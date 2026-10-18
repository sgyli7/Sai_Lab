"""Read-only data handoff check using only Python's standard library.

Never imports a simulator, the runtime adapter, or archived experiment scripts.
It does not run/compile a model or replay contact. Output is stdout only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    p = (root / relative).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError("Path escapes its declared data root: " + relative)
    return p


def verify_cart_data(root, versions, failures, check_originals):
    cart = versions.get("historical_cart_inputs", [versions.get("cart_inputs")])[0]
    if not cart:
        return None
    index = json.loads(inside(root, cart["integration_index"]).read_text())
    counts = {}
    for key in ["collision_artifact_manifest", "historical_static_archive_manifest"]:
        declaration = index[key]
        path = inside(root, declaration["path"])
        if sha(path) != declaration["sha256"]:
            failures.append("Cart producer manifest changed: " + key)
        rows = json.loads(path.read_text())["files"]
        if isinstance(rows, dict):
            rows = [dict(path=str(path.parent.relative_to(root) / name), **row) for name, row in rows.items()]
        for row in rows:
            target = inside(root, row["path"])
            if not target.is_file() or sha(target) != row["sha256"]:
                failures.append("Cart frozen artifact changed: " + row["path"])
        counts[key] = len(rows)
    review = index["independent_review"]
    if review["status"] == "FINAL_LOCKED":
        path = inside(root, review["manifest"])
        if sha(path) != review["manifest_sha256"]:
            failures.append("Independent cart manifest changed")
        for row in json.loads(path.read_text())["files"]:
            target = inside(root, row["path"])
            if not target.is_file() or sha(target) != row["sha256"]:
                failures.append("Independent cart evidence changed: " + row["path"])
    source_map = json.loads(inside(root, cart["source_copy_map"]).read_text())
    for row in source_map["files"]:
        target = inside(root, row["destination"])
        if sha(target) != row["sha256"]:
            failures.append("Cart source copy changed: " + row["destination"])
        if check_originals and sha(Path(row["source"])) != row["sha256"]:
            failures.append("Cart original source changed: " + row["source"])
    correction = inside(root, index["source_metadata_correction"]["path"])
    if sha(correction) != index["source_metadata_correction"]["sha256"]:
        failures.append("Cart metadata correction receipt changed")
    receipt = json.loads(correction.read_text())
    corrected = correction.parent / "geometry_mass_screen_corrected.json"
    old = correction.parent / "geometry_mass_screen_r2_original_reconstructed.json"
    reverse = json.loads(corrected.read_text())
    reverse["support_screen"]["caster_all_headings_conservative_support_rectangle_mm"]["x"] = [30, 570]
    reversed_sha = hashlib.sha256((json.dumps(reverse, indent=2) + "\n").encode()).hexdigest()
    if sha(corrected) != receipt["current_corrected_mass_sha256"] or sha(old) != receipt["old_R2_identity_mass_sha256"] or reversed_sha != sha(old):
        failures.append("Cart single-field metadata correction chain failed")
    xml_counts = {}
    for label, relative in [("cart", cart["cart_xml"]), ("combined", cart["combined_xml"])]:
        path = inside(root, relative)
        xml = ET.parse(path).getroot()
        compiler = xml.find("compiler")
        meshdir = compiler.get("meshdir", ".") if compiler is not None else "."
        for mesh in xml.findall("./asset/mesh"):
            target = (path.parent / meshdir / mesh.get("file")).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                failures.append("Cart unresolved mesh: " + str(target))
        geoms = [g for body in xml.find("worldbody").findall("body") for g in body.iter("geom")]
        xml_counts[label] = len(geoms)
        if label == "cart":
            types = {kind: sum(g.get("type") == kind for g in geoms) for kind in ["box", "cylinder", "mesh"]}
            if types != {"box": 8, "cylinder": 6, "mesh": 4}:
                failures.append("Cart collision_002 types changed")
            if len(xml.findall(".//body")) != 10 or len(xml.findall(".//joint")) != 9:
                failures.append("Cart body/passive-axis count changed")
            if abs(sum(float(x.get("mass")) for x in xml.findall(".//inertial")) - cart["empty_mass_kg"]) > 1e-10:
                failures.append("Cart declared mass changed")
    if xml_counts != {"cart": 18, "combined": 32}:
        failures.append("Cart/goose collision leaf count changed")
    return dict(status="STATIC_FILE_STRUCTURE_ONLY", producer_indexed_files=counts, XML_collision_geoms=xml_counts, passive_axes=9, empty_mass_kg=cart["empty_mass_kg"], independent_review=review["status"], metadata_correction_chain_verified=True, cargo_instances=0, integration_or_forward_executed_by_this_check=False)


def verify_incremental_data(root, versions, failures, check_originals):
    declaration = versions.get("low_handle_integration")
    if not declaration:
        return None
    index = json.loads(inside(root, declaration["path"]).read_text())
    if sha(inside(root, declaration["path"])) != declaration["sha256"]:
        failures.append("Low-handle integration index changed")
    counts = {}
    for item in index["locked_manifests"]:
        path = inside(root, item["path"])
        if sha(path) != item["sha256"]:
            failures.append("Increment producer manifest changed: " + item["path"])
        data = json.loads(path.read_text())
        rows = data["files"]
        if isinstance(rows, dict):
            rows = [dict(name=k, **v) for k, v in rows.items()]
        for row in rows:
            target = inside(root, row["path"]) if "path" in row else path.parent / row["name"]
            if not target.is_file() or sha(target) != row["sha256"]:
                permitted = index.get("producer_log_change", {})
                if target == inside(root, permitted.get("path", ".")) and sha(target) == permitted.get("actual_sha256"):
                    continue
                failures.append("Increment locked file mismatch: " + str(target.relative_to(root)))
        counts[item["path"]] = len(rows)
    for rel in index["copy_maps"]:
        for row in json.loads(inside(root, rel).read_text())["files"]:
            target = inside(root, row["destination"])
            if not target.is_file() or sha(target) != row["sha256"]:
                failures.append("Increment source copy mismatch: " + row["destination"])
            if check_originals and sha(Path(row["source"])) != row["sha256"]:
                failures.append("Increment original source mismatch: " + row["source"])
    cart = versions["cart_inputs"]
    counts_xml = {}
    for label, rel in [("cart", cart["cart_xml"]), ("combined", cart["combined_xml"])]:
        path = inside(root, rel)
        xml = ET.parse(path).getroot()
        compiler = xml.find("compiler")
        meshdir = compiler.get("meshdir", ".") if compiler is not None else "."
        for mesh in xml.findall("./asset/mesh"):
            target = (path.parent / meshdir / mesh.get("file")).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                failures.append("Increment unresolved mesh: " + str(target))
        geoms = [g for body in xml.find("worldbody").findall("body") for g in body.iter("geom")]
        counts_xml[label] = len(geoms)
        if label == "cart":
            if len(xml.findall(".//body")) != cart["cart_bodies"] or len(xml.findall(".//joint")) != cart["cart_passive_axes"]:
                failures.append("Current cart body/passive-axis count changed")
            if abs(sum(float(x.get("mass")) for x in xml.findall(".//inertial")) - cart["empty_mass_kg"]) > 1e-9:
                failures.append("Current cart mass mismatch")
    if counts_xml != {"cart": cart["cart_leaves"], "combined": 14 + cart["cart_leaves"]}:
        failures.append("Current cart collision XML count changed")
    return dict(status="STATIC_FILES_ONLY", locked_manifest_entries=counts, XML_collision_geoms=counts_xml, producer_log_difference_explicitly_recorded=True, no_simulator_import_or_calls=True)


def main(check_originals):
    root = Path(__file__).resolve().parents[1]
    versions = json.loads((root / "input_versions.json").read_text())
    package = inside(root, versions["model_directory"])
    archive = inside(root, versions["archive"])
    failures = []
    copied = json.loads((root / "copy_manifest.json").read_text())
    for row in copied["files"]:
        target = inside(root, row["destination"])
        if not target.is_file() or sha(target) != row["sha256"]:
            failures.append("Copied file mismatch: " + row["destination"])
        if check_originals:
            source = Path(row["source"])
            if not source.is_file() or sha(source) != row["sha256"]:
                failures.append("Original source mismatch: " + str(source))
    if sha(archive) != versions["archive_sha256"]:
        failures.append("Frozen archive SHA mismatch")
    with zipfile.ZipFile(archive) as z:
        members = [info for info in z.infolist() if not info.is_dir()]
        for info in members:
            destination = inside(root / "inputs", info.filename)
            if not destination.is_file() or hashlib.sha256(z.read(info)).hexdigest() != sha(destination):
                failures.append("Archive member mismatch: " + info.filename)
    if len(members) != 53:
        failures.append("Expected all 53 frozen input files")
    manifest = json.loads((package / "manifest.json").read_text())
    for row in manifest["files"]:
        p = inside(package, row["path"])
        if not p.is_file() or sha(p) != row["sha256"]:
            failures.append("Frozen manifest mismatch: " + row["path"])
    if sha(package / "manifest.json") != versions["manifest_sha256"]:
        failures.append("Frozen manifest identity changed")
    contract = json.loads((package / "physical_contract.json").read_text())
    actuator = json.loads((package / "actuator_contract.json").read_text())
    robot = ET.parse(package / "robot.xml").getroot()
    compiler = robot.find("compiler")
    meshdir = compiler.get("meshdir", ".") if compiler is not None else "."
    assets = robot.findall("./asset/mesh")
    for mesh in assets:
        file = mesh.get("file")
        if file and not inside(package, str(Path(meshdir) / file)).is_file():
            failures.append("Unresolved relative mesh: " + file)
    if robot.findall(".//include"):
        failures.append("Unexpected external model include")
    world = robot.find("worldbody")
    collision_geoms = [geom for body in world.findall("body") for geom in body.iter("geom")]
    expected = (23, 18, 4, 14)
    actual = (len(contract["bodies"]), len(contract["joints"]), len(contract["passive_linkage_joints"]), len(collision_geoms))
    if actual != expected:
        failures.append("Static robot body/axis/collision counts changed: " + str(actual))
    mass = sum(body["mass_kg"] for body in contract["bodies"])
    if abs(mass - 16.970203928833527) > 1e-10:
        failures.append("Frozen mass changed")
    for body in contract["bodies"]:
        matrix = body["inertia_at_com_world_zero_kg_m2"]
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            failures.append("Incomplete inertia tensor: " + body["name"])
        elif any(not math.isfinite(x) for row in matrix for x in row) or any(abs(matrix[i][j] - matrix[j][i]) > 1e-12 for i in range(3) for j in range(3)):
            failures.append("Invalid static inertia matrix: " + body["name"])
    if (contract["physics_dt_s"], contract["torque_dt_s"], contract["policy_dt_s"]) != (.005, .005, .02):
        failures.append("Frozen timing changed")
    if actuator["positive_mechanical_power_limit_w"] != 350 or actuator["peak_output_enabled"]:
        failures.append("Frozen continuous-only power contract changed")
    cart_check = verify_cart_data(root, versions, failures, check_originals)
    increment_check = verify_incremental_data(root, versions, failures, check_originals)
    closed_checks = []
    for item in versions.get("closed_local_probes", []):
        path = inside(root, item["copy_manifest"])
        if sha(path) != item["copy_manifest_sha256"]:
            failures.append("Closed probe copy manifest changed")
        rows = json.loads(path.read_text())["files"]
        for row in rows:
            if sha(inside(root, row["destination"])) != row["sha256"]:
                failures.append("Closed probe copy changed: " + row["destination"])
            if check_originals and sha(Path(row["source"])) != row["sha256"]:
                failures.append("Closed probe original changed: " + row["source"])
        producer = inside(root, item["producer_manifest"])
        if sha(producer) != item["producer_manifest_sha256"]:
            failures.append("Closed probe producer manifest changed")
        if check_originals:
            for row in json.loads(producer.read_text()):
                if sha(Path(row["path"])) != row["sha256"]:
                    failures.append("Closed probe raw record changed")
        if sha(inside(root, item["independent_review"])) != item["independent_review_sha256"]:
            failures.append("Closed probe independent review changed")
        reproduction = json.loads(inside(root, item["entry"] + "/reproduction_contract.json").read_text())
        for dep in reproduction["mesh_dependencies"].values():
            if sha(inside(root, dep["package_path"])) != dep["sha256"]:
                failures.append("Closed probe mesh dependency changed")
        closed_checks.append(dict(id=item["id"],exact_copies=len(rows),raw_indexed_records=54,decision=item["decision"],historical_steps=item["producer_physics_steps"],new_integrator_steps=0))
    # Parse only. Importing these archived files could execute old orders.
    python_files = list(root.rglob("*.py"))
    for source in python_files:
        ast.parse(source.read_bytes(), filename=str(source))
    raw_indexes = json.loads((root / "raw_data_index.json").read_text())["datasets"]
    raw_checked = 0
    for dataset in raw_indexes:
        index = json.loads((root / "history" / dataset["experiment_id"] / "raw_data_index.json").read_text())
        source_manifest = root / "history" / dataset["experiment_id"] / "selected/results_manifest.json"
        if sha(source_manifest) != dataset["manifest_sha256"]:
            failures.append("Raw manifest copy mismatch: " + dataset["experiment_id"])
        if check_originals:
            original = Path(dataset["manifest_source"])
            if not original.is_file() or sha(original) != dataset["manifest_sha256"]:
                failures.append("Original raw manifest changed: " + str(original))
            for row in index["raw_files"]:
                source = Path(row["raw_path"])
                if not source.is_file() or sha(source) != row["sha256"]:
                    failures.append("Raw data hash mismatch: " + str(source))
                raw_checked += 1
    dataset_manifest = root / "dataset_manifest.json"
    if dataset_manifest.exists():
        for row in json.loads(dataset_manifest.read_text())["files"]:
            target = inside(root, row["path"])
            if not target.is_file() or sha(target) != row["sha256"]:
                failures.append("Dataset file changed: " + row["path"])
    report = dict(schema="goose_v02_data_handoff_static_check_v1", status="PASS_STATIC_DATA_ONLY" if not failures else "REJECT", copied_files_checked=len(copied["files"]), archive_members_checked=len(members), frozen_manifest_entries=len(manifest["files"]), mesh_references_resolved=len(assets), rigid_bodies=actual[0], public_axes=actual[1], internal_coordinates=actual[2], collision_geoms=actual[3], mass_kg=mass, embedded_contract_release_id=contract["release_id"], closed_local_probes=closed_checks, cart_static_history=cart_check, low_handle_increment=increment_check, native_simulator_compiled_or_imported=False, historical_python_sources_parsed_not_imported=len(python_files), original_sources_requested=check_originals, raw_files_checked_this_invocation=raw_checked, original_raw_manifest_files=sum(row["manifest_entries"] for row in raw_indexes), physical_steps=0, forward_calls=0, contact_replays=0, training_updates=0, GPU_use=0, new_experiment_orders=0, failures=failures, scope="Byte identity, archive and static structure only; no new physical, contact, task or manufacturing validation")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-originals", action="store_true", help="Hash original Rotbots/ProjectBackups files too; never execute them")
    sys.exit(main(parser.parse_args().check_originals))
