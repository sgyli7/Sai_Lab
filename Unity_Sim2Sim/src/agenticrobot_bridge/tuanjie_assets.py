"""Deterministically prepare MicroDuck assets for the Tuanjie project."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mujoco

from .onnx_compat import convert_policy_bundle
from .policy_audit import discover_policy_paths
from .robot_manifest import RobotManifest, compile_robot_manifest
from .upstream import UpstreamLock, load_upstream_lock


GENERATED_ASSET_ROOT = "Assets/MicroDuck/Generated"
MESH_LICENSE = "Creative Commons BY-SA-NC"
COLLISION_HULL_MAX_VERTICES = 128
TIRE_MESH_NAME = "tire"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AssetPreparationError(RuntimeError):
    """Raised when a complete, provenance-preserving asset bundle cannot be made."""


@dataclass(frozen=True, slots=True)
class StlConversionResult:
    """Provenance and topology counts for one generated OBJ."""

    destination: Path
    source_sha256: str
    obj_sha256: str
    vertex_count: int
    triangle_count: int


@dataclass(frozen=True, slots=True)
class CompiledMeshGeometry:
    """Indexed mesh geometry in MuJoCo's compiled, geom-compatible frame."""

    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class AssetPreparationReport:
    """Machine-readable result for one complete Tuanjie asset preparation."""

    output_directory: Path
    document: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.document["passed"])

    def to_dict(self) -> dict[str, Any]:
        return self.document


def convert_stl_to_obj(
    source: str | Path,
    destination: str | Path,
    *,
    source_relative_path: str,
    source_repository: str,
    source_commit: str,
) -> StlConversionResult:
    """Convert binary or ASCII STL to a handedness-correct Tuanjie OBJ."""

    source_path = Path(source)
    destination_path = Path(destination)
    source_bytes = source_path.read_bytes()
    source_sha256 = _sha256_bytes(source_bytes)
    triangles = _read_stl_triangles(source_bytes, source_path)

    vertices: list[tuple[float, float, float]] = []
    vertex_indices: dict[tuple[float, float, float], int] = {}
    faces: list[tuple[int, int, int]] = []
    for triangle in triangles:
        indices: list[int] = []
        for vertex in triangle:
            transformed = _mujoco_to_tuanjie_vertex(vertex)
            index = vertex_indices.get(transformed)
            if index is None:
                vertices.append(transformed)
                index = len(vertices)
                vertex_indices[transformed] = index
            indices.append(index)
        # The basis has determinant -1, so preserve outward normals by reversing winding.
        faces.append((indices[0], indices[2], indices[1]))

    safe_name = _safe_asset_stem(source_path.stem)
    lines = [
        "# Generated MicroDuck mesh; do not edit by hand.",
        f"# Source: {source_relative_path}",
        f"# Source repository: {source_repository}@{source_commit}",
        f"# Source SHA-256: {source_sha256}",
        f"# License: {MESH_LICENSE}; see ../Licenses/ATTRIBUTION.json.",
        "# Modified: converted MuJoCo coordinates to Tuanjie coordinates.",
        "# Transform: (x,y,z) -> (-y,z,x); triangle winding reversed.",
        f"o {safe_name}",
    ]
    lines.extend(
        f"v {_format_float(x)} {_format_float(y)} {_format_float(z)}"
        for x, y, z in vertices
    )
    lines.extend(f"f {first} {second} {third}" for first, second, third in faces)
    contents = ("\n".join(lines) + "\n").encode("utf-8")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(contents)
    return StlConversionResult(
        destination=destination_path,
        source_sha256=source_sha256,
        obj_sha256=_sha256_bytes(contents),
        vertex_count=len(vertices),
        triangle_count=len(faces),
    )


def convert_compiled_mesh_to_obj(
    source: str | Path,
    destination: str | Path,
    *,
    source_relative_path: str,
    source_repository: str,
    source_commit: str,
    geometry: CompiledMeshGeometry,
) -> StlConversionResult:
    """Write compiled MuJoCo mesh data in the Tuanjie coordinate basis.

    MuJoCo recenters and may reorient imported meshes at compile time.  It also
    compensates that operation in each compiled geom pose.  Because the import
    manifest uses those compiled geom poses, its OBJ must use ``mesh_vert`` and
    ``mesh_face`` from the same compiled model rather than the raw STL frame.
    """

    source_path = Path(source)
    destination_path = Path(destination)
    source_sha256 = _sha256_path(source_path)
    if not geometry.vertices or not geometry.faces:
        raise AssetPreparationError(f"Compiled mesh {source_path} contains no geometry")

    transformed_vertices = [
        _mujoco_to_tuanjie_vertex(_finite_vertex(vertex, source_path))
        for vertex in geometry.vertices
    ]
    vertex_count = len(transformed_vertices)
    validated_faces: list[tuple[int, int, int]] = []
    for face in geometry.faces:
        if len(face) != 3 or any(index < 0 or index >= vertex_count for index in face):
            raise AssetPreparationError(
                f"Compiled mesh {source_path} contains an invalid triangle {face!r}"
            )
        validated_faces.append(face)

    safe_name = _safe_asset_stem(source_path.stem)
    lines = [
        "# Generated MicroDuck mesh; do not edit by hand.",
        f"# Source: {source_relative_path}",
        f"# Source repository: {source_repository}@{source_commit}",
        f"# Source SHA-256: {source_sha256}",
        f"# License: {MESH_LICENSE}; see ../Licenses/ATTRIBUTION.json.",
        "# Modified: exported MuJoCo-compiled mesh geometry in Tuanjie coordinates.",
        "# Transform: (x,y,z) -> (-y,z,x); triangle winding reversed.",
        f"o {safe_name}",
    ]
    lines.extend(
        f"v {_format_float(x)} {_format_float(y)} {_format_float(z)}"
        for x, y, z in transformed_vertices
    )
    lines.extend(
        f"f {first + 1} {third + 1} {second + 1}"
        for first, second, third in validated_faces
    )
    contents = ("\n".join(lines) + "\n").encode("utf-8")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(contents)
    return StlConversionResult(
        destination=destination_path,
        source_sha256=source_sha256,
        obj_sha256=_sha256_bytes(contents),
        vertex_count=vertex_count,
        triangle_count=len(validated_faces),
    )


def prepare_tuanjie_assets(
    project_root: str | Path,
    tuanjie_project: str | Path,
) -> AssetPreparationReport:
    """Build and atomically install all generated MicroDuck Tuanjie assets."""

    root = Path(project_root).resolve()
    tuanjie_root = Path(tuanjie_project).resolve()
    upstream_root = root / ".cache" / "upstream"
    lock = load_upstream_lock(root / "upstream.lock.json")
    _validate_required_repositories(lock)

    microduck_asset_root = tuanjie_root / "Assets" / "MicroDuck"
    microduck_asset_root.mkdir(parents=True, exist_ok=True)
    generated_root = microduck_asset_root / "Generated"
    _validate_generated_target(tuanjie_root, generated_root)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".Generated-stage-", dir=microduck_asset_root)
    )
    try:
        document = _build_asset_bundle(root, upstream_root, lock, staging_root)
        _write_json(staging_root / "asset-report.json", document)
        if generated_root.exists():
            shutil.rmtree(generated_root)
        staging_root.replace(generated_root)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    return AssetPreparationReport(output_directory=generated_root, document=document)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by the one-command MVP runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--tuanjie-project", required=True, type=Path)
    arguments = parser.parse_args(argv)
    report = prepare_tuanjie_assets(arguments.project_root, arguments.tuanjie_project)
    print(json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")))
    return 0 if report.passed else 1


def _build_asset_bundle(
    project_root: Path,
    upstream_root: Path,
    lock: UpstreamLock,
    staging_root: Path,
) -> dict[str, Any]:
    model_root = (
        upstream_root
        / "microduck_rl"
        / "src"
        / "mjlab_microduck"
        / "robot"
        / "microduck"
    )
    manifest_specs = (("legged", "scene.xml"), ("roller", "scene_rollers.xml"))
    manifests: list[RobotManifest] = []
    manifest_reports: list[dict[str, Any]] = []
    for variant, source_name in manifest_specs:
        manifest = compile_robot_manifest(
            model_root / source_name,
            source_root=upstream_root,
        )
        if manifest.variant != variant:
            raise AssetPreparationError(
                f"{source_name} compiled as {manifest.variant}, expected {variant}"
            )
        manifests.append(manifest)
        manifest_reports.append(
            {
                "variant": variant,
                "asset": _generated_asset_path("Manifests", f"{variant}.json"),
                "sourceFile": manifest.source_relative_path,
                "sourceSha256": manifest.source_sha256,
                "bodyCount": manifest.body_count,
                "jointCount": manifest.joint_count,
                "servoCount": manifest.servo_count,
                "passiveJointCount": len(manifest.passive_joint_names),
                "geomCount": len(manifest.geoms),
                "meshCount": len(manifest.meshes),
            }
        )

    mesh_reports = _prepare_meshes(staging_root, upstream_root, lock, manifests)
    collision_mesh_reports = _prepare_collision_meshes(
        staging_root,
        upstream_root,
        lock,
        manifests,
        mesh_reports,
    )
    _write_import_manifests(
        staging_root,
        manifests,
        mesh_reports,
        collision_mesh_reports,
    )
    collision_reports = _prepare_collision_proxies(
        staging_root,
        manifests,
        collision_mesh_reports,
    )
    policy_reports = _prepare_policies(staging_root, upstream_root, lock)
    attribution_reports = _prepare_attribution(
        project_root, staging_root, upstream_root, lock
    )

    expected_manifest_counts = {
        "legged": (15, 15, 14, 0),
        "roller": (19, 19, 14, 4),
    }
    manifest_counts_match = all(
        (
            report["bodyCount"],
            report["jointCount"],
            report["servoCount"],
            report["passiveJointCount"],
        )
        == expected_manifest_counts[report["variant"]]
        for report in manifest_reports
    )
    original_policy_names = tuple(
        path.name for path in sorted((staging_root / "Policies" / "Original").glob("*.onnx"))
    )
    barracuda_policy_names = tuple(
        path.name
        for path in sorted((staging_root / "Policies" / "Barracuda").glob("*.onnx"))
    )
    expected_policy_names = tuple(sorted(lock.policies))
    manifest_mesh_names = {
        mesh.name for manifest in manifests for mesh in manifest.meshes
    }
    generated_mesh_names = {report["name"] for report in mesh_reports}
    referenced_collision_mesh_names = {
        geom.mesh_name
        for manifest in manifests
        for geom in manifest.geoms
        if geom.collidable
        and geom.type == "mesh"
        and geom.mesh_name is not None
        and geom.mesh_name.lower() != TIRE_MESH_NAME
    }
    generated_collision_mesh_names = {
        report["name"] for report in collision_mesh_reports
    }
    checks = {
        "allReferencedObjPresent": (
            generated_mesh_names == manifest_mesh_names
            and all(
                (staging_root / Path(report["objAsset"]).relative_to(GENERATED_ASSET_ROOT)).is_file()
                for report in mesh_reports
            )
        ),
        "allCollisionMeshesPresent": (
            generated_collision_mesh_names == referenced_collision_mesh_names
            and all(
                (
                    staging_root
                    / Path(report["objAsset"]).relative_to(GENERATED_ASSET_ROOT)
                ).is_file()
                for report in collision_mesh_reports
            )
        ),
        "exactPolicySet": (
            original_policy_names == expected_policy_names
            and barracuda_policy_names == expected_policy_names
        ),
        "manifestCountsMatch": manifest_counts_match,
        "sourceHashesRecorded": _all_source_hashes_recorded(
            manifest_reports,
            mesh_reports,
            collision_mesh_reports,
            policy_reports,
            attribution_reports,
        ),
    }
    return {
        "schemaVersion": 2,
        "generatedRoot": GENERATED_ASSET_ROOT,
        "passed": all(checks.values()),
        "checks": checks,
        "manifests": manifest_reports,
        "policies": policy_reports,
        "meshes": mesh_reports,
        "collisionMeshes": collision_mesh_reports,
        "collisionProxies": collision_reports,
        "attribution": attribution_reports,
    }


def _write_import_manifests(
    staging_root: Path,
    manifests: list[RobotManifest],
    mesh_reports: list[dict[str, Any]],
    collision_mesh_reports: list[dict[str, Any]],
) -> None:
    mesh_assets = {report["name"]: report for report in mesh_reports}
    collision_mesh_assets = {
        report["name"]: report for report in collision_mesh_reports
    }
    for manifest in manifests:
        payload = manifest.to_dict()
        for mesh in payload["meshes"]:
            try:
                generated = mesh_assets[mesh["name"]]
            except KeyError as exc:
                raise AssetPreparationError(
                    f"Manifest {manifest.variant} references unprepared mesh {mesh['name']!r}"
                ) from exc
            mesh["assetFile"] = generated["objAsset"]
            mesh["sourceSha256"] = generated["sourceSha256"]
            mesh["assetSha256"] = generated["objSha256"]
            collision_mesh = collision_mesh_assets.get(mesh["name"])
            if collision_mesh is not None:
                mesh["collisionAssetFile"] = collision_mesh["objAsset"]
                mesh["collisionAssetSha256"] = collision_mesh["objSha256"]
            # The generated OBJ already contains MuJoCo's compiled vertices,
            # including the source mesh scale.  Applying mesh_scale again in
            # Tuanjie would double-scale non-unit source meshes.
            mesh["scale"] = [1.0, 1.0, 1.0]
        _write_json(staging_root / "Manifests" / f"{manifest.variant}.json", payload)


def _prepare_meshes(
    staging_root: Path,
    upstream_root: Path,
    lock: UpstreamLock,
    manifests: list[RobotManifest],
) -> list[dict[str, Any]]:
    compiled_meshes = _collect_compiled_mesh_geometry(manifests)
    mesh_sources: dict[str, tuple[str, tuple[float, float, float]]] = {}
    variants_by_mesh: dict[str, set[str]] = {}
    asset_names: dict[str, str] = {}
    for manifest in manifests:
        for mesh in manifest.meshes:
            existing = mesh_sources.get(mesh.name)
            definition = (mesh.source_file, mesh.scale)
            if existing is not None and existing != definition:
                raise AssetPreparationError(
                    f"Mesh {mesh.name!r} resolves to conflicting sources or scales"
                )
            mesh_sources[mesh.name] = definition
            variants_by_mesh.setdefault(mesh.name, set()).add(manifest.variant)
            asset_name = f"{_safe_asset_stem(mesh.name)}.obj"
            conflicting_mesh = asset_names.get(asset_name)
            if conflicting_mesh is not None and conflicting_mesh != mesh.name:
                raise AssetPreparationError(
                    f"Mesh names {conflicting_mesh!r} and {mesh.name!r} collide as {asset_name}"
                )
            asset_names[asset_name] = mesh.name

    repository = lock.repositories["microduck_rl"]
    reports: list[dict[str, Any]] = []
    for mesh_name in sorted(mesh_sources):
        source_file, scale = mesh_sources[mesh_name]
        source = upstream_root / source_file
        asset_name = f"{_safe_asset_stem(mesh_name)}.obj"
        destination = staging_root / "Meshes" / asset_name
        conversion = convert_compiled_mesh_to_obj(
            source,
            destination,
            source_relative_path=source_file,
            source_repository=repository.url,
            source_commit=repository.commit,
            geometry=compiled_meshes[mesh_name],
        )
        reports.append(
            {
                "name": mesh_name,
                "variants": sorted(variants_by_mesh[mesh_name]),
                "sourceFile": source_file,
                "sourceSha256": conversion.source_sha256,
                "sourceScale": list(scale),
                "objAsset": _generated_asset_path("Meshes", asset_name),
                "objSha256": conversion.obj_sha256,
                "vertexCount": conversion.vertex_count,
                "triangleCount": conversion.triangle_count,
                "basisTransform": ["-y", "z", "x"],
                "windingReversed": True,
                "meshFrame": "mujoco-compiled",
                "license": MESH_LICENSE,
            }
        )
    return reports


def _prepare_collision_meshes(
    staging_root: Path,
    upstream_root: Path,
    lock: UpstreamLock,
    manifests: list[RobotManifest],
    mesh_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Export capped MuJoCo hulls that PhysX can cook without partial-hull fallback.

    Unity/Tuanjie 2022 limits a convex MeshCollider to 255 triangles. A closed
    triangular convex hull with 128 vertices has at most 252 faces, so the cap
    preserves as much of MuJoCo's hull as possible while leaving deterministic
    headroom below that engine limit. Tires remain analytic cylinders in Tuanjie.
    """

    compiled_hulls = _collect_capped_collision_hulls(
        manifests,
        max_hull_vertices=COLLISION_HULL_MAX_VERTICES,
    )
    visual_reports = {report["name"]: report for report in mesh_reports}
    repository = lock.repositories["microduck_rl"]
    reports: list[dict[str, Any]] = []
    for mesh_name in sorted(compiled_hulls):
        visual_report = visual_reports[mesh_name]
        source_file = visual_report["sourceFile"]
        source = upstream_root / source_file
        asset_name = f"{_safe_asset_stem(mesh_name)}.obj"
        destination = staging_root / "CollisionMeshes" / asset_name
        conversion = convert_compiled_mesh_to_obj(
            source,
            destination,
            source_relative_path=source_file,
            source_repository=repository.url,
            source_commit=repository.commit,
            geometry=compiled_hulls[mesh_name],
        )
        if conversion.vertex_count > COLLISION_HULL_MAX_VERTICES:
            raise AssetPreparationError(
                f"Collision hull {mesh_name!r} has {conversion.vertex_count} vertices, "
                f"above the {COLLISION_HULL_MAX_VERTICES} vertex cap"
            )
        if conversion.triangle_count > 2 * COLLISION_HULL_MAX_VERTICES - 4:
            raise AssetPreparationError(
                f"Collision hull {mesh_name!r} has too many triangles for PhysX"
            )
        reports.append(
            {
                "name": mesh_name,
                "variants": visual_report["variants"],
                "sourceFile": source_file,
                "sourceSha256": conversion.source_sha256,
                "objAsset": _generated_asset_path("CollisionMeshes", asset_name),
                "objSha256": conversion.obj_sha256,
                "vertexCount": conversion.vertex_count,
                "triangleCount": conversion.triangle_count,
                "maxHullVertices": COLLISION_HULL_MAX_VERTICES,
                "basisTransform": ["-y", "z", "x"],
                "windingReversed": True,
                "meshFrame": "mujoco-compiled-capped-convex-hull",
                "license": MESH_LICENSE,
            }
        )
    return reports


def _collect_compiled_mesh_geometry(
    manifests: Sequence[RobotManifest],
) -> dict[str, CompiledMeshGeometry]:
    """Collect and cross-check compiled mesh frames from every robot variant."""

    compiled: dict[str, CompiledMeshGeometry] = {}
    for manifest in manifests:
        model = mujoco.MjModel.from_xml_path(str(manifest.source_xml))
        for mesh in manifest.meshes:
            mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh.name)
            if mesh_id < 0:
                raise AssetPreparationError(
                    f"Compiled model {manifest.source_xml} has no mesh {mesh.name!r}"
                )
            vertex_address = int(model.mesh_vertadr[mesh_id])
            vertex_count = int(model.mesh_vertnum[mesh_id])
            face_address = int(model.mesh_faceadr[mesh_id])
            face_count = int(model.mesh_facenum[mesh_id])
            geometry = CompiledMeshGeometry(
                vertices=tuple(
                    tuple(float(value) for value in vertex)
                    for vertex in model.mesh_vert[
                        vertex_address : vertex_address + vertex_count
                    ]
                ),
                faces=tuple(
                    tuple(int(index) for index in face)
                    for face in model.mesh_face[face_address : face_address + face_count]
                ),
            )
            previous = compiled.get(mesh.name)
            if previous is not None and previous != geometry:
                raise AssetPreparationError(
                    f"Mesh {mesh.name!r} compiles differently between robot variants"
                )
            compiled[mesh.name] = geometry
    return compiled


def _collect_capped_collision_hulls(
    manifests: Sequence[RobotManifest],
    *,
    max_hull_vertices: int,
) -> dict[str, CompiledMeshGeometry]:
    """Compile and cross-check deterministic MuJoCo hulls for collision geoms."""

    if max_hull_vertices <= 3 or 2 * max_hull_vertices - 4 > 255:
        raise AssetPreparationError(
            "Collision hull cap must produce a closed triangular hull below "
            "PhysX's 255-triangle limit"
        )

    compiled: dict[str, CompiledMeshGeometry] = {}
    for manifest in manifests:
        collision_mesh_names = {
            geom.mesh_name
            for geom in manifest.geoms
            if geom.collidable
            and geom.type == "mesh"
            and geom.mesh_name is not None
            and geom.mesh_name.lower() != TIRE_MESH_NAME
        }
        spec = mujoco.MjSpec.from_file(str(manifest.source_xml))
        # MJCF permits an omitted mesh name; MuJoCo then derives the compiled
        # name from the source filename even though MjSpec keeps ``name`` empty.
        spec_meshes = {
            mesh.name or Path(mesh.file).stem: mesh
            for mesh in spec.meshes
        }
        missing_meshes = collision_mesh_names - spec_meshes.keys()
        if missing_meshes:
            missing = ", ".join(sorted(missing_meshes))
            raise AssetPreparationError(
                f"Model {manifest.source_xml} has no collision meshes: {missing}"
            )
        for mesh_name in collision_mesh_names:
            spec_meshes[mesh_name].maxhullvert = max_hull_vertices
        model = spec.compile()

        for mesh_name in sorted(collision_mesh_names):
            mesh_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_MESH,
                mesh_name,
            )
            geometry = _compiled_collision_hull_geometry(
                model,
                mesh_id,
                mesh_name=mesh_name,
                max_hull_vertices=max_hull_vertices,
            )
            previous = compiled.get(mesh_name)
            if previous is not None and previous != geometry:
                raise AssetPreparationError(
                    f"Collision hull {mesh_name!r} differs between robot variants"
                )
            compiled[mesh_name] = geometry
    return compiled


def _compiled_collision_hull_geometry(
    model: mujoco.MjModel,
    mesh_id: int,
    *,
    mesh_name: str,
    max_hull_vertices: int,
) -> CompiledMeshGeometry:
    """Decode one convex hull record from MuJoCo's packed ``mesh_graph`` array."""

    if mesh_id < 0:
        raise AssetPreparationError(f"Compiled model has no mesh {mesh_name!r}")
    graph_address = int(model.mesh_graphadr[mesh_id])
    if graph_address < 0:
        raise AssetPreparationError(f"Compiled mesh {mesh_name!r} has no convex hull graph")

    vertex_count = int(model.mesh_graph[graph_address])
    face_count = int(model.mesh_graph[graph_address + 1])
    if not 3 < vertex_count <= max_hull_vertices:
        raise AssetPreparationError(
            f"Collision hull {mesh_name!r} has invalid vertex count {vertex_count}"
        )
    if not 3 < face_count <= 2 * max_hull_vertices - 4:
        raise AssetPreparationError(
            f"Collision hull {mesh_name!r} has invalid face count {face_count}"
        )

    # MuJoCo packs [counts, vertex-edge offsets, hull-to-mesh vertex IDs,
    # edge records, face-to-mesh vertex IDs] into mesh_graph. Face IDs are
    # indices into this mesh's mesh_vert block, not global mesh_vert indices.
    vertex_ids_address = graph_address + 2 + vertex_count
    hull_vertex_ids = tuple(
        int(value)
        for value in model.mesh_graph[
            vertex_ids_address : vertex_ids_address + vertex_count
        ]
    )
    if len(set(hull_vertex_ids)) != vertex_count:
        raise AssetPreparationError(
            f"Collision hull {mesh_name!r} contains duplicate vertex IDs"
        )

    face_ids_address = graph_address + 2 + 3 * vertex_count + 3 * face_count
    face_vertex_ids = tuple(
        int(value)
        for value in model.mesh_graph[
            face_ids_address : face_ids_address + 3 * face_count
        ]
    )
    local_ids = {mesh_vertex_id: index for index, mesh_vertex_id in enumerate(hull_vertex_ids)}
    try:
        faces = tuple(
            tuple(local_ids[face_vertex_ids[index + offset]] for offset in range(3))
            for index in range(0, len(face_vertex_ids), 3)
        )
    except KeyError as exc:
        raise AssetPreparationError(
            f"Collision hull {mesh_name!r} face references a non-hull vertex"
        ) from exc

    mesh_vertex_address = int(model.mesh_vertadr[mesh_id])
    mesh_vertex_count = int(model.mesh_vertnum[mesh_id])
    if any(index < 0 or index >= mesh_vertex_count for index in hull_vertex_ids):
        raise AssetPreparationError(
            f"Collision hull {mesh_name!r} references an out-of-range mesh vertex"
        )
    vertices = tuple(
        tuple(float(value) for value in model.mesh_vert[mesh_vertex_address + index])
        for index in hull_vertex_ids
    )
    return CompiledMeshGeometry(vertices=vertices, faces=faces)


def _prepare_collision_proxies(
    staging_root: Path,
    manifests: list[RobotManifest],
    collision_mesh_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    collision_mesh_assets = {
        report["name"]: report for report in collision_mesh_reports
    }
    reports: list[dict[str, Any]] = []
    for manifest in manifests:
        proxies: list[dict[str, Any]] = []
        for geom in manifest.geoms:
            if not geom.collidable or geom.type != "mesh" or geom.mesh_name is None:
                continue
            half_extents = list(geom.size)
            if not all(math.isfinite(value) and value > 0.0 for value in half_extents):
                raise AssetPreparationError(
                    f"Collidable geom {geom.name!r} has invalid half extents"
                )
            proxy_type = (
                "cylinder"
                if geom.mesh_name.lower() == TIRE_MESH_NAME
                else "convexMesh"
            )
            proxy = {
                "geomName": geom.name,
                "bodyName": geom.body_name,
                "meshName": geom.mesh_name,
                "proxyType": proxy_type,
                "localPosition": list(geom.local_position),
                "localRotationWxyz": list(geom.local_rotation_wxyz),
                "halfExtents": half_extents,
                "contype": geom.contype,
                "conaffinity": geom.conaffinity,
            }
            if proxy_type == "cylinder":
                proxy["radiusMeters"] = max(half_extents[:2])
                proxy["halfWidthMeters"] = half_extents[2]
            else:
                try:
                    collision_mesh = collision_mesh_assets[geom.mesh_name]
                except KeyError as exc:
                    raise AssetPreparationError(
                        f"Collidable geom {geom.name!r} has no generated collision mesh"
                    ) from exc
                proxy["collisionAssetFile"] = collision_mesh["objAsset"]
                proxy["collisionAssetSha256"] = collision_mesh["objSha256"]
            proxies.append(proxy)
        payload = {
            "schemaVersion": 2,
            "variant": manifest.variant,
            "strategy": "mujoco-capped-convex-hull; tire-as-cylinder",
            "maxHullVertices": COLLISION_HULL_MAX_VERTICES,
            "proxies": proxies,
        }
        asset_name = f"{manifest.variant}.json"
        _write_json(staging_root / "Collision" / asset_name, payload)
        convex_mesh_count = sum(
            proxy["proxyType"] == "convexMesh" for proxy in proxies
        )
        cylinder_count = sum(proxy["proxyType"] == "cylinder" for proxy in proxies)
        reports.append(
            {
                "variant": manifest.variant,
                "asset": _generated_asset_path("Collision", asset_name),
                "proxyCount": len(proxies),
                "convexMeshCount": convex_mesh_count,
                "cylinderCount": cylinder_count,
                "maxHullVertices": COLLISION_HULL_MAX_VERTICES,
            }
        )
    return reports


def _prepare_policies(
    staging_root: Path,
    upstream_root: Path,
    lock: UpstreamLock,
) -> list[dict[str, Any]]:
    source_paths = discover_policy_paths(upstream_root, lock)
    original_root = staging_root / "Policies" / "Original"
    original_root.mkdir(parents=True, exist_ok=True)
    for name, source in source_paths.items():
        shutil.copyfile(source, original_root / name)

    barracuda_root = staging_root / "Policies" / "Barracuda"
    compatibility = convert_policy_bundle(
        upstream_root,
        lock,
        barracuda_root,
        policy_subdirectory=".",
    )
    converted = {policy.name: policy for policy in compatibility.policies}
    return [
        {
            "name": name,
            "sourceSha256": converted[name].original_sha256,
            "originalAsset": _generated_asset_path("Policies", "Original", name),
            "barracudaAsset": _generated_asset_path("Policies", "Barracuda", name),
            "barracudaSha256": converted[name].converted_sha256,
            "targetOpset": compatibility.target_opset,
            "fixtureCount": converted[name].fixture_count,
            "maxAbsError": converted[name].max_abs_error,
        }
        for name in lock.policies
    ]


def _prepare_attribution(
    project_root: Path,
    staging_root: Path,
    upstream_root: Path,
    lock: UpstreamLock,
) -> list[dict[str, Any]]:
    license_root = staging_root / "Licenses"
    license_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    repository_documents: dict[str, dict[str, Any]] = {}
    for name in ("microduck", "microduck_rl"):
        source = upstream_root / name / "LICENSE"
        asset_name = f"{name}-LICENSE.txt"
        destination = license_root / asset_name
        shutil.copyfile(source, destination)
        source_sha256 = _sha256_path(source)
        reports.append(
            {
                "name": f"{name}-license",
                "asset": _generated_asset_path("Licenses", asset_name),
                "sourceFile": source.relative_to(project_root).as_posix(),
                "sourceSha256": source_sha256,
            }
        )
        repository = lock.repositories[name]
        repository_documents[name] = {
            "url": repository.url,
            "commit": repository.commit,
            "licenseAsset": _generated_asset_path("Licenses", asset_name),
            "licenseSha256": source_sha256,
        }

    readme_source = upstream_root / "microduck_rl" / "README.md"
    readme_destination = license_root / "microduck_rl-README.md"
    shutil.copyfile(readme_source, readme_destination)
    readme_sha256 = _sha256_path(readme_source)
    reports.append(
        {
            "name": "microduck_rl-readme",
            "asset": _generated_asset_path("Licenses", readme_destination.name),
            "sourceFile": readme_source.relative_to(project_root).as_posix(),
            "sourceSha256": readme_sha256,
        }
    )
    attribution = {
        "schemaVersion": 1,
        "repositories": repository_documents,
        "meshLicense": MESH_LICENSE,
        "meshLicenseNotice": (
            "3D model files are licensed under Creative Commons BY-SA-NC."
        ),
        "meshLicenseSource": {
            "file": reports[-1]["asset"],
            "sha256": readme_sha256,
        },
        "modificationNotice": (
            "Generated OBJ files transform coordinates from MuJoCo to Tuanjie and "
            "reverse triangle winding. Generated ONNX files only change the default "
            "opset declaration from 18 to 9 after restricted validation and parity tests."
        ),
    }
    attribution_path = license_root / "ATTRIBUTION.json"
    _write_json(attribution_path, attribution)
    reports.append(
        {
            "name": "attribution",
            "asset": _generated_asset_path("Licenses", attribution_path.name),
            "sourceFile": "upstream.lock.json + microduck_rl/README.md",
            "sourceSha256": _sha256_path(project_root / "upstream.lock.json"),
        }
    )
    return reports


def _read_stl_triangles(
    source: bytes,
    source_path: Path,
) -> list[tuple[tuple[float, float, float], ...]]:
    if len(source) >= 84:
        triangle_count = struct.unpack_from("<I", source, 80)[0]
        if 84 + triangle_count * 50 == len(source):
            triangles = []
            for index in range(triangle_count):
                values = struct.unpack_from("<12fH", source, 84 + index * 50)
                triangles.append(
                    (
                        _finite_vertex(values[3:6], source_path),
                        _finite_vertex(values[6:9], source_path),
                        _finite_vertex(values[9:12], source_path),
                    )
                )
            if not triangles:
                raise AssetPreparationError(f"STL {source_path} contains no triangles")
            return triangles

    try:
        text = source.decode("ascii")
    except UnicodeDecodeError as exc:
        raise AssetPreparationError(f"STL {source_path} is neither binary nor ASCII") from exc
    vertices: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        fields = line.strip().split()
        if fields and fields[0].lower() == "vertex":
            if len(fields) != 4:
                raise AssetPreparationError(f"Malformed vertex in ASCII STL {source_path}")
            try:
                vertex = tuple(float(value) for value in fields[1:])
            except ValueError as exc:
                raise AssetPreparationError(
                    f"Malformed numeric vertex in ASCII STL {source_path}"
                ) from exc
            vertices.append(_finite_vertex(vertex, source_path))
    if not vertices or len(vertices) % 3:
        raise AssetPreparationError(
            f"ASCII STL {source_path} must contain a non-zero multiple of three vertices"
        )
    return [
        (vertices[index], vertices[index + 1], vertices[index + 2])
        for index in range(0, len(vertices), 3)
    ]


def _finite_vertex(
    values: Sequence[float],
    source_path: Path,
) -> tuple[float, float, float]:
    vertex = tuple(float(value) for value in values)
    if len(vertex) != 3 or not all(math.isfinite(value) for value in vertex):
        raise AssetPreparationError(f"STL {source_path} contains a non-finite vertex")
    return vertex


def _mujoco_to_tuanjie_vertex(
    vertex: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z = vertex
    return tuple(0.0 if value == 0.0 else value for value in (-y, z, x))


def _safe_asset_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise AssetPreparationError(f"Asset name {value!r} has no safe filename characters")
    return safe


def _format_float(value: float) -> str:
    return format(value, ".9g")


def _generated_asset_path(*parts: str) -> str:
    return "/".join((GENERATED_ASSET_ROOT, *parts))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_required_repositories(lock: UpstreamLock) -> None:
    required = {"microduck", "microduck_rl"}
    missing = sorted(required.difference(lock.repositories))
    if missing:
        raise AssetPreparationError(f"Upstream lock is missing: {', '.join(missing)}")


def _validate_generated_target(tuanjie_root: Path, generated_root: Path) -> None:
    expected_parent = (tuanjie_root / "Assets" / "MicroDuck").resolve()
    resolved = generated_root.resolve()
    if resolved.name != "Generated" or resolved.parent != expected_parent:
        raise AssetPreparationError(f"Refusing unsafe generated asset target {resolved}")


def _all_source_hashes_recorded(*report_groups: list[dict[str, Any]]) -> bool:
    reports = [report for group in report_groups for report in group]
    return bool(reports) and all(
        isinstance(report.get("sourceSha256"), str)
        and _SHA256_PATTERN.fullmatch(report["sourceSha256"]) is not None
        for report in reports
    )


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
