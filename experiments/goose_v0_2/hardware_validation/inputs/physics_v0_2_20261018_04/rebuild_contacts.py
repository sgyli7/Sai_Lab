"""Restore the adopted native lower gripping face inside the 14-leaf budget.

All other collision OBJ files and all physical/drive contracts remain unchanged.
The small native reference is independently mapped to whole05 world-zero space.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    parent = a.parent.resolve()
    reference = json.loads(a.contact_reference.read_text())
    points = np.asarray(reference["contact_face_world_zero_vertices_mm"], dtype=float) * .001
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("A real adopted contact-face boundary is required")
    contract = json.loads((parent / "physical_contract.json").read_text())
    origin = np.asarray(next(b["origin_world_zero_m"] for b in contract["bodies"]
                             if b["name"] == "beak_hinge"))
    mf = json.loads((parent / "collision_manifest.json").read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "meshes").mkdir()
    rows = copy.deepcopy(mf["collision_geometries"])
    for row in rows:
        path = parent / row["asset"]
        dest = a.output / "meshes" / path.name
        if row["name"] != "lower_bill_envelope":
            shutil.copyfile(path, dest)
        else:
            seed = a.lower_source_mesh if a.lower_source_mesh else path
            mesh = trimesh.load(seed, force="mesh", process=False)
            vertices = np.asarray(mesh.vertices).copy() + origin
            native_plane = float(points[:, 2].mean())
            plane = float(contract["grip_reference_zero_m"][2])
            if abs(plane - native_plane) > 1e-6:
                raise ValueError("Frozen grip datum differs from the true native plane")
            if np.ptp(points[:, 2]) > 1e-10:
                raise ValueError("Native zero-angle gripping face is not planar")
            points[:, 2] = plane
            # Parent vertices within one micron of the actual plane are only
            # tessellation/float noise, not a different mechanical surface.
            near = np.abs(vertices[:, 2] - plane) < 1e-6
            vertices[near, 2] = plane
            hull = trimesh.convex.convex_hull(np.vstack((vertices, points)))
            if len(hull.vertices) > 64:
                raise ValueError("The corrected single leaf exceeds 64 supports")
            hull.vertices -= origin
            hull.export(dest, include_normals=False, digits=12)
            row.update(vertices=len(hull.vertices), triangles=len(hull.faces),
                       construction="Parent lower jaw hull plus complete adopted native contact-face boundary",
                       native_contact_plane_mm=native_plane * 1000,
                       frozen_common_plane_roundoff_alignment_m=plane - native_plane,
                       task_approximation="True lower gripping boundary restored; common plane aligned within native/GLB float roundoff to frozen grip datum; jaw body remains a coarse convex role")
        row.update(asset="meshes/" + dest.name, sha256=sha(dest))
    result = dict(schema="goose_fourteen_native_lower_pad_revision_v1",
                  status="CANDIDATE_REQUIRES_INDEPENDENT_NATIVE_CONTACT_AND_RUNTIME_GATES",
                  hardware_parent="physics_v0_2_20261018_02",
                  collision_parent=parent.name,
                  parent_collision_manifest_sha256=sha(parent / "collision_manifest.json"),
                  parent_physical_contract_sha256=sha(parent / "physical_contract.json"),
                  parent_actuator_contract_sha256=sha(parent / "actuator_contract.json"),
                  producer_sha256=sha(__file__), collision_geometries=rows,
                  native_contact_reference=dict(path=str(a.contact_reference), sha256=sha(a.contact_reference)),
                  lower_seed_mesh_sha256=sha(a.lower_source_mesh if a.lower_source_mesh else parent / next(r["asset"] for r in mf["collision_geometries"] if r["name"] == "lower_bill_envelope")),
                  actual_leaf_count=len(rows),
                  maximum_source_convex_vertices=max(r["vertices"] for r in rows),
                  source_vertex_sum=sum(r["vertices"] for r in rows),
                  explicit_exclusions=mf["explicit_exclusions"],
                  explicit_contact_pairs=mf["source_collision_metadata"]["explicit_contact_pairs"],
                  collision_geometry_contributes_to_MCI=False,
                  all_other_thirteen_OBJ_bytes_unchanged=True)
    (a.output / "collision_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "collision_geometries"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--contact-reference", type=Path, required=True)
    p.add_argument("--lower-source-mesh", type=Path,
                   help="Optional preserved lower-jaw seed for rebuilding from the self-contained04 package")
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
