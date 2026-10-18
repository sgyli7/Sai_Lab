"""Finite, independent contact checks on the shipped Goose training model.

Uses only the delivered contracts and compiled MuJoCo model. These fixtures
verify collision participation and articulation gaps, not learned task success
or a certificate of the entire configuration space.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def identity(p):
    return dict(path=str(Path(p).resolve()), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())


def name(m, kind, i):
    return mujoco.mj_id2name(m, kind, int(i))


def owner(m, g):
    return name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])


def pair_policy(m, ga, gb):
    """Replicate the documented mask/body tests independently, for audit only."""
    key = tuple(sorted((ga, gb)))
    if key in {tuple(sorted((int(x), int(y)))) for x, y in zip(m.pair_geom1, m.pair_geom2)}:
        return True, "explicit_geom_pair"
    a, b = map(int, (m.geom_bodyid[ga], m.geom_bodyid[gb]))
    if a == b:
        return False, "same_rigid_body"
    if ((min(a, b) << 16) | max(a, b)) in set(map(int, m.exclude_signature)):
        return False, "explicit_entire_body_exclusion"
    if not (int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)):
        if a and b and (m.body_parentid[a] == b or m.body_parentid[b] == a):
            return False, "simulator_parent_filter"
    allowed = bool((m.geom_contype[ga] & m.geom_conaffinity[gb])
                   or (m.geom_contype[gb] & m.geom_conaffinity[ga]))
    return allowed, "geom_masks" if not allowed else "ordinary_collision_enabled"


def pose(m, d, c, q, root_z=.5):
    mujoco.mj_resetData(m, d)
    d.qpos[:3] = [0, 0, root_z]
    d.qpos[3:7] = [1, 0, 0, 0]
    for j in c["joints"]:
        d.qpos[m.joint(j["name"]).qposadr[0]] = q.get(j["name"], 0.)
    for j in c["passive_linkage_joints"]:
        d.qpos[m.joint(j["name"]).qposadr[0]] = (
            j["mimic_multiplier"] * d.qpos[m.joint(j["mimic_joint"]).qposadr[0]]
            + j["mimic_offset_rad"])
    mujoco.mj_forward(m, d)


def contacts(m, d):
    return [dict(a=name(m, mujoco.mjtObj.mjOBJ_GEOM, x.geom1),
                 b=name(m, mujoco.mjtObj.mjOBJ_GEOM, x.geom2),
                 owner_a=owner(m, x.geom1), owner_b=owner(m, x.geom2),
                 distance_m=float(x.dist), position_m=x.pos.tolist(),
                 pair_policy=pair_policy(m, int(x.geom1), int(x.geom2))) for x in d.contact]


def mouth_object_fixtures(package, c):
    """A real 50g 20x20x16mm free box tests the saved external mouth domain."""
    root = ET.parse(package / "robot.xml").getroot()
    assets = {e.get("file"): (package / e.get("file")).read_bytes()
              for e in root.findall(".//*[@file]")}
    body = ET.SubElement(root.find("worldbody"), "body", name="g_fixture_object", pos="0 0 1")
    ET.SubElement(body, "freejoint", name="g_fixture_object_free")
    moments = .05 / 12 * np.array([.02**2 + .016**2, .02**2 + .016**2, .02**2 + .02**2])
    ET.SubElement(body, "inertial", pos="0 0 0", mass=".05", diaginertia=" ".join(map(str, moments)))
    ET.SubElement(body, "geom", name="g_fixture_object_geom", type="box", size=".01 .01 .008",
                  contype="1", conaffinity="2147483647", friction=".8 .005 .0001")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)
    d = mujoco.MjData(m)
    object_q = int(m.joint("g_fixture_object_free").qposadr[0])
    plane = float(c["grip_reference_zero_m"][2])
    rows = []
    for label, angle, z, expected in (
        ("open_mouth_real_box_clear", .55, plane - .009, "clear"),
        ("upper_contact_plane_positive", .55, plane - .007, "upper"),
        ("lower_contact_plane_positive", .24, plane - .009, "lower"),
    ):
        pose(m, d, c, dict(beak_hinge=angle), root_z=.5)
        d.qpos[object_q:object_q+7] = [.21, 0., z+.5, 1., 0., 0., 0.]
        mujoco.mj_forward(m, d)
        actual = [x for x in contacts(m, d) if "g_fixture_object" in (x["owner_a"], x["owner_b"])]
        interfering = [x for x in actual if x["distance_m"] < -.00025]
        if expected == "clear":
            passed = not interfering
        else:
            desired_owner = "head_roll" if expected == "upper" else "beak_hinge"
            passed = any(desired_owner in (x["owner_a"], x["owner_b"]) and x["distance_m"] < 0
                         for x in actual)
        rows.append(dict(name=label, beak_q_rad=angle, box_mass_kg=.05,
                         box_dimensions_m=[.02, .02, .016], box_position_m=[.21, 0., z+.5],
                         source_pad_contact_plane_zero_m=plane,
                         actual_object_contacts=actual, expected=expected, passed=passed,
                         object_dynamic_but_no_policy_or_lift_inferred=True))
    return rows


def native_lower_face_fixture(package, m, d, c):
    """Compare the actual compiled flat grip perimeter with adopted native CAD.

    The small BREP's exact face extraction is audited separately using CAD.
    This portable runtime check binds that BREP and the eight-edge reference,
    then measures all1440 stated planar support directions without CAD modules.
    """
    reference = package / "source_inputs/native_contact/lower_pad_contact_face.json"
    if not reference.is_file():
        return None
    ref = json.loads(reference.read_text())
    brep = reference.parent / "lower_grip_cassette_pad.brep"
    if identity(brep)["sha256"] != ref["source_brep_sha256"]:
        raise AssertionError("Native lower pad BREP reference changed")
    points = np.asarray(ref["contact_face_world_zero_vertices_mm"], dtype=float)
    if points.shape != (8, 3) or np.ptp(points[:, 2]) > 1e-8:
        raise AssertionError("Expected the exact eight-edge planar native pad reference")
    area = abs(float(np.dot(points[:,0], np.roll(points[:,1], -1))
                     - np.dot(points[:,1], np.roll(points[:,0], -1)))) / 2
    if abs(area - float(ref["source_face_area_mm2"])) > 1e-6:
        raise AssertionError("Native perimeter does not match the recorded full face area")
    pose(m, d, c, {}, root_z=0.)
    g = m.geom("lower_bill_envelope").id
    mid = int(m.geom_dataid[g]); first = int(m.mesh_vertadr[mid]); n = int(m.mesh_vertnum[mid])
    world = (m.mesh_vert[first:first+n] @ d.geom_xmat[g].reshape(3,3).T + d.geom_xpos[g]) * 1000
    z = float(points[0, 2])
    flat = world[np.abs(world[:,2] - z) < .01]
    if len(flat) < 3:
        raise AssertionError("Compiled lower jaw has no full planar contact face")
    theta = np.linspace(0., 2*np.pi, 1440, endpoint=False)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    native_support = np.max(points[:,:2] @ directions.T, axis=0)
    proxy_support = np.max(flat[:,:2] @ directions.T, axis=0)
    loss, excess = native_support-proxy_support, proxy_support-native_support
    # Also test each exact native corner against every edge of the planar
    # convex mesh support polygon; directions alone are not an area proof.
    ordered = sorted(set(map(tuple, flat[:,:2])))
    def cross(a,b,p): return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
    lo, hi = [], []
    for point in ordered:
        while len(lo)>1 and cross(lo[-2],lo[-1],point)<=0: lo.pop()
        lo.append(point)
    for point in reversed(ordered):
        while len(hi)>1 and cross(hi[-2],hi[-1],point)<=0: hi.pop()
        hi.append(point)
    hull = np.asarray(lo[:-1]+hi[:-1])
    signed = []
    for a,b in zip(hull, np.roll(hull,-1,axis=0)):
        signed.extend(cross(a,b,p)/np.linalg.norm(b-a) for p in points[:,:2])
    worst_outside = max(0., -min(signed))
    error = max(float(loss.max()),float(excess.max()),worst_outside)
    passed = bool(error <= .05 and float(np.max(np.abs(flat[:,2]-z))) <= .01)
    return dict(reference=identity(reference), source_brep=identity(brep),
                source_face_index=ref["native_contact_face_index"], native_face_area_mm2=area,
                native_contact_vertices_mm=points.tolist(), loaded_coplanar_vertices_mm=flat.tolist(),
                maximum_plane_error_mm=float(np.max(np.abs(flat[:,2]-z))),
                directions=1440, max_support_loss_mm=float(loss.max()),
                max_support_excess_mm=float(excess.max()),
                maximum_native_corner_outside_proxy_edge_mm=worst_outside,
                accepted_support_error_mm=.05, passed=passed,
                scope="Full exact eight-edge zero-angle native contact face; finite1440 directional supports and exact corner-to-edge containment. CAD BREP face extraction independently audited separately; no full-jaw/whole-ROM or material-compliance claim.")


def run(package):
    c = json.loads((package / "physical_contract.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(package / "robot.xml"))
    d = mujoco.MjData(m)
    robot = [g for g in range(m.ngeom) if m.geom_bodyid[g]]
    floor = [g for g in range(m.ngeom) if not m.geom_bodyid[g]
             and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    if len(floor) != 1:
        raise AssertionError("Expected one ordinary ground plane")
    ground = floor[0]
    if any(not pair_policy(m, ground, g)[0] for g in robot):
        raise AssertionError("An exterior robot collider is hidden from the environment")
    mask_groups = {}
    for g in robot:
        k = (owner(m, g), int(m.geom_contype[g]), int(m.geom_conaffinity[g]))
        mask_groups[k] = mask_groups.get(k, 0) + 1
    exclusions = []
    for sig in m.exclude_signature:
        exclusions.append([name(m, mujoco.mjtObj.mjOBJ_BODY, int(sig) >> 16),
                           name(m, mujoco.mjtObj.mjOBJ_BODY, int(sig) & 65535)])
    # The low-leaf model omits internal axis carriers as exterior roles. Its
    # approved adjacency exceptions must be exact, mechanically adjacent after
    # contracting only those empty carriers, and must not hide any other role.
    role_adjacency = []
    low_leaf_model = len(robot) == 14
    if low_leaf_model:
        approved = {frozenset(x) for x in (("torso", "neck_pitch"),
                    ("torso", "left_hip_roll"), ("torso", "right_hip_roll"),
                    ("neck_mid_pitch", "head_roll"),
                    ("left_knee_pitch", "left_ankle_roll"),
                    ("right_knee_pitch", "right_ankle_roll"))}
        if {frozenset(x) for x in exclusions} != approved:
            raise AssertionError("Low-leaf adjacency exclusions differ from approved six")
        exterior = set(map(int, m.geom_bodyid[robot]))
        for a, b in exclusions:
            ia, ib = m.body(a).id, m.body(b).id
            proven = False
            for child, parent in ((ia, ib), (ib, ia)):
                n = int(m.body_parentid[child]); skipped = []
                while n and n not in exterior:
                    skipped.append(m.body(n).name); n = int(m.body_parentid[n])
                if n == parent:
                    role_adjacency.append(dict(owners=[a, b], child=m.body(child).name,
                                               nearest_external_parent=m.body(parent).name,
                                               skipped_internal_axis_carriers=skipped))
                    proven = True; break
            if not proven:
                raise AssertionError("Excluded role pair is not external adjacency: " + str((a,b)))
        for i, ga in enumerate(robot):
            for gb in robot[i+1:]:
                a, b = owner(m, ga), owner(m, gb)
                if a == b or frozenset((a,b)) in approved:
                    continue
                enabled, reason = pair_policy(m, ga, gb)
                if not enabled and reason != "simulator_parent_filter":
                    raise AssertionError("Non-adjacent exterior role hidden: " + a + "/" + b)
    critical = [("torso", "neck_pitch"), ("torso", "left_hip_pitch"),
                ("torso", "right_hip_pitch"), ("left_ankle_roll", "right_ankle_roll")]
    critical_policy = []
    for a, b in critical:
        groups_a = [g for g in robot if owner(m, g) == a]
        groups_b = [g for g in robot if owner(m, g) == b]
        enabled = sum(pair_policy(m, ga, gb)[0] for ga in groups_a for gb in groups_b)
        critical_policy.append(dict(owners=[a, b], enabled_geom_pairs=enabled,
                                    total_geom_pairs=len(groups_a) * len(groups_b)))
        approved_adjacent = low_leaf_model and frozenset((a,b)) in approved
        critical_policy[-1]["approved_external_adjacency_exception"] = approved_adjacent
        if groups_a and groups_b and not enabled and not approved_adjacent:
            raise AssertionError("Critical entire role pair hidden: " + a + "/" + b)
    cases = dict(zero={}, neck_015=dict(neck_pitch=.15, neck_mid_pitch=-.15),
                 neck_030=dict(neck_pitch=.3, neck_mid_pitch=-.3),
                 neck_050=dict(neck_pitch=.5, neck_mid_pitch=-.5),
                 deliberately_crossed_feet=dict(left_hip_roll=-.5, right_hip_roll=.5),
                 right_positive_HP_070=dict(right_hip_pitch=.7),
                 left_positive_HP_070=dict(left_hip_pitch=.7), mouth_open=dict(beak_hinge=.55))
    rows = []
    for label, q in cases.items():
        pose(m, d, c, q)
        cc = contacts(m, d)
        unexpected = [x for x in cc if set((x["owner_a"], x["owner_b"])) in
                      ({"torso", "neck_pitch"}, {"torso", "left_hip_pitch"},
                       {"torso", "right_hip_pitch"}) and x["distance_m"] < -.00025]
        clear_fixture = label in ("zero", "neck_015", "neck_030", "neck_050")
        foot = [x for x in cc if any(x["owner_a"].startswith(s) and x["owner_b"].startswith(t)
                                    for s, t in (("left_ankle", "right_ankle"),
                                                 ("right_ankle", "left_ankle")))]
        body_k = None
        if "positive_HP" in label:
            side = label.split("_")[0]
            body_k = [x for x in cc if set((x["owner_a"], x["owner_b"])) ==
                      {"torso", side + "_knee_pitch"} and x["distance_m"] < 0]
        rows.append(dict(name=label, q_rad=q, root_z_m=.5, contacts=cc,
                         native_clear_pair_fixture=clear_fixture,
                         native_clear_pair_unexpected_contacts=unexpected if clear_fixture else None,
                         crossed_foot_contacts=foot if label == "deliberately_crossed_feet" else None,
                         body_k_owner_positive_contacts=body_k))
    # Ordinary ground contact: find the actual loaded soles' support plane, then
    # lower it 1mm. This is a kinematic fixture, never a runtime state projection.
    pose(m, d, c, {}, root_z=0.)
    sole_bounds = {}
    for side in ("left", "right"):
        points = []
        for g in robot:
            if owner(m, g) != side + "_ankle_roll" or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = int(m.geom_dataid[g])
            start, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            points.append(m.mesh_vert[start:start+n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        if not points:
            raise AssertionError("No actual sole mesh for " + side)
        points = np.vstack(points)
        sole_bounds[side] = [points.min(0).tolist(), points.max(0).tolist()]
    floor_root_z = -.001 - min(x[0][2] for x in sole_bounds.values())
    pose(m, d, c, {}, root_z=floor_root_z)
    cc = contacts(m, d)
    ground_contacts = [x for x in cc if x["owner_a"] == "world" or x["owner_b"] == "world"]
    sides = {s: [x for x in ground_contacts if s + "_ankle_roll" in (x["owner_a"], x["owner_b"])]
             for s in ("left", "right")}
    failures = []
    for row in rows:
        if row["native_clear_pair_fixture"] and row["native_clear_pair_unexpected_contacts"]:
            failures.append(row["name"] + ": false working-space contact")
        if row["name"] == "deliberately_crossed_feet" and not row["crossed_foot_contacts"]:
            failures.append("Crossed foot positive contact missing")
        if "positive_HP" in row["name"] and not row["body_k_owner_positive_contacts"]:
            failures.append(row["name"] + ": body/K-owner positive participation missing")
    for s, cc in sides.items():
        if not cc:
            failures.append(s + ": actual sole-ground contact missing")
    # Record closed/open pad distances and actual named collision assets. Do not
    # infer a grasp from an endpoint or invent a success criterion for a policy.
    pad = [g for g in robot if any(s in (m.geom(g).name or "").lower()
                                  for s in ("pad", "upper_bill", "lower_bill"))]
    mouth = []
    for q in (0., .274891, .55):
        pose(m, d, c, dict(beak_hinge=q))
        pad_pairs = []
        for i, ga in enumerate(pad):
            for gb in pad[i + 1:]:
                if m.geom_bodyid[ga] == m.geom_bodyid[gb]:
                    continue
                segment = np.zeros(6)
                dist = mujoco.mj_geomDistance(m, d, ga, gb, 1., segment)
                pad_pairs.append(dict(a=m.geom(ga).name, b=m.geom(gb).name,
                                      distance_m=float(dist), witnesses_m=segment.tolist(),
                                      policy=pair_policy(m, ga, gb)))
        mouth.append(dict(beak_hinge_rad=q, pad_pairs=pad_pairs))
    object_fixtures = mouth_object_fixtures(package, c)
    failures += [x["name"] for x in object_fixtures if not x["passed"]]
    native_face = native_lower_face_fixture(package, m, d, c)
    if native_face is not None and not native_face["passed"]:
        failures.append("Native lower-pad complete flat-face perimeter mismatch")
    return dict(schema="goose_independent_finite_training_contact_gate_v1",
                model=identity(package / "robot.xml"), script=identity(__file__),
                status="PASS_FINITE_FIXTURES" if not failures else "REJECT_FINITE_FIXTURES",
                failures=failures, actual_robot_collision_leaves=len(robot),
                mesh_support_vertices=sum(int(m.mesh_vertnum[m.geom_dataid[g]]) for g in robot
                                          if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH),
                unique_mesh_support_vertices=int(sum(m.mesh_vertnum)),
                primitive_leaves=sum(m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH for g in robot),
                all_robot_colliders_environment_enabled=True,
                bodywide_exclusions=exclusions,
                independently_verified_external_adjacency_exceptions=role_adjacency,
                owner_mask_groups=[dict(owner=k[0], contype=k[1], conaffinity=k[2], leaves=v)
                                   for k, v in mask_groups.items()], critical_pair_policy=critical_policy,
                finite_poses=rows, actual_sole_ground_positive=sides,
                loaded_sole_bounds_zero_m=sole_bounds, sole_positive_fixture_root_z_m=floor_root_z,
                mouth_endpoint_distances=mouth, real_object_mouth_fixtures=object_fixtures,
                native_lower_contact_face=native_face,
                limitations=["Finite articulation gap and collision participation tests only.",
                             "No learned getup, movement, grasp success or manufacture qualification.",
                             "Positive HP probes are diagnostics until native-source contact is confirmed."])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(a.package.resolve())
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(a.output), status=result["status"], failures=result["failures"],
                          leaves=result["actual_robot_collision_leaves"],
                          supports=result["mesh_support_vertices"])))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
