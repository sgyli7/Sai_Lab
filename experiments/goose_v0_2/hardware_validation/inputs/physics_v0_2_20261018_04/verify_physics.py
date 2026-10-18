"""Independent release gate for the fixed Goose physical-data package.

Recomputes M/COM/fullI without importing its producer, then reads the compiled
model and exercises the packaged adapter. No task, policy, training or CAD.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


JAW_OWNERS = {
    **dict.fromkeys(("beak_native_input_flange", "beak_input_steel_sleeve",
                     "beak_input_washer", "beak_input_button_screw"), "beak_input_rotor"),
    **dict.fromkeys(("beak_native_coupler", "beak_input_bush_candidate",
                     "beak_output_bush_candidate"), "beak_coupler_link"),
}


def identity(p):
    p = Path(p).resolve()
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def load(p):
    return json.loads(Path(p).read_text())


def combined(rows):
    masses = np.array([r["mass_kg"] for r in rows], dtype=float)
    centres = np.array([r["centre_zero_mm"] for r in rows], dtype=float) / 1000
    m = float(masses.sum())
    centre = np.einsum("i,ij->j", masses, centres) / m
    tensor = np.sum([np.asarray(r["inertia_at_com_zero_kg_m2"], dtype=float)
                     + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))
                     for r, mass, d in zip(rows, masses, centres - centre)], axis=0)
    return m, centre, (tensor + tensor.T) / 2


def check(ok, text):
    if not ok:
        raise AssertionError(text)


def source_input_path(package, source_root, reference):
    rel = reference["repository_path"]
    if source_root is not None and (source_root / rel).is_file():
        p = source_root / rel
    else:
        snapshot = load(package / "source_snapshot.json")
        entries = snapshot.get("embedded_source_contracts", {})
        if isinstance(entries, list):
            entries = {e["repository_path"]: e for e in entries}
        entry = entries.get(rel)
        check(entry is not None, "No source or exact embedded source for " + rel)
        p = (package / entry["package_path"]).resolve()
        check(p.is_relative_to(package), "Embedded source escapes package")
        check(entry["sha256"] == reference["sha256"], "Embedded mapping source identity differs")
    check(identity(p)["sha256"] == reference["sha256"], "Source bytes differ: " + rel)
    return p


def source_patch_expected(package, source_root, contract):
    """Rebuild the selected once-only E operations directly from source input."""
    refs = {Path(r["repository_path"]).name: r for r in contract["sources"]}
    base_ref = refs["conditional_mass_ledger.json"]
    e_ref = refs["E_simulation_mass_contract.json"]
    base_path = source_input_path(package, source_root, base_ref)
    e_path = source_input_path(package, source_root, e_ref)
    check(identity(base_path)["sha256"] == base_ref["sha256"], "Base ledger hash")
    check(identity(e_path)["sha256"] == e_ref["sha256"], "E source hash")
    rows = copy.deepcopy(load(base_path)["items"])
    before = sum(r["mass_kg"] for r in rows)
    e = load(e_path)
    for op in e["operations_for_F_to_apply_once"]:
        r = op["new"]
        m = r["mass_kg"]
        centre = (r["com_world_zero_mm"] if "com_world_zero_mm" in r
                  else (np.array(r["center_of_mass_world_m"]) * 1000).tolist())
        tensor = r.get("uniform_or_actual_inertia_com_world_kg_m2",
                       r.get("inertia_at_com_world_kg_m2"))
        if "old" in op:
            ix = [i for i, a in enumerate(rows) if a["name"] == op["old"]["name"]]
            check(len(ix) == 1, "Cap must be replaced once")
            item = rows[ix[0]]
            check(abs(item["mass_kg"] - op["old"]["mass_kg"]) < 1e-12, "Wrong old cap")
            item.update(mass_kg=m, centre_zero_mm=centre, inertia_at_com_zero_kg_m2=tensor)
        else:
            name = "engineering_allocation/" + r["name"]
            check(not any(a["name"] == name for a in rows), "Duplicate E addition")
            rows.append(dict(name=name, body=r["body"], mass_kg=m,
                             centre_zero_mm=centre, inertia_at_com_zero_kg_m2=tensor))
    for r in e["remaining_original_E25_engineering_allocations"]:
        name = "engineering_allocation/" + r["name"]
        check(not any(a["name"] == name for a in rows), "Duplicate allocation")
        rows.append(dict(name=name, body=r["body"], mass_kg=r["mass_kg"],
                         centre_zero_mm=(np.array(r["center_of_mass_world_m"]) * 1000).tolist(),
                         inertia_at_com_zero_kg_m2=r["inertia_at_com_world_kg_m2"]))
    delta = sum(r["mass_kg"] for r in rows) - before
    check(abs(delta - e["accounting"]["E_only_net_delta_vs_whole05_kg"]) < 1e-11,
          "Independent E once-only delta mismatch")
    result = {"source_inputs": [identity(base_path), identity(e_path)],
              "baseline_mass_kg": before, "once_delta_kg": delta}
    reference = contract.get("selected_material_revision")
    if reference:
        material_path = source_input_path(package, source_root, reference)
        check(identity(material_path)["sha256"] == reference["sha256"], "Selected material source hash")
        packet = load(material_path)
        parts = packet["parts"]
        names = [p["name"] for p in parts]
        check(len(names) == len(set(names)) == 6, "Expected unique six-part bounded material revision")
        carrier_names = {"left_hip_roll_carrier_arm", "right_hip_roll_carrier_arm"}
        check(carrier_names <= set(names), "Missing actual paired carrier material replacement")
        pre_material_mass = sum(r["mass_kg"] for r in rows)
        applied = []
        for p in parts:
            found = [r for r in rows if r["name"] == p["name"]]
            check(len(found) == 1, "Material source must replace an existing name exactly once")
            old = copy.deepcopy(found[0])
            check(old["body"] == p["body"], "Material revision changed rigid owner")
            new_centre = np.array(p["center_of_mass_world_m"])
            new_tensor = np.array(p["inertia_at_com_world_kg_m2"])
            native = p.get("native", dict(path=p.get("native_brep_path"),
                                         sha256=p.get("native_brep_sha256")))
            if source_root is not None and native.get("path") and native.get("sha256"):
                native_path = Path(native["path"])
                if not native_path.is_absolute():
                    native_path = source_root / native_path
                check(identity(native_path)["sha256"] == native["sha256"],
                      "Selected actual native geometry identity differs: " + p["name"])
            if p["name"] in carrier_names:
                check(p["density_kg_m3"] == 2810., "Unapproved carrier material density")
                ratio = 2810. / 7850.
                check(abs(p["mass_kg"] - old["mass_kg"] * ratio) < 1e-10,
                      "Unchanged carrier geometry mass does not follow actual material")
                check(np.max(abs(new_centre - np.array(old["centre_zero_mm"])/1000)) < 1e-10,
                      "Same-native carrier COM changed")
                check(np.max(abs(new_tensor - np.array(old["inertia_at_com_zero_kg_m2"])*ratio)) < 1e-10,
                      "Same-native carrier full inertia did not scale with selected material")
            eig = np.linalg.eigvalsh(new_tensor)
            check(p["mass_kg"] > 0 and eig[0] > 0 and eig[2] <= eig[0]+eig[1]+1e-10,
                  "Selected part MCI is not physical")
            found[0].update(mass_kg=p["mass_kg"], centre_zero_mm=(new_centre*1000).tolist(),
                            inertia_at_com_zero_kg_m2=new_tensor.tolist())
            applied.append(dict(name=p["name"], old_mass_kg=old["mass_kg"],
                                new_mass_kg=p["mass_kg"], delta_kg=p["mass_kg"]-old["mass_kg"],
                                actual_native=native))
        result["selected_material_revision"] = dict(
            source=identity(material_path), parts=applied,
            baseline_with_E_kg=pre_material_mass,
            net_delta_kg=sum(r["mass_kg"] for r in rows)-pre_material_mass,
            carrier_same_native_density_and_full_tensor_scaling=True,
            washer_and_bolt_geometry_or_location_changes_from_actual_source=True,
            body_and_source_name_changes=False)
    return rows, result


def contract_check(package, source_root=None):
    c, drive, ledger = [load(package / n) for n in
                        ("physical_contract.json", "actuator_contract.json", "mass_ledger.json")]
    rows = ledger["items"]
    if c.get("selected_material_revision"):
        check(identity(package / "material_mass_patch.json")["sha256"] ==
              c["selected_material_revision"]["sha256"], "Packaged material source not verbatim selected source")
    check(len({r["name"] for r in rows}) == len(rows), "Duplicate ledger names")
    source_result = None
    has_embedded = (package / "source_snapshot.json").is_file() and bool(
        load(package / "source_snapshot.json").get("embedded_source_contracts"))
    if source_root is not None or has_embedded:
        expected, source_result = source_patch_expected(package, source_root, c)
        a, b = {r["name"]: r for r in rows}, {r["name"]: r for r in expected}
        check(set(a) == set(b), "Ledger source names/additions differ")
        worst = dict(mass_kg=0., com_m=0., fullI_kg_m2=0.)
        for name, item in b.items():
            got = a[name]
            check(got["body"] == item["body"], "Ledger item owner differs: " + name)
            for k, field, scale in (("mass_kg", "mass_kg", 1),
                                    ("com_m", "centre_zero_mm", .001),
                                    ("fullI_kg_m2", "inertia_at_com_zero_kg_m2", 1)):
                err = float(np.max(np.abs(np.array(got[field]) - item[field]))) * scale
                worst[k] = max(worst[k], err)
        check(max(worst.values()) < 1e-11, "Source part MCI changed")
        source_result["maximum_unchanged_or_selected_item_error"] = worst
        si_ref = next(r for r in c["sources"] if r["repository_path"].endswith("whole05_full_SI_contract.json"))
        si_path = source_input_path(package, source_root, si_ref)
        check(identity(si_path)["sha256"] == si_ref["sha256"], "Original SI source hash")
        original = load(si_path)
        check(c["joint_order"] == original["joint_order"], "Original public action order changed")
        for old, new in zip(original["joints"], c["joints"]):
            expected = copy.deepcopy(old)
            if expected["name"] == "head_roll":
                expected["frictionloss_nm"] = .036
            check(expected == new, "Source axis/limit/drive changed: " + new["name"])
        for old, new in zip(original["passive_linkage_joints"], c["passive_linkage_joints"]):
            expected = copy.deepcopy(old)
            if expected["name"] == "beak_coupler_link":
                expected.update(physical_angular_stop=False,
                                range_rad_is_nominal_linkage_branch_not_separate_hard_stop=True)
            check(expected == new, "Source passive axis changed: " + new["name"])
        source_result["original_axis_limit_drive_source"] = identity(si_path)
    reassigned = copy.deepcopy(rows)
    for item in reassigned:
        leaf = item["name"].rsplit("/", 1)[-1]
        if leaf in JAW_OWNERS:
            item["body"] = JAW_OWNERS[leaf]
    owner_names = {r["body"] for r in reassigned}
    check(owner_names == {b["name"] for b in c["bodies"]}, "Physical body coverage differs")
    bresult = []
    for b in c["bodies"]:
        m, com, I = combined([r for r in reassigned if r["body"] == b["name"]])
        origin = np.array(b["origin_world_zero_m"])
        err = dict(mass_kg=abs(m - b["mass_kg"]),
                   COM_m=float(np.max(np.abs(com - origin - b["com_local_m"]))),
                   fullI_kg_m2=float(np.max(np.abs(I - b["inertia_at_com_body_kg_m2"])) ))
        eig = np.linalg.eigvalsh(I)
        check(eig[0] > 0 and eig[2] <= eig[0] + eig[1] + 1e-10, "Nonphysical body inertia")
        check(max(err.values()) < 1e-10, "Independent body MCI mismatch: " + b["name"])
        bresult.append(dict(body=b["name"], mass_kg=m, COM_zero_m=com.tolist(),
                            inertia_eigenvalues=eig.tolist(), error=err))
    mass, com, I = combined(reassigned)
    check(abs(mass - c["nominal_robot_mass_kg"]) < 1e-10, "Total mass mismatch")
    names = c["joint_order"]
    check(len(names) == len(set(names)) == 18 and names == [j["name"] for j in c["joints"]],
          "Public action order differs")
    check(len(c["passive_linkage_joints"]) == 4 and len(c["bodies"]) == 23, "Internal counts")
    check([r["name"] for r in drive["axes"]] == names, "Drive names/order")
    check(c["physics_dt_s"] == .005 and c["policy_dt_s"] == .02 and c["torque_dt_s"] == .005,
          "Clock contract")
    check(drive["default_mode"] == "continuous_only" and not drive["peak_output_enabled"],
          "Unexpected unlimited-peak profile")
    check(drive["positive_mechanical_power_limit_w"] == 350., "Power profile changed")
    for j, a in zip(c["joints"], drive["axes"]):
        check(abs(np.linalg.norm(j["axis_parent"]) - 1) < 1e-12, "Nonunit axis")
        check(j["range_rad"][0] < j["range_rad"][1], "Invalid joint range")
        for fk, ak in (("continuous_design_limit_nm", "selected_continuous_design_limit_nm"),
                       ("torque_peak_limit_nm", "selected_peak_design_limit_nm"),
                       ("speed_limit_rad_s", "selected_speed_limit_rad_s")):
            check(j[fk] == a[ak] and j[fk] > 0, "Drive/physical inconsistency: " + j["name"])
        check(j["continuous_design_limit_nm"] <= j["torque_peak_limit_nm"], "Continuous above peak")
    hr = next(j for j in c["joints"] if j["name"] == "head_roll")
    check(abs(hr["frictionloss_nm"] - .036) < 1e-12 and hr["damping_nm_s_rad"] == .01,
          "CRBT loss must be servo .006 + bearing .03 once")
    return c, drive, dict(status="PASS", source_patch=source_result, total_mass_kg=mass,
                          COM_zero_m=com.tolist(), fullI_at_whole_COM_zero=I.tolist(), bodies=bresult,
                          engineering_allocations_not_weighed_or_installed=True)


def quaternion_matrix(q):
    # MuJoCo inertial quaternions are wxyz, independent of source eigensolver.
    w, x, y, z = q
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def runtime_check(package, c, drive, adapter_file):
    import mujoco
    xml = package / "robot.xml"
    for a in ET.parse(xml).getroot().findall(".//*[@file]"):
        rel = Path(a.get("file"))
        check(not rel.is_absolute() and (package / rel).resolve().is_relative_to(package),
              "Runtime asset escapes package")
        check((package / rel).is_file(), "Missing relative asset: " + str(rel))
    start = time.perf_counter()
    m = mujoco.MjModel.from_xml_path(str(xml))
    compiled_ms = (time.perf_counter() - start) * 1000
    check(m.nu == 18 and m.njnt == 23 and m.nbody == 24, "Compiled counts differ")
    check(abs(m.opt.timestep - .005) < 1e-12, "Compiled timestep differs")
    check(np.allclose(m.opt.gravity, [0, 0, -9.81]), "Gravity is not normal")
    zero = mujoco.MjData(m)
    zero.qpos[:] = 0
    zero.qpos[3] = 1
    mujoco.mj_forward(m, zero)
    body_errors = []
    for b in c["bodies"]:
        v = m.body(b["name"])
        rot = quaternion_matrix(v.iquat)
        err = dict(body=b["name"], mass_kg=abs(float(v.mass[0]) - b["mass_kg"]),
                   COM_m=float(np.max(abs(v.ipos - b["com_local_m"]))),
                   fullI_kg_m2=float(np.max(abs(rot @ np.diag(v.inertia) @ rot.T
                                               - b["inertia_at_com_body_kg_m2"]))),
                   origin_zero_m=float(np.max(abs(zero.xpos[v.id] - b["origin_world_zero_m"]))))
        check(max(err[k] for k in ("mass_kg", "COM_m", "fullI_kg_m2", "origin_zero_m")) < 1e-10,
              "Compiled mass/tensor differs: " + b["name"])
        check(np.max(abs(zero.xmat[v.id].reshape(3, 3) - np.eye(3))) < 1e-10,
              "Source zero body orientation differs: " + b["name"])
        body_errors.append(err)
    allj = c["joints"] + c["passive_linkage_joints"]
    joint_rows = []
    for j in allj:
        i = m.joint(j["name"]).id
        check(np.max(abs(m.jnt_axis[i] - j["axis_parent"])) < 1e-12, "Compiled axis differs")
        check(np.max(abs(m.jnt_range[i] - j["range_rad"])) < 1e-12, "Compiled range differs")
        check(np.max(abs(zero.xanchor[i] - j["pivot_world_at_zero_m"])) < 1e-10,
              "Compiled joint pivot differs: " + j["name"])
        expected_parent = m.body(j["parent"]).id
        check(int(m.body_parentid[m.jnt_bodyid[i]]) == expected_parent,
              "Compiled articulated parent differs: " + j["name"])
        if j.get("physical_angular_stop") is False:
            check(not m.jnt_limited[i], "Dependent coupler incorrectly hard-stopped")
        dof = int(m.jnt_dofadr[i])
        joint_rows.append(dict(name=j["name"], qpos_index=int(m.jnt_qposadr[i]),
                               armature=float(m.dof_armature[dof]), frictionloss=float(m.dof_frictionloss[dof]),
                               damping=float(m.dof_damping[dof])))
        check(abs(m.dof_frictionloss[dof] - j.get("frictionloss_nm", 0)) < 1e-12,
              "Compiled friction differs: " + j["name"])
        check(abs(m.dof_armature[dof] - j.get("armature_kg_m2", 0)) < 1e-12,
              "Compiled armature differs: " + j["name"])
    mapping = []
    for i, j in enumerate(c["joints"]):
        target = j.get("actuation_joint", j["name"])
        check(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0])) == target,
              "Actual motor/input binding differs")
        check(np.max(abs(m.actuator_gear[i] - [1, 0, 0, 0, 0, 0])) < 1e-12,
              "Gear ratio multiplied twice")
        cont = j["continuous_design_limit_nm"]
        check(m.actuator_ctrllimited[i] and np.max(abs(m.actuator_ctrlrange[i] - [-cont, cont])) < 1e-12,
              "Native continuous control clamp differs")
        check(m.actuator_forcelimited[i] and np.max(abs(m.actuator_forcerange[i] - [-cont, cont])) < 1e-12,
              "Native continuous force clamp differs")
        mapping.append(dict(public=j["name"], driven=target, actuator_index=i))
    spec = importlib.util.spec_from_file_location("independent_packaged_adapter", adapter_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = mod.FrozenPhysics(package)
    check(env.ticks == 4 and env.substeps == 1, "Adapter effective clock")
    # Limiter checks exercise actual package code, without changing physical state.
    cap = np.array([j["continuous_design_limit_nm"] for j in c["joints"]])
    curves = drive["axes"]
    limit_rows = []
    for voltage in (20., 22.2, 24., 25.2):
        e = mod.FrozenPhysics(package, bus_voltage_v=voltage)
        v = np.array([a["selected_torque_speed_curve_rad_s_nm"][-1][0] * .85 for a in curves])
        tau = e.limit_torque(np.full(18, 1e4), v)
        check(np.all(abs(tau) <= cap + 1e-12), "Continuous cap exceeded")
        power = float(np.maximum(tau * v, 0).sum())
        check(power <= 350 + 1e-8, "Positive power cap exceeded")
        for i, a in enumerate(curves):
            curve = np.array(a["selected_torque_speed_curve_rad_s_nm"], float)
            factor = 1. if a["name"] == "head_roll" else min(voltage/24., 1.)
            available = np.interp(abs(v[i]), curve[:, 0] * factor, curve[:, 1], right=0)
            check(abs(tau[i]) <= min(cap[i], available) + 1e-10, "Voltage/curve exceeded")
        desired = np.clip(e.neutral + e.scale, e.ranges[:, 0], e.ranges[:, 1])
        factor = np.array([1. if a["reference_voltage_v"] == 5. else min(voltage/24., 1.) for a in curves])
        expected_target = e.neutral + np.clip(desired - e.neutral, -e.speed*factor*.02, e.speed*factor*.02)
        e.step(np.ones(18))
        check(np.max(abs(e.target - expected_target)) < 1e-12, "Actual 200Hz target slew differs")
        limit_rows.append(dict(bus_voltage_v=voltage, positive_power_w=power,
                               maximum_continuous_fraction=float(max(abs(tau)/cap)),
                               target_slew_after_20ms_rad=(e.target-e.neutral).tolist()))
    maximum_power_velocity = []
    pre_power = 0.
    for i, a in enumerate(curves):
        curve = np.array(a["selected_torque_speed_curve_rad_s_nm"], float)
        if a["reference_voltage_v"] != 5.:
            curve[:, 0] *= env.voltage_speed_factor
        grid = np.linspace(0., curve[-1, 0], 2001)
        available = np.minimum(cap[i], np.interp(grid, curve[:, 0], curve[:, 1]))
        k = int(np.argmax(grid * available))
        maximum_power_velocity.append(grid[k])
        pre_power += float(grid[k] * available[k])
    maximum_power_velocity = np.array(maximum_power_velocity)
    tau = env.limit_torque(np.full(18, 1e4), maximum_power_velocity)
    after_power = float(np.maximum(tau * maximum_power_velocity, 0).sum())
    check(pre_power > 350 and abs(after_power - 350) < 1e-8,
          "350W scaling branch not exercised correctly")
    # The adapter receives ordinary state and performs no post-step state projection.
    axis_rows = []
    for i, j in enumerate(c["joints"]):
        em = mod.FrozenPhysics(package)
        em.model.opt.gravity[:] = 0
        em.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        q0 = em.data.qpos[em.qidx].copy()
        action = np.zeros(18)
        action[i] = min(.08 / j["action_scale_rad"], .5)
        for _ in range(5):
            em.step(action)
        delta = float(em.data.qpos[em.qidx[i]] - q0[i])
        axis_rows.append(dict(public=j["name"], delta_rad=delta, positive_response=delta > 0,
                              scope="Isolated mapping only: gravity/contact off in memory"))
        check(delta > 0, "Independent positive-axis response failed: " + j["name"])
    env.reset()
    obs = env.observations(np.array([.1, -.2, .3]), phase_rad=.25)
    check(obs.shape == (65,) and np.isfinite(obs).all(), "65-value physical observation reader")
    check(np.max(abs(obs[9:27] - (env.data.qpos[env.qidx] - env.neutral))) < 1e-7,
          "Observed named q mapping differs")
    check(np.max(abs(obs[6:9] - [.1, -.2, .3])) < 1e-7
          and np.max(abs(obs[-2:] - [np.sin(.25), np.cos(.25)])) < 1e-7,
          "Observation command/phase layout differs")
    default_contacts = [dict(a=mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(x.geom1)),
                             b=mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(x.geom2)),
                             distance_m=float(x.dist)) for x in env.data.contact]
    samples, peaks, positive_w, branch_peaks = [], {}, 0., {}
    begin = time.perf_counter()
    for _ in range(100):
        env.step(np.zeros(18))
        mujoco.mj_forward(env.model, env.data)
        check(np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all(), "Nonfinite source step")
        check(not np.any(env.data.qfrc_applied) and not np.any(env.data.xfrc_applied), "Hidden external support")
        check(abs(env.data.time - env.policy_updates*.02) < 1e-10, "Clock drift")
        for k in range(env.data.nefc):
            if env.data.efc_type[k] == mujoco.mjtConstraint.mjCNSTR_EQUALITY:
                name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, int(env.data.efc_id[k]))
                peaks[name] = max(peaks.get(name, 0.), abs(float(env.data.efc_pos[k])))
        for j in c["passive_linkage_joints"]:
            qi = int(env.model.joint(j["name"]).qposadr[0])
            source = int(env.model.joint(j["mimic_joint"]).qposadr[0])
            delta = abs(float(env.data.qpos[qi] - j["mimic_multiplier"] * env.data.qpos[source]
                              - j["mimic_offset_rad"]))
            branch_peaks[j["name"]] = max(branch_peaks.get(j["name"], 0.), delta)
        positive_w = max(positive_w, float(np.maximum(env.last_tau * env.data.qvel[env.drive_vidx], 0).sum()))
        samples.append(dict(time_s=float(env.data.time), contact_count=int(env.data.ncon),
                            root_z_m=float(env.data.qpos[2]), warnings=int(env.data.warning.number.sum())))
    actual = dict(robot_leaves=int(sum(env.model.geom_bodyid != 0)),
                  supports=sum(int(env.model.mesh_vertnum[env.model.geom_dataid[g]])
                               for g in range(env.model.ngeom) if env.model.geom_bodyid[g] != 0))
    return dict(status="PASS_NUMERICAL_CHECKS_CONTACT_FIXTURES_SEPARATE", model=identity(xml),
                adapter=identity(adapter_file), mujoco_version=mujoco.__version__,
                compile_ms=compiled_ms, actual_counts=actual, body_errors=body_errors,
                compiled_contact_parameters=dict(
                    unique_geom_friction=np.unique(m.geom_friction, axis=0).tolist(),
                    unique_geom_solref=np.unique(m.geom_solref, axis=0).tolist(),
                    unique_geom_solimp=np.unique(m.geom_solimp, axis=0).tolist(),
                    explicit_pair_friction=m.pair_friction.tolist(),
                    source_is_design_estimate_not_material_bench_measurement=True),
                joints=joint_rows, actual_named_motor_map=mapping, limiter=limit_rows,
                positive_power_scaling_fixture=dict(before_limit_w=pre_power, after_limit_w=after_power,
                                                    velocity_rad_s=maximum_power_velocity.tolist()),
                isolated_axis_checks=axis_rows, untouched_reset_contacts=default_contacts,
                observation_shape=list(obs.shape),
                gravity_contact_step=dict(simulated_s=float(env.data.time), physics_steps=400,
                                          policy_updates=env.policy_updates, drive_updates=env.controller_updates,
                                          mean_physics_step_ms=(time.perf_counter()-begin)*1000/400,
                                          peak_equality_residual=peaks, peak_positive_power_w=positive_w,
                                          peak_passive_branch_angle_error_rad=branch_peaks,
                                          samples=samples),
                task_success_or_manufacturing_inferred=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--contract-only", action="store_true")
    p.add_argument("--adapter", type=Path)
    a = p.parse_args()
    package = a.package.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    result = dict(schema="goose_independent_frozen_training_physics_gate_v1",
                  script=identity(__file__), package=str(package), final_release_decision="PENDING")
    try:
        c, drive, r = contract_check(package, a.source_root)
        result["contract"] = r
        if not a.contract_only:
            adapter = (a.adapter or package / "runtime_adapter.py").resolve()
            result["runtime"] = runtime_check(package, c, drive, adapter)
        result["checks_complete"] = True
    except Exception as e:
        result["checks_complete"] = False
        result["failure"] = dict(type=type(e).__name__, message=str(e))
    (a.output / "independent_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("contract", "runtime")}))
    if not result["checks_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
