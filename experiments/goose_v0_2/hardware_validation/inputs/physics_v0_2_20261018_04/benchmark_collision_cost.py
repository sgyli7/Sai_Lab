"""Independent bounded MuJoCo collision-cost and actual hull-complexity gate.

Collision geometry is timed separately from articulated dynamics and loop solve.
No training, controller change, persistent model edits or deliverable masks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def identity(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def referenced_mesh_files(path):
    """Bind cost reports to file contents; identical XML can point at new meshes."""
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    compiler = root.find('compiler')
    meshdir = Path(compiler.get('meshdir', '.')) if compiler is not None else Path('.')
    rows = []
    for node in root.findall('./asset/mesh'):
        if not node.get('file'):
            continue
        p = Path(node.get('file'))
        if not p.is_absolute(): p = path.parent / meshdir / p
        row = dict(name=node.get('name'), **identity(p))
        if p.suffix.lower() == '.obj':
            counts = dict(file_vertices=0, file_face_records=0, file_triangulated_faces=0)
            for line in p.read_text().splitlines():
                words = line.split()
                if words and words[0] == 'v': counts['file_vertices'] += 1
                if words and words[0] == 'f':
                    counts['file_face_records'] += 1
                    counts['file_triangulated_faces'] += max(len(words)-3, 0)
            row.update(counts)
        rows.append(row)
    binding = hashlib.sha256(json.dumps([(r['name'], r['sha256']) for r in rows], separators=(',', ':')).encode()).hexdigest()
    return dict(files=rows, ordered_names_and_sha256=binding)


def graph_layout_fixture():
    # Bind this decoder to real MuJoCo3.10 hull construction, not raw OBJ count.
    fixtures = [("cube", "-1 -1 -1 1 -1 -1 1 1 -1 -1 1 -1 -1 -1 1 1 -1 1 1 1 1 -1 1 1", 8, 12),
                ("octahedron", "1 0 0 -1 0 0 0 1 0 0 -1 0 0 0 1 0 0 -1", 6, 8)]
    rows = []
    for name, vertices, nv, nf in fixtures:
        m = mujoco.MjModel.from_xml_string(
            '<mujoco><asset><mesh name="h" vertex="'+vertices+'"/></asset>'
            '<worldbody><body><freejoint/><geom type="mesh" mesh="h"/></body></worldbody></mujoco>')
        adr = int(m.mesh_graphadr[0])
        actual = m.mesh_graph[adr:adr+2].tolist()
        if actual != [nv, nf]:
            raise AssertionError("Unexpected convex graph layout: " + str(actual))
        rows.append(dict(name=name, expected=[nv, nf], actual=actual))
    return dict(mujoco_version=mujoco.__version__, header='mesh_graph[mesh_graphadr: +2] = hull vertices, triangulated hull faces', fixtures=rows)


def complexity(path, model=None):
    path = Path(path).resolve()
    m = model or mujoco.MjModel.from_xml_path(str(path))
    explicit = set(map(int, m.pair_geom1)) | set(map(int, m.pair_geom2))
    rows = []
    for g in range(m.ngeom):
        if not m.geom_bodyid[g] or not (m.geom_contype[g] or m.geom_conaffinity[g] or g in explicit):
            continue
        row = dict(geom=m.geom(g).name, body=m.body(int(m.geom_bodyid[g])).name,
                   type=int(m.geom_type[g]), contype=int(m.geom_contype[g]),
                   conaffinity=int(m.geom_conaffinity[g]))
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[g]); adr = int(m.mesh_graphadr[mid])
            if adr < 0:
                raise AssertionError("No actual convex graph for " + row['geom'])
            nv, nf = map(int, m.mesh_graph[adr:adr+2])
            if nv < 4 or nv > m.mesh_vertnum[mid] or nf < 4:
                raise AssertionError("Invalid actual hull graph")
            row.update(mesh=m.mesh(mid).name, loaded_vertices=int(m.mesh_vertnum[mid]),
                       loaded_faces=int(m.mesh_facenum[mid]), hull_vertices=nv, hull_faces=nf)
        else:
            row.update(hull_vertices=None, hull_faces=None,
                       primitive_support_function_not_finite_vertex_mesh=True)
        rows.append(row)
    total = sum(r.get('hull_vertices') or 0 for r in rows)
    maximum = max([r.get('hull_vertices') or 0 for r in rows] or [0])
    failures = []
    if len(rows) != 14: failures.append(f'actual collision leaves {len(rows)} !=14')
    if maximum > 64: failures.append(f'actual maximum convex supports {maximum} >64')
    if total > 896: failures.append(f'actual total convex supports {total} >896')
    if any(r.get('hull_vertices') is None for r in rows):
        failures.append('new fourteen-block gate requires explicit convex mesh topology')
    return dict(model=identity(path), referenced_mesh_assets=referenced_mesh_files(path), actual_collision_leaves=len(rows), actual_hull_vertices=total,
                actual_maximum_hull_vertices=maximum,
                actual_hull_triangles=sum(r.get('hull_faces') or 0 for r in rows),
                loaded_mesh_vertices=sum(r.get('loaded_vertices') or 0 for r in rows),
                loaded_mesh_faces=sum(r.get('loaded_faces') or 0 for r in rows),
                rows=rows, complexity_gate='PASS' if not failures else 'REJECT', failures=failures)


def summary(values):
    return dict(median_us=float(np.median(values)), p95_us=float(np.percentile(values, 95)),
                minimum_us=float(min(values)), repeats=len(values), samples_us=list(map(float, values)))


def uniform_solver(m):
    m.opt.timestep = .005
    m.opt.iterations = 100
    m.opt.ls_iterations = 50
    m.opt.tolerance = 1e-8
    m.opt.ls_tolerance = .01
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    # Collision solver budget must match too: the old Move XML explicitly
    # disables multiccd, while the shipped new model uses its enabled default.
    m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)


def pose_floor_z(m, d):
    minima = []
    for g in range(m.ngeom):
        if not m.geom_bodyid[g] or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        R = d.geom_xmat[g].reshape(3, 3); z = d.geom_xpos[g, 2]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[g]); start = int(m.mesh_vertadr[mid]); n = int(m.mesh_vertnum[mid])
            minima.append(float((m.mesh_vert[start:start+n] @ R[2]).min()+z))
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            minima.append(float(z-np.abs(R[2])@m.geom_size[g]))
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            minima.append(float(z-m.geom_size[g, 0]))
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            minima.append(float(z-m.geom_size[g, 0]-abs(R[2, 2])*m.geom_size[g, 1]))
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            minima.append(float(z-m.geom_size[g, 0]*np.linalg.norm(R[2, :2])-abs(R[2, 2])*m.geom_size[g, 1]))
        else:
            raise ValueError('Unsupported finite robot shape in ground fixture')
    return min(minima)


def source_contract(path):
    for name in ('physical_contract.json', 'hardware_contract.json', 'contract.json'):
        p = path.parent / name
        if p.is_file():
            return json.loads(p.read_text())
    return None


def make_scenarios(m, contract, seed, full):
    rng = np.random.default_rng(seed)
    cases = [('cold_saved_reset', None, None), ('standing_support', [1., 0., 0., 0.], {})]
    s = 2**-.5
    cases += [('prone_front', [s, 0., s, 0.], {}), ('supine_back', [s, 0., -s, 0.], {}),
              ('lying_left', [s, s, 0., 0.], {}), ('lying_right', [s, -s, 0., 0.], {})]
    for n in range(4 if full else 1):
        qs = {}
        for i in range(m.njnt):
            if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE: continue
            nm = m.joint(i).name
            if nm is None or 'input_rotor' in nm or 'coupler' in nm: continue
            lo, hi = m.jnt_range[i]
            qs[nm] = float(np.clip(rng.normal(0, .15), lo+.001, hi-.001)) if m.jnt_limited[i] else 0.
        cases.append((f'finite_work_seed_{n}', [1., 0., 0., 0.], qs))
    outputs = []
    for label, quat, qs in cases:
        d = mujoco.MjData(m); mujoco.mj_resetData(m, d)
        if quat is not None:
            d.qpos[:3] = [0., 0., 0.]; d.qpos[3:7] = quat
            for nm, val in (qs or {}).items():
                d.qpos[int(m.joint(nm).qposadr[0])] = val
            if contract:
                for j in contract.get('passive_linkage_joints', []):
                    source = int(m.joint(j['mimic_joint']).qposadr[0])
                    d.qpos[int(m.joint(j['name']).qposadr[0])] = (
                        j['mimic_multiplier']*d.qpos[source]+j['mimic_offset_rad'])
            mujoco.mj_forward(m, d)
            d.qpos[2] = -.0005-pose_floor_z(m, d)
        mujoco.mj_forward(m, d)
        outputs.append((label, d.qpos.copy(), d.qvel.copy(), dict(
            qpos=d.qpos.tolist(), qvel=d.qvel.tolist(), kinematic_terrain_placement=quat is not None,
            terrain_penetration_fixture_m=.0005 if quat is not None else None,
            named_hinges={m.joint(i).name: float(d.qpos[int(m.jnt_qposadr[i])])
                          for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE})))
    return outputs


def reset_snapshot(m, d, q, v):
    mujoco.mj_resetData(m, d)
    d.qpos[:] = q; d.qvel[:] = v
    mujoco.mj_forward(m, d)


def timed_fixed(m, d, fn, calls, repeats, warmup):
    for _ in range(warmup): fn(m, d)
    values = []
    for _ in range(repeats):
        begin = time.perf_counter_ns()
        for _ in range(calls): fn(m, d)
        values.append((time.perf_counter_ns()-begin)/calls/1000.)
    return summary(values)


def benchmark(path, full, seed):
    m = mujoco.MjModel.from_xml_path(str(path)); uniform_solver(m)
    c = source_contract(path); scenarios = make_scenarios(m, c, seed, full)
    no_contact = copy.copy(m); no_contact.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    no_loop = copy.copy(no_contact); no_loop.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EQUALITY)
    calls, repeats, warmup, steps = (80, 9, 20, 80) if full else (16, 5, 5, 20)
    rows = []
    for label, q, v, fixture in scenarios:
        d = mujoco.MjData(m); reset_snapshot(m, d, q, v)
        collision = timed_fixed(m, d, mujoco.mj_collision, calls, repeats, warmup)
        forward = timed_fixed(m, d, mujoco.mj_forward, calls, repeats, warmup)
        counts = dict(contacts=int(d.ncon), constraints=int(d.nefc),
                      solver_iterations=int(np.sum(d.solver_niter)), warnings=int(np.sum(d.warning.number)))
        diagnostics = {}
        for tag, am in [('no_contact', no_contact), ('no_contact_no_equality', no_loop)]:
            ad = mujoco.MjData(am); reset_snapshot(am, ad, q, v)
            diagnostics[tag] = dict(forward=timed_fixed(am, ad, mujoco.mj_forward, calls, repeats, warmup),
                                    contacts=int(ad.ncon), constraints=int(ad.nefc))
        for _ in range(warmup): mujoco.mj_step(m, d)
        durations, step_counts, warnings, finite = [], [], 0, True
        for _ in range(repeats):
            reset_snapshot(m, d, q, v)
            begin = time.perf_counter_ns()
            for _ in range(steps): mujoco.mj_step(m, d)
            durations.append((time.perf_counter_ns()-begin)/steps/1000.)
            step_counts.append(dict(contacts=int(d.ncon), constraints=int(d.nefc),
                                    solver_iterations=int(np.sum(d.solver_niter))))
            warnings += int(np.sum(d.warning.number)); finite &= bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        rows.append(dict(name=label, fixture=fixture, collision_query=collision,
                         fixed_pose_forward=forward, fixed_counts=counts,
                         source_native_step=summary(durations), terminal_step_counts=step_counts,
                         step_warnings=warnings, step_finite=finite,
                         diagnostic_ablation_not_admissible_for_delivery=diagnostics))
    return dict(model=identity(path), actual_complexity=complexity(path, m),
                body_count=m.nbody-1, articulated_velocity_dimensions=m.nv,
                equality_count=m.neq, actuator_count=m.nu, total_mass_kg=float(m.body_mass.sum()),
                source_collision_body_exclusions=m.exclude_signature.tolist(),
                effective_solver=dict(dt_s=m.opt.timestep, solver=int(m.opt.solver), iterations=m.opt.iterations,
                                      ls_iterations=m.opt.ls_iterations, tolerance=m.opt.tolerance,
                                      integrator=int(m.opt.integrator), cone=int(m.opt.cone), jacobian=int(m.opt.jacobian),
                                      enableflags=int(m.opt.enableflags), disableflags=int(m.opt.disableflags),
                                      ccd_iterations=int(m.opt.ccd_iterations), ccd_tolerance=float(m.opt.ccd_tolerance)),
                workload=dict(calls_per_repeat=calls, repeats=repeats, warmup=warmup, steps_per_rollout=steps),
                scenarios=rows)


def performance_comparison(reference, candidate):
    rows = []
    by = {s['name']: s for s in reference['scenarios']}
    for row in candidate['scenarios']:
        base = by[row['name']]
        rows.append(dict(name=row['name'], collision_median_ratio=row['collision_query']['median_us']/base['collision_query']['median_us'],
                         collision_p95_ratio=row['collision_query']['p95_us']/base['collision_query']['p95_us'],
                         forward_median_ratio=row['fixed_pose_forward']['median_us']/base['fixed_pose_forward']['median_us'],
                         whole_step_median_ratio=row['source_native_step']['median_us']/base['source_native_step']['median_us'],
                         contacts_reference=base['fixed_counts']['contacts'], contacts_candidate=row['fixed_counts']['contacts']))
    worst = max(r['collision_median_ratio'] for r in rows)
    return dict(scenarios=rows, collision_specialty_target_max_ratio=1.5,
                worst_collision_median_ratio=worst,
                collision_specialty_gate='PASS' if worst <= 1.5 else 'REJECT',
                full_step_ratio_is_descriptive_not_geometry_causal_proof=True,
                gpu_or_parallel_training_throughput_measured=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--complexity-only', action='store_true')
    p.add_argument('--reference', type=Path)
    p.add_argument('--full', action='store_true')
    p.add_argument('--seed', type=int, default=20261018)
    p.add_argument('--cpu', type=int)
    a = p.parse_args()
    result = dict(schema='goose_independent_collision_cost_gate_v1', script=identity(__file__),
                  graph_layout=graph_layout_fixture(), candidate=complexity(a.candidate))
    if not a.complexity_only:
        available = sorted(os.sched_getaffinity(0)); cpu = a.cpu if a.cpu is not None else available[-1]
        os.sched_setaffinity(0, {cpu})
        result['host'] = dict(machine=platform.machine(), platform=platform.platform(),
                              pinned_cpu=cpu, affinity=sorted(os.sched_getaffinity(0)), seed=a.seed,
                              mujoco_version=mujoco.__version__, process_id=os.getpid())
        result['candidate_benchmark'] = benchmark(a.candidate.resolve(), a.full, a.seed)
        if a.reference:
            result['reference_benchmark'] = benchmark(a.reference.resolve(), a.full, a.seed)
            result['performance_comparison'] = performance_comparison(result['reference_benchmark'], result['candidate_benchmark'])
        result['scope'] = 'Single pinned CPU collision queries, fixed forward, and bounded native integration only; no controller or training. Contact/equality ablations diagnose cost and are not admissible performance candidates.'
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result['candidate'].items() if k not in ('rows','model','referenced_mesh_assets')}))
    if result['candidate']['failures']:
        raise SystemExit(1)
    if result.get('performance_comparison', {}).get('collision_specialty_gate') == 'REJECT':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
