"""Compare a short untouched spawn rollout in main, workshop and flat mode.

Requires the local verified nine-model bundle and generated robot assets.
This checks the free central floor, not policy equivalence on obstacles.
"""
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import numpy as np
import sim2sim.godot_proc as godot_proc
from sim2sim.research.world import World
from sim2sim.research.models import NativeAnchor
from sim2sim.research.tasks import TASKS
from sim2sim.research.bundle import bundle_paths
from showcase_resource_guard import preflight, preview_lease

ROOT = Path(__file__).resolve().parents[1]

def main():
    cpus = sorted(os.sched_getaffinity(0))[:2]
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    # Keep successive headless children inside this test's two-core budget.
    godot_proc._CORE_SEQ = itertools.cycle(cpus)
    actor = NativeAnchor(bundle_paths(ROOT / 'bundles/delivery_v3')['standing'])
    traces, contracts = {}, {}
    with preview_lease():
        preflight()
        for label, scene, collisions in [('main', None, '1'),
                ('atelier', 'res://atelier/atelier.tscn', '1'),
                ('flat', 'res://atelier/atelier.tscn', '0')]:
            os.environ['MD_WORKSHOP_COLLISIONS'] = collisions
            os.environ['MD_STATIC_COURSE'] = '0'
            world = World(replace(TASKS['standing'], robot='microduck_ball_stand_fix'), scene_override=scene)
            try:
                world.reset(61000, randomize=False)
                rows = []
                for _ in range(100):
                    action = actor(world.obs()[None])[0]
                    world.step(action)
                    rows.append(np.r_[action, world.state.q, world.state.qd,
                                      world.state.base_pos, world.state.base_quat_wxyz])
                traces[label] = np.array(rows)
                if scene:
                    contracts[label] = world.backend._client.call(dict(cmd='atelier_contract'))
            finally:
                world.close()
    report = {k: dict(steps=100, max_abs=float(abs(v-traces['main']).max())) for k,v in traces.items()}
    report['headless_solid_nodes'] = {k: sum('WorkshopCollisions' in n['path'] or 'ContactCourse' in n['path']
        for n in c['physics']) for k,c in contracts.items()}
    report['headless_dynamic_bodies'] = {k: sum(n['type']=='RigidBody3D' and 'LooseProps' in n['path'] for n in c['physics']) for k,c in contracts.items()}
    out = ROOT / 'results/workshop_validation/isolation.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if report['headless_dynamic_bodies'] != {'atelier': 6, 'flat': 0}:
        raise SystemExit('Wrong free-body scene isolation')
    if report['flat']['max_abs'] != 0 or report['atelier']['max_abs'] > 1e-5:
        raise SystemExit('Spawn rollout changed; inspect the isolation report')
    if report['headless_solid_nodes']['atelier'] < 100 or report['headless_solid_nodes']['flat']:
        raise SystemExit('Incorrect headless collision mode')

if __name__ == '__main__':
    main()
