"""Compare every recorded policy trajectory against the original flat headless world."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from sim2sim.obs import build_obs
from sim2sim.research.bundle import bundle_paths
from sim2sim.research.models import NativeAnchor
from sim2sim.research.world import World
from sim2sim.research.tasks import TASKS, DT
from sim2sim.research.evaluate import record
from showcase_resource_guard import preview_lease, preflight

ROOT = Path(__file__).resolve().parents[1]

def main(prefix):
    paths = bundle_paths(ROOT / 'bundles/delivery_v3')
    results = {}
    for skill, original_task in TASKS.items():
        tag = prefix + '_' + skill
        directory = ROOT / 'results/showcase' / tag
        captured = json.loads((directory / 'capture.json').read_text())
        reference = np.load(directory / 'trajectory.npz')
        actor = NativeAnchor(paths[skill])
        task = original_task if original_task.robot == 'microduck_roller' else replace(original_task, robot='microduck_ball_stand_fix')
        preflight(30)
        world = World(task, time_input_s=actor.time_input_s, heading_input=actor.heading_input,
                      yaw_memory_input=actor.yaw_memory_input, roller_contract=skill == 'roller',
                      entry_source=paths['roller' if task.robot == 'microduck_roller' else 'standing'])
        rows = []
        try:
            world.reset(captured['seed'], captured['condition'], randomize=False)
            if captured['entry_seconds']:
                teacher, cmd = world.prepare_standing_entry()
                for _ in range(round(captured['entry_seconds'] / DT)):
                    obs = build_obs(world.state, world.last, cmd, world.home)
                    world.step(teacher(obs[None])[0])
                world.finish_standing_entry()
            for _ in range(round(captured['skill_seconds'] / DT)):
                action = actor(world.obs()[None])[0]
                world.step(action)
                rows.append(record(world, action))
        finally:
            world.close()
        errors = {key: float(np.max(np.abs(np.asarray([r[key] for r in rows], dtype=float) - reference[key].astype(float)))) for key in reference.files}
        results[skill] = {'samples': len(rows), 'maximum_absolute_errors': errors,
                          'exact': all(e == 0 for e in errors.values())}
        print(skill, results[skill]['exact'], max(errors.values()), flush=True)
    output = ROOT / 'results/showcase/physics_equivalence.json'
    output.write_text(json.dumps(results, indent=2) + '\n')
    if not all(r['exact'] for r in results.values()):
        raise SystemExit('Trajectories differ; investigate before claiming physics equivalence')

if __name__ == '__main__':
    with preview_lease():
        parser = argparse.ArgumentParser()
        parser.add_argument("--prefix", default="04")
        main(parser.parse_args().prefix)
