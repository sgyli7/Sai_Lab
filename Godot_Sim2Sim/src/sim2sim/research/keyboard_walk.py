"""Supplementary walking checks through the actual keyboard controller.

These short continuous tapes include the default 0.3 m/s key command absent
from the original fixed suite. Existing task gates are reported per segment;
the original physical_tasks_v7 suite remains unchanged.
"""
import argparse, hashlib, json, math
from dataclasses import replace
from pathlib import Path
import numpy as np
from sim2sim.obs import build_obs
from sim2sim.play_input import PlayBrain
from sim2sim.policy import OnnxPolicy
from .world import World
from .tasks import TASKS, DT
from .evaluate import record, summarize, PROTOCOL_VERSION
from .schedules import KEYBOARD_TAPES as TAPES


def run(source, out, seed=100, condition='forward', backend='godot', scene_robot='microduck_ball'):
    source = Path(source)
    actor = OnnxPolicy(source)
    bank = {'walking': actor, 'standing': OnnxPolicy(TASKS['standing'].source)}
    brain = PlayBrain(has_standing=actor.has_standing_partner, lim=actor.twist_limits)
    world = World(replace(TASKS['walking'], robot=scene_robot), backend)
    segments = []; traces = {}
    try:
        world.reset(seed)
        for label, seconds, held in TAPES[condition]:
            start = world.t; rows = []; policies = set()
            yaw = world.features['yaw']; heading = np.array([math.cos(yaw), math.sin(yaw)])
            for _ in range(round(seconds / DT)):
                control = brain.tick(held, [], DT)
                obs = build_obs(world.state, world.last, control.command, world.home)
                action = bank[control.policy].infer(obs)
                world.step(action)
                row = record(world, action)
                row['cmd'] = control.command.copy(); row['time'] -= start; row['obs'] = obs
                rows.append(row); policies.add(control.policy)
            segments.append(dict(label=label, policies=sorted(policies),
                metrics=summarize(TASKS['walking'], rows, heading)))
            for key in rows[0]: traces[label + '/' + key] = np.asarray([row[key] for row in rows])
        out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out.with_suffix('.npz'), **traces)
        sidecar = source.with_suffix('.manifest.json')
        result = dict(check='native_keyboard_walk_v2', protocol=PROTOCOL_VERSION,
            source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest() if sidecar.exists() else None,
            seed=seed, condition=condition, backend=backend, physics=world.physics,
            resets_after_start=0, seconds=world.t, segments=segments,
            fell=any(s['metrics']['fell'] for s in segments), trace=str(out.with_suffix('.npz')))
        out.write_text(json.dumps(result, indent=2)); return result
    finally:
        world.close()


def main():
    p = argparse.ArgumentParser(); p.add_argument('source', type=Path)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--seed', type=int, default=100)
    p.add_argument('--condition', choices=list(TAPES), default='forward')
    p.add_argument('--backend', choices=['godot', 'mujoco'], default='godot')
    a = p.parse_args(); r = run(a.source, a.out, a.seed, a.condition, a.backend)
    print(json.dumps(r['segments'], indent=2))


if __name__ == '__main__': main()
