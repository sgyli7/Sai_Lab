"""Bounded source-capability probe; no training or candidate promotion.

Every case starts at rest, accelerates through the existing input ramp, and
releases to idle. These command-limit probes are not the final Shift handoff
acceptance suite. Raw observations and physical trajectories are retained.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import argparse
import json
from pathlib import Path
import time

import numpy as np

from sim2sim.play_input import PlayBrain, TwistLimits
from .models import NativeAnchor
from .queue import atomic_json
from .tasks import TASKS, DT
from .world import World


def episode(source, speed, turn, seed, output, scene_robot='microduck'):
    policy = NativeAnchor(source)
    task = replace(TASKS['walking'], seconds=12.,robot=scene_robot)
    world = World(task, yaw_memory_input=policy.yaw_memory_input)
    brain = PlayBrain(has_standing=False, lim=TwistLimits(vmax_x=speed, vmax_ang=abs(turn) or 1.5))
    rows = []; observations = []; actions = []
    start = time.monotonic()
    try:
        world.reset(seed, 'idle')
        commands = []
        for step in range(600):
            held = {'fwd'} if 50 <= step < 450 else set()
            if held and turn: held.add('left' if turn > 0 else 'right')
            commands.append(brain.tick(held, [], DT).command)
        world.command_tape = np.stack(commands)
        for step in range(600):
            obs = world.obs(); action = policy(obs[None])[0]
            observations.append(obs.copy()); actions.append(action.copy())
            world.step(action)
            f = world.features
            rows.append([world.t, *f['xy'], f['z'], f['yaw'], f['tilt'], *f['vel'], *f['gyro']])
        trace = np.asarray(rows)
        heading = np.arctan2(world.heading[1], world.heading[0])
        yaw = np.unwrap(np.r_[heading, trace[:, 4]])[1:]
        moving = (trace[:, 0] >= 2.) & (trace[:, 0] < 9.)
        delta = trace[:, 1:3] - world.initial_xy
        forward = delta @ world.heading
        lateral = delta @ np.array([-world.heading[1], world.heading[0]])
        fall = bool(np.any(trace[:, 5] >= 60.) or np.any(trace[:, 3] < .05))
        result = dict(source=str(Path(source).resolve()), sha256=policy.sha256,
            seed=seed, command_speed=speed, command_turn=turn, fall=fall,
            mean_forward_speed=float(np.mean(trace[moving, 6])),
            mean_lateral_speed=float(np.mean(np.abs(trace[moving, 7]))),
            mean_yaw_rate=float(np.mean(trace[moving, 11])),
            max_tilt=float(trace[:, 5].max()), min_z=float(trace[:, 3].min()),
            max_heading_error_deg=float(np.rad2deg(np.max(np.abs(yaw-heading)))) if not turn else None,
            max_lateral_path_m=float(np.max(np.abs(lateral))) if not turn else None,
            forward_distance_m=float(forward[449]),
            idle_speed=float(np.mean(np.linalg.norm(trace[-50:, 6:8], axis=1))),
            trace=str(output.with_name(output.name+'.npz').resolve()),
            elapsed_s=time.monotonic()-start, physics=world.physics)
        np.savez_compressed(output.with_name(output.name+'.npz'), obs=observations, actions=actions,
                            commands=commands, state=trace, heading=world.heading)
        atomic_json(output.with_name(output.name+'.json'), result)
        return result
    finally:
        world.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True, help='JSON label to ONNX mapping')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--speeds', default='.3,.4,.5')
    parser.add_argument('--turns', default='0,.8,-.8')
    parser.add_argument('--seeds', default='919000,919001')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--scene-robot',choices=['microduck','microduck_ball_stand_fix'],default='microduck')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    sources = json.loads(args.sources.read_text())
    jobs = [(name, path, speed, turn, seed) for name, path in sources.items()
            for speed in map(float, args.speeds.split(','))
            for turn in map(float, args.turns.split(','))
            for seed in map(int, args.seeds.split(','))]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        for name, path, speed, turn, seed in jobs:
            output = args.out / f'{name}_v{speed}_w{turn}_{seed}'
            pending[pool.submit(episode, path, speed, turn, seed, output,args.scene_robot)] = (name, speed, turn, seed)
        for future in as_completed(pending):
            name, speed, turn, seed = pending[future]
            try:
                result = future.result(); result['label'] = name
            except Exception as error:
                result = dict(label=name, command_speed=speed, command_turn=turn, seed=seed, error=repr(error))
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('physics', 'source')}), flush=True)
    atomic_json(args.out / 'summary.json', dict(completed=len(results)==len(jobs),
        errors=sum('error' in r for r in results), episodes=results))


if __name__ == '__main__':
    main()
