"""Bounded brake-only pose search in unchanged Jolt physics.

This is a development experiment, not an automatic model promotion. All trials
retain their physical trajectories; the exported actor still requires the full
independent native keyboard suite. Positive and neutral throttle use the exact
original ONNX graph, including its internal floating-point types.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper

from sim2sim.research.budget import require_supervision
from sim2sim.research.evaluate import record
from sim2sim.research.models import NativeAnchor
from sim2sim.research.queue import atomic_json
from sim2sim.research.tasks import TASKS, DT
from sim2sim.research.world import World
from sim2sim.standalone.score import brake_metrics


BOUNDS = np.array([.35, .25, .45, .45, .45], np.float32)


def action_offset(parameters):
    values = np.asarray(parameters, np.float32)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError('Expected five finite joint offsets')
    result = np.zeros(14, np.float32)
    result[:5] = values
    result[9:] = -values
    return result


def adapted_action(anchor, observations, offset, feedback=(0.,0.)):
    observations = np.asarray(observations, np.float32).reshape(-1, 61)
    gate = np.clip(-observations[:, 48:49] / np.float32(.05), 0., 1.)
    delta=np.broadcast_to(np.asarray(offset,np.float32), (len(observations),14)).copy()
    correction=np.clip(-np.float32(feedback[0])*observations[:,3:4]-np.float32(feedback[1])*observations[:,1:2],-.12,.12)
    delta[:,4:5]+=correction;delta[:,13:14]-=correction
    return anchor(observations) + gate * delta


def export(source, offset, target, feedback=(0.,0.)):
    graph = onnx.load(source)
    original = graph.graph.output[0].name
    names = {v for n in graph.graph.node for v in [*n.input, *n.output]}
    prefix = 'brake_pose_'
    while any(n.startswith(prefix) for n in names):
        prefix += 'x_'
    def name(value): return prefix + value
    for label, value in [('index', np.array([48], np.int64)),
                         ('scale', np.array(.05, np.float32)),
                         ('zero', np.array(0., np.float32)),
                         ('one', np.array(1., np.float32)),
                         ('offset', np.asarray(offset, np.float32).reshape(1, 14))]:
        graph.graph.initializer.append(numpy_helper.from_array(value, name(label)))
    offset_name=name('offset')
    nodes=[]
    if any(feedback):
        for label,value in [('pitch_index',np.array([3],np.int64)),('gyro_index',np.array([1],np.int64)),
                            ('kp',np.array(-feedback[0],np.float32)),('kd',np.array(-feedback[1],np.float32)),
                            ('low',np.array(-.12,np.float32)),('high',np.array(.12,np.float32)),
                            ('ankles',np.eye(14,dtype=np.float32)[4:5]-np.eye(14,dtype=np.float32)[13:14])]:
            graph.graph.initializer.append(numpy_helper.from_array(value,name(label)))
        nodes.extend([
            helper.make_node('Gather',[graph.graph.input[0].name,name('pitch_index')],[name('pitch')],axis=1),
            helper.make_node('Gather',[graph.graph.input[0].name,name('gyro_index')],[name('gyro')],axis=1),
            helper.make_node('Mul',[name('pitch'),name('kp')],[name('p')]),
            helper.make_node('Mul',[name('gyro'),name('kd')],[name('d')]),
            helper.make_node('Add',[name('p'),name('d')],[name('feedback')]),
            helper.make_node('Clip',[name('feedback'),name('low'),name('high')],[name('bounded_feedback')]),
            helper.make_node('Mul',[name('bounded_feedback'),name('ankles')],[name('ankle_delta')]),
            helper.make_node('Add',[name('offset'),name('ankle_delta')],[name('total_offset')]),
        ])
        offset_name=name('total_offset')
    nodes += [
        helper.make_node('Gather', [graph.graph.input[0].name, name('index')], [name('throttle')], axis=1),
        helper.make_node('Neg', [name('throttle')], [name('negative')]),
        helper.make_node('Div', [name('negative'), name('scale')], [name('weight')]),
        helper.make_node('Clip', [name('weight'), name('zero'), name('one')], [name('gate')]),
        helper.make_node('Mul', [name('gate'), offset_name], [name('delta')]),
        helper.make_node('Add', [original, name('delta')], [name('actions')]),
    ]
    graph.graph.node.extend(nodes)
    graph.graph.output[0].name = name('actions')
    metadata = {item.key: item.value for item in graph.metadata_props}
    metadata.update(brake_pose_contract='negative_throttle_symmetric_offset_v1',
                    brake_pose_offset=json.dumps(np.asarray(offset).tolist()),
                    brake_pitch_feedback=json.dumps(list(feedback)),
                    brake_pose_parent_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    helper.set_model_props(graph, metadata)
    onnx.checker.check_model(graph)
    onnx.save(graph, target)
    old = NativeAnchor(source); new = NativeAnchor(target)
    rng = np.random.default_rng(915007)
    observations = rng.normal(size=(512, 61)).astype(np.float32)
    observations[:128, 48] = 0.
    expected = adapted_action(old, observations, offset,feedback)
    actual = new(observations)
    error = float(np.max(abs(expected - actual)))
    preserved = observations[:, 48] >= 0.
    exact_prefix = bool(np.array_equal(actual[preserved], old(observations[preserved])))
    if error >= 1e-5 or not exact_prefix:
        raise RuntimeError(f'Export parity failed: {error}, prefix={exact_prefix}')
    return dict(max_abs=error, positive_and_neutral_exact=exact_prefix,
                sha256=hashlib.sha256(Path(target).read_bytes()).hexdigest())


def objective(rows, metrics):
    braking = [row for row in rows if row['time'] >= metrics['brake_at']]
    speed = np.array([np.linalg.norm(row['vel'][:2]) for row in braking])
    early = speed[:100]
    reverse = max(0., -metrics['minimum_forward_02s'] - .03)
    settled = [row for row in braking if row['time'] >= metrics['brake_at']+1.]
    height_deficit = np.array([max(0., .085-row['z'])/.03 for row in settled])
    tilt_excess = np.array([max(0., row['tilt']-10.)/20. for row in settled])
    support_loss = np.mean([np.sum(row['contact']) == 0 for row in settled])
    # Fixed physical outcome first, then continuous guidance within failures.
    return (100. * metrics['fell'] + 30. * (not metrics['success'])
            + 12. * float(np.mean(early ** 2)) + 20. * float(np.mean(speed[-50:] ** 2))
            + 8. * reverse + min(10., metrics['braking_distance'])
            + 20. * float(np.mean(height_deficit**2))
            + 5. * float(np.mean(tilt_excess**2)) + 2. * support_loss
            + .001 * metrics['max_tilt'])


def rollout(world, anchor, parameters, seed, output, trial, case_name='roller_brake_3s', feedback=(0.,0.)):
    if world is None:
        # A new physical scene gives every parameter candidate the same initial
        # solver/contact history. Repeated reset in a reused Jolt scene can
        # slightly change the pre-braking speed even with identical seed/action.
        fresh = World(TASKS['roller'], roller_contract=True,state_input=getattr(anchor,'state_input',''))
        try:
            return rollout(fresh, anchor, parameters, seed, output, trial,case_name,feedback)
        finally:
            fresh.close()
    from sim2sim.standalone.cases import standard_cases
    case=standard_cases()[case_name]
    obs = world.reset(seed, 'keyboard_'+case_name, randomize=True)
    offset = action_offset(parameters); rows = []; observations = []
    for _ in range(round(case['seconds']/DT)):
        observations.append(obs.copy())
        action = adapted_action(anchor, obs, offset,feedback)[0]
        obs = world.step(action)
        rows.append(record(world, action))
    metrics = brake_metrics(rows, case['brake_times'][0], case['seconds'])
    cost = objective(rows, metrics)
    trajectory = output / f'{trial}_seed_{seed}.npz'
    np.savez_compressed(trajectory, observations=np.array(observations),
                        **{key: np.array([row[key] for row in rows]) for key in rows[0]})
    result = dict(trial=trial, seed=seed, parameters=np.asarray(parameters).tolist(),
                  case=case_name,feedback=list(feedback),
                  metrics=metrics, objective=cost, trajectory=str(trajectory))
    atomic_json(trajectory.with_suffix('.json'), result)
    return result


def run(source, output, generations=8, population=20, workers=4, minutes=12., initial=None, spread=.65,
        feedback=(0.,0.), cases=None):
    require_supervision(Path(os.environ['SIM2SIM_RESEARCH_DIR']))
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); rng = np.random.default_rng(915002)
    worlds = []; anchors = []; records = []
    seeds = [915000, 915001]
    cases=cases or ['roller_brake_3s']
    definition = dict(hypothesis='A symmetric brake-only joint offset can create stable deceleration while preserving push/coast exactly.',
        source=str(Path(source).resolve()), source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        bounds=BOUNDS.tolist(), seeds=seeds, generations=generations, population=population,
        minutes=minutes, protocol='standalone_keyboard_v1', task='roller_brake_3s',
        objective_version='brake_pose_outcome_standing_v2', physical_start='fresh_process_per_trial',
        source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [Path(__file__), Path(__file__).with_name('world.py')]},
        initial=None if initial is None else dict(path=str(Path(initial).resolve()),sha256=hashlib.sha256(Path(initial).read_bytes()).hexdigest()),
        initial_spread_fraction=spread,
        pitch_feedback=list(feedback),cases=cases,
        selection='development search; full native suite required before promotion')
    atomic_json(output / 'experiment.json', definition)
    try:
        for _ in range(workers):
            worlds.append(None); anchors.append(NativeAnchor(source))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            def evaluate(parameters, label, applied_feedback=feedback):
                jobs = [(i, p, seed,case) for i, p in enumerate(parameters) for seed in seeds for case in cases]
                results = []
                for start in range(0, len(jobs), workers):
                    futures = [pool.submit(rollout, worlds[lane], anchors[lane], p, seed,
                               output, f'{label}_{i:03d}'+('_'+case if len(cases)>1 else ''),case,applied_feedback)
                               for lane, (i, p, seed,case) in enumerate(jobs[start:start+workers])]
                    results.extend(f.result() for f in futures)
                candidates = []
                for i, p in enumerate(parameters):
                    trials = [r for r in results if r['trial'].startswith(f'{label}_{i:03d}')]
                    candidates.append(dict(parameters=np.asarray(p).tolist(),
                        objective=float(np.mean([r['objective'] for r in trials])),
                        passes=sum(r['metrics']['success'] for r in trials),
                        falls=sum(r['metrics']['fell'] for r in trials), trials=trials))
                return sorted(candidates, key=lambda x: x['objective'])
            baseline = evaluate([np.zeros(5)], 'baseline',(0.,0.))[0]
            atomic_json(output / 'baseline.json', baseline)
            best = baseline; mean = np.zeros(5); sigma = BOUNDS * spread
            if initial is not None:
                previous=json.loads(Path(initial).read_text())
                mean=np.asarray(previous['best']['parameters'],float)
                if mean.shape!=(5,) or not np.isfinite(mean).all() or np.any(abs(mean)>BOUNDS):
                    raise ValueError('Invalid initial pose parameters')
                best=evaluate([mean], 'initial')[0]
                atomic_json(output/'initial.json',best)
            elif any(feedback):
                best=evaluate([mean], 'initial')[0]
                atomic_json(output/'initial.json',best)
            generation_cost=3.*population*len(seeds)*len(cases)/workers
            for generation in range(generations):
                if time.monotonic()-started+generation_cost+20.>minutes*60.:break
                generation_started=time.monotonic()
                parameters = np.clip(rng.normal(mean, sigma, size=(population, 5)), -BOUNDS, BOUNDS)
                parameters[0] = best['parameters']
                candidates = evaluate(parameters, f'g{generation:02d}')
                generation_cost=(time.monotonic()-generation_started)*1.2
                if candidates[0]['objective'] < best['objective']: best = candidates[0]
                elites = np.array([c['parameters'] for c in candidates[:max(3, population // 5)]])
                mean = .3 * mean + .7 * elites.mean(0)
                sigma = np.maximum(BOUNDS * .06, .3 * sigma + .7 * elites.std(0))
                record_ = dict(generation=generation, candidates=candidates, best=best,
                               mean=mean.tolist(), sigma=sigma.tolist(), elapsed=time.monotonic()-started)
                atomic_json(output / f'generation_{generation:02d}.json', record_)
                records.append(record_)
                print(json.dumps(dict(generation=generation, objective=best['objective'],
                      passes=best['passes'], falls=best['falls'], parameters=best['parameters'],
                      elapsed=time.monotonic()-started)), flush=True)
            parity = export(source, action_offset(best['parameters']), output / 'candidate.onnx',feedback)
            result = dict(completed=True, baseline=baseline, best=best, parity=parity,
                          generations=len(records), elapsed=time.monotonic()-started,
                          decision='unpromoted; requires independent native development comparison')
            atomic_json(output / 'completed.json', result)
            return result
    finally:
        for world in worlds:
            if world is not None: world.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--generations', type=int, default=8); parser.add_argument('--population', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4); parser.add_argument('--minutes', type=float, default=12.)
    parser.add_argument('--initial');parser.add_argument('--spread',type=float,default=.65)
    parser.add_argument('--feedback',type=float,nargs=2,default=(0.,0.));parser.add_argument('--cases',nargs='+')
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.out, args.generations, args.population, args.workers, args.minutes,args.initial,args.spread,args.feedback,args.cases)))
