"""Search a small velocity-dependent brake residual in unchanged Jolt physics."""
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

from sim2sim.policy_state import BRAKE_STATE_V1
from .brake_pose_search import action_offset, rollout
from .budget import require_supervision
from .models import NativeAnchor
from .queue import atomic_json


class VelocityPose:
    def __init__(self, anchor, parameters, pitch_mode='centered'):
        if anchor.state_input != BRAKE_STATE_V1:
            raise ValueError('Velocity residual requires the declared state input')
        parameters = np.asarray(parameters, np.float32)
        if parameters.shape not in ((10,), (15,)) or not np.isfinite(parameters).all():
            raise ValueError('Expected ten or fifteen finite residual coefficients')
        self.anchor = anchor
        self.state_input = anchor.state_input
        self.bias = action_offset(parameters[:5])
        self.velocity_gain = action_offset(parameters[5:10])
        self.pitch_gain = action_offset(parameters[10:15]) if len(parameters)==15 else None
        if pitch_mode not in ('centered','excess_forward_tilt'):
            raise ValueError('Unknown pitch feedback feature')
        self.pitch_reference=np.float32(.22 if pitch_mode=='centered' else np.sin(np.deg2rad(15.)))
        self.pitch_lower=np.float32(-1. if pitch_mode=='centered' else 0.)
        self.pitch_mode=pitch_mode

    def __call__(self, observations):
        x = np.asarray(observations, np.float32).reshape(-1, 61)
        velocity = np.clip(x[:, 58:59] / np.float32(.6), -1., 1.)
        gate = np.clip(-x[:, 48:49] / np.float32(.05), 0., 1.)
        residual = self.bias + velocity * self.velocity_gain
        if self.pitch_gain is not None:
            pitch = np.clip((x[:, 3:4]-self.pitch_reference)/np.float32(.3), self.pitch_lower, 1.)
            residual = residual + pitch * self.pitch_gain
        return self.anchor(x) + gate * residual


def export(source, parameters, target, pitch_mode='centered'):
    anchor = NativeAnchor(source)
    actor = VelocityPose(anchor, parameters, pitch_mode)
    model = onnx.load(source)
    prefix = 'velocity_pose/'
    names = {name for node in model.graph.node for name in [*node.input, *node.output]}
    while any(name.startswith(prefix) for name in names):
        prefix += 'next/'
    def n(label): return prefix + label
    constants = dict(velocity_index=np.array([58], np.int64), throttle_index=np.array([48], np.int64),
        velocity_scale=np.array(.6, np.float32), throttle_scale=np.array(-.05, np.float32),
        zero=np.array(0., np.float32), one=np.array(1., np.float32), minus_one=np.array(-1., np.float32),
        bias=actor.bias.reshape(1, 14), velocity_gain=actor.velocity_gain.reshape(1, 14))
    for label, value in constants.items():
        model.graph.initializer.append(numpy_helper.from_array(value, n(label)))
    original = model.graph.output[0].name
    nodes = [
        helper.make_node('Gather', [model.graph.input[0].name, n('velocity_index')], [n('velocity')], axis=1),
        helper.make_node('Div', [n('velocity'), n('velocity_scale')], [n('normalized_velocity')]),
        helper.make_node('Clip', [n('normalized_velocity'), n('minus_one'), n('one')], [n('bounded_velocity')]),
        helper.make_node('Mul', [n('bounded_velocity'), n('velocity_gain')], [n('velocity_delta')]),
        helper.make_node('Add', [n('bias'), n('velocity_delta')], [n('residual')]),
        helper.make_node('Gather', [model.graph.input[0].name, n('throttle_index')], [n('throttle')], axis=1),
        helper.make_node('Div', [n('throttle'), n('throttle_scale')], [n('negative_weight')]),
        helper.make_node('Clip', [n('negative_weight'), n('zero'), n('one')], [n('gate')]),
    ]
    residual = n('residual')
    if actor.pitch_gain is not None:
        for label, value in dict(pitch_index=np.array([3],np.int64), pitch_reference=np.array(actor.pitch_reference,np.float32),
                pitch_lower=np.array(actor.pitch_lower,np.float32),pitch_scale=np.array(.3,np.float32),
                pitch_gain=actor.pitch_gain.reshape(1,14)).items():
            model.graph.initializer.append(numpy_helper.from_array(value,n(label)))
        nodes.extend([
            helper.make_node('Gather',[model.graph.input[0].name,n('pitch_index')],[n('pitch')],axis=1),
            helper.make_node('Sub',[n('pitch'),n('pitch_reference')],[n('pitch_error')]),
            helper.make_node('Div',[n('pitch_error'),n('pitch_scale')],[n('normalized_pitch')]),
            helper.make_node('Clip',[n('normalized_pitch'),n('pitch_lower'),n('one')],[n('bounded_pitch')]),
            helper.make_node('Mul',[n('bounded_pitch'),n('pitch_gain')],[n('pitch_delta')]),
            helper.make_node('Add',[n('residual'),n('pitch_delta')],[n('combined_residual')]),
        ])
        residual=n('combined_residual')
    nodes.extend([
        helper.make_node('Mul',[n('gate'),residual],[n('delta')]),
        helper.make_node('Add',[original,n('delta')],[n('actions')]),
    ])
    model.graph.node.extend(nodes)
    model.graph.output[0].name = n('actions')
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata['sim2sim_velocity_pose'] = json.dumps(dict(parent_sha256=anchor.sha256,
        parameters=np.asarray(parameters).tolist(), normalization=.6,
        pitch_mode=pitch_mode if actor.pitch_gain is not None else None,
        pitch_reference=float(actor.pitch_reference) if actor.pitch_gain is not None else None,
        pitch_normalization=.3 if actor.pitch_gain is not None else None))
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, target)
    x = np.random.default_rng(915003).normal(0, .3, (512, 61)).astype(np.float32)
    x[:64, 48] = 0.
    x[64:128, 48] = -.5
    x[64:128, 58] = np.linspace(-1., 1., 64)
    actual = NativeAnchor(target)(x)
    error = float(np.max(abs(actual - actor(x))))
    exact = np.array_equal(actual[x[:, 48] >= 0], anchor(x[x[:, 48] >= 0]))
    if error >= 1e-5 or not exact:
        raise RuntimeError('Velocity residual export failed parity')
    return dict(max_abs=error, positive_and_neutral_exact=exact,
        sha256=hashlib.sha256(Path(target).read_bytes()).hexdigest())


def run(experiment, output, minutes=20., generations=12, population=20, workers=4):
    require_supervision(Path(os.environ['SIM2SIM_RESEARCH_DIR']))
    spec = json.loads(Path(experiment).read_text())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rng = np.random.default_rng(915003)
    bounds = np.array(spec['bounds'], float)
    if bounds.shape != (10,) or not np.isfinite(bounds).all() or np.any(bounds <= 0):
        raise ValueError('Expected ten positive search bounds')
    mean = np.zeros(10)
    sigma = bounds * .35
    anchors = [NativeAnchor(spec['source']) for _ in range(workers)]
    if any(anchor.sha256 != spec['source_sha256'] for anchor in anchors):
        raise ValueError('Source checksum mismatch')
    spec.update(minutes=minutes, generations=generations, population=population, workers=workers,
        code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path(__file__).with_name('brake_pose_search.py')]})
    atomic_json(output / 'experiment.json', spec)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def evaluate(parameters, label):
            jobs = [(i, p, case, seed) for i, p in enumerate(parameters)
                for case in spec['cases'] for seed in spec['seeds']]
            rows = []
            for start in range(0, len(jobs), workers):
                futures = [pool.submit(rollout, None, VelocityPose(anchors[lane], p), np.zeros(5),
                    seed, output, f'{label}_{i:03d}_{case}', case)
                    for lane, (i, p, case, seed) in enumerate(jobs[start:start + workers])]
                rows.extend(f.result() for f in futures)
            candidates = []
            for i, p in enumerate(parameters):
                trials = [row for row in rows if row['trial'].startswith(f'{label}_{i:03d}_')]
                candidates.append(dict(parameters=np.asarray(p).tolist(), trials=trials,
                    objective=float(np.mean([row['objective'] for row in trials])),
                    passes=sum(row['metrics']['success'] for row in trials),
                    falls=sum(row['metrics']['fell'] for row in trials)))
            return sorted(candidates, key=lambda row: row['objective'])
        baseline = best = evaluate([mean], 'baseline')[0]
        atomic_json(output / 'baseline.json', baseline)
        cost = 3 * population * len(spec['cases']) * len(spec['seeds']) / workers
        history = []
        for generation in range(generations):
            if time.monotonic() - started + cost + 20 > minutes * 60:
                break
            begin = time.monotonic()
            parameters = np.clip(rng.normal(mean, sigma, (population, 10)), -bounds, bounds)
            parameters[0] = best['parameters']
            candidates = evaluate(parameters, f'g{generation:02d}')
            cost = (time.monotonic() - begin) * 1.2
            if candidates[0]['objective'] < best['objective']:
                best = candidates[0]
            elite = np.array([row['parameters'] for row in candidates[:max(3, population // 5)]])
            mean = .3 * mean + .7 * elite.mean(0)
            sigma = np.maximum(bounds * .025, .3 * sigma + .7 * elite.std(0))
            row = dict(generation=generation, best=best, candidates=candidates, mean=mean.tolist(),
                sigma=sigma.tolist(), elapsed=time.monotonic() - started)
            atomic_json(output / f'generation_{generation:02d}.json', row)
            history.append(row)
            print(json.dumps({k: row['best'][k] for k in ['passes', 'falls', 'objective']} |
                dict(generation=generation, elapsed=row['elapsed'])), flush=True)
        parity = export(spec['source'], best['parameters'], output / 'candidate.onnx')
        result = dict(completed=True, baseline=baseline, best=best, generations=len(history),
            parity=parity, elapsed=time.monotonic() - started, decision='unpromoted; native development suite required')
        atomic_json(output / 'completed.json', result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment'); parser.add_argument('--out', required=True)
    parser.add_argument('--minutes', type=float, default=20.)
    parser.add_argument('--generations', type=int, default=12)
    parser.add_argument('--population', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.experiment, args.out, args.minutes, args.generations, args.population, args.workers)
