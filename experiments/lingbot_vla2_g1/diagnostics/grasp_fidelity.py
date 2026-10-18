"""Frozen R4 action fidelity, stochasticity, and input-sensitivity control.

Every real observation and target is TRAIN data. Counterfactual image/state
swaps are offline controls, never robot actions or acceptance evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from experiments.lingbot_vla2_g1.constant_inverse import PointMassRuntime
from experiments.lingbot_vla2_g1.contract import encode_state, unpack_actions
from experiments.lingbot_vla2_g1.data import OWN, sha
from .precision_probe import errors, noise_pair

SCHEMA = 'lingbot_R4_TRAIN_sampled_action_fidelity_v1'


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def prepare(path):
    path = Path(path).absolute()
    if path.exists() or not path.is_relative_to(OWN):
        raise FileExistsError('Fresh owned TRAIN control manifest required')
    old = json.loads((OWN / 'diagnostics/paired_precision_manifest_v1.json').read_text())
    definitions = [
        ('native_TRAIN', Path(old['source_trace']).parent,
         [item['tick'] for item in old['samples']]),
        ('full_scripted_TRAIN', OWN / 'placement_teacher02/episode',
         [0, 100, 150, 200, 250, 350, 1000, 1500, 1800, 2400, 2600, 2800]),
    ]
    frozen, samples = {}, []
    for family, folder, ticks in definitions:
        result_path, trace_path = folder / 'result.json', folder / 'trajectory.jsonl'
        result = json.loads(result_path.read_text())
        if result['case']['seed'] != 2026101702 or sha(trace_path) != result['trace_sha256']:
            raise ValueError('Reviewed nonformal TRAIN source required')
        rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
        for name in ('config.json', 'result.json', 'trajectory.jsonl'):
            frozen[str(folder / name)] = sha(folder / name)
        for tick in ticks:
            if len(rows[tick+1:tick+51]) != 50:
                raise ValueError('Complete fifty-next-applied-action target required')
            image = folder / 'head_rgb' / f'{tick:05}.png'
            rgb = np.asarray(Image.open(image).convert('RGB'))
            if rgb.shape != (480, 640, 3) or hashlib.sha256(rgb.tobytes()).hexdigest() != rows[tick]['head_rgb_sha256']:
                raise ValueError('Exact synchronized raw TRAIN RGB required')
            frozen[str(image)] = sha(image)
            samples.append({'family': family, 'tick': tick, 'trace': str(trace_path),
                            'image': str(image), 'noise_seed': 90142 + len(samples)})
    baseline = OWN / 'paired_precision_control01/execution/paired_actions.npz'
    for name in (baseline, OWN / 'training_task_round04/final/metadata.json',
                 OWN / 'train_meanstd.json', OWN / 'contract.json'):
        frozen[str(name)] = sha(name)
    manifest = {'schema': SCHEMA, 'TRAIN_only': True, 'source_formal_or_test_data': False,
                'held_out_accuracy_claimed': False, 'samples': samples, 'noise_repetitions': 3,
                'delta': str(OWN / 'training_task_round04/final'), 'R2_baseline_npz': str(baseline),
                'frozen_inputs': frozen, 'actual_robot_success_claimed': False}
    write_new(path, manifest)
    return manifest


def run(manifest_path, output):
    manifest_path, output = [Path(p).absolute() for p in (manifest_path, output)]
    if output.exists() or not output.is_relative_to(OWN):
        raise FileExistsError('Fresh owned numerical-control output required')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != SCHEMA or manifest.get('TRAIN_only') is not True
            or manifest.get('source_formal_or_test_data') is not False
            or len(manifest['samples']) != 20 or manifest['noise_repetitions'] != 3):
        raise ValueError('Reviewed twenty-observation, three-noise TRAIN control required')
    for path, digest in manifest['frozen_inputs'].items():
        if sha(path) != digest:
            raise ValueError('Frozen control input changed: ' + path)
    traces, samples = {}, []
    for item in manifest['samples']:
        if item['trace'] not in traces:
            traces[item['trace']] = [json.loads(x) for x in Path(item['trace']).read_text().splitlines()]
        rows, tick = traces[item['trace']], item['tick']
        row = rows[tick]
        rgb = np.asarray(Image.open(item['image']).convert('RGB'))
        state = encode_state(np.asarray(row['joint_positions'], np.float32), row['source_joint_names'])
        targets = np.asarray([r['applied_action32'] for r in rows[tick+1:tick+51]], np.float32)
        if state.shape != (32,) or targets.shape != (50, 32):
            raise ValueError('Exact32-channel state and next-action targets required')
        samples.append((rgb, state, targets))
    targets = np.asarray([item[2] for item in samples])
    with np.load(manifest['R2_baseline_npz'], allow_pickle=False) as baseline:
        old = baseline['bf16'].copy()
        if not np.array_equal(targets[:8], baseline['teacher_next_actions32']):
            raise ValueError('R2/R4 paired reference actions differ')
    output.mkdir()
    started = time.monotonic()
    torch.set_num_threads(4)
    with contextlib.redirect_stdout(sys.stderr):
        runtime = PointMassRuntime(delta=Path(manifest['delta']))
    def infer(rgb, state, seed):
        inputs = runtime.inputs(rgb, state)
        noise = noise_pair(seed)[0].to(runtime.device)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), contextlib.redirect_stdout(sys.stderr):
            raw = runtime.model.sample_actions(**inputs, noise=noise)
        torch.cuda.synchronize()
        value = runtime.normalizer.denormalize(unpack_actions(raw[0].float().cpu().numpy()), 'actions')
        if value.shape != (50, 32) or not np.isfinite(value).all():
            raise ValueError('Finite actual effective32 prediction required')
        return value
    predictions = []
    with (output / 'progress.jsonl').open('x', buffering=1) as stream:
        for repetition in range(3):
            values = []
            for index, (rgb, state, target) in enumerate(samples):
                seed = manifest['samples'][index]['noise_seed'] + repetition * 1000003
                value = infer(rgb, state, seed)
                values.append(value)
                event = {'event': 'TRAIN_sampled_action', 'repetition': repetition, 'index': index,
                         **manifest['samples'][index], 'actual_noise_seed': seed,
                         'action_error': errors(value, target)}
                stream.write(json.dumps(event, allow_nan=False) + '\n')
                print(json.dumps({key: event[key] for key in ('event', 'repetition', 'index', 'family', 'tick', 'action_error')}), flush=True)
            predictions.append(values)
    predictions = np.asarray(predictions, np.float32)
    controls, control_details = {}, []
    for index, swapped in ((0, 14), (11, 8), (13, 8), (17, 8)):
        rgb, state, _target = samples[index]
        other_rgb, other_state, _ = samples[swapped]
        seed = manifest['samples'][index]['noise_seed']
        for kind, image, q in (('image_swap', other_rgb, state), ('state_swap', rgb, other_state)):
            key = f'{kind}_{index:02}'
            controls[key] = infer(image, q, seed)
            control_details.append({'index': index, 'swapped_index': swapped, 'control': kind,
                                    'output_change_RMS': errors(controls[key], predictions[0, index])})
    repeated = infer(*samples[0][:2], manifest['samples'][0]['noise_seed'])
    repeat_error = float(np.max(np.abs(repeated - predictions[0, 0])))
    if repeat_error > 1e-5:
        raise ValueError('Same-input/same-noise repeat is not stable')
    arrays = {'predictions': predictions, 'targets': targets, 'R2_native_first_noise': old,
              'same_noise_repeat': repeated, **controls}
    with (output / 'actions_and_controls.npz').open('xb') as stream:
        np.savez(stream, **arrays)
    average = predictions.mean(axis=0)
    details = []
    for index, item in enumerate(manifest['samples']):
        details.append({**item, 'per_noise_fit': [errors(p[index], targets[index]) for p in predictions],
                        'three_noise_mean_fit': errors(average[index], targets[index]),
                        'noise_output_RMS': errors(predictions[:, index], average[index][None])})
    result = {'schema': SCHEMA, 'status': 'completed', 'TRAIN_only': True,
              'held_out_accuracy_claimed': False, 'actual_robot_success_claimed': False,
              'new_physical_trial': False, 'device': runtime.device_name,
              'delta': manifest['delta'], 'manifest_sha256': sha(manifest_path),
              'sample_count': 20, 'noise_repetitions': 3,
              'same_noise_repeat_maxabs': repeat_error,
              'R2_native_first_noise_fit': errors(old, targets[:8]),
              'R4_native_paired_first_noise_fit': errors(predictions[0, :8], targets[:8]),
              'R4_all_single_noise_fit': errors(predictions, targets[None]),
              'R4_three_noise_mean_fit': errors(average, targets),
              'R4_noise_output_RMS': errors(predictions, average[None]),
              'input_swap_controls': control_details, 'samples': details,
              'arrays_sha256': sha(output / 'actions_and_controls.npz'),
              'elapsed_seconds': time.monotonic() - started}
    write_new(output / 'result.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'R2_native_first_noise_fit', 'R4_native_paired_first_noise_fit', 'R4_three_noise_mean_fit', 'R4_noise_output_RMS')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        manifest = prepare(args.prepare)
        print(json.dumps({'TRAIN_samples': len(manifest['samples']), 'GPU_used': False}))
    else:
        if args.manifest is None or args.output is None:
            parser.error('--manifest and --output required')
        run(args.manifest, args.output)


if __name__ == '__main__':
    main()
