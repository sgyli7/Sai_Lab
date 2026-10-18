"""TRAIN-only sampled action probe for placement/velocity failure localization."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from experiments.lingbot_vla2_g1.coherent_sampling import ADAPTER, infer
from experiments.lingbot_vla2_g1.constant_inverse import PointMassRuntime
from experiments.lingbot_vla2_g1.contract import encode_state
from experiments.lingbot_vla2_g1.data import OWN, sha
from .precision_probe import errors

SCHEMA = 'lingbot_coherent_TRAIN_action_probe_v1'


def run(manifest_path, output):
    manifest_path, output = [Path(p).absolute() for p in (manifest_path, output)]
    if output.exists() or not output.is_relative_to(OWN):
        raise FileExistsError('Fresh owned probe output required')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != SCHEMA or manifest.get('TRAIN_only') is not True
            or manifest.get('sampling_adapter') != ADAPTER or len(manifest['samples']) != 20):
        raise ValueError('Twenty-observation TRAIN-only coherent probe required')
    for name, digest in manifest['frozen_inputs'].items():
        if sha(name) != digest:
            raise ValueError('Probe input changed: ' + name)
    traces, observations = {}, []
    for item in manifest['samples']:
        if item['trace'] not in traces:
            traces[item['trace']] = [json.loads(x) for x in Path(item['trace']).read_text().splitlines()]
        rows, tick = traces[item['trace']], item['tick']
        row = rows[tick]
        rgb = np.asarray(Image.open(item['image']).convert('RGB'))
        state = encode_state(np.asarray(row['joint_positions'], np.float32), row['source_joint_names'])
        target = np.asarray([x['applied_action32'] for x in rows[tick+1:tick+51]], np.float32)
        if target.shape != (50, 32) or rgb.shape != (480, 640, 3):
            raise ValueError('Exact measured input and next50 applied actions required')
        observations.append((rgb, state, target))
    output.mkdir()
    started = time.monotonic()
    torch.set_num_threads(4)
    with contextlib.redirect_stdout(sys.stderr):
        runtime = PointMassRuntime(delta=Path(manifest['delta']))
    predictions, raw_samples, details = [], [], []
    with (output / 'progress.jsonl').open('x', buffering=1) as stream:
        for index, (rgb, state, target) in enumerate(observations):
            predicted, samples = infer(runtime, rgb, state)
            torch.cuda.synchronize()
            predictions.append(predicted); raw_samples.append(samples)
            item = {'index': index, **manifest['samples'][index], 'error': errors(predicted, target),
                    'predicted_height_body_mean': predicted[:, 28:32].mean(0).tolist(),
                    'target_height_body_mean': target[:, 28:32].mean(0).tolist()}
            details.append(item)
            stream.write(json.dumps(item, allow_nan=False) + '\n')
            print(json.dumps({'event': 'coherent_TRAIN_probe_sample', **item}), flush=True)
    predictions = np.asarray(predictions, np.float32)
    targets = np.asarray([o[2] for o in observations], np.float32)
    with (output / 'sampled_actions.npz').open('xb') as stream:
        np.savez(stream, predictions=predictions, targets=targets, raw_samples=np.asarray(raw_samples, np.float32))
    # Candidate gain is diagnostic only. It is never applied by this probe.
    p, t = predictions[:, :, 29:32].astype(np.float64), targets[:, :, 29:32].astype(np.float64)
    gains = (p * t).sum((0, 1)) / np.maximum((p * p).sum((0, 1)), 1e-12)
    clipped = np.clip(gains, .5, 3.)
    folds = []
    for held in range(len(p)):
        keep = np.arange(len(p)) != held
        gain = np.clip((p[keep] * t[keep]).sum((0, 1)) / np.maximum((p[keep]**2).sum((0, 1)), 1e-12), .5, 3.)
        folds.append({'held_index': held, 'gain': gain.tolist(),
                      'raw_MSE_by_axis': ((p[held] - t[held])**2).mean(0).tolist(),
                      'fitted_MSE_by_axis': ((p[held] * gain - t[held])**2).mean(0).tolist()})
    result = {'schema': SCHEMA, 'status': 'completed', 'TRAIN_only': True, 'new_physical_trial': False,
              'sampling_adapter': ADAPTER, 'delta': manifest['delta'], 'manifest_sha256': sha(manifest_path),
              'observations': details, 'sampled_action_error': errors(predictions, targets),
              'diagnostic_only_body_gain_unconstrained': gains.tolist(),
              'diagnostic_only_body_gain_positive_bounded': clipped.tolist(),
              'diagnostic_gain_applied_to_any_robot': False, 'leave_one_TRAIN_observation_out': folds,
              'calibration_is_not_heldout_task_evidence': True,
              'arrays_sha256': sha(output / 'sampled_actions.npz'),
              'actual_robot_success_claimed': False, 'elapsed_seconds': time.monotonic() - started}
    with (output / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ('status', 'sampled_action_error', 'diagnostic_only_body_gain_positive_bounded')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.output)


if __name__ == '__main__':
    main()
