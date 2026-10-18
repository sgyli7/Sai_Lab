"""Bounded inference on successful TRAIN observations; never a task score."""
import argparse
import contextlib
import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from experiments.lingbot_vla2_g1.coherent_sampling import infer, ADAPTER
from experiments.lingbot_vla2_g1.constant_inverse import PointMassRuntime
from experiments.lingbot_vla2_g1.contract import encode_state
from experiments.lingbot_vla2_g1.data import OWN, sha
from experiments.lingbot_vla2_g1.make_plan import write
from .precision_probe import errors


def samples_from_manifest(path):
    path = Path(path).absolute()
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != 'lingbot_R8_TRAIN_placement_fit_probe_v1'
            or manifest.get('TRAIN_only') is not True
            or manifest.get('formal_trajectories_used') is not False
            or manifest.get('new_optimizer_updates') is not False
            or manifest.get('sampling_adapter') != ADAPTER
            or manifest.get('noise_seed') != 42 or len(manifest['samples']) != 12
            or Path(manifest['delta']) != OWN / 'training_task_round08/selected'):
        raise ValueError('Prospectively declared actual R8 twelve-sample TRAIN probe required')
    for name, expected in ((Path(manifest['delta']) / 'metadata.json', manifest['delta_metadata_sha256']),
                           (manifest['norm'], manifest['norm_sha256'])):
        if sha(name) != expected:
            raise ValueError('R8 runtime identity changed')
    metadata = json.loads((Path(manifest['delta']) / 'metadata.json').read_text())
    if metadata['weights_sha256'] != manifest['delta_weights_sha256']:
        raise ValueError('Actual R8 weights and declared inference identity differ')
    loaded, sources = [], {}
    for item in manifest['samples']:
        for field in ('npz', 'image', 'manifest'):
            if sha(item[field]) != item[field + '_sha256']:
                raise ValueError('Frozen TRAIN input changed: ' + item[field])
        key = item['manifest']
        if key not in sources:
            data = json.loads(Path(key).read_text())
            audit = json.loads(Path(data['independent_audit']).read_text())
            if (data.get('train_only') is not True or data.get('validation') != [] or data.get('test') != []
                    or data.get('counts_as_LingBot_success') is not False
                    or data.get('source_autonomous_model_policy') is not False
                    or data.get('formal_trajectories_used_for_training') is not False
                    or data.get('physical_outcome', {}).get('success') is not True
                    or audit.get('passed') is not True
                    or sha(data['independent_audit']) != data['independent_audit_sha256']
                    or any(check.get('passed') is not True for check in audit['checks'].values())):
                raise ValueError('Only independently successful scripted TRAIN episodes required')
            arrays = np.load(item['npz'])
            sources[key] = data, arrays
        data, arrays = sources[key]
        tick = item['tick']
        target = np.asarray(arrays['actions'][tick:tick + 50], np.float32)
        state = np.asarray(arrays['state'][tick], np.float32)
        measured = np.asarray(arrays['measured_joint_positions43'][tick], np.float32)
        if (target.shape != (50, 32) or state.shape != (32,)
                or not np.array_equal(encode_state(measured, data['source_joint_names']), state)
                or data['raw_rgb_frames'][tick]['path'] != item['image']
                or not np.isfinite(target).all() or not np.isfinite(state).all()):
            raise ValueError('Exact measured input and fifty next-applied targets required')
        rgb = np.asarray(Image.open(item['image']).convert('RGB'))
        if rgb.shape != (480, 640, 3):
            raise ValueError('Unchanged full-resolution head observation required')
        loaded.append((item, rgb, state, target))
    if len(sources) != 3 or {s[0]['family'] for s in loaded} != {
            'privileged_recovery_teacher04', 'privileged_recovery_teacher08', 'privileged_recovery_teacher09'}:
        raise ValueError('All three existing successful TRAIN recovery families required')
    return manifest, loaded


def run(manifest_path, output):
    output = Path(output).absolute()
    if output.exists() or not output.is_relative_to(OWN):
        raise FileExistsError('Fresh owned inference evidence required')
    manifest, samples = samples_from_manifest(manifest_path)
    output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    with contextlib.redirect_stdout(sys.stderr):
        runtime = PointMassRuntime(delta=Path(manifest['delta']), norm_stats=Path(manifest['norm']))
    targets, states, means, natives, records = [], [], [], [], []
    with (output / 'progress.jsonl').open('x', buffering=1) as progress:
        for item, rgb, state, target in samples:
            clock = time.monotonic()
            mean, native = infer(runtime, rgb, state)
            torch.cuda.synchronize()
            if not np.array_equal(native.mean(0, dtype=np.float32), mean):
                raise ValueError('Actual native model predictions must reconstruct coherent mean')
            record = {'family': item['family'], 'tick': item['tick'],
                      'target_fit': errors(mean, target),
                      'native_each_target_fit': [errors(x, target) for x in native],
                      'target_height_chunk_mean_m': float(target[:, 28].mean()),
                      'inferred_height_chunk_mean_m': float(mean[:, 28].mean()),
                      'native_each_height_chunk_mean_m': [float(x[:, 28].mean()) for x in native],
                      'elapsed_inference_seconds': time.monotonic() - clock,
                      'task_success_claim': False}
            targets.append(target); states.append(state); means.append(mean); natives.append(native); records.append(record)
            progress.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps({'event': 'actual_TRAIN_placement_inference', **record}, allow_nan=False), flush=True)
    payload = output / 'actual_predictions.npz'
    np.savez_compressed(payload, targets=np.asarray(targets), states=np.asarray(states),
                        coherent=np.asarray(means), native=np.asarray(natives))
    report = {'schema': 'lingbot_R8_TRAIN_placement_fit_execution_v1', 'status': 'completed',
              'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'manifest': str(Path(manifest_path).absolute()), 'manifest_sha256': sha(manifest_path),
              'model_weights_sha256': manifest['delta_weights_sha256'],
              'model_metadata_sha256': manifest['delta_metadata_sha256'],
              'TRAIN_only': True, 'formal_trajectories_used': False, 'new_optimizer_updates': False,
              'samples': len(records), 'predictions_per_sample': 2, 'sampling_adapter': ADAPTER,
              'payload': str(payload), 'payload_sha256': sha(payload), 'records': records,
              'elapsed_seconds': time.monotonic() - started,
              'parameter_elements': runtime.parameter_elements,
              'trainable_elements': runtime.trainable_elements,
              'actual_autonomous_task_success_established': False, 'overall_robot_goal_completed': False}
    write(output / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        manifest, samples = samples_from_manifest(args.manifest)
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU preparation must not initialize a GPU context')
        print(json.dumps({'status': 'verified', 'TRAIN_samples': len(samples), 'GPU_initialized': torch.cuda.is_initialized(),
                          'manifest_sha256': sha(args.manifest)}), flush=True)
    elif args.output is None:
        parser.error('--output is required for actual inference')
    else:
        report = run(args.manifest, args.output)
        print(json.dumps({'status': report['status'], 'samples': report['samples'],
                          'overall_robot_goal_completed': False}), flush=True)


if __name__ == '__main__':
    main()
