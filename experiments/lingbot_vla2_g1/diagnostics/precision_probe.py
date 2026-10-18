"""Paired inference arithmetic control on immutable TRAIN observations.

Both paths use the same existing BF16-stored base/delta and identical rounded
noise values. FP32 computation here cannot recover original FP32 checkpoint
weights and does not reproduce the official RoboTwin benchmark.
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from experiments.lingbot_vla2_g1.contract import encode_state, unpack_actions
from experiments.lingbot_vla2_g1.data import OWN, sha
from experiments.lingbot_vla2_g1.runtime import Runtime


GROUPS = {'arms_rad': (0, 14), 'hands_rad': (14, 28), 'height_m': (28, 29),
          'base_xy_m_s': (29, 31), 'yaw_rad_s': (31, 32)}


def errors(predicted, target):
    return {name: float(np.sqrt(np.mean((predicted[..., start:stop] - target[..., start:stop]) ** 2)))
            for name, (start, stop) in GROUPS.items()}


def noise_pair(seed, shape=(1, 50, 55)):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    # Values are exactly representable in both formats; no dtype-dependent RNG.
    bf16 = torch.randn(shape, generator=generator, dtype=torch.float32).bfloat16()
    return bf16, bf16.float()


def run(args):
    output = Path(args.output).absolute()
    if output.exists() or not output.is_relative_to(OWN):
        raise FileExistsError('Fresh owned numerical-control result required')
    manifest = json.loads(args.manifest.read_text())
    if (manifest.get('schema') != 'lingbot_paired_precision_TRAIN_manifest_v1'
            or manifest.get('TRAIN_only') is not True
            or manifest.get('official_benchmark_reproduced') is not False
            or len(manifest['samples']) != 8):
        raise ValueError('Explicit eight-observation TRAIN arithmetic control required')
    for name, expected in manifest['frozen_inputs'].items():
        if sha(name) != expected:
            raise ValueError('Frozen precision-control input changed: ' + name)
    trace = [json.loads(line) for line in Path(manifest['source_trace']).read_text().splitlines()]
    source = json.loads(Path(manifest['source_result']).read_text())
    if source['case']['seed'] != 2026101702:
        raise ValueError('Prospectively identified TRAIN source required')
    samples = []
    for item in manifest['samples']:
        tick = item['tick']
        row = trace[tick]
        rgb = np.asarray(Image.open(item['image']).convert('RGB'))
        if rgb.shape != (480, 640, 3):
            raise ValueError('Exact frozen head image shape required')
        state = encode_state(np.asarray(row['joint_positions'], np.float32), row['source_joint_names'])
        target = np.asarray([r['applied_action32'] for r in trace[tick + 1:tick + 51]], np.float32)
        if target.shape != (50, 32):
            raise ValueError('Fifty next-applied TRAIN actions required')
        samples.append((item, rgb, state, target))
    output.mkdir()
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    with contextlib.redirect_stdout(sys.stderr):
        runtime = Runtime(delta=Path(manifest['delta']))
    outputs, details = {}, {}
    with (output / 'progress.jsonl').open('x', buffering=1) as progress:
        for name, dtype in (('bf16', torch.bfloat16), ('fp32', torch.float32)):
            if name == 'fp32':
                runtime.model.to(dtype=torch.float32)
                if any(p.is_floating_point() and p.dtype != torch.float32 for p in runtime.model.parameters()):
                    raise ValueError('FP32 control has non-FP32 parameters')
            outputs[name], details[name] = [], []
            for item, rgb, state, target in samples:
                inputs = runtime.inputs(rgb, state)
                inputs = {key: (value.to(dtype=dtype) if value.is_floating_point() else value)
                          for key, value in inputs.items()}
                noise = noise_pair(item['noise_seed'])[0 if name == 'bf16' else 1].to(runtime.device)
                tick = time.monotonic()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=name == 'bf16'):
                    with contextlib.redirect_stdout(sys.stderr):
                        value = runtime.model.sample_actions(**inputs, noise=noise)
                torch.cuda.synchronize()
                actions = runtime.normalizer.denormalize(unpack_actions(value[0].float().cpu().numpy()), 'actions')
                if actions.shape != (50, 32) or not np.isfinite(actions).all():
                    raise ValueError('Finite paired control actions required')
                outputs[name].append(actions)
                record = {'event': 'paired_precision_inference', 'precision': name, 'tick': item['tick'],
                          'elapsed_seconds': time.monotonic() - tick,
                          'TRAIN_action_fit': errors(actions, target),
                          'allocated_bytes': torch.cuda.memory_allocated()}
                details[name].append(record)
                progress.write(json.dumps(record, allow_nan=False) + '\n')
                print(json.dumps(record, allow_nan=False), flush=True)
    arrays = {key: np.asarray(value, np.float32) for key, value in outputs.items()}
    arrays['teacher_next_actions32'] = np.asarray([sample[3] for sample in samples], np.float32)
    with (output / 'paired_actions.npz').open('xb') as stream:
        np.savez(stream, **arrays)
    result = {'schema': 'lingbot_paired_precision_TRAIN_result_v1', 'status': 'completed',
              'device': runtime.device_name, 'manifest_sha256': sha(args.manifest),
              'same_stored_BF16_weights': True, 'original_FP32_checkpoint_weights_restored': False,
              'same_representable_initial_noise_values': True, 'TF32_enabled': False,
              'sample_count': len(samples), 'new_physical_trial': False,
              'official_benchmark_reproduced': False, 'LingBot_task_success_proved': False,
              'bf16_TRAIN_action_fit': errors(arrays['bf16'], arrays['teacher_next_actions32']),
              'fp32_TRAIN_action_fit': errors(arrays['fp32'], arrays['teacher_next_actions32']),
              'paired_output_difference': errors(arrays['bf16'], arrays['fp32']),
              'observations': details, 'paired_actions_sha256': sha(output / 'paired_actions.npz'),
              'elapsed_seconds': time.monotonic() - started,
              'scope': 'Numerical precision control on eight TRAIN images only; neither held-out task evaluation nor official checkpoint reproduction.'}
    with (output / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ('status', 'bf16_TRAIN_action_fit', 'fp32_TRAIN_action_fit', 'paired_output_difference', 'elapsed_seconds')}), flush=True)


def prepare_manifest(output):
    output = Path(output).absolute()
    if output.exists() or not output.is_relative_to(OWN):
        raise FileExistsError('Fresh owned precision manifest required')
    source = OWN / 'resumed_turn01/episodes/resume01_native_teacher01'
    result = json.loads((source / 'result.json').read_text())
    if result['case']['seed'] != 2026101702 or sha(source / 'trajectory.jsonl') != result['trace_sha256']:
        raise ValueError('Actual unchanged TRAIN source required')
    paths = [source / name for name in ('config.json', 'result.json', 'trajectory.jsonl')]
    delta = OWN / 'training_round02/selected'
    paths += [delta / 'metadata.json', OWN / 'train_meanstd.json', OWN / 'contract.json',
              OWN / 'bf16_conversion_receipt.json', OWN / 'runtime_overlay_receipt.json']
    samples = []
    for index, tick in enumerate((0, 50, 100, 150, 200, 250, 1000, 1400)):
        image = source / 'head_rgb' / (f'{tick:05}.png')
        paths.append(image)
        samples.append({'tick': tick, 'image': str(image), 'noise_seed': 90142 + index})
    manifest = {'schema': 'lingbot_paired_precision_TRAIN_manifest_v1', 'TRAIN_only': True,
                'source_trace': str(source / 'trajectory.jsonl'), 'source_result': str(source / 'result.json'),
                'delta': str(delta), 'samples': samples, 'official_benchmark_reproduced': False,
                'source_test_or_formal_trajectories_used': False,
                'frozen_inputs': {str(path): sha(path) for path in paths}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        manifest = prepare_manifest(args.prepare)
        # CPU negative control: standard independent FP32 generation is not
        # assumed equal to BF16 generation; this helper guarantees exact values.
        a, b = noise_pair(90142)
        if not torch.equal(a.float(), b):
            raise AssertionError('Paired initial noise differs')
        print(json.dumps({'manifest': str(args.prepare), 'samples': len(manifest['samples']),
                          'CPU_noise_equality_passed': True, 'GPU_started': False}))
    else:
        if args.manifest is None or args.output is None:
            parser.error('--manifest and --output required for GPU control')
        run(args)


if __name__ == '__main__':
    main()
