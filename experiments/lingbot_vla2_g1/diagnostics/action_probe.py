"""Bounded CUDA diagnostics on original validation and admitted TRAIN prefix.

No optimizer, new checkpoint, simulator, task-score selection, or formal-test
training. Predictions and grouped errors remain diagnostics, never task success.
Run only inside the shared FIFO and the owned GPU budget supervisor.
"""
import argparse
import contextlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from experiments.pi05_g1.training import EpisodeDataset, memory_snapshot
from experiments.pi05_g1_mobile.carry_prefix_data import CarryPrefix
from experiments.lingbot_vla2_g1.contract import ACTION_NAMES, ACTION_MASK
from experiments.lingbot_vla2_g1.data import OWN, SHARED, SPLIT, MeanStd, sha
from experiments.lingbot_vla2_g1.runtime import Runtime


GROUPS = {'arms_rad': slice(0, 14), 'pretrained_hand_slots_rad': slice(14, 26),
          'new_hand_slots_rad': slice(26, 28), 'height_m': slice(28, 29),
          'vx_m_s': slice(29, 30), 'vy_m_s': slice(30, 31),
          'yaw_rad_s': slice(31, 32)}


def errors(prediction, target, normalizer):
    p, t = np.asarray(prediction), np.asarray(target)
    difference = p - t
    normalized = normalizer.normalize(p, 'actions') - normalizer.normalize(t, 'actions')
    return {'physical_rmse_by_group': {n: float(np.sqrt(np.mean(difference[..., s] ** 2))) for n, s in GROUPS.items()},
            'normalized_rmse_by_group': {n: float(np.sqrt(np.mean(normalized[..., s] ** 2))) for n, s in GROUPS.items()},
            'per_channel_rmse': dict(zip(ACTION_NAMES, np.sqrt(np.mean(difference.reshape(-1, 32) ** 2, axis=0)).tolist()))}


def infer(runtime, rgb, state, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with contextlib.redirect_stdout(__import__('sys').stderr):
        return runtime.infer(rgb, state)


def cache_equivalence(runtime, sample):
    """Compare actual training joint attention and inference cached attention."""
    fm = runtime.model.model
    module = __import__(fm.__module__, fromlist=['make_att_2d_masks'])
    rgb, state, target, mask = sample
    inputs = runtime.inputs(rgb, state, target, mask)
    inputs.pop('joint_mask')
    outcomes = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.manual_seed(4242)
        noise = torch.randn_like(inputs['actions'])
        for t in (.05, .5, .95):
            timestep = torch.full((1,), t, device=runtime.device, dtype=inputs['state'].dtype)
            captured = []
            handle = fm.action_out_proj.register_forward_hook(lambda module, args, output: captured.append(output.detach().clone()))
            try:
                fm(**inputs, noise=noise, time=timestep)
            finally:
                handle.remove()
            joint = captured[-1]
            prefix, pads, att, pos, visual, stack = fm.embed_prefix(inputs['images'], inputs['img_masks'], inputs['lang_tokens'], inputs['lang_masks'], image_grid_thw=inputs['image_grid_thw'])
            _, cache, _ = fm.qwenvl_with_expert.forward(attention_mask=module.make_att_2d_masks(pads, att), position_ids=pos, vlm_position_ids=pos, past_key_values=None, inputs_embeds=[prefix, None], use_cache=True, fill_kv_cache=True, visual_pos_masks=visual, deepstack_visual_embeds=stack)
            x = timestep[:, None, None] * noise + (1 - timestep[:, None, None]) * inputs['actions']
            cached = fm.predict_velocity(inputs['state'], pads, cache, x, timestep, prefix_position_ids=pos)
            difference = (joint.float() - cached.float()).abs()
            outcomes.append({'time': t, 'max_abs_velocity_difference': float(difference.max()), 'mean_abs_velocity_difference': float(difference.mean()), 'joint_velocity_rms': float(joint.float().square().mean().sqrt())})
    return outcomes


def padding_ablation(runtime, sample):
    """Probe the zero-target unused slots' training-noise path, diagnostic only."""
    fm = runtime.model.model
    original = fm.predict_velocity
    valid = torch.as_tensor(ACTION_MASK, device=runtime.device)[None, None, :]
    first_noise = None

    def constrained(state, pads, cache, x, timestep, **kwargs):
        nonlocal first_noise
        if first_noise is None:
            first_noise = x.detach().clone()
        velocity = original(state, pads, cache, x, timestep, **kwargs)
        return torch.where(valid, velocity, first_noise)

    rgb, state, target, mask = sample
    normal = runtime.normalizer
    expected = normal.denormalize(target[mask], 'actions')
    baseline = infer(runtime, rgb, state)[mask]
    try:
        fm.predict_velocity = constrained
        candidate = infer(runtime, rgb, state)[mask]
    finally:
        fm.predict_velocity = original
    return {'scope': 'Diagnostic sampled-action comparison only; no policy deployment or autonomous success claim.',
            'baseline': errors(baseline, expected, normal), 'unused_slots_follow_training_noise_path': errors(candidate, expected, normal),
            'prediction_max_abs_change': float(np.max(np.abs(candidate - baseline)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delta', type=Path, required=True)
    parser.add_argument('--prefix-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve diagnostics from every attempt')
    started = time.monotonic()
    report = {'schema': 'lingbot_resumed_sampled_action_probe_v1', 'status': 'running', 'GPU_used': True, 'optimizer_updates': 0, 'source_test_opened': False, 'formal_observations_used_for_training': False}
    args.output.mkdir(parents=True)

    def event(name, **values):
        print(json.dumps({'event': name, 'elapsed_seconds': time.monotonic() - started, **values}), flush=True)

    try:
        torch.set_num_threads(4)
        normalizer = MeanStd(OWN / 'train_meanstd.json')
        validation = EpisodeDataset(SHARED / 'data/lerobot', SPLIT, 'validation', normalizer)
        prefix = CarryPrefix(args.prefix_manifest, normalizer)
        runtime = Runtime(delta=args.delta)
        event('model_loaded', load_seconds=runtime.load_seconds, **memory_snapshot(torch))
        report['cache_equivalence'] = cache_equivalence(runtime, prefix.sample(0))
        event('joint_and_cached_forward_compared', measurements=report['cache_equivalence'])
        report['unused_padding_ablation'] = padding_ablation(runtime, prefix.sample(0))
        records = []
        for label, dataset, indices in [('original_validation', validation, validation.validation_indices()),
                                         ('trained_development_prefix', prefix, range(0, len(prefix.indices), 50))]:
            predictions, targets, details = [], [], []
            for count, index in enumerate(indices):
                rgb, state, actions, mask = dataset.sample(index)
                prediction = infer(runtime, rgb, state)
                expected = normalizer.denormalize(actions, 'actions')
                predictions.append(prediction[mask]); targets.append(expected[mask])
                details.append({'index': index, 'valid_frames': int(mask.sum()), **errors(prediction[mask], expected[mask], normalizer)})
                if count % 25 == 0:
                    event('sampling_progress', split=label, chunks=count + 1)
            p, t = np.concatenate(predictions), np.concatenate(targets)
            np.savez_compressed(args.output / (label + '_predictions.npz'), prediction=p, target=t)
            value = {'source': label, 'chunks': len(details), 'frames': len(p), **errors(p, t, normalizer), 'details': details}
            records.append(value)
            event('sampled_split_complete', split=label, chunks=len(details), frames=len(p), errors=value['normalized_rmse_by_group'])
        report.update(status='completed', delta=str(args.delta.absolute()), delta_sha256=sha(args.delta / 'trainable.safetensors'),
                      prefix_manifest_sha256=sha(args.prefix_manifest), norm_sha256=sha(OWN / 'train_meanstd.json'),
                      sources=records, model_load_seconds=runtime.load_seconds, **memory_snapshot(torch))
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        with (args.output / 'result.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        event('diagnostic_finished', status=report['status'], result=str(args.output / 'result.json'))


if __name__ == '__main__':
    main()
