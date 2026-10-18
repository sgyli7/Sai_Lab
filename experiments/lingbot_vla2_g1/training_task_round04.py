"""R4 task adaptation: successful TRAIN placement, retained final checkpoint.

Original validation selects its own best candidate. The actual last update is
always retained separately for the prospectively declared physical development
screen, even when original short-task validation regresses. Neither diagnostic
TRAIN loss nor training completion is a robot success criterion.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from experiments.pi05_g1.training import EpisodeDataset, MasterAdamW, memory_snapshot
from experiments.pi05_g1_mobile.carry_prefix_data import raw_png
from .data import OWN, SHARED, SPLIT, SPLIT_SHA, MeanStd, sha
from .contract import PROMPT
from .runtime import Runtime
from .training import validate, snapshot_trainable, save_delta, write_new
from .training_carry_round03_dual import admit as admit_carry
from .grasp_balanced_prefix import GraspBalancedPrefix
from .carry_data_v2 import BalancedNewCarryPrefix, EqualCarryFamilies
from .scripted_success_data import FullScriptedSuccess


class TaskTrainingMixture:
    def __init__(self, original, carry, complete):
        self.sources = [original, carry, complete]
        self.indices = range(sum(len(x.indices) for x in self.sources))
        self.episodes = [ep for source in self.sources for ep in source.episodes]
        self.draw_counts = {'original80TRAIN': 0, 'carry_prefixes': 0, 'full_scripted_TRAIN': 0}

    def sample(self, index):
        value = random.random()
        family = 0 if value < .2 else 1 if value < .4 else 2
        name = ('original80TRAIN', 'carry_prefixes', 'full_scripted_TRAIN')[family]
        self.draw_counts[name] += 1
        source = self.sources[family]
        return source.sample(random.randrange(len(source.indices)))


class DiagnosticTRAINView:
    def __init__(self, source):
        self.source = source
        self.episodes = source.episodes

    def validation_indices(self):
        return range(0, len(self.source.actions), 50)

    def sample(self, index):
        source = self.source
        stop = min(index + 50, len(source.actions)); count = stop - index
        targets = np.repeat(source.actions[stop - 1:stop], 50, axis=0)
        targets[:count] = source.actions[index:stop]
        return raw_png(source.frames[index]), source.states[index], targets, np.arange(50) < count


def admission(args):
    inherited = admit_carry(args)
    path = Path(args.full_teacher_manifest).absolute()
    data = json.loads(path.read_text())
    # The real constructor additionally checks every frozen proof/image/tensor.
    if (data.get('schema') != 'lingbot_privileged_physical_full_success_TRAIN_episode_v1'
            or data.get('task_success_teacher') is not True
            or data.get('physical_outcome', {}).get('success') is not True
            or data.get('source_autonomous_model_policy') is not False
            or data.get('counts_as_LingBot_success') is not False):
        raise ValueError('Explicit independently successful physical TRAIN teacher required')
    return {**inherited, 'full_teacher_manifest': str(path), 'full_teacher_manifest_sha256': sha(path),
            'task_success_teacher': True, 'teacher_is_LingBot_success': False,
            'teacher_autonomous_model_policy': False,
            'training_source_weights': {'original80TRAIN': .2, 'carry_prefixes': .2, 'full_scripted_TRAIN': .6},
            'training_source_weights_by_family': {'original': .2, 'native201': .1,
                                                 'fresh_native2026101702': .1, 'full_scripted_TRAIN': .6},
            'full_teacher_phase_probabilities': {'approach_grasp': .45, 'transport': .20, 'placement': .35},
            'full_teacher_used_for_normalizer_fit': False,
            'physical_candidate_selection': 'Prospectively evaluate final actual optimizer boundary; source-validation best retained separately. No formal trajectory informs training or checkpoint choice.'}


def train(args):
    if args.output.exists():
        raise FileExistsError('New immutable output required')
    if not 0 < args.hours <= .82 or not 0 < args.max_steps <= 400:
        raise ValueError('Finite400update R4 recipe required')
    if sha(SPLIT) != SPLIT_SHA:
        raise ValueError('Original split changed')
    provenance = admission(args)
    args.output.mkdir(parents=True)
    started = time.monotonic(); deadline = started + args.hours * 3600
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    progress = (args.output / 'progress.jsonl').open('x', buffering=1)
    step = best_step = last_validated_step = 0
    best_snapshot = best = last = baseline = None
    identity = {'base_receipt_sha256': sha(OWN / 'bf16_conversion_receipt.json'),
                'norm_path': str(args.norm_stats.absolute()), 'norm_sha256': sha(args.norm_stats),
                'contract_sha256': sha(OWN / 'contract.json'), 'split_sha256': SPLIT_SHA,
                'runtime_overlay_receipt_sha256': sha(OWN / 'runtime_overlay_receipt.json'),
                'prompt': PROMPT, 'test_opened': False, **provenance}
    write_new(args.output / 'recipe.json', {'hours': args.hours, 'max_steps': args.max_steps,
              'microbatch': 1, 'accumulation': 8, 'lr': 2e-5, 'optimizer': 'FP32-master AdamW',
              'validation_every': 100, 'source_selection_loss': 'complete original10 validation flow_loss',
              'diagnostic_TRAIN_view_is_validation': False, 'freeze_vlm': True,
              'auxiliary_teachers': False, 'native_queries_preserved': True,
              'checkpoint_strategy': 'Always save final actual update; additionally retain strict original-validation best if new. At most two new payloads; no optimizer checkpoint.',
              **identity})
    def event(name, **payload):
        doc = {'event': name, 'step': step, 'elapsed_seconds': time.monotonic() - started, **payload}
        line = json.dumps(doc, allow_nan=False); progress.write(line + '\n')
        if name != 'optimizer_step' or step % 10 == 0:
            print(line, flush=True)
    try:
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42); torch.set_num_threads(4)
        normalizer = MeanStd(args.norm_stats)
        original = EpisodeDataset(SHARED / 'data/lerobot', SPLIT, 'train', normalizer)
        valset = EpisodeDataset(SHARED / 'data/lerobot', SPLIT, 'validation', normalizer)
        carry = EqualCarryFamilies(GraspBalancedPrefix(args.carry_prefix_manifest, normalizer),
                                   BalancedNewCarryPrefix(args.new_carry_manifest, normalizer))
        complete = FullScriptedSuccess(args.full_teacher_manifest, normalizer)
        trainset = TaskTrainingMixture(original, carry, complete)
        train_view = DiagnosticTRAINView(complete)
        runtime = Runtime(training=True, delta=args.initial_delta, norm_stats=args.norm_stats)
        parameters = [p for p in runtime.model.parameters() if p.requires_grad]
        if runtime.model.config.bias_update_speed != 0:
            raise ValueError('Fixed upstream router bias required')
        persistent = set(runtime.model.state_dict())
        buffers = {name: value.detach().cpu().clone() for name, value in runtime.model.named_buffers() if name in persistent}
        optimizer = MasterAdamW(parameters, lr=2e-5, betas=(.9, .95), eps=1e-8, weight_decay=.01, foreach=False)
        event('training_started', parameters=runtime.parameter_elements, trainable=runtime.trainable_elements, **memory_snapshot(torch))
        baseline = validate(runtime, valset, deadline - 300, lambda: stopping)
        if baseline is None:
            raise RuntimeError('No updates before complete fixed validation')
        best = last = baseline
        event('validation_complete', validation=baseline, selected=True)
        diagnostic = validate(runtime, train_view, deadline - 300, lambda: stopping)
        event('task_TRAIN_diagnostic', metrics=diagnostic, held_out_validation=False, used_for_checkpoint_selection=False)
        while step < args.max_steps and time.monotonic() < deadline - 300 and not stopping:
            tick = time.monotonic(); optimizer.zero_grad(set_to_none=True); loss_sum = 0.; complete_accum = True
            for micro in range(8):
                if stopping or time.monotonic() >= deadline - 300:
                    complete_accum = False; break
                rgb, state, actions, mask = trainset.sample(random.randrange(len(trainset.indices)))
                loss, flow = runtime.loss(rgb, state, actions, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite task adaptation loss')
                (loss / 8).backward(); loss_sum += float(loss.detach())
            if not complete_accum:
                optimizer.zero_grad(set_to_none=True); break
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(); step += 1
            event('optimizer_step', loss=loss_sum / 8, gradient_norm=float(norm), step_seconds=time.monotonic() - tick, **memory_snapshot(torch))
            if step % 100 == 0:
                value = validate(runtime, valset, deadline - 180, lambda: stopping)
                if value is not None:
                    last = value; last_validated_step = step; selected = value['flow_loss'] < best['flow_loss']
                    if selected:
                        best = value; best_step = step; best_snapshot = snapshot_trainable(runtime.model, best_snapshot)
                    event('validation_complete', validation=value, selected=selected)
                else:
                    event('validation_incomplete')
        event('training_stopped', reason='signal' if stopping else 'step_limit' if step >= args.max_steps else 'time_budget')
        if last_validated_step != step and not stopping:
            value = validate(runtime, valset, deadline - 150, lambda: stopping)
            if value is not None:
                last = value; last_validated_step = step; selected = value['flow_loss'] < best['flow_loss']
                if selected:
                    best = value; best_step = step; best_snapshot = snapshot_trainable(runtime.model, best_snapshot)
                event('validation_complete', validation=value, selected=selected)
        diagnostic = validate(runtime, train_view, deadline - 100, lambda: stopping)
        event('task_TRAIN_diagnostic', metrics=diagnostic, held_out_validation=False, used_for_checkpoint_selection=False)
        if step == 0:
            raise RuntimeError('No actual optimizer update')
        for name, value in runtime.model.named_buffers():
            if name in buffers and not torch.equal(value.detach().cpu(), buffers[name]):
                raise ValueError('Unsaved persistent buffer changed: ' + name)
        event('fixed_buffers_verified', persistent_buffer_count=len(buffers), elements=sum(v.numel() for v in buffers.values()))
        selected_path = Path(args.initial_delta).absolute()
        if best_snapshot is not None:
            selected_path = args.output / 'selected'
            save_delta(selected_path, best_snapshot, {**identity, 'step': best_step, 'validation': best,
                       'selected_by': 'complete original10 validation flow_loss only'})
        if best_step == step and best_snapshot is not None:
            final_path = selected_path
        else:
            final_path = args.output / 'final'
            best_snapshot = snapshot_trainable(runtime.model, best_snapshot)
            save_delta(final_path, best_snapshot, {**identity, 'step': step,
                       'validation': last if last_validated_step == step else None,
                       'selected_by': 'Actual final optimizer boundary, prospectively designated for physical development; not selected by held-out task performance'})
        result = {'schema': 'lingbot_vla2_g1_training_task_round04_v1', 'status': 'completed',
                  'optimizer_steps': step, 'selected_step': best_step, 'selected_delta': str(selected_path),
                  'physical_candidate_delta': str(final_path), 'physical_candidate_step': step,
                  'final_delta': str(final_path), 'final_weights_saved': True,
                  'baseline_validation': baseline, 'selected_validation': best, 'last_validation': last,
                  'final_validation_complete': last_validated_step == step, 'stop_requested': stopping,
                  'task_TRAIN_diagnostic': diagnostic, 'task_TRAIN_diagnostic_is_held_out': False,
                  'training_draw_counts': trainset.draw_counts,
                  'carry_family_draw_counts': carry.family_draw_counts, 'carry_phase_draw_counts': carry.phase_draw_counts,
                  'full_teacher_phase_draw_counts': complete.phase_draw_counts,
                  'model_weight_lineage_optimizer_updates': 1000 + step,
                  'elapsed_seconds': time.monotonic() - started, 'overall_robot_goal_completed': False, **identity}
        write_new(args.output / 'training_result.json', result)
        event('training_completed', selected_step=best_step, selected_delta=str(selected_path), final_delta=str(final_path))
        return result
    except BaseException as error:
        if not (args.output / 'training_result.json').exists():
            write_new(args.output / 'training_result.json', {'schema': 'lingbot_vla2_g1_training_task_round04_v1',
                      'status': 'failed', 'optimizer_steps': step, 'error': repr(error), **identity})
        event('training_failed', error=repr(error)); raise
    finally:
        progress.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('output', 'initial-delta', 'carry-prefix-manifest', 'new-carry-manifest', 'full-teacher-manifest'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--norm-stats', type=Path, default=OWN / 'train_meanstd.json')
    parser.add_argument('--supplementary-manifest', type=Path)
    parser.add_argument('--hours', type=float, default=.82)
    parser.add_argument('--max-steps', type=int, default=400)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
