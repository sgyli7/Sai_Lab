"""R7 adaptation: physical recovery supervision after actual student long carry.

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
from .grasp_balanced_prefix import GraspBalancedPrefix
from .carry_data_v2 import BalancedNewCarryPrefix, EqualCarryFamilies
from .robust_training_v5 import (FullTransportBalanced, PHASE_PROBABILITIES, require_complete_source_validation)
from .robust_training_v7 import (FullRecoveryBalanced, EqualRecoveryFamilies, RecoveryTrainingMixture, SOURCE_PROBABILITIES,
    RECOVERY_PHASE_PROBABILITIES, AUGMENTATION, GlobalSourceBest, PERMANENT_SOURCE_BEST)


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
    if args.supplementary_manifest is not None:
        raise ValueError('Only explicitly admitted physical TRAIN demonstrations allowed')
    initial = Path(args.initial_delta).absolute()
    if initial != OWN / 'training_task_round06/final' or Path(args.norm_stats).absolute() != OWN / 'train_meanstd.json':
        raise ValueError('Actual R6 final and original normalization required')
    prior_path = OWN / 'training_task_round06/training_result.json'
    prior = json.loads(prior_path.read_text())
    metadata = json.loads((initial / 'metadata.json').read_text())
    expected_weights = 'f7403098fff697a034022bbbb00369f9f3cf065fd953bbcdc9fbf0e2dbd8b8c5'
    if (prior.get('status') != 'completed' or prior.get('optimizer_steps') != 240
            or prior.get('physical_candidate_delta') != str(initial)
            or metadata.get('step') != 240 or metadata.get('tensor_count') != 959
            or metadata.get('weights_sha256') != expected_weights
            or sha(initial / 'trainable.safetensors') != expected_weights
            or prior.get('final_validation_complete') is not True
            or metadata.get('validation') != prior.get('last_validation')
            or metadata.get('norm_sha256') != sha(args.norm_stats)
            or metadata.get('contract_sha256') != sha(OWN / 'contract.json')):
        raise ValueError('Reviewed actual R6 payload differs')
    source_best = Path(prior['selected_delta']).absolute()
    source_meta = json.loads((source_best / 'metadata.json').read_text())
    if (source_best != OWN / 'training_round02/selected' or source_meta.get('step') != 200
            or any(source_meta.get('validation', {}).get(k) != prior['selected_validation'].get(k)
                   for k in ('flow_loss', 'frames', 'chunks', 'episodes', 'complete', 'noise_seed'))
            or source_meta.get('weights_sha256') != prior['source_validation_best_reference_provenance']['weights_sha256']):
        raise ValueError('Inherited original-validation best differs')
    require_complete_source_validation(prior['selected_validation'])
    for flag in ('carry_prefix_manifest', 'new_carry_manifest', 'full_teacher_manifest'):
        path = Path(getattr(args, flag)).absolute()
        if str(path) != prior[flag] or sha(path) != prior[flag + '_sha256']:
            raise ValueError('Previously admitted TRAIN source differs: ' + flag)
    recovery = Path(args.recovery_teacher_manifest).absolute()
    if recovery != OWN / 'prepared/privileged_recovery_teacher08/manifest.json':
        raise ValueError('The prospectively declared recovery TRAIN source is required')
    data = json.loads(recovery.read_text())
    if (data.get('schema') != 'lingbot_privileged_physical_full_success_TRAIN_episode_v1'
            or data.get('task_success_teacher') is not True or data.get('train_only') is not True
            or data.get('physical_outcome', {}).get('success') is not True
            or data.get('source_autonomous_model_policy') is not False
            or data.get('counts_as_LingBot_success') is not False
            or data.get('source_case_id') != 'privileged_TRAIN_recovery08'):
        raise ValueError('Actual independently audited full recovery success required')
    prior_recovery = Path(args.prior_recovery_teacher_manifest).absolute()
    if str(prior_recovery) != prior['recovery_teacher_manifest'] or sha(prior_recovery) != prior['recovery_teacher_manifest_sha256']:
        raise ValueError('Retain independently admitted previous recovery TRAIN family')
    return {'initial_delta': str(initial), 'initial_weights_sha256': expected_weights,
            'initial_metadata_sha256': sha(initial / 'metadata.json'),
            'initial_training_result_sha256': sha(prior_path), 'initial_validation': prior['last_validation'],
            'initial_step': 240, 'initial_total_optimizer_updates': 2040,
            'source_best_reference_delta': str(source_best), 'source_best_reference_validation': prior['selected_validation'],
            'source_validation_best_reference_provenance': {'origin': 'inherited_R2', 'lineage_updates': 1000,
                'R7_updates': 0, 'metadata_sha256': sha(source_best / 'metadata.json'), 'weights_sha256': source_meta['weights_sha256']},
            **{key: value for flag in ('carry_prefix_manifest', 'new_carry_manifest', 'full_teacher_manifest')
               for key, value in ((flag, str(Path(getattr(args, flag)).absolute())),
                                   (flag + '_sha256', sha(getattr(args, flag))))},
            'recovery_teacher_manifest': str(recovery), 'recovery_teacher_manifest_sha256': sha(recovery),
            'prior_recovery_teacher_manifest': str(prior_recovery), 'prior_recovery_teacher_manifest_sha256': sha(prior_recovery),
            'task_success_teacher': True, 'teacher_is_LingBot_success': False,
            'teacher_autonomous_model_policy': False, 'optimizer_resume': False,
            'training_source_weights': SOURCE_PROBABILITIES,
            'full_teacher_phase_probabilities': PHASE_PROBABILITIES,
            'recovery_teacher_phase_probabilities': RECOVERY_PHASE_PROBABILITIES,
            'full_teacher_used_for_normalizer_fit': False, 'state_augmentation': AUGMENTATION,
            'original80train_norm_unchanged': True,
            'former_nonformal_development_scene_explicitly_reclassified_TRAIN': True,
            'fresh_independent_development_cohort_required': True,
            'formal_or_original_test_trajectories_used_for_training': False,
            'physical_candidate_selection': 'Prospectively evaluate final actual R7 optimizer boundary; original-validation best retained separately. No formal trajectory informs training.'}


def train(args):
    if args.output.exists():
        raise FileExistsError('New immutable output required')
    if not 0 < args.hours <= .55 or not 0 < args.max_steps <= 240:
        raise ValueError('Finite240update R7 recovery recipe required')
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
    best_snapshot = best = last = baseline = source_best = None
    identity = {'base_receipt_sha256': sha(OWN / 'bf16_conversion_receipt.json'),
                'norm_path': str(args.norm_stats.absolute()), 'norm_sha256': sha(args.norm_stats),
                'contract_sha256': sha(OWN / 'contract.json'), 'split_sha256': SPLIT_SHA,
                'runtime_overlay_receipt_sha256': sha(OWN / 'runtime_overlay_receipt.json'),
                'new_source_best_permanent_path': str(PERMANENT_SOURCE_BEST),
                'prompt': PROMPT, 'test_opened': False, **provenance}
    write_new(args.output / 'recipe.json', {'hours': args.hours, 'max_steps': args.max_steps,
              'microbatch': 1, 'accumulation': 8, 'lr': 2e-5, 'optimizer': 'FP32-master AdamW',
              'validation_every': 100, 'source_selection_loss': 'complete original10 validation flow_loss',
              'diagnostic_TRAIN_view_is_validation': False, 'state_augmentation': AUGMENTATION, 'freeze_vlm': True,
              'auxiliary_teachers': False, 'native_queries_preserved': True,
              'checkpoint_strategy': 'Save actual final and its independent permanent backup. A distinct new original-validation best is written directly to permanent storage once. At most three physical payloads including backup; no optimizer checkpoint.',
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
        complete = FullTransportBalanced(args.full_teacher_manifest, normalizer)
        recovery = EqualRecoveryFamilies(FullRecoveryBalanced(args.prior_recovery_teacher_manifest, normalizer),
                                         FullRecoveryBalanced(args.recovery_teacher_manifest, normalizer))
        trainset = RecoveryTrainingMixture(original, carry, complete, recovery)
        train_view = DiagnosticTRAINView(complete)
        recovery_view = DiagnosticTRAINView(recovery.sources[1])
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
        require_complete_source_validation(baseline)
        if abs(baseline['flow_loss'] - provenance['initial_validation']['flow_loss']) > 1e-7:
            raise ValueError('Restored R6 initial validation differs before any R7 update')
        source_best = GlobalSourceBest(Path(provenance['source_best_reference_delta']),
                                       provenance['source_best_reference_validation'], PERMANENT_SOURCE_BEST)
        best = source_best.validation
        last = baseline
        event('validation_complete', validation=baseline, selected=False, initialized_from_R6=True, inherited_source_best=best)
        diagnostic = validate(runtime, train_view, deadline - 300, lambda: stopping)
        event('task_TRAIN_diagnostic', metrics=diagnostic, held_out_validation=False, used_for_checkpoint_selection=False)
        recovery_diagnostic = validate(runtime, recovery_view, deadline - 100, lambda: stopping)
        event('recovery_TRAIN_diagnostic', metrics=recovery_diagnostic, held_out_validation=False, used_for_checkpoint_selection=False)
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
                    last = value; last_validated_step = step; selected = source_best.consider(step, value)
                    if selected:
                        best = value; best_step = step; best_snapshot = snapshot_trainable(runtime.model, best_snapshot)
                    event('validation_complete', validation=value, selected=selected)
                else:
                    event('validation_incomplete')
        event('training_stopped', reason='signal' if stopping else 'step_limit' if step >= args.max_steps else 'time_budget')
        if last_validated_step != step and not stopping:
            value = validate(runtime, valset, deadline - 150, lambda: stopping)
            if value is not None:
                last = value; last_validated_step = step; selected = source_best.consider(step, value)
                if selected:
                    best = value; best_step = step; best_snapshot = snapshot_trainable(runtime.model, best_snapshot)
                event('validation_complete', validation=value, selected=selected)
        diagnostic = validate(runtime, train_view, deadline - 100, lambda: stopping)
        event('task_TRAIN_diagnostic', metrics=diagnostic, held_out_validation=False, used_for_checkpoint_selection=False)
        recovery_diagnostic = validate(runtime, recovery_view, deadline - 100, lambda: stopping)
        event('recovery_TRAIN_diagnostic', metrics=recovery_diagnostic, held_out_validation=False, used_for_checkpoint_selection=False)
        if step == 0:
            raise RuntimeError('No actual optimizer update')
        for name, value in runtime.model.named_buffers():
            if name in buffers and not torch.equal(value.detach().cpu(), buffers[name]):
                raise ValueError('Unsaved persistent buffer changed: ' + name)
        event('fixed_buffers_verified', persistent_buffer_count=len(buffers), elements=sum(v.numel() for v in buffers.values()))
        selected_path, final_path = source_best.paths(args.output, step)
        if source_best.step != best_step:
            raise ValueError('Source-best decision and retained snapshot differ')
        if best_snapshot is not None:
            selected_path.parent.mkdir(parents=True, exist_ok=True)
            save_delta(selected_path, best_snapshot, {**identity, 'step': best_step, 'validation': best,
                       'model_weight_lineage_optimizer_updates': 2040 + best_step,
                       'selected_by': 'Complete original10 validation flow_loss strictly below inherited R2 best'})
        if final_path != selected_path:
            best_snapshot = snapshot_trainable(runtime.model, best_snapshot)
            save_delta(final_path, best_snapshot, {**identity, 'step': step,
                       'model_weight_lineage_optimizer_updates': 2040 + step,
                       'validation': last if last_validated_step == step else None,
                       'selected_by': 'Actual final optimizer boundary, prospectively designated for physical development; not selected by held-out task performance'})
        elif best_snapshot is None:
            raise ValueError('Actual final must be materialized independently of inherited R2')
        if (sum(carry.family_draw_counts.values()) != trainset.draw_counts['carry_prefixes']
                or sum(carry.phase_draw_counts.values()) != trainset.draw_counts['carry_prefixes']
                or sum(complete.phase_draw_counts.values()) != trainset.draw_counts['full_scripted_TRAIN']
                or sum(recovery.phase_draw_counts.values()) != trainset.draw_counts['recovery_scripted_TRAIN']
                or sum(recovery.family_draw_counts.values()) != trainset.draw_counts['recovery_scripted_TRAIN']
                or any(trainset.perturbation.counts[name]['draws'] != count
                       for name, count in trainset.draw_counts.items())):
            raise ValueError('Actual sampling and perturbation counts do not reconcile')
        result = {'schema': 'lingbot_vla2_g1_training_task_round07_v1', 'status': 'completed',
                  'optimizer_steps': step, 'selected_step': best_step, 'selected_delta': str(selected_path),
                  'source_validation_best_origin': 'round07' if best_step else 'inherited_R2',
                  'physical_candidate_is_source_validation_best': best_step == step,
                  'physical_candidate_delta': str(final_path), 'physical_candidate_step': step,
                  'final_delta': str(final_path), 'final_weights_saved': True,
                  'baseline_validation': baseline, 'selected_validation': best, 'last_validation': last,
                  'final_validation_complete': last_validated_step == step, 'stop_requested': stopping,
                  'task_TRAIN_diagnostic': diagnostic, 'task_TRAIN_diagnostic_is_held_out': False,
                  'training_draw_counts': trainset.draw_counts,
                  'state_augmentation_counts': trainset.perturbation.counts,
                  'carry_family_draw_counts': carry.family_draw_counts, 'carry_phase_draw_counts': carry.phase_draw_counts,
                  'full_teacher_phase_draw_counts': complete.phase_draw_counts,
                  'recovery_teacher_phase_draw_counts': recovery.phase_draw_counts,
                  'recovery_family_draw_counts': recovery.family_draw_counts,
                  'recovery_phase_boundaries': [source.boundaries if hasattr(source, 'boundaries') else {'synthetic': True} for source in recovery.sources],
                  'recovery_TRAIN_diagnostic': recovery_diagnostic,
                  'model_weight_lineage_optimizer_updates': 2040 + step,
                  'elapsed_seconds': time.monotonic() - started, 'overall_robot_goal_completed': False, **identity}
        write_new(args.output / 'training_result.json', result)
        event('training_completed', selected_step=best_step, selected_delta=str(selected_path), final_delta=str(final_path))
        return result
    except BaseException as error:
        if not (args.output / 'training_result.json').exists():
            write_new(args.output / 'training_result.json', {'schema': 'lingbot_vla2_g1_training_task_round07_v1',
                      'status': 'failed', 'optimizer_steps': step, 'error': repr(error), **identity})
        event('training_failed', error=repr(error)); raise
    finally:
        progress.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('output', 'initial-delta', 'carry-prefix-manifest', 'new-carry-manifest', 'full-teacher-manifest', 'recovery-teacher-manifest', 'prior-recovery-teacher-manifest'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--norm-stats', type=Path, default=OWN / 'train_meanstd.json')
    parser.add_argument('--supplementary-manifest', type=Path)
    parser.add_argument('--hours', type=float, default=.55)
    parser.add_argument('--max-steps', type=int, default=240)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
