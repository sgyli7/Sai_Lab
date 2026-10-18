"""Freeze R8 inputs after actual R6 failure and CPU robustness/retention checks."""
import datetime
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from experiments.vla_resources import gpu_queue
from .budget_reallocated_v7 import admit_allocation
from .data import OWN, ROOT, SHARED, MeanStd, sha
from .make_plan import experiment_sources, write
from .round06_selection import admit as admit_R6
from .training_task_round08 import admission
from .robust_training_v5 import FullTransportBalanced
from .robust_training_v8 import FullRecoveryBalanced, StatePerturbation, PERMANENT_SOURCE_BEST


def main():
    output, prep = OWN / 'training_task_round08', OWN / 'round08_CPU_preparation'
    job_path, snapshot = OWN / 'training_task_round08_job.json', OWN / 'source_snapshots/training_task_round08'
    if any(path.exists() for path in (output, prep, job_path, snapshot, PERMANENT_SOURCE_BEST)):
        raise FileExistsError('Preserve all previous training and preparation')
    allocation = admit_allocation(OWN / 'gpu_budget.json')
    ledger = json.loads((OWN / 'gpu_budget.json').read_text())
    used = sum(row['elapsed_seconds'] for row in ledger['runs'] if row['stage'] == 'training')
    if (allocation['total_limit_seconds'] - allocation['used_total_seconds'] < 1475
            or allocation['stage_limits_seconds']['training'] - used < 1475):
        raise ValueError('Complete finite1475-second training turn must fit')
    _result, initial, frozen, payloads, R6_provenance = admit_R6(OWN / 'training_task_round06/training_result.json')
    gate_path = OWN / 'round08_CPU_gate.json'
    gate = json.loads(gate_path.read_text())
    if gate.get('passed') is not True or gate.get('tests_run') != 13 or gate.get('GPU_initialized') is not False:
        raise ValueError('Actual13 CPU weighting/robustness/retention regressions required')
    for path, digest in gate['source_files_sha256'].items():
        if sha(path) != digest:
            raise ValueError('R8 changed after CPU tests: ' + path)
    args = SimpleNamespace(initial_delta=initial, norm_stats=OWN / 'train_meanstd.json',
        carry_prefix_manifest=SHARED / 'prepared/carry_prefix_native201/manifest.json',
        new_carry_manifest=OWN / 'prepared/fresh_native2026101702_carry_prefix/manifest.json',
        full_teacher_manifest=OWN / 'prepared/privileged_placement_teacher01/manifest.json',
        recovery_teacher_manifest=OWN / 'prepared/privileged_recovery_teacher09/manifest.json',
        prior_recovery_teacher_manifest=OWN / 'prepared/privileged_recovery_teacher04/manifest.json',
        intermediate_recovery_teacher_manifest=OWN / 'prepared/privileged_recovery_teacher08/manifest.json', supplementary_manifest=None)
    role_gate_path = OWN / 'round08_candidate_roles_CPU_gate.json'
    role_gate = json.loads(role_gate_path.read_text())
    if role_gate.get('passed') is not True or role_gate.get('tests_run') != 5 or role_gate.get('GPU_initialized') is not False:
        raise ValueError('Five actual R8 candidate-role CPU regressions required')
    for name, digest in role_gate['source_files_sha256'].items():
        if sha(name) != digest:
            raise ValueError('R8 final-role code changed after its CPU gate')
    parent_closure_path = OWN / 'matched_parent_R6_against_R7_preparation/Root_actual_negative_closure.json'
    parent_closure = json.loads(parent_closure_path.read_text())
    if (parent_closure.get('status') != 'completed_valid_negative'
            or parent_closure.get('all7_checks_both_cases_passed') is not True
            or parent_closure.get('full_task_success') is not False):
        raise ValueError('Closed actual matched R6 parent comparison required')
    provenance = admission(args)
    teacher = FullTransportBalanced(args.full_teacher_manifest, MeanStd(args.norm_stats))
    prior_recovery = FullRecoveryBalanced(args.prior_recovery_teacher_manifest, MeanStd(args.norm_stats))
    intermediate_recovery = FullRecoveryBalanced(args.intermediate_recovery_teacher_manifest, MeanStd(args.norm_stats))
    recovery = FullRecoveryBalanced(args.recovery_teacher_manifest, MeanStd(args.norm_stats))
    for manifest in (args.prior_recovery_teacher_manifest, args.intermediate_recovery_teacher_manifest, args.recovery_teacher_manifest):
        recovery_data = json.loads(manifest.read_text())
        frozen.update(recovery_data['extra_frozen_files'])
        frozen[str(manifest)] = sha(manifest)
    noise = StatePerturbation(42)
    originals = teacher.states.copy()
    for tick in (0, 100, 150, 200, 250, 350, 1000, 1500, 2040, 2400, 2600, 2800):
        state = teacher.states[tick]
        transformed = noise.apply(state, 'full_scripted_TRAIN')
        if np.shares_memory(transformed, state) or not np.array_equal(transformed[14:28], state[14:28]) or transformed[31] != 0:
            raise ValueError('Real TRAIN perturbation must preserve original measurements')
    if not np.array_equal(originals, teacher.states):
        raise ValueError('Stored TRAIN states were changed')
    recovery_originals = recovery.states.copy()
    for tick in (0, 150, 1000, 1700, len(recovery.states)-1):
        state = recovery.states[tick]
        transformed = noise.apply(state, 'recovery_scripted_TRAIN')
        if np.shares_memory(transformed, state) or not np.array_equal(transformed[14:28], state[14:28]) or transformed[31] != 0:
            raise ValueError('Recovery TRAIN perturbation must preserve original measurements')
    if not np.array_equal(recovery_originals, recovery.states):
        raise ValueError('Stored recovery states were changed')
    backup = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z')
    combined = sum(p.stat().st_size for folder in (OWN, backup) for p in folder.rglob('*') if p.is_file())
    metadata = json.loads((initial / 'metadata.json').read_text())
    growth = 3 * (metadata['weights_bytes'] + 1024**2) + 3 * 1024**3
    if (combined + growth > 64 * 1024**3 or growth > 20 * 1024**3
            or shutil.disk_usage(OWN).free - growth < 32 * 1024**3):
        raise RuntimeError('At most three physical checkpoint payloads and3GiB evidence reserve must fit64GiB')
    prep.mkdir()
    write(prep / 'before_ledger.json', ledger)
    decision = {'schema': 'lingbot_R8_prospective_root_training_decision_v1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'initial_R6_weights_sha256': metadata['weights_sha256'], 'initial_lineage_updates': 2040,
        'max_new_updates': 200, 'lr': 2e-5, 'max_GPU_seconds': 1475,
        'source_probabilities': provenance['training_source_weights'],
        'full_teacher_phase_probabilities': provenance['full_teacher_phase_probabilities'],
        'recovery_teacher_phase_probabilities': provenance['recovery_teacher_phase_probabilities'],
        'TRAIN_measurement_augmentation': provenance['state_augmentation'],
        'TRAIN_loss_weighting': provenance['training_loss_weighting'],
        'original_validation_loss_weights_unchanged': True, 'rejected_R7_not_in_weight_lineage': True,
        'internal_hours': .40, 'optimization_reserve_seconds': 180, 'validation_reserve_seconds': 90, 'diagnostic_reserve_seconds': 45,
        'original_normalization_and_live_contract_unchanged': True,
        'former_nonformal_development_scene_reclassified_TRAIN': True, 'formal_data_for_training': False,
        'fresh_development_cohort_required': True,
        'source_best_reference': provenance['source_validation_best_reference_provenance'],
        'physical_candidate': 'Actual final R8 update; global original-validation best is a separate role.',
        'combined_bytes_before': combined, 'prospective_growth_bound_bytes': growth,
        'combined_cap_bytes': 64 * 1024**3, 'current_training_and_backup_growth_bound_bytes': growth,
        'formal20_storage_forecast_required_after_actual_checkpoint_roles_known': True,
        'distinct_new_source_best_written_directly_to_permanent_storage': str(PERMANENT_SOURCE_BEST),
        'actual_final_has_independent_scratch_and_permanent_copies': True,
        'same_FIFO_memory_and_acceptance': True, 'goal_completed': False}
    write(prep / 'decision.json', decision)
    write(prep / 'real_TRAIN_CPU_gate.json', {'status': 'passed', 'GPU_initialized': False,
          'all_teacher_images_and_targets_admitted': True, 'real_states_untouched': True,
          'probe_augmentation_counts': noise.counts, 'teacher_frames': len(teacher.states), 'recovery_frames': len(recovery.states),
          'all_recovery_frames': [len(prior_recovery.states), len(intermediate_recovery.states), len(recovery.states)],
          'teacher_is_LingBot_success': False})
    starts = [Path(__file__), ROOT / 'experiments/lingbot_vla2_g1/training_task_round08.py',
              ROOT / 'experiments/lingbot_vla2_g1/budget_reallocated_v7.py', ROOT / 'experiments/lingbot_vla2_g1/job.py']
    for path in experiment_sources(starts):
        frozen[str(path)] = sha(path)
    for path in (gate_path, role_gate_path, OWN / 'round08_development_cohort.json',
                 OWN / 'round08_prospective_recipe_CPU_plan.json', OWN / 'training_task_round07/training_result.json',
                 OWN / 'diagnostics/development_to_TRAIN_recovery_v9.json',
                 OWN / 'placement_recovery_teacher09/Root_actual_positive_TRAIN_closure.json',
                 OWN / 'round07_development_preparation/Root_actual_negative_closure.json',
                 OWN / 'R1_selected_permanent_working_alias_receipt.json', OWN / 'R2_selected_permanent_working_alias_receipt.json',
                 OWN / 'R7_permanent_working_alias_receipt.json',
                 OWN / 'gpu_budget_allocation_user_resume_v7.json', OWN / 'gpu_budget_before_reallocation_v7.json',
                 OWN / 'budget_completion_v7_CPU_preparation.json',
                 OWN / 'diagnostics/historical_teacher_attribution_correction_v3_v4.json',
                 OWN / 'matched_parent_R6_against_R7_preparation/Root_actual_negative_closure.json',
                 OWN / 'diagnostics/R7_all60_query_state_range_CPU_fact.json',
                 prep / 'before_ledger.json', prep / 'decision.json', prep / 'real_TRAIN_CPU_gate.json'):
        frozen[str(path)] = sha(path)
    for name, digest in frozen.items():
        if sha(name) != digest:
            raise ValueError('Frozen training dependency changed: ' + name)
    py = str(OWN / 'venv/bin/python')
    argv = [py, '-m', 'experiments.lingbot_vla2_g1.training_task_round08', '--output', str(output),
        '--initial-delta', str(initial), '--norm-stats', str(args.norm_stats),
        '--carry-prefix-manifest', str(args.carry_prefix_manifest), '--new-carry-manifest', str(args.new_carry_manifest),
        '--full-teacher-manifest', str(args.full_teacher_manifest), '--recovery-teacher-manifest', str(args.recovery_teacher_manifest),
        '--prior-recovery-teacher-manifest', str(args.prior_recovery_teacher_manifest),
        '--intermediate-recovery-teacher-manifest', str(args.intermediate_recovery_teacher_manifest), '--hours', '.40', '--max-steps', '200']
    result_path, schema = output / 'training_result.json', 'lingbot_vla2_g1_training_task_round08_v1'
    ticket = gpu_queue.request(ROOT / '.scratch/vla_gpu_coordination',
                               '01a145c6-0b86-7623-a869-086b33292199', 'training', 1475)
    try:
        supervisor = [py, '-m', 'experiments.lingbot_vla2_g1.budget_reallocated_v7',
            '--ledger', str(OWN / 'gpu_budget.json'), '--stage', 'training', '--max-seconds', '1475',
            '--memory-high-gib', '80', '--memory-max-gib', '90', '--host-reserve-gib', '16',
            '--result-file', str(result_path), '--result-schema', schema, '--',
            py, '-m', 'experiments.lingbot_vla2_g1.job', '--job', str(job_path)]
        write(job_path, {'schema': 'lingbot_vla2_verified_job_v1', 'stage': 'training', 'max_seconds': 1475,
            'fifo_ticket': ticket['ticket'], 'result_file': str(result_path), 'result_schema': schema,
            'workload_argv': argv, 'supervisor_argv': supervisor, 'frozen_files': frozen,
            'immutable_payloads': payloads, 'budget_allocation': allocation, 'root_training_decision': decision})
        copied = {}
        for name, digest in frozen.items():
            if Path(name).suffix != '.py':
                continue
            target = snapshot / 'files' / name.lstrip('/')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
            if sha(target) != digest:
                raise ValueError('Training snapshot changed while copied')
            copied[str(target)] = digest
        write(snapshot / 'manifest.json', {'schema': 'lingbot_training_source_snapshot_v2', 'status': 'completed',
            'job_sha256': sha(job_path), 'original_hashes': frozen, 'copied_files_sha256': copied,
            'actual_copied_bytes': sum(Path(p).stat().st_size for p in copied), 'large_inputs_copied': False})
        write(prep / 'launch.json', {'FIFO_ticket': ticket['ticket'], 'job': str(job_path),
            'job_sha256': sha(job_path), 'coordinator_unit': 'lingbot-train-task-round08-' + ticket['ticket'] + '.service',
            'command': [py, '-m', 'experiments.vla_resources.gpu_queue', 'run', '--ticket', ticket['ticket'],
                '--ledger', str(OWN / 'gpu_budget.json'), '--wait-seconds', '7200', '--', *supervisor]})
        print(json.dumps({'ticket': ticket['ticket'], 'job_sha256': sha(job_path),
                         'frozen_files': len(frozen), 'copied_sources': len(copied)}))
    except BaseException:
        gpu_queue.cancel(ROOT / '.scratch/vla_gpu_coordination', ticket['ticket'], '01a145c6-0b86-7623-a869-086b33292199')
        raise


if __name__ == '__main__':
    main()
