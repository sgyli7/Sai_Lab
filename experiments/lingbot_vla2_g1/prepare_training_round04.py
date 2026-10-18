"""CPU-only R4 admission after an actually successful physical TRAIN teacher.

This command cannot run until collector cleanup, independent full-success audit,
and every synchronized teacher RGB/state/action admission have completed.
"""
import datetime
import json
from pathlib import Path
import shutil

from experiments.vla_resources import gpu_queue
from .budget_continuation import admit_allocation
from .data import OWN, ROOT, SHARED, SPLIT_SHA, MeanStd, sha
from .make_plan import experiment_sources, write
from .scripted_success_data import FullScriptedSuccess


def main():
    output = OWN / 'training_task_round04'
    preparation = OWN / 'round04_CPU_preparation'
    job_path = OWN / 'training_task_round04_job.json'
    snapshot = OWN / 'source_snapshots/training_task_round04'
    if any(p.exists() for p in (output, preparation, job_path, snapshot)):
        raise FileExistsError('Preserve every previous model and preparation')
    allocation = admit_allocation(OWN / 'gpu_budget.json')
    ledger = json.loads((OWN / 'gpu_budget.json').read_text())
    used_training = sum(row['elapsed_seconds'] for row in ledger['runs'] if row['stage'] == 'training')
    if allocation['total_limit_seconds'] - allocation['used_total_seconds'] < 3000 or allocation['stage_limits_seconds']['training'] - used_training < 3000:
        raise ValueError('Full finite3000second training turn must fit actual remaining budget')
    teacher_manifest = OWN / 'prepared/privileged_placement_teacher01/manifest.json'
    teacher = json.loads(teacher_manifest.read_text())
    complete = FullScriptedSuccess(teacher_manifest, MeanStd(OWN / 'train_meanstd.json'))
    if teacher['source_split_sha256'] != SPLIT_SHA or teacher['physical_outcome']['success'] is not True:
        raise ValueError('Actual complete physical TRAIN success required')
    gate_path = OWN / 'round04_CPU_checkpoint_gate.json'
    gate = json.loads(gate_path.read_text())
    if gate.get('passed') is not True or gate.get('tests_run') != 3:
        raise ValueError('Real CPU final retention/save/reload regression checks required')
    for path, expected in gate['source_files_sha256'].items():
        if sha(path) != expected:
            raise ValueError('R4 source changed after CPU regression gate')
    storage_path = OWN / 'storage_continuation_R4_v1.json'
    storage = json.loads(storage_path.read_text())
    if storage.get('prospective_remaining_growth_commitment_bytes') != 24 * 1024 ** 3 or storage.get('GPU_budget_total_seconds_unchanged') != 43200:
        raise ValueError('Prospectively communicated bounded storage commitment required')
    initial = OWN / 'training_round02/selected'
    metadata = json.loads((initial / 'metadata.json').read_text())
    if metadata['step'] != 200 or metadata['weights_sha256'] != sha(initial / 'trainable.safetensors'):
        raise ValueError('Actual R2 selected initialization required')
    prior = json.loads((OWN / 'training_round03_dual_job.json').read_text())
    payloads = prior['immutable_payloads']
    frozen = {**prior['frozen_files'], **teacher['extra_frozen_files'], **gate['source_files_sha256']}
    starts = [Path(__file__), ROOT / 'experiments/lingbot_vla2_g1/training_task_round04.py',
              ROOT / 'experiments/lingbot_vla2_g1/budget_continuation.py', ROOT / 'experiments/lingbot_vla2_g1/job.py']
    for path in experiment_sources(starts):
        frozen[str(path)] = sha(path)
    for path in (teacher_manifest, gate_path, storage_path, initial / 'metadata.json',
                 OWN / 'training_round03_dual/training_result.json',
                 OWN / 'round03_CPU_preparation/Root_actual_negative_trial_closure.json'):
        frozen[str(path)] = sha(path)
    for name, expected in frozen.items():
        if sha(name) != expected:
            raise ValueError('Frozen training input changed: ' + name)
    backup = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z')
    combined = sum(p.stat().st_size for base in (OWN, backup) for p in base.rglob('*') if p.is_file())
    source_bytes = sum(Path(n).stat().st_size for n in frozen if Path(n).suffix == '.py')
    growth = 4 * (metadata['weights_bytes'] + 1024 ** 2) + source_bytes + 2 * 1024 ** 3
    if combined + growth > storage['combined_cap_bytes'] or growth > storage['prospective_remaining_growth_commitment_bytes'] or shutil.disk_usage(OWN).free - growth < 32 * 1024 ** 3:
        raise RuntimeError('Two candidate payloads/permanent copies and complete raw evidence must fit committed storage/floor')
    preparation.mkdir()
    write(preparation / 'before_ledger.json', ledger)
    write(preparation / 'actual_teacher_CPU_data_gate.json', {
          'schema': 'lingbot_full_scripted_TRAIN_actual_CPU_data_gate_v1', 'status': 'passed',
          'manifest_sha256': sha(teacher_manifest), 'frames': len(complete.states),
          'every_raw_PNG_and_state_action_alignment_checked': True,
          'physical_teacher_success': True, 'teacher_is_LingBot_success': False,
          'GPU_initialized': False})
    decision = {'schema': 'lingbot_R4_prospective_root_training_decision_v1',
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'max_new_updates': 400, 'initial_weight_lineage_updates': 1000,
                'historical_R3_updates324_not_in_this_model_initialization': True,
                'learning_rate': 2e-5, 'source_probabilities': {'original80TRAIN': .2, 'native201': .1, 'native2026101702_prefix': .1, 'successful_scripted_TRAIN': .6},
                'full_teacher_phase_probabilities': {'approach_grasp': .45, 'transport': .2, 'placement': .35},
                'VLM_frozen': True, 'original80_10_10_split_and80TRAIN_norm_unchanged': True,
                'source_selection': 'Complete original10 validation flow_loss only; preserve original-best lineage independently.',
                'physical_candidate': 'Actual last complete optimizer update is always retained and is the prospectively designated development candidate, even if original short-task validation regresses.',
                'task_TRAIN_loss_is_held_out': False, 'formal_test_training_or_selection': False,
                'checkpoint_storage': 'At most two new full959BF16 trainable payloads; full final retained even on cooperative early stop; no optimizer payload.',
                'finite_GPU_seconds': 3000, 'combined_bytes_before': combined,
                'prospective_growth_bound_bytes': growth, 'combined_cap_bytes': storage['combined_cap_bytes'],
                'minimum_global_free_bytes': 32 * 1024 ** 3, 'minimum_host_available_GiB': 16,
                'actual_robot_goal_completed': False}
    write(preparation / 'decision.json', decision)
    for path in (preparation / 'before_ledger.json', preparation / 'actual_teacher_CPU_data_gate.json', preparation / 'decision.json'):
        frozen[str(path)] = sha(path)
    py = str(OWN / 'venv/bin/python')
    argv = [py, '-m', 'experiments.lingbot_vla2_g1.training_task_round04', '--output', str(output),
            '--initial-delta', str(initial), '--carry-prefix-manifest', str(SHARED / 'prepared/carry_prefix_native201/manifest.json'),
            '--new-carry-manifest', str(OWN / 'prepared/fresh_native2026101702_carry_prefix/manifest.json'),
            '--full-teacher-manifest', str(teacher_manifest), '--norm-stats', str(OWN / 'train_meanstd.json'),
            '--hours', '.82', '--max-steps', '400']
    result_path = output / 'training_result.json'; schema = 'lingbot_vla2_g1_training_task_round04_v1'
    ticket = gpu_queue.request(ROOT / '.scratch/vla_gpu_coordination', '01a145c6-0b86-7623-a869-086b33292199', 'training', 3000)
    try:
        supervisor = [py, '-m', 'experiments.lingbot_vla2_g1.budget_continuation', '--ledger', str(OWN / 'gpu_budget.json'),
                      '--stage', 'training', '--max-seconds', '3000', '--memory-high-gib', '80', '--memory-max-gib', '90',
                      '--host-reserve-gib', '16', '--result-file', str(result_path), '--result-schema', schema,
                      '--', py, '-m', 'experiments.lingbot_vla2_g1.job', '--job', str(job_path)]
        write(job_path, {'schema': 'lingbot_vla2_verified_job_v1', 'stage': 'training', 'max_seconds': 3000,
              'fifo_ticket': ticket['ticket'], 'result_file': str(result_path), 'result_schema': schema,
              'workload_argv': argv, 'supervisor_argv': supervisor, 'frozen_files': frozen,
              'immutable_payloads': payloads, 'budget_allocation': allocation, 'root_training_decision': decision})
        copied = {}
        for name, expected in frozen.items():
            if Path(name).suffix == '.py' or name in (str(teacher_manifest), str(preparation / 'decision.json')):
                target = snapshot / 'files' / name.lstrip('/')
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, target)
                if sha(target) != expected:
                    raise ValueError('Actual training snapshot changed during copy')
                copied[str(target)] = expected
        write(snapshot / 'manifest.json', {'schema': 'lingbot_training_source_snapshot_v2', 'status': 'completed',
              'job_sha256': sha(job_path), 'original_hashes': frozen, 'copied_files_sha256': copied,
              'actual_copied_bytes': sum(Path(n).stat().st_size for n in copied),
              'model_assets_and_prior_raw_evidence_copied': False})
        write(preparation / 'launch.json', {'coordinator_unit': 'lingbot-train-task-round04-' + ticket['ticket'] + '.service',
              'job': str(job_path), 'job_sha256': sha(job_path), 'FIFO_ticket': ticket['ticket'],
              'command': [py, '-m', 'experiments.vla_resources.gpu_queue', 'run', '--ticket', ticket['ticket'],
                          '--ledger', str(OWN / 'gpu_budget.json'), '--wait-seconds', '7200', '--', *supervisor]})
        print(json.dumps({'FIFO_ticket': ticket['ticket'], 'job': str(job_path), 'job_sha256': sha(job_path),
                          'frozen_files': len(frozen), 'copied_files': len(copied)}))
    except BaseException:
        gpu_queue.cancel(ROOT / '.scratch/vla_gpu_coordination', ticket['ticket'], '01a145c6-0b86-7623-a869-086b33292199')
        raise


if __name__ == '__main__':
    main()
