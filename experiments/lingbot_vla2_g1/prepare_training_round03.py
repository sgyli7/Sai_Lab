"""Freeze one finite stronger LingBot adaptation turn after failed grasp probes."""
import datetime
import json
from pathlib import Path
import shutil

from experiments.vla_resources import gpu_queue
from .budget_continuation import admit_allocation
from .carry_data_v2 import admit as admit_new_prefix
from .data import OWN, ROOT, SHARED, SPLIT, SPLIT_SHA, sha
from .make_plan import experiment_sources, write

OWNER = '01a145c6-0b86-7623-a869-086b33292199'


def prepare():
    run = 'training_round03_dual'
    output, job_path = OWN / run, OWN / (run + '_job.json')
    preparation = OWN / 'round03_CPU_preparation'
    snapshot = OWN / 'source_snapshots' / run
    if any(p.exists() for p in (output, job_path, preparation, snapshot)):
        raise FileExistsError('Preserve every prior preparation and model')
    allocation = admit_allocation(OWN / 'gpu_budget.json')
    ledger = json.loads((OWN / 'gpu_budget.json').read_text())
    used_training = sum(row['elapsed_seconds'] for row in ledger['runs'] if row['stage'] == 'training')
    if allocation['total_limit_seconds'] - allocation['used_total_seconds'] < 3000 or allocation['stage_limits_seconds']['training'] - used_training < 3000:
        raise ValueError('Full reviewed finite3000second training turn must fit remaining allocation')
    review = json.loads((OWN / 'resumed_turn01/independent_CPU_review.json').read_text())
    if review.get('status') != 'passed_evidence_audits' or review.get('actual_robot_task_goal_completed') is not False:
        raise ValueError('Independently checked task failure must remain explicit')
    fresh_path = OWN / 'prepared/fresh_native2026101702_carry_prefix/manifest.json'
    fresh = admit_new_prefix(fresh_path)
    proof_path = OWN / 'resumed_turn01/fresh_prefix_CPU_admission.json'
    proof = json.loads(proof_path.read_text())
    if (proof.get('status') != 'passed' or proof.get('frames') != 1462
            or proof.get('manifest_sha256') != sha(fresh_path)
            or proof.get('full_task_success_teacher') is not False or sha(SPLIT) != SPLIT_SHA):
        raise ValueError('Fresh carry TRAIN prefix must pass actual full CPU alignment admission')
    gate = json.loads((OWN / 'round03_training_CPU_gate.json').read_text())
    if gate.get('passed') is not True or gate.get('tests_run') != 7:
        raise ValueError('Completed current-source CPU training/data gate required')
    for name, digest in gate['source_files_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Source changed after CPU training/data gate: ' + name)
    initial = OWN / 'training_round02/selected'
    metadata = json.loads((initial / 'metadata.json').read_text())
    if metadata.get('step') != 200 or metadata.get('norm_sha256') != sha(OWN / 'train_meanstd.json'):
        raise ValueError('Use the actual selected round02 initialization')
    old = json.loads((OWN / 'training_round02_job.json').read_text())
    frozen = {**old['frozen_files'], **fresh['extra_frozen_files']}
    starts = [Path(__file__), ROOT / 'experiments/lingbot_vla2_g1/training_carry_round03_dual.py',
              ROOT / 'experiments/lingbot_vla2_g1/budget_continuation.py',
              ROOT / 'experiments/lingbot_vla2_g1/job.py']
    for path in experiment_sources(starts):
        frozen[str(path)] = sha(path)
    for path in (fresh_path, proof_path, initial / 'metadata.json', OWN / 'training_round02/training_result.json',
                 OWN / 'gpu_budget_allocation_user_resume_v4.json', OWN / 'gpu_budget_before_user_resume_v4.json',
                 OWN / 'resumed_turn01/independent_CPU_review.json',
                 OWN / 'diagnostics/round03_dual_source_derivation.json',
                 OWN / 'round03_training_CPU_gate.json'):
        frozen[str(path)] = sha(path)
    for name, digest in frozen.items():
        if sha(name) != digest:
            raise ValueError('Frozen training evidence/source changed: ' + name)
    payloads = json.loads((OWN / 'resumed_turn01/preparation.json').read_text())['immutable_payloads']
    for name, identity in payloads.items():
        if Path(name).stat().st_size != identity['bytes']:
            raise ValueError('Previously verified immutable payload size changed')
    if payloads[str(initial / 'trainable.safetensors')]['sha256'] != metadata['weights_sha256']:
        raise ValueError('Selected initialization payload differs')
    backup = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z')
    combined = sum(p.stat().st_size for folder in (OWN, backup) for p in folder.rglob('*') if p.is_file())
    source_bytes = sum(Path(name).stat().st_size for name in frozen if Path(name).suffix == '.py')
    growth = 2 * (metadata['weights_bytes'] + 1024 ** 2) + source_bytes + 2 * 1024 ** 3
    if combined + growth > 48 * 1024 ** 3 or shutil.disk_usage(OWN).free - growth < 32 * 1024 ** 3:
        raise RuntimeError('Preserve combined48GiB cap and global32GiB training reserve')
    preparation.mkdir()
    write(preparation / 'before_ledger.json', ledger)
    frozen[str(preparation / 'before_ledger.json')] = sha(preparation / 'before_ledger.json')
    decision = {'schema': 'lingbot_round03_root_training_decision_v1',
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'actual_initial_total_updates': 1000, 'max_new_updates': 500,
                'learning_rate': 5e-5, 'optimizer': 'FP32-master AdamW', 'hours': .82,
                'source_weights': {'original80TRAIN': .6, 'native201_carry_prefix': .2,
                                   'fresh_native2026101702_carry_prefix': .2},
                'within_each_prefix': {'approach_and_contact_lift_plus2s': .5, 'carry_remainder': .5},
                'full_success_teacher': False, 'original_norm_and80_10_10_unchanged': True,
                'validation': 'Original10complete190chunks9240frames only; lowest flow_loss selects one retained payload.',
                'checkpoint_storage': 'One selected new payload at most; final metrics retained without a separate worse payload.',
                'physical_development_next': 'Require actual grasp, then complete5m carry and stable release; training loss does not complete the task.',
                'planned_inference_variant': 'original_train_exact_constant_point_mass_inverse_v1',
                'combined_owned_bytes_before': combined, 'prospective_growth_bound_bytes': growth,
                'combined_cap_bytes': 48 * 1024 ** 3, 'global_training_reserve_bytes': 32 * 1024 ** 3,
                'finite_GPU_seconds': 3000, 'host_reserve_GiB': 16,
                'causal_attribution_to_one_change': False}
    write(preparation / 'decision.json', decision)
    frozen[str(preparation / 'decision.json')] = sha(preparation / 'decision.json')
    py = str(OWN / 'venv/bin/python')
    argv = [py, '-m', 'experiments.lingbot_vla2_g1.training_carry_round03_dual', '--output', str(output),
            '--initial-delta', str(initial), '--carry-prefix-manifest', str(SHARED / 'prepared/carry_prefix_native201/manifest.json'),
            '--new-carry-manifest', str(fresh_path), '--norm-stats', str(OWN / 'train_meanstd.json'),
            '--hours', '.82', '--max-steps', '500']
    result = output / 'training_result.json'
    schema = 'lingbot_vla2_g1_training_carry_round03_dual_v1'
    ticket = gpu_queue.request(ROOT / '.scratch/vla_gpu_coordination', OWNER, 'training', 3000)
    try:
        supervisor = [py, '-m', 'experiments.lingbot_vla2_g1.budget_continuation', '--ledger', str(OWN / 'gpu_budget.json'),
                      '--stage', 'training', '--max-seconds', '3000', '--memory-high-gib', '80', '--memory-max-gib', '90',
                      '--host-reserve-gib', '16', '--result-file', str(result), '--result-schema', schema,
                      '--', py, '-m', 'experiments.lingbot_vla2_g1.job', '--job', str(job_path)]
        job = {'schema': 'lingbot_vla2_verified_job_v1', 'stage': 'training', 'max_seconds': 3000,
               'fifo_ticket': ticket['ticket'], 'result_file': str(result), 'result_schema': schema,
               'workload_argv': argv, 'supervisor_argv': supervisor, 'frozen_files': frozen,
               'immutable_payloads': payloads, 'budget_allocation': allocation, 'root_training_decision': decision}
        write(job_path, job)
        copied = {}
        for name, digest in frozen.items():
            source = Path(name)
            if source.suffix == '.py' or source in (fresh_path, proof_path, initial / 'metadata.json', preparation / 'decision.json'):
                target = snapshot / 'files' / name.lstrip('/')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if sha(target) != digest:
                    raise ValueError('Training source changed during actual snapshot copy')
                copied[str(target)] = digest
        write(snapshot / 'manifest.json', {'schema': 'lingbot_training_source_snapshot_v2', 'status': 'completed',
              'job_sha256': sha(job_path), 'original_hashes': frozen, 'copied_files_sha256': copied,
              'actual_copied_bytes': sum(Path(name).stat().st_size for name in copied),
              'model_assets_and_prior_raw_evidence_copied': False,
              'large_inputs': 'Remain in place under exact SHA roster, previous snapshots and existing permanent backup.'})
        write(preparation / 'launch.json', {'coordinator_unit': 'lingbot-train-round03-dual-' + ticket['ticket'] + '.service',
              'job': str(job_path), 'job_sha256': sha(job_path), 'FIFO_ticket': ticket['ticket'],
              'command': [py, '-m', 'experiments.vla_resources.gpu_queue', 'run', '--ticket', ticket['ticket'],
                          '--ledger', str(OWN / 'gpu_budget.json'), '--wait-seconds', '7200', '--', *supervisor]})
        print(json.dumps({'FIFO_ticket': ticket['ticket'], 'job': str(job_path), 'job_sha256': sha(job_path),
                          'files': len(frozen), 'payloads': len(payloads), 'source_snapshot_files': len(copied),
                          'source_snapshot_bytes': sum(Path(name).stat().st_size for name in copied)}), flush=True)
    except BaseException:
        gpu_queue.cancel(ROOT / '.scratch/vla_gpu_coordination', ticket['ticket'], OWNER)
        raise


if __name__ == '__main__':
    prepare()
