"""Verify separate original-validation best and prospectively retained final."""
import json
from pathlib import Path

from safetensors import safe_open

from experiments.vla_resources.gpu_queue import cleanup_confirmed
from .data import OWN, ROOT, sha
from .make_plan import artifact
from .round03_selection import select_from_progress


def roles_from_progress(result, events):
    if result.get('schema') != 'lingbot_vla2_g1_training_task_round04_v1':
        raise ValueError('Actual R4 training required')
    # Reuse the already tested strict original-validation verifier. Its schema
    # adapter is temporary CPU data; original receipts are never rewritten.
    source_best = select_from_progress({**result, 'schema': 'lingbot_vla2_g1_training_carry_round03_dual_v1'}, events)
    if (result.get('final_weights_saved') is not True
            or result.get('physical_candidate_step') != result['optimizer_steps']
            or result.get('physical_candidate_delta') != result.get('final_delta')
            or result.get('task_TRAIN_diagnostic_is_held_out') is not False
            or result.get('overall_robot_goal_completed') is not False):
        raise ValueError('The final actual optimizer boundary must be separately and truthfully retained')
    complete = [e for e in events if e['event'] == 'training_completed']
    if complete[0].get('final_delta') != result['final_delta']:
        raise ValueError('Actual completion did not retain the declared final payload')
    counts = result['training_draw_counts']
    draws = sum(counts.values())
    if not result['optimizer_steps'] * 8 <= draws <= result['optimizer_steps'] * 8 + 7:
        raise ValueError('Actual applied/partial accumulation draw counts differ')
    if (sum(result['carry_family_draw_counts'].values()) != counts['carry_prefixes']
            or sum(result['carry_phase_draw_counts'].values()) != counts['carry_prefixes']
            or sum(result['full_teacher_phase_draw_counts'].values()) != counts['full_scripted_TRAIN']):
        raise ValueError('Actual training family/phase counts differ')
    return source_best


def admit(training_result):
    path = Path(training_result).absolute()
    if path != OWN / 'training_task_round04/training_result.json':
        raise ValueError('This reviewed R4 output only')
    result = json.loads(path.read_text())
    progress = path.parent / 'progress.jsonl'
    events = [json.loads(line) for line in progress.read_text().splitlines()]
    source_best = roles_from_progress(result, events)
    delta = Path(result['physical_candidate_delta']).absolute()
    expected = path.parent / ('selected' if source_best['step'] == result['optimizer_steps'] else 'final')
    if delta != expected:
        raise ValueError('Actual final payload must belong to this run')
    metadata = json.loads((delta / 'metadata.json').read_text())
    for name, expected_value in (
        ('step', result['optimizer_steps']), ('norm_sha256', sha(OWN / 'train_meanstd.json')),
        ('contract_sha256', sha(OWN / 'contract.json')), ('base_receipt_sha256', sha(OWN / 'bf16_conversion_receipt.json')),
        ('full_teacher_manifest_sha256', result['full_teacher_manifest_sha256']),
    ):
        if metadata.get(name) != expected_value:
            raise ValueError('Actual final checkpoint identity differs: ' + name)
    identity = artifact(delta / 'trainable.safetensors', metadata['weights_sha256'])
    if identity['bytes'] != metadata['weights_bytes'] or metadata['tensor_count'] != 959:
        raise ValueError('Full trainable payload coverage differs')
    initial = OWN / 'training_round02/selected/trainable.safetensors'
    if identity['sha256'] == result['initial_weights_sha256']:
        raise ValueError('Do not repeat physical tests of unchanged R2 weights')
    with safe_open(str(initial), framework='pt', device='cpu') as old, safe_open(str(delta / 'trainable.safetensors'), framework='pt', device='cpu') as new:
        if set(old.keys()) != set(new.keys()) or len(new.keys()) != 959:
            raise ValueError('All959 trainable tensors must restore')
        for name in new.keys():
            if old.get_slice(name).get_shape() != new.get_slice(name).get_shape() or new.get_slice(name).get_dtype() != 'BF16':
                raise ValueError('Actual final tensor shape/dtype differs')
    job_path = OWN / 'training_task_round04_job.json'
    job = json.loads(job_path.read_text())
    queue = json.loads((ROOT / '.scratch/vla_gpu_coordination/queue.json').read_text())
    ticket = next(t for t in queue['tickets'] if t['ticket'] == job['fifo_ticket'])
    ledger = json.loads((OWN / 'gpu_budget.json').read_text())
    row = ledger['runs'][ticket['ledger_runs_before']]
    if ticket['status'] != 'completed' or ticket.get('cleanup_confirmed') is not True or row['status'] != 'completed' or not cleanup_confirmed(row):
        raise ValueError('Actual owned R4 training must complete and release resources')
    snapshot_path = OWN / 'source_snapshots/training_task_round04/manifest.json'
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get('status') != 'completed' or snapshot.get('job_sha256') != sha(job_path):
        raise ValueError('Actual R4 source archive differs')
    for name, expected_hash in snapshot['copied_files_sha256'].items():
        if sha(name) != expected_hash:
            raise ValueError('Archived R4 source changed')
    frozen = dict(job['frozen_files'])
    for value in (path, progress, path.parent / 'recipe.json', delta / 'metadata.json', job_path, snapshot_path):
        frozen[str(value)] = sha(value)
    payloads = {p: i for p, i in job['immutable_payloads'].items() if Path(p).parent == OWN / 'base_bf16'}
    if len(payloads) != 6:
        raise ValueError('Exactly six unchanged base shards required')
    payloads[str(delta / 'trainable.safetensors')] = identity
    provenance = {'job': str(job_path), 'job_sha256': sha(job_path), 'FIFO_ticket': ticket,
                  'actual_closed_GPU_charge': row, 'snapshot_sha256': sha(snapshot_path),
                  'new_optimizer_updates': result['optimizer_steps'], 'initial_lineage_updates': 1000,
                  'deployed_total_updates': 1000 + result['optimizer_steps'],
                  'source_validation_best_step': result['selected_step'],
                  'physical_candidate_step': result['physical_candidate_step'],
                  'physical_candidate_is_source_validation_best': result['selected_step'] == result['physical_candidate_step'],
                  'physical_candidate_chosen_before_task_evaluation': True,
                  'full_physical_scripted_TRAIN_teacher_used': True, 'teacher_is_LingBot_success': False,
                  'formal_or_original_test_trajectories_used_for_training': False}
    return result, delta, frozen, payloads, provenance
