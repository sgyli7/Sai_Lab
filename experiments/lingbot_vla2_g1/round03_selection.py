"""Admit a real round03 checkpoint using complete original validation only."""
import json
import math
from pathlib import Path

from safetensors import safe_open

from experiments.vla_resources.gpu_queue import cleanup_confirmed
from .data import OWN, ROOT, SPLIT_SHA, sha
from .make_plan import artifact


def select_from_progress(result, events):
    if (result.get('schema') != 'lingbot_vla2_g1_training_carry_round03_dual_v1'
            or result.get('status') != 'completed' or result.get('test_opened') is not False
            or result.get('split_sha256') != SPLIT_SHA or result.get('optimizer_steps', 0) <= 0):
        raise ValueError('Completed actual round03 training required')
    count = result['optimizer_steps']
    if [e['step'] for e in events if e['event'] == 'optimizer_step'] != list(range(1, count + 1)):
        raise ValueError('Every real optimizer update must be present exactly once')
    if sum(e['event'] == 'training_completed' for e in events) != 1:
        raise ValueError('Unique completed training event required')
    candidates = [e for e in events if e['event'] == 'validation_complete']
    if not candidates or candidates[0]['step'] != 0 or candidates[0]['validation'] != result['baseline_validation']:
        raise ValueError('Actual restored initial checkpoint must be the baseline candidate')
    for event in candidates:
        value = event['validation']
        if (value.get('complete') is not True or value.get('frames') != 9240
                or value.get('chunks') != 190 or value.get('episodes') != 10
                or not math.isfinite(value['flow_loss']) or not 0 <= event['step'] <= count):
            raise ValueError('Only complete original validation candidates may select weights')
    selected = min(candidates, key=lambda e: (e['validation']['flow_loss'], e['step']))
    if selected['step'] != result['selected_step'] or selected['validation'] != result['selected_validation']:
        raise ValueError('Checkpoint differs from complete validation minimum')
    return selected


def admit(training_result):
    path = Path(training_result).absolute()
    if not path.is_relative_to(OWN):
        raise ValueError('LingBot-owned actual training required')
    result = json.loads(path.read_text())
    progress = path.parent / 'progress.jsonl'
    events = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    selected = select_from_progress(result, events)
    if selected['step'] == 0:
        raise ValueError('Selected weights are unchanged from R2; do not repeat their physical test')
    delta = Path(result['selected_delta']).absolute()
    if delta != path.parent / 'selected':
        raise ValueError('Selected new payload must belong to this training run')
    meta = json.loads((delta / 'metadata.json').read_text())
    for field, actual in [('step', selected['step']), ('validation', selected['validation']),
                          ('norm_sha256', sha(OWN / 'train_meanstd.json')),
                          ('contract_sha256', sha(OWN / 'contract.json')),
                          ('base_receipt_sha256', sha(OWN / 'bf16_conversion_receipt.json'))]:
        if meta.get(field) != actual:
            raise ValueError('Selected trained payload metadata differs: ' + field)
    identity = artifact(delta / 'trainable.safetensors', meta['weights_sha256'])
    if identity['bytes'] != meta['weights_bytes'] or meta['tensor_count'] != 959:
        raise ValueError('Selected payload coverage differs')
    reference = OWN / 'training_round02/selected/trainable.safetensors'
    with safe_open(str(reference), framework='pt', device='cpu') as old, safe_open(str(delta / 'trainable.safetensors'), framework='pt', device='cpu') as new:
        if set(old.keys()) != set(new.keys()) or len(new.keys()) != 959:
            raise ValueError('Same959 trainable parameter names required')
        for name in new.keys():
            if old.get_slice(name).get_shape() != new.get_slice(name).get_shape() or new.get_slice(name).get_dtype() != 'BF16':
                raise ValueError('Selected tensor shape/dtype differs: ' + name)
    job_path = OWN / 'training_round03_dual_job.json'
    job = json.loads(job_path.read_text())
    queue = json.loads((ROOT / '.scratch/vla_gpu_coordination/queue.json').read_text())
    ticket = next(t for t in queue['tickets'] if t['ticket'] == job['fifo_ticket'])
    ledger = json.loads((OWN / 'gpu_budget.json').read_text())
    row = ledger['runs'][ticket['ledger_runs_before']]
    if ticket['status'] != 'completed' or ticket.get('cleanup_confirmed') is not True or row['status'] != 'completed' or not cleanup_confirmed(row):
        raise ValueError('Actual owned training must complete and release its resources')
    snapshot_path = OWN / 'source_snapshots/training_round03_dual/manifest.json'
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get('status') != 'completed' or snapshot.get('job_sha256') != sha(job_path):
        raise ValueError('Actual training source snapshot differs')
    for name, digest in snapshot['copied_files_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Archived training source changed: ' + name)
    frozen = dict(job['frozen_files'])
    for p in (path, progress, path.parent / 'recipe.json', path.parent / 'carry_admission.json',
              delta / 'metadata.json', job_path, snapshot_path):
        frozen[str(p)] = sha(p)
    families = result.get('prefix_family_draw_counts', {})
    phases = result.get('prefix_phase_draw_counts', {})
    if sum(families.values()) != result['training_draw_counts']['carry_prefix'] or sum(phases.values()) != sum(families.values()):
        raise ValueError('Actual family/phase draw counts differ')
    payloads = {p: i for p, i in job['immutable_payloads'].items() if Path(p).parent == OWN / 'base_bf16'}
    if len(payloads) != 6:
        raise ValueError('Exactly six unchanged pretrained base shards required')
    payloads[str(delta / 'trainable.safetensors')] = identity
    provenance = {'job': str(job_path), 'job_sha256': sha(job_path), 'FIFO_ticket': ticket,
                  'actual_closed_GPU_charge': row, 'snapshot': str(snapshot_path), 'snapshot_sha256': sha(snapshot_path),
                  'new_optimizer_updates': result['optimizer_steps'], 'selected_new_step': result['selected_step'],
                  'initial_total_updates': 1000, 'deployed_total_updates': 1000 + result['selected_step'],
                  'new_source_task_success_teacher': False, 'original10_validation_only_selection': True}
    return result, delta, frozen, payloads, provenance
