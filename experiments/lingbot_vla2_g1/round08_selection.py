"""Audit the actual R8 final independently of the inherited global source best."""
import json
from pathlib import Path

from safetensors import safe_open

from experiments.vla_resources.gpu_queue import cleanup_confirmed
from .data import OWN, ROOT, sha
from .make_plan import artifact
from .robust_training_v5 import PHASE_PROBABILITIES
from .robust_training_v8 import (GlobalSourceBest, PERMANENT_SOURCE_BEST,
    SOURCE_PROBABILITIES, RECOVERY_PHASE_PROBABILITIES, AUGMENTATION, RECOVERY_FAMILY_PROBABILITIES)
from .body_preserving_training import LOSS_WEIGHTING


def roles_from_progress(result, events):
    if (result.get('schema') != 'lingbot_vla2_g1_training_task_round08_v1'
            or result.get('status') != 'completed' or result.get('overall_robot_goal_completed') is not False
            or result.get('initial_total_optimizer_updates') != 2040
            or result.get('training_source_weights') != SOURCE_PROBABILITIES
            or result.get('full_teacher_phase_probabilities') != PHASE_PROBABILITIES
            or result.get('state_augmentation') != AUGMENTATION
            or result.get('training_loss_weighting') != LOSS_WEIGHTING
            or result.get('validation_loss_weighting') != 'original_unit_valid_mask'
            or result.get('rejected_R7_not_in_weight_lineage') is not True
            or result.get('recovery_teacher_phase_probabilities') != RECOVERY_PHASE_PROBABILITIES
            or result.get('former_nonformal_development_scene_explicitly_reclassified_TRAIN') is not True
            or result.get('fresh_independent_development_cohort_required') is not True):
        raise ValueError('Actual reviewed R8 recipe and truthful attribution required')
    steps = result['optimizer_steps']
    if type(steps) is not int or not 0 < steps <= 200:
        raise ValueError('Real positive bounded R8 optimizer updates required')
    updates = [event['step'] for event in events if event.get('event') == 'optimizer_step']
    started = [event for event in events if event.get('event') == 'training_started']
    if len(started) != 1 or started[0]['step'] != 0:
        raise ValueError('Unique actual training start required')
    if updates != list(range(1, steps + 1)) or result.get('model_weight_lineage_optimizer_updates') != 2040 + steps:
        raise ValueError('Contiguous actual updates and weight lineage required')
    validations = [event for event in events if event.get('event') == 'validation_complete']
    if not validations or validations[0]['step'] != 0 or validations[0]['validation'] != result['baseline_validation']:
        raise ValueError('Actual R6 baseline validation required')
    if validations[0].get('selected') is not False:
        raise ValueError('Worse R6 baseline cannot replace inherited R2 best')
    source_best = GlobalSourceBest(result['source_best_reference_delta'], result['source_best_reference_validation'],
                                   result['new_source_best_permanent_path'])
    for event in validations[1:]:
        if not 0 < event['step'] <= steps or event['step'] % 100 and event['step'] != steps:
            raise ValueError('Undeclared validation boundary')
        selected = source_best.consider(event['step'], event['validation'])
        if event.get('selected') is not selected:
            raise ValueError('Source-best flag differs from strict global flow selection')
    if result.get('final_validation_complete') is not True or validations[-1]['step'] != steps:
        raise ValueError('Complete original10 validation of the actual final update required')
    if (source_best.step != result['selected_step'] or source_best.validation != result['selected_validation']
            or result['last_validation'] != validations[-1]['validation']
            or result['final_validation_complete'] is not (validations[-1]['step'] == steps)
            or result.get('source_validation_best_origin') != ('round08' if source_best.step else 'inherited_R2')):
        raise ValueError('Source-validation best or final validation role differs')
    end = [e for e in events if e.get('event') == 'training_completed']
    stopped = [e for e in events if e.get('event') == 'training_stopped']
    if (len(end) != 1 or len(stopped) != 1 or end[0]['step'] != steps or stopped[0]['step'] != steps
            or end[0].get('selected_step') != source_best.step
            or end[0].get('selected_delta') != result['selected_delta']
            or end[0].get('final_delta') != result['final_delta']):
        raise ValueError('Actual completion and checkpoint roles differ')
    if (result.get('physical_candidate_step') != steps
            or result.get('final_weights_saved') is not True
            or result.get('final_delta') != result.get('physical_candidate_delta')
            or result.get('physical_candidate_is_source_validation_best') is not (source_best.step == steps)
            or result.get('task_TRAIN_diagnostic_is_held_out') is not False):
        raise ValueError('Actual final update must be retained separately')
    counts = result['training_draw_counts']
    if set(counts) != set(SOURCE_PROBABILITIES) or not steps * 8 <= sum(counts.values()) <= steps * 8 + 7:
        raise ValueError('Actual optimizer accumulation draws differ')
    if (sum(result['carry_family_draw_counts'].values()) != counts['carry_prefixes']
            or sum(result['carry_phase_draw_counts'].values()) != counts['carry_prefixes']
            or sum(result['full_teacher_phase_draw_counts'].values()) != counts['full_scripted_TRAIN']
            or sum(result['recovery_teacher_phase_draw_counts'].values()) != counts['recovery_scripted_TRAIN']
            or sum(result['recovery_family_draw_counts'].values()) != counts['recovery_scripted_TRAIN']):
        raise ValueError('Actual TRAIN family/phase draws differ')
    if (set(result['recovery_family_draw_counts']) != set(RECOVERY_FAMILY_PROBABILITIES)
            or any(type(v) is not int or v <= 0 for v in result['recovery_family_draw_counts'].values())):
        raise ValueError('All three admitted recovery families must actually be sampled')
    for family, count in counts.items():
        measured = result['state_augmentation_counts'][family]
        if (measured['draws'] != count or measured['eligible'] != (0 if family == 'original80TRAIN' else count)
                or not 0 <= measured['augmented'] <= measured['eligible']):
            raise ValueError('Actual TRAIN state-perturbation counts differ')
    return source_best


def admit(training_result):
    path = Path(training_result).absolute()
    if path != OWN / 'training_task_round08/training_result.json':
        raise ValueError('This reviewed R8 output only')
    result = json.loads(path.read_text())
    if result.get('synthetic') is True:
        raise ValueError('Synthetic fixtures cannot qualify a real candidate')
    if result.get('new_source_best_permanent_path') != str(PERMANENT_SOURCE_BEST):
        raise ValueError('Prospective distinct source-best permanent storage required')
    if (result.get('initial_weights_sha256') != 'f7403098fff697a034022bbbb00369f9f3cf065fd953bbcdc9fbf0e2dbd8b8c5'
            or Path(result['source_best_reference_delta']).absolute() != OWN / 'training_round02/selected'):
        raise ValueError('R6 initialization and inherited R2 source best required')
    progress = path.parent / 'progress.jsonl'
    source_best = roles_from_progress(result, [json.loads(line) for line in progress.read_text().splitlines()])
    selected, delta = source_best.paths(path.parent, result['optimizer_steps'])
    if str(selected) != result['selected_delta'] or str(delta) != result['physical_candidate_delta']:
        raise ValueError('Physical final or source-best path differs')
    metadata = json.loads((delta / 'metadata.json').read_text())
    for key, expected in (('step', result['optimizer_steps']), ('norm_sha256', sha(OWN / 'train_meanstd.json')),
            ('base_receipt_sha256', sha(OWN / 'bf16_conversion_receipt.json')),
            ('contract_sha256', sha(OWN / 'contract.json')), ('model_weight_lineage_optimizer_updates', 2040 + result['optimizer_steps']),
            ('initial_weights_sha256', 'f7403098fff697a034022bbbb00369f9f3cf065fd953bbcdc9fbf0e2dbd8b8c5')):
        if metadata.get(key) != expected:
            raise ValueError('Actual final metadata differs: ' + key)
    if (metadata.get('training_loss_weighting') != LOSS_WEIGHTING
            or metadata.get('validation_loss_weighting') != 'original_unit_valid_mask'
            or metadata.get('validation') != result['last_validation']
            or metadata.get('rejected_R7_not_in_weight_lineage') is not True):
        raise ValueError('Actual final loss recipe or complete final validation metadata differs')
    identity = artifact(delta / 'trainable.safetensors', metadata['weights_sha256'])
    if identity['bytes'] != metadata['weights_bytes'] or metadata.get('tensor_count') != 959 or identity['sha256'] == metadata['initial_weights_sha256']:
        raise ValueError('Genuinely new complete actual final payload required')
    with safe_open(str(OWN / 'training_task_round06/final/trainable.safetensors'), framework='pt', device='cpu') as old, \
            safe_open(str(delta / 'trainable.safetensors'), framework='pt', device='cpu') as new:
        if set(old.keys()) != set(new.keys()) or len(new.keys()) != 959:
            raise ValueError('All959 trainable tensors required')
        for key in new.keys():
            if old.get_slice(key).get_shape() != new.get_slice(key).get_shape() or new.get_slice(key).get_dtype() != 'BF16':
                raise ValueError('Actual trainable tensor shape/dtype differs')
    source_metadata = json.loads((selected / 'metadata.json').read_text())
    for key in ('flow_loss', 'frames', 'chunks', 'episodes', 'complete', 'noise_seed'):
        if source_metadata.get('validation', {}).get(key) != result['selected_validation'].get(key):
            raise ValueError('Retained source-validation best differs')
    if selected != delta:
        artifact(selected / 'trainable.safetensors', source_metadata['weights_sha256'])
    job_path = OWN / 'training_task_round08_job.json'
    job = json.loads(job_path.read_text())
    queue = json.loads((ROOT / '.scratch/vla_gpu_coordination/queue.json').read_text())
    ticket = next(t for t in queue['tickets'] if t['ticket'] == job['fifo_ticket'])
    row = json.loads((OWN / 'gpu_budget.json').read_text())['runs'][ticket['ledger_runs_before']]
    if ticket['status'] != 'completed' or ticket.get('cleanup_confirmed') is not True or row['status'] != 'completed' or not cleanup_confirmed(row):
        raise ValueError('Actual R8 training and owned cleanup must complete')
    snapshot_path = OWN / 'source_snapshots/training_task_round08/manifest.json'
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get('status') != 'completed' or snapshot['job_sha256'] != sha(job_path):
        raise ValueError('Actual R8 source snapshot required')
    for name, digest in snapshot['copied_files_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Archived training source changed: ' + name)
    frozen = dict(job['frozen_files'])
    for value in (path, progress, path.parent / 'recipe.json', delta / 'metadata.json', selected / 'metadata.json', job_path, snapshot_path):
        frozen[str(value)] = sha(value)
    payloads = {p: value for p, value in job['immutable_payloads'].items() if Path(p).parent == OWN / 'base_bf16'}
    if len(payloads) != 6:
        raise ValueError('Exactly six unchanged base shards required')
    payloads[str(delta / 'trainable.safetensors')] = identity
    provenance = {'job': str(job_path), 'job_sha256': sha(job_path), 'FIFO_ticket': ticket,
        'actual_closed_GPU_charge': row, 'snapshot_sha256': sha(snapshot_path),
        'new_optimizer_updates': result['optimizer_steps'], 'initial_lineage_updates': 2040,
        'deployed_total_updates': 2040 + result['optimizer_steps'],
        'source_validation_best_step': result['selected_step'], 'physical_candidate_step': result['physical_candidate_step'],
        'physical_candidate_is_source_validation_best': selected == delta,
        'physical_candidate_chosen_before_task_evaluation': True, 'teacher_is_LingBot_success': False,
        'formal_or_original_test_trajectories_used_for_training': False,
        'TRAIN_only_state_augmentation': result['state_augmentation_counts']}
    return result, delta, frozen, payloads, provenance
