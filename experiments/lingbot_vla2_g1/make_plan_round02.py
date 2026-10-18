"""Freeze a carry-augmented checkpoint on the same, explicitly reused cohort."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil

from .acceptance_context import measurement_identity
from .benchmark_pose_speed import load_plan
from .data import OWN, ROOT, SPLIT, SPLIT_SHA, sha
from .make_plan import artifact, experiment_sources, training_provenance, write


def selected_checkpoint(path):
    path = Path(path).absolute()
    result = json.loads(path.read_text())
    if (result.get('schema') != 'lingbot_vla2_g1_training_carry_round02_v1'
            or result.get('status') != 'completed'
            or result.get('test_opened') is not False
            or result.get('optimizer_steps', 0) <= 0
            or result.get('split_sha256') != SPLIT_SHA or sha(SPLIT) != SPLIT_SHA):
        raise ValueError('Completed independently recorded second-round training required')
    if (result.get('carry_prefix_manifest_sha256') != '3dab70a1de3b387197fabbbf73cb83af1db0eebe4e6c146f6d7577238d00b9df'
            or result.get('training_source_weights') != {'original': .6, 'carry_prefix': .4}
            or result.get('optimizer_resume') is not False
            or result.get('original80train_norm_unchanged') is not True
            or result.get('task_success_teacher') is not False):
        raise ValueError('Predeclared train-only carry recipe differs')
    norm = Path(result['norm_path']).absolute()
    if norm != OWN / 'train_meanstd.json':
        raise ValueError('Original train-only LingBot normalization required')
    for field, identity in [('norm_sha256', norm),
                            ('base_receipt_sha256', OWN / 'bf16_conversion_receipt.json'),
                            ('contract_sha256', OWN / 'contract.json'),
                            ('runtime_overlay_receipt_sha256', OWN / 'runtime_overlay_receipt.json')]:
        if result.get(field) != sha(identity):
            raise ValueError('Training identity changed: ' + field)
    progress_path = path.parent / 'progress.jsonl'
    progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
    updates = [x['step'] for x in progress if x['event'] == 'optimizer_step']
    if updates != list(range(1, result['optimizer_steps'] + 1)):
        raise ValueError('Actual update sequence differs')
    split = json.loads(SPLIT.read_text())
    episodes = [x for x in split['files'] if x['episode_id'] in split['validation']]
    expected = {'frames': sum(x['length'] for x in episodes),
                'chunks': sum(math.ceil(x['length'] / 50) for x in episodes), 'episodes': 10}
    candidates = [x for x in progress if x['event'] == 'validation_complete']
    if not candidates or candidates[0]['step'] != 0:
        raise ValueError('Initial checkpoint must be a validation candidate')
    for entry in candidates:
        metric = entry['validation']
        if (metric.get('complete') is not True
                or any(metric.get(k) != v for k, v in expected.items())
                or not math.isfinite(metric['flow_loss'])
                or not 0 <= entry['step'] <= result['optimizer_steps']):
            raise ValueError('Original complete validation candidates required')
    selected = min(candidates, key=lambda x: (x['validation']['flow_loss'], x['step']))
    if selected['step'] != result['selected_step'] or selected['validation'] != result['selected_validation']:
        raise ValueError('Checkpoint differs from the predeclared validation minimum')
    delta = Path(result['selected_delta']).absolute()
    meta_path = delta / 'metadata.json'
    meta = json.loads(meta_path.read_text())
    if selected['step']:
        if not delta.is_relative_to(path.parent) or meta['step'] != selected['step'] or meta['validation'] != selected['validation']:
            raise ValueError('Selected new delta metadata differs')
    elif delta != Path(result['initial_delta']).absolute():
        raise ValueError('Step zero must use the recorded initial checkpoint')
    for key in ['norm_sha256', 'base_receipt_sha256', 'contract_sha256']:
        if meta[key] != result[key]:
            raise ValueError('Selected parameter identity differs: ' + key)
    payload = artifact(delta / 'trainable.safetensors', meta['weights_sha256'])
    if payload['bytes'] != meta['weights_bytes']:
        raise ValueError('Selected parameter byte length differs')
    from safetensors import safe_open
    with safe_open(str(OWN / 'training_round01/selected/trainable.safetensors'), framework='pt', device='cpu') as reference, safe_open(str(delta / 'trainable.safetensors'), framework='pt', device='cpu') as actual:
        if set(reference.keys()) != set(actual.keys()) or len(actual.keys()) != 959:
            raise ValueError('Selected trainable parameter coverage differs')
        for name in reference.keys():
            a, b = actual.get_slice(name), reference.get_slice(name)
            if a.get_shape() != b.get_shape() or a.get_dtype() != 'BF16':
                raise ValueError('Selected tensor shape/dtype differs: ' + name)
    frozen = {str(p): sha(p) for p in [path, progress_path, path.parent / 'recipe.json',
                                      path.parent / 'carry_admission.json',
                                      meta_path, norm, SPLIT]}
    return result, delta, norm, frozen, payload


def make_plan(previous_path, training_result, output, episodes_root):
    previous_path, output, episodes_root = map(lambda p: Path(p).absolute(),
                                              [previous_path, output, episodes_root])
    if output.exists() or episodes_root.exists() or not all(p.is_relative_to(OWN) for p in [output, episodes_root]):
        raise ValueError('Fresh owned second-round outputs required')
    previous = load_plan(previous_path)
    result, delta, norm, additions, payload = selected_checkpoint(training_result)
    provenance = training_provenance(training_result, additions)
    previous_configs = {j['config_path'] for j in previous['jobs']}
    frozen = {p: digest for p, digest in previous['frozen_files'].items() if p not in previous_configs}
    frozen.update(additions)
    frozen[str(previous_path)] = sha(previous_path)
    starts = [Path(__file__), ROOT / 'experiments/lingbot_vla2_g1/evaluate_round02.py',
              ROOT / 'experiments/lingbot_vla2_g1/evaluate_round02_driver.py',
              ROOT / 'experiments/lingbot_vla2_g1/benchmark_pose_speed_v2.py']
    for p in experiment_sources(starts):
        frozen[str(p)] = sha(p)
    allocation = OWN / 'gpu_budget_allocation_12h_v3.json'
    frozen[str(allocation)] = sha(allocation)
    for name in ['round02_cpu_gate_confirmation.json', 'round02_evaluator_deadline_guard.json',
                 'round02_queue_wait_timeout_cpu_gate.json', 'original_goal_wall_deadline.json',
                 'round02_resource_stop_rule.json', 'round02_actual_resource_stop_signal.json',
                 'round02_actual_GPU_admission.json']:
        p = OWN / name
        frozen[str(p)] = sha(p)
    payloads = {p: value for p, value in previous['immutable_payloads'].items()
                if Path(p).parent == OWN / 'base_bf16'}
    if len(payloads) != 6:
        raise ValueError('Exactly six original base shards required')
    payloads[str(delta / 'trainable.safetensors')] = payload
    for p, identity in payloads.items():
        if artifact(p, identity['sha256'])['bytes'] != identity['bytes']:
            raise ValueError('Immutable selected model changed')
    selection = {'schema': 'lingbot_vla2_round02_execution_selection_v1',
                 'training_result': str(Path(training_result).absolute()),
                 'training_provenance': provenance, 'selected_step': result['selected_step'],
                 'optimizer_steps': result['optimizer_steps'], 'selected_delta': str(delta),
                 'norm_path': str(norm), 'norm_sha256': sha(norm),
                 'selected_validation': result['selected_validation'],
                 'baseline_validation': result['baseline_validation'],
                 'selection_rule': 'Minimum complete original10 validation flow loss; earliest tie; initial round01 candidate included',
                 'acceptance_measurement': measurement_identity(),
                 'training_only_carry_augmentation': True, 'full_success_teacher': False,
                 'test_observations_opened': False, 'same_cohort_reused_after_baseline': True,
                 'limitation': 'Second recipe chosen after seeing round01 task outcomes; this cohort is a repeated benchmark, not a fresh blind generalization estimate.',
                 'immutable_payloads': payloads}
    original_selection = json.loads(Path(previous['selection_receipt']).read_text())
    for field in ['hardware', 'runtime', 'scenarios_sha256', 'offline_unused_assets']:
        selection[field] = original_selection[field]
    selection['shared_placement_qualification'] = previous['shared_placement_qualification']
    output.mkdir(parents=True)
    selection_path = output / 'selection_receipt.json'
    write(selection_path, selection)
    frozen[str(selection_path)] = sha(selection_path)
    jobs = []
    for old in previous['jobs']:
        config = json.loads(Path(old['config_path']).read_text())
        folder = episodes_root / old['case_id']
        command = list(config['policy_command'])
        command[command.index('--decoded-evidence') + 1] = str(folder / 'decoded.jsonl')
        command[command.index('--delta') + 1] = str(delta)
        config.update(output_dir=str(folder), policy_command=command, frozen_files=dict(frozen))
        config['HFmodel']['delta'] = str(delta)
        config_path = output / 'configs' / (old['case_id'] + '.json')
        write(config_path, config)
        jobs.append({**old, 'config_path': str(config_path), 'config_sha256': sha(config_path),
                     'result_path': str(folder / 'result.json')})
    frozen.update({j['config_path']: j['config_sha256'] for j in jobs})
    plan = copy.deepcopy(previous)
    plan.update(jobs=jobs, frozen_files=frozen, immutable_payloads=payloads,
                selection_receipt=str(selection_path), episode_root=str(episodes_root),
                round=2, training_recipe='round01_selected_plus_train_only_carry60_40',
                same_cohort_reused_after_baseline=True,
                previous_baseline_plan={'path': str(previous_path), 'sha256': sha(previous_path)},
                storage_admission={'schema': 'lingbot_storage_admission_v2',
                                   'reserve_bytes': 24 * 1024**3,
                                   'per_case_buffer_bytes': 1024**3,
                                   'own_cap_bytes': 36 * 1024**3})
    write(output / 'plan.json', plan)
    for name, digest in frozen.items():
        p = Path(name)
        if p.suffix not in ['.py', '.json', '.yaml', '.usda'] or p.stat().st_size > 20 * 1024**2:
            continue
        target = output / 'source_snapshot' / name.lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        if sha(target) != digest:
            raise ValueError('Source changed during snapshot')
    load_plan(output / 'plan.json')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['previous-plan', 'training-result', 'output', 'episodes-root']:
        parser.add_argument('--' + key, type=Path, required=True)
    a = parser.parse_args()
    plan = make_plan(a.previous_plan, a.training_result, a.output, a.episodes_root)
    print(json.dumps({'cases': len(plan['jobs']), 'round': 2,
                      'frozen_files': len(plan['frozen_files']),
                      'same_cohort_reused_after_baseline': True}))
