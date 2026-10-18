"""Freeze validation-selected LingBot on the peer's unchanged twenty G1 cases."""
from __future__ import annotations
import argparse
import ast
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from experiments.pi05_g1_mobile.benchmark import read
from experiments.pi05_g1_mobile.simulator import validate_config
from experiments.pi05_g1_mobile.make_benchmark_plan import docker_identity, verify_offline_assets
from .contract import PROMPT
from .data import ROOT, OWN, SHARED, SPLIT, SPLIT_SHA, sha

SCENARIOS = SHARED / 'scenarios_frozen_v3.json'
SCENARIOS_SHA = '6c7885cedb68826ae431319c0d65f91e29d6749d350853732f2c7a559c050953'
TEMPLATE = SHARED / 'prepared_long5m_v4_native_seed201_dev04_60s.json'
TEMPLATE_SHA = 'd36224e983b7c193d7f026558ce500b30ab0842925aa4d1f8a6afbb8c7da23d3'


def write(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)


def artifact(path, expected=None):
    path = Path(path).absolute()
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError('Immutable payload SHA mismatch: ' + str(path))
    result = {'sha256': actual, 'bytes': path.stat().st_size}
    with path.open('rb') as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return result


def verify_selection(result_path):
    result_path = Path(result_path).absolute()
    if not result_path.is_relative_to(OWN):
        raise ValueError('LingBot-owned training required')
    result = read(result_path)
    if result.get('schema') != 'lingbot_vla2_g1_training_v1' or result.get('status') != 'completed' or result.get('test_opened') is not False or result.get('optimizer_steps', 0) <= 0:
        raise ValueError('Completed real fine-tuning, with test unopened, required')
    if result.get('split_sha256') != SPLIT_SHA or sha(SPLIT) != SPLIT_SHA or result.get('supplementary_manifest') is not None:
        raise ValueError('This first-round plan requires the unchanged source-only split')
    norm = Path(result['norm_path']).absolute()
    for field, path in [('norm_sha256', norm), ('base_receipt_sha256', OWN / 'bf16_conversion_receipt.json'), ('contract_sha256', OWN / 'contract.json'), ('runtime_overlay_receipt_sha256', OWN / 'runtime_overlay_receipt.json')]:
        if result.get(field) != sha(path):
            raise ValueError('Training identity changed: ' + field)
    split = read(SPLIT)
    episodes = [entry for entry in split['files'] if entry['episode_id'] in split['validation']]
    expected = {'frames': sum(entry['length'] for entry in episodes), 'chunks': sum(math.ceil(entry['length'] / 50) for entry in episodes), 'episodes': 10}
    progress_path = result_path.parent / 'progress.jsonl'
    progress = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
    updates = [entry['step'] for entry in progress if entry['event'] == 'optimizer_step']
    completed = [entry for entry in progress if entry['event'] == 'training_completed']
    stopped = [entry for entry in progress if entry['event'] == 'training_stopped']
    if updates != list(range(1, result['optimizer_steps'] + 1)) or len(completed) != 1 or len(stopped) != 1 or completed[0]['step'] != result['optimizer_steps'] or stopped[0]['step'] != result['optimizer_steps']:
        raise ValueError('Every real update and a completed update-boundary stop must be recorded')
    candidates = [entry for entry in progress if entry['event'] == 'validation_complete']
    if not candidates or candidates[0]['step'] != 0 or candidates[0]['validation'] != result['baseline_validation']:
        raise ValueError('Unchanged base must be the step0 validation candidate')
    for entry in candidates:
        metric = entry['validation']
        if not 0 <= entry['step'] <= result['optimizer_steps'] or metric.get('complete') is not True or any(metric.get(k) != v for k, v in expected.items()) or not math.isfinite(metric['flow_loss']):
            raise ValueError('Only complete original validation candidates may select weights')
    selected = min(candidates, key=lambda entry: (entry['validation']['flow_loss'], entry['step']))
    if selected['step'] != result['selected_step'] or selected['validation'] != result['selected_validation']:
        raise ValueError('Recorded selected weights differ from validation minimum')
    delta = Path(result['selected_delta']).absolute() if result['selected_delta'] else None
    frozen = {str(path): sha(path) for path in [result_path, progress_path, result_path.parent / 'recipe.json', norm, SPLIT, OWN / 'contract.json', OWN / 'bf16_conversion_receipt.json', OWN / 'runtime_overlay_receipt.json']}
    payloads = {}
    converted = read(OWN / 'bf16_conversion_receipt.json')
    if converted.get('status') != 'completed':
        raise ValueError('Verified pretrained conversion required')
    for path in sorted((OWN / 'base_bf16').glob('*.safetensors')):
        identity = artifact(path)
        if sum(identity['sha256'] in shard.values() for shard in converted['shards']) != 1:
            raise ValueError('Pretrained shard no longer matches conversion receipt')
        payloads[str(path)] = identity
    if len(payloads) != 6:
        raise ValueError('Exactly six pretrained shards required')
    if delta:
        if not delta.is_relative_to(result_path.parent) or selected['step'] == 0:
            raise ValueError('Recorded selected delta must be owned by this training run')
        meta_path = delta / 'metadata.json'
        meta = read(meta_path)
        if meta['step'] != selected['step'] or meta['validation'] != selected['validation'] or meta['norm_sha256'] != result['norm_sha256'] or meta['base_receipt_sha256'] != result['base_receipt_sha256'] or meta['contract_sha256'] != result['contract_sha256']:
            raise ValueError('Selected delta metadata identity differs')
        payloads[str(delta / 'trainable.safetensors')] = artifact(delta / 'trainable.safetensors', meta['weights_sha256'])
        if payloads[str(delta / 'trainable.safetensors')]['bytes'] != meta['weights_bytes']:
            raise ValueError('Selected delta bytes differ')
        # Shape/name inspection uses a meta model and safetensor headers only.
        # A malformed restore must be rejected before reserving a GPU turn.
        from safetensors import safe_open
        from .runtime import meta_model
        model, _ = meta_model()
        expected_parameters = {name: list(value.shape) for name, value in model.named_parameters() if value.requires_grad}
        with safe_open(str(delta / 'trainable.safetensors'), framework='pt', device='cpu') as tensors:
            if set(tensors.keys()) != set(expected_parameters):
                raise ValueError('Selected delta does not cover exactly the trainable parameters')
            for name, shape in expected_parameters.items():
                if tensors.get_slice(name).get_shape() != shape or tensors.get_slice(name).get_dtype() != 'BF16':
                    raise ValueError('Selected delta tensor shape/dtype differs: ' + name)
        frozen[str(meta_path)] = sha(meta_path)
    elif selected['step'] != 0:
        raise ValueError('Selected nonzero update has no delta')
    return result, delta, norm, frozen, payloads


def training_provenance(result_path, frozen):
    from experiments.vla_resources.gpu_queue import cleanup_confirmed
    result_path = Path(result_path).absolute()
    run_name = result_path.parent.name
    job_path = OWN / (run_name + '_job.json')
    snapshot = OWN / 'source_snapshots' / run_name
    manifest_path = snapshot / 'manifest.json'
    job, manifest = read(job_path), read(manifest_path)
    if job.get('schema') != 'lingbot_vla2_verified_job_v1' or job.get('stage') != 'training' or job.get('result_file') != str(result_path) or manifest.get('status') != 'completed' or manifest.get('job_sha256') != sha(job_path) or manifest.get('original_hashes') != job['frozen_files']:
        raise ValueError('Completed training work item/source snapshot identity differs')
    for name, expected in job['frozen_files'].items():
        archived = snapshot / 'files' / name.lstrip('/')
        if sha(archived) != expected:
            raise ValueError('Executed training source snapshot changed: ' + name)
        frozen[str(archived)] = expected
    queue = read(ROOT / '.scratch/vla_gpu_coordination/queue.json')
    tickets = [row for row in queue['tickets'] if row['ticket'] == job['fifo_ticket']]
    if len(tickets) != 1 or tickets[0].get('status') != 'completed' or tickets[0].get('cleanup_confirmed') is not True:
        raise ValueError('Training FIFO turn did not close with confirmed cleanup')
    row = read(OWN / 'gpu_budget.json')['runs'][tickets[0]['ledger_runs_before']]
    expected_command = [str(OWN / 'venv/bin/python'), '-m', 'experiments.lingbot_vla2_g1.job', '--job', str(job_path)]
    if row['command'] != expected_command or row['stage'] != 'training' or row['status'] != 'completed' or not cleanup_confirmed(row):
        raise ValueError('Actual owned training command/accounting differs')
    log_path = OWN / (run_name + '_supervisor.log')
    verified = []
    for line in log_path.read_text().splitlines():
        if 'frozen_work_item_verified' in line:
            verified.append(json.loads(line))
    if len(verified) != 1 or verified[0].get('job_sha256') != sha(job_path):
        raise ValueError('Execution lacks its actual frozen-work-item verification')
    for path in [job_path, manifest_path, log_path]:
        frozen[str(path)] = sha(path)
    return {'job': str(job_path), 'job_sha256': sha(job_path), 'source_snapshot_manifest': str(manifest_path), 'source_snapshot_manifest_sha256': sha(manifest_path), 'actual_owned_GPU_accounting': row, 'closed_FIFO_turn': tickets[0]}


def experiment_sources(starts):
    """Pin the actual local Python import closure without unrelated peer reports."""
    seen = set()
    pending = list(starts)
    while pending:
        path = Path(pending.pop()).absolute()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        parts = path.relative_to(ROOT).with_suffix('').parts
        package = list(parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                base = prefix + (node.module.split('.') if node.module else [])
                modules = ['.'.join(base), *['.'.join([*base, alias.name]) for alias in node.names]]
            for module in modules:
                if not module.startswith('experiments.'):
                    continue
                target = ROOT.joinpath(*module.split('.'))
                for candidate in [target.with_suffix('.py'), target / '__init__.py']:
                    if candidate.is_file():
                        pending.append(candidate)
    return seen


def make_plan(training_result, output, episodes_root):
    output = Path(output).absolute()
    episodes_root = Path(episodes_root).absolute()
    if not output.is_relative_to(OWN) or not episodes_root.is_relative_to(OWN) or output.exists() or episodes_root.exists():
        raise ValueError('Fresh LingBot-owned plan and raw-evidence roots required')
    if sha(SCENARIOS) != SCENARIOS_SHA or sha(TEMPLATE) != TEMPLATE_SHA:
        raise ValueError('Shared frozen scenario/template changed')
    scenarios, template = read(SCENARIOS), read(TEMPLATE)
    cases = scenarios['cases']
    if len(cases) != 20 or len({case['case_id'] for case in cases}) != 20 or {case['seed'] for case in cases} != {1040, 1041, 1042, 1043} or len({case['cluster_id'] for case in cases}) != 5:
        raise ValueError('Same frozen twenty cases required')
    if (scenarios['ticks'], scenarios['episode_seconds'], scenarios['native_action_horizon'], scenarios['replan_interval_ticks'], scenarios['control_hz'], scenarios['physics_hz']) != (3000, 60., 50, 50, 50, 200):
        raise ValueError('Common60s/50Hz/200Hz protocol required')
    result, delta, norm, frozen, payloads = verify_selection(training_result)
    provenance = training_provenance(training_result, frozen)
    frozen.update({str(SCENARIOS): SCENARIOS_SHA, str(TEMPLATE): TEMPLATE_SHA})
    offline = verify_offline_assets(template, frozen)
    for name, expected in template['frozen_files'].items():
        if Path(name).suffix != '.py':
            if sha(name) != expected:
                raise ValueError('Admitted asset changed: ' + name)
            frozen[name] = expected
    starts = [ROOT / 'experiments/lingbot_vla2_g1' / (name + '.py') for name in ['run_case', 'service', 'sim_launcher', 'evaluate', 'benchmark']]
    starts += [ROOT / 'experiments/pi05_g1_mobile' / (name + '.py') for name in ['simulator', 'sim_launcher', 'audit_episodes_v3', 'benchmark_v2']]
    sources = experiment_sources(starts)
    for directory in [Path(template['source_harness_dir']), ROOT / 'upstream/unitree_g1/isaaclab_arena_static', ROOT / 'upstream/unitree_g1/isaaclab_static/source', OWN / 'runtime_source']:
        files = list(directory.rglob('*.py'))
        if not files:
            raise ValueError('Required simulator/model source root is missing: ' + str(directory))
        sources.update(files)
    for path in sources:
        frozen[str(path.absolute())] = sha(path)
    for path in [*(OWN / 'qwen_processor').glob('*'), *(OWN / 'base_bf16').glob('*.json'), OWN / 'environment_receipt.json', OWN / 'gpu_budget_allocation_v2.json', SHARED / 'simulator_inventory.json', SHARED / 'runtime_precision_rule.json']:
        if path.is_file():
            frozen[str(path)] = sha(path)
    runtime = docker_identity(read(SHARED / 'simulator_inventory.json'))
    gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True, timeout=10).strip().splitlines()
    expected_gpu = ['NVIDIA GB10', 'GPU-21b3808c-f26c-1b0c-c75e-7957e971606d']
    fields = [part.strip() for part in gpu_lines[0].split(',')] if len(gpu_lines) == 1 else []
    if fields[:2] != expected_gpu or len(fields) != 3:
        raise ValueError('Same single physical NVIDIA GB10 required')
    hardware = {'name': fields[0], 'uuid': fields[1], 'driver': fields[2], 'robot': 'Same frozen Unitree G1 with Homie controllers and14 hand joints'}
    frozen[str(OWN / 'hardware_and_scope.json')] = sha(OWN / 'hardware_and_scope.json')
    selection = {'schema': 'lingbot_vla2_execution_selection_v1', 'training_result': str(Path(training_result).absolute()), 'training_provenance': provenance, 'selected_step': result['selected_step'], 'optimizer_steps': result['optimizer_steps'], 'selected_delta': str(delta) if delta else None, 'selected_validation': result['selected_validation'], 'baseline_validation': result['baseline_validation'], 'norm_path': str(norm), 'norm_sha256': sha(norm), 'selection_rule': 'Strictly lowest complete validation flow loss, earliest tie; unchanged base candidate included', 'formal_outcomes_read': False, 'test_observations_opened': False, 'scenarios_sha256': SCENARIOS_SHA, 'offline_unused_assets': offline, 'runtime': runtime, 'hardware': hardware, 'immutable_payloads': payloads}
    output.mkdir(parents=True)
    selection_path = output / 'selection_receipt.json'
    write(selection_path, selection)
    frozen[str(selection_path)] = sha(selection_path)
    jobs = []
    for case in cases:
        folder = episodes_root / case['case_id']
        command = [str(OWN / 'venv/bin/python'), '-m', 'experiments.lingbot_vla2_g1.service', '--norm-stats', str(norm), '--decoded-evidence', str(folder / 'decoded.jsonl')]
        if delta:
            command += ['--delta', str(delta)]
        config = copy.deepcopy(template)
        config.update(mode='policy', policy_id='lingbot_vla2_selected', case={k: v for k, v in case.items() if k != 'scene'}, output_dir=str(folder), scene=copy.deepcopy(case['scene']), ticks=3000, episode_seconds=60., native_action_horizon=50, replan_interval_ticks=50, evaluate_acceptance=True, stop_on_verified_success=True, save_full_rgb=False, offline_unused_assets=True, capture_preflight_scene_state=False, max_run_wall_seconds=840, prompt=PROMPT, observer_camera=copy.deepcopy(scenarios['observer_camera']), acceptance=copy.deepcopy(scenarios['acceptance']), action_limits_file=scenarios['action_limits_file'], policy_command=command, norm_stats=str(norm), frozen_files=dict(frozen), HFmodel={'repo_id': 'robbyant/lingbot-vla-v2-6b', 'revision': '11c703bf6a5c1f45b3b69168482da11fdbba53d7', 'selected_by_validation': True, 'delta': str(delta) if delta else None})
        config = validate_config(config)
        config_path = output / 'configs' / (case['case_id'] + '.json')
        write(config_path, config)
        jobs.append({'policy': config['policy_id'], 'case_id': case['case_id'], 'config_path': str(config_path), 'config_sha256': sha(config_path), 'result_path': str(folder / 'result.json'), 'max_seconds': 900})
    frozen.update({job['config_path']: job['config_sha256'] for job in jobs})
    plan = {'schema': 'lingbot_vla2_g1_benchmark_plan_v1', 'primary_policies': ['lingbot_vla2_selected'], 'cases': cases, 'jobs': jobs, 'frozen_files': frozen, 'immutable_payloads': payloads, 'selection_receipt': str(selection_path), 'scenarios_sha256': SCENARIOS_SHA, 'episode_root': str(episodes_root), 'order_rule': 'Frozen original case order; no outcome-dependent skipping', 'recording': scenarios['recording'], 'scope': scenarios['task_scope'], 'shared_physical_rules': 'Unchanged peer simulator, acceptance, Homie, robot, scene, head camera, named32 action/state and observer; model-specific preprocessing/normalization declared'}
    write(output / 'plan.json', plan)
    snapshot = output / 'source_snapshot'
    for name, expected in frozen.items():
        path = Path(name)
        if path.suffix not in ['.py', '.json', '.usda', '.yaml'] or path.stat().st_size > 20 * 1024**2:
            continue
        target = snapshot / str(path).lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target) != expected:
            raise ValueError('Source changed during snapshot: ' + name)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes-root', type=Path, required=True)
    args = parser.parse_args()
    plan = make_plan(args.training_result, args.output, args.episodes_root)
    print(json.dumps({'schema': plan['schema'], 'cases': len(plan['jobs']), 'selected': plan['selection_receipt'], 'frozen_files': len(plan['frozen_files'])}))


if __name__ == '__main__':
    main()
