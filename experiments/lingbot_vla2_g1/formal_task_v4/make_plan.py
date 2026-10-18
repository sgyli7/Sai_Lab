"""Freeze an admitted actual-final candidate on the untouched twenty-case cohort.

A formal plan is written only after autonomous development has fully succeeded.
This module does not request a GPU turn.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

from experiments.lingbot_vla2_g1.contract import PROMPT
from experiments.lingbot_vla2_g1.data import OWN, ROOT, SHARED, sha
from experiments.lingbot_vla2_g1.make_plan import (
    SCENARIOS, SCENARIOS_SHA, TEMPLATE, TEMPLATE_SHA, artifact, experiment_sources, write)
from .gates import (
    CPU_ROOT, HISTORICAL_ROOTS, admit_development, assert_common_cohort, bind_candidate_role,
    load_formal_plan, same_path, service_command, storage_admission)


def assert_fresh(path):
    path = Path(path).absolute()
    if path.exists():
        raise FileExistsError('Fresh formal output required')
    if any(part in HISTORICAL_ROOTS for part in path.parts) or path.is_relative_to(CPU_ROOT):
        raise ValueError('Refusing to write a formal plan into an existing benchmark, training, or CPU-preparation root')
    if not path.is_relative_to(OWN):
        raise ValueError('Fresh LingBot-owned plan root required')
    return path


def make_plan(training_result, development_plan, development_result, output, episodes_root):
    preview = admit_development(development_plan, development_result, require_actual=True)
    sampling_adapter = preview['plan'].get('sampling_adapter')
    output, episodes_root = assert_fresh(output), assert_fresh(episodes_root)
    training_schema = json.loads(Path(training_result).read_text()).get('schema')
    if training_schema == 'lingbot_vla2_g1_training_task_round04_v1':
        from experiments.lingbot_vla2_g1.round04_selection import admit
        role_name = 'actual_final_task_round04'
    elif training_schema == 'lingbot_vla2_g1_training_task_round05_v1':
        from experiments.lingbot_vla2_g1.round05_selection import admit
        role_name = 'actual_final_task_round05'
    elif training_schema == 'lingbot_vla2_g1_training_task_round06_v1':
        from experiments.lingbot_vla2_g1.round06_selection import admit
        role_name = 'actual_final_task_round06'
    else:
        raise ValueError('Reviewed completed R4, R5 or R6 training required')
    result, delta, frozen, payloads, provenance = admit(training_result)
    if len(payloads) != 7 or not same_path(preview['selected_delta'], delta) or preview['selected_new_step'] != result['physical_candidate_step']:
        raise ValueError('Development checkpoint differs from the actual final candidate')
    role = bind_candidate_role(provenance, role_name, result['selected_step'] == result['physical_candidate_step'])
    admitted = admit_development(development_plan, development_result, delta, result['physical_candidate_step'],
                                 result['selected_step'], payloads[str(Path(delta) / 'trainable.safetensors')], require_actual=True)
    if sha(SCENARIOS) != SCENARIOS_SHA or sha(TEMPLATE) != TEMPLATE_SHA:
        raise ValueError('Shared frozen scenario/template changed')
    scenarios, template = json.loads(SCENARIOS.read_text()), json.loads(TEMPLATE.read_text())
    assert_common_cohort(scenarios['cases'], scenarios)
    from experiments.lingbot_vla2_g1.acceptance_context import admit_qualification, measurement_identity
    from experiments.pi05_g1_mobile.make_benchmark_plan import docker_identity, verify_offline_assets
    from experiments.pi05_g1_mobile.simulator import validate_config
    qualified, qualification_files = admit_qualification(SHARED / 'placement_oracle_attempt03_pose_speed_qualification.json')
    identity = measurement_identity()
    frozen = dict(frozen)
    frozen.update(admitted['frozen_files'])
    frozen.update(qualification_files)
    frozen.update({str(SCENARIOS): SCENARIOS_SHA, str(TEMPLATE): TEMPLATE_SHA})
    offline = verify_offline_assets(template, frozen)
    for name, expected in template['frozen_files'].items():
        if Path(name).suffix != '.py':
            if sha(name) != expected:
                raise ValueError('Admitted asset changed: ' + name)
            frozen[name] = expected
    starts = [ROOT / 'experiments/lingbot_vla2_g1/formal_task_v4' / (name + '.py')
              for name in ('gates', 'make_plan', 'evaluate', 'aggregate', 'cpu_manifest')]
    starts += [ROOT / 'experiments/lingbot_vla2_g1' / (name + '.py')
               for name in ('run_case_with_backup', 'sim_launcher_with_backup', 'service_pointmass', 'budget_reallocated_v6', 'benchmark_pose_speed', 'acceptance_context')]
    starts += [ROOT / 'experiments/lingbot_vla2_g1/diagnostics/development_audit.py']
    starts += [ROOT / 'experiments/lingbot_vla2_g1/diagnostics/development_audit_v2.py',
               ROOT / 'experiments/lingbot_vla2_g1/service_coherent_noise.py']
    starts += [ROOT / 'experiments/pi05_g1_mobile' / (name + '.py')
               for name in ('simulator', 'sim_launcher', 'audit_episodes_v3', 'benchmark_v2', 'acceptance_pose_speed', 'acceptance_body_ids', 'sim_launcher_pose_speed')]
    sources = experiment_sources(starts)
    for directory in [Path(template['source_harness_dir']), ROOT / 'upstream/unitree_g1/isaaclab_arena_static',
                      ROOT / 'upstream/unitree_g1/isaaclab_static/source', OWN / 'runtime_source']:
        files = list(directory.rglob('*.py'))
        if not files:
            raise ValueError('Required simulator/model source root is missing: ' + str(directory))
        sources.update(files)
    for path in sources:
        frozen[str(path.absolute())] = sha(path)
    for path in [*(OWN / 'qwen_processor').glob('*'), *(OWN / 'base_bf16').glob('*.json'),
                 OWN / 'environment_receipt.json', OWN / 'gpu_budget_allocation_user_resume_v6.json',
                 SHARED / 'simulator_inventory.json', SHARED / 'runtime_precision_rule.json',
                 OWN / 'train_meanstd.json', OWN / 'contract.json']:
        if path.is_file():
            frozen[str(path)] = sha(path)
    runtime = docker_identity(json.loads((SHARED / 'simulator_inventory.json').read_text()))
    gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True, timeout=10).strip().splitlines()
    fields = [part.strip() for part in gpu_lines[0].split(',')] if len(gpu_lines) == 1 else []
    if fields[:2] != ['NVIDIA GB10', 'GPU-21b3808c-f26c-1b0c-c75e-7957e971606d'] or len(fields) != 3:
        raise ValueError('Same single physical NVIDIA GB10 required')
    hardware = {'name': fields[0], 'uuid': fields[1], 'driver': fields[2],
                'robot': 'Same frozen Unitree G1 with Homie controllers and14 hand joints'}
    norm = OWN / 'train_meanstd.json'
    output.mkdir(parents=True)
    selection = {'schema': 'lingbot_vla2_execution_selection_v1', 'candidate_role': role['candidate_role'],
                 'sampling_adapter': sampling_adapter,
                 'initial_lineage_updates': provenance['initial_lineage_updates'],
                 'source_validation_best_step': role['source_validation_best_step'],
                 'physical_candidate_step': role['physical_candidate_step'],
                 'selected_by_validation': role['selected_by_validation'], 'selected_delta': str(delta),
                 'norm_path': str(norm), 'norm_sha256': sha(norm), 'training_result': str(Path(training_result).absolute()),
                 'training_provenance': provenance, 'development_plan': str(Path(development_plan).absolute()),
                 'development_plan_sha256': sha(development_plan), 'development_result': str(Path(development_result).absolute()),
                 'development_result_sha256': sha(development_result), 'acceptance_measurement': identity,
                 'candidate_formal_rollouts_read_before_selection': False,
                 'historical_predecessor_evaluated_same_frozen_cohort': True,
                 'formal_trajectories_used_for_training': False, 'actual_robot_success_claimed': False,
                 'offline_unused_assets': offline, 'runtime': runtime, 'hardware': hardware,
                 'immutable_payloads': payloads, 'scenarios_sha256': SCENARIOS_SHA}
    selection_path = output / 'selection_receipt.json'
    write(selection_path, selection)
    frozen[str(selection_path)] = sha(selection_path)
    jobs = []
    for case in scenarios['cases']:
        folder = episodes_root / case['case_id']
        config = copy.deepcopy(template)
        config.update(mode='policy', policy_id='lingbot_vla2_selected',
                      case={key: value for key, value in case.items() if key != 'scene'},
                      output_dir=str(folder), scene=copy.deepcopy(case['scene']), ticks=scenarios['ticks'],
                      episode_seconds=scenarios['episode_seconds'], native_action_horizon=scenarios['native_action_horizon'],
                      replan_interval_ticks=scenarios['replan_interval_ticks'], evaluate_acceptance=True,
                      stop_on_verified_success=True, save_full_rgb=False, offline_unused_assets=True,
                      capture_preflight_scene_state=False, max_run_wall_seconds=840, prompt=PROMPT,
                      observer_camera=copy.deepcopy(scenarios['observer_camera']),
                      acceptance=copy.deepcopy(scenarios['acceptance']),
                      action_limits_file=scenarios['action_limits_file'],
                      policy_command=service_command(delta, folder, sampling_adapter), norm_stats=str(norm),
                      frozen_files=dict(frozen), acceptance_measurement=identity,
                      HFmodel={'repo_id': 'robbyant/lingbot-vla-v2-6b',
                               'revision': '11c703bf6a5c1f45b3b69168482da11fdbba53d7',
                               'selected_by_validation': role['selected_by_validation'], 'delta': str(delta),
                               'sampling_adapter': sampling_adapter})
        config = validate_config(config)
        config_path = output / 'configs' / (case['case_id'] + '.json')
        write(config_path, config)
        jobs.append({'policy': 'lingbot_vla2_selected', 'case_id': case['case_id'], 'config_path': str(config_path),
                     'config_sha256': sha(config_path), 'result_path': str(folder / 'result.json'), 'max_seconds': 900})
    frozen.update({job['config_path']: job['config_sha256'] for job in jobs})
    plan = {'schema': 'lingbot_vla2_g1_pose_speed_benchmark_plan_v1', 'primary_policies': ['lingbot_vla2_selected'],
            'sampling_adapter': sampling_adapter,
            'initial_lineage_updates': provenance['initial_lineage_updates'],
            'candidate_role': role['candidate_role'], 'source_validation_best_step': role['source_validation_best_step'],
            'physical_candidate_step': role['physical_candidate_step'], 'selected_by_validation': role['selected_by_validation'],
            'cases': scenarios['cases'], 'jobs': jobs, 'frozen_files': frozen, 'immutable_payloads': payloads,
            'selection_receipt': str(selection_path), 'scenarios_sha256': SCENARIOS_SHA,
            'episode_root': str(episodes_root), 'acceptance_measurement': identity,
            'shared_placement_qualification': qualified, 'storage_admission': storage_admission(),
            'development_plan': str(Path(development_plan).absolute()), 'development_plan_sha256': sha(development_plan),
            'development_result': str(Path(development_result).absolute()), 'development_result_sha256': sha(development_result),
            'order_rule': 'Frozen original case order; no outcome-dependent skipping or retries into the same raw directory',
            'recording': scenarios['recording'], 'scope': scenarios['task_scope'],
            'policy_inputs': 'rgb_q_text_only', 'actual_robot_success_claimed': False,
            'shared_physical_rules': 'Unchanged peer simulator, acceptance, Homie, robot, scene, head camera, named32 action/state and observer; service_pointmass constant channels use original TRAIN mean'}
    write(output / 'plan.json', plan)
    copied = {}
    for name, expected in frozen.items():
        path = Path(name)
        if path.suffix not in ('.py', '.json', '.usda', '.yaml') or path.stat().st_size > 20 * 1024 ** 2:
            continue
        target = output / 'source_snapshot' / str(path).lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target) != expected:
            raise ValueError('Source changed during snapshot: ' + name)
        copied[str(target)] = expected
    write(output / 'source_snapshot_manifest.json', {
        'schema': 'lingbot_formal_task_v4_source_snapshot_v1', 'status': 'completed',
        'plan_sha256': sha(output / 'plan.json'), 'before_first_physical_case': True,
        'copied_files_sha256': copied, 'formal_plan_generated': True})
    load_formal_plan(output / 'plan.json')
    return json.loads((output / 'plan.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-result', type=Path, required=True)
    parser.add_argument('--development-plan', type=Path, required=True)
    parser.add_argument('--development-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes-root', type=Path, required=True)
    args = parser.parse_args()
    plan = make_plan(args.training_result, args.development_plan, args.development_result, args.output, args.episodes_root)
    print(json.dumps({'schema': plan['schema'], 'cases': len(plan['jobs']), 'candidate_role': plan['candidate_role'],
                      'selected_by_validation': plan['selected_by_validation'], 'formal_plan_generated': True}))


if __name__ == '__main__':
    main()
