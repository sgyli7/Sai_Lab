"""CPU gates for the unchanged formal task and admitted R4/R5/R6 candidates."""
from __future__ import annotations
import json
from pathlib import Path

from experiments.lingbot_vla2_g1.data import OWN, sha
from experiments.lingbot_vla2_g1.coherent_sampling import ADAPTER as COHERENT_ADAPTER
from experiments.lingbot_vla2_g1.benchmark_pose_speed import load_plan as load_common_plan

COMBINED_CAP = 64 * 1024 ** 3
HEADROOM = 1024 ** 3
GLOBAL_FREE = 24 * 1024 ** 3
FREE_BUFFER = 1024 ** 3
CASE_SECONDS = 900
TARGET_SUCCESS = 16
BACKUP = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z')
CPU_ROOT = OWN / 'formal_task_v4_CPU_preparation'
OWNER = '01a145c6-0b86-7623-a869-086b33292199'
REQUIRED_CHECKS = frozenset({
    'physical_counter_timing', 'initial_kinematic_sha', 'actual_runtime_identity',
    'actions', 'policy_feedback', 'videos', 'acceptance'})
HISTORICAL_ROOTS = frozenset({
    'evaluation_plan_pose_speed', 'evaluation_plan_pose_speed_resume01',
    'evaluation_plan_round02', 'evaluation_plan_storage_v2', 'training_task_round04', 'training_task_round05',
    'training_task_round06', 'training_round02', 'training_round03', 'resumed_turn01'})
PROTOCOL = ('ticks', 'episode_seconds', 'native_action_horizon', 'replan_interval_ticks',
            'control_hz', 'physics_hz')
PROTOCOL_VALUE = (3000, 60.0, 50, 50, 50, 200)


def read_json(path):
    return json.loads(Path(path).read_text())


def same_path(left, right):
    return Path(left).absolute() == Path(right).absolute()


SAMPLING_SERVICES = {None: 'experiments.lingbot_vla2_g1.service_pointmass',
                    COHERENT_ADAPTER: 'experiments.lingbot_vla2_g1.service_coherent_noise'}


def service_command(delta, folder, sampling_adapter=None):
    norm = OWN / 'train_meanstd.json'
    if sampling_adapter not in SAMPLING_SERVICES:
        raise ValueError('Unknown prospective sampling adapter')
    return [str(OWN / 'venv/bin/python'), '-m', SAMPLING_SERVICES[sampling_adapter],
            '--delta', str(Path(delta).absolute()), '--norm-stats', str(norm),
            '--decoded-evidence', str(Path(folder) / 'decoded.jsonl')]


def assert_autonomous_service(config):
    if config.get('privileged_TRAIN_teacher') or config.get('teacher'):
        raise ValueError('Privileged or scripted teacher is not autonomous development')
    identity = {'policy_id': config.get('policy_id'), 'policy_command': config.get('policy_command')}
    text = json.dumps(identity).lower()
    if 'privileged' in text or 'scripted' in text or 'teacher' in text:
        raise ValueError('Privileged or scripted teacher is not autonomous development')
    command = config.get('policy_command') or []
    sampling = config.get('HFmodel', {}).get('sampling_adapter')
    if (sampling not in SAMPLING_SERVICES
            or command[:3] != [str(OWN / 'venv/bin/python'), '-m', SAMPLING_SERVICES.get(sampling)]):
        raise ValueError('Prospective declared LingBot sampling policy required')
    if config.get('mode') != 'policy':
        raise ValueError('Autonomous policy mode required')
    if '--delta' not in command or '--norm-stats' not in command or '--decoded-evidence' not in command:
        raise ValueError('Prospective service_pointmass policy required')


def assert_seven_checks(audit):
    checks = audit.get('checks') or {}
    if (audit.get('passed') is not True or audit.get('autonomous_policy_trial') is not True
            or audit.get('anomalies') or set(checks) != REQUIRED_CHECKS
            or any(not isinstance(item, dict) or item.get('passed') is not True for item in checks.values())):
        raise ValueError('Failed independent audit')
    streams = checks['videos'].get('streams') or {}
    if set(streams) != {'head_video_path', 'observer_video_path'}:
        raise ValueError('Both head and observer videos required')


def outcomes_match(recorded, audit, case_result, screen):
    recomputed = audit.get('recomputed_acceptance')
    if recorded != recomputed:
        raise ValueError('Physical outcome differs from independent audit')
    acceptance = case_result.get('acceptance') or {}
    for key, value in recorded.items():
        if acceptance.get(key) != value:
            raise ValueError('Physical outcome differs from case result')
    if case_result.get('success') is not recorded.get('success'):
        raise ValueError('Physical outcome differs from case result')
    if (case_result.get('physical_failure') is True or case_result.get('termination_reason') in
            ('source_object_dropped_predicate', 'source_fall_guard', 'source_robot_fallen_predicate')):
        raise ValueError('Source fall/drop termination cannot qualify autonomous development')
    if screen:
        if recorded.get('grasped') is not True or recorded.get('fell') is not False or recorded.get('dropped') is not False:
            raise ValueError('Grasp screen physical outcome disagrees')
    elif recorded.get('success') is not True:
        raise ValueError('Full development physical outcome disagrees')


def admit_development(plan_path, result_path, candidate_delta=None, candidate_step=None,
                      source_best_step=None, candidate_payload=None, require_actual=False):
    """Require a completed autonomous two-case development execution."""
    plan_path, result_path = Path(plan_path).absolute(), Path(result_path).absolute()
    plan, result = read_json(plan_path), read_json(result_path)
    if plan.get('schema') != 'lingbot_task_v4_development_plan_v1':
        raise ValueError('Development plan schema lingbot_task_v4_development_plan_v1 required')
    if result.get('schema') != 'lingbot_task_v4_development_execution_v1':
        raise ValueError('Development result schema lingbot_task_v4_development_execution_v1 required')
    if result.get('status') != 'completed' or plan.get('formal20_benchmark') is not False or result.get('formal20_benchmark') is not False:
        raise ValueError('Incomplete full development cannot open the formal cohort')
    if result.get('grasp_screen_passed') is not True:
        raise ValueError('Incomplete full development cannot open the formal cohort')
    if result.get('full_task_case_ran') is not True or result.get('full_task_case_success') is not True:
        raise ValueError('Skipped or incomplete full development cannot open the formal cohort')
    if result.get('overall_robot_task_goal_completed') is not False:
        raise ValueError('Development must leave the overall robot task goal unfinished')
    if result.get('plan_sha256') != sha(plan_path):
        raise ValueError('Development plan SHA differs')
    planned, done = plan.get('jobs') or [], result.get('jobs') or []
    if len(planned) != 2 or len(done) != 2:
        raise ValueError('Exactly two completed development jobs required')
    declared = plan.get('selected_delta')
    if not declared or not isinstance(plan.get('selected_new_step'), int) or plan['selected_new_step'] <= 0:
        raise ValueError('Development checkpoint differs from the actual final candidate')
    if candidate_delta is not None and not same_path(declared, candidate_delta):
        raise ValueError('Development checkpoint differs from the actual final candidate')
    if candidate_step is not None and plan['selected_new_step'] != candidate_step:
        raise ValueError('Development checkpoint differs from the actual final candidate')
    if candidate_payload is not None:
        key = str(Path(declared).absolute() / 'trainable.safetensors')
        if (plan.get('immutable_payloads') or {}).get(key) != candidate_payload:
            raise ValueError('Development checkpoint differs from the actual final candidate')
    truthful = None if source_best_step is None or candidate_step is None else candidate_step == source_best_step
    frozen = {str(plan_path): sha(plan_path), str(result_path): sha(result_path)}
    for index, (job, record) in enumerate(zip(planned, done)):
        if record.get('status') != 'completed' or record.get('index') != index or record.get('case_id') != job.get('case_id'):
            raise ValueError('Skipped or incomplete full development cannot open the formal cohort')
        config_path = Path(job['config_path'])
        if sha(config_path) != job.get('config_sha256'):
            raise ValueError('Development config SHA differs')
        config = read_json(config_path)
        assert_autonomous_service(config)
        if config.get('HFmodel', {}).get('sampling_adapter') != plan.get('sampling_adapter'):
            raise ValueError('Development sampling adapter differs from the prospective plan')
        command = config['policy_command']
        if not same_path(config.get('HFmodel', {}).get('delta'), declared) or not same_path(command[command.index('--delta') + 1], declared):
            raise ValueError('Development checkpoint differs from the actual final candidate')
        if not same_path(command[command.index('--norm-stats') + 1], OWN / 'train_meanstd.json'):
            raise ValueError('Original TRAIN normalization required')
        if truthful is not None and config.get('HFmodel', {}).get('selected_by_validation') is not truthful:
            raise ValueError('Final versus source-validation-best role cannot be relabelled')
        if sha(record['result_file']) != record.get('result_sha256') or sha(record['audit']) != record.get('audit_sha256'):
            raise ValueError('Development result or audit SHA differs')
        folder = Path(config['output_dir'])
        actual = {str(path): sha(path) for path in sorted(folder.rglob('*')) if path.is_file()}
        if actual != record.get('raw_files_sha256'):
            raise ValueError('Development raw inventory differs')
        audit = read_json(record['audit'])
        assert_seven_checks(audit)
        if audit.get('case_id') != job['case_id'] or not same_path(audit.get('result_path'), record['result_file']):
            raise ValueError('Failed independent audit')
        outcomes_match(record.get('physical_outcome'), audit, read_json(record['result_file']), index == 0)
        if require_actual:
            case_result = read_json(record['result_file'])
            if any(value.get('synthetic') is True for value in (plan, result, config, record, audit, case_result)):
                raise ValueError('Synthetic fixtures cannot qualify an actual formal plan')
            outcome = record['physical_outcome']
            if index == 0 and (outcome.get('success') is not True
                    and case_result.get('completed_control_ticks', -1) < config.get('ticks', 1000)):
                raise ValueError('Actual grasp screen must complete its full frozen horizon without source failure')
            if index == 1 and (outcome.get('stable_placement') is not True
                    or outcome.get('fell') is not False or outcome.get('dropped') is not False
                    or outcome.get('max_carried_box_displacement_m', 0) < 4
                    or outcome.get('carry_turn_degrees', 0) < 60
                    or outcome.get('source_target_distance_m', 0) < 5
                    or outcome.get('policy_external_interventions') != 0):
                raise ValueError('Actual complete autonomous five-metre development required')
        frozen[str(config_path)] = job['config_sha256']
        frozen[str(Path(record['result_file']).absolute())] = record['result_sha256']
        frozen[str(Path(record['audit']).absolute())] = record['audit_sha256']
        frozen.update(actual)
    return {'selected_delta': str(Path(declared).absolute()), 'selected_new_step': plan['selected_new_step'],
            'frozen_files': frozen, 'plan': plan, 'result': result}


def bind_candidate_role(provenance, declared_role, declared_selected_by_validation):
    best, physical = provenance['source_validation_best_step'], provenance['physical_candidate_step']
    initial = provenance.get('initial_lineage_updates', 1000)
    expected_role = {1000: 'actual_final_task_round04', 1400: 'actual_final_task_round05',
                     1800: 'actual_final_task_round06'}.get(initial)
    if type(best) is not int or type(physical) is not int or physical <= 0 or not 0 <= best <= physical:
        raise ValueError('Physical candidate step required')
    truthful = best == physical
    if (provenance.get('physical_candidate_is_source_validation_best') is not truthful
            or expected_role is None or declared_role != expected_role
            or declared_selected_by_validation is not truthful):
        raise ValueError('Final versus source-validation-best role cannot be relabelled')
    return {'candidate_role': expected_role, 'source_validation_best_step': best,
            'physical_candidate_step': physical, 'selected_by_validation': truthful}


def scenario_document():
    from experiments.lingbot_vla2_g1.make_plan import SCENARIOS, SCENARIOS_SHA
    if sha(SCENARIOS) != SCENARIOS_SHA:
        raise ValueError('Shared frozen scenario cohort changed')
    return json.loads(SCENARIOS.read_text()), SCENARIOS_SHA


def assert_common_cohort(cases, scenarios):
    original, _digest = scenario_document()
    if cases != original['cases']:
        raise ValueError('Formal case order/scene differs from the frozen cohort')
    if len(cases) != 20 or len({case['case_id'] for case in cases}) != 20:
        raise ValueError('Formal case order/scene differs from the frozen cohort')
    clusters = [case['cluster_id'] for case in cases]
    if len(set(clusters)) != 5 or any(clusters.count(cluster) != 4 for cluster in set(clusters)):
        raise ValueError('Five physical positions with four seeds required')
    if tuple(scenarios.get(field) for field in PROTOCOL) != PROTOCOL_VALUE:
        raise ValueError('Common cadence changed')
    if scenarios.get('acceptance') != original['acceptance']:
        raise ValueError('Common acceptance thresholds changed')
    if scenarios.get('observer_camera') != original['observer_camera'] or scenarios.get('action_limits_file') != original['action_limits_file']:
        raise ValueError('Common camera or action bounds changed')
    return original


def assert_case_protocol(config, scenarios):
    fields = ('ticks', 'episode_seconds', 'native_action_horizon', 'replan_interval_ticks',
              'observer_camera', 'acceptance', 'action_limits_file')
    if any(config.get(field) != scenarios.get(field) for field in fields):
        raise ValueError('Common cadence changed')
    original, _ = scenario_document()
    if scenarios.get('acceptance') != original['acceptance']:
        raise ValueError('Common acceptance thresholds changed')


def storage_admission():
    return {'schema': 'lingbot_storage_admission_combined64_v1', 'combined_cap_bytes': COMBINED_CAP,
            'next_case_headroom_bytes': HEADROOM, 'global_free_floor_bytes': GLOBAL_FREE,
            'global_free_buffer_bytes': FREE_BUFFER, 'host_reserve_gib': 16, 'memory_high_gib': 80,
            'memory_max_gib': 90, 'backup_root': str(BACKUP), 'budget_policy': 'budget_reallocated_v6',
            'wall_deadline_enforced': False}


def admit_storage(storage):
    expected = storage_admission()
    if not isinstance(storage, dict) or storage.get('schema') != expected['schema'] or storage.get('combined_cap_bytes') != COMBINED_CAP:
        raise ValueError('Combined 64GiB continuation storage required')
    obsolete = {25 * 1024 ** 3, 36 * 1024 ** 3}
    if storage.get('own_cap_bytes') in obsolete or storage.get('combined_cap_bytes') == 48 * 1024 ** 3:
        raise ValueError('Obsolete own or combined storage cap')
    if storage.get('budget_policy') != 'budget_reallocated_v6' or storage.get('wall_deadline_enforced') is not False:
        raise ValueError('budget_reallocated_v6 required; expired wall cutoff is not admission')
    for key, value in expected.items():
        if key in ('schema', 'combined_cap_bytes', 'budget_policy', 'wall_deadline_enforced'):
            continue
        if storage.get(key) != value:
            raise ValueError('Combined 64GiB headroom or cgroup commitment differs')
    return storage


def recheck_turns(allocation, ledger, turns):
    if allocation.get('old_wall_deadline_enforced') is not False:
        raise ValueError('Expired wall cutoff cannot admit a turn')
    if type(turns) is not int or turns < 1:
        raise ValueError('No full turn to admit')
    need = CASE_SECONDS * turns
    used_eval = sum(row['elapsed_seconds'] for row in ledger['runs'] if row['stage'] == 'evaluation')
    if allocation['total_limit_seconds'] - allocation['used_total_seconds'] < need:
        raise ValueError('Remaining total allocation cannot fit the full next turn')
    if allocation['stage_limits_seconds']['evaluation'] - used_eval < need:
        raise ValueError('Remaining evaluation allocation cannot fit the full next turn')


def assert_result_destination(path):
    path = Path(path).absolute()
    if path.exists() or path.with_suffix('.csv').exists():
        raise FileExistsError('Preserve earlier benchmark evidence')
    if any(part in HISTORICAL_ROOTS for part in path.parts):
        raise ValueError('CPU fixtures cannot be written into real benchmark outputs')


def assert_role_fields(plan, selection):
    provenance = {'source_validation_best_step': plan['source_validation_best_step'],
                  'physical_candidate_step': plan['physical_candidate_step'],
                  'initial_lineage_updates': plan.get('initial_lineage_updates', 1000),
                  'physical_candidate_is_source_validation_best': plan['source_validation_best_step'] == plan['physical_candidate_step']}
    if plan.get('initial_lineage_updates') != selection.get('initial_lineage_updates'):
        raise ValueError('Initial model lineage differs from selection receipt')
    if plan.get('sampling_adapter') != selection.get('sampling_adapter'):
        raise ValueError('Sampling adapter differs from selection receipt')
    role = bind_candidate_role(provenance, plan.get('candidate_role'), selection.get('selected_by_validation'))
    for key, value in role.items():
        if plan.get(key) != value or selection.get(key) != value:
            raise ValueError('Final versus source-validation-best role cannot be relabelled')
    return role


def load_formal_plan(path):
    """Common-cohort checker plus R4 role and development provenance."""
    plan = load_common_plan(path)
    selection = read_json(plan['selection_receipt'])
    assert_role_fields(plan, selection)
    admit_storage(plan.get('storage_admission'))
    original, digest = scenario_document()
    if plan.get('scenarios_sha256') != digest:
        raise ValueError('Formal case order/scene differs from the frozen cohort')
    assert_common_cohort(plan['cases'], original)
    if len(plan.get('immutable_payloads') or {}) != 7:
        raise ValueError('All seven model payloads required')
    if any(job.get('max_seconds') != CASE_SECONDS for job in plan['jobs']):
        raise ValueError('Each case requires a separate finite 900 second budget')
    for job in plan['jobs']:
        config = read_json(job['config_path'])
        assert_autonomous_service(config)
        assert_case_protocol(config, original)
        if not same_path(config.get('HFmodel', {}).get('delta'), selection['selected_delta']):
            raise ValueError('Development checkpoint differs from the actual final candidate')
        sampling = selection.get('sampling_adapter')
        if (config.get('HFmodel', {}).get('sampling_adapter') != sampling
                or config['policy_command'] != service_command(selection['selected_delta'], Path(config['output_dir']), sampling)):
            raise ValueError('Actual policy arguments differ from the declared selected candidate')
        if config.get('HFmodel', {}).get('selected_by_validation') is not selection['selected_by_validation']:
            raise ValueError('Final versus source-validation-best role cannot be relabelled')
    if sha(plan['development_plan']) != plan.get('development_plan_sha256') or sha(plan['development_result']) != plan.get('development_result_sha256'):
        raise ValueError('Development plan SHA differs')
    admit_development(plan['development_plan'], plan['development_result'], selection['selected_delta'],
                      plan['physical_candidate_step'], plan['source_validation_best_step'], require_actual=True)
    return plan


def _teacher_text(document):
    if document.get('teacher') or document.get('privileged_scripted_teacher'):
        return 'teacher'
    return json.dumps({'policy': document.get('policy'), 'policy_id': document.get('policy_id')}).lower()


def apply_target(report, documents):
    """Sixteen full successes across a complete twenty can set the performance flag."""
    report = dict(report)
    report.pop('goal_performance_passed', None)
    jobs, blocked = {}, None
    synthetic = [doc.get('synthetic') is True for doc in documents]
    if any(synthetic) and not all(synthetic):
        blocked = 'mixed synthetic and actual receipts'
    for doc in documents:
        if doc.get('teacher_data') is True or doc.get('privileged_or_scripted_teacher') is True:
            blocked = blocked or 'teacher'
        for record in doc.get('jobs') or []:
            if 'privileged' in _teacher_text(record) or 'scripted' in _teacher_text(record) or 'teacher' in _teacher_text(record):
                blocked = blocked or 'teacher'
            if record.get('status') != 'completed':
                blocked = blocked or 'infrastructure'
            if record.get('audit_passed') is False:
                blocked = blocked or 'failed audit'
            index = record.get('index')
            if index in jobs and jobs[index] != record:
                blocked = blocked or 'conflicting records'
            jobs[index] = record
    full = 0
    for record in jobs.values():
        if record.get('partial_carry_only') is True or record.get('full_task_success') is False:
            if record.get('success') is True:
                blocked = blocked or 'partial carry'
            continue
        if record.get('success') is True and record.get('status') == 'completed':
            full += 1
    complete = (report.get('status') == 'completed' and report.get('formal_verdict_available') is True
                and report.get('completed_evidence_cases') == 20 and report.get('expected_cases') == 20
                and not report.get('errors') and set(jobs) == set(range(20)) and blocked is None)
    success = report.get('success_count')
    passed = bool(complete and success == full and full >= TARGET_SUCCESS)
    if report.get('status') != 'completed' or report.get('completed_evidence_cases') != 20:
        blocked = blocked or 'incomplete'
    elif success is not None and success < TARGET_SUCCESS:
        blocked = blocked or 'below target'
    elif success != full:
        blocked = blocked or 'forged success count'
    report['target_success_count'] = TARGET_SUCCESS
    report['goal_performance_passed'] = passed
    report['goal_performance_block'] = None if passed else blocked
    report['actual_robot_success_claimed'] = False
    report['acceptance_conclusion_owner'] = 'parent'
    if any(synthetic) or report.get('synthetic') is True:
        report['synthetic'] = True
    return report
