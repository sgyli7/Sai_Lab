"""LingBot decoder identity plus the peer's unchanged physical evidence checks."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from experiments.pi05_g1_mobile.benchmark import read, checked_json, distribution, FLAGS, METRICS
from experiments.pi05_g1.summarize_evaluation import wilson
from .data import OWN, sha


def decoded_evidence(config, result):
    command = config['policy_command']
    if command[:3] != [str(OWN / 'venv/bin/python'), '-m', 'experiments.lingbot_vla2_g1.service']:
        raise ValueError('Frozen LingBot policy service required')
    def argument(flag, optional=False):
        if optional and flag not in command:
            return None
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
            raise ValueError('Exactly one explicit ' + flag + ' required')
        return command[command.index(flag) + 1]
    path = Path(argument('--decoded-evidence')).absolute()
    if not path.is_relative_to(Path(config['output_dir']).absolute()) or path == Path(config['output_dir']).absolute():
        raise ValueError('Case-owned decoder evidence required')
    events = [checked_json(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    if not events or events[0].get('event') != 'model_loaded' or sum(event.get('event') == 'model_loaded' for event in events) != 1:
        raise ValueError('Unique real model-load identity required')
    loaded = events[0]
    norm = Path(argument('--norm-stats')).absolute()
    identities = [('norm_sha256', norm), ('contract_sha256', OWN / 'contract.json'), ('base_conversion_receipt_sha256', OWN / 'bf16_conversion_receipt.json'), ('runtime_overlay_receipt_sha256', OWN / 'runtime_overlay_receipt.json')]
    for field, identity in identities:
        if loaded.get(field) != config['frozen_files'].get(str(identity)):
            raise ValueError('Loaded LingBot identity differs: ' + field)
    if loaded.get('norm_path') != str(norm) or loaded.get('delta') != argument('--delta', optional=True) or loaded.get('parameter_elements') != 6375906359 or loaded.get('trainable_elements') != 1938090551:
        raise ValueError('Loaded LingBot payload/schema differs')
    decoded = events[1:]
    calls = result['inference_calls']
    if not decoded or any(event.get('event') != 'lingbot_decoded' for event in decoded) or [event.get('control_tick') for event in decoded] != [call['observation_control_tick'] for call in calls]:
        raise ValueError('Decoder rejected/missing/extra queries')
    for event, call in zip(decoded, calls):
        array = np.asarray(event.get('actions32'), np.float32)
        if event.get('episode_id') != config['case']['case_id'] or event.get('action_horizon') != 50 or array.shape != (50, 32) or not np.isfinite(array).all() or not np.array_equal(array, np.asarray(event.get('actions'), np.float32)):
            raise ValueError('Exact named32 decoder chunk/case mismatch')
        if event.get('inference_seconds') != call.get('inference_seconds'):
            raise ValueError('Model-reported inference timing differs')
    return path, events


def load_plan(path):
    from .make_plan import SCENARIOS, SCENARIOS_SHA, TEMPLATE
    path = Path(path).absolute()
    plan = read(path)
    if plan.get('schema') != 'lingbot_vla2_g1_benchmark_plan_v1' or plan.get('primary_policies') != ['lingbot_vla2_selected']:
        raise ValueError('Frozen LingBot selected twenty-case plan required')
    ids = [case['case_id'] for case in plan['cases']]
    if len(ids) != 20 or len(set(ids)) != 20 or len(plan['jobs']) != 20 or [job['case_id'] for job in plan['jobs']] != ids or any(job['policy'] != 'lingbot_vla2_selected' for job in plan['jobs']):
        raise ValueError('Frozen complete case order/coverage required')
    for name, expected in plan['frozen_files'].items():
        if sha(name) != expected:
            raise ValueError('Frozen execution dependency changed: ' + name)
    scenarios, template = read(SCENARIOS), read(TEMPLATE)
    if plan.get('scenarios_sha256') != SCENARIOS_SHA or sha(SCENARIOS) != SCENARIOS_SHA or plan['cases'] != scenarios['cases']:
        raise ValueError('Formal cases differ from the common untouched scenario cohort')
    selection = read(plan['selection_receipt'])
    for job, case in zip(plan['jobs'], scenarios['cases']):
        config = read(job['config_path'])
        if sha(job['config_path']) != job['config_sha256'] or config['case'] != {k: v for k, v in case.items() if k != 'scene'} or config['scene'] != case['scene']:
            raise ValueError('Frozen case configuration differs from common case')
        if any(config.get(field) != template.get(field) for field in ['assets', 'mesh_url_paths', 'source_harness_dir', 'geometry_definition', 'robot_geometry']):
            raise ValueError('Robot/physics/source assets differ from the shared simulator')
        if any(config.get(field) != scenarios.get(field) for field in ['ticks', 'episode_seconds', 'native_action_horizon', 'replan_interval_ticks', 'observer_camera', 'acceptance', 'action_limits_file']):
            raise ValueError('Task horizon/cadence/camera/acceptance differs')
        if config.get('norm_stats') != selection['norm_path'] or config.get('HFmodel', {}).get('delta') != selection['selected_delta'] or config.get('save_full_rgb') is not False or config.get('offline_unused_assets') is not True:
            raise ValueError('Selected weights, normalization or recording differ')
    return plan


def carry_metrics(audit, rows):
    # This is the existing acceptance event; no second arrival threshold.
    judgment = audit['recomputed_acceptance']
    tick = judgment['carry_qualified_control_tick']
    first = judgment['first_grasp_control_tick']
    if tick is None:
        return {'long_carry_qualified': False, 'carry_qualified_control_tick': None, 'time_to_qualified_carry_sim_seconds': None, 'time_to_qualified_carry_wall_seconds': None, 'grasp_to_qualified_carry_sim_seconds': None}
    if type(tick) is not int or type(first) is not int or not 0 <= first <= tick < len(rows) or rows[tick]['control_tick'] != tick:
        raise ValueError('Audited transport event lacks measured physical timing')
    return {'long_carry_qualified': True, 'carry_qualified_control_tick': tick, 'time_to_qualified_carry_sim_seconds': tick / 50, 'time_to_qualified_carry_wall_seconds': (rows[tick]['monotonic_ns'] - rows[0]['monotonic_ns']) / 1e9, 'grasp_to_qualified_carry_sim_seconds': (tick - first) / 50}


def aggregate(plan_path, receipt_paths, output):
    from experiments.pi05_g1_mobile.benchmark_v2 import load_case
    plan_path, output = Path(plan_path).absolute(), Path(output).absolute()
    if output.exists() or output.with_suffix('.csv').exists():
        raise FileExistsError('Preserve earlier benchmark evidence')
    plan = load_plan(plan_path)
    receipts = {}
    inputs = {str(plan_path): sha(plan_path)}
    for path in map(Path, receipt_paths):
        doc = read(path)
        if doc.get('schema') != 'lingbot_vla2_g1_evaluation_segment_v1' or doc['plan_sha256'] != inputs[str(plan_path)]:
            raise ValueError('Evaluation receipt plan identity differs')
        if doc.get('model_payloads_verified_before') is not True or doc.get('model_payloads_verified_end') is not True:
            raise ValueError('Both phase boundaries must verify immutable model payloads')
        inputs[str(path.absolute())] = sha(path)
        for record in doc['jobs']:
            if record.get('status') != 'completed':
                continue
            index = record['index']
            if index in receipts and receipts[index] != record:
                raise ValueError('Conflicting immutable completed records')
            receipts[index] = record
    rows, errors = [], []
    for index, job in enumerate(plan['jobs']):
        try:
            record = receipts[index]
            if (record['case_id'], record['policy']) != (job['case_id'], job['policy']):
                raise ValueError('Completed receipt case/policy differs')
            for name, expected in record['case_files_sha256'].items():
                if sha(name) != expected:
                    raise ValueError('Raw case evidence changed: ' + name)
            audit_path = Path(record['cpu_audit_path'])
            if sha(audit_path) != record['cpu_audit_sha256']:
                raise ValueError('Independent CPU audit changed')
            audit = read(audit_path)
            if audit.get('passed') is not True or audit.get('autonomous_policy_trial') is not True or audit.get('result_path') != job['result_path'] or audit.get('case_id') != job['case_id']:
                raise ValueError('Independent physical/action/video audit failed')
            from .evaluate import raw_files
            if raw_files(Path(job['result_path']).parent) != record['case_files_sha256']:
                raise ValueError('Raw case inventory changed')
            ledger = read(OWN / 'gpu_budget.json')
            actual = [entry for entry in ledger['runs'] if entry.get('budget_token') == record['budget_row']['budget_token']]
            if actual != [record['budget_row']] or record['budget_row']['status'] != 'completed':
                raise ValueError('Independent owned GPU accounting differs')
            row = load_case(job, plan, plan_path, decoded_checker=decoded_evidence)
            trace = Path(job['result_path']).with_name('trajectory.jsonl')
            trace_rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
            row.update(carry_metrics(audit, trace_rows))
            rows.append(row)
            inputs.update(row['inputs_sha256'])
            inputs[str(audit_path)] = record['cpu_audit_sha256']
        except (OSError, ValueError, KeyError) as error:
            errors.append({'case_id': job['case_id'], 'error': repr(error)})
    clusters = {case['case_id']: case['cluster_id'] for case in plan['cases']}
    cluster_counts = {cluster: list(clusters.values()).count(cluster) for cluster in sorted(set(clusters.values()))}
    if len(cluster_counts) != 5 or set(cluster_counts.values()) != {4}:
        errors.append({'error': 'Expected five physical positions with four seeds each'})
    if len({row['frozen_interface_sha256'] for row in rows}) > 1:
        errors.append({'error': 'Measured interface changed across cases'})
    complete = len(rows) == 20 and not errors
    counts = {flag: sum(row[flag] for row in rows) for flag in FLAGS}
    failures = {flag: counts[flag] for flag in ['fell', 'dropped', 'timeout']}
    failures.update(no_verified_grasp=sum(not row['grasped'] for row in rows), no_released_stable_placement=sum(not row['stable_placement'] for row in rows))
    for reason in sorted({reason for row in rows for reason in row['termination_reasons']}):
        failures[reason] = sum(reason in row['termination_reasons'] for row in rows)
    public = [{k: v for k, v in row.items() if k not in ['config', 'inputs_sha256']} for row in rows]
    fields = ['policy', 'case_id', 'cluster_id', *FLAGS, 'long_carry_qualified', 'carry_qualified_control_tick', 'time_to_qualified_carry_sim_seconds', 'time_to_qualified_carry_wall_seconds', 'grasp_to_qualified_carry_sim_seconds', *METRICS, 'task_sim_seconds', 'task_wall_seconds', 'episode_wall_seconds', 'action_clipping_fraction', 'initial_physical_state_sha256', 'frozen_interface_sha256', 'result_path']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(public)
    report = {'schema': 'lingbot_vla2_g1_benchmark_summary_v1', 'status': 'completed' if complete else 'incomplete', 'formal_verdict_available': complete, 'completed_evidence_cases': len(rows), 'expected_cases': 20, 'success_count': counts['success'] if complete else None, 'success_rate': counts['success'] / 20 if complete else None, 'success_wilson95ci': wilson(counts['success'], 20) if complete else None, 'observed_counts': counts, 'nonexclusive_failure_counts': failures, 'physical_position_clusters': cluster_counts, 'success_only_sim_seconds': distribution([row['task_sim_seconds'] for row in rows if row['success']]), 'success_only_task_wall_seconds': distribution([row['task_wall_seconds'] for row in rows if row['success']]), 'warm_inference_seconds': distribution([v for row in rows for v in row['warm_inference_seconds']]), 'cold_inference_seconds': distribution([v for row in rows for v in row['cold_inference_seconds']]), 'warm_RPC_pause_wall_seconds': distribution([v for row in rows for v in row['warm_RPC_pause_wall_seconds']]), 'model_load_seconds': distribution([v for row in rows for v in row['model_load_seconds']]), 'physical_metrics': {key: distribution([row[key] for row in rows]) for key in METRICS}, 'episodes': public, 'errors': errors, 'selection_receipt': plan['selection_receipt'], 'scope': plan['scope'], 'timing_scope': 'Simulation pauses during synchronous model RPC. Warm inference compute, task sim time, and actual task wall are distinct. Fast failure does not establish task completion speed.', 'statistical_scope': 'Five physical-position clusters, four repeated seeds each. Wilson interval is descriptive, not twenty independent-scene evidence.', 'inputs_sha256': inputs, 'tool_sha256': sha(__file__)}
    carry_count = sum(row['long_carry_qualified'] for row in rows)
    report.update(long_carry_qualified_count=carry_count if complete else None, long_carry_qualified_rate=carry_count / 20 if complete else None,
                  time_to_qualified_carry_sim_seconds=distribution([row['time_to_qualified_carry_sim_seconds'] for row in rows if row['long_carry_qualified']]),
                  time_to_qualified_carry_wall_seconds=distribution([row['time_to_qualified_carry_wall_seconds'] for row in rows if row['long_carry_qualified']]),
                  carry_scope='Existing contact-qualified4m/60deg event from independently recomputed acceptance. This partial transport event does not mean arrival, release or complete task success.')
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.plan, args.receipt, args.output)
    print(json.dumps({k: report[k] for k in ['status', 'completed_evidence_cases', 'success_count', 'errors']}))


if __name__ == '__main__':
    main()
