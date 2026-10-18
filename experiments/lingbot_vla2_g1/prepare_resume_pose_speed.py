"""Rebind an audited prefix and preserve one interrupted infrastructure attempt.

Only output paths change for uncompleted cases. No weight, selection, scenario,
acceptance, policy input or action parameter may change during recovery.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time

from experiments.vla_resources import gpu_queue
from .benchmark_pose_speed import load_plan
from .data import OWN, ROOT, sha
from .evaluate_pose_speed import previous_records, raw_files, verify_payloads
from .make_plan import experiment_sources, write


def completed_prefix(receipt, plan):
    if receipt.get('schema') != 'lingbot_vla2_g1_pose_speed_evaluation_segment_v1' or receipt.get('status') != 'infrastructure_failed':
        raise ValueError('An immutable interrupted evaluation receipt is required')
    if not receipt.get('model_payloads_verified_before') or not receipt.get('model_payloads_verified_end'):
        raise ValueError('Both historical immutable-model boundary checks required')
    jobs = receipt['jobs']
    done = [job for job in jobs if job.get('status') == 'completed']
    if [job['index'] for job in done] != list(range(len(done))) or len(jobs) != len(done) + 1:
        raise ValueError('Exactly one failed tail after a contiguous completed prefix required')
    failed = jobs[-1]
    if failed.get('status') != 'infrastructure_failed' or failed['index'] != len(done):
        raise ValueError('Physical policy failures must not be retried as infrastructure')
    if receipt.get('completed_total_cases') != len(done):
        raise ValueError('Historical completion counter differs')
    for job in jobs:
        expected = plan['jobs'][job['index']]
        if any(job.get(key) != expected.get(key) for key in ['case_id', 'policy', 'result_path', 'config_sha256']):
            raise ValueError('Interrupted receipt case identity differs')
    return done, failed


def prepare(previous_plan, interrupted_receipt, output, episodes_root, driver_source):
    previous_plan, interrupted_receipt = Path(previous_plan).absolute(), Path(interrupted_receipt).absolute()
    output, episodes_root, driver_source = map(lambda p: Path(p).absolute(), [output, episodes_root, driver_source])
    if output.exists() or episodes_root.exists() or not output.is_relative_to(OWN) or not episodes_root.is_relative_to(OWN) or not driver_source.is_relative_to(OWN):
        raise ValueError('Fresh owned recovery paths and owned finite driver required')
    previous = load_plan(previous_plan)
    source_receipt = json.loads(interrupted_receipt.read_text())
    if source_receipt.get('plan_sha256') != sha(previous_plan):
        raise ValueError('Interrupted receipt belongs to a different plan')
    done, failed = completed_prefix(source_receipt, previous)
    if len(done) != 2 or failed['fifo_ticket'] != 't000020':
        raise ValueError('This recovery is scoped to the captured t20 interruption only')
    ledger = json.loads((OWN / 'gpu_budget.json').read_text())
    for record in done:
        if raw_files(Path(record['result_path']).parent) != record['case_files_sha256'] or sha(record['cpu_audit_path']) != record['cpu_audit_sha256']:
            raise ValueError('Completed prefix raw evidence/audit changed')
        audit = json.loads(Path(record['cpu_audit_path']).read_text())
        if audit.get('passed') is not True or audit.get('autonomous_policy_trial') is not True or audit.get('case_id') != record['case_id']:
            raise ValueError('Completed prefix lacks independent autonomous audit')
        actual = [row for row in ledger['runs'] if row.get('budget_token') == record['budget_row']['budget_token']]
        if actual != [record['budget_row']] or not gpu_queue.cleanup_confirmed(record['budget_row']):
            raise ValueError('Completed prefix owned accounting changed')
    ticket = next(row for row in gpu_queue.status(ROOT / '.scratch/vla_gpu_coordination')['tickets'] if row['ticket'] == failed['fifo_ticket'])
    if ticket.get('status') != 'failed' or ticket.get('cleanup_confirmed') is not True:
        raise ValueError('Interrupted GPU turn is not completely closed')
    failure_row = ledger['runs'][ticket['ledger_runs_before']]
    if failure_row.get('status') != 'interrupted' or failure_row.get('exit_code') != 137 or not gpu_queue.cleanup_confirmed(failure_row):
        raise ValueError('Captured failure is not the admitted signal interruption')
    if failure_row['elapsed_seconds'] >= failure_row['allowed_seconds'] or failure_row['minimum_host_memavailable_bytes'] <= failure_row['host_memavailable_stop_bytes']:
        raise ValueError('Do not silently retry a time or memory-budget failure')
    failed_result = Path(failed['result_path'])
    if failed_result.exists():
        raise ValueError('Captured interruption had no producer result; preserve any unexpected result')
    for job in previous['jobs'][len(done) + 1:]:
        if Path(job['result_path']).parent.exists():
            raise ValueError('An unaccounted later attempt already exists')
    diagnostic = OWN / 'diagnostics/formal_interruption_t20'
    attached = json.loads((diagnostic / 'probe_attached01/result.json').read_text())
    isolated = json.loads((diagnostic / 'probe_isolated01/result.json').read_text())
    if attached.get('passed') is not False or attached.get('observed_state_after_external_group_signal') != 'stopped' or isolated.get('passed') is not True or isolated.get('observed_state_after_external_group_signal') != 'running':
        raise ValueError('CPU signal isolation must pass its controlled differential probe')
    if attached['driver_source_sha256'] != isolated['driver_source_sha256'] or attached['driver_source_sha256'] != sha(OWN / 'preparation_scripts/evaluate_pose_speed_driver.py'):
        raise ValueError('CPU signal probe did not use the recorded original driver')
    verify_payloads(previous)
    output.mkdir(parents=True)
    failure_inventory = raw_files(failed_result.parent)
    admission = {'schema': 'lingbot_signal_interruption_recovery_admission_v1', 'prepared_unix': time.time(),
                 'historical_signal_sender_identified': False, 'retry_reason': 'Captured parent-driver signal interruption; CPU service isolation qualified separately',
                 'previous_plan': str(previous_plan), 'previous_plan_sha256': sha(previous_plan),
                 'interrupted_receipt': str(interrupted_receipt), 'interrupted_receipt_sha256': sha(interrupted_receipt),
                 'audited_completed_prefix': len(done), 'failed_case_id': failed['case_id'], 'failed_ticket': ticket,
                 'failed_budget_row': failure_row, 'failed_raw_files_sha256': failure_inventory,
                 'failed_attempt_is_unknown_task_outcome': True, 'automatic_retry': False,
                 'model_selection_weights_data_scenarios_physics_thresholds_unchanged': True}
    admission_path = output / 'recovery_admission.json'
    write(admission_path, admission)
    frozen = dict(previous['frozen_files'])
    frozen.update(failure_inventory)
    for path in [previous_plan, interrupted_receipt, admission_path, driver_source]:
        frozen[str(path)] = sha(path)
    for path in experiment_sources([Path(__file__).absolute()]):
        frozen[str(path)] = sha(path)
    for path in (OWN / 'diagnostics/formal_interruption_t20').glob('probe_*/result.json'):
        frozen[str(path)] = sha(path)
    jobs = copy.deepcopy(previous['jobs'][:len(done)])
    for old_job in previous['jobs'][len(done):]:
        config = json.loads(Path(old_job['config_path']).read_text())
        folder = episodes_root / old_job['case_id']
        command = list(config['policy_command'])
        command[command.index('--decoded-evidence') + 1] = str(folder / 'decoded.jsonl')
        config.update(output_dir=str(folder), policy_command=command, frozen_files=dict(frozen))
        path = output / 'configs' / (old_job['case_id'] + '.json')
        write(path, config)
        jobs.append({**old_job, 'config_path': str(path), 'config_sha256': sha(path), 'result_path': str(folder / 'result.json')})
    frozen.update({job['config_path']: job['config_sha256'] for job in jobs})
    prefix_path = output / 'completed_prefix_receipt.json'
    plan = copy.deepcopy(previous)
    plan.update(jobs=jobs, frozen_files=frozen, episode_root=str(episodes_root),
                recovery={'admission': str(admission_path), 'admission_sha256': sha(admission_path),
                          'previous_plan': str(previous_plan), 'previous_plan_sha256': sha(previous_plan),
                          'completed_prefix_receipt': str(prefix_path), 'completed_prefix_cases': len(done),
                          'scope': 'CPU orchestration and uncompleted output paths only; one infrastructure attempt preserved; no model/task changes'})
    plan_path = output / 'plan.json'
    write(plan_path, plan)
    bootstrap = {'schema': 'lingbot_vla2_g1_pose_speed_evaluation_segment_v1', 'status': 'completed',
                 'plan': str(plan_path), 'plan_sha256': sha(plan_path), 'start_index': 0,
                 'stop_index_exclusive': len(done), 'previous_completed_cases': 0,
                 'jobs': copy.deepcopy(done), 'model_payloads_verified_before': True,
                 'model_payloads_verified_end': True, 'completed_total_cases': len(done),
                 'benchmark_complete': False, 'kind': 'rebound_previously_completed_audited_prefix',
                 'original_receipt': str(interrupted_receipt), 'original_receipt_sha256': sha(interrupted_receipt),
                 'original_plan_sha256': sha(previous_plan), 'budget_allocation': source_receipt['budget_allocation'],
                 'payloads_reverified_during_rebinding': True, 'new_gpu_execution': False,
                 'tool_sha256': sha(__file__)}
    write(prefix_path, bootstrap)
    if len(previous_records([prefix_path], sha(plan_path), plan)) != len(done):
        raise ValueError('Rebound prefix failed the production resume gate')
    load_plan(plan_path)
    for name, expected in frozen.items():
        path = Path(name)
        if path.suffix not in ['.py', '.json', '.usda', '.yaml'] or path.stat().st_size > 20 * 1024**2:
            continue
        target = output / 'source_snapshot' / name.lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target) != expected:
            raise ValueError('Dependency changed during recovery preparation')
    return {'plan': str(plan_path), 'plan_sha256': sha(plan_path), 'prefix_receipt': str(prefix_path),
            'prefix_receipt_sha256': sha(prefix_path), 'completed_prefix': len(done), 'remaining_cases': len(jobs) - len(done)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-plan', type=Path, required=True)
    parser.add_argument('--interrupted-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes-root', type=Path, required=True)
    parser.add_argument('--driver-source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.previous_plan, args.interrupted_receipt, args.output, args.episodes_root, args.driver_source)))
