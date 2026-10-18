"""Finite continuation turns for the unchanged formal cohort.

Allocation and the case supervisor both use budget_reallocated_v8. Obsolete own
caps and the expired wall cutoff are not admission rules.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time

from experiments.vla_resources import gpu_queue
from experiments.lingbot_vla2_g1.budget_reallocated_v8 import admit_allocation
from experiments.lingbot_vla2_g1.data import OWN, ROOT, sha
from .gates import (
    BACKUP, CASE_SECONDS, COMBINED_CAP, FREE_BUFFER, GLOBAL_FREE, HEADROOM, OWNER,
    admit_storage, assert_result_destination, assert_seven_checks, load_formal_plan, recheck_turns)

LEDGER = OWN / 'gpu_budget.json'
COORDINATION = ROOT / '.scratch/vla_gpu_coordination'


def persist(path, document):
    temporary = path.with_name(path.name + '.incomplete')
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False))
    temporary.replace(path)


def raw_files(folder):
    return {str(path.absolute()): sha(path) for path in sorted(Path(folder).rglob('*')) if path.is_file()}


def verify_payloads(plan):
    payloads = plan.get('immutable_payloads') or {}
    if len(payloads) != 7:
        raise ValueError('All seven model payloads required')
    from experiments.lingbot_vla2_g1.make_plan import artifact
    for path, identity in payloads.items():
        if artifact(path, identity['sha256'])['bytes'] != identity['bytes']:
            raise ValueError('Immutable model bytes changed')


def verify_snapshot(plan_path):
    manifest_path = Path(plan_path).parent / 'source_snapshot_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'completed' or manifest.get('before_first_physical_case') is not True or manifest.get('plan_sha256') != sha(plan_path):
        raise ValueError('Source snapshot must exist before the first physical case')
    for name, digest in manifest['copied_files_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Snapshotted source changed: ' + name)


def combined_bytes():
    return sum(path.stat().st_size for folder in (OWN, BACKUP) for path in folder.rglob('*') if path.is_file())


def assert_storage_room():
    if combined_bytes() + HEADROOM > COMBINED_CAP:
        raise RuntimeError('Combined 64GiB cap requires 1GiB next-case headroom')
    if shutil.disk_usage(OWN).free < GLOBAL_FREE + FREE_BUFFER:
        raise RuntimeError('Global free 24GiB plus 1GiB buffer required')


def previous_records(paths, plan_sha, plan):
    records = {}
    for path in paths:
        document = json.loads(Path(path).read_text())
        if document.get('schema') != 'lingbot_vla2_g1_pose_speed_evaluation_segment_v1' or document.get('plan_sha256') != plan_sha:
            raise ValueError('Resume receipt differs from frozen plan')
        if document.get('model_payloads_verified_before') is not True or document.get('model_payloads_verified_end') is not True:
            raise ValueError('Resume lacks both immutable-model phase boundary checks')
        for record in document['jobs']:
            if record.get('status') != 'completed':
                continue
            index = record['index']
            if not 0 <= index < len(plan['jobs']) or (record['policy'], record['case_id']) != (plan['jobs'][index]['policy'], plan['jobs'][index]['case_id']):
                raise ValueError('Resume case identity differs')
            if index in records and records[index] != record:
                raise ValueError('Conflicting completed resume records')
            if raw_files(Path(plan['jobs'][index]['result_path']).parent) != record['case_files_sha256']:
                raise ValueError('Completed raw evidence inventory changed')
            if sha(record['cpu_audit_path']) != record['cpu_audit_sha256']:
                raise ValueError('Completed audit changed')
            records[index] = record
    if set(records) != set(range(len(records))):
        raise ValueError('Only a contiguous frozen-order completed prefix may resume')
    return records


def reservation(job):
    if job.get('max_seconds') != CASE_SECONDS:
        raise ValueError('Each case requires a separate finite 900 second budget')
    return (COORDINATION, OWNER, 'evaluation', CASE_SECONDS)


def supervisor_argv(result_path, config_path):
    py = str(OWN / 'venv/bin/python')
    return [py, '-m', 'experiments.lingbot_vla2_g1.budget_reallocated_v8', '--ledger', str(LEDGER),
            '--stage', 'evaluation', '--max-seconds', str(CASE_SECONDS), '--memory-high-gib', '80',
            '--memory-max-gib', '90', '--host-reserve-gib', '16', '--result-file', str(result_path),
            '--result-schema', 'pi05_g1_mobile_case_result_v1', '--', py, '-m',
            'experiments.lingbot_vla2_g1.run_case_with_backup', '--config', str(config_path)]


def policy_decoder():
    from experiments.lingbot_vla2_g1.diagnostics.development_audit_v2 import decoded_evidence
    return decoded_evidence


def run(plan_path, receipt_path, max_jobs, resume_paths):
    plan_path, receipt_path = Path(plan_path).absolute(), Path(receipt_path).absolute()
    if receipt_path.exists() or not receipt_path.is_relative_to(OWN) or max_jobs not in range(1, 4):
        raise ValueError('Fresh owned receipt and one to three cases per CPU segment required')
    assert_result_destination(receipt_path)
    plan = load_formal_plan(plan_path)
    admit_storage(plan.get('storage_admission'))
    plan_sha = sha(plan_path)
    records = previous_records(resume_paths, plan_sha, plan)
    start, stop = len(records), min(len(plan['jobs']), len(records) + max_jobs)
    if start == stop:
        raise ValueError('All planned cases already completed')
    allocation = admit_allocation(LEDGER)
    ledger = json.loads(LEDGER.read_text())
    recheck_turns(allocation, ledger, stop - start)
    assert_storage_room()
    verify_payloads(plan)
    verify_snapshot(plan_path)
    receipt = {'schema': 'lingbot_vla2_g1_pose_speed_evaluation_segment_v1', 'status': 'prepared',
               'plan': str(plan_path), 'plan_sha256': plan_sha, 'start_index': start,
               'stop_index_exclusive': stop, 'previous_completed_cases': start, 'jobs': [],
               'started_unix': time.time(), 'resume_receipts_sha256': {str(Path(path).absolute()): sha(path) for path in resume_paths},
               'tool_sha256': sha(__file__), 'model_payloads_verified_before': True, 'model_payloads_verified_end': False,
               'budget_allocation': allocation, 'storage_admission': plan['storage_admission'],
               'budget_policy': 'budget_reallocated_v8', 'wall_deadline_enforced': False}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    persist(receipt_path, receipt)
    cancelled, child, tickets = False, None, {}

    def stop_requested(signum, frame):
        nonlocal cancelled
        cancelled = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    previous = {sig: signal.signal(sig, stop_requested) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for index in range(start, stop):
            tickets[index] = gpu_queue.request(*reservation(plan['jobs'][index]))
        receipt['reserved_tickets'] = {str(index): ticket['ticket'] for index, ticket in tickets.items()}
        persist(receipt_path, receipt)
        for index in range(start, stop):
            if cancelled:
                raise InterruptedError('Stopped at completed case boundary')
            load_formal_plan(plan_path)
            allocation = admit_allocation(LEDGER)
            recheck_turns(allocation, json.loads(LEDGER.read_text()), 1)
            assert_storage_room()
            job = plan['jobs'][index]
            config_path, result_path = Path(job['config_path']), Path(job['result_path'])
            log_path = receipt_path.parent / ('case_' + str(index).zfill(2) + '_supervisor.log')
            audit_path = receipt_path.parent / ('case_' + str(index).zfill(2) + '_cpu_audit.json')
            if result_path.parent.exists() or log_path.exists() or audit_path.exists():
                raise FileExistsError('Preserve prior/partial case; no automatic overwrite or rerun')
            ticket = tickets[index]
            pending = {'index': index, 'policy': job['policy'], 'case_id': job['case_id'], 'status': 'waiting',
                       'fifo_ticket': ticket['ticket'], 'config_sha256': job['config_sha256'], 'result_path': str(result_path)}
            receipt['jobs'].append(pending)
            receipt['status'] = 'running'
            persist(receipt_path, receipt)
            py = str(OWN / 'venv/bin/python')
            command = [py, '-m', 'experiments.vla_resources.gpu_queue', 'run', '--ticket', ticket['ticket'],
                       '--ledger', str(LEDGER), '--', *supervisor_argv(result_path, config_path)]
            print(json.dumps({'event': 'case_queued', **pending}), flush=True)
            try:
                with log_path.open('x') as stream:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    code = child.wait()
                child = None
                if code:
                    raise RuntimeError('Owned case supervisor exit ' + str(code))
                queue_row = next(row for row in gpu_queue.status(COORDINATION)['tickets'] if row['ticket'] == ticket['ticket'])
                if queue_row['status'] != 'completed' or queue_row.get('cleanup_confirmed') is not True:
                    raise RuntimeError('FIFO cleanup was not confirmed')
                row = json.loads(LEDGER.read_text())['runs'][queue_row['ledger_runs_before']]
                if row['status'] != 'completed' or not gpu_queue.cleanup_confirmed(row):
                    raise RuntimeError('Owned ledger did not complete and release all resources')
                from experiments.lingbot_vla2_g1.acceptance_context import audit_case
                from experiments.lingbot_vla2_g1.make_plan import write
                audit = audit_case(config_path, decoded_checker=policy_decoder())
                write(audit_path, audit)
                assert_seven_checks(audit)
                if audit.get('result_path') != str(result_path) or audit.get('case_id') != job['case_id']:
                    raise RuntimeError('Independent physical/action/video audit failed')
                case_result = json.loads(result_path.read_text())
                pending.update(status='completed', success=case_result['success'], result_sha256=sha(result_path),
                               case_files_sha256=raw_files(result_path.parent), cpu_audit_path=str(audit_path),
                               cpu_audit_sha256=sha(audit_path), budget_row=row, supervisor_log=str(log_path),
                               supervisor_log_sha256=sha(log_path))
                records[index] = pending
                print(json.dumps({'event': 'case_completed', 'index': index, 'case_id': job['case_id'],
                                  'success': case_result['success'], 'owned_GPU_seconds': row['elapsed_seconds']}), flush=True)
                persist(receipt_path, receipt)
            except BaseException as error:
                pending.update(status='infrastructure_failed', error=repr(error))
                current = next(row for row in gpu_queue.status(COORDINATION)['tickets'] if row['ticket'] == ticket['ticket'])
                if current['status'] == 'waiting':
                    gpu_queue.cancel(COORDINATION, ticket['ticket'], OWNER)
                raise
        verify_payloads(plan)
        receipt.update(status='completed', model_payloads_verified_end=True, completed_total_cases=len(records),
                       benchmark_complete=len(records) == len(plan['jobs']), ended_unix=time.time())
        persist(receipt_path, receipt)
        return receipt
    except BaseException as error:
        if child is None or child.poll() is not None:
            try:
                verify_payloads(plan)
                receipt['model_payloads_verified_end'] = True
            except BaseException as verification_error:
                receipt['model_integrity_error'] = repr(verification_error)
        receipt.update(status='infrastructure_failed', error=repr(error), completed_total_cases=len(records),
                       benchmark_complete=False, ended_unix=time.time())
        persist(receipt_path, receipt)
        raise
    finally:
        queue = gpu_queue.status(COORDINATION)
        for ticket in tickets.values():
            current = next(row for row in queue['tickets'] if row['ticket'] == ticket['ticket'])
            if current['status'] == 'waiting':
                gpu_queue.cancel(COORDINATION, ticket['ticket'], OWNER)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--max-jobs', type=int, default=3)
    parser.add_argument('--resume-receipt', type=Path, action='append', default=[])
    args = parser.parse_args()
    result = run(args.plan, args.receipt, args.max_jobs, args.resume_receipt)
    print(json.dumps({key: result[key] for key in ('status', 'completed_total_cases', 'benchmark_complete')}))


if __name__ == '__main__':
    main()
