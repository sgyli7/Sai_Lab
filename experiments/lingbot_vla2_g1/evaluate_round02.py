"""Finite FIFO turns per physical case; preserve every attempt and independent audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from experiments.vla_resources import gpu_queue
from .benchmark_pose_speed import load_plan, decoded_evidence
from .data import OWN, ROOT, sha
from .make_plan import artifact, write
from .budget_12h import admit_allocation, wall_remaining

OWNER = '01a145c6-0b86-7623-a869-086b33292199'
LEDGER = OWN / 'gpu_budget.json'
COORDINATION = ROOT / '.scratch/vla_gpu_coordination'


def persist(path, doc):
    temporary = path.with_name(path.name + '.incomplete')
    temporary.write_text(json.dumps(doc, indent=2, allow_nan=False))
    temporary.replace(path)


def raw_files(folder):
    return {str(path.absolute()): sha(path) for path in sorted(Path(folder).rglob('*')) if path.is_file()}


def verify_payloads(plan):
    for path, identity in plan['immutable_payloads'].items():
        if artifact(path, identity['sha256'])['bytes'] != identity['bytes']:
            raise ValueError('Immutable model bytes changed')


def previous_records(paths, plan_sha, plan):
    records = {}
    for path in paths:
        doc = json.loads(Path(path).read_text())
        if doc.get('schema') != 'lingbot_vla2_g1_pose_speed_evaluation_segment_v1' or doc['plan_sha256'] != plan_sha:
            raise ValueError('Resume receipt differs from frozen plan')
        if doc.get('model_payloads_verified_before') is not True or doc.get('model_payloads_verified_end') is not True:
            raise ValueError('Resume lacks both immutable-model phase boundary checks')
        for record in doc['jobs']:
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


def run(plan_path, receipt_path, max_jobs, resume_paths):
    from .acceptance_context import audit_case
    plan_path, receipt_path = Path(plan_path).absolute(), Path(receipt_path).absolute()
    if receipt_path.exists() or not receipt_path.is_relative_to(OWN) or max_jobs not in range(1, 4):
        raise ValueError('Fresh owned receipt and one to three cases per CPU segment required')
    plan = load_plan(plan_path)
    allocation = admit_allocation(LEDGER)
    storage = plan.get('storage_admission', {})
    if storage.get('schema') != 'lingbot_storage_admission_v2' or storage.get('reserve_bytes') != 24 * 1024**3 or storage.get('per_case_buffer_bytes') != 1024**3:
        raise ValueError('Explicit versioned24GiB storage admission required')
    plan_sha = sha(plan_path)
    records = previous_records(resume_paths, plan_sha, plan)
    start, stop = len(records), min(len(plan['jobs']), len(records) + max_jobs)
    if start == stop:
        raise ValueError('All planned cases already completed')
    verify_payloads(plan)
    receipt = {'schema': 'lingbot_vla2_g1_pose_speed_evaluation_segment_v1', 'status': 'prepared', 'plan': str(plan_path), 'plan_sha256': plan_sha, 'start_index': start, 'stop_index_exclusive': stop, 'previous_completed_cases': start, 'jobs': [], 'started_unix': time.time(), 'resume_receipts_sha256': {str(Path(path).absolute()): sha(path) for path in resume_paths}, 'tool_sha256': sha(__file__), 'model_payloads_verified_before': True, 'model_payloads_verified_end': False, 'budget_allocation': allocation, 'storage_admission': storage}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    persist(receipt_path, receipt)
    cancelled = False
    child = None
    def stop_requested(signum, frame):
        nonlocal cancelled
        cancelled = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    previous = {sig: signal.signal(sig, stop_requested) for sig in [signal.SIGINT, signal.SIGTERM]}
    tickets = {}
    try:
        # Both chats agreed to at most three cases per FIFO segment. Separate
        # owned budgets release all GPU resources between those finite turns.
        for index in range(start, stop):
            tickets[index] = gpu_queue.request(COORDINATION, OWNER, 'evaluation', plan['jobs'][index]['max_seconds'])
        receipt['reserved_tickets'] = {str(index): ticket['ticket'] for index, ticket in tickets.items()}
        persist(receipt_path, receipt)
        for index in range(start, stop):
            if cancelled:
                raise InterruptedError('Stopped at completed case boundary')
            load_plan(plan_path)
            if shutil.disk_usage(OWN).free < 25 * 1024**3:
                raise RuntimeError('Storage v2: keep24GiB free plus1GiB for the next complete case')
            own_bytes = sum(path.stat().st_size for path in OWN.rglob('*') if path.is_file())
            if own_bytes > 35 * 1024**3:
                raise RuntimeError('Keep one GiB headroom below the prospective own36GiB storage cap')
            job = plan['jobs'][index]
            wait_limit = wall_remaining() - job['max_seconds'] - 300
            if wait_limit <= 0:
                raise RuntimeError('Original wall window: reserve the full case plus five minutes for audit and delivery')
            config_path = Path(job['config_path'])
            config = json.loads(config_path.read_text())
            result_path = Path(job['result_path'])
            log_path = receipt_path.parent / ('case_' + str(index).zfill(2) + '_supervisor.log')
            audit_path = receipt_path.parent / ('case_' + str(index).zfill(2) + '_cpu_audit.json')
            if result_path.parent.exists() or log_path.exists() or audit_path.exists():
                raise FileExistsError('Preserve prior/partial case; no automatic overwrite or rerun')
            ticket = tickets[index]
            pending = {'index': index, 'policy': job['policy'], 'case_id': job['case_id'], 'status': 'waiting', 'fifo_ticket': ticket['ticket'], 'config_sha256': job['config_sha256'], 'result_path': str(result_path)}
            receipt['jobs'].append(pending)
            receipt['status'] = 'running'
            persist(receipt_path, receipt)
            py = str(OWN / 'venv/bin/python')
            supervisor = [py, '-m', 'experiments.lingbot_vla2_g1.budget_12h', '--ledger', str(LEDGER), '--stage', 'evaluation', '--max-seconds', str(job['max_seconds']), '--memory-high-gib', '80', '--memory-max-gib', '90', '--host-reserve-gib', '16', '--result-file', str(result_path), '--result-schema', 'pi05_g1_mobile_case_result_v1', '--', py, '-m', 'experiments.lingbot_vla2_g1.run_case_pose_speed', '--config', str(config_path)]
            command = [py, '-m', 'experiments.vla_resources.gpu_queue', 'run', '--ticket', ticket['ticket'], '--ledger', str(LEDGER), '--wait-seconds', str(wait_limit), '--', *supervisor]
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
                ledger = json.loads(LEDGER.read_text())
                row = ledger['runs'][queue_row['ledger_runs_before']]
                if row['status'] != 'completed' or not gpu_queue.cleanup_confirmed(row):
                    raise RuntimeError('Owned ledger did not complete and release all resources')
                audit = audit_case(config_path, decoded_checker=decoded_evidence)
                write(audit_path, audit)
                if audit.get('passed') is not True or audit.get('autonomous_policy_trial') is not True:
                    raise RuntimeError('Independent physical/action/video audit failed: ' + repr(audit.get('anomalies')))
                result = json.loads(result_path.read_text())
                pending.update(status='completed', success=result['success'], result_sha256=sha(result_path), case_files_sha256=raw_files(result_path.parent), cpu_audit_path=str(audit_path), cpu_audit_sha256=sha(audit_path), budget_row=row, supervisor_log=str(log_path), supervisor_log_sha256=sha(log_path))
                records[index] = pending
                print(json.dumps({'event': 'case_completed', 'index': index, 'case_id': job['case_id'], 'success': result['success'], 'owned_GPU_seconds': row['elapsed_seconds']}), flush=True)
                persist(receipt_path, receipt)
            except BaseException as error:
                pending.update(status='infrastructure_failed', error=repr(error))
                current = next(row for row in gpu_queue.status(COORDINATION)['tickets'] if row['ticket'] == ticket['ticket'])
                if current['status'] == 'waiting':
                    gpu_queue.cancel(COORDINATION, ticket['ticket'], OWNER)
                raise
        verify_payloads(plan)
        receipt.update(status='completed', model_payloads_verified_end=True, completed_total_cases=len(records), benchmark_complete=len(records) == len(plan['jobs']), ended_unix=time.time())
        persist(receipt_path, receipt)
        return receipt
    except BaseException as error:
        if child is None or child.poll() is not None:
            try:
                verify_payloads(plan)
                receipt['model_payloads_verified_end'] = True
            except BaseException as verification_error:
                receipt['model_integrity_error'] = repr(verification_error)
        receipt.update(status='infrastructure_failed', error=repr(error), completed_total_cases=len(records), benchmark_complete=False, ended_unix=time.time())
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
    print(json.dumps({k: result[k] for k in ['status', 'completed_total_cases', 'benchmark_complete']}))


if __name__ == '__main__':
    main()
