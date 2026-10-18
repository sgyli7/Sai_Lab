"""Prospective stage transfer inside the existing finite sixteen-hour allocation.

Historical charges, the sixteen-hour v7 allocation, and its supervisor remain immutable.
Activation requires all owned work to be closed and no queued owned ticket.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import sys

from experiments.vla_resources.gpu_queue import cleanup_confirmed
from . import budget
from .budget_allocated import runs_digest
from .data import OWN, ROOT, sha

TOTAL = 57600
OWNER = '01a145c6-0b86-7623-a869-086b33292199'
STAGES = {'preflight', 'training', 'evaluation'}
ALLOCATION = 'gpu_budget_allocation_user_resume_v8.json'
SNAPSHOT = 'gpu_budget_before_reallocation_v8.json'


def valid_limits(limits):
    if (not isinstance(limits, dict) or set(limits) != STAGES
            or any(type(value) is not int or value <= 0 for value in limits.values())
            or sum(limits.values()) != TOTAL):
        raise ValueError('Positive integer stage limits must sum to finite57600seconds')


def valid_rows(rows, limits):
    if any(not cleanup_confirmed(row) for row in rows):
        raise ValueError('An owned run is unfinished')
    if any(row.get('stage') not in STAGES
           or type(row.get('elapsed_seconds')) not in (int, float)
           or not math.isfinite(row['elapsed_seconds']) or row['elapsed_seconds'] < 0 for row in rows):
        raise ValueError('Invalid historical GPU charges')
    used = sum(row['elapsed_seconds'] for row in rows)
    if used > TOTAL or any(sum(r['elapsed_seconds'] for r in rows if r['stage'] == stage) > limit
                           for stage, limit in limits.items()):
        raise ValueError('Historical charges exceed proposed allocation')
    return used


def admit_allocation(ledger_path):
    ledger_path = Path(ledger_path).absolute()
    if ledger_path != OWN / 'gpu_budget.json':
        raise ValueError('Independent LingBot ledger required')
    path = OWN / ALLOCATION
    allocation, ledger = [json.loads(p.read_text()) for p in (path, ledger_path)]
    limits = allocation['new_stage_limits_seconds']
    valid_limits(limits)
    if (allocation.get('schema') != 'lingbot_vla2_stage_reallocation_v8'
            or allocation.get('total_limit_seconds') != TOTAL
            or allocation.get('owner_thread') != OWNER
            or allocation.get('user_authorization') != '开始执行计划把， 直到目标达成'
            or allocation.get('old_wall_deadline_enforced') is not False
            or allocation.get('supervisor_sha256') != sha(__file__)):
        raise ValueError('Prospective continuation allocation identity differs')
    previous = OWN / SNAPSHOT
    old_allocation = OWN / 'gpu_budget_allocation_user_resume_v7.json'
    if (allocation.get('previous_ledger_snapshot') != str(previous)
            or allocation.get('previous_ledger_sha256') != sha(previous)
            or allocation.get('previous_allocation_path') != str(old_allocation)
            or allocation.get('previous_allocation_sha256') != sha(old_allocation)):
        raise ValueError('Preserved historical allocation changed')
    old = json.loads(previous.read_text())
    if (allocation.get('previous_total_limit_seconds') != old.get('total_limit_seconds')
            or old.get('total_limit_seconds') != TOTAL
            or allocation.get('finite_completion_continuation_under_latest_user_instruction') is not True
            or allocation.get('historical_charges_not_reset') is not True
            or allocation.get('total_GPU_limit_unchanged') is not True):
        raise ValueError('Same finite sixteen-hour total and preserved historical charges required')
    count = allocation['preserved_run_count']
    rows = ledger['runs']
    if (type(count) is not int or count != len(old['runs']) or rows[:count] != old['runs']
            or runs_digest(rows[:count]) != allocation['preserved_runs_sha256']):
        raise ValueError('Historical actual GPU charges changed')
    if (ledger.get('total_limit_seconds') != TOTAL or ledger.get('stage_limits_seconds') != limits
            or ledger.get('active_allocation_sha256') != sha(path)):
        raise ValueError('Ledger and prospective allocation differ')
    used = valid_rows(rows, limits)
    return {'path': str(path), 'sha256': sha(path), 'total_limit_seconds': TOTAL,
            'stage_limits_seconds': limits, 'used_total_seconds': used,
            'old_wall_deadline_enforced': False}


def save_fresh(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def activate(limits, reason):
    valid_limits(limits)
    if not reason.strip():
        raise ValueError('Concrete prospective reallocation reason required')
    from .budget_reallocated_v7 import admit_allocation as admit_v7
    ledger_path = OWN / 'gpu_budget.json'
    previous_admission = admit_v7(ledger_path)
    ledger = json.loads(ledger_path.read_text())
    used = valid_rows(ledger['runs'], limits)
    queue = json.loads((ROOT / '.scratch/vla_gpu_coordination/queue.json').read_text())
    if any(t['owner'] == OWNER and t['status'] in ('waiting', 'running') for t in queue['tickets']):
        raise ValueError('Close or cancel all owned tickets before changing stage admission')
    path, snapshot = OWN / ALLOCATION, OWN / SNAPSHOT
    if path.exists() or snapshot.exists():
        raise FileExistsError('Preserve prior reallocation evidence')
    save_fresh(snapshot, ledger)
    allocation = {'schema': 'lingbot_vla2_stage_reallocation_v8',
                  'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'owner_thread': OWNER, 'user_authorization': '开始执行计划把， 直到目标达成',
                  'total_limit_seconds': TOTAL, 'previous_total_limit_seconds': ledger['total_limit_seconds'],
                  'finite_completion_continuation_under_latest_user_instruction': True,
                  'historical_charges_not_reset': True, 'total_GPU_limit_unchanged': True,
                  'new_stage_limits_seconds': limits,
                  'previous_stage_limits_seconds': ledger['stage_limits_seconds'],
                  'previous_ledger_snapshot': str(snapshot), 'previous_ledger_sha256': sha(snapshot),
                  'previous_allocation_path': previous_admission['path'],
                  'previous_allocation_sha256': previous_admission['sha256'],
                  'preserved_run_count': len(ledger['runs']),
                  'preserved_runs_sha256': runs_digest(ledger['runs']),
                  'historical_used_GPU_seconds': used, 'reason': reason,
                  'old_wall_deadline_enforced': False, 'supervisor_sha256': sha(__file__),
                  'shared_FIFO_unchanged': True, 'memory_and_storage_limits_unchanged': True}
    save_fresh(path, allocation)
    ledger.update(total_limit_seconds=TOTAL, stage_limits_seconds=limits, active_allocation_sha256=sha(path))
    temporary = ledger_path.with_name(ledger_path.name + '.reallocation_v8')
    save_fresh(temporary, ledger)
    temporary.replace(ledger_path)
    return admit_allocation(ledger_path)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--preflight-seconds', type=int)
    parser.add_argument('--training-seconds', type=int)
    parser.add_argument('--evaluation-seconds', type=int)
    parser.add_argument('--reason', default='')
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--max-seconds', type=float)
    args, _ = parser.parse_known_args()
    if args.activate:
        result = activate({stage: getattr(args, stage + '_seconds') for stage in STAGES}, args.reason)
        print(json.dumps({'event': 'stage_reallocation_activated', 'allocation': result}))
        return
    prefix = sys.argv[1:sys.argv.index('--')] if '--' in sys.argv else sys.argv[1:]
    count = sum(value == '--max-seconds' or value.startswith('--max-seconds=') for value in prefix)
    if (args.ledger is None or count != 1 or args.max_seconds is None
            or not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= TOTAL):
        raise ValueError('One finite positive supervisor duration and owned ledger required')
    allocation = admit_allocation(args.ledger)
    print(json.dumps({'event': 'reallocated_continuation_admitted', 'allocation': allocation,
                      'finite_GPU_turn_seconds': args.max_seconds}), flush=True)
    budget.TOTAL, budget.LIMITS = TOTAL, dict(allocation['stage_limits_seconds'])
    budget.main()


if __name__ == '__main__':
    main()
