"""Prospective user-authorized twelve-hour allocation; retain every earlier charge."""
import argparse
import json
from pathlib import Path
import sys
import time

from experiments.vla_resources.gpu_queue import cleanup_confirmed
from . import budget
from .budget_allocated import runs_digest
from .data import OWN, sha

TOTAL = 43200
LIMITS = {'preflight': 3600, 'training': 10800, 'evaluation': 28800}


def wall_remaining():
    path = OWN / 'original_goal_wall_deadline.json'
    timing = json.loads(path.read_text())
    if (timing.get('schema') != 'lingbot_original_goal_wall_deadline_v1'
            or timing.get('goal_created_unix') != 1792171987
            or timing.get('wall_limit_seconds') != TOTAL
            or timing.get('wall_deadline_unix') != 1792171987 + TOTAL
            or timing.get('queue_wait_counts_toward_wall') is not True):
        raise ValueError('Original goal wall window changed')
    return max(0., timing['wall_deadline_unix'] - time.time() - 10.)


def admit_allocation(ledger_path):
    ledger_path = Path(ledger_path).absolute()
    if ledger_path != OWN / 'gpu_budget.json':
        raise ValueError('Independent LingBot ledger required')
    path = OWN / 'gpu_budget_allocation_12h_v3.json'
    allocation = json.loads(path.read_text())
    ledger = json.loads(ledger_path.read_text())
    if (allocation.get('schema') != 'lingbot_vla2_user_12h_allocation_v1'
            or allocation.get('total_limit_seconds') != TOTAL
            or allocation.get('new_stage_limits_seconds') != LIMITS
            or allocation.get('user_authorization') != '我也给你 12各小时时间 ，加油 做出最好成绩'):
        raise ValueError('Explicit twelve-hour allocation identity differs')
    previous_path = Path(allocation['previous_ledger_snapshot'])
    if sha(previous_path) != allocation['previous_ledger_sha256']:
        raise ValueError('Historical ledger snapshot changed')
    previous = json.loads(previous_path.read_text())
    count = allocation['preserved_run_count']
    rows = ledger['runs']
    if (len(previous['runs']) != count or len(rows) < count
            or rows[:count] != previous['runs']
            or runs_digest(rows[:count]) != allocation['preserved_runs_sha256']):
        raise ValueError('Historical actual GPU charges changed')
    if (ledger.get('total_limit_seconds') != TOTAL
            or ledger.get('stage_limits_seconds') != LIMITS
            or ledger.get('active_allocation_sha256') != sha(path)):
        raise ValueError('Ledger and allocation differ')
    if any(not cleanup_confirmed(row) for row in rows):
        raise ValueError('An owned run remains unfinished')
    total = sum(row['elapsed_seconds'] for row in rows)
    if total > TOTAL or any(sum(row['elapsed_seconds'] for row in rows
                               if row['stage'] == stage) > limit
                            for stage, limit in LIMITS.items()):
        raise ValueError('Actual usage exceeds allocation')
    return {'path': str(path), 'sha256': sha(path),
            'total_limit_seconds': TOTAL, 'stage_limits_seconds': LIMITS,
            'used_total_seconds': total}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    args, _ = parser.parse_known_args()
    admit_allocation(args.ledger)
    remaining = wall_remaining()
    if remaining < 1:
        raise RuntimeError('Original twelve-hour wall window exhausted')
    duration = min(args.max_seconds, remaining)
    if sys.argv.count('--max-seconds') != 1:
        raise ValueError('One explicit finite supervisor duration required')
    sys.argv[sys.argv.index('--max-seconds') + 1] = str(duration)
    print(json.dumps({'event': 'original_wall_window_admitted',
                      'remaining_wall_seconds': remaining,
                      'allowed_GPU_turn_seconds': duration}), flush=True)
    budget.TOTAL = TOTAL
    budget.LIMITS = dict(LIMITS)
    budget.main()


if __name__ == '__main__':
    main()
