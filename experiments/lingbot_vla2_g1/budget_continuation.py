"""User-directed continuation with unchanged finite GPU limits and old charges.

The expired original wall window remains hashed historical evidence. The user's
explicit continuation replaces its stopping rule; it does not reset accounting.
"""
import argparse
import json
import math
from pathlib import Path
import sys

from experiments.vla_resources.gpu_queue import cleanup_confirmed
from . import budget
from .budget_allocated import runs_digest
from .data import OWN, sha

TOTAL = 43200
LIMITS = {'preflight': 3600, 'training': 10800, 'evaluation': 28800}


def admit_allocation(ledger_path):
    ledger_path = Path(ledger_path).absolute()
    if ledger_path != OWN / 'gpu_budget.json':
        raise ValueError('Independent LingBot ledger required')
    path = OWN / 'gpu_budget_allocation_user_resume_v4.json'
    allocation = json.loads(path.read_text())
    ledger = json.loads(ledger_path.read_text())
    if (allocation.get('schema') != 'lingbot_vla2_user_resume_allocation_v4'
            or allocation.get('total_limit_seconds') != TOTAL
            or allocation.get('new_stage_limits_seconds') != LIMITS
            or allocation.get('user_authorization') != '没成功你停下来个屁啊'
            or allocation.get('old_wall_deadline_enforced') is not False):
        raise ValueError('Explicit user continuation allocation identity differs')
    previous_path = Path(allocation['previous_ledger_snapshot'])
    if sha(previous_path) != allocation['previous_ledger_sha256']:
        raise ValueError('Historical ledger snapshot changed')
    previous = json.loads(previous_path.read_text())
    rows, count = ledger['runs'], allocation['preserved_run_count']
    if (type(count) is not int or count != len(previous['runs'])
            or len(rows) < count or rows[:count] != previous['runs']
            or runs_digest(rows[:count]) != allocation['preserved_runs_sha256']):
        raise ValueError('Historical actual GPU charges changed')
    wall_path = Path(allocation['previous_wall_window_path'])
    if wall_path != OWN / 'original_goal_wall_deadline.json' or sha(wall_path) != allocation['previous_wall_window_sha256']:
        raise ValueError('Historical wall-window evidence changed')
    if (ledger.get('total_limit_seconds') != TOTAL
            or ledger.get('stage_limits_seconds') != LIMITS
            or ledger.get('active_allocation_sha256') != sha(path)):
        raise ValueError('Ledger and continuation allocation differ')
    if any(not cleanup_confirmed(row) for row in rows):
        raise ValueError('An owned run remains unfinished')
    if any(type(row.get('elapsed_seconds')) not in (int, float)
           or not math.isfinite(row['elapsed_seconds']) or row['elapsed_seconds'] < 0
           or row.get('stage') not in LIMITS for row in rows):
        raise ValueError('Invalid historical actual GPU accounting')
    used = sum(row['elapsed_seconds'] for row in rows)
    if used > TOTAL or any(sum(row['elapsed_seconds'] for row in rows if row['stage'] == stage) > limit
                           for stage, limit in LIMITS.items()):
        raise ValueError('Actual usage exceeds unchanged GPU allocation')
    return {'path': str(path), 'sha256': sha(path), 'total_limit_seconds': TOTAL,
            'stage_limits_seconds': LIMITS, 'used_total_seconds': used,
            'old_wall_deadline_enforced': False}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    args, _ = parser.parse_known_args()
    prefix = sys.argv[1:sys.argv.index('--')] if '--' in sys.argv else sys.argv[1:]
    count = sum(value == '--max-seconds' or value.startswith('--max-seconds=') for value in prefix)
    if count != 1 or not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= TOTAL:
        raise ValueError('One positive finite supervisor duration required')
    allocation = admit_allocation(args.ledger)
    print(json.dumps({'event': 'user_continuation_admitted', 'allocation': allocation,
                      'finite_GPU_turn_seconds': args.max_seconds}), flush=True)
    budget.TOTAL, budget.LIMITS = TOTAL, dict(LIMITS)
    budget.main()


if __name__ == '__main__':
    main()
