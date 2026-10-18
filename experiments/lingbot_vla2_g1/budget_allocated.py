"""Same8h total, audited1h/2h/5h allocation after completed first-round training."""
import argparse
import hashlib
import json
from pathlib import Path
from experiments.vla_resources.gpu_queue import cleanup_confirmed
from .data import OWN, sha
from . import budget

LIMITS = {'preflight': 3600, 'training': 7200, 'evaluation': 18000}


def runs_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()


def admit_allocation(ledger_path):
    ledger_path = Path(ledger_path).absolute()
    if ledger_path != OWN / 'gpu_budget.json':
        raise ValueError('This allocation belongs to the independent LingBot ledger')
    allocation_path = OWN / 'gpu_budget_allocation_v2.json'
    allocation = json.loads(allocation_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    identity = sha(allocation_path)
    if allocation.get('schema') != 'lingbot_vla2_budget_reallocation_v1' or allocation.get('total_limit_seconds') != budget.TOTAL or allocation.get('new_stage_limits_seconds') != LIMITS:
        raise ValueError('Unsupported allocation or increased total allowance')
    if ledger.get('total_limit_seconds') != budget.TOTAL or ledger.get('stage_limits_seconds') != LIMITS or ledger.get('active_allocation_sha256') != identity:
        raise ValueError('Ledger and audited allocation differ')
    rows = ledger['runs']
    prefix = allocation['preserved_run_count']
    if len(rows) < prefix or runs_digest(rows[:prefix]) != allocation['preserved_runs_sha256']:
        raise ValueError('Prior actual GPU charges were changed')
    if any(not cleanup_confirmed(row) for row in rows):
        raise ValueError('An owned run is unfinished')
    total = sum(row['elapsed_seconds'] for row in rows)
    if total > budget.TOTAL or any(sum(row['elapsed_seconds'] for row in rows if row['stage'] == stage) > limit for stage, limit in LIMITS.items()):
        raise ValueError('Actual charges exceed the unchanged allocated allowance')
    return {'path': str(allocation_path), 'sha256': identity, 'total_limit_seconds': budget.TOTAL, 'stage_limits_seconds': LIMITS, 'used_total_seconds': total}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ledger', type=Path, required=True)
    args, _ = parser.parse_known_args()
    admit_allocation(args.ledger)
    budget.LIMITS = dict(LIMITS)
    budget.main()


if __name__ == '__main__':
    main()
