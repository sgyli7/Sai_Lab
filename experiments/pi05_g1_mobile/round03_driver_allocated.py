"""Finite legacy stage driver with reviewed ledger-allocation-aware supervisor.

Only GPU command substitution is budget -> budget_allocated. Legacy typed result
predicates, fresh outputs, FIFO cancellation, accounting and CPU cleanup stay in
round03_driver. New source pins and allocation are checked before FIFO requests.
python -m experiments.pi05_g1_mobile.round03_driver_allocated --plan PLAN \
    --output NEW_JOURNAL [--dry-run]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import threading
from . import round03_driver as legacy

_OLD_MODULE='experiments.pi05_g1_mobile.budget'
_NEW_MODULE='experiments.pi05_g1_mobile.budget_allocated'
_ORIGINAL_COMMAND=legacy.budget_command
_ORIGINAL_VALIDATE=legacy.validate_plan
_BIND_LOCK=threading.Lock()


def budget_command(plan,stage):
    command=_ORIGINAL_COMMAND(plan,stage)
    if len(command)<3 or command[1:3]!=['-m',_OLD_MODULE]:
        raise ValueError('Legacy budget command identity changed')
    return [*command[:2],_NEW_MODULE,*command[3:]]


def validate_allocated_plan(plan):
    """CPU only; no queue operations, workload imports or model payload hashes."""
    _ORIGINAL_VALIDATE(plan)
    for path in (Path(__file__).resolve(),Path(__file__).with_name('budget_allocated.py').resolve()):
        if plan.get('frozen_files',{}).get(str(path))!=legacy.sha(path):
            raise ValueError('Both allocated-driver and allocated-budget sources require exact plan frozen SHA: '+str(path))
    from . import budget_allocated
    # Public CPU API supplied by budget_allocated, checked again at every request.
    return budget_allocated.validate_ledger(Path(plan['ledger']))


class _ValidatedQueue:
    def __init__(self,queue,plan):self.queue=queue;self.plan=plan
    def request(self,*args,**kwargs):
        validate_allocated_plan(self.plan)
        return self.queue.request(*args,**kwargs)
    def run(self,*args,**kwargs):return self.queue.run(*args,**kwargs)
    def snapshot(self,*args,**kwargs):return self.queue.snapshot(*args,**kwargs)
    def cancel(self,*args,**kwargs):return self.queue.cancel(*args,**kwargs)


def execute(plan_path,output,*,dry_run=False,queue=None,cpu_runner=legacy.cpu_execute):
    """Bind only during this foreground execution; always restore legacy globals."""
    if not _BIND_LOCK.acquire(blocking=False):raise RuntimeError('Allocated driver already executing in this process')
    original_command=legacy.budget_command;original_validate=legacy.validate_plan
    try:
        if original_command is not _ORIGINAL_COMMAND or original_validate is not _ORIGINAL_VALIDATE:
            raise RuntimeError('Refusing unexpected prior legacy driver bindings')
        plan=legacy.read(plan_path)
        validate_allocated_plan(plan)
        def checked_validate(document):
            if document!=plan:raise ValueError('Plan changed before delegated driver validation')
            validate_allocated_plan(document)
            return document
        legacy.budget_command=budget_command
        legacy.validate_plan=checked_validate
        wrapped=None if dry_run else _ValidatedQueue(queue if queue is not None else legacy.QueueAPI(plan['coord_root']),plan)
        return legacy.execute(plan_path,output,dry_run=dry_run,queue=wrapped,cpu_runner=cpu_runner)
    finally:
        legacy.budget_command=original_command;legacy.validate_plan=original_validate
        _BIND_LOCK.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True);parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();result=execute(args.plan,args.output,dry_run=args.dry_run)
    print(json.dumps({'status':result['status'],'output':args.output}))
    return 0 if result['status'] in ('completed','dry_run') else 2

if __name__=='__main__':raise SystemExit(main())
