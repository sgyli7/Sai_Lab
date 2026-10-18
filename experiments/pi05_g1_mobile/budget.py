"""Twelve-hour owned mobile budget with a mandatory producer completion contract.

Resource/process accounting reuses the original supervisor unchanged. Its raw
exit status is preserved; a zero systemd-run status alone cannot prove workload
completion. Every GPU command must produce a new status=completed JSON receipt.
CPU-only tools do not require this wrapper.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from experiments.pi05_g1 import budget as supervisor

ROOT=Path(__file__).resolve().parents[2]
LEDGER_ROOT=ROOT/'.scratch/pi05_g1/round03_mobile'
LIMITS={'preflight':5400,'training':16200,'evaluation':21600}
TOTAL=43200


@contextmanager
def ledger_lock(path):
    with Path(path).with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def remaining_seconds(ledger,stage):
    """Includes carried shutdown overlap and every failed owned run."""
    return min(TOTAL-sum(row['elapsed_seconds'] for row in ledger['runs']),LIMITS[stage]-sum(row['elapsed_seconds'] for row in ledger['runs'] if row['stage']==stage))


def completion_contract(path,schema,started_ns):
    path=Path(path);receipt={'path':str(path.resolve()),'expected_schema':schema,'valid':False}
    try:
        if not path.is_file() or path.is_symlink():raise ValueError('Producer completion file missing or symlinked')
        stat=path.stat()
        if stat.st_size>4*1024**2:raise ValueError('Producer completion receipt exceeds4MiB')
        if stat.st_mtime_ns<started_ns:raise ValueError('Producer completion receipt is stale')
        raw=path.read_bytes();doc=json.loads(raw,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))
        if not isinstance(doc,dict) or doc.get('schema')!=schema:raise ValueError('Producer completion schema mismatch')
        if doc.get('status')!='completed':raise ValueError('Producer did not declare true status=completed')
        receipt.update(valid=True,schema=doc['schema'],status=doc['status'],sha256=hashlib.sha256(raw).hexdigest(),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        if 'success' in doc:receipt['task_success']=doc['success']
    except (OSError,ValueError,TypeError) as error:receipt['error']=str(error)
    return receipt


def finalize_workload(ledger_path,row_index,contract,raw_code):
    """Only mutate this new round's row after its owned resources are closed."""
    ledger_path=Path(ledger_path)
    if not ledger_path.resolve().is_relative_to(LEDGER_ROOT.resolve()):raise ValueError('Refusing to update a ledger outside this mobile round')
    with ledger_lock(ledger_path):
        ledger=json.loads(ledger_path.read_text())
        if len(ledger['runs'])!=row_index+1:raise RuntimeError('Owned ledger row identity changed')
        row=ledger['runs'][row_index]
        if row.get('supervisor_pid')!=os.getpid():raise RuntimeError('Refusing to derive status for a foreign supervisor row')
        state=row.get('last_unit_properties',{}).get('ActiveState')
        if row.get('accounting_open') or state not in ('inactive','failed') or row.get('docker_cleanup',{}).get('complete') is False:raise RuntimeError('Resources not confirmed closed; preserve raw cleanup/accounting state')
        row['raw_supervisor_status']=row['status'];row['raw_supervisor_exit_code']=row.get('exit_code',raw_code)
        row['workload_completion_contract']=contract
        code=raw_code
        if not contract['valid']:
            row['status']='result_contract_failed';code=2
        elif raw_code!=0:
            code=raw_code  # A valid producer receipt cannot forgive process failure.
        row['exit_code']=code;row['process_exit_is_not_producer_completion']=True
        temporary=ledger_path.with_suffix('.workload.tmp');temporary.write_text(json.dumps(ledger,indent=2,allow_nan=False));os.replace(temporary,ledger_path)
        return code


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--result-file',type=Path,required=True)
    parser.add_argument('--result-schema',required=True)
    # Pass all stage/memory/duration/command flags directly to the unchanged
    # supervisor. Our flags must appear before the command's -- separator.
    own,rest=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    ledger=own.ledger.resolve();result=own.result_file.resolve()
    if sum(LIMITS.values())!=TOTAL:raise RuntimeError('Stage allocation must equal12h')
    if not ledger.is_relative_to(LEDGER_ROOT.resolve()) or not result.is_relative_to(LEDGER_ROOT.resolve()):raise ValueError('Mobile ledger and completion evidence must stay inside round03_mobile')
    if own.result_file.exists() or own.result_file.is_symlink():raise FileExistsError('Completion file must not exist before workload launch')
    with ledger_lock(ledger):
        existing=json.loads(ledger.read_text())
        if existing.get('total_limit_seconds')!=TOTAL or existing.get('stage_limits_seconds')!=LIMITS:raise ValueError('New mobile authority/ledger allocation mismatch')
        index=len(existing['runs'])
    started_ns=time.time_ns();old_limits,old_total=supervisor.LIMITS,supervisor.TOTAL;old_argv=sys.argv
    raw_code=1
    try:
        supervisor.LIMITS=dict(LIMITS);supervisor.TOTAL=TOTAL
        sys.argv=[old_argv[0],'--ledger',str(ledger),*rest]
        try:supervisor.main()
        except SystemExit as error:
            raw_code=error.code if isinstance(error.code,int) else (0 if error.code is None else 1)
    finally:
        supervisor.LIMITS,supervisor.TOTAL=old_limits,old_total;sys.argv=old_argv
    contract=completion_contract(result,own.result_schema,started_ns)
    code=finalize_workload(ledger,index,contract,raw_code)
    print('MOBILE_WORKLOAD_COMPLETION',json.dumps({'exit_code':code,'contract':contract}),flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
