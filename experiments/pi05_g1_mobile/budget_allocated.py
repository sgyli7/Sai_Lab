"""Ledger-allocated12h mobile budget; mandatory typed completion stays unchanged.

--check-only is read-only CPU admission, no supervisor/systemd/queue/GPU calls.
Allocation caps are latched for one workload and restored in finally. A changed
allocation at workload return is rejected, never silently adopted mid-run.
"""
from __future__ import annotations
import argparse,datetime,fcntl,hashlib,json,math,sys
from pathlib import Path
from . import budget as legacy
from .quiescence import TERMINAL

TOTAL=43200
INITIAL={'preflight':7200,'training':21600,'evaluation':14400}
KEYS=set(INITIAL)


def number(value,*,positive=False):
    return type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0)


def caps(value):
    if not isinstance(value,dict) or set(value)!=KEYS or not all(number(x,positive=True) for x in value.values()) or sum(value.values())!=TOTAL:raise ValueError('Exactly3 positive finite stage caps summing43200 required')
    return dict(value)


def timestamp(value):
    parsed=datetime.datetime.fromisoformat(value)
    if parsed.tzinfo is None:raise ValueError('Timezone-aware accounting timestamp required')
    return parsed.timestamp()


def small(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>16*1024**2:raise ValueError('Small regular ledger/authority file required')
    raw=path.read_bytes();return json.loads(raw,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON:'+value))),hashlib.sha256(raw).hexdigest()


def allocation_fingerprint(document):
    fields={key:document.get(key) for key in ('schema','total_limit_seconds','stage_limits_seconds','authorization_receipt','allocation_amendments')}
    return hashlib.sha256(json.dumps(fields,sort_keys=True,allow_nan=False).encode()).hexdigest()


def validate_document(document,authority):
    if document.get('schema')!='pi05_g1_gpu_budget_v1' or document.get('total_limit_seconds')!=TOTAL:raise ValueError('Mobile ledger total/schema must remain43200')
    current=caps(document['stage_limits_seconds'])
    if authority.get('schema')!='pi05_g1_mobile_new_budget_authorization_v1' or authority.get('total_limit_seconds')!=TOTAL or caps(authority.get('stage_limits_seconds'))!=INITIAL:raise ValueError('Initial root12h authority differs')
    rows=document.get('runs')
    if not isinstance(rows,list):raise ValueError('Ledger runs list required')
    events=[];spent={key:0. for key in KEYS}
    for row in rows:
        if row.get('stage') not in KEYS or not number(row.get('elapsed_seconds')):raise ValueError('Finite nonnegative charged elapsed_seconds required')
        if row.get('status') not in TERMINAL or not row.get('ended_utc') or row.get('accounting_open') is True or row.get('launcher_cleanup_pending') is True or row.get('last_unit_properties',{}).get('ActiveState') not in ('inactive','failed') or row.get('docker_cleanup',{}).get('complete') is not True:raise ValueError('Every owned ledger row must be terminal and resources confirmed closed')
        start=timestamp(row['started_utc']);end=timestamp(row['ended_utc'])
        if end<start:raise ValueError('Accounting end before start')
        events.append((start,end,row));spent[row['stage']]+=row['elapsed_seconds']
    if sum(spent.values())>TOTAL or any(spent[key]>current[key] for key in KEYS):raise ValueError('Charged GPU usage exceeds total or stage cap')
    expected=dict(INITIAL);previous_time=timestamp(authority['accounting_start_utc']);proofs=[]
    amendments=document.get('allocation_amendments',[])
    if not isinstance(amendments,list):raise ValueError('Allocation amendment list required')
    for index,amendment in enumerate(amendments):
        if amendment.get('schema')!='pi05_g1_mobile_stage_allocation_amendment_v1' or amendment.get('total_limit_seconds_unchanged')!=TOTAL or caps(amendment.get('previous_stage_limits_seconds'))!=expected:raise ValueError('Broken allocation amendment chain/changed authority')
        new=caps(amendment['new_stage_limits_seconds']);when=timestamp(amendment['utc'])
        if when<previous_time:raise ValueError('Allocation amendments not chronological')
        if any(start<=when<end for start,end,row in events):raise ValueError('Allocation changed while an owned GPU run was active')
        prefix=[row for start,end,row in events if end<=when];consumed=sum(row['elapsed_seconds'] for row in prefix)
        if not number(amendment.get('consumed_seconds_unchanged')) or not math.isclose(consumed,amendment['consumed_seconds_unchanged'],rel_tol=0,abs_tol=1e-5):raise ValueError('Allocation amendment altered/excluded charged usage')
        prior=amendment.get('prior_ledger_sha256')
        if not isinstance(prior,str) or len(prior)!=64 or any(c not in '0123456789abcdef' for c in prior):raise ValueError('Prior ledger SHA binding missing/malformed')
        at_stage={key:sum(row['elapsed_seconds'] for row in prefix if row['stage']==key) for key in KEYS}
        if any(at_stage[key]>expected[key] or at_stage[key]>new[key] for key in KEYS):raise ValueError('Amendment stage cap below already charged usage')
        proofs.append({'index':index,'utc':amendment['utc'],'previous':expected,'new':new,'consumed_seconds':consumed,'prior_ledger_sha256':prior})
        expected=new;previous_time=when
    if expected!=current:raise ValueError('Latest amended caps differ from live ledger')
    return {'limits':current,'total_limit_seconds':TOTAL,'spent_by_stage':spent,'spent_total_seconds':sum(spent.values()),'remaining_by_stage':{key:min(TOTAL-sum(spent.values()),current[key]-spent[key]) for key in KEYS},'run_count':len(rows),'all_owned_rows_closed':True,'amendment_chain':proofs,'allocation_fingerprint':allocation_fingerprint(document),'prior_sha_scope':'Prior ledger SHA strings validated structurally; historical backup bytes are not inferred/scanned. Allocation continuity,UTCclosedness,andunchangedcharge are checked against all live ledger rows.'}


def validate_ledger(path):
    """Pure CPU readonly admission; raises ValueError for any invalid allocation."""
    path=Path(path).resolve()
    if not path.is_relative_to(legacy.LEDGER_ROOT.resolve()):raise ValueError('Only this mobile ledger root allowed')
    lock_path=path.with_suffix('.lock')
    if not lock_path.is_file() or lock_path.is_symlink():raise ValueError('Existing ledger lock required;readonly check creates nothing')
    try:
        with lock_path.open('rb') as lock:
            fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            document,digest=small(path);authority_path=Path(document['authorization_receipt']).resolve()
            if not authority_path.is_relative_to(legacy.LEDGER_ROOT.resolve()):raise ValueError('Mobile authority path escaped round')
            authority,authority_sha=small(authority_path);report=validate_document(document,authority)
            if small(path)[1]!=digest:raise ValueError('Ledger changed during readonly admission')
    except BlockingIOError as error:raise ValueError('Owned ledger busy;do not admit another GPU workload') from error
    report.update(schema='pi05_g1_mobile_allocated_budget_cpu_check_v1',status='verified',ledger=str(path),ledger_sha256=digest,authorization_receipt=str(authority_path),authorization_sha256=authority_sha,no_gpu_systemd_or_queue_calls=True)
    return report


def invoke(argv):
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--ledger',required=True)
    own,_=parser.parse_known_args(argv);path=Path(own.ledger)
    admitted=validate_ledger(path);old_limits=legacy.LIMITS;old_total=legacy.TOTAL
    try:
        # Only this process's compatibility binding; no source or ledger edits.
        legacy.LIMITS=dict(admitted['limits']);legacy.TOTAL=TOTAL
        print('MOBILE_ALLOCATED_ADMISSION',json.dumps(admitted,allow_nan=False),flush=True)
        # Result-file/schema validation and owned resource cleanup remain in the
        # existing mobile wrapper; do not duplicate/weaken its mandatory contract.
        legacy.main(argv)
    finally:
        legacy.LIMITS,legacy.TOTAL=old_limits,old_total
        after=validate_ledger(path)
        if after['allocation_fingerprint']!=admitted['allocation_fingerprint'] or after['authorization_sha256']!=admitted['authorization_sha256']:raise ValueError('Allocation/authority changed during workload;cannot claim normal allocated completion')


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if '--check-only' in args:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-only',action='store_true');p.add_argument('--ledger',required=True);a=p.parse_args(args)
        print(json.dumps(validate_ledger(a.ledger),indent=2,allow_nan=False));return
    invoke(args)

if __name__=='__main__':main()
