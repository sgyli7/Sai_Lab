"""Finite root-reviewed CPU/GPU stage driver. No retries, shell, or inferred stages.

Dry-run never accesses the real FIFO. GPU execution always wraps the existing
mobile budget in the shared queue; queue waiting does not charge GPU ledger.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,math,os,signal,subprocess,time
from pathlib import Path

SCHEMA='pi05_g1_mobile_explicit_stage_plan_v1'
ROOT=Path(__file__).resolve().parents[2]
MOBILE=ROOT/'.scratch/pi05_g1/round03_mobile'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.incomplete')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def read(path):return json.loads(Path(path).read_text())


def finite_positive(value):return type(value) in (int,float) and math.isfinite(value) and value>0


def argv(value):
    if not isinstance(value,list) or not value or any(not isinstance(x,str) or not x or '\x00' in x for x in value):raise ValueError('Explicit nonempty string argv array required')
    return value



def verify_frozen(files):
    """Verify the reviewed map verbatim; never refresh an expected digest."""
    if not isinstance(files,dict):raise ValueError('frozen_files must be an absolute-path toSHA map')
    for name,digest in files.items():
        path=Path(name)
        if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size>16*1024**2 or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid frozen small-file identity:'+name)
        if sha(path)!=digest:raise ValueError('Root-reviewed frozen input changed:'+name)


def snapshot_frozen(files,output):
    folder=output/'frozen_snapshot';folder.mkdir();records=[]
    for index,(name,digest) in enumerate(sorted(files.items())):
        source=Path(name);payload=source.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Frozen input changed during snapshot:'+name)
        target=folder/f'{index:04d}_{source.name}'
        with target.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        records.append({'source':name,'sha256':digest,'snapshot':str(target)})
    atomic(folder/'manifest.json',{'schema':'pi05_g1_mobile_driver_frozen_snapshot_v1','files':records})


def validate_plan(plan):
    if plan.get('schema')!=SCHEMA or plan.get('root_reviewed') is not True:raise ValueError('Root-reviewed explicit plan required')
    if not isinstance(plan.get('owner'),str) or not plan['owner']:raise ValueError('Explicit ROOT owner thread ID required')
    for key in ('coord_root','ledger','supervisor_python'):
        if not Path(plan[key]).is_absolute():raise ValueError('Absolute '+key+' required')
    if Path(plan['ledger']).resolve()!=MOBILE/'gpu_budget.json':raise ValueError('Only new mobile ledger allowed')
    verify_frozen(plan.get('frozen_files',{}))
    stages=plan['stages']
    if not isinstance(stages,list) or not 1<=len(stages)<=64:raise ValueError('Finite1..64 explicit stages required')
    if len({x['id'] for x in stages})!=len(stages):raise ValueError('Unique stage IDs required')
    paths=[]
    for stage in stages:
        if stage['kind'] not in ('gpu','cpu'):raise ValueError('Only explicit CPU/GPU stages supported')
        argv(stage['argv'])
        if not finite_positive(stage.get('max_seconds')):raise ValueError('Finite positive stage deadline required')
        result=stage['result'];path=Path(result['path'])
        if not path.is_absolute() or not path.resolve().is_relative_to(MOBILE):raise ValueError('Fresh mobile-owned absolute result path required')
        if not isinstance(result.get('schema'),str) or not result['schema']:raise ValueError('Exact typed result schema required')
        predicates=result.get('predicates')
        if not isinstance(predicates,list) or not predicates:raise ValueError('Explicit completion predicates required even for CPU exit0')
        for predicate in predicates:
            if not isinstance(predicate.get('path'),list) or not predicate['path'] or any(not isinstance(k,str) or not k for k in predicate['path']) or predicate.get('op') not in ('eq','gte','lte'):raise ValueError('Explicit dictionary-key predicate path/op required')
            if 'value' not in predicate:raise ValueError('Predicate comparison value required')
        if stage['kind']=='gpu':
            if stage.get('budget_stage') not in ('preflight','training','evaluation') or not finite_positive(stage.get('wait_seconds')):raise ValueError('GPU budget stage and finite FIFO wait timeout required')
            if stage['max_seconds']>21600:raise ValueError('GPU stage exceeds per-stage mobile limit')
        elif stage['max_seconds']>3600:raise ValueError('CPU stage must be bounded to3600s')
        if stage.get('role','generic') not in ('generic','oracle_gate','training'):raise ValueError('Unknown explicit stage role')
        paths.append(str(path.resolve()))
    if len(set(paths))!=len(paths):raise ValueError('Each stage requires its own fresh result path')
    if sum(s['max_seconds'] for s in stages if s['kind']=='gpu')>43200:raise ValueError('Planned GPU caps exceed12h authority')
    return plan


def budget_command(plan,stage):
    return [plan['supervisor_python'],'-m','experiments.pi05_g1_mobile.budget','--ledger',plan['ledger'],
            '--stage',stage['budget_stage'],'--max-seconds',str(stage['max_seconds']),'--host-reserve-gib','16',
            '--result-file',stage['result']['path'],'--result-schema',stage['result']['schema'],'--',*stage['argv']]


def value_at(document,path):
    current=document
    for key in path:
        if not isinstance(current,dict) or key not in current:raise ValueError('Missing required receipt field:'+'.'.join(path))
        current=current[key]
    return current


def check_result(stage,started_ns):
    path=Path(stage['result']['path'])
    if path.is_symlink() or not path.is_file() or path.stat().st_size>16*1024**2 or path.stat().st_mtime_ns<started_ns:raise ValueError('Missing,stale,symlink,or oversized completion receipt')
    document=read(path)
    if document.get('schema')!=stage['result']['schema']:raise ValueError('Completion schema mismatch')
    if stage['kind']=='gpu' and document.get('status')!='completed':raise ValueError('GPU producer did not truly complete')
    for predicate in stage['result']['predicates']:
        actual=value_at(document,predicate['path']);expected=predicate['value'];op=predicate['op']
        if op=='eq':valid=type(actual) is type(expected) and actual==expected
        else:
            valid=type(actual) in (int,float) and type(expected) in (int,float) and math.isfinite(actual) and math.isfinite(expected) and (actual>=expected if op=='gte' else actual<=expected)
        if not valid:raise ValueError('Completion predicate failed:'+'.'.join(predicate['path']))
    role=stage.get('role','generic')
    if role=='oracle_gate':
        ready=value_at(document,['acceptance','max_ready_span_seconds'])
        if document.get('status')!='completed' or document.get('success') is not False or type(ready) not in (int,float) or not math.isfinite(ready) or ready<2:raise ValueError('Oracle gate requires completed,2s ready span,and taskFalse;never teacher')
    elif role=='training':
        stop=document.get('training_stopped',{});validation=document.get('final_validation',{})
        if document.get('status')!='completed' or stop.get('complete_optimizer_update_boundary') is not True or type(document.get('step')) is not int or stop.get('step')!=document['step'] or validation.get('complete') is not True or type(validation.get('flow_loss')) not in (int,float) or not math.isfinite(validation['flow_loss']) or not document.get('final_checkpoint') or not document.get('selected_checkpoint'):raise ValueError('Training lacks real stop/final complete validation/checkpoint selection')
    return {'path':str(path),'sha256':sha(path),'schema':document['schema'],'predicate_count':len(stage['result']['predicates']),'role':role}


class QueueAPI:
    def __init__(self,root):self.root=Path(root)
    def request(self,owner,stage,seconds):
        from experiments.vla_resources import gpu_queue
        return gpu_queue.request(self.root,owner,stage,seconds)
    def run(self,ticket,ledger,command,wait):
        from experiments.vla_resources import gpu_queue
        return gpu_queue.run(self.root,ticket,ledger,command,wait)
    def snapshot(self,ticket):
        queue=read(self.root/'queue.json')
        matches=[x for x in queue['tickets'] if x['ticket']==ticket]
        if len(matches)!=1:raise ValueError('Cannot identify owned FIFO ticket')
        return matches[0]
    def cancel(self,ticket,owner):
        from experiments.vla_resources import gpu_queue
        return gpu_queue.cancel(self.root,ticket,owner)


def runs(ledger):
    if not Path(ledger).exists():return []
    rows=read(ledger)['runs']
    if not isinstance(rows,list):raise ValueError('Invalid GPU ledger')
    return rows


def closed(row):
    return (row.get('last_unit_properties',{}).get('ActiveState') in ('inactive','failed') and row.get('accounting_open') is not True and row.get('docker_cleanup',{}).get('complete') is True and row.get('launcher_cleanup_pending') is not True and row.get('status') not in ('running','cleanup_pending'))


def stop_cpu_group(process,term_seconds=2.,kill_seconds=2.):
    """Only this freshly created session's group; never another owned GPU unit."""
    for sig,timeout in ((signal.SIGTERM,term_seconds),(signal.SIGKILL,kill_seconds)):
        try:os.killpg(process.pid,sig)
        except ProcessLookupError:pass
        try:process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:pass
        # Even if the direct child exited early, send KILL to any remaining
        # descendants in the same new group before returning.
    if process.poll() is None:raise RuntimeError('CPU direct child not reaped after bounded cleanup')


def cpu_execute(command,seconds):
    environment=dict(os.environ);environment.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    child=subprocess.Popen(command,env=environment,shell=False,start_new_session=True)
    previous={}
    def interrupted(signum,frame):raise InterruptedError('CPU stage interrupted by signal'+str(signum))
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):previous[sig]=signal.signal(sig,interrupted)
        return child.wait(timeout=seconds)
    except BaseException:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        stop_cpu_group(child)
        raise
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def execute(plan_path,output,*,dry_run=False,queue=None,cpu_runner=cpu_execute):
    plan_path=Path(plan_path).resolve();plan=validate_plan(read(plan_path));output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh driver journal directory required;no implicit resume/retry')
    output.mkdir(parents=True);plan_hash=sha(plan_path);atomic(output/'plan_snapshot.json',plan)
    snapshot_frozen(plan.get('frozen_files',{}),output)
    state={'schema':'pi05_g1_mobile_stage_driver_result_v1','status':'dry_run' if dry_run else 'running','plan_path':str(plan_path),'plan_sha256':plan_hash,'owner':plan['owner'],'completed_stages':[],'current_stage':None,'root_interpretation_required':True}
    journal=output/'events.jsonl'
    def event(name,**fields):
        row={'event':name,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields}
        with journal.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        atomic(output/'state.json',state)
    event('driver_prepared',dry_run=dry_run)
    if dry_run:
        state['commands']=[{'id':s['id'],'kind':s['kind'],'argv':budget_command(plan,s) if s['kind']=='gpu' else s['argv'],'fifo_request':{'owner':plan['owner'],'stage':s['budget_stage'],'max_seconds':s['max_seconds']} if s['kind']=='gpu' else None} for s in plan['stages']]
        event('dry_run_complete',real_fifo_accessed=False);atomic(output/'driver_result.json',state);return state
    queue=queue or QueueAPI(plan['coord_root']);ticket=None
    try:
        for stage in plan['stages']:
            if sha(plan_path)!=plan_hash:raise ValueError('Root-reviewed plan changed while executing')
            verify_frozen(plan.get('frozen_files',{}))
            state['current_stage']=stage['id'];ticket=None
            if Path(stage['result']['path']).exists() or Path(stage['result']['path']).is_symlink():raise FileExistsError('Stage result must not exist before execution')
            started_ns=time.time_ns();event('stage_started',stage=stage['id'],kind=stage['kind'],argv=stage['argv'])
            if stage['kind']=='gpu':
                before=len(runs(plan['ledger']))
                row=queue.request(plan['owner'],stage['budget_stage'],stage['max_seconds']);ticket=row['ticket']
                if row.get('owner')!=plan['owner'] or row.get('status')!='waiting':raise ValueError('FIFO request identity invalid')
                event('fifo_ticket_requested',ticket=ticket,stage=stage['id'],queue_wait_gpu_charge_seconds=0)
                code=queue.run(ticket,plan['ledger'],budget_command(plan,stage),stage['wait_seconds'])
                row=queue.snapshot(ticket)
                if row.get('owner')!=plan['owner']:raise ValueError('FIFO owner changed')
                if row.get('status')=='waiting':
                    queue.cancel(ticket,plan['owner']);event('waiting_ticket_cancelled',ticket=ticket,code=code)
                    raise RuntimeError('FIFO wait interrupted/timedout;cancelled waiting ticket,no retry')
                if row.get('status')!='completed' or row.get('cleanup_confirmed') is not True:raise RuntimeError('FIFO failed/cancelled/cleanup_unknown;do not continue')
                after=runs(plan['ledger'])
                if len(after)!=before+1 or not closed(after[-1]) or after[-1].get('stage')!=stage['budget_stage'] or after[-1].get('status')!='completed' or after[-1].get('exit_code')!=0 or after[-1].get('workload_completion_contract',{}).get('valid') is not True:raise RuntimeError('GPU accounting/typed-result/resource closure not confirmed')
                contract=after[-1]['workload_completion_contract']
                if Path(contract['path']).resolve()!=Path(stage['result']['path']).resolve() or contract.get('expected_schema')!=stage['result']['schema']:raise ValueError('Supervisor completion contract differs from plan')
                event('gpu_cleanup_confirmed',ticket=ticket,budget_token=after[-1].get('budget_token'),elapsed_seconds=after[-1].get('elapsed_seconds'))
            else:code=cpu_runner(stage['argv'],stage['max_seconds'])
            verify_frozen(plan.get('frozen_files',{}))
            if code!=0:raise RuntimeError('Stage command nonzero:'+str(code))
            if sha(plan_path)!=plan_hash:raise ValueError('Root-reviewed plan changed during stage/wait')
            receipt=check_result(stage,started_ns)
            state['completed_stages'].append({'id':stage['id'],'result':receipt,'ticket':ticket});event('stage_completed',stage=stage['id'],receipt=receipt)
        state.update(status='completed',current_stage=None);event('driver_completed')
    except BaseException as error:
        # A signal before queue.run takes over can leave a waiting ticket. Only
        # cancel our own confirmed waiting ticket, never a running/unknown one.
        if ticket:
            try:
                row=queue.snapshot(ticket)
                if row.get('owner')==plan['owner'] and row.get('status')=='waiting':queue.cancel(ticket,plan['owner']);event('waiting_ticket_cancelled',ticket=ticket)
            except BaseException as cleanup_error:state['ticket_state_error']=repr(cleanup_error)
        state.update(status='attention',error=repr(error),automatic_retry=False);event('driver_stopped_attention')
    atomic(output/'driver_result.json',state);return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    result=execute(a.plan,a.output,dry_run=a.dry_run);print(json.dumps({'status':result['status'],'output':a.output}));return 0 if result['status'] in ('completed','dry_run') else 2

if __name__=='__main__':raise SystemExit(main())
