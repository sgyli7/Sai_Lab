"""Owned serial frozen60 evaluation in bounded3/5/10-job segments.

No queue wait, GPU budget creation, score selection, or task-success requirement.
Wrap each segment with mobile.budget stage=evaluation; FIFO belongs outside it.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .benchmark import sha,read,path_from
from .simulator import validate_config

ROOT=Path(__file__).resolve().parents[2]
LEDGER=ROOT/'.scratch/pi05_g1/round03_mobile/gpu_budget.json'
SEGMENT_SCHEMA='pi05_g1_mobile_batch_segment_result_v1'
BATCH_SCHEMA='pi05_g1_mobile_batch_result_v1'


def persist(path,value):
    path=Path(path);temp=path.with_name(path.name+'.incomplete')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def verify_files(files):
    for name,expected in files.items():
        if sha(name)!=expected:raise ValueError('Immutable batch evidence changed: '+name)


def case_files(directory):
    directory=Path(directory).resolve();result={}
    for p in sorted(directory.rglob('*')):
        if p.is_symlink():raise ValueError('Case evidence symlink refused: '+str(p))
        if p.is_file():result[str(p.resolve())]=sha(p)
    return result


def ownership(ledger):
    token=os.environ.get('PI05_G1_BUDGET_TOKEN');directory=os.environ.get('PI05_G1_BUDGET_OWNERSHIP_DIR')
    if not token or not directory:raise ValueError('An outer mobile evaluation budget owner is mandatory; queue outside GPU budget')
    doc=read(ledger);matches=[r for r in doc['runs'] if r.get('budget_token')==token]
    if len(matches)!=1:raise ValueError('Unique owned budget token missing from ledger')
    row=matches[0]
    if row['stage']!='evaluation' or row['status']!='running' or Path(row['ownership_receipt_dir']).resolve()!=Path(directory).resolve():raise ValueError('Owned budget is not a running evaluation stage')
    deadline=row['started_unix']+row['child_work_limit_seconds']
    return {'budget_token':token,'ownership_receipt_dir':str(Path(directory).resolve()),'owned_unit':row['owned_unit'],'ledger':str(Path(ledger).resolve()),'work_deadline_unix':deadline}


def load_plan(plan_path):
    plan=read(plan_path)
    if plan.get('schema')!='pi05_g1_mobile_benchmark_plan_v1':raise ValueError('Expected frozen mobile benchmark plan')
    jobs=plan['jobs'];policies=plan.get('primary_policies',plan.get('policies',sorted({j['policy'] for j in jobs})))
    ids=plan.get('case_ids',[c['case_id'] for c in plan.get('cases',[])])
    if len(policies)!=3 or len(set(policies))!=3 or len(ids)!=20 or len(set(ids))!=20 or len(jobs)!=60:raise ValueError('Exactly3policies x20cases =60jobs required')
    if {(j['policy'],j['case_id']) for j in jobs}!={(p,c) for p in policies for c in ids}:raise ValueError('Unique complete Cartesian policy/case coverage required')
    frozen={str(path_from(plan_path,p)):digest for p,digest in plan.get('frozen_files',{}).items()}
    if not frozen:raise ValueError('Frozen input SHA manifest required')
    frozen[str(plan_path)]=sha(plan_path);configs={};outputs=[]
    for i,job in enumerate(jobs):
        path=path_from(plan_path,job['config_path']);expected=job.get('config_sha256',frozen.get(str(path)))
        if not expected:raise ValueError('Each config needs predeclared SHA, not a runtime repin')
        frozen[str(path)]=expected;config=read(path)
        if config['mode']!='policy' or config.get('policy_id')!=job['policy'] or config['case']['case_id']!=job['case_id']:raise ValueError('Policy/case mode differs from plan')
        result=path_from(plan_path,job['result_path']);job['config_path']=str(path);job['result_path']=str(result);out=Path(config['output_dir']).resolve()
        if result!=out/'result.json':raise ValueError('Planned result must be inside exact case output_dir')
        configs[i]=(path,config);outputs.append(str(out))
    if len(set(outputs))!=60:raise ValueError('Every case must have a separate evidence directory')
    verify_files(frozen);return plan,configs,frozen


def completed_record(index,job,config_path,config,launch_path,log_path):
    folder=Path(config['output_dir']);rp=Path(job['result_path']);result=read(rp)
    if (result.get('schema')!='pi05_g1_mobile_case_result_v1' or result.get('status')!='completed' or result.get('source_interface_passed') is not True or result.get('policy')!=job['policy'] or result.get('case_id')!=job['case_id'] or result.get('cleanup_errors')):raise ValueError('Producer infrastructure/interface did not complete')
    if read(folder/'config.json')!=validate_config(config):raise ValueError('Producer executed config differs from frozen config')
    if type(result.get('success')) is not bool or result.get('acceptance',{}).get('success') is not result['success']:raise ValueError('Completed producer lacks real task outcome')
    trace=folder/'trajectory.jsonl'
    if not result.get('trace_sha256') or sha(trace)!=result['trace_sha256']:raise ValueError('Completed producer trace SHA mismatch')
    launch=read(launch_path)
    if launch.get('schema')!='pi05_g1_mobile_case_launch_v1' or launch.get('exit_code')!=0 or launch.get('config_sha256')!=sha(config_path):raise ValueError('Case launch receipt identity/exit mismatch')
    return {'index':index,'policy':job['policy'],'case_id':job['case_id'],'status':'completed','success':result['success'],'result_path':str(rp),'result_sha256':sha(rp),'config_sha256':sha(config_path),'case_files_sha256':case_files(folder),'launch_path':str(launch_path),'launch_sha256':sha(launch_path),'launch_log':str(log_path),'launch_log_sha256':sha(log_path),'source_interface_passed':True}


def resume_records(paths,plan_sha,plan,configs):
    previous={};input_receipts={}
    for path in paths:
        path=Path(path).resolve();doc=read(path)
        if doc.get('schema') not in [SEGMENT_SCHEMA,BATCH_SCHEMA] or doc.get('plan_sha256')!=plan_sha:raise ValueError('Resume receipt schema/frozen plan differs')
        input_receipts[str(path)]=sha(path)
        for record in doc.get('jobs',[]):
            if record.get('status')!='completed':continue
            i=record['index']
            if type(i) is not int or not 0<=i<60 or (record['policy'],record['case_id'])!=(plan['jobs'][i]['policy'],plan['jobs'][i]['case_id']):raise ValueError('Resume job identity/order mismatch')
            if i in previous and previous[i]!=record:raise ValueError('Conflicting completed job receipts')
            verify_files(record['case_files_sha256'])
            if case_files(configs[i][1]['output_dir'])!=record['case_files_sha256']:raise ValueError('Case evidence inventory changed after completion')
            verify_files({record['launch_path']:record['launch_sha256'],record['launch_log']:record['launch_log_sha256']})
            refreshed=completed_record(i,plan['jobs'][i],*configs[i],Path(record['launch_path']),Path(record['launch_log']))
            if refreshed!=record:raise ValueError('Completed job receipt changed on resume')
            previous[i]=record
    return previous,input_receipts


class Cancellation:
    def __init__(self):self.requested=False;self.signum=None;self.child=None
    def handler(self,signum,frame):
        self.requested=True;self.signum=int(signum)
        if self.child is not None and self.child.poll() is None:self.child.terminate()


def launch(command,log,timeout,cancel,cleanup_seconds=20):
    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);cancel.child=child
    try:
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:child.wait(timeout=cleanup_seconds)
        except subprocess.TimeoutExpired:raise RuntimeError('Owned launcher cleanup_pending; outer budget must reconcile owned Docker CID, no next job')
        raise TimeoutError('Owned case reached batch work deadline; stopped launcher with bounded cleanup')
    finally:cancel.child=None


def run(plan_path,receipt_path,batch_result,*,max_jobs=5,start_index=None,resume_receipts=(),ledger=LEDGER,min_job_seconds=300,cleanup_reserve_seconds=45):
    plan_path=Path(plan_path).resolve();receipt_path=Path(receipt_path).resolve();batch_result=Path(batch_result).resolve()
    if receipt_path.exists() or receipt_path.with_name(receipt_path.name+'.incomplete').exists():raise FileExistsError('New segment receipt required; preserve prior attempts')
    if max_jobs not in [3,5,10] or not math.isfinite(min_job_seconds) or not math.isfinite(cleanup_reserve_seconds) or min_job_seconds<=0 or cleanup_reserve_seconds<40:raise ValueError('Segments3or5or10, positive job admission and>=40s cleanup reserve required')
    plan,configs,frozen=load_plan(plan_path)
    if receipt_path==batch_result or any(receipt_path.is_relative_to(Path(c['output_dir']).resolve()) or batch_result.is_relative_to(Path(c['output_dir']).resolve()) for _,c in configs.values()):raise ValueError('Batch receipts must be distinct and outside immutable raw case outputs')
    owner=ownership(ledger);started=time.time();plan_sha=sha(plan_path)
    records,resume_inputs=resume_records(resume_receipts,plan_sha,plan,configs);frozen.update(resume_inputs)
    prefix=0
    while prefix in records:prefix+=1
    if set(records)!=set(range(prefix)):raise ValueError('Resume must establish a contiguous fixed-order prefix')
    start=prefix if start_index is None else start_index
    if start!=prefix:raise ValueError('Segment must start at first uncompleted frozen job; no skipping or score-based reordering')
    stop=min(60,start+max_jobs)
    if start>=60:raise ValueError('All60 already complete; do not start a fresh budget segment')
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    receipt={'schema':SEGMENT_SCHEMA,'status':'prepared','overall_status':'partial','benchmark_complete':False,'plan':str(plan_path),'plan_sha256':plan_sha,'frozen_files':frozen,'owner':owner,'started_unix':started,'segment_start_index':start,'segment_stop_index_exclusive':stop,'expected_segment_jobs':stop-start,'expected_total_jobs':60,'jobs':[],'previous_completed_jobs':len(records),'resume_receipts_sha256':resume_inputs,'tool_sha256':sha(__file__)}
    persist(receipt_path,receipt);cancel=Cancellation();prior={s:signal.getsignal(s) for s in [signal.SIGTERM,signal.SIGINT]}
    for signum in prior:signal.signal(signum,cancel.handler)
    lock=(plan_path.parent/'.mobile_batch_execution.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for index in range(start,stop):
            verify_files(frozen)
            if ownership(ledger)!=owner:raise RuntimeError('Evaluation owner identity/deadline changed')
            if sha(__file__)!=receipt['tool_sha256']:raise RuntimeError('Batch runner source changed')
            if cancel.requested:receipt['stop_reason']='signal_boundary';break
            job=plan['jobs'][index];cp,config=configs[index]
            estimate=float(job.get('estimated_wall_seconds',0))
            if not math.isfinite(estimate) or estimate<0:raise ValueError('Invalid predeclared job wall-time estimate')
            needed=max(min_job_seconds,estimate)
            left=owner['work_deadline_unix']-time.time()-cleanup_reserve_seconds
            if left<needed:
                receipt.update(stop_reason='insufficient_owned_work_time_for_next_complete_job',remaining_work_seconds=left,next_job_admission_seconds=needed);break
            folder=Path(config['output_dir']);lp=cp.with_suffix('.launch.json');log_path=cp.with_suffix('.launch.log')
            if folder.exists() or lp.exists() or log_path.exists():raise FileExistsError('Existing/partial case or launch evidence has no verified completed resume receipt: '+str(folder))
            command=[sys.executable,'-m','experiments.pi05_g1_mobile.sim_launcher','--docker','--config',str(cp)]
            job_start=time.time();pending={'index':index,'policy':job['policy'],'case_id':job['case_id'],'status':'running','config_path':str(cp),'started_unix':job_start}
            receipt['status']='running';receipt['jobs'].append(pending);persist(receipt_path,receipt)
            launch_receipt={'schema':'pi05_g1_mobile_case_launch_v1','status':'running','command':command,'config_sha256':sha(cp),'budget_token':owner['budget_token'],'started_unix':job_start}
            persist(lp,launch_receipt)
            print('CASE_START',index,job['policy'],job['case_id'],flush=True)
            try:
                with log_path.open('x') as log:code=launch(command,log,max(.1,left),cancel)
                launch_receipt.update(status='completed' if code==0 else 'failed',exit_code=code,elapsed_seconds=time.time()-job_start)
                persist(lp,launch_receipt)
                if code:raise RuntimeError('Owned case launcher failed with exit'+str(code))
                record=completed_record(index,job,cp,config,lp,log_path);records[index]=record;receipt['jobs'][-1]=record
                persist(receipt_path,receipt);print('CASE_COMPLETE',index,job['policy'],job['case_id'],'success='+str(record['success']),flush=True)
            except BaseException as error:
                pending.update(status='infrastructure_failed',error=repr(error));launch_receipt.update(status='failed',error=repr(error),elapsed_seconds=time.time()-job_start)
                persist(lp,launch_receipt);raise
        verify_files(frozen)
        if len(receipt['jobs'])==stop-start and all(r['status']=='completed' for r in receipt['jobs']):receipt['status']='completed'
        else:receipt['status']='partial'
        receipt.update(completed_total_jobs=len(records),benchmark_complete=len(records)==60,overall_status='completed' if len(records)==60 else 'partial',ended_unix=time.time(),signal=cancel.signum)
        if len(records)==60:
            if batch_result.exists():raise FileExistsError('Preserve existing final batch result')
            final={'schema':BATCH_SCHEMA,'status':'completed','benchmark_complete':True,'plan':str(plan_path),'plan_sha256':plan_sha,'expected_total_jobs':60,'completed_total_jobs':60,'jobs':[records[i] for i in range(60)],'final_segment_receipt':str(receipt_path),'resume_receipts_sha256':resume_inputs,'tool_sha256':receipt['tool_sha256'],'task_success_not_required_for_producer_completion':True,'independent_physics_audit_and_benchmark_pending':True,'ended_unix':time.time()}
            persist(batch_result,final);receipt['final_batch_result_path']=str(batch_result);receipt['final_batch_result_sha256']=sha(batch_result)
        persist(receipt_path,receipt);return receipt
    except BaseException as error:
        receipt.update(status='partial' if cancel.requested else 'infrastructure_failed',overall_status='partial',benchmark_complete=False,error=repr(error),completed_total_jobs=len(records),ended_unix=time.time(),signal=cancel.signum)
        persist(receipt_path,receipt);raise
    finally:
        lock.close()
        for signum,handler in prior.items():signal.signal(signum,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--batch-result',type=Path,required=True);p.add_argument('--ledger',type=Path,default=LEDGER);p.add_argument('--max-jobs',type=int,choices=[3,5,10],default=5);p.add_argument('--start-index',type=int);p.add_argument('--resume-receipt',type=Path,action='append',default=[]);p.add_argument('--min-job-seconds',type=float,default=300);p.add_argument('--cleanup-reserve-seconds',type=float,default=45);a=p.parse_args()
    result=run(a.plan,a.receipt,a.batch_result,max_jobs=a.max_jobs,start_index=a.start_index,resume_receipts=a.resume_receipt,ledger=a.ledger,min_job_seconds=a.min_job_seconds,cleanup_reserve_seconds=a.cleanup_reserve_seconds)
    print(json.dumps({'schema':result['schema'],'status':result['status'],'benchmark_complete':result['benchmark_complete'],'completed_total_jobs':result['completed_total_jobs']}))
    if result['status']!='completed':raise SystemExit(2)

if __name__=='__main__':main()
