"""Plan/bind/summarize held-out action diagnostics; planning never loads test observations.

The run-job subcommand explicitly invokes the unchanged GPU evaluate_offline
implementation. Its separate receipts provide identities absent from raw output.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
POLICIES=('pi05_round01','pi05_round02','official_pi05_base')
GROUP_UNITS={'arms14':'rad','activehands_train_nonconstant':'rad','allhands14':'rad',
             'height':'m','vx':'m/s','vy':'m/s','yaw':'rad/s','nav3':None}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+v)))


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def persist(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.incomplete')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def make_plan(selection_path,train_bounds,output,*,python=None,hours_per_model=.25):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('New offline diagnostic directory required')
    if not finite(hours_per_model) or not 0<hours_per_model<=8:raise ValueError('Positive bounded per-model hours required')
    selection_path=Path(selection_path).resolve();selection=read(selection_path)
    if selection.get('schema')!='pi05_g1_round02_final_selection_v1':raise ValueError('Main final-selection receipt required')
    if selection['selected']['validation_flow_loss']!=min(c['validation_flow_loss'] for c in selection['all_complete_validation_candidates']):
        raise ValueError('Selection receipt does not certify complete-validation minimum')
    identity=selection['data_identity'];sources=[s for s in identity['sources'] if s['kind']=='lerobot']
    if len(sources)!=1:raise ValueError('Exactly one original source identity required')
    manifest=Path(sources[0]['manifest']).resolve()
    if sha(manifest)!=sources[0]['manifest_sha256']:raise ValueError('Original split manifest changed')
    data=read(manifest);ids=data['test'];train=data['train']
    if len(ids)!=21 or len(set(ids))!=len(ids) or set(ids)&set(train):raise ValueError('Expected original21 held-out whole episodes')
    files={r['episode_id']:r for r in data['files']};coverage=[]
    for eid in ids:
        length=files[eid]['length']
        if type(length) is not int or length<=0:raise ValueError('Manifest lacks positive audited episode lengths')
        coverage.append({'episode_id':eid,'length':length,'chunk_starts':list(range(0,length,50)),'chunks':math.ceil(length/50)})
    expected_chunks=sum(r['chunks'] for r in coverage);expected_frames=sum(r['length'] for r in coverage)
    norm_paths=[Path(p).resolve() for p,h in selection['frozen_files'].items() if h==identity['norm_sha256'] and Path(p).name=='norm_stats_v2.json']
    if len(norm_paths)!=1 or sha(norm_paths[0])!=identity['norm_sha256']:raise ValueError('Unambiguous frozen norm v2 required')
    norm=norm_paths[0];bounds_path=Path(train_bounds).resolve();bounds=read(bounds_path)
    if bounds['source_manifest_sha256']!=sha(manifest) or set(bounds['train_episodes'])!=set(train) or not bounds['source_split'].startswith('train_only'):
        raise ValueError('Active hand selection requires original TRAIN-only raw range provenance')
    lo,hi=bounds['lower'],bounds['upper']
    if len(lo)!=32 or len(hi)!=32 or any(not finite(a) or not finite(b) or a>b for a,b in zip(lo,hi)):raise ValueError('Invalid raw training ranges')
    groups={'arms14':list(range(14)),'activehands_train_nonconstant':[i for i in range(14,28) if hi[i]-lo[i]>0],
            'allhands14':list(range(14,28)),'height':[28],'vx':[29],'vy':[30],'yaw':[31],'nav3':[29,30,31]}
    frozen={str(selection_path):sha(selection_path),str(manifest):sha(manifest),str(norm):sha(norm),str(bounds_path):sha(bounds_path)}
    for name in ('evaluate_offline.py','model_runtime.py','training.py','contract.py'):
        path=ROOT/'experiments/pi05_g1'/name;frozen[str(path)]=sha(path)
    evaluator=ROOT/'experiments/pi05_g1/evaluate_offline.py'
    # Checkpoint payloads are not read while planning; phase receipts bound by final selection are reused.
    jobs=[]
    for policy in POLICIES:
        artifact=selection['model_artifacts'][policy];checkpoint=Path(artifact['path']).resolve().parent
        if policy=='pi05_round02' and checkpoint!=Path(selection['selected']['path']).resolve():raise ValueError('New model differs from frozen selected checkpoint')
        result=output/(policy+'.json');receipt=output/(policy+'.receipt.json')
        argv=[str(Path(python or ROOT/'.scratch/pi05_g1/venv/bin/python').absolute()),'-m','experiments.pi05_g1.evaluate_offline',
              '--dataset',data['dataset_root'],'--manifest',str(manifest),'--norm-stats',str(norm),'--checkpoint',str(checkpoint),
              '--split','test','--hours',str(hours_per_model),'--output',str(result)]
        jobs.append({'policy':policy,'argv':argv,'output':str(result),'receipt':str(receipt),'checkpoint':str(checkpoint),
                     'weights':artifact,'norm_sha256':identity['norm_sha256'],'manifest_sha256':sha(manifest)})
    plan={'schema':'pi05_g1_round02_offline_diagnostic_plan_v1','selection_receipt':str(selection_path),'selection_receipt_sha256':sha(selection_path),
          'frozen_files':frozen,'jobs':jobs,'split':'test','horizon':50,'test_episode_ids':ids,'episode_coverage':coverage,
          'expected_chunks_per_policy':expected_chunks,'expected_valid_frames_per_policy':expected_frames,'groups':groups,'group_units':GROUP_UNITS,
          'activehands_rule':'Original TRAIN raw upper-lower >0; selected before reading test observations',
          'source_action_names':bounds['action_names'],'physical_success_evaluated':False,'normalization':'common norm_stats_v2; no simulator/action-limit clipping',
          'scope':'Auxiliary raw action MAE/RMSE only. Never physical success, checkpoint selection or evidence of closed-loop task benefit.',
          'target_caveat':'Unchanged evaluator denormalizes its normalized dataset target. Nonconstant channels with collapsed quantiles would lose raw target information; common v2 repairs the known vx/vy issue.',
          'evaluator_path':str(evaluator),'evaluator_sha256':sha(evaluator)}
    output.mkdir(parents=True);write_new(output/'plan.json',plan);return plan


def verify_inputs(plan,job):
    observed={}
    for path,expected in {**plan['frozen_files'],job['weights']['path']:job['weights']['sha256']}.items():
        observed[path]=sha(path)
        if observed[path]!=expected:raise ValueError('Offline diagnostic immutable input changed: '+path)
    return observed


def run_job(plan_path,policy):
    plan_path=Path(plan_path).resolve();plan=read(plan_path)
    if plan.get('schema')!='pi05_g1_round02_offline_diagnostic_plan_v1':raise ValueError('Unsupported plan')
    job=next(j for j in plan['jobs'] if j['policy']==policy);rp=Path(job['receipt']);result=Path(job['output']);log=rp.with_suffix('.log')
    if any(p.exists() for p in (rp,result,log)):raise FileExistsError('Preserve existing result/receipt/log; generate a new plan directory')
    receipt={'schema':'pi05_g1_round02_offline_job_receipt_v1','status':'preparing','policy':policy,'plan':str(plan_path),
             'plan_sha256':sha(plan_path),'started_utc':now(),'command':job['argv'],'physical_success_evaluated':False}
    write_new(rp,receipt)
    try:
        receipt['before_sha256']=verify_inputs(plan,job);receipt['status']='running';persist(rp,receipt)
        with log.open('x') as stream:
            process=subprocess.run(job['argv'],cwd=ROOT,env=dict(os.environ),stdout=stream,stderr=subprocess.STDOUT)
        receipt['exit_code']=process.returncode;receipt['after_sha256']=verify_inputs(plan,job)
        if sha(plan_path)!=receipt['plan_sha256']:raise ValueError('Plan changed during diagnostic execution')
        if process.returncode or not result.is_file():raise ValueError('Original evaluator failed; preserve log')
        raw=read(result);receipt.update(status=raw['status'],raw_result_sha256=sha(result),ended_utc=now())
        persist(rp,receipt);return receipt
    except BaseException as error:
        receipt.update(status='infrastructure_failed',error=repr(error),ended_utc=now());persist(rp,receipt);raise


def aggregate(plan_path,output):
    plan_path=Path(plan_path).resolve();plan=read(plan_path)
    if plan.get('schema')!='pi05_g1_round02_offline_diagnostic_plan_v1':raise ValueError('Unsupported diagnostic plan')
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new summary output')
    policies={};errors=[];input_shas={str(plan_path):sha(plan_path)}
    for job in plan['jobs']:
        policy=job['policy'];row={'status':'incomplete','errors':[]};policies[policy]=row
        try:
            receipt_path=Path(job['receipt']);result_path=Path(job['output']);receipt=read(receipt_path);raw=read(result_path)
            input_shas[str(receipt_path)]=sha(receipt_path);input_shas[str(result_path)]=sha(result_path)
            expected={**plan['frozen_files'],job['weights']['path']:job['weights']['sha256']}
            if (receipt.get('schema')!='pi05_g1_round02_offline_job_receipt_v1' or receipt.get('policy')!=policy
                or receipt.get('plan_sha256')!=sha(plan_path) or receipt.get('command')!=job['argv']
                or receipt.get('before_sha256')!=expected or receipt.get('after_sha256')!=expected
                or receipt.get('raw_result_sha256')!=sha(result_path) or receipt.get('exit_code')!=0):raise ValueError('Result identity receipt missing/inconsistent')
            if receipt['status']!='complete' or raw['status']!='complete':raise ValueError('Evaluator partial/incomplete: '+str(raw['status']))
            if (raw['split']!='test' or Path(raw['checkpoint']).resolve()!=Path(job['checkpoint']).resolve()
                or raw.get('physical_success_evaluated') is not False
                or raw['chunks_planned']!=plan['expected_chunks_per_policy'] or raw['chunks_evaluated']!=plan['expected_chunks_per_policy']
                or raw['valid_frames']!=plan['expected_valid_frames_per_policy']):raise ValueError('Incomplete/different split or chunk/frame coverage')
            mae,rmse=raw['mae_per_dimension'],raw['rmse_per_dimension']
            if len(mae)!=32 or len(rmse)!=32 or any(not finite(a) or not finite(r) or a<0 or r<a-1e-7 for a,r in zip(mae,rmse)):
                raise ValueError('Invalid per-dimension action error')
            groups={}
            for name,dims in plan['groups'].items():
                groups[name]={'dimensions':dims,'unit':plan['group_units'][name],
                    'physical_unit_defined':name!='nav3',
                    'interpretation':'mixed-units numeric-only; no physical unit and no cross-group ranking' if name=='nav3' else 'Same-unit component error; compare like units and dimensions only',
                    'mae':sum(mae[i] for i in dims)/len(dims) if dims else None,
                    'rmse':math.sqrt(sum(rmse[i]**2 for i in dims)/len(dims)) if dims else None}
            row.update(status='complete',checkpoint=job['checkpoint'],checkpoint_sha256=job['weights']['sha256'],norm_sha256=job['norm_sha256'],
                       manifest_sha256=job['manifest_sha256'],chunks=raw['chunks_evaluated'],valid_frames=raw['valid_frames'],
                       mae_per_dimension=mae,rmse_per_dimension=rmse,groups=groups,
                       inference_seconds_mean=raw['inference_seconds_mean'],inference_seconds_p95=raw['inference_seconds_p95'])
        except (OSError,ValueError,KeyError,TypeError) as error:row['errors'].append(str(error));errors.append(policy+': '+str(error))
    complete=set(policies)==set(POLICIES) and all(r['status']=='complete' for r in policies.values())
    report={'schema':'pi05_g1_round02_offline_diagnostic_summary_v1','complete':complete,'policies':policies,'errors':errors,
            'same_split_norm_and_frozen_checkpoint_bindings_verified':complete,'expected_chunks_per_policy':plan['expected_chunks_per_policy'],
            'expected_valid_frames_per_policy':plan['expected_valid_frames_per_policy'],'groups':plan['groups'],'group_units':plan['group_units'],'action_names':plan['source_action_names'],
            'group_aggregation':'MAE is arithmetic mean across dimensions; RMSE is sqrt(mean of per-dimension MSE).',
            'units':'arms14/activehands/allhands14: rad; height: m; vx/vy: m/s; yaw: rad/s. nav3 is mixed-units numeric-only, has no physical unit, and must not be used for cross-group ranking. Per-dimension and separately grouped vx/vy/yaw are retained.',
            'physical_success_evaluated':False,'selection_affected':False,'scope':plan['scope'],'target_caveat':plan['target_caveat'],'inputs_sha256':input_shas}
    write_new(output,report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    plan=sub.add_parser('plan')
    for flag in ('selection-receipt','train-raw-bounds','output'):plan.add_argument('--'+flag,type=Path,required=True)
    plan.add_argument('--python',type=Path);plan.add_argument('--hours-per-model',type=float,default=.25)
    run=sub.add_parser('run-job');run.add_argument('--plan',type=Path,required=True);run.add_argument('--policy',choices=POLICIES,required=True)
    summary=sub.add_parser('summarize');summary.add_argument('--plan',type=Path,required=True);summary.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='plan':result=make_plan(a.selection_receipt,a.train_raw_bounds,a.output,python=a.python,hours_per_model=a.hours_per_model)
    elif a.mode=='run-job':result=run_job(a.plan,a.policy)
    else:result=aggregate(a.plan,a.output)
    print(json.dumps({'schema':result['schema'],'complete':result.get('complete'),'status':result.get('status'),'expected_chunks':result.get('expected_chunks_per_policy')}))


if __name__=='__main__':main()
