"""Export only completed final-plan cases after the GPU ledger is fully closed.

Pure CPU subprocesses reuse video_export without modifying raw evidence. The
existing budget ledger lock is held to exclude managed GPU launches during export.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
REPRESENTATIVE='p0_seed40'
PRIMARY=('gr00t_n17','pi05_round01','pi05_round02')
STANDARD=('head_raw','observer_raw','head_wall_annotated','observer_wall_annotated')
WALL=('head_wall_clock','observer_wall_clock')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def persist(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.incomplete')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def closed_ledger(path):
    ledger=read(path)
    if ledger.get('schema')!='pi05_g1_gpu_budget_v1' or not isinstance(ledger.get('runs'),list):raise ValueError('Recognized complete GPU ledger required')
    blocked=[r.get('token',r.get('command')) for r in ledger['runs']
             if r.get('status') in ('running','cleanup_pending','preparing') or r.get('accounting_open') is True or r.get('launcher_cleanup_pending') is True]
    if blocked:raise ValueError('GPU accounting/cleanup not closed: '+str(blocked))
    return sha(path)


def collect_jobs(evaluation_root):
    root=Path(evaluation_root).resolve();receipts=[]
    # Only top-level batch receipts matched by their exact formal plan identity.
    for path in sorted(root.glob('*.json')):
        doc=read(path)
        if doc.get('schema')=='pi05_g1_round02_batch_receipt_v1':receipts.append((path,doc))
    jobs=[];phases=[];errors=[];seen=set()
    for phase,policies in [('primary',PRIMARY),('optional_base',('official_pi05_base',))]:
        plan_path=root/(phase+'_plan.json')
        if not plan_path.is_file():
            if phase=='primary':raise ValueError('Formal primary plan is missing')
            phases.append({'phase':phase,'status':'not_planned'});continue
        plan=read(plan_path)
        if plan.get('schema')!='pi05_g1_round02_frozen_benchmark_plan_v1' or plan.get('phase')!=phase or plan.get('policies')!=list(policies):raise ValueError('Unknown/incompatible formal plan')
        exporter_path=ROOT/'experiments/pi05_g1_round02/video_export.py'
        if plan.get('frozen_files',{}).get(str(exporter_path))!=sha(exporter_path):raise ValueError('Frozen exporter source identity missing/changed')
        matches=[(p,r) for p,r in receipts if Path(r['plan']).resolve()==plan_path]
        if not matches:
            if phase=='primary':raise ValueError('Primary completed batch receipt is missing')
            phases.append({'phase':phase,'status':'not_executed'});continue
        if len(matches)!=1:raise ValueError('Ambiguous batch receipts; retain an unambiguous audited final root')
        receipt_path,receipt=matches[0]
        if receipt.get('status')!='completed':
            if phase=='primary':raise ValueError('Primary batch is not completed')
            errors.append('Optional official-base phase exists but is incomplete; no partial policy exported')
            phases.append({'phase':phase,'status':'incomplete','receipt':str(receipt_path)});continue
        if receipt.get('plan_sha256')!=sha(plan_path):raise ValueError('Completed receipt plan SHA mismatch')
        expected_ids={f'p{p}_seed{s}' for p in range(5) for s in range(40,44)}
        expected_pairs={(policy,cid) for policy in policies for cid in expected_ids};actual=set()
        records={r['name']:r for r in receipt['jobs']}
        if len(records)!=len(receipt['jobs']) or len(plan['jobs'])!=len(expected_pairs) or len(records)!=len(expected_pairs):raise ValueError('Incomplete/duplicate formal jobs')
        for job in plan['jobs']:
            path=Path(job['config_path']).resolve()
            if path.parent!=root/'configs':raise ValueError('Case configs must be the formal evaluation root configs')
            digest=sha(path);config=read(path);policy=config['policy_id'];cid=config['case']['case_id'];pair=(policy,cid)
            if pair in actual or pair not in expected_pairs:raise ValueError('Development/teacher/duplicate case forbidden')
            positions=((0,0),(.02,0),(-.02,0),(0,.02),(0,-.02))
            if config['case']!={'case_id':cid,'seed':int(cid.split('_seed')[1]),'apple_offset_xy_m':list(positions[int(cid[1])])}:
                raise ValueError('Formal case ID must retain its frozen physical position and seed')
            actual.add(pair);record=records[job['name']]
            if job['config_sha256']!=digest or record['config_file_sha256']!=digest or receipt['frozen_files'].get(str(path))!=digest:
                raise ValueError('Frozen case config SHA mismatch')
            raw=Path(config['output_dir']).resolve()
            if raw!=root/policy/cid or Path(record['output']).resolve()!=raw or record.get('status')!='completed':raise ValueError('Completed record differs from formal case output')
            result_path=raw/'result.json';result=read(result_path)
            if record.get('result_sha256')!=sha(result_path) or result.get('status')!='completed' or result.get('case')!=config['case'] or result.get('policy')!=policy:
                raise ValueError('Completed raw result identity mismatch')
            normalized=dict(config)
            if 'action_limits_file' in config:normalized['action_limits']=config['action_limits_file']
            if read(raw/'config.json')!=normalized:raise ValueError('Saved raw config differs from executed config')
            destination=Path(config['video_output_dir']).resolve()
            if destination.is_relative_to(raw) or destination!=root/'video_exports'/policy/cid:raise ValueError('Only config-bound external video export destinations accepted')
            if destination in seen:raise ValueError('Duplicate external export destination')
            seen.add(destination)
            jobs.append({'name':job['name'],'policy':policy,'case_id':cid,'raw':str(raw),'output':str(destination),
                         'wall_clock':cid==REPRESENTATIVE,'result_sha256':sha(result_path),'trace_sha256':sha(raw/'trajectory.jsonl'),
                         'config_sha256':digest,'batch_receipt':str(receipt_path),'batch_receipt_sha256':sha(receipt_path)})
        if actual!=expected_pairs:raise ValueError('Formal case coverage incomplete')
        phases.append({'phase':phase,'status':'completed','plan':str(plan_path),'plan_sha256':sha(plan_path),'receipt':str(receipt_path),'receipt_sha256':sha(receipt_path)})
    return jobs,phases,errors


def verify_export(job):
    """Full CPU decode and hash check, including compatibility before any reuse."""
    import av
    raw=Path(job['raw']);output=Path(job['output']);manifest_path=output/'video_manifest.json';manifest=read(manifest_path)
    if (manifest.get('schema')!='pi05_g1_round02_video_manifest_v1' or manifest.get('policy')!=job['policy'] or manifest.get('case_id')!=job['case_id']
        or manifest.get('raw_result_sha256')!=job['result_sha256'] or manifest.get('trace_sha256')!=job['trace_sha256']
        or Path(manifest.get('source_case_dir','')).resolve()!=raw):raise ValueError('Existing export is incompatible with immutable raw evidence')
    result=read(raw/'result.json');rows=[json.loads(line) for line in (raw/'trajectory.jsonl').read_text().splitlines()]
    if len(rows)!=result['completed_control_ticks']+1 or any(r['control_tick']!=i for i,r in enumerate(rows)):raise ValueError('Raw control/frame coverage mismatch')
    first,last=rows[0]['monotonic_ns'],rows[-1]['monotonic_ns'];start=min(first,result.get('cold_process_started_monotonic_ns',first))
    wall_seconds=(last-start)/1e9
    if not math.isclose(manifest['timing']['total_preserved_wall_seconds'],wall_seconds,abs_tol=1e-8):raise ValueError('Manifest wall timing differs from trace')
    required=STANDARD+(WALL if job['wall_clock'] else ())
    if set(manifest['videos'])!=set(required):raise ValueError('Existing export has incompatible representative-wall-clock coverage')
    verified={}
    for key in required:
        record=manifest['videos'][key];path=Path(record['path']);path=path if path.is_absolute() else output/path
        expected_path=raw/('head_50fps.mp4' if key=='head_raw' else 'observer_50fps.mp4') if key.endswith('_raw') else output/(key+'.mp4')
        if path.resolve()!=expected_path.resolve():raise ValueError('Manifest path is not the config-bound raw/derivative file: '+key)
        fps=25 if key in WALL else 50;count=math.ceil(wall_seconds*fps)+1 if key in WALL else len(rows)
        kind='measured_wall_clock_resampled' if key in WALL else ('simulation_time_raw' if key.endswith('_raw') else 'simulation_time_with_wall_subtitles')
        if (record.get('fps')!=fps or record.get('frames')!=count or record.get('playback_kind')!=kind
            or record.get('wall_annotation') is not (not key.endswith('_raw')) or sha(path)!=record['sha256']):raise ValueError('Export file SHA/timing/type mismatch: '+key)
        with av.open(str(path)) as container:
            stream=container.streams.video[0]
            if float(stream.average_rate)!=fps:raise ValueError('Decoded FPS differs: '+key)
            n=0
            for n,frame in enumerate(container.decode(stream),1):
                if frame.pts is None or abs(float(frame.pts*frame.time_base)-(n-1)/fps)>1e-7:raise ValueError('Decoded frame time mismatch: '+key)
            if n!=count:raise ValueError('Decoded frame count differs: '+key)
        verified[key]={'path':str(path.resolve()),'sha256':record['sha256'],'fps':fps,'frames':count}
    if sha(raw/'result.json')!=job['result_sha256'] or sha(raw/'trajectory.jsonl')!=job['trace_sha256']:raise ValueError('Raw evidence changed during CPU export validation')
    return {'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),'videos':verified}


def export_job(job,workdir,ledger):
    receipt_path=workdir/(job['name']+'.json');log_path=workdir/(job['name']+'.log')
    receipt={'schema':'pi05_g1_round02_cpu_video_export_job_v1','status':'running','job':job,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    persist(receipt_path,receipt)
    print('CPU_VIDEO_EXPORT_START',job['policy'],job['case_id'],flush=True)
    try:
        closed_ledger(ledger);output=Path(job['output'])
        if output.exists() and any(output.iterdir()):
            # No exporter invocation can overwrite an incompatible/partial prior export.
            receipt.update(reused=True,evidence=verify_export(job))
            log_path.write_text('Reused existing compatible complete export after full SHA/decode verification.\n')
        else:
            command=[sys.executable,'-m','experiments.pi05_g1_round02.video_export','--case-dir',job['raw'],'--output-dir',job['output']]
            if job['wall_clock']:command.append('--wall-clock')
            env=dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
            receipt.update(command=command,cpu_environment_overrides={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
            with log_path.open('x') as log:process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            receipt['exit_code']=process.returncode
            if process.returncode:raise ValueError('CPU video exporter failed; preserve partial exports/log')
            receipt.update(reused=False,evidence=verify_export(job))
        closed_ledger(ledger)
        receipt.update(status='complete',ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except Exception as error:
        receipt.update(status='incomplete',error=repr(error),ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if not log_path.exists():log_path.write_text(repr(error)+'\n')
    persist(receipt_path,receipt);return receipt


def run(evaluation_root,gpu_ledger,workers=4):
    if type(workers) is not int or not 1<=workers<=4:raise ValueError('CPU workers must be 1..4')
    root=Path(evaluation_root).resolve();ledger=Path(gpu_ledger).resolve()
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger_sha=closed_ledger(ledger);jobs,phases,errors=collect_jobs(root)
        workdir=root/'video_export_batches'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);workdir.mkdir(parents=True)
        progress={'schema':'pi05_g1_round02_cpu_video_export_batch_v1','status':'running','cpu_only':True,'workers':workers,
                  'tool_sha256':sha(__file__),'exporter_sha256':sha(ROOT/'experiments/pi05_g1_round02/video_export.py'),
                  'gpu_ledger':str(ledger),'gpu_ledger_sha256':ledger_sha,'gpu_ledger_exclusive_lock_held':True,
                  'representative_rule':'Frozen case p0_seed40 per policy gets additional real wall-clock views; all other cases exactly four standard views.',
                  'phases':phases,'planned_jobs':len(jobs),'errors':errors,'jobs':[]}
        path=workdir/'batch_receipt.json';persist(path,progress)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(export_job,job,workdir,ledger):job for job in jobs}
            for future in as_completed(futures):
                receipt=future.result();progress['jobs'].append(receipt)
                print('CPU_VIDEO_EXPORT',receipt['job']['policy'],receipt['job']['case_id'],receipt['status'],len(progress['jobs']),len(jobs),flush=True)
                persist(path,progress)
        progress['jobs'].sort(key=lambda r:r['job']['name'])
        progress['complete']=not errors and len(progress['jobs'])==len(jobs) and all(r['status']=='complete' for r in progress['jobs'])
        progress.update(status='complete' if progress['complete'] else 'incomplete',ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if closed_ledger(ledger)!=ledger_sha:progress.update(status='incomplete',complete=False);progress['errors'].append('GPU ledger changed during locked CPU export')
        persist(path,progress);return path,progress


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation-root',type=Path,required=True);p.add_argument('--gpu-ledger',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();path,report=run(a.evaluation_root,a.gpu_ledger,a.workers)
    print(json.dumps({'receipt':str(path),'status':report['status'],'jobs':len(report['jobs'])}))
    if not report['complete']:raise SystemExit(1)


if __name__=='__main__':main()
