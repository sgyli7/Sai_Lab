"""Training-only native roll/stand demonstrations for a single timed student."""
from .budget import legacy_deadline
import argparse,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from .tasks import TASKS,SESSION,DT
from .models import NativeAnchor,Policy,export_policy,parity
from .world import World
from .evaluate import record,summarize,run_suite
from .time_input import prepare


def demonstration(job):
    source,seed,entry,out=job
    roll=NativeAnchor(source);stand=NativeAnchor(TASKS['standing'].source)
    w=World(TASKS['roulade'],time_input_s=5.);rows=[];observations=[];actions=[];is_stand=[]
    switched=False;switch_time=None;net=0.;pivot=False;inverted=False
    try:
        obs=w.reset(seed)
        if entry=='standing':obs=w.enter_from_standing()
        for k in range(250):
            f=w.features
            if not switched and pivot and inverted and 4.5<net<7.5 and f['up']>.85 and f['z']>.09 and f['contact'].sum()>0:
                switched=True;switch_time=w.t
            teacher_obs=obs.copy();teacher_obs[48:]=0.
            a=(stand if switched else roll)(teacher_obs[None])[0]
            observations.append(obs.copy());actions.append(a.copy());is_stand.append(switched)
            obs=w.step(a);rows.append(record(w,a));f=w.features;net+=float(f['gyro'][1])*DT
            pivot|=bool(f['head_contact'] and f['head_up']<-.3 and .35<net<2.97)
            inverted|=bool(pivot and f['up']<-.7)
        result=summarize(w.task,rows,w.heading)
        path=Path(out)/f'{entry}_{seed}.npz'
        np.savez_compressed(path,obs=np.stack(observations),actions=np.stack(actions),is_stand=np.array(is_stand),
            **{k:np.asarray([r[k] for r in rows]) for k in rows[0] if k!='actions'})
        result.update(seed=seed,entry=entry,trace=str(path),teacher_switch_time=switch_time,
            teacher_demonstration_only=True,roll_teacher=str(source),roll_teacher_sha256=roll.sha256,physics=w.physics)
        return result
    finally:w.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--name',default='distillation_roll_recovery_v1')
    p.add_argument('--probe',action='store_true');p.add_argument('--seeds',type=int,default=12)
    p.add_argument('--time-gate',default='');p.add_argument('--demonstrations',type=Path)
    args=p.parse_args();root=SESSION/args.name;root.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(929)
    start=time.time();deadline=min(start+25*60,legacy_deadline(SESSION)-3600)
    if args.demonstrations:
        reports=json.loads(args.demonstrations.read_text());parent_hash=NativeAnchor(args.source).sha256
        if any(r['roll_teacher_sha256']!=parent_hash for r in reports):raise ValueError('Demonstration parent mismatch')
    else:
        jobs=[(args.source,90000+s,entry,root) for s in range(args.seeds) for entry in ('reset','standing')]
        with ThreadPoolExecutor(max_workers=4) as pool:reports=list(pool.map(demonstration,jobs))
    (root/'demonstrations.json').write_text(json.dumps(reports,indent=2));valid=[r for r in reports if r['success']]
    print(json.dumps(dict(teacher_only=True,successful_demos=len(valid),total=len(reports),switch_times=[r['teacher_switch_time'] for r in reports])),flush=True)
    if args.probe:return
    if len(valid)<4:raise RuntimeError('Too few complete native training demonstrations')
    data=[]
    for r in valid:
        with np.load(r['trace']) as a:data.append({k:a[k] for k in ('obs','actions','is_stand')})
    x=torch.from_numpy(np.concatenate([d['obs'] for d in data]));target=torch.from_numpy(np.concatenate([d['actions'] for d in data]))
    post=np.concatenate([d['is_stand'] for d in data]);groups=[torch.from_numpy(np.flatnonzero(post==v)) for v in (False,True)]
    source=prepare(args.source,root/'time_teacher.onnx')
    time_gate=None if not args.time_gate else tuple(float(x) for x in args.time_gate.split(','))
    actor=Policy(source,'plain',template=TASKS['roulade'].source,time_gate=time_gate);actor.task_name='roulade';anchor=actor.anchor_values(x)
    optimizer=torch.optim.Adam(actor.delta.net.parameters(),lr=1e-5);best_key=(-1.,-1.);evaluations=[]
    config=dict(source=str(args.source),timed_anchor_sha256=actor.anchor.sha256,start_unix=start,deadline_unix=deadline,
        teacher_demonstrations=len(valid),training_samples=len(x),learning_rate=1e-5,time_gate=args.time_gate)
    (root/'config.json').write_text(json.dumps(config,indent=2))
    for step in range(1,5001):
        if time.time()>deadline:break
        ix=torch.cat([g[torch.randint(len(g),(256,))] for g in groups])
        loss=(actor(x[ix],anchor[ix])-target[ix]).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.delta.net.parameters(),1.);optimizer.step()
        if step%250==0:print(json.dumps(dict(step=step,loss=float(loss.detach()),elapsed=time.time()-start)),flush=True)
        if step in (500,1000,2000,3000,5000):
            path=export_policy(actor,root/f'step_{step:04d}.onnx');check=parity(actor,path,n=1000)
            if not check['passed']:raise RuntimeError('Nonzero student export parity failed')
            result=run_suite('roulade',path,seeds=(100,101,102),workers=4,out=root/f'eval_{step}',entry='both')
            rec=dict(step=step,success=result['success_rate'],score=result['score'],errors=result['errors'],parity=check,path=str(path))
            evaluations.append(rec);(root/'evaluations.json').write_text(json.dumps(evaluations,indent=2));print(json.dumps(rec),flush=True)
            torch.save(dict(policy=actor.state_dict(),optimizer=optimizer.state_dict(),config=config,step=step),root/f'step_{step:04d}.pt')
            key=(result['success_rate'],result['score'])
            if not result['errors'] and key>best_key:
                best_key=key;(root/'best.onnx').write_bytes(path.read_bytes());(root/'best.json').write_text(json.dumps(rec,indent=2))
    (root/'completed.json').write_text(json.dumps(dict(step=step,elapsed=time.time()-start),indent=2))


if __name__=='__main__':main()
