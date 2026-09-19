"""On-policy state aggregation for walking-to-standing transitions.

The student controls every collection rollout. Teachers only label its states:
factory standing at zero command, the verified parent at moving commands.
Every physical evaluation executes the single exported student throughout.
"""
from .budget import legacy_deadline
import hashlib,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from .tasks import TASKS,BASELINE,SESSION,DT,conditions
from .models import NativeAnchor,Policy,export_policy,parity
from .world import World
from .evaluate import run_suite


def collect(job):
    student_path,parent_path,condition,seed,entry=job
    student=NativeAnchor(student_path);parent=NativeAnchor(parent_path)
    stand=NativeAnchor(TASKS["standing"].source);w=World(TASKS["walking"])
    observations=[];targets=[];idle=[];fell=False
    try:
        obs=w.reset(seed,condition)
        if entry=="standing":obs=w.enter_from_standing()
        for k in range(round(w.task.seconds/DT)):
            stopped=np.linalg.norm(w.command()[:3])<.01
            target=(stand if stopped else parent)(obs[None])[0]
            observations.append(obs.copy());targets.append(target);idle.append(stopped)
            obs=w.step(student(obs[None])[0])
            if w.features["tilt"]>70 or w.features["z"]<.045:
                fell=True;break
        return dict(obs=np.stack(observations),target=np.stack(targets),idle=np.array(idle),
            seed=np.full(len(idle),seed),condition=condition,entry=entry,fell=fell,physics=w.physics)
    finally:w.close()


def main():
    torch.set_num_threads(2);torch.manual_seed(819)
    root=SESSION/"distillation_transitions_v1";root.mkdir(exist_ok=False)
    start=time.time();deadline=min(start+25*60,legacy_deadline(SESSION)-3600)
    parent=SESSION/"walking_heading_probe/coupling_0.9.onnx";student=parent
    actor=Policy(parent,"plain",template=BASELINE/"Walk_Godot.onnx");actor.task_name="walking"
    optimizer=torch.optim.Adam(actor.delta.net.parameters(),lr=5e-6)
    config=dict(parent=str(parent),parent_sha256=actor.anchor.sha256,teacher=str(TASKS["standing"].source),
        start_unix=start,deadline_unix=deadline,learning_rate=5e-6,student_controls_collection=True)
    (root/"config.json").write_text(json.dumps(config,indent=2));datasets=[];evaluations=[]
    best_key=(-1.,-1.);last_step=0
    for stage in range(2):
        if time.time()>deadline:break
        cases=conditions(TASKS["walking"])+["random_seq"]*3
        jobs=[(student,parent,c,80000+stage*10+s,"standing" if s%2 else "reset") for c in cases for s in range(3)]
        with ThreadPoolExecutor(max_workers=4) as pool:parts=list(pool.map(collect,jobs))
        data={k:np.concatenate([p[k] for p in parts]) for k in ("obs","target","idle","seed")};datasets.append(data)
        dataset=root/f"states_{stage}.npz";np.savez_compressed(dataset,**data)
        (root/f"states_{stage}.json").write_text(json.dumps([dict((k,v) for k,v in p.items() if k not in ("obs","target","idle","seed")) for p in parts],indent=2))
        (root/f"states_{stage}.sha256").write_text(hashlib.sha256(dataset.read_bytes()).hexdigest())
        combined={k:np.concatenate([d[k] for d in datasets]) for k in data}
        x=torch.from_numpy(combined["obs"]);target=torch.from_numpy(combined["target"]);anchor=actor.anchor_values(x)
        training=combined["seed"]%10!=2;validation=~training
        groups=[torch.from_numpy(np.flatnonzero(training&(combined["idle"]==v))) for v in (True,False)]
        print(json.dumps(dict(stage=stage,samples=len(x),idle=int(combined['idle'].sum()),collection_falls=sum(p['fell'] for p in parts))),flush=True)
        for step in range(1,1501):
            if time.time()>deadline:break
            ix=torch.cat([g[torch.randint(len(g),(256,))] for g in groups])
            loss=(actor(x[ix],anchor[ix])-target[ix]).square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.delta.net.parameters(),1.);optimizer.step()
            last_step=stage*1500+step
            if step%250==0:
                with torch.no_grad():val=float((actor(x[validation],anchor[validation])-target[validation]).square().mean())
                print(json.dumps(dict(stage=stage,step=step,loss=float(loss.detach()),validation_mse=val,elapsed=time.time()-start)),flush=True)
            if step in (500,1000,1500):
                path=export_policy(actor,root/f"stage_{stage}_step_{step:04d}.onnx")
                check=parity(actor,path,n=1000)
                if not check['passed']:raise RuntimeError('Export parity failed')
                result=run_suite("walking",path,seeds=(100,101,102),workers=4,out=root/f"eval_{stage}_{step}",entry="both")
                falls=sum(e.get('fell',True) for e in result['episodes'])
                report=dict(stage=stage,step=step,success=result['success_rate'],score=result['score'],falls=falls,errors=result['errors'],parity=check,path=str(path))
                evaluations.append(report);(root/'evaluations.json').write_text(json.dumps(evaluations,indent=2));print(json.dumps(report),flush=True)
                key=(result['success_rate'],result['score'])
                if not falls and not result['errors'] and key>best_key:
                    best_key=key;(root/'best.onnx').write_bytes(path.read_bytes());(root/'best.json').write_text(json.dumps(report,indent=2))
        # Aggregate another state distribution from the current student, even
        # when an earlier checkpoint remains the best deployable candidate.
        student=export_policy(actor,root/f"stage_{stage}_collector.onnx")
        torch.save(dict(policy=actor.state_dict(),optimizer=optimizer.state_dict(),config=config,step=last_step),root/f"stage_{stage}.pt")
    (root/'completed.json').write_text(json.dumps(dict(elapsed=time.time()-start,step=last_step),indent=2))


if __name__=="__main__":main()
