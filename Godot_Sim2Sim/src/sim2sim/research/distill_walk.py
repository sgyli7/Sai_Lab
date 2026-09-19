"""Distill standing stability and an existing gait into one neural actor.

Expert switching is used only to collect training demonstrations. Deployment
and physical evaluation use the single exported MLP increment plus its native
anchor, with no command gate or auxiliary standing controller.
"""
from .budget import legacy_deadline
import json,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from .tasks import TASKS,BASELINE,SESSION,conditions,command,DT
from .models import NativeAnchor,Policy,export_policy,parity
from .world import World
from .evaluate import run_suite


def demonstration(job):
    condition,seed=job;task=TASKS["walking"];w=World(task)
    walk=NativeAnchor(BASELINE/"Walk_Godot.onnx");stand=NativeAnchor(TASKS["standing"].source)
    observations=[];actions=[];idle=[];valid=True
    try:
        obs=w.reset(seed,condition)
        for k in range(round(task.seconds/DT)):
            stopped=np.linalg.norm(command(task,w.t,condition)[:3])<.01
            teacher=stand if stopped else walk;action=teacher(obs[None])[0]
            observations.append(obs.copy());actions.append(action);idle.append(stopped)
            if condition=="walk_push" and k==200:w.nudge(np.array([.15,-.12,0]))
            obs=w.step(action)
            valid &= w.features["tilt"]<70 and w.features["z"]>.045
        return dict(obs=np.stack(observations),actions=np.stack(actions),idle=np.array(idle),
                    valid=bool(valid),condition=condition,seed=seed,physics=w.physics)
    finally:w.close()


def main():
    torch.set_num_threads(2);torch.manual_seed(734)
    out=SESSION/"distillation_walk_v1";out.mkdir(exist_ok=True)
    start=time.time();deadline=min(start+20*60,legacy_deadline(SESSION)-3600)
    source=BASELINE/"Walk_Godot.onnx";dataset=out/"demonstrations.npz"
    if not dataset.exists():
        jobs=[(c,60000+s) for c in conditions(TASKS["walking"]) for s in range(3)]
        with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(demonstration,jobs))
        valid=[r for r in results if r["valid"]]
        np.savez_compressed(dataset,**{k:np.concatenate([r[k] for r in valid]) for k in ("obs","actions","idle")},
                            seed=np.concatenate([np.full(len(r["obs"]),r["seed"]) for r in valid]))
        (out/"demonstrations.json").write_text(json.dumps([dict((k,v) for k,v in r.items() if k not in ("obs","actions","idle")) for r in results],indent=2))
    data=np.load(dataset);x=torch.from_numpy(data["obs"]);target=torch.from_numpy(data["actions"])
    actor=Policy(source,"plain");actor.task_name="walking";anchor=actor.anchor_values(x)
    optimizer=torch.optim.Adam(actor.delta.net.parameters(),lr=1e-5)
    training=data["seed"]!=60002;validation=~training
    groups=[torch.from_numpy(np.flatnonzero(training&(data["idle"]==v))) for v in (True,False)]
    config=dict(source=str(source),source_sha256=actor.anchor.sha256,standing_teacher=str(TASKS["standing"].source),
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),train_samples=int(training.sum()),
        validation_samples=int(validation.sum()),start_unix=start,deadline_unix=deadline,learning_rate=1e-5)
    (out/"config.json").write_text(json.dumps(config,indent=2));records=[]
    print(json.dumps(config),flush=True)
    for step in range(1,2001):
        if time.time()>deadline:break
        ix=torch.cat([g[torch.randint(len(g),(256,))] for g in groups])
        predicted=actor(x[ix],anchor[ix]);loss=(predicted-target[ix]).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.delta.net.parameters(),1.);optimizer.step()
        if step%100==0:
            with torch.no_grad():val=float((actor(x[validation],anchor[validation])-target[validation]).square().mean())
            print(json.dumps(dict(step=step,loss=float(loss.detach()),validation_mse=val,elapsed=time.time()-start)),flush=True)
        if step in (250,500,1000,2000):
            path=export_policy(actor,out/f"step_{step:04d}.onnx")
            torch.save(dict(policy=actor.state_dict(),config=config,step=step),out/f"step_{step:04d}.pt")
            check=parity(actor,path,n=1000)
            result=run_suite("walking",path,seeds=(100,101,102),workers=4,out=out/f"eval_{step:04d}",entry="both")
            report=dict(step=step,success=result["success_rate"],score=result["score"],errors=result["errors"],parity=check)
            records.append(report);(out/"evaluations.json").write_text(json.dumps(records,indent=2));print(json.dumps(report),flush=True)
    (out/"completed.json").write_text(json.dumps(dict(elapsed=time.time()-start,step=step),indent=2))


if __name__=="__main__":main()
