"""Independent native roller tasks, with the source command contract."""
import argparse,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from .tasks import TASKS,DT
from .world import World
from .models import NativeAnchor
from .evaluate import record
from .roller_tasks import CONDITIONS,PROTOCOL,initial_speed,summarize


def episode(source,condition,seed,backend='godot',reference_profile='xml',trace=None):
    actor=NativeAnchor(source);w=World(TASKS['roller'],backend,reference_profile=reference_profile,roller_contract=True,state_input=actor.state_input,task_input=actor.task_input)
    try:
        obs=w.reset(seed,condition);rows=[]
        for _ in range(round(w.task.seconds/DT)):
            a=actor(obs[None])[0];obs=w.step(a);row=record(w,a);row['condition']=condition;rows.append(row)
        result=summarize(rows,w.heading,w.roller_target_yaw,initial_speed(condition))
        result.update(seed=seed,backend=backend,physics=w.physics,sha256=actor.sha256,entry='task_initial_state')
        if trace:
            trace=Path(trace);trace.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(trace,**{k:np.asarray([r[k] for r in rows]) for k in rows[0] if k!='condition'})
            result['trace']=str(trace)
        return result
    finally:w.close()


def run_suite(skill,onnx,backend='godot',seeds=(100,101,102),workers=4,out=None,selected_conditions=None,entry='reset',reference_profile='xml'):
    if skill!='roller' or entry!='reset':raise ValueError('Roller tasks define their own rest/moving starts')
    if out:Path(out).mkdir(parents=True,exist_ok=True)
    jobs=[(c,s) for c in (selected_conditions or CONDITIONS) for s in seeds];rows=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(episode,onnx,c,s,backend,reference_profile,None if out is None else Path(out)/f'{c}_{s}.npz'):(c,s) for c,s in jobs}
        for f in as_completed(futures):
            c,s=futures[f]
            try:rows.append(f.result())
            except Exception as e:rows.append(dict(condition=c,seed=s,success=False,score=0.,error=repr(e)))
    rows.sort(key=lambda r:(r['condition'],r['seed']))
    result=dict(protocol=PROTOCOL,skill=skill,onnx=str(onnx),backend=backend,entry='task_initial_state',
        success_rate=float(np.mean([r['success'] for r in rows])),score=float(np.mean([r['score'] for r in rows])),
        errors=sum('error' in r for r in rows),episodes=rows)
    if out:(Path(out)/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',default='godot');p.add_argument('--reference-profile',default='xml')
    p.add_argument('--seed-start',type=int,default=100);p.add_argument('--seeds',type=int,default=3)
    a=p.parse_args();r=run_suite('roller',a.onnx,a.backend,range(a.seed_start,a.seed_start+a.seeds),out=a.out,reference_profile=a.reference_profile)
    print(json.dumps({k:v for k,v in r.items() if k!='episodes'},indent=2))


if __name__=='__main__':main()
