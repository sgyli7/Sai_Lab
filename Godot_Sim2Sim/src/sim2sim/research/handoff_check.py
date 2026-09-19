"""Additional native skill-exit checks; never replace primary task outcomes."""
import argparse,json,math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from sim2sim.obs import build_obs
from .tasks import TASKS,DT
from .models import NativeAnchor
from .world import World
from .evaluate import record,summarize,PROTOCOL_VERSION


def episode(skill,source,exit_source,seed=100,entry="standing",backend="godot",seconds=3.):
    task=TASKS[skill];actor=NativeAnchor(source);balance=NativeAnchor(exit_source)
    if task.mode not in ("zeros","phase"):raise ValueError("Exit checks require a finite maneuver")
    if balance.time_input_s:raise ValueError("The exit actor must accept an ordinary zero command")
    w=World(task,backend,time_input_s=actor.time_input_s,heading_input=actor.heading_input)
    try:
        obs=w.reset(seed)
        if entry=="standing":obs=w.enter_from_standing()
        primary=[]
        for _ in range(round(task.seconds/DT)):
            a=actor(obs[None])[0];obs=w.step(a);primary.append(record(w,a))
        before=summarize(task,primary,w.heading)
        start=w.t;exit_heading=np.array([math.cos(w.features['yaw']),math.sin(w.features['yaw'])])
        prior=w.last.copy();rows=[];cmd=np.zeros(13,np.float32);jump=None
        for _ in range(round(seconds/DT)):
            # No reset, warmup or action-history erasure at the real boundary.
            obs=build_obs(w.state,w.last,cmd,w.home);a=balance(obs[None])[0]
            if jump is None:jump=float(np.sqrt(np.mean((a-prior)**2)))
            w.step(a);row=record(w,a);row['cmd']=cmd.copy();row['time']-=start;rows.append(row)
        after=summarize(TASKS['standing'],rows,exit_heading)
        dy=w.features['yaw']-math.atan2(w.heading[1],w.heading[0])
        return dict(skill=skill,seed=seed,entry=entry,backend=backend,protocol=PROTOCOL_VERSION,
            actor_sha256=actor.sha256,exit_actor_sha256=balance.sha256,physics=w.physics,
            primary=before,exit=after,action_jump_rms=jump,
            final_original_heading_error_deg=abs(math.degrees(math.atan2(math.sin(dy),math.cos(dy)))),
            success=bool(before['success'] and after['success']))
    finally:w.close()


def suite(skill,source,exit_source,out,seeds=(100,101,102),workers=4,backend="godot"):
    jobs=[(s,e) for s in seeds for e in ('reset','standing')]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(lambda x:episode(skill,source,exit_source,*x,backend),jobs))
    result=dict(check='native_skill_exit_v1',source=str(source),exit_source=str(exit_source),
        primary_success_rate=float(np.mean([r['primary']['success'] for r in rows])),
        exit_success_rate=float(np.mean([r['exit']['success'] for r in rows])),
        combined_success_rate=float(np.mean([r['success'] for r in rows])),episodes=rows)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',required=True);p.add_argument('--source',required=True)
    p.add_argument('--exit-source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seeds',default='100,101,102');p.add_argument('--backend',default='godot')
    a=p.parse_args();r=suite(a.skill,a.source,a.exit_source,a.out,tuple(map(int,a.seeds.split(','))),backend=a.backend)
    print(json.dumps({k:v for k,v in r.items() if k!='episodes'}))


if __name__=='__main__':main()
