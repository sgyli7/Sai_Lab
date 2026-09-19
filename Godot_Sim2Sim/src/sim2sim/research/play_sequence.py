"""Exercise the real PlayBrain across skills in one continuous native world."""
import argparse,json,math
from dataclasses import replace
from pathlib import Path
import numpy as np
from sim2sim.obs import build_obs
from sim2sim.policy import OnnxPolicy
from sim2sim.policy_time import time_command
from sim2sim.play_input import PlayBrain
from sim2sim.play import kick_ball_position
from .world import World
from .tasks import TASKS,DT
from .evaluate import record,summarize,PROTOCOL_VERSION

SEQUENCE=[('idle',1.,set(),None),('forward',2.,{'fwd'},None),('brake',2.,set(),None),
    ('turn',1.,{'left'},None),('idle',1.,set(),None),('ground_pick',4.,set(),'pick'),
    ('idle',2.,set(),None),('kick_left',5.,set(),'kick_left'),('idle',2.,set(),None),
    ('roulade',5.,set(),'roulade'),('idle',2.,set(),None),('sit',6.,set(),'sit'),
    ('rise',6.,set(),'sit'),('kick_right',5.,set(),'kick_right'),('idle',2.,set(),None)]


def run(paths,out,seed=100,scene_robot='microduck_ball'):
    bank={k:OnnxPolicy(Path(p)) for k,p in paths.items()}
    brain=PlayBrain(has_standing='standing' in bank and bank['walking'].has_standing_partner,
        lim=bank['walking'].twist_limits)
    w=World(replace(TASKS['kick_left'],robot=scene_robot));rows=[];segments=[];events=[];active_policy=None
    try:
        w.report_names=list(w.meta)
        w.reset(seed);w.pending_ball=[5.,5.,.035]
        for label,seconds,held,tap in SEQUENCE:
            start=w.t;segment=[];policies=[];heading=w.heading.copy()
            for k in range(round(seconds/DT)):
                control=brain.tick(held,[tap] if tap and k==0 else [],DT)
                actor=bank[control.policy];cmd=control.command
                if active_policy!=control.policy:
                    actor.reset_context();active_policy=control.policy
                if control.started_skill:
                    yaw=w.features['yaw'];w.heading=np.array([math.cos(yaw),math.sin(yaw)]);heading=w.heading.copy()
                    events.append(dict(time=w.t,skill=control.started_skill,q=w.state.q.tolist(),
                        last_action=w.last.tolist(),gyro=w.features['gyro'].tolist(),tilt=w.features['tilt'],z=w.features['z']))
                if actor.time_input_s:
                    duration=brain.roulade_duration if control.policy=='roulade' else (
                        brain.kick_duration if control.policy in ('kick_left','kick_right') else 0.)
                    if not duration or actor.time_input_s!=duration:raise ValueError('Unexpected timed skill')
                    cmd=time_command(duration-brain.behavior_t,actor.time_input_s,
                        w.features['rot'] if actor.heading_input else None,w.heading)
                if control.started_skill in ('kick_left','kick_right'):
                    w.pending_ball=kick_ball_position(w.state,control.started_skill)
                # The task tag affects telemetry only. The native body/ball world
                # remains continuous and action history is never erased.
                w.task=TASKS[control.policy]
                obs=build_obs(w.state,w.last,cmd,w.home);a=actor.infer(obs);w.step(a)
                row=record(w,a);row['cmd']=cmd.copy();row['time']-=start
                segment.append(row);rows.append(dict(time=w.t,label=label,policy=control.policy,
                    z=w.features['z'],tilt=w.features['tilt'],yaw=w.features['yaw']))
                policies.append(control.policy)
            if label in ('ground_pick','kick_left','kick_right','roulade'):
                summary=summarize(TASKS[label],segment,heading)
            elif label in ('idle','brake'):
                summary=summarize(TASKS['standing'],segment,heading)
            else:
                summary=dict(final_z=w.features['z'],final_tilt=w.features['tilt'],
                    unintended_fall=any(x['z']<.045 or x['tilt']>70 for x in segment))
            item=dict(label=label,start=start,seconds=seconds,policies=sorted(set(policies)),metrics=summary)
            if label in ('idle','brake'):
                item['settled_metrics']=summarize(TASKS['walking'],segment,heading)
            segments.append(item)
        result=dict(check='native_play_sequence_v2',protocol=PROTOCOL_VERSION,seed=seed,paths=paths,
            physics=w.physics,seconds=w.t,resets_after_start=0,events=events,segments=segments,trajectory=rows)
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));return result
    finally:w.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=100)
    p.add_argument('--scene-robot',choices=['microduck_ball','microduck_ball_stand_fix'],default='microduck_ball_stand_fix')
    a=p.parse_args();r=run(json.loads(a.bank.read_text()),a.out,a.seed,a.scene_robot)
    for s in r['segments']:print(s['label'],s['policies'],s['metrics'])


if __name__=='__main__':main()
