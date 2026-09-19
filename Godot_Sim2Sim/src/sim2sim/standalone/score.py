"""Score player traces with the unchanged research telemetry and task scorer."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from sim2sim.coords import quat_wxyz_to_mat
from sim2sim.paths import load_robot_json,sim2sim_root
from sim2sim.research.evaluate import record,summarize,PROTOCOL_VERSION
from sim2sim.research.tasks import TASKS,DT
from sim2sim.research.world import World
from sim2sim.train.reset_poses import HomePoseSampler
from .replay import raw_state


def moving_average(values,count):
    values=np.asarray(values,float)
    return np.convolve(values,np.ones(count)/count,mode='valid')


def brake_metrics(rows,brake_at,end):
    selected=[row for row in rows if brake_at-1e-9<=row['time']<=end+1e-9]
    if not selected:raise ValueError('Brake interval has no samples')
    times=np.array([row['time'] for row in selected])-brake_at
    speed=np.array([np.linalg.norm(row['vel'][:2]) for row in selected])
    xy=np.array([row['xy'] for row in selected])
    initial_speed=float(speed[0]);yaw=float(selected[0]['yaw'])
    # Project world travel onto the heading at S press, preserving signs during turns.
    forward=np.array([row['vel'][0]*math.cos(row['yaw']-yaw)-row['vel'][1]*math.sin(row['yaw']-yaw) for row in selected])
    stable=np.array([row['tilt']<15 and row['z']>.08 and np.sum(row['contact'])>0 for row in selected])
    fell=np.array([row['tilt']>70 or row['z']<.055 for row in selected])
    speed_window=moving_average(speed,10)
    stable_window=stable[9:]
    confirmed=None;onset=None
    for index in range(49,len(speed_window)):
        window=slice(index-49,index+1)
        if np.all(speed_window[window]<.05) and np.mean(stable_window[window])>=.9:
            confirmed=index+9;onset=index-49+9;break
    final_index=len(times)-1 if confirmed is None else confirmed
    distance=float(np.linalg.norm(np.diff(xy[:final_index+1],axis=0),axis=1).sum())
    distance_limit=max(.15,initial_speed*1.2)
    backward=moving_average(forward,10)<-.05
    sustained_backwards=any(np.all(backward[i:i+10]) for i in range(max(0,len(backward)-9)))
    confirmation_time=None if confirmed is None else float(times[confirmed])
    success=bool(not fell.any() and confirmed is not None and confirmation_time<=2.+1e-9
                 and distance<=distance_limit and not sustained_backwards)
    return dict(brake_at=brake_at,initial_speed=initial_speed,stop_onset_s=None if onset is None else float(times[onset]),
        stop_confirmed_s=confirmation_time,braking_distance=distance,distance_limit=distance_limit,
        fell=bool(fell.any()),first_fall_s=next((float(t) for t,f in zip(times,fell) if f),None),
        sustained_backwards=bool(sustained_backwards),minimum_forward_02s=float(moving_average(forward,10).min()),
        final_speed=float(speed[-50:].mean()),max_tilt=float(max(row['tilt'] for row in selected)),
        success=success)


class TraceWorld:
    """Only MuJoCo metadata/FK is loaded; no physical rollout or TCP is started."""
    _godot_bodies=World._godot_bodies
    measure=World.measure

    def __init__(self,mode,skill):
        robot='microduck_roller' if mode=='roller' else 'microduck_ball_stand_fix'
        self.task=replace(TASKS[skill],robot=robot)
        cfg=load_robot_json(sim2sim_root()/'robots'/f'{robot}.json')
        self.sampler=HomePoseSampler(cfg);self.mj=self.sampler.mj
        self.backend_name='godot'
        self.meta={mujoco.mj_id2name(self.mj.model,mujoco.mjtObj.mjOBJ_BODY,i):i for i in range(1,self.mj.model.nbody)}
        self.site_id=mujoco.mj_name2id(self.mj.model,mujoco.mjtObj.mjOBJ_SITE,'mouth_tip')
        self.report_names=[]
        spec=json.loads(Path(cfg['godot_spec']).read_text())
        base=next(b for b in spec['bodies'] if b['name']==cfg.get('base_body','trunk_base'))
        self.robot=dict(ipos=base['ipos'],iquat=base['iquat_wxyz'])

    def append_state(self,raw,action,command,time):
        self.state=raw_state(raw,self.robot);self.state.extra={'raw':raw}
        self.report_names=[b['name'] for b in raw['body_states'] if b['name'] in self.meta]
        self.features=self.measure()
        self.executed_command=np.array(command,np.float32);self.t=time
        return record(self,action)

    def close(self):self.sampler.close()


def score(trace_path,case_path):
    trace=json.loads(Path(trace_path).read_text());case=json.loads(Path(case_path).read_text())
    if 'training_exploration' in trace['summary'] or 'training_exploration' in case:
        raise ValueError('Exploratory training rollouts cannot be used as acceptance evidence')
    if trace['summary'].get('error'):raise RuntimeError(trace['summary']['error'])
    if trace['summary'].get('resets') or trace['summary'].get('switches'):
        raise ValueError('Task acceptance cases cannot mask outcomes with resets or robot replacement')
    world=TraceWorld(case['mode'],case['skill']);rows=[]
    try:
        actions=trace['rows']
        for index,source in enumerate(actions):
            # The controller records the state before its action. Research scoring
            # consumes the state after that same action, including accumulated contacts.
            raw=actions[index+1]['raw'] if index+1<len(actions) else trace['summary']['final_raw']
            requested=source.get('requested_command',source['command'])
            if 'requested_command' not in source and trace['summary'].get('control_config',{}).get(case['mode'],{}):
                raise ValueError('Controlled traces must distinguish requested from policy commands; replay with the current recorder')
            rows.append(world.append_state(raw,source['action'],requested,source['t']+DT))
        start=float(case['scoring']['start']);end=float(case['scoring']['end'])
        task_rows=[{**row,'time':row['time']-start} for row in rows if start+1e-9<row['time']<=end+1e-9]
        first=next(source for source in actions if source['t']>=start-1e-9)
        rotation=quat_wxyz_to_mat(first['body']['base_quat']);yaw=math.atan2(rotation[1,0],rotation[0,0])
        heading=np.array([math.cos(yaw),math.sin(yaw)])
        metrics=summarize(world.task,task_rows,heading)
        if case.get('protocol')=='walking_sprint_v1':
            from .sprint import metrics as sprint_metrics
            metrics=sprint_metrics(rows,actions,case)
        result=dict(case=case['case'],skill=case['skill'],seed=case.get('seed'),
            randomized_start=case.get('randomized_start',False),protocol=case['protocol'],
            task_protocol=PROTOCOL_VERSION,models=trace['summary']['models'],task_metrics=metrics)
        result['control_config']=trace['summary'].get('control_config',{})
        result['controller_scoring_protocol']='requested_command_v1'
        if case.get('brake_times'):
            brakes=[]
            for at in case['brake_times']:
                following=[s['at'] for s in case['segments'] if s['at']>at and 'fwd' in s.get('held',[])]
                brakes.append(brake_metrics(rows,at,min(following,default=case['seconds'])))
            result['brakes']=brakes;result['brake_success']=all(b['success'] for b in brakes)
        moving=[row for row,source in zip(rows,actions) if 'fwd' in source['held']]
        result['moving_mean_vx']=float(np.mean([row['vel'][0] for row in moving])) if moving else None
        translating=[row for row,source in zip(rows,actions) if set(source['held'])&{'fwd','back','strafe_l','strafe_r'}]
        result['moving_mean_speed']=float(np.mean([np.linalg.norm(row['vel'][:2]) for row in translating])) if translating else None
        result['trace']=str(Path(trace_path).resolve())
        return result
    finally:world.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('trace');parser.add_argument('case')
    parser.add_argument('--out',type=Path)
    a=parser.parse_args();result=score(a.trace,a.case);text=json.dumps(result,indent=2,allow_nan=False)
    if a.out:a.out.write_text(text+'\n')
    print(text)
