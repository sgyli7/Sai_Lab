"""Python reference replay and shadow audit for standalone Godot trajectories."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from sim2sim.backends import SimState
from sim2sim.backends.godot_backend import GodotBackend, inertial_to_body
from sim2sim.coords import quat_wxyz_to_mat
from sim2sim.fall import fallen, tilt_deg
from sim2sim.obs import build_obs
from sim2sim.motion_control import MotionControl
from sim2sim.paths import sim2sim_root
from sim2sim.play import kick_ball_position, ROLLER_LIMITS
from sim2sim.play_input import PlayBrain
from sim2sim.policy import OnnxPolicy
from sim2sim.policy_time import time_command


def deployment(project=None):
    root=Path(project) if project else sim2sim_root()/'godot'
    return json.loads((root/'runtime_assets/deployment.json').read_text())


def bank_and_brain(config, mode, control_config=None, project=None):
    root=Path(project) if project else sim2sim_root()/'godot'
    bank={skill:OnnxPolicy(root/item['path'].removeprefix('res://'))
          for skill,item in config['policies'].items()}
    roller=mode=='roller'
    control_config=config.get('control_config',{}) if control_config is None else control_config
    limits=ROLLER_LIMITS if roller else bank['walking'].twist_limits
    overrides=control_config.get(mode,{}).get('twist_limits',{})
    if not all(np.isfinite(value) for value in overrides.values()):raise ValueError('Nonfinite control limit')
    limits=replace(limits,**overrides)
    brain=PlayBrain(has_standing=not roller and bank['walking'].has_standing_partner,
        has_sitstand=not roller,has_pick=not roller,has_kick_left=not roller,
        has_kick_right=not roller,has_roulade=not roller,has_roller_crouch=roller,has_stand_hold=not roller,
        lim=limits,has_sprint=not roller and 'sprint' in bank)
    brain.motion_control=MotionControl(control_config.get(mode,{}))
    brain.roller_support_groups=config['robots'][mode].get('support_groups')
    return bank,brain


def control(brain,bank,state,held,taps,order,last,home,heading,mode):
    out=brain.tick(set(held),taps,.02,press_order=order)
    skill='roller' if mode=='roller' and out.policy=='walking' else out.policy
    if mode=='walk' and out.sprint:skill='sprint'
    actor=bank[skill]
    cmd=out.command
    if actor.time_input_s:
        rotation=quat_wxyz_to_mat(state.base_quat_wxyz)
        if out.started_skill==skill:
            yaw=np.arctan2(rotation[1,0],rotation[0,0]);heading[:]=[np.cos(yaw),np.sin(yaw)]
        cmd=time_command(actor.time_input_s-brain.behavior_t,actor.time_input_s,
                         rotation if actor.heading_input else None,heading)
    cmd=brain.motion_control.command(cmd,state,skill)
    obs=build_obs(state,last,cmd,home)
    from sim2sim.policy_state import inject_state
    obs=inject_state(obs,state,actor.state_input)
    if actor.task_state is not None:
        from sim2sim.policy_task_state import contacts_from_raw
        contacts=contacts_from_raw(state.extra['raw'],brain.roller_support_groups)
        obs=actor.task_state.observe(obs,contacts,state.t)
    elif 'roller' in bank:
        bank['roller'].reset_context()
    action=actor.infer(obs)
    ball=kick_ball_position(state,out.started_skill) if out.started_skill in ('kick_left','kick_right') else None
    return skill,cmd,obs,action,ball,out


def raw_state(raw, robot):
    pos,quat=inertial_to_body(raw['base_pos'],raw['base_quat'],robot['ipos'],robot['iquat'])
    return SimState(t=raw['t'],q=np.array(raw['q']),qd=np.array(raw['qd']),base_pos=pos,
        base_quat_wxyz=quat,base_linvel=np.array(raw['base_linvel']),
        base_angvel_local=np.array(raw['base_angvel_local']),extra={'raw':raw})


def shadow(trace,project=None):
    data=json.loads(Path(trace).read_text());cfg=deployment(project)
    if data['summary']['models']!={k:v['sha256'] for k,v in cfg['policies'].items()}:
        raise ValueError('Shadow model bank does not match trace; use the frozen suite --project')
    previous_mode=None;bank={};brain=None
    maxima={key:0. for key in ('command','obs','action','ctrl','last_action','body_pos','body_quat')}
    mismatches=[];last=np.zeros(14,np.float32);heading=np.array([1.,0.])
    previous_episode_t=-1.
    for index,row in enumerate(data['rows']):
        mode=row['mode'];robot=cfg['robots'][mode];home=np.array(robot['home'],np.float32)
        if mode!=previous_mode:
            bank,brain=bank_and_brain(cfg,mode,data['summary'].get('control_config'),project);last[:]=0;heading[:]=[1.,0.]
            previous_mode=mode;previous_episode_t=-1.
        elif row['episode_t']<previous_episode_t:
            brain.reset_motion();brain.motion_control.reset();last[:]=0;heading[:]=[1.,0.]
            for actor in bank.values():actor.reset_context()
        previous_episode_t=row['episode_t']
        state=raw_state(row['raw'],robot)
        skill,cmd,obs,action,_,_=control(brain,bank,state,row['held'],row['taps'],row['order'],last,home,heading,mode)
        if skill!=row['skill']:mismatches.append(dict(index=index,expected=skill,actual=row['skill']))
        pairs=dict(command=(cmd,row['command']),obs=(obs,row['obs']),action=(action,row['action']),
          ctrl=(home+action*robot['action_scale'],row['ctrl']),last_action=(last,row['last_action']),
          body_pos=(state.base_pos,row['body']['base_pos']),body_quat=(state.base_quat_wxyz,row['body']['base_quat']))
        for key,(expected,actual) in pairs.items():
            error=float(np.max(np.abs(np.asarray(expected)-actual)));maxima[key]=max(maxima[key],error)
        # Teacher forcing isolates each step; do not accumulate inference rounding in shadow mode.
        last=np.asarray(row['action'],np.float32)
    return dict(check='native_shadow',rows=len(data['rows']),max_abs=maxima,
        skill_mismatches=mismatches[:20],passed=not mismatches and max(maxima.values())<1e-5)


def reference(replay_path, output):
    replay=json.loads(Path(replay_path).read_text());cfg=deployment();mode=replay.get('mode','walk')
    controls=replay.get('control_config',cfg.get('control_config',{}))
    robot=cfg['robots'][mode];bank,brain=bank_and_brain(cfg,mode,controls)
    home=np.array(robot['home'],np.float32);last=np.zeros(14,np.float32);heading=np.array([1.,0.])
    backend=GodotBackend(sim2sim_root()/'godot'/robot['spec'].removeprefix('res://'),
        headless=True,current_limit_a=robot['current_limit_a'],recv_timeout=30)
    rows=[];previous=-1
    try:
        poses=replay.get('initial_poses',{}).get(mode,robot['poses'])
        state=backend.reset(ctrl=home,bodies=poses,report_bodies=[b['name'] for b in backend.spec['bodies']])
        for tick in range(round(replay['seconds']/.02)):
            elapsed=tick*.02
            selected=max((i for i,s in enumerate(replay['segments']) if s.get('at',0)<=elapsed+1e-9),default=-1)
            segment=replay['segments'][selected] if selected>=0 else {}
            held=segment.get('held',[]);order=segment.get('order',held)
            taps=segment.get('taps',[]) if selected!=previous else [];previous=selected
            skill,cmd,obs,action,ball,out=control(brain,bank,state,held,taps,order,last,home,heading,mode)
            if out.switch_robot:raise ValueError('Reference runner: split robot-switch cases into episodes')
            if out.reset:
                state=backend.reset(ctrl=home,bodies=robot['poses'],
                    report_bodies=[b['name'] for b in backend.spec['bodies']])
                brain.reset_motion();last[:]=0
                brain.motion_control.reset()
                for actor in bank.values():actor.reset_context()
                continue
            ctrl=home+action*robot['action_scale']
            rows.append(dict(t=elapsed,mode=mode,skill=skill,command=cmd.tolist(),obs=obs.tolist(),
              requested_command=(out.command if skill in ('walking','roller','sprint') else cmd).tolist(),
              action=action.tolist(),ctrl=ctrl.tolist(),held=held,taps=taps,order=order,body=dict(base_pos=state.base_pos.tolist(),
              base_quat=state.base_quat_wxyz.tolist()),raw=state.extra['raw'],
              fell=fallen(state.base_quat_wxyz,state.base_pos),tilt=tilt_deg(state.base_quat_wxyz)))
            last=action.astype(np.float32)
            state=backend.step(ctrl,n_substeps=4,report='research',place_ball=ball)
    finally:backend.close()
    summary=dict(check='python_reference',steps=len(rows),first_fall=next((x['t'] for x in rows if x['fell']),None),
                 models={k:v['sha256'] for k,v in cfg['policies'].items()},final_raw=state.extra['raw'],control_config=controls)
    Path(output).write_text(json.dumps(dict(summary=summary,rows=rows)))
    return summary


def paired(reference_path,native_path):
    ref=json.loads(Path(reference_path).read_text())['rows'];native=json.loads(Path(native_path).read_text())['rows']
    errors={key:[] for key in ['obs','action','ctrl']}
    for a,b in zip(ref,native):
        for key in errors:errors[key].append(float(np.max(np.abs(np.array(a[key])-b[key]))))
    return dict(check='closed_loop_pair',steps=[len(ref),len(native)],
                max_abs={key:max(values) for key,values in errors.items()},
                first_difference={key:next((i for i,v in enumerate(values) if v>1e-5),None) for key,values in errors.items()},
                first_ten={key:values[:10] for key,values in errors.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('shadow');s.add_argument('trace');s.add_argument('--project',type=Path)
    s=sub.add_parser('reference');s.add_argument('replay');s.add_argument('--out',required=True)
    s=sub.add_parser('paired');s.add_argument('reference');s.add_argument('native')
    a=p.parse_args()
    if a.cmd=='shadow':result=shadow(a.trace,a.project)
    elif a.cmd=='reference':result=reference(a.replay,a.out)
    else:result=paired(a.reference,a.native)
    print(json.dumps(result,indent=2))
    if result.get('passed') is False:raise SystemExit(1)


if __name__=='__main__':main()
