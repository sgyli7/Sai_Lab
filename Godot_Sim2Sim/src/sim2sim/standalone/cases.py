"""Versioned keyboard cases and precomputed starts for independent native replay."""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from sim2sim.paths import load_robot_json, sim2sim_root
from sim2sim.train.reset_poses import HomePoseSampler

PROTOCOL='standalone_keyboard_v1'


def brake_case(push=3.,after=None):
    brake=1.+push
    return dict(mode='roller',skill='roller',seconds=brake+6.,brake_times=[brake],
        segments=[dict(at=0.,held=[]),dict(at=1.,held=['fwd']),
                  dict(at=brake,held=['back'] if after is None else after)],
        scoring=dict(start=0.,end=brake+6.))


def standard_cases():
    cases={}
    cases['standing']=dict(mode='walk',skill='standing',seconds=10.,
        segments=[dict(at=0.,taps=['stand'])],scoring=dict(start=0.,end=10.))
    cases['walking_straight']=dict(mode='walk',skill='walking',seconds=8.,
        segments=[dict(at=0.,held=[]),dict(at=1.,held=['fwd']),dict(at=4.,held=[])],scoring=dict(start=0.,end=8.))
    cases['walking_forward_long']=dict(mode='walk',skill='walking',seconds=11.,
        segments=[dict(at=0.),dict(at=1.,held=['fwd']),dict(at=7.,held=[])],scoring=dict(start=0.,end=11.))
    for direction in ['left','right']:
        cases['walking_turn_'+direction]=dict(mode='walk',skill='walking',seconds=10.,
            segments=[dict(at=0.),dict(at=1.,held=['fwd']),dict(at=3.,held=['fwd',direction]),
                      dict(at=5.,held=['fwd']),dict(at=7.,held=[])],scoring=dict(start=0.,end=10.))
    cases['walking_back']=dict(mode='walk',skill='walking',seconds=8.,
        segments=[dict(at=0.),dict(at=1.,held=['back']),dict(at=4.,held=[])],scoring=dict(start=0.,end=8.))
    cases['walking_strafe']=dict(mode='walk',skill='walking',seconds=9.,
        segments=[dict(at=0.),dict(at=1.,held=['strafe_l']),dict(at=3.,held=[]),
                  dict(at=4.,held=['strafe_r']),dict(at=6.,held=[])],scoring=dict(start=0.,end=9.))
    cases['walking_repeated_start']=dict(mode='walk',skill='walking',seconds=9.,
        segments=[dict(at=float(i),held=['fwd'] if i in [1,3,5] else []) for i in range(7)],
        scoring=dict(start=0.,end=9.))
    cases['sitstand']=dict(mode='walk',skill='sitstand',seconds=13.,
        segments=[dict(at=0.),dict(at=1.,taps=['sit']),dict(at=7.,taps=['sit'])],scoring=dict(start=1.,end=13.))
    for skill,tap,period in [('ground_pick','pick',4.),('kick_left','kick_left',5.),
                            ('kick_right','kick_right',5.),('roulade','roulade',5.)]:
        cases[skill]=dict(mode='walk',skill=skill,seconds=period+3.,
            segments=[dict(at=0.),dict(at=1.,taps=[tap])],scoring=dict(start=1.,end=period+1.))
    cases['roller_crouch']=dict(mode='roller',skill='roller_crouch',seconds=12.,
        segments=[dict(at=0.),dict(at=1.,held=['fwd']),dict(at=4.,held=[]),dict(at=4.5,taps=['sit'])],
        scoring=dict(start=4.5,end=9.5))
    cases['roller_brake_3s']=brake_case()
    cases['roller_release_3s']=brake_case(after=[])
    cases['roller_space_3s']=brake_case(after=['idle'])
    for push in [1.,2.,5.]:cases[f'roller_brake_{push:g}s']=brake_case(push)
    turn=brake_case(4.)
    turn['segments'].insert(2,dict(at=3.,held=['fwd','left']))
    cases['roller_turn_brake']=turn
    repeat=brake_case();repeat.update(seconds=22.,brake_times=[4.,10.,16.])
    repeat['scoring']['end']=22.
    repeat['segments'].extend([dict(at=7.,held=['fwd']),dict(at=10.,held=['back']),
                               dict(at=13.,held=['fwd']),dict(at=16.,held=['back'])])
    cases['roller_repeated_brake']=repeat
    crouch=copy.deepcopy(cases['roller_crouch']);crouch.update(skill='roller',seconds=16.,brake_times=[10.])
    crouch['segments'].extend([dict(at=9.5,held=['fwd']),dict(at=10.,held=['back'])])
    crouch['scoring']=dict(start=0.,end=16.)
    cases['roller_crouch_brake']=crouch
    for case in cases.values():case['protocol']=PROTOCOL
    return cases


def randomized_poses(mode,seed):
    name='microduck_roller' if mode=='roller' else 'microduck_ball_stand_fix'
    cfg=load_robot_json(sim2sim_root()/'robots'/f'{name}.json')
    sampler=HomePoseSampler(cfg);rng=np.random.default_rng(seed)
    try:
        yaw=float(rng.uniform(-math.pi,math.pi))
        poses,_,_=sampler.sample(rng,yaw_range=(yaw,yaw),joint_noise_rad=.015)
        for pose in poses:
            if pose['name']=='ball':pose['pos']=[5.,5.,.035]
        return poses
    finally:sampler.close()


def write_cases(directory,seeds=(),selected=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    written=[]
    for name,template in standard_cases().items():
        if selected and name not in selected:continue
        for seed in list(seeds) or [None]:
            case=copy.deepcopy(template);case['case']=name;case['seed']=seed
            case['randomized_start']=seed is not None
            if seed is not None:
                case['initial_poses']={case['mode']:randomized_poses(case['mode'],seed)}
            path=directory/f'{name}_{"nominal" if seed is None else seed}.json'
            path.write_text(json.dumps(case,indent=2));written.append(str(path))
    return written


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--seed-start',type=int,default=915000);p.add_argument('--seeds',type=int,default=0)
    p.add_argument('--cases',nargs='*')
    a=p.parse_args();print(json.dumps(write_cases(a.directory,range(a.seed_start,a.seed_start+a.seeds),a.cases)))
