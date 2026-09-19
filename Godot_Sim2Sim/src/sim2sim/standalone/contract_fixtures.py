"""Independent Python oracle for the native controller state and observation contract."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from sim2sim.backends import SimState
from sim2sim.backends.godot_backend import inertial_to_body
from sim2sim.obs import build_obs
from sim2sim.play import ROLLER_LIMITS, kick_ball_position
from sim2sim.play_input import PlayBrain, TwistLimits
from sim2sim.paths import sim2sim_root


def generate(destination):
    deployment = json.loads((sim2sim_root()/'godot/runtime_assets/deployment.json').read_text())
    brain_cases = []
    for mode, options in [('walk', dict(has_standing=False,has_stand_hold=True)), ('partner', {}),
                          ('roller', dict(has_standing=False, has_sitstand=False, has_pick=False,
                           has_kick_left=False, has_kick_right=False, has_roulade=False,
                           has_roller_crouch=True, lim=ROLLER_LIMITS))]:
        brain = PlayBrain(**options)
        flags = {key: getattr(brain, 'has_'+('pick' if key=='ground_pick' else key))
                 for key in ['walking','standing','sitstand','ground_pick','kick_left',
                             'kick_right','roulade','roller_crouch','stand_hold']}
        cases = []
        triggers = {150:['sit'], 200:['sit'], 400:['pick'], 410:['kick_left'],
                    650:['kick_left'], 920:['kick_right'], 1200:['roulade'],
                    1460:['reset'], 1490:['sit'], 1530:['reset'], 1540:['push'],
                    1560:['switch_robot'], 1570:['stand'], 1590:['quit']}
        for tick in range(1600):
            order = [['fwd'],['fwd','back'],['left','right'],['fwd','left'],
                     ['fwd','idle'],['idle','fwd'],[],['strafe_r']][(tick//20)%8]
            taps = triggers.get(tick, [])
            out = brain.tick(set(order), taps, .02, press_order=order)
            expected = asdict(out)
            expected['command'] = out.command.tolist()
            expected['started_skill'] = out.started_skill or ''
            cases.append(dict(held=order, order=order, taps=taps, dt=.02,
                              expected=expected, vel=brain.vel.tolist(),
                              behavior_t=brain.behavior_t,pick_phase=brain.pick_phase,rise_t=brain.rise_t))
        brain_cases.append(dict(mode=mode, flags=flags,limits=asdict(brain.lim),cases=cases))
    rng = np.random.default_rng(915000)
    observations = []
    for mode, robot in deployment['robots'].items():
        for index in range(512):
            quat = rng.normal(size=4);quat/=np.linalg.norm(quat)
            if index < 4: quat=np.eye(4)[index]
            raw = dict(base_pos=rng.normal(size=3).tolist(),base_quat=quat.tolist(),
                       base_angvel_local=rng.normal(size=3).tolist(),
                       q=rng.normal(size=14).tolist(),qd=rng.normal(size=14).tolist())
            pos, body_quat = inertial_to_body(raw['base_pos'],raw['base_quat'],robot['ipos'],robot['iquat'])
            state = SimState(t=0,q=np.array(raw['q']),qd=np.array(raw['qd']),base_pos=pos,
                             base_quat_wxyz=body_quat,base_linvel=np.zeros(3),
                             base_angvel_local=np.array(raw['base_angvel_local']))
            last=rng.normal(size=14).astype(np.float32)
            command=rng.normal(size=13).astype(np.float32)
            home=np.array(robot['home'],np.float32)
            scale=robot['action_scale']
            observations.append(dict(mode=mode,raw=raw,last=last.tolist(),command=command.tolist(),
              body=dict(base_pos=pos.tolist(),base_quat=body_quat.tolist()),
              obs=build_obs(state,last,command,home).tolist(),
              control=(home+last*scale).tolist(),ball_left=kick_ball_position(state,'kick_left'),
              ball_right=kick_ball_position(state,'kick_right')))
    Path(destination).write_text(json.dumps(dict(brains=brain_cases,observations=observations)))
    print(json.dumps(dict(brain_ticks=sum(len(x['cases']) for x in brain_cases),observations=len(observations))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination')
    generate(p.parse_args().destination)
