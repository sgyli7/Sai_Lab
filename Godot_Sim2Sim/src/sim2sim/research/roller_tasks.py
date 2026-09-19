"""Source roller semantics: positive push, zero coast, negative brake.

The third actor command is target-minus-current heading error in radians,
as implemented by upstream RelativeHeadingVelocityCommand._update_command.
"""
import math
import numpy as np

CONDITIONS=('push','coast','brake','heading_l','heading_r','push_coast_brake')
PROTOCOL='roller_throttle_heading_v2'


def initial_speed(condition):return .3 if condition in ('coast','brake') else 0.

def target_offset(condition):return .8 if condition=='heading_l' else (-.8 if condition=='heading_r' else 0.)

def command(w):
    c=w.condition;t=w.t
    throttle={'push':.5,'coast':0.,'brake':-.5,'heading_l':.3,'heading_r':.3}.get(c)
    if c=='push_coast_brake':throttle=.5 if t<3 else (0. if t<6 else -.5)
    if throttle is None:raise ValueError('Unknown native roller condition: '+c)
    delta=w.roller_target_yaw-w.features['yaw']
    error=math.atan2(math.sin(delta),math.cos(delta))
    out=np.zeros(13,np.float32);out[0]=throttle;out[2]=np.clip(error,-1.,1.);return out


def summarize(rows,initial_heading,target_yaw,initial_velocity):
    a={k:np.asarray([r[k] for r in rows]) for k in rows[0]};t=a['time'];v=a['vel']
    failed=(a['z']<.045)|(a['tilt']>70);last=t>t[-1]-1.
    heading_error=np.arctan2(np.sin(target_yaw-a['yaw']),np.cos(target_yaw-a['yaw']))
    final_speed=float(np.mean(np.linalg.norm(v[last,:2],axis=1)))
    final_heading=float(np.sqrt(np.mean(heading_error[last]**2)))
    displacement=a['xy'][-1]-a['xy'][0]
    c=rows[0]['condition'];no_fall=not failed.any();forward=float(displacement@initial_heading)
    mean_push=float(v[(a['cmd'][:,0]>.01)&(t>1),0].mean()) if np.any((a['cmd'][:,0]>.01)&(t>1)) else 0.
    action_rate=float(np.mean(np.sum(np.diff(a['actions'],axis=0)**2,axis=1)))
    stable=bool(np.mean((a['tilt'][last]<15)&(a['z'][last]>.08)&(a['contact'][last].sum(1)>0))>=.9)
    minimum_smoothed_vx=float(np.convolve(v[:,0],np.ones(50)/50,mode='valid').min())
    if c=='push':
        outcome=mean_push>.2 and forward>.5
        quality=min(1.,max(0.,mean_push)/.3)*math.exp(-final_heading/.3)
    elif c=='coast':
        # Coast permits momentum to carry the duck; it is not a position hold.
        maximum_speed=float(np.max(np.convolve(np.linalg.norm(v[:,:2],axis=1),np.ones(50)/50,mode='valid')))
        outcome=maximum_speed<initial_velocity+.15 and minimum_smoothed_vx>-.05
        quality=math.exp(-action_rate/.2-final_heading/.3)
    elif c in ('brake','push_coast_brake'):
        outcome=final_speed<.05 and minimum_smoothed_vx>-.1
        if c=='push_coast_brake':outcome=outcome and mean_push>.2
        quality=math.exp(-final_speed/.1-final_heading/.3-max(0.,-minimum_smoothed_vx-.03)/.1)
    else:
        outcome=final_heading<.15 and mean_push>.1
        quality=math.exp(-final_heading/.3)
    return dict(protocol=PROTOCOL,condition=c,success=bool(no_fall and stable and outcome),
        score=float(quality*no_fall*stable),fell=bool(failed.any()),final_standing=stable,
        final_speed=final_speed,heading_rmse_final_rad=final_heading,forward_displacement=forward,
        mean_push_vx=mean_push,min_smoothed_vx=minimum_smoothed_vx,action_rate=action_rate,
        final_z=float(a['z'][-1]),final_tilt=float(a['tilt'][-1]),max_tilt=float(a['tilt'].max()),
        initial_velocity=float(initial_velocity),target_heading=float(target_yaw))
