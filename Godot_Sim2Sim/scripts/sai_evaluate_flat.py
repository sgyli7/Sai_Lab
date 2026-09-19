"""Check recorded real Godot key events and motor-driven motion, independently of reward."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--directory',type=Path,default=Path('artifacts'))
p.add_argument('--prefix',default='godot-flat-')
p.add_argument('--out',type=Path,required=True)
p.add_argument('--require-pass',action='store_true')
p.add_argument('--cases',nargs='+',choices=['stop','W','S','A','D','WA','shift','W_shift'])
a=p.parse_args()
rows=[]
for name,vx,wz,shift in [('stop',0,0,False),('W',.16,0,False),('S',-.16,0,False),('A',0,.45,False),('D',0,-.45,False),('WA',.14,.3,False),('shift',0,0,True),('W_shift',.14,0,True)]:
    if a.cases and name not in a.cases:continue
    raw=json.loads((a.directory/f'{a.prefix}{name}.json').read_text())
    samples=raw['samples'];t=np.array([s['time'] for s in samples])
    xyz=np.array([s['base_position'] for s in samples])
    obs=np.array([s['policy_observation'] for s in samples])
    rotations=np.array([s['base_rotation_columns'] for s in samples]).transpose(0,2,1)
    yaw=np.unwrap(np.arctan2(rotations[:,1,0],rotations[:,0,0]))
    upright=np.array([s['upright'] for s in samples])
    steady=t>=2;normal=(t>=1)&(t<3);low=(t>=5)&(t<8);recovered=t>=9
    displacement=xyz[-1,:2]-xyz[0,:2]
    row=dict(case=name,duration=float(t[-1]),displacement_xy_m=displacement.tolist(),
        yaw_change_rad=float(yaw[-1]-yaw[0]),velocity_mae=float(abs(obs[steady,3]-vx).mean()),
        yaw_mae=float(abs(obs[steady,8]-wz).mean()),lateral_mae=float(abs(obs[steady,4]).mean()),
        min_upright=float(upright.min()),crouch_drop_m=float(xyz[normal,2].mean()-xyz[low,2].mean()),
        recovered_height_difference_m=float(abs(xyz[normal,2].mean()-xyz[recovered,2].mean())))
    checks=dict(completed_12s=t[-1]>=11.999,upright=row['min_upright']>.9,
        velocity_tracking=row['velocity_mae']<.06,yaw_tracking=row['yaw_mae']<.22,
        lateral_tracking=row['lateral_mae']<.04,full_articulation=raw['body_count']==26 and raw['hinges']==23 and raw['sliders']==2,
        finite_observations=np.isfinite(obs).all())
    if name in ('stop','shift'):checks['stationary_drift']=np.linalg.norm(displacement)<.12
    if name in ('W','S'):
        checks['travel_direction']=np.sign(vx)*displacement[0]>.7
        checks['straight_heading']=abs(row['yaw_change_rad'])<.35
    if name in ('A','D'):
        checks['turn_direction']=np.sign(wz)*row['yaw_change_rad']>1.5
        checks['turn_center_drift']=np.linalg.norm(displacement)<.4
    if shift:
        checks['crouch_drop']=.023<row['crouch_drop_m']<.055
        checks['stand_recovery']=row['recovered_height_difference_m']<.012
        events=[e for e in raw['input_events'] if e['key']==4194325]
        checks['shift_press_release']=len(events)==2 and events[0]['pressed'] and not events[1]['pressed']
    for char,code in [('W',87),('S',83),('A',65),('D',68)]:
        if char in name and name not in ('shift',):
            checks[f'{char}_key_event']=any(e['key']==code and e['pressed'] for e in raw['input_events'])
    row['checks']={k:bool(v) for k,v in checks.items()};row['passed']=all(checks.values());rows.append(row)
    print(json.dumps(row),flush=True)
result=dict(suite='godot-real-keyboard-v1',physics='Godot/Jolt',engine=raw['engine'],cases=rows,passed=all(r['passed'] for r in rows))
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
if a.require_pass and not result['passed']:raise SystemExit('Godot command acceptance failed')
