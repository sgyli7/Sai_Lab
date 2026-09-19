"""Independent sprint keyboard cases and physical acceptance metrics.

No training reward is imported. Ordinary paired controls use the same cases
with the modifier removed; acceleration gains are measured from body motion.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from .cases import randomized_poses

PROTOCOL = 'walking_sprint_v1'


def templates():
    f=['sprint','fwd']
    def case(seconds,segments):
        return dict(mode='walk',skill='walking',seconds=seconds,
                    segments=[dict(at=at,held=held) for at,held in segments],
                    scoring=dict(start=0.,end=seconds),protocol=PROTOCOL)
    return {
        'shift_first':case(12,[(0,[]),(.5,['sprint']),(1,f),(9,[])]),
        'w_first':case(16,[(0,[]),(1,['fwd']),(3,f),(10,['fwd']),(12,[])]),
        'long':case(25,[(0,[]),(1,f),(21,[])]),
        'left':case(14,[(0,[]),(1,f),(3,f+['left']),(8,f),(10,[])]),
        'right':case(14,[(0,[]),(1,f),(3,f+['right']),(8,f),(10,[])]),
        'alternate':case(18,[(0,[]),(1,f),(3,f+['left']),(6,f+['right']),(9,f),(14,[])]),
        'turn_release':case(16,[(0,[]),(1,f),(3,f+['left']),(7,['fwd','left']),(9,['fwd']),(12,[])]),
        'repeat':case(17,[(0,[]),(1,f),(4,['fwd']),(6,f),(9,[]),(12,f),(14,[])]),
    }


def write_cases(directory,seeds,speed=.4,paired=False,selected=None,control=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);written=[]
    for name,template in templates().items():
        if selected and name not in selected:continue
        for seed in seeds:
            poses=randomized_poses('walk',seed)
            for ordinary in ([False,True] if paired else [False]):
                case=copy.deepcopy(template)
                case.update(case='sprint_'+name+('_ordinary' if ordinary else ''),seed=seed,
                    pair_id='sprint_'+name,randomized_start=True,
                    initial_poses={'walk':poses},ordinary_control=ordinary)
                case['control_config']=copy.deepcopy(control or {})
                settings=case['control_config'].setdefault('walk',{}).setdefault('twist_limits',{})
                settings.update(sprint_vmax_x=speed,sprint_vmax_ang=.8)
                # Preserve the intended sprint intervals for paired measurement.
                case['sprint_intervals']=[]
                for i,segment in enumerate(case['segments']):
                    end=case['segments'][i+1]['at'] if i+1<len(case['segments']) else case['seconds']
                    if 'sprint' in segment['held'] and 'fwd' in segment['held']:
                        case['sprint_intervals'].append([segment['at'],end])
                    if ordinary:segment['held']=[bit for bit in segment['held'] if bit!='sprint']
                path=directory/f"{case['case']}_{seed}.json"
                path.write_text(json.dumps(case,indent=2)+'\n');written.append(path)
    return written


def metrics(rows,actions,case):
    if len(rows)!=round(case['seconds']/.02):raise ValueError('Incomplete sprint trajectory')
    times=np.array([a['t'] for a in actions]);vel=np.array([r['vel'] for r in rows])
    xy=np.array([r['xy'] for r in rows]);yaw=np.unwrap([r['yaw'] for r in rows])
    tilt=np.array([r['tilt'] for r in rows]);z=np.array([r['z'] for r in rows])
    fell=(tilt>60.)|(z<.055)
    active=np.zeros(len(rows),bool)
    for start,end in case['sprint_intervals']:active|=(times>=start+1.-1e-9)&(times<end-1e-9)
    straight=[];turns=[];stops=[]
    for i,segment in enumerate(case['segments']):
        start=segment['at'];end=case['segments'][i+1]['at'] if i+1<len(case['segments']) else case['seconds']
        held=set(segment['held']);index=np.flatnonzero((times>=start-1e-9)&(times<end-1e-9))
        if not len(index):continue
        if 'fwd' in held and end-start>=1.5:
            direction=1 if 'left' in held else (-1 if 'right' in held else 0)
            if direction:
                use=index[times[index]>=start+1.-1e-9]
                requested=np.mean([abs(actions[j]['requested_command'][2]) for j in use])
                # World yaw derivative avoids body-axis projection while tilted.
                signed=np.diff(yaw[index])/.02*direction
                averaged=np.convolve(signed,np.ones(10)/10,mode='valid')
                reverse=any(np.all(averaged[j:j+25]<-.05) for j in range(max(0,len(averaged)-24)))
                actual=(yaw[use[-1]]-yaw[use[0]])/max(.02,times[use[-1]]-times[use[0]])*direction
                turns.append(dict(at=start,requested=float(requested),signed_yaw_rate=float(actual),
                                  sustained_reverse=bool(reverse),passed=bool(actual>=.6*requested and not reverse)))
            else:
                anchor=index[0];heading=np.array([math.cos(yaw[anchor]),math.sin(yaw[anchor])])
                displacement=xy[index]-xy[anchor]
                forward=displacement@heading;lateral=displacement@np.array([-heading[1],heading[0]])
                error=float(np.rad2deg(np.max(np.abs(yaw[index]-yaw[anchor]))))
                side=float(np.max(np.abs(lateral)));limit=max(.05,float(np.max(forward))*.1)
                straight.append(dict(at=start,heading_error_deg=error,lateral_path_m=side,lateral_limit_m=limit,
                                     passed=bool(error<=15. and side<=limit)))
        elif not held.intersection({'fwd','back','left','right','strafe_l','strafe_r'}) and start>0 and end-start>=3.:
            speed=np.linalg.norm(vel[index,:2],axis=1)
            average=np.convolve(speed,np.ones(10)/10,mode='valid')
            stable=(tilt[index][9:]<15.)&(z[index][9:]>.08)
            onset=None
            for j in range(max(0,len(average)-49)):
                if np.all(average[j:j+50]<.05) and np.mean(stable[j:j+50])>=.9:
                    onset=float(times[index[j+9]]-start);break
            stops.append(dict(at=start,onset_s=onset,passed=onset is not None and onset<=2.+1e-9))
    expected=not case.get('ordinary_control',False)
    selection=all((a['skill']=='sprint')==expected for a,m in zip(actions,active) if m)
    passed=bool(not fell.any() and selection and all(x['passed'] for x in straight+turns+stops))
    return dict(success=passed,fell=bool(fell.any()),first_fall_s=next((float(t) for t,f in zip(times,fell) if f),None),
        score=float(np.mean(vel[active,0])) if passed else 0.,
        sustained_mean_vx=float(np.mean(vel[active,0])),sustained_p10_vx=float(np.quantile(vel[active,0],.1)),
        max_tilt=float(tilt.max()),min_z=float(z.min()),straight=straight,turns=turns,stops=stops,
        actor_selection_passed=selection,speed_gain_gate='requires paired ordinary result')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    p.add_argument('--seed-start',type=int,default=919000);p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--speed',type=float,default=.4);p.add_argument('--paired',action='store_true')
    p.add_argument('--control',type=Path);p.add_argument('--cases',nargs='*')
    a=p.parse_args();control=None if a.control is None else json.loads(a.control.read_text())
    print(json.dumps([str(x) for x in write_cases(a.directory,range(a.seed_start,a.seed_start+a.seeds),a.speed,a.paired,a.cases,control)]))
