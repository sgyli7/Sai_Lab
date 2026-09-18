"""Accept actual Godot workshop traces; no mocks or policy substitutes."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); d=json.loads((a.directory/'hub.json').read_text());events=d['events']
    expected=['microduck','roller','sai','sai','roller','sai','microduck']
    checks={
        'expected_switches_and_reset':[e['to'] for e in events]==expected,
        'one_process':len({e['pid'] for e in events})==1,
        'persistent_hub_and_workshop':len({(e['hub'],e['atelier']) for e in events})==1,
        'loose_body_ids_and_positions_preserved':all(e['props_before']==e['props_after'] for e in events),
        'separate_jolt_spaces':all(e['space']!=e['new_space'] for e in events),
        'qualified_sai_settings':all(e['physics_hz']==1000 and e['settings']=={'speculative':.0005,'velocity_steps':10.} for e in events if e['to']=='sai'),
        'original_md_settings':all(e['physics_hz']==200 and e['settings']=={'speculative':0.,'velocity_steps':32.} for e in events if e['to']!='sai'),
        'all_body_counts':all(e['bodies']=={'microduck':16,'roller':19,'sai':26}[e['to']] for e in events),
    }
    native=json.loads((a.directory/'native-0.json').read_text())['rows']
    moving=[r for r in native if r['skill']=='walking' and 1.1<r['t']<3]
    roll=[r for r in native if r['skill']=='roulade']
    last=native[-1]
    checks['microduck_walks_without_fall']=moving[-1]['body']['base_pos'][0]-moving[0]['body']['base_pos'][0]>.20 and all(not r['fell'] for r in moving)
    checks['microduck_roll_and_recover']=bool(roll) and max(r['tilt'] for r in roll)>90 and last['tilt']<15 and not last['fell']
    roller=json.loads((a.directory/'native-1.json').read_text())
    rr=roller['rows']
    checks['roller_motion_and_crouch']=roller['first_fall'] is None and max(r['body']['base_pos'][0] for r in rr)>.8 and any(r['skill']=='roller_crouch' for r in rr)
    ss=[json.loads(x) for x in (a.directory/'sai-trace.jsonl').read_text().splitlines()]
    # First Sai session runs hub t21..40; release state t0..19.
    first=[]
    for r in ss:
        if first and r['state']['time']<first[-1]['state']['time']:break
        first.append(r)
    def segment(start,end):return [r['state'] for r in first if start<r['state']['time']<end]
    fwd=segment(1.1,2.9);rev=segment(3.6,5.4)
    checks['sai_forward']=fwd[-1]['base_position'][0]-fwd[0]['base_position'][0]>.15
    checks['sai_reverse']=rev[-1]['base_position'][0]-rev[0]['base_position'][0]<-.10
    left=segment(6.1,7.9);right=segment(9.1,10.9)
    def yaw(r):
        c=r['base_rotation_columns'][0];return float(np.arctan2(c[1],c[0]))
    checks['sai_left_turn']=yaw(left[-1])-yaw(left[0])>.3
    checks['sai_right_turn']=yaw(right[-1])-yaw(right[0])<-.3
    standing=float(np.mean([r['base_position'][2] for r in segment(10.5,11.8)]))
    low=float(np.mean([r['base_position'][2] for r in segment(13,14.7)]))
    recovered=float(np.mean([r['base_position'][2] for r in segment(16.5,18)]))
    checks['shift_hold_and_release']=standing-low>.025 and recovered-low>.025
    checks['sai_reset_returns_to_start']=np.linalg.norm(ss[-1]['state']['base_position'][:2])<.04
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(passed=all(checks.values()),checks=checks,pid=d['pid'],switches=len(events)-1,
                height_m=dict(standing=standing,crouched=low,recovered=recovered),
                note='In-process InputEventKey replay. OS-level physical keyboard delivery is a separate check.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
