"""Observe heading reference and body travel at an existing handoff failure."""
from pathlib import Path
import json,math
from types import SimpleNamespace
import numpy as np
from sim2sim.motion_control import MotionControl
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912');out={}
for arm in ['baseline_s05','baseline_cat_split','frozen_cat_joint']:
 s=json.loads((r/arm/'suite/summary.json').read_text());rows=[]
 for name,seed in [('sprint_repeat',925002),('sprint_repeat',927013)]:
  e=next(x for x in s['episodes'] if (x['case'],x['seed'])==(name,seed));data=json.loads(Path(e['trace']).read_text())['rows'];control=MotionControl(e['control_config']['walk']);commands=[];ref=[];yaw=[];pos=[]
  for row in data:
   body=row['body'];q=body['base_quat'];yaw.append(math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3])));pos.append(body['base_pos'][:2])
   state=SimpleNamespace(base_pos=np.array(body['base_pos']),base_quat_wxyz=np.array(q),base_linvel=np.array(body['base_linvel']))
   command=control.command(row['requested_command'],state,row['skill']);commands.append(float(np.max(np.abs(command-np.array(row['command'],np.float32)))));ref.append([control.walk_path_yaw,control.target_yaw])
  assert max(commands)<1e-5
  p=np.array(pos);y=np.unwrap(yaw);ref=np.unwrap(np.array(ref),axis=0);phases=[]
  for start,end in [(1,4),(4,6),(6,9)]:
   i,j=round(start/.02),round(end/.02);d=p[j-1]-p[i];heading=y[i];cross=-math.sin(heading)*(p[i:j,0]-p[i,0])+math.cos(heading)*(p[i:j,1]-p[i,1]);first=y[50]
   phases.append(dict(start=start,end=end,skill=data[i]['skill'],heading_from_first_deg=float(np.degrees(heading-first)),path_reference_from_body_deg=float(np.degrees(ref[i,0]-heading)),heading_change_deg=float(np.degrees(y[j-1]-heading)),net_body_lateral_m=float(-math.sin(heading)*d[0]+math.cos(heading)*d[1]),max_body_lateral_m=float(abs(cross).max()),net_original_lateral_m=float(-math.sin(first)*d[0]+math.cos(first)*d[1]),mean_corrective_command=np.array([x['command'][:3] for x in data[i:j]]).mean(0).tolist()))
  rows.append(dict(case=name,seed=seed,command_max_error=max(commands),phases=phases))
 out[arm]=rows
atomic_json(r/'path_diagnosis.json',out)
for arm,rows in out.items():
 for row in rows:print(arm,row['seed'],row['phases'][1],flush=True)
