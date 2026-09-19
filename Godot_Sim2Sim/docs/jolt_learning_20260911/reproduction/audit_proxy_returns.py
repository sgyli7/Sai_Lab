"""Apply the frozen training objective to paired native traces, without re-simulation."""
import json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sim2sim.research.rewards import Objective
from sim2sim.research.tasks import TASKS
from sim2sim.standalone.replay import raw_state
from sim2sim.coords import quat_wxyz_to_mat
from sim2sim.policy_task_state import contacts_from_raw
p=Path('results/jolt_learning_20260911');robot=json.loads((p/'contract_export_project/runtime_assets/deployment.json').read_text())['robots']['roller'];home=np.array(robot['home'],np.float32)
cases=['roller_brake_1s','roller_brake_2s','roller_brake_3s','roller_brake_5s','roller_turn_brake']
def state_features(row):
 state=raw_state(row['raw'],robot);rot=quat_wxyz_to_mat(state.base_quat_wxyz);yaw=math.atan2(rot[1,0],rot[0,0]);c,s=math.cos(yaw),math.sin(yaw)
 f=dict(rot=rot,yaw=yaw,z=float(state.base_pos[2]),xy=state.base_pos[:2].copy(),up=float(rot[2,2]),tilt=float(row['tilt']),vel=np.array([[c,s,0],[-s,c,0],[0,0,1]])@state.base_linvel,gyro=state.base_angvel_local,contact=contacts_from_raw(row['raw'],robot['support_groups']))
 return state,f
reports=[]
for name,directory in [('frozen','baseline_candidate'),('r1_full_s71_1m','candidate_evaluations/r1_full_s71_1m/suite'),('r2_sagittal_s71_1m','candidate_evaluations/r2_sagittal_s71_1m/suite')]:
 summary=json.loads((p/directory/'summary.json').read_text())
 for e in summary['episodes']:
  if e['case'] not in cases:continue
  rows=json.loads(Path(e['trace']).read_text())['rows'];state,features=state_features(rows[0])
  w=SimpleNamespace(task=TASKS['roller'],roller_contract=True,state=state,features=features,home=home,t=0.,last=np.zeros(14),old_last=np.zeros(14),roller_target_yaw=features['yaw'])
  objective=Objective(w,{'brake':2.,'height':4.,'upright':2.},{'brake_velocity_variance':.0025},'stop_hold_v1');total=0.;terms={};terminated=False;discounted=0.;brake_discounted=0.;brake_index=0;transitions=0
  for before,after in zip(rows,rows[1:]):
   if before['skill']!='roller' or after['skill']!='roller':raise RuntimeError('Proxy audit requires a single actor tape')
   w.t=float(after['t']);w.state,w.features=state_features(after)
   w.last=np.array(before['action']);w.old_last=np.array(before['last_action']);w.executed_command=np.array(before['command'])
   _,old=state_features(before);w.executed_heading_target=old['yaw']+w.executed_command[2]
   reward,terminal,detail=objective.compute();total+=reward
   discounted+=(.99**transitions)*reward;transitions+=1
   if w.executed_command[0]<-.01:
    brake_discounted+=(.99**brake_index)*reward;brake_index+=1
   for key,value in detail.items():terms[key]=terms.get(key,0.)+value*.02
   if terminal:terminated=True;break
  report=dict(model=name,case=e['case'],seed=e['seed'],success=e['brake_success'],proxy_return=total,discounted_return_gamma_099=discounted,brake_discounted_return_gamma_099=brake_discounted,return_per_planned_step=total/(len(rows)-1),terminated=terminated,terms=terms)
  reports.append(report)
result=dict(method='Native pre-action trace pairs are scored as executed transition and successor state. First training termination is absorbing with zero subsequent reward; final successor is not in the row array, so the last transition is omitted identically on both sides. The whole native tape is used (8–12s), not claimed identical to the trainer 10s timeout.',objective='stop_hold_v1',rows=reports)
(p/'proxy_return_audit.json').write_text(json.dumps(result,indent=2))
for name in ['frozen','r1_full_s71_1m','r2_sagittal_s71_1m']:
 a=[r for r in reports if r['model']==name];print(name,'passes',sum(r['success'] for r in a),'mean return',np.mean([r['proxy_return'] for r in a]),'mean reward',np.mean([r['return_per_planned_step'] for r in a]),'discounted',np.mean([r['discounted_return_gamma_099'] for r in a]),'brake-discounted',np.mean([r['brake_discounted_return_gamma_099'] for r in a]))
