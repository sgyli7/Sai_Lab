from pathlib import Path
import json,subprocess
import numpy as np
import onnxruntime as ort
from sim2sim.standalone.sprint import write_cases
from sim2sim.standalone.suite import run
R=Path(__file__).resolve().parent
control=json.loads((R/'base/runtime_assets/deployment.json').read_text())['control_config']
cases=write_cases(R/'preflight_cases',[929100],.3,control=control,selected=['long','turn_release'])
results={}
for label in ['base','neck_0','neck_5','neck_10']:
 p=R/label;check=subprocess.run(['godot','--headless','--fixed-fps','200','--path',str(p),'res://standalone/main.tscn','--','--self-test'],capture_output=True,text=True,timeout=40)
 (R/(label+'_self_test.log')).write_text(check.stdout+check.stderr)
 assert check.returncode==0 and '"passed":true' in check.stdout.replace(' ',''),check.stdout+check.stderr
 s=run(cases,R/('preflight_'+label),workers=2,project=p);assert s['errors']==0
 results[label]=s
 print(label,[(e['case'],e['task_metrics']['success']) for e in s['episodes']],flush=True)
base={e['case']:json.loads(Path(e['trace']).read_text())['rows'] for e in results['base']['episodes']}
zero={e['case']:json.loads(Path(e['trace']).read_text())['rows'] for e in results['neck_0']['episodes']}
equality={}
for case in base:
 errors={key:float(np.max(np.abs(np.array([r[key] for r in base[case]])-np.array([r[key] for r in zero[case]])))) for key in ['action','last_action','obs','ctrl','requested_command']}
 assert max(errors.values())<1e-7,errors
 equality[case]=errors
contracts={}
for label,angle in [('neck_0',0),('neck_5',-5),('neck_10',-10)]:
 dep=json.loads((R/label/'runtime_assets/deployment.json').read_text());scale=dep['robots']['walk']['action_scale'];home=np.array(dep['robots']['walk']['home'],np.float32)
 errors=dict(bias=0.,action=0.,previous=0.,observation_previous=0.,control=0.);frames=0
 for ep in results[label]['episodes']:
  rows=json.loads(Path(ep['trace']).read_text())['rows'];bias=0.;previous=np.zeros(14,np.float32)
  for row in rows:
   target=np.deg2rad(angle) if row['skill']=='sprint' and row['requested_command'][0]>0 else 0
   delta=abs(np.deg2rad(angle))*.02/.5;bias+=np.clip(target-bias,-delta,delta)
   errors['bias']=max(errors['bias'],abs(bias-row['experiment_neck_bias_rad']))
   policy=np.array(row['experiment_policy_action'],np.float32);actual=np.array(row['action'],np.float32);expected=policy.copy();expected[5]+=bias/scale
   errors['action']=max(errors['action'],float(np.abs(actual-expected).max()))
   errors['previous']=max(errors['previous'],float(np.abs(np.array(row['last_action'])-previous).max()))
   errors['observation_previous']=max(errors['observation_previous'],float(np.abs(np.array(row['obs'][34:48])-previous).max()))
   expected_ctrl=home+np.float32(scale)*actual
   errors['control']=max(errors['control'],float(np.abs(np.array(row['ctrl'])-expected_ctrl).max()))
   assert len(row['raw']['body_states'])>=15
   previous=actual;frames+=1
 assert max(errors.values())<1e-6,errors
 contracts[label]=dict(errors=errors,frames=frames)
(R/'preflight_completed.json').write_text(json.dumps(dict(passed=True,zero_intervention_equivalence=equality,control_contracts=contracts),indent=2)+'\n')
print('PRECHECK PASSED',flush=True)
