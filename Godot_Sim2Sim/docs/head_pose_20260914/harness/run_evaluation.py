from pathlib import Path
import hashlib,json,time
from sim2sim.standalone.sprint import write_cases
from sim2sim.standalone.suite import run,runtime_inputs
R=Path(__file__).resolve().parent
assert json.loads((R/'preflight_completed.json').read_text())['passed']
control=json.loads((R/'base/runtime_assets/deployment.json').read_text())['control_config']
freeze={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'protocol.json',Path(__file__),Path('/home/ethan/Projects/MicroDuck-SpeedControls/scripts/native_neck_bias_experiment.py')]}
(R/'evaluation_code.json').write_text(json.dumps(freeze,indent=2)+'\n')
results=[]
for speed in [.30,.45]:
 for angle in [0,-5,-10]:
  label=f'v{round(speed*100):02d}_neck{abs(angle):02d}';project=R/('neck_'+str(abs(angle)));record=json.loads(project.with_suffix('.json').read_text());assert runtime_inputs(project)==record['candidate_inputs']
  cases=write_cases(R/label/'cases',range(929100,929104),speed,control=control)
  start=time.monotonic();s=run(cases,R/label/'suite',workers=2,project=project)
  assert s['errors']==0,'Preserve suite errors and stop the experiment'
  result=dict(label=label,speed=speed,neck_degrees=angle,passes=sum(e['task_metrics']['success'] for e in s['episodes']),count=len(s['episodes']),falls=sum(e['task_metrics']['fell'] for e in s['episodes']),long_speed=[e['task_metrics']['sustained_mean_vx'] for e in s['episodes'] if e['case']=='sprint_long'],runtime_id=s['runtime_id'],elapsed_s=time.monotonic()-start,completed=True)
  (R/label/'completed.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result)
  (R/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print('NECK_RESULT',result,flush=True)
(R/'evaluation_completed.json').write_text(json.dumps(results,indent=2)+'\n')
