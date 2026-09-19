from pathlib import Path
import copy,json,shutil,subprocess
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.suite import run
r=Path('results/sprint_exit_20260912');out=r/'frame_probe';out.mkdir(exist_ok=False)
project=out/'prepared';subprocess.run(['cp','--reflink=auto','-a','results/sprint_joint_20260912/delivery/runtime',str(project)],check=True,timeout=60)
shutil.copy2('godot/standalone/motion_control.gd',project/'standalone/motion_control.gd')
cases=[]
for source in [r/'minimize/remove_forward_prefix.json',*sorted((r/'baseline_exits/cases').glob('*.json'))]:
 case=json.loads(source.read_text());case['control_config']['walk']['walk_path_project_velocity']=True
 p=out/(case['case']+'_'+str(case['seed'])+'.json');atomic_json(p,case);cases.append(p)
s=run(cases,out/'suite',workers=4,project=project)
result=dict(count=len(cases),errors=s['errors'],passed=sum(x['task_metrics']['success'] for x in s['episodes']),
 mini=next(x for x in s['episodes'] if x['case']=='remove_forward_prefix'),
 failures=[dict(case=x['case'],seed=x['seed'],metrics=x['task_metrics']) for x in s['episodes'] if not x['task_metrics']['success']])
atomic_json(out/'completed.json',result);print({k:v for k,v in result.items() if k not in ['mini','failures']},flush=True)
