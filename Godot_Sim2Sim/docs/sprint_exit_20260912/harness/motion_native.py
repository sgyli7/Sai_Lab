from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
from sim2sim.standalone.prepare import prepare
from sim2sim.standalone.sprint import write_cases
from sim2sim.standalone.sprint_accept import paired_acceptance
from sim2sim.standalone.suite import run
from sim2sim.research.queue import atomic_json
p=argparse.ArgumentParser();p.add_argument('--actor',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(exist_ok=False,parents=True);models=a.out/'models'
shutil.copytree('results/research_20260911/delivery/candidate_models',models)
for name,skill in [('Walk_Godot.onnx','walking'),('Sprint_Godot.onnx','sprint')]:
 target=models/name;shutil.copy2(a.actor,target)
 manifest=json.loads(a.actor.with_suffix('.manifest.json').read_text())
 manifest.update(sha256=hashlib.sha256(target.read_bytes()).hexdigest(),status='experimental_motion_state_not_promoted')
 manifest.setdefault('sim2sim',{}).update(skill=skill,use_stand_policy=False)
 atomic_json(target.with_suffix('.manifest.json'),manifest)
control=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text())
atomic_json(a.out/'control.json',control)
project=a.out/'prepared';subprocess.run(['cp','--reflink=auto','-a','godot',str(project)],check=True,timeout=60)
prepare(models,project=project,sprint=models/'Sprint_Godot.onnx',control_config=a.out/'control.json')
cases=write_cases(a.out/'cases',range(923000,923016),.3,paired=True,control=control)
mini=Path('results/sprint_exit_20260912/minimize/remove_forward_prefix.json')
s=run([*cases,mini],a.out/'suite',workers=4,project=project)
main={**s,'episodes':[e for e in s['episodes'] if e['case']!='remove_forward_prefix']}
accept=paired_acceptance(main,cases);atomic_json(a.out/'paired_acceptance.json',accept)
c=[e for e in main['episodes'] if not json.loads(Path(e['case_path']).read_text())['ordinary_control']]
minimal=next(e for e in s['episodes'] if e['case']=='remove_forward_prefix')
result=dict(errors=s['errors'],candidate_count=len(c),candidate_pass=sum(e['task_metrics']['success'] for e in c),candidate_falls=sum(e['task_metrics']['fell'] for e in c),accepted=accept['accepted'],mini=minimal['task_metrics'],models=s['models'],comparisons=[{k:v for k,v in x.items() if k!='pairs'} for x in accept['comparisons']])
atomic_json(a.out/'completed.json',result);print(result,flush=True)
