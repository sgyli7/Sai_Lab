from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
from sim2sim.standalone.prepare import prepare
from sim2sim.standalone.sprint import write_cases
from sim2sim.standalone.sprint_accept import paired_acceptance
from sim2sim.standalone.suite import run
from sim2sim.research.queue import atomic_json
p=argparse.ArgumentParser();p.add_argument('--actor',type=Path,required=True);p.add_argument('--ordinary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--seeds',type=int,default=16);a=p.parse_args()
a.out.mkdir(exist_ok=False,parents=True);models=a.out/'models'
shutil.copytree('results/research_20260911/delivery/candidate_models',models)
for name,skill in [('Walk_Godot.onnx','walking'),('Sprint_Godot.onnx','sprint')]:
 source=a.actor if skill=='sprint' else a.ordinary
 target=models/name;shutil.copy2(source,target)
 manifest=json.loads(source.with_suffix('.manifest.json').read_text())
 manifest.update(sha256=hashlib.sha256(target.read_bytes()).hexdigest(),status='experimental_gpu_sprint_not_promoted')
 manifest.setdefault('sim2sim',{}).update(skill=skill,use_stand_policy=False)
 atomic_json(target.with_suffix('.manifest.json'),manifest)
control=json.loads(a.control.read_text())
atomic_json(a.out/'control.json',control)
project=a.out/'prepared';subprocess.run(['cp','--reflink=auto','-a','results/sprint_exit_20260912/state_contract/suite/runtime',str(project)],check=True,timeout=60)
shutil.copy2('godot/standalone/motion_control.gd',project/'standalone/motion_control.gd')
prepare(models,project=project,sprint=models/'Sprint_Godot.onnx',control_config=a.out/'control.json')
cases=write_cases(a.out/'cases',range(927000,927000+a.seeds),.3,paired=True,control=control)
mini=Path('results/sprint_exit_20260912/minimize/remove_forward_prefix.json')
previous=json.loads(Path('results/sprint_gpu_match_20260912/native_cat/suite/summary.json').read_text())
failed=json.loads(Path('results/sprint_gpu_match_20260912/native_comparison.json').read_text())['native_cat']['failed']
extra=[Path(next(e['case_path'] for e in previous['episodes'] if [e['case'],e['seed']]==f['case'])) for f in failed]+[mini]
patched=[]
for i,old in enumerate(extra):
 case=json.loads(old.read_text());case['control_config']=control
 target=a.out/'regressions'/f'{i}.json';target.parent.mkdir(exist_ok=True);atomic_json(target,case);patched.append(target)
extra=patched
s=run([*cases,*extra],a.out/'suite',workers=4,project=project)
if s['errors']:raise RuntimeError('Native replay errors; preserve attempts')
main_paths={str(Path(c).resolve()) for c in cases}
main={**s,'episodes':[e for e in s['episodes'] if e['case_path'] in main_paths]}
accept=paired_acceptance(main,cases);atomic_json(a.out/'paired_acceptance.json',accept)
c=[e for e in main['episodes'] if not json.loads(Path(e['case_path']).read_text())['ordinary_control']]
minimal=next(e for e in s['episodes'] if e['case']=='remove_forward_prefix')
regressions=[e for e in s['episodes'] if e['case_path'] not in main_paths]
result=dict(regression_count=len(regressions),regression_pass=sum(e['task_metrics']['success'] for e in regressions),regression_failures=[dict(case=e['case'],seed=e['seed'],metrics=e['task_metrics']) for e in regressions if not e['task_metrics']['success']],ordinary_pass=sum(e['task_metrics']['success'] for e in main['episodes'] if json.loads(Path(e['case_path']).read_text())['ordinary_control']),errors=s['errors'],candidate_count=len(c),candidate_pass=sum(e['task_metrics']['success'] for e in c),candidate_falls=sum(e['task_metrics']['fell'] for e in c),accepted=accept['accepted'] and all(e['task_metrics']['success'] for e in regressions),mini=minimal['task_metrics'],models=s['models'],comparisons=[{k:v for k,v in x.items() if k!='pairs'} for x in accept['comparisons']])
atomic_json(a.out/'completed.json',result);print(result,flush=True)
