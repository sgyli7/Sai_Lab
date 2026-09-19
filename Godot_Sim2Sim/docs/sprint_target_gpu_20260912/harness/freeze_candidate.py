"""Run only after every fixed development endpoint is complete and qualified."""
from pathlib import Path
import hashlib,json,shutil,subprocess,time
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.prepare import prepare
from sim2sim.standalone.package import package
from sim2sim.standalone.sprint import write_cases
from sim2sim.standalone.sprint_accept import paired_acceptance
from sim2sim.standalone.suite import run
r=Path('results/sprint_target_gpu_20260912');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
arms=[r/f'native_{i:03d}' for i in [6,12]]+[r/f'native_warm_{i:03d}' for i in [6,12]]+[r/f'native_initial_teacher_{i:03d}' for i in [6,12]]
eligible=[]
for arm in arms:
 c=json.loads((arm/'comparison.json').read_text());assert all(not x['errors'] for x in c.values())
 if all(x['development_eligible'] for x in c.values()):
  speed=next(x['candidate_vx'] for x in c['vs_initial']['paired_speed'] if x['case']=='sprint_long');eligible.append((speed,arm))
if not eligible:raise RuntimeError('No development-qualified candidate; final seeds remain unused')
_,selected=max(eligible,key=lambda x:x[0]);out=r/'delivery';out.mkdir(exist_ok=False)
models=out/'models';shutil.copytree(selected/'models',models)
control=json.loads((selected/'control.json').read_text());atomic_json(out/'control.json',control)
summary=json.loads((selected/'suite/summary.json').read_text());assert summary['models']=={k:v['sha256'] for k,v in json.loads((selected/'suite/runtime/runtime_assets/deployment.json').read_text())['policies'].items()}
assert sha(models/'Walk_Godot.onnx')==sha('results/sprint_20260912/runs/s05_native_handoff/final.onnx')
atomic_json(out/'freeze.json',dict(created_unix=time.time(),source=str(selected),models=summary['models'],control=control,development_summary_sha256=sha(selected/'suite/summary.json'),final_seed_start=928000,final_seed_stop=928050,expected_episodes=800,rule='One frozen candidate; no selection or parameter change after final results',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),promoted=False))
project=out/'runtime';subprocess.run(['cp','--reflink=auto','-a',str(selected/'suite/runtime'),str(project)],check=True,timeout=60)
traces={}
for e in json.loads(Path('results/research_20260911/final_nominal_candidate/summary.json').read_text())['episodes']:
 if e.get('completed') and e.get('trace'):traces.setdefault(e['skill'],e['trace'])
for ordinary,skill in [(False,'sprint'),(True,'walking')]:
 traces[skill]=next(e['trace'] for e in summary['episodes'] if e['case']=='sprint_long'+('_ordinary' if ordinary else ''))
prepare(models,project=project,sprint=models/'Sprint_Godot.onnx',control_config=out/'control.json',real_traces=list(traces.values()))
fixtures=json.loads((project/'runtime_assets/self_test.json').read_text());assert len(fixtures['cases'])==10 and all(x['real_count']>0 for x in fixtures['cases'])
pack=package('dist/MicroDuck-ARM64-20260912-target-trial',project=project);atomic_json(out/'package.json',pack)
cases=write_cases(out/'final_cases',range(928000,928050),.3,paired=True,control=control)
atomic_json(out/'final_case_index.json',{str(p.resolve()):sha(p) for p in cases})
result=run(cases,out/'final_suite',workers=4,executable=pack['executable'],timeout=45,container_image='ubuntu@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254')
acceptance=paired_acceptance(result,cases)
all_physical=all(e['task_metrics']['success'] and not e['task_metrics']['fell'] for e in result['episodes'])
atomic_json(out/'final_acceptance.json',dict(paired=acceptance,all_physical=all_physical,passed=not result['errors'] and len(result['episodes'])==800 and acceptance['accepted'] and all_physical))
atomic_json(out/'completed.json',dict(completed=True,accepted=acceptance['accepted'] and all_physical and not result['errors'],promoted=False,engineering_pending=True))
print((out/'final_acceptance.json').read_text(),flush=True)
