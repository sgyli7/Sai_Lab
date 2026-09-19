from pathlib import Path
import copy,json,subprocess,shutil
from sim2sim.standalone.suite import run
from sim2sim.standalone.replay import shadow
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_exit_20260912');out=r/'fast_entry_check';out.mkdir(exist_ok=False)
project=r/'running_zero_shot/suite/runtime'
trace=r/'running_zero_shot/suite/fast_source_2.2_923000/attempt_01/trace.json'
audit=shadow(trace,project);atomic_json(out/'shadow.json',audit);assert audit['passed']
with (out/'native_self_test.log').open('w') as log:
 subprocess.run(['godot','--headless','--path',str(project),'res://standalone/main.tscn','--','--self-test'],stdout=log,stderr=subprocess.STDOUT,timeout=30,check=True)
records={}
for mode in ['cold','source_idle']:
 prepared=project
 if mode=='source_idle':
  prepared=out/'prepared';subprocess.run(['cp','--reflink=auto','-a',str(project),str(prepared)],check=True,timeout=60)
  deployment=prepared/'runtime_assets/deployment.json';d=json.loads(deployment.read_text());d['policies']['walking']=copy.deepcopy(d['policies']['sprint']);d['policies']['walking']['manifest']['sim2sim']['skill']='walking';atomic_json(deployment,d)
 cases=[]
 for seed in [923000,923001,923002,923003]:
  source=json.loads((r/f'running_zero_shot/fast_source_2.2_{seed}.json').read_text());source['case']='fast_'+mode+'_'+str(seed)
  if mode=='cold':source['segments']=[dict(at=0,held=['sprint','fwd']),dict(at=3,held=[])];source['seconds']=6;source['scoring']['end']=6;source['sprint_intervals']=[[0,3]]
  p=out/(source['case']+'.json');atomic_json(p,source);cases.append(p)
 s=run(cases,out/mode,workers=4,project=prepared)
 records[mode]=[dict(case=x['case'],metrics=x['task_metrics']) for x in s['episodes']]
 atomic_json(out/'progress.json',records)
atomic_json(out/'completed.json',dict(shadow=audit,results=records));print(records,flush=True)
