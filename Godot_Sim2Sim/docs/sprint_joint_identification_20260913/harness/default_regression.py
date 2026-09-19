from pathlib import Path
import json
from sim2sim.standalone.suite import run
from sim2sim.research.queue import atomic_json
R=Path('results/sprint_joint_identification_20260913').resolve();out=R/'default_regression';out.mkdir(exist_ok=False)
prior=json.loads(Path('results/sprint_joint_20260912/final_engineering/nine_new/summary.json').read_text());cases=[Path(e['case_path']) for e in prior['episodes']]
summaries=[]
for name,project in [('before',R/'default_before/runtime'),('after',Path('/home/ethan/Projects/MicroDuck-Sprint-Default/godot'))]:
 summaries.append(run(cases,out/name,workers=4,project=project))
a,b=summaries;assert not a['errors'] and not b['errors']
index={(e['case'],e['seed']):e for e in a['episodes']};pairs=[]
for e in b['episodes']:
 old=index[e['case'],e['seed']];assert old['case_sha256']==e['case_sha256']
 pairs.append(dict(case=e['case'],seed=e['seed'],before=old['task_metrics']['success'],after=e['task_metrics']['success'],before_brake=old.get('brake_success'),after_brake=e.get('brake_success')))
losses=[v for v in pairs if v['before'] and not v['after']];brake_losses=[v for v in pairs if v['before_brake'] is True and v['after_brake'] is not True]
result=dict(completed=True,passed=not losses and not brake_losses,count=len(pairs),before_pass=sum(v['before'] for v in pairs),after_pass=sum(v['after'] for v in pairs),losses=losses,brake_losses=brake_losses,pairs=pairs,models_before=a['models'],models_after=b['models'])
atomic_json(out/'completed.json',result);print({k:v for k,v in result.items() if k not in ['pairs','models_before','models_after']},flush=True)
