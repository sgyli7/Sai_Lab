"""Verify the frozen exported candidate without adapting its policy or control."""
from pathlib import Path
import hashlib,json,subprocess,time
import numpy as np
from sim2sim.standalone.suite import run
from sim2sim.standalone.replay import shadow
from sim2sim.research.queue import atomic_json
R=Path('results/sprint_joint_identification_20260913').resolve();out=R/'package_engineering';out.mkdir(exist_ok=False)
binary=Path(json.loads((R/'package_completed.json').read_text())['executable'])
image='ubuntu@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254'
prior=Path('results/sprint_joint_20260912/final_engineering/nine_new/summary.json');old=json.loads(prior.read_text())
models={k:v['sha256'] for k,v in json.loads((binary.parent/'models.json').read_text()).items()}
assert all(models[k]==v for k,v in old['models'].items() if k!='sprint')
cases=[Path(e['case_path']) for e in old['episodes']]
dev=json.loads((R/'integrated_dev/suite/summary.json').read_text())
extra=[next(e for e in dev['episodes'] if e['case']==name and e['seed']==927001) for name in ['sprint_alternate','sprint_long_ordinary']]
summary=run(cases+[Path(e['case_path']) for e in extra],out/'clean_container',workers=4,executable=binary,container_image=image)
assert not summary['errors']
index={(e['case'],e['seed']):e for e in summary['episodes']}
pairs=[]
for e in old['episodes']:
 new=index[e['case'],e['seed']]
 assert e['case_sha256']==new['case_sha256']
 pairs.append(dict(case=e['case'],seed=e['seed'],before=e['task_metrics']['success'],after=new['task_metrics']['success'],old_brake=e.get('brake_success'),new_brake=new.get('brake_success')))
losses=[v for v in pairs if v['before'] and not v['after']];brake_losses=[v for v in pairs if v['old_brake'] is True and v['new_brake'] is not True]
equivalence=[]
for e in extra:
 new=index[e['case'],e['seed']];a=json.loads(Path(e['trace']).read_text())['rows'];b=json.loads(Path(new['trace']).read_text())['rows']
 maxima={k:float(np.max(np.abs(np.asarray([x[k] for x in a])-np.asarray([x[k] for x in b])))) for k in ['obs','action','command','ctrl','last_action']}
 assert max(maxima.values())<1e-5,maxima
 equivalence.append(dict(case=e['case'],seed=e['seed'],max_abs=maxima,success=new['task_metrics']['success']))
atomic_json(out/'nine_regression.json',dict(passed=not losses and not brake_losses,baseline=str(prior),baseline_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),count=len(pairs),old_pass=sum(v['before'] for v in pairs),new_pass=sum(v['after'] for v in pairs),lost_successes=losses,brake_losses=brake_losses,pairs=pairs,export_equivalence=equivalence))
assert not losses and not brake_losses
case=Path('results/sprint_20260912/preview_release_lifecycle/case.json').resolve();runs=[];traces=[]
for fps in [30,144]:
 trace=out/f'lifecycle_{fps}.json';command=[str(binary),'--','--render-fps='+str(fps),'--replay='+str(case),'--trace='+str(trace)]
 with (out/f'lifecycle_{fps}.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=60,check=True)
 data=json.loads(trace.read_text());s=data['summary'];assert not s['error'] and s['steps']==600 and s['resets']==1 and s['switches']==6
 ref=shadow(trace,R/'integrated_dev/suite/runtime');assert ref['passed'],ref
 runs.append(dict(fps=fps,summary=s,shadow=ref));traces.append(data)
maxima={k:float(np.max(np.abs(np.asarray([x[k] for x in traces[0]['rows']])-np.asarray([x[k] for x in traces[1]['rows']])))) for k in ['obs','action','command','ctrl','last_action']}
assert max(maxima.values())<1e-5
assert [x['skill'] for x in traces[0]['rows']]==[x['skill'] for x in traces[1]['rows']]
atomic_json(out/'fps_lifecycle.json',dict(passed=True,max_abs=maxima,runs=runs,input_source='exported replay, not OS keyboard'))
atomic_json(out/'completed.json',dict(completed=True,nine_regression_passed=True,export_equivalence_passed=True,fps_lifecycle_passed=True))
print('package engineering passed',flush=True)
