from pathlib import Path
import json,copy,subprocess,shutil
import numpy as np
from sim2sim.godot_proc import _headless_overlay
from sim2sim.standalone.score import score
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_observability_20260913').resolve();out=r/'reproduce';out.mkdir(exist_ok=False)
prior=Path('results/sprint_stop_state_20260912/native_joint_fd').resolve();summary=json.loads((prior/'suite/summary.json').read_text());entry=next(e for e in summary['episodes'] if e['case']=='sprint_alternate' and e['seed']==927001)
original=json.loads(Path(entry['case_path']).read_text());case=copy.deepcopy(original);case['seconds']=9.;case['scoring']['end']=9.;case['segments']=[s for s in case['segments'] if s['at']<9.];case['sprint_intervals']=[[1.,9.]]
variants=[]
for i in range(3):variants.append((f'original_{i}',copy.deepcopy(case)))
x=copy.deepcopy(case);x['seconds']=8.;x['scoring']['end']=8.;x['segments']=[dict(s,at=s['at']-1.) for s in x['segments'] if s['at']>=1.];x['sprint_intervals']=[[0.,8.]];variants.append(('remove_idle',x))
x=copy.deepcopy(case);x['segments'][2]['held']=['sprint','fwd'];variants.append(('remove_first_turn',x))
results=[]
for name,c in variants:
 folder=out/name;folder.mkdir();path=folder/'case.json';atomic_json(path,c);overlay=_headless_overlay(prior/'suite/runtime')
 try:
  with (folder/'player.log').open('w') as log:subprocess.run(['godot','--headless','--fixed-fps','200','--path',str(overlay),'res://standalone/main.tscn','--','--replay='+str(path),'--trace='+str(folder/'trace.json')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=20)
 finally:shutil.rmtree(overlay)
 v=score(folder/'trace.json',path);atomic_json(folder/'score.json',v);results.append(dict(name=name,task=v['task_metrics'],trace=str(folder/'trace.json'),case=str(path)));print(name,v['task_metrics']['success'],v['task_metrics']['turns'],flush=True)
assert all(not x['task']['success'] and any(not t['passed'] for t in x['task']['turns']) and not x['task']['fell'] for x in results[:3])
old=json.loads(Path(entry['trace']).read_text())['rows'][:450];new=json.loads(Path(results[0]['trace']).read_text())['rows'];maximum={k:float(np.abs(np.array([x[k] for x in old])-np.array([x[k] for x in new])).max()) for k in ['obs','action','command','ctrl','last_action']};assert max(maximum.values())==0.,maximum
atomic_json(out/'completed.json',dict(reproduced=True,results=results,original_prefix_max_abs=maximum,scope='Frozen nine-second turn-reversal failure; ablations diagnose prefix dependence, not policy improvements'))
