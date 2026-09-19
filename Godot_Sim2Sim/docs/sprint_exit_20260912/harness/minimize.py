from pathlib import Path
import copy,json
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.suite import run
r=Path('results/sprint_exit_20260912');out=r/'minimize';out.mkdir(exist_ok=False)
source=json.loads((r/'baseline_exits/cases/sprint_right_923000.json').read_text());records=[]
def probe(label,seconds,segments):
 case=copy.deepcopy(source);case.update(case=label,seconds=seconds,segments=[dict(at=t,held=keys) for t,keys in segments],scoring=dict(start=0,end=seconds))
 case['sprint_intervals']=[]
 for i,s in enumerate(case['segments']):
  end=case['segments'][i+1]['at'] if i+1<len(case['segments']) else seconds
  if 'fwd' in s['held'] and 'sprint' in s['held']:case['sprint_intervals'].append([s['at'],end])
 p=out/(label+'.json');atomic_json(p,case)
 s=run([p],out/label,workers=1,executable='dist/MicroDuck-ARM64-20260912-joint-trial/MicroDuck.arm64')
 row=s['episodes'][0];record=dict(label=label,seconds=seconds,case_path=str(p),metrics=row['task_metrics'],trace=row['trace']);records.append(record)
 atomic_json(out/'progress.json',records)
 return record
f=['sprint','fwd'];turn=f+['right']
a=probe('trim_stop',10,[(0,[]),(1,f),(3,turn),(8,f)])
b=probe('trim_stop_repeat',10,[(0,[]),(1,f),(3,turn),(8,f)])
assert not a['metrics']['success'] and not b['metrics']['success']
ra=json.loads(Path(a['trace']).read_text())['rows'];rb=json.loads(Path(b['trace']).read_text())['rows']
assert all(all(x[k]==y[k] for k in ['obs','action','command','ctrl']) for x,y in zip(ra,rb))
c=probe('remove_forward_prefix',8,[(0,[]),(1,turn),(6,f)])
if not c['metrics']['success']:
 chosen=c;idle=probe('remove_initial_idle',7,[(0,turn),(5,f)])
 if not idle['metrics']['success']:chosen=idle
else:
 chosen=a;probe('remove_idle_keep_forward',9,[(0,f),(2,turn),(7,f)])
atomic_json(out/'completed.json',dict(reproduced=True,deterministic=True,chosen=chosen,attempts=records))
print([(x['label'],x['metrics']['success'],x['metrics']['straight']) for x in records],flush=True)
