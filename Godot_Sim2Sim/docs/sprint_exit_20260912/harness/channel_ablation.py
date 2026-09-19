from pathlib import Path
import copy,json
from sim2sim.standalone.suite import run
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_exit_20260912');out=r/'channel_ablation';out.mkdir(exist_ok=False)
mini=json.loads((r/'minimize/remove_forward_prefix.json').read_text());base=list((r/'baseline_exits/cases').glob('*.json'))
records={}
for name,change in [('position_only',{'walk_path_lookahead':0.}),('steering_only',{'walk_path_limit':0.})]:
 d=out/name;d.mkdir();cases=[]
 for case in [mini]+[json.loads(p.read_text()) for p in sorted(base)]:
  case=copy.deepcopy(case);case['control_config']['walk'].update(change)
  p=d/(case['case']+'_'+str(case['seed'])+'.json');atomic_json(p,case);cases.append(p)
 s=run(cases,d/'suite',workers=4,executable='dist/MicroDuck-ARM64-20260912-joint-trial/MicroDuck.arm64')
 records[name]=dict(errors=s['errors'],count=len(cases),passed=sum(x['task_metrics']['success'] for x in s['episodes']),
  mini=next(x['task_metrics'] for x in s['episodes'] if x['case']=='remove_forward_prefix'),
  failures=[dict(case=x['case'],seed=x['seed'],metrics=x['task_metrics']) for x in s['episodes'] if not x['task_metrics']['success']])
 atomic_json(out/'progress.json',records)
atomic_json(out/'completed.json',records);print(records,flush=True)
