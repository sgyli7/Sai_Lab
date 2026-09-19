from pathlib import Path
import argparse,json,math
import numpy as np
from sim2sim.standalone.replay import shadow
from sim2sim.research.queue import atomic_json
p=argparse.ArgumentParser();p.add_argument('arm');p.add_argument('--shadow',action='store_true');a=p.parse_args()
r=Path('results/sprint_joint_gpu_20260912');suite=r/a.arm/'suite';comparison=json.loads((r/'comparison.json').read_text())[a.arm]
episodes=json.loads((suite/'summary.json').read_text())['episodes'];baseline=json.loads((r/'baseline_s05/suite/summary.json').read_text())['episodes'];records=[]
for failure in comparison['failed']:
 name,seed=failure['case'];e=next(x for x in episodes if (x['case'],x['seed'])==(name,seed));old=next(x for x in baseline if (x['case'],x['seed'])==(name,seed))
 rows=json.loads(Path(e['trace']).read_text())['rows'];case=json.loads(Path(e['case_path']).read_text());segments=[]
 for kind in ['straight','turns','stops']:
  for failed in failure[kind]:
   start=failed['at'];idx=next(i for i,x in enumerate(case['segments']) if x['at']==start);end=case['segments'][idx+1]['at'] if idx+1<len(case['segments']) else case['seconds'];active=[x for x in rows if start-1e-9<=x['t']<end-1e-9]
   before=max((x for x in rows if x['t']<start),key=lambda x:x['t']);first=active[0]
   segments.append(dict(kind=kind,start=start,end=end,held=case['segments'][idx]['held'],skills=sorted({x['skill'] for x in active}),previous_skill=before['skill'],action_jump_l2=float(np.linalg.norm(np.array(first['action'])-before['action'])),metrics=failed))
 verification=shadow(e['trace'],suite/'runtime') if a.shadow else None
 if verification:assert verification['passed']
 records.append(dict(case=name,seed=seed,baseline_success=old['task_metrics']['success'],segments=segments,shadow=verification))
result=dict(arm=a.arm,records=records,all_failures_after_sprint_to_walking=bool(records) and all(x['previous_skill']=='sprint' and x['skills']==['walking'] for r in records for x in r['segments']))
atomic_json(r/(a.arm+'_handoff.json'),result)
print(a.arm,'failures',len(records),'all after sprint to walking',result['all_failures_after_sprint_to_walking'],flush=True)
