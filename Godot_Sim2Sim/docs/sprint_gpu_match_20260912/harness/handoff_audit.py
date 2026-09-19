from pathlib import Path
import json
import numpy as np
from sim2sim.standalone.replay import shadow
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_gpu_match_20260912');comparison=json.loads((r/'native_comparison.json').read_text())['native_cat'];suite=r/'native_cat/suite';episodes=json.loads((suite/'summary.json').read_text())['episodes'];baseline=json.loads((r/'native_s05/suite/summary.json').read_text())['episodes'];records=[]
for failure in comparison['failed']:
 name,seed=failure['case'];e=next(x for x in episodes if x['case']==name and x['seed']==seed);old=next(x for x in baseline if x['case']==name and x['seed']==seed)
 rows=json.loads(Path(e['trace']).read_text())['rows'];case=json.loads(Path(e['case_path']).read_text());segments=[]
 for kind in ['straight','turns','stops']:
  for failed in failure[kind]:
   start=failed['at'];idx=next(i for i,x in enumerate(case['segments']) if x['at']==start);end=case['segments'][idx+1]['at'] if idx+1<len(case['segments']) else case['seconds'];active=[x for x in rows if start-1e-9<=x['t']<end-1e-9]
   before=max((x for x in rows if x['t']<start),key=lambda x:x['t']);first=active[0]
   segments.append(dict(kind=kind,start=start,end=end,held=case['segments'][idx]['held'],skills=sorted({x['skill'] for x in active}),previous_skill=before['skill'],action_jump_l2=float(np.linalg.norm(np.array(first['action'])-before['action'])),metrics=failed))
 verification=shadow(e['trace'],suite/'runtime');assert verification['passed']
 records.append(dict(case=name,seed=seed,baseline_success=old['task_metrics']['success'],segments=segments,shadow=verification))
result=dict(records=records,all_failures_after_sprint_to_walking=all(x['previous_skill']=='sprint' and x['skills']==['walking'] for r in records for x in r['segments']),all_old_cases_pass=all(x['baseline_success'] for x in records),interpretation='All six new failing segments run frozen ordinary S05 after sprint. This supports an off-distribution handoff hypothesis; not proof of a specific mechanical cause or a tested unified-policy remedy.')
atomic_json(r/'handoff_audit.json',result);print(result,flush=True)
