from pathlib import Path
import json
import numpy as np
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_gpu_match_20260912')
names=['native_s05','native_source_graph','native_jolt_graph','native_positive','native_cat']
all_results={}
for name in names:
 summary=json.loads((r/name/'suite/summary.json').read_text());completed=json.loads((r/name/'completed.json').read_text())
 assert summary['errors']==0
 indexed={}
 for e in summary['episodes']:
  case=json.loads(Path(e['case_path']).read_text())
  if case.get('ordinary_control',False):continue
  indexed[(e['case'],int(case.get('seed',e.get('seed',-1))))]=e
 all_results[name]=(completed,indexed)
baseline=all_results[names[0]][1]
report={}
for name,(completed,rows) in all_results.items():
 assert rows.keys()==baseline.keys()
 lost=[];gained=[];failed=[]
 for key,e in rows.items():
  before=baseline[key]['task_metrics'];after=e['task_metrics']
  if before['success'] and not after['success']:lost.append(key)
  if not before['success'] and after['success']:gained.append(key)
  if not after['success']:
   failed.append(dict(case=key,fall=after['fell'],straight=[x for x in after.get('straight',[]) if not x['passed']],turns=[x for x in after.get('turns',[]) if not x['passed']],stops=[x for x in after.get('stops',[]) if not x['passed']]))
 ordinary_before={}
 for e in json.loads((r/names[0]/'suite/summary.json').read_text())['episodes']:
  case=json.loads(Path(e['case_path']).read_text())
  if case.get('ordinary_control',False):ordinary_before[(e['case'],e['seed'])]=e
 ordinary=[]
 for e in json.loads((r/name/'suite/summary.json').read_text())['episodes']:
  if (e['case'],e['seed']) not in ordinary_before:continue
  old=ordinary_before[(e['case'],e['seed'])]['task_metrics'];new=e['task_metrics'];ordinary.append(dict(case=e['case'],same_metrics=new==old,lost_success=old['success'] and not new['success'],old_speed=old['sustained_mean_vx'],new_speed=new['sustained_mean_vx']))
 report[name]=dict(completed=completed,lost_successes=lost,gained_successes=gained,failed=failed,ordinary_count=len(ordinary),ordinary_identical=sum(x['same_metrics'] for x in ordinary),ordinary_lost=sum(x['lost_success'] for x in ordinary),eligible=completed['accepted'] and not lost and not any(x['lost_success'] for x in ordinary))
atomic_json(r/'native_comparison.json',report)
for name,s in report.items():print(name,s['completed']['candidate_pass'],'/128, falls',s['completed']['candidate_falls'],'lost',len(s['lost_successes']),'gained',len(s['gained_successes']),'eligible',s['eligible'],flush=True)
