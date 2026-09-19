"""Summarize a completed held-out comparison without modifying candidates."""
from pathlib import Path
import collections,hashlib,json,shutil
p=Path('results/jolt_learning_20260911');out=Path('docs/jolt_learning_20260911')
record=json.loads((p/'final_validation_complete.json').read_text());f=Path(record['directory'])
comparison=json.loads((f/'comparison.json').read_text())
summaries={label:json.loads((f/(label+'_suite/summary.json')).read_text()) for label in ['baseline','candidate']}
rows=[]
for name in comparison['cases']:
    row=dict(case=name)
    for label,s in summaries.items():
        es=[e for e in s['episodes'] if e['case']==name]
        row['skill']=es[0]['skill'];row['episodes']=len(es)
        bs=[b for e in es for b in e.get('brakes',[])]
        row[label]=dict(task_success=sum(e['task_metrics']['success'] for e in es),
          brake_success=sum(bool(e.get('brake_success')) for e in es),
          brake_events=len(bs),brake_falls=sum(b['fell'] for b in bs),
          unconfirmed_or_late=sum(b['stop_confirmed_s'] is None or b['stop_confirmed_s']>2 for b in bs),
          backwards=sum(b['sustained_backwards'] for b in bs))
    row['moving_speed_ratio']=comparison['cases'][name]['moving_speed_ratio'];rows.append(row)
active=[r for r in rows if r['skill']=='roller' and r['case'] not in ['roller_release_3s','roller_space_3s']]
counts={label:sum(r[label]['brake_success'] for r in active) for label in summaries}
cruise=all(r['moving_speed_ratio'] is None or r['moving_speed_ratio']>=.95 for r in active)
new_falls=sum(r['candidate']['brake_falls'] for r in active)
result=dict(protocol='Frozen final seeds 918000–918029, no post-final selection',model_sha256=record['frozen']['model_sha256'],cases=rows,active_braking=counts,active_episodes=210,regressions=comparison['regressions'],cruise_retained=cruise,candidate_brake_falls=new_falls,brake_hard_gate=counts['candidate']==210 and new_falls==0 and cruise,paired_quality_improvement=not comparison['regressions'] and cruise and counts['candidate']>counts['baseline'],final_seed_range_consumed=True,automatic_promotion=False,raw_directory=str(f),summaries_sha256={label:hashlib.sha256((f/(label+'_suite/summary.json')).read_bytes()).hexdigest() for label in summaries})
(p/'final_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for src,dest in [(p/'final_analysis.json','FINAL_RESULTS.json'),(p/'final_validation_complete.json','FINAL_VALIDATION.json'),(p/'release_extension_recovery.json','RELEASE_RECOVERY.json'),(f/'comparison.json','FINAL_COMPARISON.json'),(f/'switch_shadow.json','FINAL_SWITCH_SHADOW.json'),(f/'candidate_clean/container_environment.json','FINAL_CLEAN_ENVIRONMENT.json')]:shutil.copy2(src,out/dest)
print(json.dumps({k:v for k,v in result.items() if k not in ['cases','summaries_sha256']}))
