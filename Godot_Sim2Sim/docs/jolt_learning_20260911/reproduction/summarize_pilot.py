"""Read completed native comparisons, not training reward or directory presence."""
import hashlib,json,statistics
from pathlib import Path
p=Path('results/jolt_learning_20260911');rows=[]
for filename in ['r0_progress.json','r1_progress.json','r2_progress.json','extension_progress.json','r3_progress.json']:
 f=p/filename
 if not f.exists():continue
 for r in json.loads(f.read_text()):
  entry={k:r[k] for k in ['name','seed','completed','passes','regressions','eligible','elapsed_s'] if k in r}
  if not r.get('completed'):
   entry['error']=r.get('error');rows.append(entry);continue
  cp=Path(r['comparison']);c=json.loads(cp.read_text());s=json.loads((cp.parent/'suite/summary.json').read_text())
  es=[e for e in s['episodes'] if e['case'] not in ['roller_release_3s','roller_space_3s']];brakes=[b for e in es for b in e['brakes']]
  stop=[b['stop_confirmed_s'] for b in brakes if b['stop_confirmed_s'] is not None]
  entry.update(new_samples=sum(s['training']['samples'] for s in r.get('segments',[])) or r['training']['samples'],cumulative_samples=r.get('cumulative_samples',r['training']['samples']),model_sha256=s['models']['roller'],falls=sum(b['fell'] for b in brakes),late_or_unconfirmed=sum(b['stop_confirmed_s'] is None or b['stop_confirmed_s']>2 for b in brakes),reversing_brakes=sum(b['sustained_backwards'] for b in brakes),brake_events=len(brakes),median_stop_confirmed_s=statistics.median(stop) if stop else None,max_stop_confirmed_s=max(stop) if stop else None,cases={k:dict(passes=v['brake_passes_after'],baseline=v['brake_passes_before'],moving_speed_ratio=v['moving_speed_ratio']) for k,v in c['cases'].items()},raw_comparison=str(cp),export_parity=r['training']['final_parity'])
  entry['roller_cruise_retained']=all(v['moving_speed_ratio'] is None or v['moving_speed_ratio']>=.95 for v in c['cases'].values())
  entry['eligible_improvement']=r['eligible'] and r['passes']>51 and entry['roller_cruise_retained']
  rows.append(entry)
result=dict(baseline_active_passes=51,active_cases=56,development_seed_range='917000–917007',final_seed_range='918000–918029',trials=rows,promoted=False)
(p/'trial_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps([dict(name=r['name'],passes=r.get('passes'),regressions=r.get('regressions'),cruise=r.get('roller_cruise_retained'),improved=r.get('eligible_improvement')) for r in rows],indent=2))
