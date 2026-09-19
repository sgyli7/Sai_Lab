"""Compare fixed endpoints and fail closed on missing cases or ordinary regressions."""
from pathlib import Path
import argparse,json,math
from sim2sim.research.queue import atomic_json
ROOT=Path('results/sprint_joint_gpu_20260912')

def compare(base_dir,candidate_dir):
 b=json.loads((base_dir/'suite/summary.json').read_text());s=json.loads((candidate_dir/'suite/summary.json').read_text())
 bc=json.loads((base_dir/'completed.json').read_text());c=json.loads((candidate_dir/'completed.json').read_text())
 errors=[]
 def index(summary):
  out={}
  for e in summary['episodes']:
   key=(e['case'],e['seed'])
   if key in out:errors.append('Duplicate case '+str(key))
   if not e.get('completed') or e.get('error') or e.get('returncode')!=0:errors.append('Incomplete case '+str(key))
   out[key]=e
  return out
 old,new=index(b),index(s)
 if set(old)!=set(new):errors.append('Case inventory changed')
 if b['errors'] or s['errors']:errors.append('Runtime errors')
 if len(new)!=263:errors.append('Expected 263 physical cases')
 losses=[];fixes=[];failed=[]
 for key in sorted(set(old)&set(new)):
  before,after=old[key],new[key]
  if before['case_sha256']!=after['case_sha256'] or before['control_config']!=after['control_config']:errors.append('Case or control changed '+str(key))
  for skill,sha in before['models'].items():
   if skill not in ('walking','sprint') and after['models'].get(skill)!=sha:errors.append('Unrelated skill model changed '+skill)
  bm,am=before['task_metrics'],after['task_metrics']
  if bm['success'] and not am['success']:losses.append(list(key))
  if am['success'] and not bm['success']:fixes.append(list(key))
  if not am['success']:
   failed.append(dict(case=list(key),fell=am['fell'],straight=[v for v in am['straight'] if not v['passed']],turns=[v for v in am['turns'] if not v['passed']],stops=[v for v in am['stops'] if not v['passed']]))
 old_speed={v['case']:v for v in bc['comparisons']};retention=[]
 for v in c['comparisons']:
  prev=old_speed[v['case']]['ordinary_vx'];now=v['ordinary_vx']
  retention.append(dict(case=v['case'],baseline_vx=prev,candidate_vx=now,ratio=now/prev,passed=math.isfinite(now) and now>=.95*prev))
 if set(v['case'] for v in retention)!=set(old_speed):errors.append('Ordinary speed comparison missing')
 all_physical=all(e['task_metrics']['success'] and not e['task_metrics']['fell'] for e in new.values())
 eligible=bool(not errors and all_physical and c['accepted'] and not losses and all(x['passed'] for x in retention))
 return dict(arm=candidate_dir.name,errors=errors,counts=dict(sprint_pass=c['candidate_pass'],sprint_count=c['candidate_count'],ordinary_pass=c['ordinary_pass'],ordinary_count=128,regression_pass=c['regression_pass'],regression_count=c['regression_count']),falls=sum(e['task_metrics']['fell'] for e in new.values()),lost_baseline_successes=losses,fixed_baseline_failures=fixes,failed=failed,paired_speed=c['comparisons'],ordinary_retention=retention,all_physical_pass=all_physical,development_eligible=eligible,final_seeds_used=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('arms',nargs='+');p.add_argument('--output',type=Path,default=ROOT/'comparison.json');a=p.parse_args()
 out={name:compare(ROOT/'baseline_s05',ROOT/name) for name in a.arms};atomic_json(a.output,out)
 for name,v in out.items():print(name,json.dumps({k:v[k] for k in ['counts','falls','lost_baseline_successes','fixed_baseline_failures','development_eligible']}))
