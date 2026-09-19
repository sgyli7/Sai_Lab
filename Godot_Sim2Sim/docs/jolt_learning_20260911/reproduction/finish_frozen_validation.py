"""Recover engineering shadow verification; never rescore reset tapes as task success."""
import json,time
from pathlib import Path
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.replay import shadow
from sim2sim.research.budget import live_group_members
p=Path('results/jolt_learning_20260911').resolve();f=p/'final_validation_retry'
start=time.monotonic()
while not (f/'candidate_switch/summary.json').exists():
 if not live_group_members(1535837) or time.monotonic()-start>1200:
  raise RuntimeError('Validation stopped before the engineering switch trace')
 time.sleep(5)
# The native reset tape is intentionally ineligible for the task-success scorer.
# Reuse its successful native execution for control parity, not task promotion.
switch=json.loads((f/'candidate_switch/summary.json').read_text())
assert len(switch['episodes'])==1
item=switch['episodes'][0]
assert switch['errors']==1 and item['error']=='ValueError: Task acceptance cases cannot mask outcomes with resets or robot replacement',item
trace=Path(item['attempt'])/'trace.json';data=json.loads(trace.read_text())
assert not data['summary']['error'] and len(data['rows'])==1250
assert data['summary']['resets']==2 and data['summary']['switches']==2
check=shadow(trace,project=f/'candidate_project');assert check['passed']
atomic_json(f/'switch_shadow.json',check)
for label in ['baseline','candidate']:
 summary=json.loads((f/(label+'_suite/summary.json')).read_text())
 assert summary['errors']==0 and len(summary['episodes'])==690
 assert {e['seed'] for e in summary['episodes']}==set(range(918000,918030))
clean=json.loads((f/'candidate_clean/summary.json').read_text());assert clean['errors']==0 and len(clean['episodes'])==23
comparison=json.loads((f/'comparison.json').read_text())
recovery=dict(reason='Runtime reset/switch trace mistakenly submitted to the strict task scorer. The scorer correctly refused resets; retained the successful trace only for engineering shadow parity.',replayed=False,task_scorer_changed=False,failed_attempt=item,engineering_shadow=check)
atomic_json(f/'switch_verification_recovery.json',recovery)
result=dict(completed=True,directory=str(f),frozen=json.loads((p/'final_candidate_freeze.json').read_text()),packages={label:json.loads((f/(label+'_package.json')).read_text()) for label in ['baseline','candidate']},paired_episodes=comparison['paired_episodes'],regressions=comparison['regressions'],final_seeds_consumed=True,clean_episodes=23,switch_shadow=check,engineering_recovery=str(f/'switch_verification_recovery.json'),automatic_promotion=False)
atomic_json(p/'final_validation_complete.json',result)
print('FINAL_VALIDATION_COMPLETE '+json.dumps({k:v for k,v in result.items() if k not in ['frozen','packages']}),flush=True)
