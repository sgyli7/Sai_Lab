from pathlib import Path
import json,copy
from compare import compare
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_joint_gpu_20260912');base=r/'baseline_s05';candidate=r/'native_feedback'
original=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text());expected=json.loads((r/'feedback_control.json').read_text())
assert expected==dict(original,version='sprint_turn_tracking_v1',walk=dict(original['walk'],walk_sprint_turn_tracking=True))
b={(e['case'],e['seed']):e for e in json.loads((base/'suite/summary.json').read_text())['episodes']};new=json.loads((candidate/'suite/summary.json').read_text())['episodes']
for e in new:
 prior=b[(e['case'],e['seed'])];old_case=json.loads(Path(prior['case_path']).read_text());new_case=json.loads(Path(e['case_path']).read_text())
 assert new_case.pop('control_config')==expected and old_case.pop('control_config')==original
 assert old_case==new_case and e['control_config']==expected and prior['control_config']==original
v=compare(base,candidate)
v['errors']=[e for e in v['errors'] if not e.startswith('Case or control changed ')]
v['development_eligible']=bool(not v['errors'] and v['all_physical_pass'] and not v['lost_baseline_successes'] and all(x['passed'] for x in v['ordinary_retention']) and all(x['speed_pass'] for x in v['paired_speed']))
v.update(kind='Control intervention, not additional model improvement',reference_routing='Same tracking sprint actor and S05 ordinary as native_tracking_split',contract_change=expected,only_declared_control_changed=True)
atomic_json(r/'feedback_comparison.json',v)
print({k:v[k] for k in ['counts','falls','lost_baseline_successes','development_eligible']},flush=True)
