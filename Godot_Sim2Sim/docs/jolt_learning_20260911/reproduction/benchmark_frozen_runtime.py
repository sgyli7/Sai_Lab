"""Measure only after final replay workers have stopped."""
import json,time
from pathlib import Path
from sim2sim.research.budget import live_group_members
from sim2sim.standalone.benchmark import benchmark
p=Path('results/jolt_learning_20260911');start=time.monotonic()
while not (p/'final_validation_complete.json').exists() or live_group_members(1535837) or live_group_members(1674525):
 if time.monotonic()-start>1320:raise RuntimeError('Final validation did not finish in the reserved interval')
 time.sleep(5)
record=json.loads((p/'final_validation_complete.json').read_text())
template=json.loads((p/'contract_switch_case.json').read_text())
case=dict(case='frozen_runtime_30_sim_minutes',mode='roller',seconds=1800,segments=[])
for cycle in range(72):
 for segment in template['segments']:case['segments'].append({**segment,'at':segment['at']+cycle*25})
replay=p/'frozen_runtime_benchmark_case.json';replay.write_text(json.dumps(case,indent=2)+'\n')
r=benchmark(record['packages']['candidate']['executable'],p/'frozen_runtime_benchmark',seconds=1800,replay=replay,timeout=360)
assert r['completed']
assert r['player']['resets']==144 and r['player']['switches']==144
print(json.dumps({k:v for k,v in r.items() if k not in ['memory_samples','player']}),flush=True)
