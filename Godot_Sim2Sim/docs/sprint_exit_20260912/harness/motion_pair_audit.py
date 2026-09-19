from pathlib import Path
import json,hashlib
import onnx
import numpy as np
from onnx import numpy_helper
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_exit_20260912');names=['e04_motion_visible','e05_motion_masked']
configs=[json.loads((r/'runs'/n/'config.json').read_text()) for n in names]
ignore={'name','mask_motion_state','start_unix','deadline_unix'}
changed={k:[configs[0].get(k),configs[1].get(k)] for k in set(configs[0])|set(configs[1]) if k not in ignore and configs[0].get(k)!=configs[1].get(k)}
assert not changed,changed
models=[onnx.load(r/'runs'/n/'initial.onnx') for n in names]
assert [n.SerializeToString() for n in models[0].graph.node]==[n.SerializeToString() for n in models[1].graph.node]
tensors=[{x.name:numpy_helper.to_array(x) for x in m.graph.initializer} for m in models]
assert tensors[0].keys()==tensors[1].keys()
differences=[k for k in tensors[0] if not np.array_equal(tensors[0][k],tensors[1][k])]
assert differences==['adapt/observation_mask'],differences
expected=tensors[0][differences[0]].copy();expected[58:61]=0
np.testing.assert_array_equal(expected,tensors[1][differences[0]])
completed=[json.loads((r/'runs'/n/'completed.json').read_text()) for n in names]
assert all(x['iterations']==96 and x['samples']==393216 and x['status']=='iteration_limit' for x in completed)
records=[]
for name,c in zip(names,configs):
 rows=[json.loads(line) for line in (r/'runs'/name/'metrics.jsonl').read_text().splitlines()]
 records.append(dict(name=name,learner_device=c['learner_device'],accelerator=c['accelerator'],iterations=len(rows),samples=rows[-1]['samples'],collection_seconds=sum(x['collection_seconds'] for x in rows),learner_seconds=sum(x['learner_seconds'] for x in rows),updates=sum(x['actor_updates'] for x in rows),final_sha256=hashlib.sha256((r/'runs'/name/'final.onnx').read_bytes()).hexdigest()))
assert all(x['learner_device']=='cuda' for x in records)
atomic_json(r/'motion_pair_audit.json',dict(passed=True,shared_config_except=sorted(ignore),unexpected_config_differences=changed,initial_graph_nodes_equal=True,initial_tensor_differences=differences,only_feature_difference='hide slots 58:61 from residual; identical critic and normalization',runs=records))
print(records,flush=True)
