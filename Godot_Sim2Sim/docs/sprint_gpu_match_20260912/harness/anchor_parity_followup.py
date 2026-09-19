from pathlib import Path
import json,time
import numpy as np
import torch
from sim2sim.research.torch_anchor import TorchAnchor
from sim2sim.research.models import NativeAnchor
from sim2sim.research.queue import atomic_json
from sim2sim.train.onnx_import import _sample_realistic_obs61
r=Path('results/sprint_gpu_match_20260912');out=r/'anchor_parity_followup';out.mkdir(exist_ok=False)
assert torch.cuda.is_available();torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
sources={'motion_state':'results/sprint_exit_20260912/runs/e04_motion_visible/final.onnx'}
real=np.array([row['obs'] for row in json.loads(Path('results/sprint_exit_20260912/state_shadow/trace.json').read_text())['rows']],np.float32)
rng=np.random.default_rng(925991);random=rng.normal(0,.5,(256,61)).astype(np.float32)
inputs=np.concatenate([random,real[::2]])
results={}
for name,path in sources.items():
 ort=NativeAnchor(path);expected=ort(inputs)
 actor=TorchAnchor(path).cuda().eval()
 with torch.inference_mode():
  actual=actor(torch.from_numpy(inputs).cuda()).cpu().numpy()
  cpu=TorchAnchor(path)(torch.from_numpy(inputs)).numpy()
  single=actor(torch.from_numpy(inputs[:1]).cuda()).cpu().numpy()
 difference=np.abs(actual-expected);record=dict(sha256=ort.sha256,rows=len(inputs),cuda_max_abs=float(difference.max()),cpu_max_abs=float(np.abs(cpu-expected).max()),batch_vs_single=float(np.abs(actual[:1]-single).max()),passed=bool(difference.max()<1e-5),dtypes=sorted({str(x.dtype) for x in actor.buffers()}),device=torch.cuda.get_device_name())
 results[name]=record;atomic_json(out/'progress.json',results);print(name,record,flush=True)
 assert record['passed'],name
atomic_json(out/'completed.json',dict(passed=True,models=results))
