from pathlib import Path
import sys,json,torch,numpy as np
sys.path.insert(0,str(Path('scripts').resolve()))
from sprint_native_correct import SOURCE,TEMPLATE,PRIOR,audit_behavior
from sim2sim.research.native_sprint import prepare_sampler,collect,dataset
from sim2sim.research.models import Policy
from sim2sim.research.torch_anchor import TorchAnchor
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_target_gpu_20260912');out=r/'prefix_probe';out.mkdir(exist_ok=False)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
project=prepare_sampler(PRIOR/'native_joint_fd/suite/runtime',out/'runtime',PRIOR/'runtime_patches/target_sampler_retry/runtime/standalone/driver.gd')
seeds=[927001,927007,927008,955208]
rollouts,deployment=collect(project,PRIOR/'train_joint_fd/final.onnx',out/'rollouts',seeds,95610000,prefix_seeds=seeds,explore_after=7.,selected=['alternate'])
reference=json.loads((r/'noise_probe/completed.json').read_text());old=next(x for x in reference['rows'] if x['seed']==927001 and x['sigma']==0.);before=json.loads(Path(old['trace']).read_text())['rows'];after=json.loads(Path(rollouts['rows'][0]['trace']).read_text())['rows']
maximum={k:float(np.abs(np.array([x[k] for x in before[:350]])-np.array([x[k] for x in after[:350]])).max()) for k in ['obs','action','last_action','ctrl','command']}
entry_error=float(np.abs(np.array(before[350]['obs'])-after[350]['obs']).max());assert max(maximum.values())==0. and entry_error==0.,(maximum,entry_error)
data,audit=dataset(rollouts,deployment)
policy=Policy(SOURCE,'residual',std=.02,bound=.2,template=TEMPLATE).cuda();policy.load_state_dict(torch.load(PRIOR/'train_joint_fd/latest.pt',map_location='cuda',weights_only=False)['policy'])
flat,behavior=audit_behavior(data,policy,TorchAnchor(SOURCE).cuda().eval(),TorchAnchor(PRIOR/'train_joint_fd/final.onnx').cuda().eval())
assert audit['deterministic_prefix_sprint_rows']==1200 and audit['actor_rows']==1400
atomic_json(out/'completed.json',dict(passed=True,prefix_max_abs=maximum,exploration_entry_obs_max_abs=entry_error,data=audit,behavior=behavior,training_updates=0,scope='Live deterministic history through7s, then stochastic actions; no cold-state restore'))
print((out/'completed.json').read_text(),flush=True)
