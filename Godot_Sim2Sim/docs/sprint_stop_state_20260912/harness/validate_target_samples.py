from pathlib import Path
import json,hashlib
import numpy as np
import torch
from sim2sim.research.torch_anchor import TorchAnchor
from sim2sim.research.queue import atomic_json
from sim2sim.standalone.score import score
r=Path('results/sprint_stop_state_20260912');out=r/'target_sampler_retry';x=json.loads((out/'completed.json').read_text());assert x['completed'];torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
actor=TorchAnchor(r/'train_joint_fd/final.onnx').cuda().eval();ordinary=TorchAnchor('results/sprint_20260912/runs/s05_native_handoff/final.onnx').cuda().eval();obs=[];act=[];sprint=[];explore=[];episode=[];times=[];maximum_last=0.;rejects=0
for i,e in enumerate(x['rows']):
 try:score(e['trace'],e['case'])
 except ValueError as err:
  assert 'Exploratory training rollouts' in str(err);rejects+=1
 else:raise AssertionError('Training trace accepted as gameplay evidence')
 rows=json.loads(Path(e['trace']).read_text())['rows'];last=np.zeros(14)
 for z in rows:
  maximum_last=max(maximum_last,float(np.max(np.abs(np.array(z['last_action'])-last))));last=np.array(z['action']);obs.append(z['obs']);act.append(z['action']);sprint.append(z['skill']=='sprint');explore.append(z['skill']=='sprint' and e['sigma']>0);episode.append(i);times.append(z['t'])
obs=np.array(obs,np.float32);act=np.array(act,np.float32);sprint=np.array(sprint,bool);explore=np.array(explore,bool);means=[]
with torch.inference_mode():
 for offset in range(0,len(obs),2048):
  t=torch.from_numpy(obs[offset:offset+2048]).cuda();sel=torch.from_numpy(sprint[offset:offset+2048]).cuda();means.append(torch.where(sel[:,None],actor(t),ordinary(t)).cpu().numpy())
means=np.concatenate(means);noise=act-means;std=noise[explore].std(0);mean=noise[explore].mean(0);parity=float(np.max(np.abs(noise[~explore])))
assert maximum_last==0. and parity<1e-5,(maximum_last,parity)
assert np.max(np.abs(mean))<.0015 and np.all((std>.0185)&(std<.0215)),(mean,std)
logprob=-.5*((noise/.02)**2+np.log(2*np.pi*.02**2)).sum(-1)
np.savez_compressed(out/'samples.npz',obs=obs,action=act,mean=means,old_logprob=logprob,sprint=sprint,actor_mask=explore,episode=np.array(episode),time=np.array(times))
atomic_json(r/'target_sampler_validation.json',dict(passed=True,rows=len(obs),exploratory_actor_rows=int(explore.sum()),actor_mean_device='cuda',deterministic_action_max_abs=parity,last_action_max_abs=maximum_last,noise_mean=mean.tolist(),noise_std=std.tolist(),acceptance_rejected=rejects,source_actor_sha256=actor.sha256,samples_sha256=hashlib.sha256((out/'samples.npz').read_bytes()).hexdigest(),training_updates=0,scope='Stochastic native rollout and CUDA likelihood reconstruction verified; this is infrastructure/data, no trained or accepted new policy'))
print(len(obs),int(explore.sum()),parity,std,flush=True)
