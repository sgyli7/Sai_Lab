"""Cold projected restart at the remaining Jolt stop failure; no learning."""
from pathlib import Path
import json,sys
import numpy as np
import torch
from sim2sim.research.torch_anchor import TorchAnchor
from sim2sim.research.torch_walking import rotate
from sim2sim.research.queue import atomic_json
from post_score import score_samples
sys.path.insert(0,str(Path('scripts').resolve()))
from sprint_gpu_world import GpuWorld
r=Path('results/sprint_joint_gpu_20260912');out=r/'stop_state_probe';out.mkdir(exist_ok=False);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
case_name='sprint_shift_first';seed=927008;cut=450
native=json.loads((r/'native_trained_tracking_split/suite/summary.json').read_text());e=next(e for e in native['episodes'] if (e['case'],e['seed'])==(case_name,seed));trace=json.loads(Path(e['trace']).read_text());rows=trace['rows'];case=json.loads(Path(e['case_path']).read_text())
gpu=json.loads((r/'post_native_trained_tracking_split/completed.json').read_text());j=next(i for i,e in enumerate(gpu['results']) if (e['case'],e['seed'])==(case_name,seed))
with np.load(r/'post_native_trained_tracking_split/trajectories.npz') as a:
 gs=a['states'][:,j].copy();go=a['obs'][:,j].copy();gc=a['commands'][:,j].copy();gr=a['requested'][:,j].copy();sel=a['sprint'][:,j].copy()
ns=np.array([[*x['body']['base_pos'],*x['body']['base_quat'],*x['raw']['base_angvel_local'],*x['body']['base_linvel']] for x in rows]);nr=np.array([x['requested_command'] for x in rows],np.float32);nc=np.array([x['command'] for x in rows],np.float32)
assert np.array_equal(nr,gr[:len(rows)])
settings=e['control_config']['walk'];world=GpuWorld(2,settings,feet='jolt',graphs=True);actor=TorchAnchor('results/sprint_20260912/runs/s05_native_handoff/final.onnx').cuda().eval();history_error=0.
try:
 for i in range(cut):
  pair=np.stack((ns[i],gs[i]));cmd=world.control.command(torch.tensor(np.stack((nr[i],gr[i])),device='cuda'),torch.tensor(pair[:,:3],device='cuda'),torch.tensor(pair[:,3:7],device='cuda'),torch.tensor(pair[:,10:13],device='cuda'),torch.tensor([sel[i],sel[i]],device='cuda'))
  history_error=max(history_error,float(np.max(np.abs(cmd.cpu().numpy()-np.stack((nc[i],gc[i]))))))
 assert history_error<1e-5,history_error
 snapshot=np.stack((ns[cut],gs[cut]));world.qpos[:,world.qa:world.qa+3]=torch.tensor(snapshot[:,:3],device='cuda');world.qpos[:,world.qa+3:world.qa+7]=torch.tensor(snapshot[:,3:7],device='cuda')
 world.qpos[:,world.qi]=torch.tensor(np.stack((np.array(rows[cut]['raw']['q']),world.home.cpu().numpy()+go[cut,6:20])),device='cuda',dtype=torch.float32)
 world.qvel[:,world.vi]=torch.tensor(np.stack((rows[cut]['raw']['qd'],go[cut,20:34])),device='cuda',dtype=torch.float32)
 q=world.qpos[:,world.qa+3:world.qa+7];angular=torch.tensor(snapshot[:,7:10],device='cuda',dtype=torch.float32);world.qvel[:,world.va+3:world.va+6]=angular
 velocity=torch.tensor(snapshot[:,10:13],device='cuda',dtype=torch.float32)
 world.qvel[:,world.va:world.va+3]=velocity-torch.cross(rotate(q,angular),rotate(q,world.ipos.expand(2,-1)),dim=-1)
 world.last.copy_(torch.tensor(np.stack((rows[cut]['last_action'],go[cut,34:48])),device='cuda',dtype=torch.float32));world.forward()
 samples=[np.concatenate((ns[:cut+1],np.zeros((len(rows)-cut,13)))),np.concatenate((gs[:cut+1],np.zeros((len(rows)-cut,13))))];initial_error=None
 with torch.inference_mode():
  for i in range(cut,len(rows)):
   obs,cmd=world.observe(torch.tensor(np.stack((nr[i],gr[i])),device='cuda'),torch.tensor([sel[i],sel[i]],device='cuda'))
   if i==cut:
    initial_error=np.max(np.abs(obs.cpu().numpy()-np.stack((rows[cut]['obs'],go[cut]))),axis=1).tolist();assert max(initial_error)<1e-5,initial_error
   world.step(actor(obs));pos,q,ang,v=world.state();s=torch.cat((pos,q,ang,v),dim=1).cpu().numpy()
   for k in range(2):samples[k][i+1]=s[k]
 results=[score_samples(s,case,nr.tolist(),sel) for s in samples]
 atomic_json(out/'completed.json',dict(completed=True,cut_t=9.,native_case=case_name,seed=seed,history_command_max_abs=history_error,initial_obs_max_abs=initial_error,native_original=e['task_metrics'],gpu_original=gpu['results'][j]['metrics'],jolt_projected_restart=results[0],gpu_own_cold_restart=results[1],scope='Observation/controller history preserved; generalized pose and velocity projected, contact caches and non-FK native joint compliance not transferred. No model update or deployment selection.'))
 np.savez_compressed(out/'trajectories.npz',jolt_projected=samples[0],gpu_cold=samples[1]);print('stop outcomes',[x['stops'] for x in results],flush=True)
finally:world.close()
