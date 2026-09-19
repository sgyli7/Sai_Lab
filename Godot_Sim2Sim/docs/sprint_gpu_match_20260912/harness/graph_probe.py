from pathlib import Path
import json,sys,time
import numpy as np
import torch
import warp as wp
import mujoco_warp as mw
from sim2sim.research.queue import atomic_json
sys.path.insert(0,str(Path('scripts').resolve()))
from sprint_gpu_world import GpuWorld
r=Path('results/sprint_gpu_match_20260912');out=r/'graph_probe';out.mkdir(exist_ok=False)
settings=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text())['walk'];torch.set_num_threads(4)
snap=np.load('results/sprint_exit_20260912/gpu_dynamics_probe/states.npz');sel=np.load('results/sprint_exit_20260912/gpu_dynamics_probe/torque_only.npz')['selected'];ids=np.resize(sel,512)
records={};outputs={}
for graphs in [False,True]:
 world=GpuWorld(512,settings,feet='jolt',graphs=graphs)
 try:
  qp=torch.tensor(snap['qpos'][ids],device='cuda',dtype=world.qpos.dtype);qv=torch.tensor(snap['qvel'][ids],device='cuda',dtype=world.qvel.dtype);action=torch.tensor(snap['target'][ids],device='cuda',dtype=torch.float32)-world.home
  def reset():
   world.qpos.copy_(qp);world.qvel.copy_(qv);wp.to_torch(world.wd.qacc_warmstart).zero_()
   with wp.ScopedStream(world.stream):
    if world.forward_graph is None:mw.forward(world.wm,world.wd)
    else:wp.capture_launch(world.forward_graph)
  reset();world.step(action);torch.cuda.synchronize();reset()
  torch.cuda.synchronize();start=time.monotonic();nefc=0;nacon=0
  for step in range(20):
   world.step(action);nefc=max(nefc,int(wp.to_torch(world.wd.nefc).max()));nacon=max(nacon,int(wp.to_torch(world.wd.nacon).max()))
   if step==0:one=(world.qpos.cpu().numpy().copy(),world.qvel.cpu().numpy().copy())
  torch.cuda.synchronize();elapsed=time.monotonic()-start
  end=(world.qpos.cpu().numpy().copy(),world.qvel.cpu().numpy().copy());outputs[graphs]=(one,end)
  result=dict(graphs=graphs,worlds=512,decision_steps=20,elapsed_s=elapsed,max_nefc=nefc,max_nacon=nacon,overflow=nefc>=2048 or nacon>=world.wd.naconmax)
  if graphs:
   for name,i in [('20ms',0),('400ms',1)]:
    result[name]=dict(qpos_max_abs=float(np.abs(outputs[True][i][0]-outputs[False][i][0]).max()),qvel_max_abs=float(np.abs(outputs[True][i][1]-outputs[False][i][1]).max()))
  records[str(graphs)]=result;atomic_json(out/'progress.json',records);np.savez_compressed(out/(str(graphs)+'.npz'),one_qpos=one[0],one_qvel=one[1],end_qpos=end[0],end_qvel=end[1]);print(result,flush=True)
 finally:world.close()
atomic_json(out/'completed.json',dict(status='completed',results=records,speedup=records['False']['elapsed_s']/records['True']['elapsed_s'],scope='Same exact game foot hulls, 2048 constraint capacity and motor equation; only raw launch versus captured graph changes. Close-to-fall snapshots included.'))
