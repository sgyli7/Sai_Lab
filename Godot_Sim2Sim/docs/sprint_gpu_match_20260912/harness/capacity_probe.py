from pathlib import Path
import json,sys,time
import numpy as np
import torch
import warp as wp
import mujoco_warp as mw
from sim2sim.research.queue import atomic_json
sys.path.insert(0,str(Path('scripts').resolve()))
from sprint_gpu_world import GpuWorld
r=Path('results/sprint_gpu_match_20260912');out=r/'capacity_probe';out.mkdir(exist_ok=False)
settings=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text())['walk'];torch.set_num_threads(4)
snap=np.load('results/sprint_exit_20260912/gpu_dynamics_probe/states.npz');sel=np.load('results/sprint_exit_20260912/gpu_dynamics_probe/torque_only.npz')['selected'];ids=np.resize(sel,512)
original=mw.put_data;records={};outputs={}
for cap in [2048,512,256]:
 def put_data(*args,**kwargs):kwargs['njmax']=cap;return original(*args,**kwargs)
 mw.put_data=put_data
 world=GpuWorld(512,settings,feet='source');mw.put_data=original
 try:
  qp=torch.tensor(snap['qpos'][ids],device='cuda',dtype=world.qpos.dtype);qv=torch.tensor(snap['qvel'][ids],device='cuda',dtype=world.qvel.dtype);target=torch.tensor(snap['target'][ids],device='cuda',dtype=torch.float32)
  def reset():
   world.qpos.copy_(qp);world.qvel.copy_(qv);wp.to_torch(world.wd.qacc_warmstart).zero_()
   with wp.ScopedStream(world.stream):mw.forward(world.wm,world.wd)
  reset();world.step(target-world.home);torch.cuda.synchronize();reset()
  nefc=torch.zeros((),device='cuda',dtype=torch.int32);nacon=nefc.clone();torch.cuda.synchronize();start=time.monotonic()
  with torch.no_grad(),wp.ScopedStream(world.stream):
   for step in range(80):
    v=world.qvel[:,world.vi];world.ctrl.copy_((.55*(target-world.qpos[:,world.qi])).clamp(-world.limit,world.limit)-.053*v-.0048*torch.tanh(v/.05));mw.step(world.wm,world.wd)
    nefc=torch.maximum(nefc,wp.to_torch(world.wd.nefc).max());nacon=torch.maximum(nacon,wp.to_torch(world.wd.nacon).max())
  torch.cuda.synchronize();elapsed=time.monotonic()-start
  result=dict(njmax=cap,worlds=512,physical_steps=80,elapsed_s=elapsed,max_nefc=int(nefc),max_nacon=int(nacon),naconmax=int(world.wd.naconmax),overflow=bool(nefc>=cap or nacon>=world.wd.naconmax))
  outputs[cap]=(world.qpos.cpu().numpy().copy(),world.qvel.cpu().numpy().copy());np.savez_compressed(out/(str(cap)+'.npz'),qpos=outputs[cap][0],qvel=outputs[cap][1])
  if cap!=2048:result.update(qpos_max_abs_vs_2048=float(np.abs(outputs[cap][0]-outputs[2048][0]).max()),qvel_max_abs_vs_2048=float(np.abs(outputs[cap][1]-outputs[2048][1]).max()))
  records[str(cap)]=result;atomic_json(out/'progress.json',records);print(result,flush=True)
 finally:world.close();mw.put_data=original
atomic_json(out/'completed.json',dict(status='completed',results=records,scope='Allocation capacity only; same model/timestep/motor equation/initial states/held targets. Includes near-fall snapshots and checks every substep for overflow.'))
