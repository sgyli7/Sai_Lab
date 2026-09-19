"""Bounded PPO with CUDA physics, frozen CUDA anchor, and native walking courses.

Split control freezes ordinary/idle S05; shared control learns every phase.
Optional reference retention anchors only the supplied successful teacher states.
Finite keyboard tasks terminate after their final stop, so their endpoint is
an MDP terminal rather than a continuing-task time-limit bootstrap.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,time,traceback
import numpy as np
import torch
from sim2sim.play_input import PlayBrain,TwistLimits
from sim2sim.standalone.sprint import templates
from sim2sim.research.models import Policy,Critic,NativeAnchor,export_policy
from sim2sim.research.torch_anchor import TorchAnchor
from sim2sim.research.torch_walking import rotate
from sim2sim.research.queue import atomic_json
from sprint_gpu_world import GpuWorld
from sim2sim.research.constraint_returns import ConstraintReturns, advantages as constrained_advantages
from sim2sim.research.sprint_constraints import SprintConstraints, yaw as constraint_yaw
from sim2sim.research.locomotion_objectives import command_tracking


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--feet',choices=['source','jolt'],required=True)
 p.add_argument('--iterations',type=int,default=128);p.add_argument('--envs',type=int,default=512);p.add_argument('--steps',type=int,default=64)
 p.add_argument('--objective',choices=['progress','tracking'],default='progress');p.add_argument('--controller',choices=['split','shared'],default='split');p.add_argument('--teacher-replay',type=Path);p.add_argument('--teacher-weight',type=float,default=.02);p.add_argument('--constraints',choices=['none','positive','cat'],default='none');p.add_argument('--cuda-graphs',action='store_true');p.add_argument('--seed',type=int,default=951101);p.add_argument('--resume',type=Path);a=p.parse_args()
 session=Path(os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR','/nonexistent'))
 if not session.exists() or not a.output.resolve().is_relative_to(session.resolve()):raise RuntimeError('Active-budget supervisor required')
 if not 1<=a.iterations<=256 or not 1<=a.envs<=512 or not 1<=a.steps<=128:raise ValueError('Unbounded training request')
 a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(a.seed)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 source=Path('results/sprint_20260912/runs/s05_native_handoff/final.onnx')
 template=Path('results/sprint_20260912/gpu_proxy/jolt_torque_512/final.onnx')
 control=Path('results/sprint_joint_20260912/delivery/control.json');settings=json.loads(control.read_text())['walk']
 policy=Policy(source,'residual',std=.02,bound=.2,template=template).cuda();policy.task_name='walking';policy.log_std.requires_grad_(False)
 anchor=TorchAnchor(source).cuda().eval();critic=Critic(template,extra_dim=10).cuda()
 retention=None
 if a.teacher_replay:
  from sim2sim.research.walking_retention import WalkingTeacherRetention
  if not np.isfinite(a.teacher_weight) or a.teacher_weight<0:raise ValueError('Invalid teacher weight')
  retention=WalkingTeacherRetention(policy,'replay_kl',a.teacher_replay,384).to('cuda')
 ao=torch.optim.Adam(policy.delta.parameters(),lr=1e-4);co=torch.optim.Adam(critic.parameters(),lr=3e-4)
 world=GpuWorld(a.envs,settings,feet=a.feet,graphs=a.cuda_graphs)
 constraints=SprintConstraints(a.envs,'cuda') if a.constraints!='none' else None
 constraint_returns=ConstraintReturns(4,'cuda') if constraints else None
 # All eight actual input programs, including released modifier and final idle.
 tapes=[];selection=[];lengths=[]
 for case in templates().values():
  brain=PlayBrain(has_standing=False,has_sprint=True,lim=TwistLimits(**settings['twist_limits']));commands=[];chosen=[]
  lengths.append(round(case['seconds']/.02))
  for i in range(1250):
   held=next(s['held'] for s in reversed(case['segments']) if s['at']<=i*.02+1e-9)
   step=brain.tick(set(held),[],.02,press_order=sorted(held));commands.append(step.command);chosen.append(step.sprint)
  tapes.append(commands);selection.append(chosen)
 tapes=torch.tensor(np.array(tapes),device='cuda',dtype=torch.float32);selection=torch.tensor(np.array(selection),device='cuda',dtype=torch.bool);lengths=torch.tensor(lengths,device='cuda')
 program=torch.zeros(a.envs,device='cuda',dtype=torch.long);phase=program.clone();all_ids=torch.arange(a.envs,device='cuda')
 lo=torch.tensor(world.model.jnt_range[world.model.actuator_trnid[:,0],0],device='cuda',dtype=torch.float32)
 hi=torch.tensor(world.model.jnt_range[world.model.actuator_trnid[:,0],1],device='cuda',dtype=torch.float32)
 def reset(ids):
  program[ids]=torch.randint(8,(len(ids),),device='cuda');phase[ids]=0
  jitter=(torch.rand((len(ids),14),device='cuda')*2-1)*.015
  jitter=(world.home+jitter).clamp(lo,hi)-world.home
  world.reset(ids,jitter)
  if constraints:constraints.reset(ids)
  yaw=(torch.rand(len(ids),device='cuda')*2-1)*np.pi
  world.qpos[ids,world.qa+3]=torch.cos(yaw/2);world.qpos[ids,world.qa+6]=torch.sin(yaw/2)
 def state_features(obs,sprint):
  pos,q,ang,vel=world.state();w,x,y,z=q.unbind(-1);yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
  cy,sy=yaw.cos(),yaw.sin();local=torch.stack((cy*vel[:,0]+sy*vel[:,1],-sy*vel[:,0]+cy*vel[:,1],vel[:,2]),dim=1)
  diff=world.control.yaw_target-yaw;heading=torch.atan2(diff.sin(),diff.cos()).float()
  displacement=pos[:,:2].double()-world.control.origin;path=-world.control.path_yaw.sin()*displacement[:,0]+world.control.path_yaw.cos()*displacement[:,1]
  fraction=phase.float()/lengths[program]
  extra=torch.cat((local,pos[:,2:3]/.125,heading[:,None],path.float()[:,None]/.05,sprint.float()[:,None],fraction[:,None],((lengths[program]-phase).float()/1250)[:,None],(program.float()/7)[:,None]),dim=1)
  return torch.cat((obs,extra),dim=1)
 def observe():
  requested=tapes[program,phase];sprint=selection[program,phase];obs,command=world.observe(requested,sprint)
  return obs,state_features(obs,sprint),requested,command,sprint
 def reward(requested,command,action,previous):
  pos,q,ang,vel=world.state();w,x,y,z=q.unbind(-1)
  yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z));tilt=torch.acos((1-2*(x*x+y*y)).clamp(-1,1))
  moving=requested[:,0]>.01;straight=moving&(requested[:,2].abs()<.05)&(requested[:,1].abs()<.01)
  local_forward=yaw.cos()*vel[:,0]+yaw.sin()*vel[:,1]
  path_forward=world.control.path_yaw.cos()*vel[:,0]+world.control.path_yaw.sin()*vel[:,1]
  speed=torch.where(straight,path_forward.float(),local_forward)
  delta=pos[:,:2].double()-world.control.origin;cross=-world.control.path_yaw.sin()*delta[:,0]+world.control.path_yaw.cos()*delta[:,1]
  difference=world.control.yaw_target-yaw;heading=torch.atan2(difference.sin(),difference.cos())
  yaw_rate=rotate(q,ang)[:,2];turn=requested[:,2].abs()>.05
  progression=torch.where(moving,6*speed.clamp(-.1,.6),1.-10.*vel[:,:2].square().sum(-1))
  tracking=torch.where(straight,20*cross.square()+2*heading.square(),0.).float()+torch.where(turn,.5*(yaw_rate-requested[:,2]).square(),0.)
  if a.objective=='tracking':
   local=torch.stack((local_forward,-yaw.sin()*vel[:,0]+yaw.cos()*vel[:,1],vel[:,2]),dim=-1)
   progression=command_tracking(command,local,yaw_rate)
   tracking=torch.where(straight,20*cross.square()+2*heading.square(),0.).float()
  balance=8*torch.relu(tilt-.12).square();smooth=.05*(action-previous).square().sum(-1)
  fell=(pos[:,2]<.055)|(tilt>np.pi/3)
  value=progression-tracking-balance-smooth-10*fell.float()
  return value,fell,dict(progress=progression.mean(),tracking=tracking.mean(),balance=balance.mean(),smooth=smooth.mean())
 files=[Path(__file__),Path('scripts/sprint_gpu_world.py'),Path('scripts/sprint_gpu_proxy.py'),Path('src/sim2sim/research/torch_anchor.py'),Path('src/sim2sim/research/torch_walking.py'),Path('src/sim2sim/research/models.py'),Path('src/sim2sim/research/constraint_returns.py'),Path('src/sim2sim/research/sprint_constraints.py'),Path('src/sim2sim/research/walking_retention.py'),Path('src/sim2sim/research/locomotion_objectives.py')]
 config=dict(source=str(source),template=str(template),template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),source_sha256=anchor.sha256,controller=a.controller,teacher_retention=retention.audit if retention else None,teacher_weight=a.teacher_weight if retention else 0.,constraints=a.constraints,constraint_probability_schedule='0.05 to0.25 over128 iterations, EMA .95' if a.constraints=='cat' else 'zero',variant='residual',std=.02,bound=.2,feet=a.feet,cuda_graphs=a.cuda_graphs,iterations=a.iterations,envs=a.envs,steps=a.steps,seed=a.seed,control=str(control),control_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),code_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},device=torch.cuda.get_device_name(),collection_device='cuda',learner_device='cuda',ordinary_controller='shared learned actor' if a.controller=='shared' else 'frozen S05',actor_mask='all phases' if a.controller=='shared' else 'actual sprint selection only',reward='command_tracking_cat_table1_v1' if a.objective=='tracking' else 'native_progress_v1',gamma=.99,lam=.95,epochs=4,minibatch=4096,actor_lr=1e-4,critic_lr=3e-4,target_kl=.008,critic_warmup=2,reset='uniform yaw and joint noise +/- .015 rad, clipped joint limits',task_termination='finite keyboard episode ends after final stop; no continuing-task bootstrap',game_physics_changed=False,foot_hulls=world.feet,inertia=world.inertia,torch=torch.__version__,cuda=torch.version.cuda,started_unix=time.time())
 atomic_json(a.output/'config.json',config)
 initial_iteration=0
 if a.resume:
  old=torch.load(a.resume,map_location='cuda',weights_only=False)
  for key in ['source_sha256','template_sha256','variant','std','bound','feet','cuda_graphs','envs','steps','seed','control_sha256','code_sha256','reward','constraints','controller','teacher_retention','teacher_weight']:
   if config[key]!=old['config'][key]:raise ValueError('Resume changed '+key)
  policy.load_state_dict(old['policy']);critic.load_state_dict(old['critic']);ao.load_state_dict(old['actor_optimizer']);co.load_state_dict(old['critic_optimizer']);initial_iteration=old['iteration']
  torch.set_rng_state(old['rng_cpu'].cpu());torch.cuda.set_rng_state(old['rng_cuda'].cpu())
  if constraint_returns:constraint_returns.scale.copy_(old['constraint_scale'])
 reset(all_ids)
 def checkpoint(iteration):
  target=a.output/'latest.pt';tmp=target.with_suffix('.partial')
  torch.save(dict(policy=policy.state_dict(),critic=critic.state_dict(),actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),iteration=iteration,config=config,factory_sha256=anchor.sha256,constraint_scale=constraint_returns.scale if constraint_returns else None,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),resume_note='New physical episodes from complete iteration; preserve this failed attempt'),tmp);tmp.replace(target)
 export_policy(policy,a.output/'initial.onnx')
 start=time.monotonic();records=[];falls_total=0;completed_total=0
 try:
  with torch.no_grad():obs,cobs,requested,command,sprint=observe()
  for iteration in range(initial_iteration+1,a.iterations+1):
   torch.cuda.synchronize();collect_start=time.monotonic();buffer={k:[] for k in ['obs','cobs','anchor','action','mean','value','logprob','reward','done','mask']};terms=[];violations=[]
   with torch.no_grad():
    for step in range(a.steps):
     base=anchor(obs);dist=policy.distribution(obs,base);active=torch.ones_like(sprint) if a.controller=='shared' else sprint;action=torch.where(active[:,None],dist.sample(),base);value=critic(cobs)
     previous=world.last.clone()
     if constraints:
      old_pos,old_q,_,_=world.state();old_pos=old_pos.clone();old_yaw=constraint_yaw(old_q)
     world.step(action)
     if constraints:
      pos,q,_,v=world.state();violations.append(constraints.observe(requested,old_pos,old_yaw,pos,q,v))
     rew,fell,detail=reward(requested,command,action,previous);terms.append(torch.stack(list(detail.values())))
     phase+=1;finished=phase>=lengths[program];done=fell|finished
     for k,v in [('obs',obs),('cobs',cobs),('anchor',base),('action',action),('mean',dist.mean),('value',value),('logprob',dist.log_prob(action).sum(-1)),('reward',rew),('done',done),('mask',active)]:buffer[k].append(v.clone())
     falls_total+=int(fell.sum());completed_total+=int(finished.sum())
     ids=done.nonzero(as_tuple=False).flatten()
     if len(ids):reset(ids)
     obs,cobs,requested,command,sprint=observe()
    b={k:torch.stack(v) for k,v in buffer.items()};next_value=critic(cobs);gae=torch.zeros(a.envs,device='cuda');adv=torch.zeros_like(b['reward'])
    if constraint_returns:
     cost=torch.stack(violations);probability=constraint_returns.probability(cost,(.05+.2*min(1.,iteration/128)) if a.constraints=='cat' else 0.)
     adv,return_values=constrained_advantages(b['reward'],b['value'],next_value,b['done'],probability)
     returns=return_values.flatten()
    else:
     for step in reversed(range(a.steps)):
      alive=(~b['done'][step]).float();delta=b['reward'][step]+.99*next_value*alive-b['value'][step];gae=delta+.99*.95*alive*gae;adv[step]=gae;next_value=b['value'][step]
     returns=(adv+b['value']).flatten()
    masked=adv[b['mask']]
    adv=((adv-masked.mean())/(masked.std()+1e-8) if masked.numel()>1 else torch.zeros_like(adv)).flatten()
    flat={k:v.flatten(0,1) for k,v in b.items()}
   torch.cuda.synchronize();collection_s=time.monotonic()-collect_start;learn_start=time.monotonic();before=copy.deepcopy(policy.state_dict());before_optimizer=copy.deepcopy(ao.state_dict());updates=0;stop=False;teacher_losses=[]
   for epoch in range(4):
    for ix in torch.randperm(a.steps*a.envs,device='cuda').split(4096):
     closs=(critic(flat['cobs'][ix])-returns[ix]).square().mean()
     if not torch.isfinite(closs):raise FloatingPointError('Nonfinite critic loss')
     co.zero_grad();closs.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);co.step()
     if iteration<=2 or stop:continue
     ix=ix[flat['mask'][ix]]
     if not len(ix):continue
     dist=policy.distribution(flat['obs'][ix],flat['anchor'][ix])
     kl=((dist.mean-flat['mean'][ix]).square()/(2*.02**2)).sum(-1).mean()
     if float(kl.detach())>.016:stop=True;continue
     ratio=(dist.log_prob(flat['action'][ix]).sum(-1)-flat['logprob'][ix]).clamp(-20,20).exp()
     loss=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean()+.02*policy.delta(flat['obs'][ix]).square().mean()
     if retention:
      teacher_loss=retention.loss(policy,flat['obs']);teacher_losses.append(float(teacher_loss.detach()));loss=loss+a.teacher_weight*teacher_loss
     if not torch.isfinite(loss):raise FloatingPointError('Nonfinite actor loss')
     ao.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.delta.parameters(),.5);ao.step();updates+=1
   with torch.no_grad():
    ix=flat['mask'].nonzero(as_tuple=False).flatten()[::max(1,int(flat['mask'].sum())//4096)]
    actual_kl=float(((policy(flat['obs'][ix],flat['anchor'][ix])-flat['mean'][ix]).square()/(2*.02**2)).sum(-1).mean()) if len(ix) else 0.
   rejected=not np.isfinite(actual_kl) or actual_kl>.15
   if rejected:
    policy.load_state_dict(before);ao.load_state_dict(before_optimizer)
    for g in ao.param_groups:g['lr']*=.5
   elif actual_kl>.012:
    for g in ao.param_groups:g['lr']=max(1e-6,g['lr']*.8)
   torch.cuda.synchronize();learning_s=time.monotonic()-learn_start
   rec=dict(iteration=iteration,samples=iteration*a.steps*a.envs,collection_s=collection_s,learning_s=learning_s,reward=float(b['reward'].mean()),actor_samples=int(b['mask'].sum()),actor_updates=updates,kl=actual_kl,rejected=rejected,actor_lr=ao.param_groups[0]['lr'],falls=falls_total,completed_tasks=completed_total,terms=torch.stack(terms).mean(0).tolist(),elapsed_s=time.monotonic()-start)
   if retention:rec['teacher_kl']=sum(teacher_losses)/max(1,len(teacher_losses))
   if constraint_returns:rec.update(constraint_fraction=(cost>0).float().mean((0,1)).tolist(),constraint_probability=float(probability.mean()),constraint_scale=constraint_returns.scale.tolist())
   records.append(rec)
   with (a.output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(rec)+'\n')
   if iteration%8==0 or iteration==a.iterations:atomic_json(a.output/'progress.json',rec);print(rec,flush=True)
   if iteration%16==0 or iteration==a.iterations:checkpoint(iteration)
  export_policy(policy,a.output/'final.onnx')
  inputs=np.concatenate([np.random.default_rng(a.seed+1).normal(0,.5,(256,61)).astype(np.float32),flat['obs'][::max(1,len(flat['obs'])//256)].detach().cpu().numpy()[:256]])
  reference=NativeAnchor(a.output/'final.onnx');expected=reference(inputs)
  with torch.no_grad():x=torch.tensor(inputs,device='cuda');actual=policy(x,anchor(x)).cpu().numpy()
  error=float(np.abs(actual-expected).max());atomic_json(a.output/'parity.json',dict(passed=error<1e-5,max_abs=error,rows=len(inputs)))
  if error>=1e-5:raise RuntimeError('Learned GPU actor differs from exported ONNX')
  atomic_json(a.output/'completed.json',dict(status='completed',iterations=a.iterations,samples=a.iterations*a.steps*a.envs,collection_device='cuda',learner_device='cuda',collection_s=sum(x['collection_s'] for x in records),learning_s=sum(x['learning_s'] for x in records),elapsed_s=time.monotonic()-start,falls=falls_total,completed_tasks=completed_total,parity_max_abs=error,final_sha256=reference.sha256,accepted=False,acceptance='Requires independent native Jolt evaluation'))
 except BaseException:
  (a.output/'error.txt').write_text(traceback.format_exc());raise
 finally:world.close()


if __name__=='__main__':main()
