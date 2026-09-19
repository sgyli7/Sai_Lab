"""Bounded target-engine PPO correction; native Jolt episodes, CUDA learning."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import time

import numpy as np
import torch

from sim2sim.research.native_sprint import prepare_sampler, collect, dataset
from sim2sim.research.models import Policy, Critic, NativeAnchor, export_policy
from sim2sim.research.torch_anchor import TorchAnchor
from sim2sim.research.walking_retention import WalkingTeacherRetention
from sim2sim.research.constraint_returns import ConstraintReturns, advantages
from sim2sim.research.queue import atomic_json


SOURCE = Path('results/sprint_20260912/runs/s05_native_handoff/final.onnx')
TEMPLATE = Path('results/sprint_20260912/gpu_proxy/jolt_torque_512/final.onnx')
PRIOR = Path('results/sprint_stop_state_20260912')
TEACHER = Path('results/sprint_joint_gpu_20260912/teacher/manifest.json')


def audit_behavior(data, policy, anchor, exported):
    flat = {k:v.flatten(0,1) for k,v in data.items()}
    means, bases = [], []
    with torch.no_grad():
        for obs in flat['obs'].split(2048):
            base = anchor(obs)
            bases.append(base)
            means.append(exported(obs))
        base = torch.cat(bases)
        mean = torch.cat(means)
        actual = torch.cat([policy(o,b) for o,b in zip(flat['obs'].split(2048),base.split(2048))])
        parity = float((actual-mean)[flat['valid']].abs().max())
        ordinary = flat['valid'] & ~flat['sprint']
        ordinary_error = float((flat['action']-base)[ordinary].abs().max())
        prefix = flat['valid'] & flat['sprint'] & ~flat['mask']
        prefix_error = float((flat['action']-mean)[prefix].abs().max()) if prefix.any() else 0.
        noise = (flat['action']-mean)[flat['mask']]
        noise_mean, noise_std = noise.mean(0), noise.std(0)
        assert parity < 1e-5 and ordinary_error < 1e-5 and prefix_error < 1e-5, (parity,ordinary_error,prefix_error)
        assert noise_mean.abs().max() < .002 and ((noise_std>.018)&(noise_std<.022)).all()
        flat['anchor'] = base
        flat['mean'] = mean
        flat['logprob'] = torch.distributions.Normal(mean,.02).log_prob(flat['action']).sum(-1)
    return flat,dict(policy_export_max_abs=parity,ordinary_max_abs=ordinary_error,deterministic_prefix_max_abs=prefix_error,
                     noise_mean=noise_mean.tolist(),noise_std=noise_std.tolist())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=12)
    p.add_argument('--seeds-per-batch',type=int,default=4)
    p.add_argument('--seed',type=int,default=955101)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--curriculum',choices=['uniform','warm_prefix'],default='uniform')
    a=p.parse_args()
    session=Path(os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR','/nonexistent'))
    if not session.exists() or not a.output.resolve().is_relative_to(session.resolve()):
        raise RuntimeError('Active-budget supervisor required')
    if not 1<=a.iterations<=24 or not 1<=a.seeds_per_batch<=8:
        raise ValueError('Unbounded native correction')
    if not torch.cuda.is_available():raise RuntimeError('CUDA learning required; CPU fallback disabled')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    policy=Policy(SOURCE,'residual',std=.02,bound=.2,template=TEMPLATE).cuda()
    policy.task_name='walking';policy.log_std.requires_grad_(False)
    anchor=TorchAnchor(SOURCE).cuda().eval()
    critic=Critic(TEMPLATE,extra_dim=10).cuda()
    checkpoint=torch.load(PRIOR/'train_joint_fd/latest.pt',map_location='cuda',weights_only=False)
    policy.load_state_dict(checkpoint['policy']);critic.load_state_dict(checkpoint['critic'])
    retention=WalkingTeacherRetention(policy,'replay_kl',TEACHER,384).to('cuda')
    constraints=ConstraintReturns(4,'cuda');constraints.scale.copy_(checkpoint['constraint_scale'])
    # Reset optimizer moments for the new target distribution; weights are intact.
    ao=torch.optim.Adam(policy.delta.parameters(),lr=1e-4)
    co=torch.optim.Adam(critic.parameters(),lr=3e-4)
    code=[Path(__file__),Path('src/sim2sim/research/native_sprint.py'),
          Path('src/sim2sim/research/models.py'),Path('src/sim2sim/research/constraint_returns.py'),
          Path('src/sim2sim/research/sprint_constraints.py'),Path('src/sim2sim/research/torch_walking.py')]
    if a.curriculum=='warm_prefix' and a.seeds_per_batch!=4:
        raise ValueError('Warm-prefix curriculum requires exactly four reset seeds per batch')
    config=dict(seed=a.seed,iterations=a.iterations,seeds_per_batch=a.seeds_per_batch,curriculum=a.curriculum,
                initial_checkpoint=str(PRIOR/'train_joint_fd/latest.pt'),
                initial_actor_sha256=hashlib.sha256((PRIOR/'train_joint_fd/final.onnx').read_bytes()).hexdigest(),
                collection_device='cpu_native_jolt',learner_device='cuda',workers=4,
                actor_lr=1e-4,critic_lr=3e-4,std=.02,epochs=4,minibatch=4096,gamma=.99,lam=.95,
                critic_warmup=2,teacher_weight=.02,teacher=retention.audit,
                physics_hz=200,policy_hz=50,physics_changed=False,control_changed=False,
                ordinary_controller='frozen S05',actor_mask='stochastic sprint transitions before first fall; deterministic prefixes excluded',
                seed_range_note='Fresh resets 955200 + iteration*8. warm_prefix mixes 927001 and third fresh seed with exploration after 7s. No final seeds.',
                constraint_maximum=.25,checkpoint_initial_config=checkpoint['config'],
                code_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in code})
    atomic_json(a.output/'config.json',config)
    if a.audit_only:
        rollouts=json.loads((PRIOR/'target_sampler_retry/completed.json').read_text())
        deployment=json.loads((PRIOR/'native_joint_fd/suite/runtime/runtime_assets/deployment.json').read_text())
        data,audit=dataset(rollouts,deployment)
        flat,behavior=audit_behavior(data,policy,anchor,TorchAnchor(PRIOR/'train_joint_fd/final.onnx').cuda().eval())
        with torch.no_grad():
            values=critic(flat['cobs']).reshape(data['reward'].shape)
            prob=constraints.probability(data['cost'],.25)
            adv,returns=advantages(data['reward'],values,torch.zeros(values.shape[1],device='cuda'),data['done'],prob)
        assert torch.isfinite(adv).all() and torch.isfinite(returns).all()
        atomic_json(a.output/'completed.json',dict(passed=True,data=audit,behavior=behavior,
                    values_mean=float(values[data['valid']].mean()),returns_mean=float(returns[data['valid']].mean()),
                    shape=list(values.shape),training_updates=0))
        np.savez_compressed(a.output/'reconstructed.npz',**{k:v.cpu().numpy() for k,v in data.items()})
        print(audit,behavior,flush=True)
        return
    project=prepare_sampler(PRIOR/'native_joint_fd/suite/runtime',a.output/'runtime',
                            PRIOR/'runtime_patches/target_sampler_retry/runtime/standalone/driver.gd')
    export_policy(policy,a.output/'initial.onnx')
    actor=a.output/'initial.onnx'
    start=time.monotonic();records=[];total_samples=0
    for iteration in range(1,a.iterations+1):
        out=a.output/f'iteration_{iteration:03d}';out.mkdir()
        collection_start=time.monotonic()
        seeds=list(range(955200+iteration*8,955200+iteration*8+a.seeds_per_batch))
        prefix_seeds=[]
        if a.curriculum=='warm_prefix':
            seeds[-1]=927001
            prefix_seeds=seeds[2:]
        rollouts,deployment=collect(project,actor,out/'rollouts',seeds,95520000+iteration*100,
                                    prefix_seeds=prefix_seeds,explore_after=7.)
        collection_s=time.monotonic()-collection_start
        reconstruct_start=time.monotonic()
        data,audit=dataset(rollouts,deployment)
        flat,behavior=audit_behavior(data,policy,anchor,TorchAnchor(actor).cuda().eval())
        atomic_json(out/'data_audit.json',dict(data=audit,behavior=behavior))
        with torch.no_grad():
            values=torch.cat([critic(x) for x in flat['cobs'].split(4096)]).reshape(data['reward'].shape)
            probability=constraints.probability(data['cost'],.25)
            adv,returns=advantages(data['reward'],values,torch.zeros(values.shape[1],device='cuda'),data['done'],probability)
            selected=adv[data['mask']]
            adv=((adv-selected.mean())/(selected.std()+1e-8)).flatten()
            returns=returns.flatten()
        torch.cuda.synchronize();reconstruct_s=time.monotonic()-reconstruct_start
        learning_start=time.monotonic();before=copy.deepcopy(policy.state_dict());before_optimizer=copy.deepcopy(ao.state_dict())
        indices=flat['valid'].nonzero(as_tuple=False).flatten()
        updates=0;stop=False;teacher_losses=[];critic_losses=[]
        for epoch in range(4):
            for ix in indices[torch.randperm(len(indices),device='cuda')].split(4096):
                closs=(critic(flat['cobs'][ix])-returns[ix]).square().mean()
                if not torch.isfinite(closs):raise FloatingPointError('Nonfinite critic loss')
                co.zero_grad();closs.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);co.step()
                critic_losses.append(float(closs.detach()))
                if iteration<=2 or stop:continue
                ix=ix[flat['mask'][ix]]
                if not len(ix):continue
                dist=policy.distribution(flat['obs'][ix],flat['anchor'][ix])
                kl=((dist.mean-flat['mean'][ix]).square()/(2*.02**2)).sum(-1).mean()
                if float(kl.detach())>.016:stop=True;continue
                ratio=(dist.log_prob(flat['action'][ix]).sum(-1)-flat['logprob'][ix]).clamp(-20,20).exp()
                loss=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean()+.02*policy.delta(flat['obs'][ix]).square().mean()
                teacher_loss=retention.loss(policy,flat['obs'][indices]);teacher_losses.append(float(teacher_loss.detach()))
                loss=loss+.02*teacher_loss
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite actor loss')
                ao.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.delta.parameters(),.5);ao.step();updates+=1
        with torch.no_grad():
            ix=flat['mask'].nonzero(as_tuple=False).flatten()
            actual=torch.cat([policy(flat['obs'][i],flat['anchor'][i]) for i in ix.split(4096)])
            actual_kl=float(((actual-flat['mean'][ix]).square()/(2*.02**2)).sum(-1).mean())
        rejected=not np.isfinite(actual_kl) or actual_kl>.15
        if rejected:
            policy.load_state_dict(before);ao.load_state_dict(before_optimizer)
            for group in ao.param_groups:group['lr']*=.5
        elif actual_kl>.012:
            for group in ao.param_groups:group['lr']=max(1e-6,group['lr']*.8)
        torch.cuda.synchronize();learning_s=time.monotonic()-learning_start
        actor=out/'actor.onnx';export_policy(policy,actor)
        probe=flat['obs'][indices[::max(1,len(indices)//256)]].cpu().numpy()[:256]
        probe=np.concatenate((probe,np.random.default_rng(a.seed+iteration).normal(0,.5,(256,61)).astype(np.float32)))
        expected=NativeAnchor(actor)(probe)
        with torch.no_grad():x=torch.as_tensor(probe,device='cuda');actual=policy(x,anchor(x)).cpu().numpy()
        parity=float(np.abs(expected-actual).max())
        if parity>=1e-5:raise RuntimeError('Export parity failed: '+str(parity))
        total_samples+=audit['rows']
        record=dict(iteration=iteration,samples=total_samples,data=audit,collection_s=collection_s,
                    reconstruction_s=reconstruct_s,learning_s=learning_s,actor_updates=updates,kl=actual_kl,
                    rejected=rejected,actor_lr=ao.param_groups[0]['lr'],parity_max_abs=parity,
                    reward=float(data['reward'][data['valid']].mean()),critic_loss=float(np.mean(critic_losses)),
                    teacher_kl=float(np.mean(teacher_losses)) if teacher_losses else None,
                    constraint_probability=float(probability[data['valid']].mean()),elapsed_s=time.monotonic()-start)
        records.append(record);atomic_json(out/'completed.json',record)
        atomic_json(a.output/'progress.json',record)
        print(record,flush=True)
        tmp=a.output/'latest.partial'
        torch.save(dict(policy=policy.state_dict(),critic=critic.state_dict(),actor_optimizer=ao.state_dict(),
                        critic_optimizer=co.state_dict(),constraint_scale=constraints.scale,iteration=iteration,
                        config=config,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),
                        resume_note='New full native episodes after complete iteration only; preserve failed attempts'),tmp)
        tmp.replace(a.output/'latest.pt')
    export_policy(policy,a.output/'final.onnx')
    atomic_json(a.output/'completed.json',dict(completed=True,iterations=a.iterations,samples=total_samples,
                collection_device='cpu_native_jolt',learner_device='cuda',elapsed_s=time.monotonic()-start,
                collection_s=sum(x['collection_s'] for x in records),reconstruction_s=sum(x['reconstruction_s'] for x in records),
                learning_s=sum(x['learning_s'] for x in records),max_parity=max(x['parity_max_abs'] for x in records),
                final_sha256=hashlib.sha256((a.output/'final.onnx').read_bytes()).hexdigest(),accepted=False))


if __name__=='__main__':main()
