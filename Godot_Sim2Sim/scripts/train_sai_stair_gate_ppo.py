#!/usr/bin/env python3
"""Train a continuous state/terrain gate over two feasible Sai motion priors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_sai_stair_ppo import SaiStairWorld


def load_actor(path: Path, input_size: int) -> nn.Sequential:
    model = onnx.load(str(path), load_external_data=True)
    values = {item.name: numpy_helper.to_array(item).copy() for item in model.graph.initializer}
    layers: list[nn.Module] = []
    dimensions = [input_size, 256, 192, 128, 16]
    for layer_index, onnx_index in enumerate((0, 2, 4, 6)):
        layer = nn.Linear(dimensions[layer_index], dimensions[layer_index + 1])
        layer.weight.data.copy_(torch.from_numpy(values[f"mlp.{onnx_index}.weight"]))
        layer.bias.data.copy_(torch.from_numpy(values[f"mlp.{onnx_index}.bias"]))
        layers.append(layer)
        if layer_index < 3:
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers).cuda().eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


class GateActorCritic(nn.Module):
    def __init__(self, gate_std: float):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(242, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))
        self.critic = nn.Sequential(nn.Linear(254, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.tensor(float(np.log(gate_std))))

    def forward(self, policy: torch.Tensor, critic: torch.Tensor):
        return self.gate(policy).squeeze(-1), self.critic(critic).squeeze(-1)


class DeployPolicy(nn.Module):
    def __init__(self, low: nn.Module, high: nn.Module, gate: nn.Module):
        super().__init__();self.low = low;self.high = high;self.gate = gate

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        amount = torch.sigmoid(self.gate(observation))
        return (1. - amount) * self.low(observation[:, :104]) + amount * self.high(observation)


def expert_actions(low: nn.Module, high: nn.Module, observation: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        amount = torch.sigmoid(latent)[:, None]
        return ((1. - amount) * low(observation[:, :104]) + amount * high(observation)).clamp(-1., 1.)


def pretrain_gate(env: SaiStairWorld, model: GateActorCritic, low: nn.Module, high: nn.Module,
                  steps: int, epochs: int) -> dict:
    observations, labels = [], []
    with torch.no_grad():
        for _ in range(steps):
            obs = env.get_observations()["policy"]
            label = (env.rises[env.lane] > .05).float()
            observations.append(obs.clone());labels.append(label.clone())
            latent = torch.where(label > .5, torch.full_like(label, 6.), torch.full_like(label, -6.))
            env.step(expert_actions(low, high, obs, latent))
    x, y = torch.cat(observations), torch.cat(labels)
    optimizer = torch.optim.Adam(model.gate.parameters(), lr=5e-4)
    for _ in range(epochs):
        for ids in torch.randperm(len(x), device="cuda").split(4096):
            loss = F.binary_cross_entropy_with_logits(model.gate(x[ids]).squeeze(-1), y[ids])
            optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():
        probability = torch.sigmoid(model.gate(x).squeeze(-1))
        accuracy = ((probability > .5) == (y > .5)).float().mean()
    env.reset(torch.arange(env.num_envs, device="cuda"))
    return {"samples": len(x), "epochs": epochs, "accuracy": float(accuracy),
            "initialization": "terrain labels initialize the learned gate; PPO retains continuous gate authority"}


def train(env: SaiStairWorld, model: GateActorCritic, low: nn.Module, high: nn.Module,
          iterations: int, rollout_steps: int, learning_rate: float, gate_rate_weight: float) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    clip = .2;gamma = .99;lam = .95
    last_gate=torch.zeros(env.num_envs,device="cuda")
    for iteration in range(iterations):
        policy_rows=[];critic_rows=[];latent_rows=[];logp_rows=[];value_rows=[];reward_rows=[];done_rows=[]
        for _ in range(rollout_steps):
            observation = env.get_observations();policy=observation["policy"];critic=observation["critic"]
            with torch.no_grad():
                mean,value=model(policy,critic);std=model.log_std.exp().expand_as(mean)
                distribution=torch.distributions.Normal(mean,std);latent=distribution.sample();logp=distribution.log_prob(latent)
                action=expert_actions(low,high,policy,latent)
                _,reward,done,_=env.step(action)
                gate=torch.sigmoid(latent)
                reward=reward-gate_rate_weight*(gate-last_gate).square()
                last_gate=torch.where(done,torch.zeros_like(gate),gate)
            policy_rows.append(policy);critic_rows.append(critic);latent_rows.append(latent)
            logp_rows.append(logp);value_rows.append(value);reward_rows.append(reward);done_rows.append(done)
        with torch.no_grad():
            observation=env.get_observations();_,next_value=model(observation["policy"],observation["critic"])
        rewards=torch.stack(reward_rows);dones=torch.stack(done_rows);values=torch.stack(value_rows)
        advantages=torch.zeros_like(rewards);gae=torch.zeros(env.num_envs,device="cuda")
        for step in reversed(range(rollout_steps)):
            continuation=1.-dones[step].float()
            following=next_value if step+1==rollout_steps else values[step+1]
            delta=rewards[step]+gamma*following*continuation-values[step]
            gae=delta+gamma*lam*continuation*gae;advantages[step]=gae
        returns=advantages+values
        policy=torch.cat(policy_rows);critic=torch.cat(critic_rows);latent=torch.cat(latent_rows)
        old_logp=torch.cat(logp_rows);advantage=advantages.flatten();target=returns.flatten()
        advantage=(advantage-advantage.mean())/(advantage.std()+1e-6)
        for _ in range(5):
            for ids in torch.randperm(len(policy),device="cuda").split(2048):
                mean,value=model(policy[ids],critic[ids]);std=model.log_std.exp().expand_as(mean)
                distribution=torch.distributions.Normal(mean,std);logp=distribution.log_prob(latent[ids])
                ratio=(logp-old_logp[ids]).exp();surrogate=ratio*advantage[ids]
                policy_loss=-torch.minimum(surrogate,ratio.clamp(1.-clip,1.+clip)*advantage[ids]).mean()
                value_loss=.5*(value-target[ids]).square().mean()
                loss=policy_loss+value_loss*.5-.002*distribution.entropy().mean()
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError(f"Non-finite gate parameter after iteration {iteration}")
        if (iteration + 1) % 10 == 0:
            print(json.dumps({"iteration":iteration+1,"success_events":env.successes,
                              "falls":env.falls,"cargo_losses":env.cargo_losses,
                              "gate_std":float(model.log_std.exp().detach())}),flush=True)
    return {"success_events":env.successes,"falls":env.falls,"cargo_losses":env.cargo_losses,
            "gate_std":float(model.log_std.exp().detach())}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--low",type=Path,required=True)
    parser.add_argument("--high",type=Path,required=True);parser.add_argument("--envs",type=int,default=64)
    parser.add_argument("--iterations",type=int,default=120);parser.add_argument("--steps",type=int,default=24)
    parser.add_argument("--pretrain-steps",type=int,default=120);parser.add_argument("--pretrain-epochs",type=int,default=20)
    parser.add_argument("--learning-rate",type=float,default=3e-5);parser.add_argument("--seed",type=int,default=170917)
    parser.add_argument("--initialize-gate",type=Path);parser.add_argument("--gate-std",type=float,default=.22)
    parser.add_argument("--gate-rate-weight",type=float,default=2.);parser.add_argument("--cargo-accel-weight",type=float,default=.3)
    parser.add_argument("--action-rate-weight",type=float,default=.3);parser.add_argument("--lift-event-weight",type=float,default=.5)
    parser.add_argument("--position-reward-scale",type=float,default=250.)
    parser.add_argument("--domain-randomization",action="store_true")
    args=parser.parse_args()
    if args.output.exists():raise SystemExit(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True);torch.manual_seed(args.seed);started=time.time()
    env=SaiStairWorld(args.envs,args.seed,teacher_weight_low=0.,teacher_weight_high=0.,cargo_accel_weight=args.cargo_accel_weight,
                      action_rate_weight=args.action_rate_weight,lift_event_weight=args.lift_event_weight,
                      position_objective=True,payload_clamp=True,position_reward_scale=args.position_reward_scale,
                      domain_randomization=args.domain_randomization)
    low=load_actor(args.low,104);high=load_actor(args.high,242);model=GateActorCritic(args.gate_std).cuda()
    if args.initialize_gate:
        packed=onnx.load(str(args.initialize_gate),load_external_data=True)
        values={item.name:numpy_helper.to_array(item).copy() for item in packed.graph.initializer}
        with torch.no_grad():
            for module_index,onnx_index in zip((0,2,4),(0,2,4)):
                model.gate[module_index].weight.copy_(torch.from_numpy(values[f"gate.{onnx_index}.weight"]).cuda())
                model.gate[module_index].bias.copy_(torch.from_numpy(values[f"gate.{onnx_index}.bias"]).cuda())
    try:
        warm=({"source":str(args.initialize_gate),"method":"feasible learned gate refinement"}
              if args.initialize_gate else pretrain_gate(env,model,low,high,args.pretrain_steps,args.pretrain_epochs))
        metrics=train(env,model,low,high,args.iterations,args.steps,args.learning_rate,args.gate_rate_weight)
        deploy=DeployPolicy(low,high,model.gate).cuda().eval();actor=args.output/"final.onnx"
        torch.onnx.export(deploy,torch.zeros((1,242),device="cuda"),str(actor),input_names=["obs"],output_names=["action"],opset_version=17,dynamo=False)
        packed=onnx.load(str(actor),load_external_data=True);onnx.save_model(packed,str(actor),save_as_external_data=False)
        sidecar=actor.with_name(actor.name+".data")
        if sidecar.exists():sidecar.unlink()
        profile={"schema_version":2,"id":"sai-stairs-learned-gate-v1","actor":"final.onnx",
                 "onnx_sha256":hashlib.sha256(actor.read_bytes()).hexdigest(),"observation_size":242,"action_size":16,
                 "contract":"sai-phase-free-stairs-v3","algorithm":"continuous learned gate over frozen feasible motion priors, position-task PPO with asymmetric critic",
                 "training":{"seed":args.seed,"envs":args.envs,"iterations":args.iterations,"steps":args.steps,
                             "low_sha256":hashlib.sha256(args.low.read_bytes()).hexdigest(),"high_sha256":hashlib.sha256(args.high.read_bytes()).hexdigest(),
                             "warm_start":warm,"metrics":metrics,"payload_clamp":True,
                             "domain_randomization":args.domain_randomization},
                 "control":{"speed":.16,"wheel_residual_scale":6.,"yaw_correction_limit":.4},
                 "payload_clamp_target_rad":.067/.01909859317102744,"payload_settle_seconds":8.}
        profile["training"]["comfort_constraints"]={"cargo_accel_weight":args.cargo_accel_weight,
            "action_rate_weight":args.action_rate_weight,"lift_event_weight":args.lift_event_weight,
            "gate_rate_weight":args.gate_rate_weight,"position_reward_scale":args.position_reward_scale}
        torch.save(model.state_dict(),args.output/"gate_actor_critic.pt")
        (args.output/"profile.json").write_text(json.dumps(profile,indent=2)+"\n")
        completed={"status":"trained_not_yet_accepted","elapsed_s":time.time()-started,**metrics,"onnx_sha256":profile["onnx_sha256"]}
        (args.output/"completed.json").write_text(json.dumps(completed,indent=2)+"\n");print(json.dumps(completed),flush=True)
    finally:env.close()


if __name__=="__main__":main()
