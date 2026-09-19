"""Batch GPU learning with a synchronized CPU policy for Jolt collection.

The frozen ORT anchor stays on CPU. It is evaluated during collection and its
exact outputs are included in the batch, so SGD never reimplements that graph.
"""
import copy
import torch


class LearnerDevice:
    def __init__(self, policy, critic, requested='auto'):
        if requested not in ('auto','cpu','cuda'):raise ValueError('Unknown learner device')
        selected=('cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else requested
        if selected=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA learning requested but CUDA is unavailable')
        self.device=torch.device(selected)
        self.policy=policy;self.critic=critic
        if selected=='cuda':
            # Preserve declared FP64 actor / FP32 critic arithmetic.
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            # ORT sessions cannot be deep-copied; this immutable anchor is shared.
            self.collect_policy=copy.deepcopy(policy,{id(policy.anchor):policy.anchor})
            self.collect_critic=copy.deepcopy(critic)
            policy.to(self.device);critic.to(self.device)
        else:
            self.collect_policy=policy;self.collect_critic=critic

    def sync_collectors(self):
        if self.device.type=='cpu':return
        for source,target in [(self.policy,self.collect_policy),(self.critic,self.collect_critic)]:
            target.load_state_dict({key:value.detach().cpu() for key,value in source.state_dict().items()})

    def batch(self, values):
        return {key:value.to(self.device) for key,value in values.items()}

    def synchronize(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
