"""Keep a fast walking teacher on balanced, successful state occupancy."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def balanced_sample(observations, count):
    idle=observations[:,48].abs()<.01
    turn=(~idle)&(observations[:,50].abs()>.25)
    sprint=(~idle)&(~turn)&(observations[:,48]>=.275)
    normal=(~idle)&(~turn)&(~sprint)
    pools=[observations[mask] for mask in [idle,normal,sprint,turn] if mask.any()]
    if not pools:return observations[:0]
    sizes=[count//len(pools)+(i<count%len(pools)) for i in range(len(pools))]
    return torch.cat([pool[torch.randint(len(pool),(size,),device=pool.device)] for pool,size in zip(pools,sizes)])


class WalkingTeacherRetention:
    def __init__(self,policy,mode,replay=None,samples=384):
        if mode not in ('online_kl','replay_kl'):raise ValueError('Unknown walking teacher mode')
        if policy.anchor.obs_dim!=61 or policy.delta.command_gate or any(
            getattr(policy.anchor,name,False) for name in ['time_input_s','heading_input','yaw_memory_input','state_input','task_input']):
            raise ValueError('Walking retention requires an ungated 61D actor')
        if samples<4:raise ValueError('Walking teacher retention needs four phase samples')
        self.mode=mode;self.samples=samples;self.reference=None
        self.audit=dict(version='walking_command_balanced_teacher_kl_v1',mode=mode,samples=samples,
                        anchor_sha256=policy.anchor.sha256,
                        phases='idle | normal | sprint command >=0.275 | turning command abs(yaw)>0.25')
        if mode=='online_kl':
            if replay:raise ValueError('Online retention must not read teacher replay')
            return
        if not replay:raise ValueError('Walking replay retention requires a data manifest')
        manifest=Path(replay);record=json.loads(manifest.read_text())
        if record.get('version')!='walking_teacher_replay_v1':raise ValueError('Unknown walking replay version')
        data=Path(record['data'])
        if not data.is_absolute():data=manifest.parent/data
        if hashlib.sha256(data.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Walking teacher data checksum mismatch')
        if record['anchor_sha256']!=policy.anchor.sha256:raise ValueError('Walking teacher source mismatch')
        with np.load(data,allow_pickle=False) as archive:x=archive['obs'].copy()
        if x.ndim!=2 or x.shape[1]!=61 or len(x)<samples or not np.isfinite(x).all():
            raise ValueError('Invalid walking teacher observations')
        self.reference=torch.from_numpy(x.astype(np.float32))
        self.audit.update(data_sha256=record['sha256'],manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),frames=len(x))

    def to(self,device):
        if self.reference is not None:self.reference=self.reference.to(device)
        return self

    def loss(self,policy,online_observations):
        x=balanced_sample(self.reference if self.reference is not None else online_observations,self.samples)
        if not len(x):return online_observations.new_zeros(())
        # Policy = frozen native teacher + differentiable increment. This is
        # the exact Gaussian mean KL with the exploration scale held fixed.
        delta=policy.delta(x)
        std=policy.log_std.detach().clamp(np.log(.005),np.log(.3)).exp()
        return .5*(delta/std).square().sum(-1).mean()
