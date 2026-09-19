"""Versioned teacher retention on current or successful reference occupancy.

The frozen anchor is the teacher, so its exact mean difference is policy.delta;
there is no approximate imported teacher or extra ONNX call in the gradient.
Sampling uses torch's checkpointed RNG, including on resumed physical episodes.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def phase_sample(observations, count):
    braking=observations[observations[:,48]<-.01]
    if not len(braking):return braking
    elapsed=braking[:,61]
    pools=[braking[elapsed<1.],braking[(elapsed>=1.)&(elapsed<2.)],braking[elapsed>=2.]]
    pools=[pool for pool in pools if len(pool)]
    sizes=[count//len(pools)+(i<count%len(pools)) for i in range(len(pools))]
    return torch.cat([pool[torch.randint(len(pool),(size,))] for pool,size in zip(pools,sizes)])


class TeacherRetention:
    def __init__(self, policy, mode, replay=None, samples=384):
        if mode not in ('online_kl','replay_kl'):
            raise ValueError('Unknown teacher retention mode')
        if policy.anchor.obs_dim!=68 or policy.delta.command_gate!='negative_throttle':
            raise ValueError('Teacher retention requires the 68D brake-gated actor')
        if samples<3:raise ValueError('Teacher retention needs at least three samples')
        self.mode=mode;self.samples=samples;self.reference=None
        self.audit=dict(version='phase_balanced_teacher_kl_v1',mode=mode,samples=samples)
        if mode=='online_kl':
            if replay:raise ValueError('Online control group must not read teacher replay')
        else:
            if not replay:raise ValueError('Replay retention requires a completed data manifest')
            manifest=Path(replay);record=json.loads(manifest.read_text())
            data=Path(record['data'])
            if not data.is_absolute():data=manifest.parent/data
            if hashlib.sha256(data.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('Teacher data checksum mismatch')
            if record['anchor_sha256']!=policy.anchor.sha256:
                raise ValueError('Teacher replay belongs to a different frozen anchor')
            with np.load(data,allow_pickle=False) as archive:x=archive['obs'].copy()
            if x.ndim!=2 or x.shape[1]!=68 or len(x)<samples or not np.isfinite(x).all():
                raise ValueError('Invalid teacher observation matrix')
            if np.any(x[:,48]>=-.01):raise ValueError('Teacher replay must contain braking frames only')
            if not all(np.any(mask) for mask in [x[:,61]<1.,(x[:,61]>=1.)&(x[:,61]<2.),x[:,61]>=2.]):
                raise ValueError('Teacher replay is missing a task phase')
            self.reference=torch.from_numpy(x.astype(np.float32))
            self.audit.update(data_sha256=record['sha256'],manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                anchor_sha256=record['anchor_sha256'],teacher_sha256=record['teacher_sha256'],frames=len(x))

    def to(self, device):
        if self.reference is not None:self.reference=self.reference.to(device)
        return self

    def loss(self, policy, online_observations):
        reference=self.reference if self.reference is not None else online_observations
        x=phase_sample(reference,self.samples)
        if not len(x):return online_observations.new_zeros(())
        delta=policy.delta(x)
        std=policy.log_std.detach().clamp(np.log(.005),np.log(.3)).exp()
        std=std*policy.delta.command_weight(x)*policy.delta.action_mask.to(std.dtype)+1e-5
        return .5*(delta/std).square().sum(-1).mean()
