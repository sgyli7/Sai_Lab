import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from sim2sim.research.teacher_retention import TeacherRetention,phase_sample


class Delta(torch.nn.Module):
    command_gate='negative_throttle'
    def __init__(self):
        super().__init__();self.bias=torch.nn.Parameter(torch.full((14,),.01));self.action_mask=torch.ones(14)
    def command_weight(self,x):return (-x[:,48:49]/.05).clamp(0,1)
    def forward(self,x):return self.bias.expand(len(x),-1)*self.command_weight(x)


class RetentionTests(unittest.TestCase):
    def observations(self):
        x=torch.zeros(600,68);x[:,48]=-.5;x[:100,61]=.2;x[100:300,61]=1.4;x[300:,61]=3.
        return x
    def policy(self):
        return SimpleNamespace(anchor=SimpleNamespace(obs_dim=68,sha256='anchor'),
            delta=Delta(),log_std=torch.log(torch.full((14,),.02)))
    def test_uneven_occupancy_is_phase_balanced_and_push_is_excluded(self):
        x=self.observations();x[500:,48]=.6
        s=phase_sample(x,300)
        self.assertEqual(int((s[:,61]<1).sum()),100)
        self.assertEqual(int(((s[:,61]>=1)&(s[:,61]<2)).sum()),100)
        self.assertEqual(int((s[:,61]>=2).sum()),100)
        self.assertTrue(torch.all(s[:,48]<-.01))
    def test_anchor_kl_has_expected_units_and_moves_increment_toward_teacher(self):
        p=self.policy();r=TeacherRetention(p,'online_kl')
        loss=r.loss(p,self.observations())
        self.assertAlmostEqual(float(loss.detach()),.5*14*(.01/.02001)**2,places=5)
        loss.backward();self.assertTrue(torch.all(p.delta.bias.grad>0))
        with torch.no_grad():p.delta.bias.zero_()
        self.assertEqual(float(r.loss(p,self.observations()).detach()),0.)
    def test_replay_is_independent_of_current_occupancy_and_checks_its_source(self):
        p=self.policy()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);data=path/'data.npz';np.savez(data,obs=self.observations().numpy())
            record=dict(data='data.npz',sha256=hashlib.sha256(data.read_bytes()).hexdigest(),anchor_sha256='anchor',teacher_sha256='teacher')
            manifest=path/'manifest.json';manifest.write_text(json.dumps(record))
            r=TeacherRetention(p,'replay_kl',manifest)
            online=self.observations();online[:,48]=.6
            self.assertGreater(float(r.loss(p,online).detach()),0.)
            self.assertEqual(float(TeacherRetention(p,'online_kl').loss(p,online)),0.)
            p.anchor.sha256='other'
            with self.assertRaisesRegex(ValueError,'different frozen anchor'):TeacherRetention(p,'replay_kl',manifest)
            p.anchor.sha256='anchor';data.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):TeacherRetention(p,'replay_kl',manifest)


if __name__=='__main__':unittest.main()
