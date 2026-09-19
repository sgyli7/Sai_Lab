import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from sim2sim.research.walking_retention import WalkingTeacherRetention,balanced_sample


class WalkingRetention(unittest.TestCase):
    def test_trained_teacher_is_frozen_and_does_not_pull_back_to_original_anchor(self):
        delta=torch.nn.Linear(61,14);delta.command_gate=''
        with torch.no_grad():delta.weight.zero_();delta.bias.fill_(.2)
        policy=SimpleNamespace(anchor=SimpleNamespace(obs_dim=61,sha256='teacher'),
                               delta=delta,log_std=torch.full((14,),np.log(.1)))
        keeper=WalkingTeacherRetention(policy,'online_kl',samples=4,reference_increment=delta)
        inputs=torch.zeros(8,61)
        self.assertEqual(float(keeper.loss(policy,inputs)),0.)
        with torch.no_grad():delta.bias.add_(.02)
        loss=keeper.loss(policy,inputs)
        self.assertAlmostEqual(float(loss.detach()),.28,places=5)
        loss.backward()
        self.assertTrue(torch.all(delta.bias.grad>0))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in keeper.reference_increment.parameters()))
        torch.testing.assert_close(keeper.reference_increment.bias,torch.full((14,),.2))

    def test_sampling_keeps_rare_idle_and_turn_states(self):
        x=torch.zeros(103,61);x[:100,48]=.3;x[100,48]=.25;x[102,48]=.25;x[102,50]=.8
        y=balanced_sample(x,40)
        self.assertEqual(int((y[:,48]==0).sum()),10)
        self.assertEqual(int((y[:,48]==.3).sum()),10)
        self.assertEqual(int((y[:,50]==.8).sum()),10)
        self.assertEqual(len(balanced_sample(torch.zeros(3,61),12)),12)

    def test_loss_matches_teacher_mean_kl_and_has_gradient(self):
        weight=torch.nn.Parameter(torch.full((14,),.02))
        policy=SimpleNamespace(anchor=SimpleNamespace(obs_dim=61,sha256='teacher'),
            delta=SimpleNamespace(command_gate=''),log_std=torch.full((14,),np.log(.1)))
        keeper=WalkingTeacherRetention(policy,'online_kl',samples=4)
        policy.delta=lambda obs:weight.expand(len(obs),-1)
        loss=keeper.loss(policy,torch.zeros(10,61));self.assertAlmostEqual(float(loss.detach()),.28,places=6)
        loss.backward();self.assertTrue(torch.all(weight.grad>0))

    def test_replay_rejects_changed_data_or_wrong_teacher(self):
        policy=SimpleNamespace(anchor=SimpleNamespace(obs_dim=61,sha256='teacher'),delta=SimpleNamespace(command_gate=''))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data.npz';np.savez(data,obs=np.zeros((8,61),np.float32))
            record=dict(version='walking_teacher_replay_v1',data=data.name,
                        sha256=hashlib.sha256(data.read_bytes()).hexdigest(),anchor_sha256='teacher')
            manifest=root/'manifest.json';manifest.write_text(json.dumps(record))
            keep=WalkingTeacherRetention(policy,'replay_kl',manifest,samples=4)
            self.assertEqual(tuple(keep.reference.shape),(8,61))
            record['anchor_sha256']='wrong';manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'source mismatch'):WalkingTeacherRetention(policy,'replay_kl',manifest,samples=4)
            record['anchor_sha256']='teacher';manifest.write_text(json.dumps(record));data.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):WalkingTeacherRetention(policy,'replay_kl',manifest,samples=4)
