import tempfile
import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from sim2sim.paths import sim2sim_root
from sim2sim.research.models import Policy,NativeAnchor,export_policy
from sim2sim.research.schedules import roller_keyboard_commands
from sim2sim.standalone.score import TraceWorld
from sim2sim.backends import SimState
from sim2sim.coords import quat_wxyz_to_mat


class BrakeAdaptationTests(unittest.TestCase):
    def test_actual_side_fall_keeps_both_wheel_support_features_finite(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/roller_terminal_support.json').read_text())
        world=TraceWorld('roller','roller');self.addCleanup(world.close)
        bodies=copy.deepcopy(fixture['bodies'])
        for body in bodies.values():
            for key in ['pos','rot','linvel']:body[key]=np.array(body[key])
        world._godot_bodies=lambda reset=False:bodies
        world.state=SimState(t=fixture['time'],q=np.zeros(14),qd=np.zeros(14),
            base_pos=np.array(fixture['base_pos']),base_quat_wxyz=np.array(fixture['base_quat']),
            base_linvel=np.zeros(3),base_angvel_local=np.zeros(3))
        rot=quat_wxyz_to_mat(world.state.base_quat_wxyz)
        old_groups=[[],[]]
        for name,body in bodies.items():
            if name.startswith('tire'):
                side=0 if (rot.T@(body['pos']-world.state.base_pos))[1]>=0 else 1
                old_groups[side].append(name)
        self.assertFalse(all(old_groups)) # The captured failure emptied a support group.
        features=world.measure()
        self.assertGreater(features['tilt'],75.)
        self.assertTrue(np.isfinite(features['foot_pos']).all())
        self.assertTrue(np.isfinite(features['foot_vel']).all())
        self.assertEqual(world._roller_supports,[['tire','tire_2'],['tire_3','tire_4']])

    def test_training_tape_contains_original_brake_transition(self):
        commands=roller_keyboard_commands('roller_brake_3s')
        np.testing.assert_array_equal(commands[:50],0.)
        self.assertAlmostEqual(float(commands[51,0]),.48,places=6)
        self.assertAlmostEqual(float(commands[199,0]),.6,places=6)
        self.assertAlmostEqual(float(commands[200,0]),.2,places=6)
        self.assertAlmostEqual(float(commands[201,0]),-.2,places=6)
        self.assertAlmostEqual(float(commands[203,0]),-.5,places=6)
        np.testing.assert_array_equal(commands[:,1:],0.)

    def test_nonbraking_commands_remain_exact_factory_after_learning_and_export(self):
        source=sim2sim_root()/'policies/roller.onnx'
        if not source.exists():self.skipTest('Factory policies not downloaded')
        torch.set_num_threads(1)
        actor=Policy(source,'residual',command_gate='negative_throttle')
        with torch.no_grad():actor.delta.net[-1].bias.fill_(.4)
        observations=np.random.default_rng(915000).normal(0,.1,(12,61)).astype(np.float32)
        observations[:6,48]=[0.,.02,.1,.3,.5,.6];observations[6:,48]=-.5
        factory=actor.anchor(observations);adapted=actor.predict(observations)
        np.testing.assert_array_equal(adapted[:6],factory[:6])
        self.assertGreater(np.max(np.abs(adapted[6:]-factory[6:])),.01)
        with tempfile.TemporaryDirectory() as directory:
            path=export_policy(actor,Path(directory)/'actor.onnx');native=NativeAnchor(path)(observations)
        np.testing.assert_array_equal(native[:6],factory[:6])
        np.testing.assert_allclose(native,adapted,rtol=0,atol=1e-5)
        std=actor.distribution(torch.from_numpy(observations)).stddev.detach().numpy()
        self.assertLess(float(std[:6].max()),1.1e-5)
        self.assertGreater(float(std[6:].min()),.02)


if __name__=='__main__':unittest.main()
