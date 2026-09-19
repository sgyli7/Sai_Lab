import unittest

import numpy as np

from sim2sim.sai_stair_v2 import OBSERVATION_SIZE, observation_numpy, targets_numpy, actions_from_targets_numpy
from sim2sim.sai_stair_v3 import OBSERVATION_SIZE as DENSE_OBSERVATION_SIZE, observation_numpy as dense_observation_numpy
from sim2sim.sai_stair_v4 import OBSERVATION_SIZE as EVENT_OBSERVATION_SIZE, observation_numpy as event_observation_numpy
from sim2sim.sai_stair_v5 import JOINT_ACTION_SCALES, TARGET_SLEW_RAD_S, targets_numpy as safe_targets_numpy


class SaiStairV2ContractTests(unittest.TestCase):
    def state(self):
        return dict(q=[0.] * 25, v=[0.] * 25, base_position=[0., 0., .2192],
            base_rotation_columns=np.eye(3).tolist(), base_linear_world=[.1, 0., 0.],
            base_angular_world=[0., 0., 0.], terrain_heights=[0.] * 24,
            terrain_path_heights=[0.] * 15, wheel_ground_heights=[0.] * 4,
            terrain_edge_heights=[0.] * 138,
            wheel_positions=[[.15, .146, .048], [.15, -.146, .048],
                             [-.15, .146, .048], [-.15, -.146, .048]])

    def test_observation_has_no_clock_and_exposes_per_wheel_clearance(self):
        state = self.state()
        a = observation_numpy(dict(state, time=.1), [.16, 0., 0.], np.zeros(16))
        b = observation_numpy(dict(state, time=9.7), [.16, 0., 0.], np.zeros(16))
        self.assertEqual(a.shape, (OBSERVATION_SIZE,))
        np.testing.assert_array_equal(a, b)
        lifted = self.state();lifted["wheel_positions"][0][2] += .03
        c = observation_numpy(lifted, [.16, 0., 0.], np.zeros(16))
        self.assertAlmostEqual(float(c[99] - a[99]), .6, places=6)

    def test_direct_action_can_cancel_every_leg_motion(self):
        zero = targets_numpy(np.zeros(16), [.16, 0., 0.], 0.)
        np.testing.assert_allclose(zero[:3], 0., atol=1e-12)
        np.testing.assert_allclose(zero[4:7], 0., atol=1e-12)
        np.testing.assert_allclose(zero[3::4], [10/3, -10/3, 10/3, -10/3], atol=1e-12)
        action = np.zeros(16);action[1] = 1.;action[2] = -1.
        target = targets_numpy(action, [.16, 0., 0.], 0.)
        self.assertEqual(target[1], .70);self.assertEqual(target[2], -1.20)

    def test_direct_target_mapping_is_invertible(self):
        action = np.linspace(-.8, .8, 16, dtype=np.float32)
        command = np.array([.16, .03, 0.])
        target = targets_numpy(action, command, 0.)
        np.testing.assert_allclose(actions_from_targets_numpy(target, command), action, atol=1e-6)

    def test_dense_contract_preserves_v2_prefix(self):
        state = self.state()
        base = observation_numpy(state, [.16, 0., 0.], np.zeros(16))
        dense = dense_observation_numpy(state, [.16, 0., 0.], np.zeros(16))
        self.assertEqual(dense.shape, (DENSE_OBSERVATION_SIZE,))
        np.testing.assert_array_equal(dense[:OBSERVATION_SIZE], base)

    def test_event_memory_is_one_bit_and_preserves_dense_prefix(self):
        state = dict(self.state(), high_obstacle_latched=1.)
        dense = dense_observation_numpy(state, [.16, 0., 0.], np.zeros(16))
        event = event_observation_numpy(state, [.16, 0., 0.], np.zeros(16))
        self.assertEqual(event.shape, (EVENT_OBSERVATION_SIZE,))
        np.testing.assert_array_equal(event[:DENSE_OBSERVATION_SIZE], dense)
        self.assertEqual(event[-1], 1.)

    def test_safe_residual_contract_keeps_margin_and_rate_limits_targets(self):
        action = np.ones(16)
        previous = np.zeros(16)
        target = safe_targets_numpy(action, [.16, 0., 0.], previous)
        np.testing.assert_allclose(target[[0, 4, 8, 12]], .02 * TARGET_SLEW_RAD_S[[0, 4, 8, 12]])
        self.assertTrue(np.all(JOINT_ACTION_SCALES[[0, 4, 8, 12]] < .45))
        self.assertTrue(np.all(JOINT_ACTION_SCALES[[1, 5, 9, 13]] < .70))
        self.assertTrue(np.all(JOINT_ACTION_SCALES[[2, 6, 10, 14]] < 1.20))
        for _ in range(100):
            target = safe_targets_numpy(action, [.16, 0., 0.], target)
        np.testing.assert_allclose(target[[0, 4, 8, 12]], .30)
        np.testing.assert_allclose(target[[1, 5, 9, 13]], .60)
        np.testing.assert_allclose(target[[2, 6, 10, 14]], 1.02)


if __name__ == "__main__":
    unittest.main()
