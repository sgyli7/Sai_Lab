"""Pure-numpy reward kernels (no Godot)."""

from __future__ import annotations

import unittest

import numpy as np

from sim2sim.train.rewards import (
    AirTimeTracker,
    HeadPoseBias,
    action_rate_l2,
    air_time_reward,
    track_ang_vel,
    track_lin_vel,
    upright,
)


class TestTrackingKernels(unittest.TestCase):
    def test_track_lin_vel_is_one_at_zero_error(self) -> None:
        v_cmd = np.array([[0.25, -0.1], [0.0, 0.0]], dtype=np.float32)
        got = track_lin_vel(v_cmd, v_cmd, std2=0.1)
        self.assertEqual(got.shape, (2,))
        self.assertTrue(np.allclose(got, 1.0))
        miss = track_lin_vel(v_cmd, np.zeros_like(v_cmd), std2=0.1)
        self.assertLess(float(miss[0]), 1.0)
        self.assertEqual(float(miss[1]), 1.0)

    def test_track_ang_vel_is_one_at_zero_error(self) -> None:
        wz = np.array([0.5, 0.0], dtype=np.float32)
        got = track_ang_vel(wz, wz, std2=0.5)
        self.assertTrue(np.allclose(got, 1.0))
        miss = track_ang_vel(wz, np.zeros_like(wz), std2=0.5)
        self.assertLess(float(miss[0]), 1.0)

    def test_upright_is_one_when_projected_grav_is_down(self) -> None:
        grav = np.array([[0.0, 0.0, -1.0], [0.5, 0.0, -np.sqrt(0.75)]], dtype=np.float32)
        got = upright(grav, std2=0.05)
        self.assertAlmostEqual(float(got[0]), 1.0, places=5)
        self.assertLess(float(got[1]), 1.0)


class TestAirTimeWindow(unittest.TestCase):
    def test_in_window_air_time_rewards_each_step(self) -> None:
        """mjlab feet_air_time: +1/foot/step while current air time is in (tmin, tmax)."""
        tr = AirTimeTracker(num_envs=1, n_feet=2)
        dt = 0.02
        cmd = np.array([0.2], dtype=np.float32)
        contact = np.array([[False, True]])
        last = 0.0
        for _ in range(10):
            tr.step(contact, dt)
            last = float(air_time_reward(tr.air_time, cmd, tmin=0.125, tmax=0.3, cmd_threshold=0.01)[0])
        # 10 * 0.02 = 0.20 s airborne on the left foot → in (0.125, 0.3).
        self.assertAlmostEqual(last, 1.0, places=5)

    def test_outside_window_is_zero(self) -> None:
        tr = AirTimeTracker(num_envs=1, n_feet=2)
        dt = 0.02
        cmd = np.array([0.2], dtype=np.float32)
        contact = np.array([[False, True]])
        for _ in range(2):  # 0.04 s, below 0.125
            tr.step(contact, dt)
        r = air_time_reward(tr.air_time, cmd, tmin=0.125, tmax=0.3)
        self.assertAlmostEqual(float(r[0]), 0.0, places=5)

        tr.reset([0])
        for _ in range(20):  # 0.40 s, above 0.3
            tr.step(contact, dt)
        r = air_time_reward(tr.air_time, cmd, tmin=0.125, tmax=0.3)
        self.assertAlmostEqual(float(r[0]), 0.0, places=5)

    def test_standing_command_gates_air_time(self) -> None:
        tr = AirTimeTracker(num_envs=1, n_feet=2)
        dt = 0.02
        contact = np.array([[False, True]])
        for _ in range(10):
            tr.step(contact, dt)
        r = air_time_reward(tr.air_time, np.array([0.0], dtype=np.float32), tmin=0.125, tmax=0.3)
        self.assertAlmostEqual(float(r[0]), 0.0, places=5)


class TestActionRateAndEma(unittest.TestCase):
    def test_action_rate_l2_is_nonnegative_and_zero_when_equal(self) -> None:
        a = np.array([[1.0, -0.5, 0.0]], dtype=np.float32)
        prev = np.array([[0.0, -0.5, 1.0]], dtype=np.float32)
        r = action_rate_l2(a, prev)
        self.assertEqual(r.shape, (1,))
        self.assertGreater(float(r[0]), 0.0)
        self.assertAlmostEqual(float(r[0]), 1.0 + 1.0, places=5)
        self.assertAlmostEqual(float(action_rate_l2(a, a)[0]), 0.0, places=6)

    def test_head_pose_bias_ema_tracks_constant_error(self) -> None:
        bias = HeadPoseBias(num_envs=1, n_joints=4, tau_s=1.0, dt=0.02)
        err = np.ones((1, 4), dtype=np.float32)
        last = 0.0
        for _ in range(300):
            last = float(bias.step(err)[0])
        self.assertGreater(last, 0.0)
        self.assertAlmostEqual(last, 1.0, places=2)
        bias.reset([0])
        z = float(bias.step(np.zeros((1, 4), dtype=np.float32))[0])
        self.assertAlmostEqual(z, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
