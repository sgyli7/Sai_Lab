import unittest

import numpy as np

from sim2sim.sai_stair_v6 import (
    JOINT_SOFT_LIMITS,
    project_target_safety,
    TARGET_SLEW_RAD_S,
    _foot_from_joints,
    _joints_from_foot,
    continuous_stair_heights,
    localize_terrain_levels,
    project_action_safety,
    targets_numpy,
)


class SaiStairV6Tests(unittest.TestCase):
    def test_final_target_projection_catches_post_skill_contributions(self):
        target = np.array([.4, .7, 1.2, 9.] * 4)
        actual = project_target_safety(target)
        finite = np.isfinite(JOINT_SOFT_LIMITS)
        self.assertTrue(np.all(np.abs(actual[finite]) <= JOINT_SOFT_LIMITS[finite] + 1e-6))
        np.testing.assert_allclose(actual[3::4], 9.)

    def test_pitch_guard_only_overrides_commands_that_increase_large_tilt(self):
        action = np.zeros(16)
        action[14] = -.7
        small_pitch = np.array([-np.sin(np.deg2rad(5.)), 0., np.cos(np.deg2rad(5.))])
        np.testing.assert_allclose(project_action_safety(action, small_pitch), action)

        large_nose_up = np.array([-np.sin(np.deg2rad(12.)), 0., np.cos(np.deg2rad(12.))])
        corrected = project_action_safety(action, large_nose_up)
        self.assertAlmostEqual(float(corrected[14]), 1.0, places=5)

        action[14] = .8
        self.assertAlmostEqual(float(project_action_safety(action, large_nose_up)[14]), 1.0, places=5)

    def test_continuous_stair_observation_matches_physical_box_levels(self):
        start, tread, rise = -.10, .18, .04
        x = np.array([start - 1e-3, start, start + .09,
                      start + tread, start + 2 * tread, start + 5 * tread])
        actual = continuous_stair_heights(x, start, tread, rise)
        np.testing.assert_allclose(actual, [0., .04, .04, .08, .12, .16], atol=1e-7)

    def test_leg_kinematics_round_trip(self):
        hip = np.array([0.12, -0.08, 0.06, -0.10])
        knee = np.array([-0.35, 0.42, 0.30, -0.38])
        dx, down = _foot_from_joints(hip, knee)
        actual_hip, actual_knee = _joints_from_foot(dx, down)
        np.testing.assert_allclose(actual_hip, hip, atol=1e-10)
        np.testing.assert_allclose(actual_knee, knee, atol=1e-10)

    def test_zero_intensity_preserves_rolling_prior(self):
        base = np.array([0.02, 0.10, -0.25, 3.0] * 4)
        action = np.ones(16)
        action[15] = -1.0
        actual = targets_numpy(action, base, np.zeros(16))
        np.testing.assert_allclose(actual, base, atol=1e-10)

        changed_base = base + np.array([0.01, -0.02, 0.03, 0.2] * 4)
        actual = targets_numpy(action, changed_base, np.zeros(16))
        np.testing.assert_allclose(actual, changed_base, atol=1e-10)

    def test_policy_cannot_directly_move_haa(self):
        base = np.array([0.07, 0.0, 0.0, 2.0] * 4)
        action = np.zeros(16)
        action[:4] = [1.0, -1.0, 1.0, -1.0]
        action[15] = 1.0
        actual = targets_numpy(action, base, np.zeros(16))
        np.testing.assert_allclose(actual[0::4], base[0::4])

    def test_forward_wheel_residual_respects_mirrored_joint_axes(self):
        base = np.zeros(16)
        action = np.zeros(16)
        action[8:12] = 1.0
        action[15] = 1.0
        actual = targets_numpy(action, base, np.zeros(16))
        np.testing.assert_allclose(actual[3::4], np.array([.4, -.4, .4, -.4]))

    def test_wheel_residual_cannot_fight_heading_hold_with_left_right_difference(self):
        base = np.zeros(16)
        action = np.zeros(16)
        action[8:12] = [1., -1., .5, -.5]
        action[15] = 1.
        actual = targets_numpy(action, base, np.zeros(16))
        np.testing.assert_allclose(actual[3::4], 0., atol=1e-10)

    def test_endpoint_skill_obeys_joint_and_rate_envelopes(self):
        base = np.zeros(16)
        action = np.ones(16)
        previous_residual = np.zeros(16)
        actual = targets_numpy(action, base, previous_residual)
        finite = np.isfinite(JOINT_SOFT_LIMITS)
        self.assertTrue(np.all(np.abs(actual[finite]) <= JOINT_SOFT_LIMITS[finite] + 1e-12))
        self.assertTrue(np.all(np.abs(actual - base - previous_residual)
                               <= TARGET_SLEW_RAD_S * 0.02 + 1e-12))
        self.assertLess(actual[2], 0.0)  # front-left knee flexes to lift its wheel

    def test_multistep_terrain_is_reduced_to_nearest_local_level(self):
        observation = np.zeros(258, dtype=np.float32)
        observation[56:80] = np.repeat([0., 0., 0., 0.2, 0.4, 0.6, 0.8, 0.8], 3)
        observation[104:242] = np.repeat([0., .2, .4, .6], [75, 24, 24, 15])
        localized = localize_terrain_levels(observation)
        self.assertAlmostEqual(float(localized[56:242].max()), .2, places=6)
        np.testing.assert_allclose(localized[95:104], observation[95:104])


if __name__ == "__main__":
    unittest.main()
