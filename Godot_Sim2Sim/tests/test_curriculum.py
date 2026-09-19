"""Curriculum interpolation and GodotVecEnv.set_curriculum."""

from __future__ import annotations

import unittest

from sim2sim.train.commands import CommandConfig
from sim2sim.train.curriculum import Curriculum, apply_curriculum, interpolate
from sim2sim.train.rewards import RewardConfig
from sim2sim.train.vec_env import GodotVecEnv


class TestInterpolate(unittest.TestCase):
    def test_clamped_and_midpoint(self) -> None:
        self.assertAlmostEqual(interpolate(0, 0.02, 0.25, 0, 1000), 0.02)
        self.assertAlmostEqual(interpolate(-10, 0.02, 0.25, 0, 1000), 0.02)
        self.assertAlmostEqual(interpolate(1000, 0.02, 0.25, 0, 1000), 0.25)
        self.assertAlmostEqual(interpolate(2000, 0.02, 0.25, 0, 1000), 0.25)
        self.assertAlmostEqual(interpolate(500, 0.02, 0.25, 0, 1000), 0.135)

    def test_delayed_start_stays_at_start(self) -> None:
        self.assertAlmostEqual(interpolate(0, 0.0, -3.0, 300, 1000), 0.0)
        self.assertAlmostEqual(interpolate(300, 0.0, -3.0, 300, 1000), 0.0)
        self.assertAlmostEqual(interpolate(650, 0.0, -3.0, 300, 1000), -1.5)
        self.assertAlmostEqual(interpolate(1000, 0.0, -3.0, 300, 1000), -3.0)

    def test_degenerate_window(self) -> None:
        self.assertAlmostEqual(interpolate(0, -0.1, -1.0, 5, 5), -0.1)
        self.assertAlmostEqual(interpolate(5, -0.1, -1.0, 5, 5), -1.0)


class TestCurriculumDict(unittest.TestCase):
    def test_values_at_matches_yaml_shape(self) -> None:
        cur = Curriculum.from_dict(
            {
                "standing_frac": [0.02, 0.25, 0, 1000],
                "action_rate_l2": [-0.1, -1.0, 0, 1000],
                "head_pose_bias": [0.0, -3.0, 300, 1000],
            }
        )
        v0 = cur.values_at(0)
        self.assertAlmostEqual(v0["standing_frac"], 0.02)
        self.assertAlmostEqual(v0["action_rate_l2"], -0.1)
        self.assertAlmostEqual(v0["head_pose_bias"], 0.0)
        v_mid = cur.values_at(500)
        self.assertAlmostEqual(v_mid["standing_frac"], 0.135)
        self.assertAlmostEqual(v_mid["action_rate_l2"], -0.55)
        self.assertAlmostEqual(v_mid["head_pose_bias"], 0.0 + (500 - 300) / 700 * (-3.0))
        v_end = cur.values_at(1000)
        self.assertAlmostEqual(v_end["standing_frac"], 0.25)
        self.assertAlmostEqual(v_end["action_rate_l2"], -1.0)
        self.assertAlmostEqual(v_end["head_pose_bias"], -3.0)

    def test_empty_and_bad_spec(self) -> None:
        self.assertEqual(Curriculum.from_dict(None).values_at(0), {})
        with self.assertRaises(ValueError):
            Curriculum.from_dict({"standing_frac": [0.02, 0.25]})


class TestApplyAndSetter(unittest.TestCase):
    def test_apply_curriculum_writes_cfgs(self) -> None:
        rew = RewardConfig()
        cmd = CommandConfig()
        apply_curriculum(rew, cmd, {"standing_frac": 0.07, "action_rate_l2": -0.4, "head_pose_bias": -1.2})
        self.assertAlmostEqual(cmd.standing_frac, 0.07)
        self.assertAlmostEqual(rew.action_rate_l2, -0.4)
        self.assertAlmostEqual(rew.head_pose_bias, -1.2)

    def test_godot_vec_env_set_curriculum_method(self) -> None:
        class Fake:
            def __init__(self) -> None:
                self.rew = type("R", (), {})()
                self.rew.cfg = RewardConfig(action_rate_l2=-0.1, head_pose_bias=0.0)
                self.commands = type("C", (), {})()
                self.commands.cfg = CommandConfig(standing_frac=0.02)
                self.curriculum = Curriculum.from_dict(
                    {
                        "standing_frac": [0.02, 0.25, 0, 1000],
                        "action_rate_l2": [-0.1, -1.0, 0, 1000],
                        "head_pose_bias": [0.0, -3.0, 300, 1000],
                    }
                )
                self._curriculum_values = {}

        env = Fake()
        vals = GodotVecEnv.set_curriculum(env, 0)  # type: ignore[arg-type]
        self.assertAlmostEqual(vals["standing_frac"], 0.02)
        self.assertAlmostEqual(env.commands.cfg.standing_frac, 0.02)
        self.assertAlmostEqual(env.rew.cfg.action_rate_l2, -0.1)
        vals = GodotVecEnv.set_curriculum(env, 1000)  # type: ignore[arg-type]
        self.assertAlmostEqual(env.commands.cfg.standing_frac, 0.25)
        self.assertAlmostEqual(env.rew.cfg.action_rate_l2, -1.0)
        self.assertAlmostEqual(env.rew.cfg.head_pose_bias, -3.0)
        self.assertEqual(env._curriculum_values, vals)


if __name__ == "__main__":
    unittest.main()
