"""Command modes and skill reward kernels (no Godot)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim2sim.obs import DEFAULT_HOME
from sim2sim.paths import sim2sim_root
from sim2sim.train.commands import CommandConfig, CommandSampler, VALID_MODES
from sim2sim.train.rewards import (
    RewardComputer,
    RewardConfig,
    RewardInputs,
    kick_swing_reward,
    mouth_proximity,
    pick_phase_from_cmd,
    posture_height,
    posture_pose,
    sit_target_q,
)


class TestCommandModes(unittest.TestCase):
    def test_zeros_stay_zero(self) -> None:
        rng = np.random.default_rng(0)
        s = CommandSampler({"mode": "zeros"}, 4, rng)
        cmd = s.step(0.02)
        self.assertTrue(np.allclose(cmd, 0.0))
        s.reset([0, 2])
        self.assertTrue(np.allclose(s.cmd, 0.0))

    def test_sit_flag_is_zero_or_one(self) -> None:
        rng = np.random.default_rng(1)
        s = CommandSampler({"mode": "sit_flag", "sit_prob": 0.5, "resample_s": [1.0, 1.0]}, 32, rng)
        flags = s.cmd[:, 0]
        self.assertTrue(np.all((flags == 0.0) | (flags == 1.0)))
        self.assertTrue(np.allclose(s.cmd[:, 1:], 0.0))
        s.step(1.01)
        flags2 = s.cmd[:, 0]
        self.assertTrue(np.all((flags2 == 0.0) | (flags2 == 1.0)))

    def test_pick_phase_is_unit_circle(self) -> None:
        rng = np.random.default_rng(2)
        s = CommandSampler({"mode": "pick_phase", "pick_period": 4.0}, 3, rng)
        mag = np.hypot(s.cmd[:, 0], s.cmd[:, 1])
        self.assertTrue(np.allclose(mag, 1.0, atol=1e-5))
        before = s.phase.copy()
        s.step(1.0)
        self.assertTrue(np.allclose((before + 0.25) % 1.0, s.phase, atol=1e-6))
        mag2 = np.hypot(s.cmd[:, 0], s.cmd[:, 1])
        self.assertTrue(np.allclose(mag2, 1.0, atol=1e-5))

    def test_twist_default_unchanged(self) -> None:
        rng = np.random.default_rng(3)
        s = CommandSampler(CommandConfig(), 8, rng)
        self.assertEqual(s.cfg.mode, "twist")
        self.assertTrue(np.allclose(s.cmd[:, 3:], 0.0))

    def test_bad_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            CommandConfig.from_dict({"mode": "nope"})
        self.assertIn("sit_flag", VALID_MODES)


class TestSkillRewards(unittest.TestCase):
    def test_posture_height_peaks_at_target(self) -> None:
        sit = np.array([1.0, 0.0], dtype=np.float32)
        z = np.array([0.060, 0.115], dtype=np.float32)
        got = posture_height(z, sit)
        self.assertTrue(np.allclose(got, 1.0, atol=1e-5))
        miss = posture_height(np.array([0.115, 0.060]), sit)
        self.assertTrue(np.all(miss < 0.2))

    def test_posture_pose_sit_target(self) -> None:
        home = DEFAULT_HOME.copy()
        sit_q = sit_target_q(home)
        n = 2
        q = np.stack([sit_q, home])
        flag = np.array([1.0, 0.0], dtype=np.float32)
        got = posture_pose(q, home, sit_q, flag)
        self.assertTrue(np.allclose(got, 1.0, atol=1e-5))

    def test_mouth_proximity_gated_on_approach(self) -> None:
        z = np.array([0.01, 0.01], dtype=np.float32)
        phase = np.array([0.25, 0.75], dtype=np.float32)
        got = mouth_proximity(z, phase, std=0.03)
        self.assertGreater(float(got[0]), 0.8)
        self.assertEqual(float(got[1]), 0.0)

    def test_pick_phase_from_unit_circle(self) -> None:
        cmd = np.zeros((2, 13), dtype=np.float32)
        cmd[0, 0], cmd[0, 1] = 1.0, 0.0
        cmd[1, 0], cmd[1, 1] = -1.0, 0.0
        ph = pick_phase_from_cmd(cmd)
        self.assertAlmostEqual(float(ph[0]), 0.0, places=5)
        self.assertAlmostEqual(float(ph[1]), 0.5, places=5)

    def test_kick_swing_gates_on_window_and_foot(self) -> None:
        h = np.array([[0.05, 0.0], [0.0, 0.05]], dtype=np.float32)
        v = np.array([[0.4, 0.0], [0.0, 0.4]], dtype=np.float32)
        t = np.array([0.5, 2.0], dtype=np.float32)
        left = kick_swing_reward(h, v, t, foot="left", window_s=1.2)
        self.assertGreater(float(left[0]), 0.4)
        self.assertEqual(float(left[1]), 0.0)

    def test_reward_computer_zeros_new_terms_for_walk_cfg(self) -> None:
        n = 2
        cfg = RewardConfig()
        rew = RewardComputer(cfg, n, 0.02, DEFAULT_HOME, -np.ones(14), np.ones(14) * 2)
        inp = RewardInputs(
            q=np.tile(DEFAULT_HOME, (n, 1)),
            gyro=np.zeros((n, 3), dtype=np.float32),
            grav=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32),
            base_linvel_yaw=np.zeros((n, 3), dtype=np.float32),
            cmd13=np.zeros((n, 13), dtype=np.float32),
            action=np.zeros((n, 14), dtype=np.float32),
            last_action=np.zeros((n, 14), dtype=np.float32),
            contact=np.ones((n, 2), dtype=np.float32),
            foot_height=np.zeros((n, 2), dtype=np.float32),
            foot_xy_speed=np.zeros((n, 2), dtype=np.float32),
            joint_lo=-np.ones(14, dtype=np.float32),
            joint_hi=np.ones(14, dtype=np.float32),
            home=DEFAULT_HOME,
        )
        total, terms = rew.compute(inp)
        self.assertEqual(total.shape, (n,))
        for name in (
            "posture_pose",
            "mouth_proximity",
            "approach_height",
            "kick_swing",
            "roulade_progress",
        ):
            self.assertTrue(np.allclose(terms[name], 0.0))

    def test_approach_height_pays_on_crouch_phase(self) -> None:
        from sim2sim.train.rewards import approach_height

        z = np.array([0.075, 0.075, 0.115], dtype=np.float32)
        phase = np.array([0.25, 0.75, 0.25], dtype=np.float32)
        got = approach_height(z, phase)
        self.assertGreater(float(got[0]), 0.95)
        self.assertEqual(float(got[1]), 0.0)
        self.assertLess(float(got[2]), 0.5)


class TestSkillYamlsLoad(unittest.TestCase):
    def test_all_skill_yamls_parse(self) -> None:
        from sim2sim.train.vec_env import load_walk_cfg

        root = sim2sim_root()
        names = (
            "stand_godot.yaml",
            "sitstand_godot.yaml",
            "pick_godot.yaml",
            "kick_left_godot.yaml",
            "kick_right_godot.yaml",
            "roulade_godot.yaml",
            "roller_godot.yaml",
            "roller_crouch_godot.yaml",
        )
        for name in names:
            cfg = load_walk_cfg(root / "configs" / name)
            self.assertIn("init_onnx", cfg)
            self.assertIn("export", cfg)
            self.assertIn("onnx", cfg["export"])
            mode = (cfg.get("commands") or {}).get("mode", "twist")
            self.assertIn(mode, VALID_MODES)


class TestSupportBodies(unittest.TestCase):
    def test_walk_and_roller_pairs(self) -> None:
        from sim2sim.train.reset_poses import support_body_pair

        walk = [{"name": "trunk_base"}, {"name": "ankle_left"}, {"name": "ankle_right"}]
        self.assertEqual(support_body_pair(walk), ("ankle_left", "ankle_right"))
        roller = [
            {"name": "trunk_base"},
            {"name": "ankle_l_v1"},
            {"name": "ankle_r_v1"},
            {"name": "tire"},
            {"name": "tire_3"},
        ]
        self.assertEqual(support_body_pair(roller), ("ankle_l_v1", "ankle_r_v1"))
        with self.assertRaises(KeyError):
            support_body_pair([{"name": "trunk_base"}])


class TestEvalReport(unittest.TestCase):
    def test_sitstand_notes_split_pose(self) -> None:
        from sim2sim.train.eval_skill import write_report

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sit.md"
            write_report(
                [
                    {
                        "name": "sitstand",
                        "kind": "sitstand",
                        "A": {
                            "fell_rate": 0.0,
                            "mean_trunk_z": 0.09,
                            "sit": {"final_trunk_z": 0.057, "pose_err_sit": 0.071},
                            "stand": {"final_trunk_z": 0.116, "pose_err_home": 0.029},
                        },
                        "B": {
                            "fell_rate": 0.0,
                            "mean_trunk_z": 0.09,
                            "sit": {"final_trunk_z": 0.062, "pose_err_sit": 0.057},
                            "stand": {"final_trunk_z": 0.117, "pose_err_home": 0.045},
                        },
                        "b_fewer_falls": True,
                    }
                ],
                path,
            )
            text = path.read_text()
            self.assertIn("pose_sit", text)
            self.assertIn("stand_z", text)
            self.assertIn("0.062", text)


class TestPlayPrefersGodot(unittest.TestCase):
    def test_prefer_helper_falls_back(self) -> None:
        from sim2sim.play import _prefer

        missing = _prefer("DefinitelyMissing_Godot.onnx", Path("/no/such/alpha.onnx"))
        self.assertIsNone(missing)

    def test_policy_paths_prefers_godot_over_alpha(self) -> None:
        import sim2sim.play as play

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "Stand_Godot.onnx").write_bytes(b"godot")
            (d / "alpha_stand.onnx").write_bytes(b"alpha")
            (d / "Walk_Godot.onnx").write_bytes(b"walkg")
            (d / "alpha_walking.onnx").write_bytes(b"walka")
            orig = play.policy_search_dirs
            play.policy_search_dirs = lambda: [d]  # type: ignore[method-assign]
            try:
                paths = play.policy_paths(local_ppo=False)
            finally:
                play.policy_search_dirs = orig
            self.assertEqual(paths["standing"], d / "Stand_Godot.onnx")
            self.assertEqual(paths["walking"], d / "Walk_Godot.onnx")
