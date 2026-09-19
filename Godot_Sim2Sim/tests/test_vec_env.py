"""GodotVecEnv with real workers (skip if spec/Godot missing)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from sim2sim.godot_proc import godot_bin
from sim2sim.paths import sim2sim_root

ROOT = sim2sim_root()
YAML = ROOT / "configs/walk_godot.yaml"
CFG_PATH = ROOT / "robots/microduck.json"


def _skip_reason() -> str | None:
    try:
        import torch  # noqa: F401
        import tensordict  # noqa: F401
    except ImportError:
        return "train extra not installed"
    from sim2sim.runner import load_robot_cfg

    cfg = load_robot_cfg(CFG_PATH)
    spec = Path(cfg["godot_spec"])
    if not spec.is_file():
        return f"missing spec {spec}"
    if not Path(godot_bin()).is_file():
        return f"missing godot {godot_bin()}"
    if not YAML.is_file():
        return f"missing {YAML}"
    return None


_SKIP = _skip_reason()


def _walk_cfg() -> dict:
    from sim2sim.train.vec_env import load_walk_cfg

    cfg = load_walk_cfg(YAML)
    cfg["obs_noise"]["enabled"] = False
    cfg["pushes"]["enabled"] = False
    cfg["recv_timeout_s"] = 15.0
    cfg["spawn_stagger_s"] = 0.1
    cfg["reset"] = {"yaw_range": [0.0, 0.0], "joint_noise_rad": 0.0}
    cfg["faults_jsonl"] = str(Path("/tmp/sim2sim_test_faults.jsonl"))
    return cfg


class TestGodotVecEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _SKIP:
            raise unittest.SkipTest(_SKIP)
        from sim2sim.train.vec_env import GodotVecEnv

        cls.env = GodotVecEnv(_walk_cfg(), num_envs=2, device="cpu", seed=1, headless=True)

    @classmethod
    def tearDownClass(cls) -> None:
        env = getattr(cls, "env", None)
        if env is not None:
            env.close()

    def test_01_zero_action_rollout(self) -> None:
        import torch

        env = self.env
        obs0 = env.get_observations()
        self.assertEqual(tuple(obs0["actor"].shape), (2, 61))
        self.assertEqual(tuple(obs0["critic"].shape), (2, 70))
        self.assertEqual(obs0["actor"].dtype, torch.float32)
        self.assertEqual(obs0["critic"].dtype, torch.float32)
        self.assertEqual(obs0["actor"].device.type, "cpu")
        self.assertFalse(torch.isnan(obs0["actor"]).any())
        self.assertFalse(torch.isnan(obs0["critic"]).any())

        length0 = env.episode_length_buf.clone()
        zeros = torch.zeros(2, 14, dtype=torch.float32)
        last_contact = None
        for _ in range(30):
            obs, rew, dones, extras = env.step(zeros)
            self.assertEqual(tuple(obs["actor"].shape), (2, 61))
            self.assertEqual(tuple(obs["critic"].shape), (2, 70))
            self.assertFalse(torch.isnan(obs["actor"]).any())
            self.assertFalse(torch.isnan(obs["critic"]).any())
            self.assertFalse(torch.isnan(rew).any())
            self.assertEqual(tuple(rew.shape), (2,))
            self.assertEqual(tuple(dones.shape), (2,))
            self.assertIn("time_outs", extras)
            self.assertEqual(tuple(extras["time_outs"].shape), (2,))
            self.assertIn("log", extras)
            last_contact = obs["critic"][:, 64:66]
        self.assertTrue(torch.all(env.episode_length_buf >= length0))
        # No fall: lengths should have grown by 30 (or less if a timeout, which should not happen).
        self.assertTrue(torch.all(env.episode_length_buf > length0))
        self.assertIsNotNone(last_contact)
        self.assertTrue(bool(torch.all(last_contact > 0.5)), msg=f"feet contact={last_contact}")

    def test_02_forced_reset(self) -> None:
        import torch

        env = self.env
        zeros = torch.zeros(2, 14, dtype=torch.float32)
        env.step(zeros)
        env.step(zeros)
        self.assertGreater(int(env.episode_length_buf[0].item()), 0)
        env.reset_idx([0])
        self.assertEqual(int(env.episode_length_buf[0].item()), 0)
        self.assertGreaterEqual(int(env.episode_length_buf[1].item()), 0)
        obs = env.get_observations()
        self.assertFalse(torch.isnan(obs["actor"]).any())

    def test_03_fault_respawn(self) -> None:
        import torch

        env = self.env
        faults0 = int(env.faults)
        proc = env._workers[0]._proc
        proc.kill()
        proc.wait(timeout=5)
        zeros = torch.zeros(2, 14, dtype=torch.float32)
        obs, rew, dones, extras = env.step(zeros)
        self.assertEqual(int(env.faults), faults0 + 1)
        self.assertTrue(bool(extras["time_outs"][0]))
        self.assertTrue(bool(dones[0]))
        self.assertTrue(env._workers[0].is_alive())
        self.assertFalse(torch.isnan(obs["actor"]).any())

    def test_04_critic_prefix_stays_clean_when_actor_noisy(self) -> None:
        import torch

        env = self.env
        prev = env.noise_enabled
        env.noise_enabled = True
        try:
            zeros = torch.zeros(2, 14, dtype=torch.float32)
            obs, *_ = env.step(zeros)
            actor = obs["actor"]
            critic_prefix = obs["critic"][:, :61]
            delta = (actor - critic_prefix).abs().max().item()
            self.assertGreater(delta, 1e-4, msg="actor noise leaked into critic 61-D prefix")
        finally:
            env.noise_enabled = prev


if __name__ == "__main__":
    unittest.main(verbosity=2)
