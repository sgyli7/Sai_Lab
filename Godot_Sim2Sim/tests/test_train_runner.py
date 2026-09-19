"""Train runner: train_cfg builder, actor freeze, Godot 2-iter save/resume."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sim2sim.godot_proc import godot_bin
from sim2sim.paths import policies_dir, sim2sim_root

ROOT = sim2sim_root()
YAML = ROOT / "configs/walk_godot.yaml"
CFG_PATH = ROOT / "robots/microduck.json"
ALPHA_ONNX = policies_dir() / "alpha_walking.onnx"


def _train_extra_installed() -> bool:
    try:
        import rsl_rl  # noqa: F401
        import tensordict  # noqa: F401
        import torch  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        return False
    return True


def _skip_godot() -> str | None:
    if not _train_extra_installed():
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
    if not ALPHA_ONNX.is_file():
        return f"missing {ALPHA_ONNX}"
    return None


_SKIP = not _train_extra_installed()
_SKIP_GODOT = _skip_godot()


@unittest.skipIf(_SKIP, "train extra not installed")
class TestTrainCfgBuilder(unittest.TestCase):
    def test_expected_keys_and_architecture(self) -> None:
        from sim2sim.train.runner import build_train_cfg

        ppo = {
            "learning_rate": 3.0e-4,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "entropy_coef": 0.001,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "init_std": 0.25,
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
        }
        cfg = build_train_cfg(
            ppo,
            num_envs=8,
            samples_per_iter=2048,
            save_interval=50,
            seed=0,
            run_name="unit",
            max_iterations=10,
        )
        self.assertEqual(cfg["obs_groups"], {"actor": ["actor"], "critic": ["critic"]})
        self.assertEqual(cfg["actor"]["hidden_dims"], [512, 256, 128])
        self.assertEqual(cfg["actor"]["activation"], "elu")
        self.assertTrue(cfg["actor"]["obs_normalization"])
        self.assertEqual(cfg["actor"]["class_name"], "MLPModel")
        self.assertEqual(cfg["actor"]["distribution_cfg"]["class_name"], "GaussianDistribution")
        self.assertEqual(cfg["actor"]["distribution_cfg"]["init_std"], 0.25)
        self.assertEqual(cfg["actor"]["distribution_cfg"]["std_type"], "scalar")
        self.assertEqual(cfg["critic"]["hidden_dims"], [512, 256, 128])
        self.assertTrue(cfg["critic"]["obs_normalization"])
        self.assertNotIn("distribution_cfg", cfg["critic"])
        self.assertEqual(cfg["algorithm"]["learning_rate"], 3.0e-4)
        self.assertEqual(cfg["algorithm"]["entropy_coef"], 0.001)
        self.assertEqual(cfg["algorithm"]["desired_kl"], 0.01)
        self.assertEqual(cfg["algorithm"]["schedule"], "adaptive")
        self.assertEqual(cfg["algorithm"]["num_learning_epochs"], 5)
        self.assertEqual(cfg["algorithm"]["num_mini_batches"], 4)
        self.assertIn("PPOFinetune", cfg["algorithm"]["class_name"])
        self.assertEqual(cfg["num_steps_per_env"], 256)
        self.assertEqual(cfg["save_interval"], 50)
        self.assertEqual(cfg["logger"], "tensorboard")
        self.assertIsNone(cfg["clip_actions"])

    def test_num_steps_clamped(self) -> None:
        from sim2sim.train.runner import STEPS_HI, STEPS_LO, choose_num_steps_per_env

        self.assertEqual(choose_num_steps_per_env(2, 10), STEPS_LO)
        self.assertEqual(choose_num_steps_per_env(2, 48), 24)
        self.assertEqual(choose_num_steps_per_env(8, 1536), 192)
        self.assertEqual(choose_num_steps_per_env(2, 100_000), STEPS_HI)


@unittest.skipIf(_SKIP, "train extra not installed")
class TestFreeze(unittest.TestCase):
    def test_actor_frozen_critic_updates(self) -> None:
        import torch
        from rsl_rl.models import MLPModel
        from tensordict import TensorDict

        from sim2sim.train.runner import set_actor_trainable

        obs = TensorDict(
            {"actor": torch.zeros(8, 61), "critic": torch.zeros(8, 70)},
            batch_size=[8],
        )
        groups = {"actor": ["actor"], "critic": ["critic"]}
        dist = {
            "class_name": "GaussianDistribution",
            "init_std": 0.25,
            "std_type": "scalar",
        }
        actor = MLPModel(
            obs, groups, "actor", 14, hidden_dims=[32, 16], activation="elu",
            obs_normalization=True, distribution_cfg=dict(dist),
        )
        critic = MLPModel(
            obs, groups, "critic", 1, hidden_dims=[32, 16], activation="elu", obs_normalization=True
        )
        set_actor_trainable(actor, False)
        opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-2)
        actor_before = {k: v.detach().clone() for k, v in actor.state_dict().items() if v.dtype.is_floating_point}
        critic_before = {k: v.detach().clone() for k, v in critic.state_dict().items() if v.dtype.is_floating_point}

        batch = TensorDict(
            {"actor": torch.randn(8, 61), "critic": torch.randn(8, 70)},
            batch_size=[8],
        )
        loss = actor(batch).pow(2).mean() + critic(batch).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        for k, v in actor.state_dict().items():
            if k not in actor_before:
                continue
            self.assertTrue(torch.equal(v.cpu(), actor_before[k].cpu()), msg=f"actor {k} changed")
        changed = False
        for k, v in critic.state_dict().items():
            if k not in critic_before or not v.dtype.is_floating_point:
                continue
            if not torch.equal(v.cpu(), critic_before[k].cpu()):
                changed = True
                break
        self.assertTrue(changed, "critic parameters did not change")

    def test_normalizer_frozen_mean_unchanged(self) -> None:
        import torch
        from rsl_rl.models import MLPModel
        from tensordict import TensorDict

        from sim2sim.train.runner import freeze_actor_normalizer, snapshot_normalizer_mean

        obs = TensorDict({"actor": torch.zeros(4, 61)}, batch_size=[4])
        dist = {"class_name": "GaussianDistribution", "init_std": 0.25, "std_type": "scalar"}
        actor = MLPModel(
            obs,
            {"actor": ["actor"]},
            "actor",
            14,
            hidden_dims=[32, 16],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=dict(dist),
        )
        actor.train()
        freeze_actor_normalizer(actor)
        mean0 = snapshot_normalizer_mean(actor)
        batch = TensorDict({"actor": torch.randn(32, 61)}, batch_size=[32])
        actor.update_normalization(batch)
        self.assertTrue(torch.equal(actor.obs_normalizer._mean.detach().cpu(), mean0))
        # Unfrozen control: a fresh normalizer must move.
        actor2 = MLPModel(
            obs,
            {"actor": ["actor"]},
            "actor",
            14,
            hidden_dims=[32, 16],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=dict(dist),
        )
        actor2.train()
        mean_b = actor2.obs_normalizer._mean.detach().clone()
        actor2.update_normalization(batch)
        self.assertFalse(torch.equal(actor2.obs_normalizer._mean.detach().cpu(), mean_b.cpu()))


@unittest.skipIf(_SKIP_GODOT is not None, _SKIP_GODOT or "godot skip")
class TestGodotTrainRunner(unittest.TestCase):
    def test_two_iter_checkpoint_and_resume(self) -> None:
        import torch

        from sim2sim.train.runner import train
        from sim2sim.train.vec_env import load_walk_cfg

        cfg = load_walk_cfg(YAML)
        cfg["obs_noise"]["enabled"] = False
        cfg["pushes"]["enabled"] = False
        cfg["recv_timeout_s"] = 20.0
        cfg["spawn_stagger_s"] = 0.1
        cfg["ppo"]["critic_warmup_iters"] = 0
        with tempfile.TemporaryDirectory(prefix="walk_godot_runner_") as tmp:
            tmp_p = Path(tmp)
            cfg.setdefault("export", {})["onnx"] = str(tmp_p / "test_walk.onnx")
            run1 = train(
                cfg,
                init_onnx=str(ALPHA_ONNX),
                num_envs=2,
                max_iterations=2,
                run_name="unit2",
                device="cpu",
                seed=1,
                log_root=tmp_p / "logs",
                samples_per_iter=48,
                critic_warmup_iters=0,
                save_interval=50,
            )
            ckpt = run1 / "model_1.pt"
            self.assertTrue(ckpt.is_file(), f"missing {ckpt}; dir={list(run1.iterdir())}")
            payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            self.assertEqual(int(payload["iter"]), 1)
            self.assertIn("actor_state_dict", payload)
            self.assertIn("critic_state_dict", payload)
            self.assertIn("optimizer_state_dict", payload)
            init_check = json.loads((run1 / "params" / "init_check.json").read_text())
            self.assertLess(float(init_check["max_abs_err"]), 1e-5)
            self.assertTrue(init_check["ok"])

            run2 = train(
                cfg,
                resume=ckpt,
                num_envs=2,
                max_iterations=1,
                run_name="unit2_resume",
                device="cpu",
                seed=1,
                log_root=tmp_p / "logs",
                samples_per_iter=48,
                critic_warmup_iters=0,
            )
            ckpt2 = run2 / "model_2.pt"
            self.assertTrue(ckpt2.is_file(), f"missing {ckpt2}; dir={list(run2.iterdir())}")
            payload2 = torch.load(str(ckpt2), map_location="cpu", weights_only=False)
            self.assertEqual(int(payload2["iter"]), 2)
            self.assertGreater(int(payload2["iter"]), int(payload["iter"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
