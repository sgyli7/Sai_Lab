"""Real-file tests for ONNX → rsl_rl 5.0.1 actor recovery."""

from __future__ import annotations

import unittest
import os
from pathlib import Path

from sim2sim.paths import microduck_rl, policies_dir

ALPHA_ONNX = policies_dir() / "alpha_walking.onnx"
LOCAL_ONNX = policies_dir() / "local-ppo/local_velocity_walk_run_idle.onnx"
RSL_CKPT = Path(os.environ.get("SIM2SIM_TEST_CHECKPOINT") or
    microduck_rl() / "logs/rsl_rl/local_ppo_velocity/2026-09-02_01-47-57_local_walk_run_idle/model_2999.pt")

_EXPECTED_ACTOR_KEYS = {
    "obs_normalizer._mean",
    "obs_normalizer._var",
    "obs_normalizer._std",
    "obs_normalizer.count",
    "distribution.std_param",
    "mlp.0.weight",
    "mlp.0.bias",
    "mlp.2.weight",
    "mlp.2.bias",
    "mlp.4.weight",
    "mlp.4.bias",
    "mlp.6.weight",
    "mlp.6.bias",
}


def _train_extra_installed() -> bool:
    try:
        import onnx  # noqa: F401
        import rsl_rl  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


_SKIP_ALPHA = (not _train_extra_installed()) or (not ALPHA_ONNX.is_file())
_SKIP_CKPT = (not _train_extra_installed()) or (not RSL_CKPT.is_file()) or (not LOCAL_ONNX.is_file())


class TestOnnxImportAlpha(unittest.TestCase):
    @unittest.skipIf(_SKIP_ALPHA, "alpha_walking.onnx or train extra missing")
    def test_parse_recovers_mlp_layout(self) -> None:
        from sim2sim.train.onnx_import import parse_mlp_onnx

        rec = parse_mlp_onnx(ALPHA_ONNX)
        self.assertEqual(rec.obs_dim, 61)
        self.assertEqual(rec.act_dim, 14)
        self.assertEqual(tuple(rec.hidden_dims), (512, 256, 128))
        self.assertEqual(rec.activation, "elu")
        self.assertEqual(rec.mean.shape, (61,))
        self.assertEqual(rec.std.shape, (61,))
        self.assertEqual(len(rec.layers), 4)
        self.assertIn("joint_names", rec.metadata)

    @unittest.skipIf(_SKIP_ALPHA, "alpha_walking.onnx or train extra missing")
    def test_state_dict_keys_and_eps_compensation(self) -> None:
        import torch

        from sim2sim.train.onnx_import import build_actor_state_dict, parse_mlp_onnx

        rec = parse_mlp_onnx(ALPHA_ONNX)
        sd = build_actor_state_dict(rec, init_std=0.25, count=1_000_000_000)
        self.assertEqual(set(sd.keys()), _EXPECTED_ACTOR_KEYS)
        self.assertEqual(tuple(sd["distribution.std_param"].shape), (14,))
        self.assertTrue(torch.allclose(sd["distribution.std_param"], torch.full((14,), 0.25)))
        # rsl_rl forward is (x-mean)/(_std+eps); ONNX Div uses rec.std.
        eps = 1e-2
        recovered_div = sd["obs_normalizer._std"].reshape(-1) + eps
        self.assertTrue(torch.allclose(recovered_div, torch.as_tensor(rec.std), atol=1e-7))
        self.assertTrue(
            torch.allclose(sd["obs_normalizer._std"], torch.sqrt(sd["obs_normalizer._var"]), atol=1e-7)
        )
        self.assertGreaterEqual(int(sd["obs_normalizer.count"].item()), 1_000_000_000)

    @unittest.skipIf(_SKIP_ALPHA, "alpha_walking.onnx or train extra missing")
    def test_alpha_parity_under_1e_5(self) -> None:
        from sim2sim.train.onnx_import import parse_mlp_onnx, verify_parity

        rec = parse_mlp_onnx(ALPHA_ONNX)
        err = verify_parity(rec, ALPHA_ONNX, n=10000, seed=0)
        self.assertLess(err, 1e-5, f"alpha_walking parity max_abs_err={err}")

    @unittest.skipIf(_SKIP_ALPHA, "alpha_walking.onnx or train extra missing")
    def test_zero_variance_std_is_clamped(self) -> None:
        from sim2sim.train.onnx_import import PARITY_FAIL_ABS, build_actor_state_dict, parse_mlp_onnx, verify_parity

        sit = policies_dir() / "alpha_sitstand.onnx"
        if not sit.is_file():
            self.skipTest("alpha_sitstand.onnx missing")
        rec = parse_mlp_onnx(sit)
        sd = build_actor_state_dict(rec, init_std=0.18, count=1_000_000_000)
        self.assertTrue((sd["obs_normalizer._std"] >= 0).all())
        err = verify_parity(rec, sit, n=10000, seed=0)
        self.assertLess(err, PARITY_FAIL_ABS, f"sitstand parity max_abs_err={err}")


class TestNativeCheckpoint(unittest.TestCase):
    @unittest.skipIf(_SKIP_CKPT, "model_2999.pt / local-ppo ONNX or train extra missing")
    def test_checkpoint_actor_strict_load(self) -> None:
        from sim2sim.train.onnx_import import load_rsl_checkpoint_actor

        sd, meta = load_rsl_checkpoint_actor(RSL_CKPT)
        self.assertEqual(set(sd.keys()), _EXPECTED_ACTOR_KEYS)
        self.assertEqual(tuple(meta["hidden_dims"]), (512, 256, 128))
        self.assertEqual(meta["obs_dim"], 61)
        self.assertEqual(meta["act_dim"], 14)

    @unittest.skipIf(_SKIP_CKPT, "model_2999.pt / local-ppo ONNX or train extra missing")
    def test_checkpoint_actor_parity_vs_local_onnx(self) -> None:
        from sim2sim.train.onnx_import import load_rsl_checkpoint_actor, make_rsl_actor_from_state_dict, verify_parity

        sd, meta = load_rsl_checkpoint_actor(RSL_CKPT)
        actor = make_rsl_actor_from_state_dict(sd, meta, device="cpu")
        err = verify_parity(actor, LOCAL_ONNX, n=10000, seed=0)
        self.assertLess(err, 1e-5, f"model_2999 vs local-ppo ONNX max_abs_err={err}")


if __name__ == "__main__":
    unittest.main()
