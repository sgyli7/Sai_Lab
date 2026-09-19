"""PolicyBundle: ONNX dims, sidecar, NaN (no synthetic onnx package)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim2sim.obs import DEFAULT_HOME, build_obs
from sim2sim.paths import policies_dir
from sim2sim.play_input import TwistLimits
from sim2sim.policy import (
    OnnxPolicy,
    PolicyBundle,
    PolicyNumericError,
    PolicyShapeError,
    expected_obs_dim,
)


def _walking_onnx() -> Path:
    return policies_dir() / "alpha_walking.onnx"


class TestPolicyBundle(unittest.TestCase):
    def test_expected_obs_matches_build_obs(self) -> None:
        home = DEFAULT_HOME
        self.assertEqual(expected_obs_dim(int(home.size)), 61)
        from sim2sim.backends import SimState

        st = SimState(
            t=0.0,
            q=home.astype(np.float64),
            qd=np.zeros(home.size, dtype=np.float64),
            base_pos=np.array([0.0, 0.0, 0.12]),
            base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            base_linvel=np.zeros(3),
            base_angvel_local=np.zeros(3),
        )
        obs = build_obs(st, np.zeros(home.size), np.zeros(13), home=home)
        self.assertEqual(int(obs.shape[-1]), expected_obs_dim(int(home.size)))

    def test_alias_and_graph_dims(self) -> None:
        onnx = _walking_onnx()
        if not onnx.is_file():
            self.skipTest(f"missing {onnx}")
        self.assertIs(OnnxPolicy, PolicyBundle)
        bundle = PolicyBundle(onnx)
        self.assertEqual(bundle.obs_dim, 61)
        self.assertEqual(bundle.act_dim, 14)
        bundle.check_dims(int(DEFAULT_HOME.size))
        with self.assertRaises(PolicyShapeError):
            bundle.check_dims(7)
        y = bundle.infer(np.zeros(61, dtype=np.float32))
        self.assertEqual(y.shape, (14,))
        self.assertTrue(np.isfinite(y).all())
        self.assertEqual(bundle.action_scale, 1.0)
        self.assertTrue(bundle.has_standing_partner)
        self.assertAlmostEqual(bundle.twist_limits.vmax_x, TwistLimits().vmax_x)

    def test_sidecar_twist_and_stand(self) -> None:
        onnx = _walking_onnx()
        if not onnx.is_file():
            self.skipTest(f"missing {onnx}")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            link = tmp / "walk.onnx"
            os.symlink(onnx.resolve(), link)
            (tmp / "walk.manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "action_scale": 1.0,
                        "sim2sim": {
                            "twist_limits": {"vmax_x": 0.4, "vmin_x": -0.3},
                            "use_stand_policy": False,
                        },
                    }
                )
            )
            bundle = PolicyBundle(link)
            self.assertIsNotNone(bundle.manifest)
            self.assertAlmostEqual(bundle.twist_limits.vmax_x, 0.4)
            self.assertAlmostEqual(bundle.twist_limits.vmin_x, -0.3)
            self.assertAlmostEqual(bundle.twist_limits.vmax_y, TwistLimits().vmax_y)
            self.assertFalse(bundle.has_standing_partner)
            bundle.check_dims(14)

    def test_nan_obs_raises(self) -> None:
        onnx = _walking_onnx()
        if not onnx.is_file():
            self.skipTest(f"missing {onnx}")
        bundle = PolicyBundle(onnx)
        obs = np.zeros(61, dtype=np.float32)
        obs[0] = np.nan
        with self.assertRaises(PolicyNumericError):
            bundle.infer(obs)
        obs = np.zeros(61, dtype=np.float32)
        obs[3] = np.inf
        with self.assertRaises(PolicyNumericError):
            bundle.infer(obs)


if __name__ == "__main__":
    unittest.main()
