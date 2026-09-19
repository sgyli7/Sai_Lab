"""A/B walk eval: yaw rotation math, synthetic metrics, Godot identical-policy Δ≈0."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim2sim.paths import policies_dir, sim2sim_root
from sim2sim.play_input import TwistLimits

ALPHA_ONNX = policies_dir() / "alpha_walking.onnx"


def _godot_ready() -> bool:
    cfg = sim2sim_root() / "godot/generated/microduck/robot_spec.json"
    return ALPHA_ONNX.is_file() and cfg.is_file()


class TestRotatePosesYaw(unittest.TestCase):
    def test_90deg_rotates_pos_and_left_multiplies_quat(self) -> None:
        from sim2sim.train.eval_walk import rotate_poses_yaw

        poses = [
            {
                "name": "trunk_base",
                "pos": [1.0, 0.0, 0.12],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "linvel": [0.2, 0.0, 0.0],
                "angvel": [0.0, 0.0, 0.1],
            }
        ]
        orig = [p["pos"][:] for p in poses]
        out = rotate_poses_yaw(poses, np.pi / 2)
        self.assertEqual(orig[0], [1.0, 0.0, 0.12])
        pos = np.asarray(out[0]["pos"], dtype=np.float64)
        np.testing.assert_allclose(pos, [0.0, 1.0, 0.12], atol=1e-9)
        quat = np.asarray(out[0]["quat"], dtype=np.float64)
        half = np.sqrt(0.5)
        np.testing.assert_allclose(quat, [half, 0.0, 0.0, half], atol=1e-9)
        lin = np.asarray(out[0]["linvel"], dtype=np.float64)
        np.testing.assert_allclose(lin, [0.0, 0.2, 0.0], atol=1e-9)

    def test_zero_yaw_is_identity(self) -> None:
        from sim2sim.train.eval_walk import rotate_poses_yaw

        poses = [{"name": "b", "pos": [0.3, -0.2, 0.1], "quat": [0.9, 0.1, 0.2, 0.3]}]
        q = np.asarray(poses[0]["quat"], dtype=np.float64)
        q = q / np.linalg.norm(q)
        poses[0]["quat"] = q.tolist()
        out = rotate_poses_yaw(poses, 0.0)
        np.testing.assert_allclose(out[0]["pos"], poses[0]["pos"], atol=1e-12)
        np.testing.assert_allclose(out[0]["quat"], q, atol=1e-12)


class TestEpisodeMetricsSynthetic(unittest.TestCase):
    def test_perfect_forward_track_zero_rmse(self) -> None:
        from sim2sim.train.eval_walk import episode_metrics

        n, dt = 100, 0.02
        pos = np.zeros((n, 3))
        pos[:, 0] = np.linspace(0.0, 0.25 * (n - 1) * dt, n)
        pos[:, 2] = 0.12
        quat = np.zeros((n, 4))
        quat[:, 0] = 1.0
        linvel = np.zeros((n, 3))
        linvel[:, 0] = 0.25
        angvel = np.zeros((n, 3))
        q = np.zeros((n, 14))
        q[:, 3] = np.sin(2 * np.pi * 2.5 * np.arange(n) * dt)
        action = np.zeros((n, 14))
        cmd = np.tile(np.array([0.25, 0.0, 0.0]), (n, 1))
        m = episode_metrics(
            t=np.arange(n) * dt,
            pos=pos,
            quat=quat,
            linvel=linvel,
            angvel=angvel,
            q=q,
            action=action,
            cmd_xyw=cmd,
            dt=dt,
            fell=False,
            survival_s=(n - 1) * dt,
        )
        self.assertLess(m["lin_vel_rmse"], 1e-6)
        self.assertLess(m["yaw_rate_rmse"], 1e-6)
        self.assertLess(m["mean_vel_err"], 1e-6)
        self.assertLess(m["mean_yaw_rate_err"], 1e-6)
        self.assertLess(m["vel_err_1s_rmse"], 1e-6)
        self.assertLess(m["yaw_err_1s_rmse"], 1e-6)
        self.assertAlmostEqual(m["mean_vx"], 0.25, places=6)
        self.assertAlmostEqual(m["mean_vy"], 0.0, places=6)
        self.assertAlmostEqual(m["mean_wz"], 0.0, places=6)
        self.assertLess(abs(m["yaw_drift_deg"]), 1e-6)
        self.assertAlmostEqual(m["distance"], 0.25 * (n - 1) * dt, places=6)
        self.assertFalse(m["fell"])
        self.assertAlmostEqual(m["survival_s"], (n - 1) * dt)
        self.assertEqual(m["mean_abs_daction"], 0.0)
        self.assertAlmostEqual(m["mean_trunk_z"], 0.12)
        self.assertGreater(m["cadence_hz_left_knee"], 2.0)
        self.assertLess(m["cadence_hz_left_knee"], 3.0)

    def test_gait_oscillation_hurts_instant_rmse_not_smoothed(self) -> None:
        from sim2sim.train.eval_walk import episode_metrics

        n, dt = 150, 0.02
        t = np.arange(n) * dt
        pos = np.zeros((n, 3))
        pos[:, 2] = 0.12
        quat = np.zeros((n, 4))
        quat[:, 0] = 1.0
        linvel = np.zeros((n, 3))
        linvel[:, 0] = 0.15 + 0.20 * np.sin(2 * np.pi * 2.5 * t)
        cmd = np.tile(np.array([0.15, 0.0, 0.0]), (n, 1))
        m = episode_metrics(
            t=t,
            pos=pos,
            quat=quat,
            linvel=linvel,
            angvel=np.zeros((n, 3)),
            q=np.zeros((n, 14)),
            action=np.zeros((n, 14)),
            cmd_xyw=cmd,
            dt=dt,
            fell=False,
            survival_s=t[-1],
        )
        self.assertGreater(m["lin_vel_rmse"], 0.10)
        self.assertLess(m["mean_vel_err"], 0.02)
        self.assertLess(m["vel_err_1s_rmse"], 0.02)
        self.assertAlmostEqual(m["mean_vx"], 0.15, places=2)

    def test_settle_window_drops_first_second(self) -> None:
        from sim2sim.train.eval_walk import episode_metrics

        n, dt = 100, 0.02
        linvel = np.zeros((n, 3))
        linvel[:50, 0] = 0.0
        linvel[50:, 0] = 0.25
        pos = np.zeros((n, 3))
        pos[:, 2] = 0.12
        quat = np.zeros((n, 4))
        quat[:, 0] = 1.0
        cmd = np.tile(np.array([0.25, 0.0, 0.0]), (n, 1))
        m = episode_metrics(
            t=np.arange(n) * dt,
            pos=pos,
            quat=quat,
            linvel=linvel,
            angvel=np.zeros((n, 3)),
            q=np.zeros((n, 14)),
            action=np.zeros((n, 14)),
            cmd_xyw=cmd,
            dt=dt,
            fell=False,
            survival_s=(n - 1) * dt,
        )
        self.assertLess(m["mean_vel_err"], 1e-6)
        self.assertAlmostEqual(m["mean_vx"], 0.25, places=6)
        self.assertGreater(m["lin_vel_rmse"], 0.10)

    def test_vel_err_1s_uses_per_step_command(self) -> None:
        from sim2sim.train.eval_walk import episode_metrics

        n, dt = 150, 0.02
        cmd = np.zeros((n, 3))
        cmd[:50, 0] = 0.10
        cmd[50:100, 0] = 0.20
        cmd[100:, 0] = 0.30
        linvel = np.zeros((n, 3))
        linvel[:, 0] = cmd[:, 0]
        pos = np.zeros((n, 3))
        pos[:, 2] = 0.12
        quat = np.zeros((n, 4))
        quat[:, 0] = 1.0
        m = episode_metrics(
            t=np.arange(n) * dt,
            pos=pos,
            quat=quat,
            linvel=linvel,
            angvel=np.zeros((n, 3)),
            q=np.zeros((n, 14)),
            action=np.zeros((n, 14)),
            cmd_xyw=cmd,
            dt=dt,
            fell=False,
            survival_s=(n - 1) * dt,
        )
        self.assertLess(m["lin_vel_rmse"], 1e-9)
        # Causal 1 s MA lags the step command, so smoothed RMSE is not zero.
        self.assertGreater(m["vel_err_1s_rmse"], 0.02)

    def test_yaw_drift_and_body_frame_strafe(self) -> None:
        from sim2sim.train.eval_walk import episode_metrics, yaw_quat_wxyz

        n, dt = 20, 0.02
        yaw0, yaw1 = 0.0, np.deg2rad(10.0)
        yaws = np.linspace(yaw0, yaw1, n)
        quat = np.stack([yaw_quat_wxyz(y) for y in yaws])
        pos = np.zeros((n, 3))
        pos[:, 2] = 0.11
        # World +Y motion while yaw=0 would be body +Y; here yaw grows but
        # linvel is given already in world frame as body-strafe equivalent at yaw=0.
        linvel = np.zeros((n, 3))
        linvel[:, 1] = 0.2
        angvel = np.zeros((n, 3))
        angvel[:, 2] = (yaw1 - yaw0) / ((n - 1) * dt)
        cmd = np.tile(np.array([0.0, 0.2, float(angvel[0, 2])]), (n, 1))
        m = episode_metrics(
            t=np.arange(n) * dt,
            pos=pos,
            quat=quat,
            linvel=linvel,
            angvel=angvel,
            q=np.zeros((n, 14)),
            action=np.linspace(0.0, 1.0, n * 14).reshape(n, 14),
            cmd_xyw=cmd,
            dt=dt,
            fell=True,
            survival_s=0.15,
        )
        self.assertAlmostEqual(m["yaw_drift_deg"], 10.0, places=5)
        self.assertTrue(m["fell"])
        self.assertGreater(m["mean_abs_daction"], 0.0)

    def test_eval_uses_default_twist_limits_not_sidecar(self) -> None:
        """A/B must share TwistLimits() so a sidecar cannot bias game_seq."""
        from sim2sim.train.eval_walk import CONDITIONS, EVAL_TWIST_LIMITS

        self.assertEqual(EVAL_TWIST_LIMITS, TwistLimits())
        prim = {c.name: c.primary for c in CONDITIONS}
        self.assertEqual(prim["walk_015"], "vel_err_1s_rmse")
        self.assertEqual(prim["turn_r"], "yaw_err_1s_rmse")
        self.assertEqual(prim["game_seq"], "vel_err_1s_rmse")
        self.assertEqual(prim["idle"], "yaw_drift_deg")
        self.assertEqual(prim["walk_push"], "fell")


class TestEvalWalkGodot(unittest.TestCase):
    @unittest.skipUnless(_godot_ready(), "alpha ONNX or godot spec missing")
    def test_identical_policies_delta_zero(self) -> None:
        from sim2sim.train.eval_walk import main as eval_main

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "eval"
            rc = eval_main(
                [
                    "--a",
                    str(ALPHA_ONNX),
                    "--b",
                    str(ALPHA_ONNX),
                    "--label-a",
                    "alpha",
                    "--label-b",
                    "alpha_copy",
                    "--seeds",
                    "1",
                    "--seconds",
                    "1",
                    "--workers",
                    "1",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(rc, 0)
            metrics = json.loads((out / "metrics.json").read_text())
            report = (out / "report.md").read_text()
            self.assertIn("VERDICT:", report)
            self.assertRegex(report, r"VERDICT: (improved|mixed|regressed)")
            rows = metrics["episodes"]
            by_key: dict[tuple, dict] = {}
            for ep in rows:
                by_key[(ep["condition"], ep["seed"], ep["label"])] = ep
            labels = {ep["label"] for ep in rows}
            self.assertEqual(labels, {"alpha", "alpha_copy"})
            a_eps = [ep for ep in rows if ep["label"] == "alpha"]
            from sim2sim.train.eval_walk import CONDITIONS

            primary = {c.name: c.primary for c in CONDITIONS}
            primary_tol = {
                "vel_err_1s_rmse": 0.02,
                "yaw_err_1s_rmse": 0.05,
                "yaw_drift_deg": 2.0,
                "fell": 0.0,
            }
            for ep in a_eps:
                other = by_key[(ep["condition"], ep["seed"], "alpha_copy")]
                self.assertAlmostEqual(ep["yaw0"], other["yaw0"], places=9, msg=f"{ep['condition']} yaw0")
                self.assertEqual(ep["fell"], other["fell"])
                self.assertAlmostEqual(ep["survival_s"], other["survival_s"], delta=0.05)
                met = primary[ep["condition"]]
                if met == "fell":
                    continue
                self.assertAlmostEqual(
                    ep[met], other[met], delta=primary_tol[met], msg=f"{ep['condition']} primary {met}"
                )
            self.assertEqual(metrics["verdict"], "mixed")


if __name__ == "__main__":
    unittest.main()
