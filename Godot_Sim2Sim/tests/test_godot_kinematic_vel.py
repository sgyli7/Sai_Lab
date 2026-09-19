"""Godot kinematic ω/qd: idle is quiet; moving hinge matches Δq/dt."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from sim2sim.backends.godot_backend import GodotBackend
from sim2sim.backends.mujoco_backend import MujocoBackend
from sim2sim.obs import DEFAULT_HOME
from sim2sim.paths import sim2sim_root
from sim2sim.runner import apply_home_qpos, load_robot_cfg

ROOT = sim2sim_root()
CFG_PATH = ROOT / "robots/microduck.json"
HEAD_YAW = 7
PHYS_DT = 0.005


class TestGodotKinematicVel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg = load_robot_cfg(CFG_PATH)
        spec = Path(cfg["godot_spec"])
        if not spec.is_file():
            raise unittest.SkipTest(f"missing spec {spec}")
        cls.cfg = cfg
        cls.home = np.asarray(cfg.get("home", DEFAULT_HOME), dtype=np.float64)
        mj = MujocoBackend(Path(cfg["mjcf"]), timestep=cfg.get("timestep", 0.005), current_limit_a=0.0)
        apply_home_qpos(mj, cls.home, z=float(cfg.get("reset_z", 0.125)))
        cls.home_poses = mj.body_poses_mujoco()
        mj.close()
        cls.be = GodotBackend(
            spec,
            headless=True,
            current_limit_a=float(cfg.get("current_limit_a", 1.75)),
            recv_timeout=30.0,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        be = getattr(cls, "be", None)
        if be is not None:
            be.close()

    def _reset_home(self, *, pin_base: bool = False):
        return self.be.reset(ctrl=self.home, bodies=self.home_poses, pin_base=pin_base)

    def test_01_idle_home_hold_quiet(self) -> None:
        # Pure HOME PD without a policy falls (C2). Pin the trunk so the
        # assertion tests kinematic ω/qd, not balance.
        self._reset_home(pin_base=True)
        angs = []
        qds = []
        for _ in range(100):
            st = self.be.step(self.home, n_substeps=4)
            angs.append(np.asarray(st.base_angvel_local, dtype=np.float64))
            qds.append(np.asarray(st.qd, dtype=np.float64))
        # Skip contact settle; same window as the idle-phase defect (steps 20–100).
        ang = np.stack(angs)[20:]
        qd = np.stack(qds)[20:]
        mean_ang = np.mean(ang, axis=0)
        mean_abs_qd = np.mean(np.abs(qd), axis=0)
        for i, v in enumerate(mean_ang):
            self.assertLess(
                abs(float(v)),
                0.02,
                f"mean base_angvel_local[{i}]={v:.4f} (want |mean|<0.02)",
            )
        for i, v in enumerate(mean_abs_qd):
            lim = 0.03 if i in (0, 7, 9) else 0.05
            self.assertLess(
                float(v),
                lim,
                f"mean |qd[{i}]|={v:.4f} (want <{lim})",
            )

    def test_02_qd_matches_joint_angle_fd(self) -> None:
        self._reset_home()
        qs = []
        qds = []
        for i in range(80):
            t = i * PHYS_DT
            ctrl = self.home.copy()
            ctrl[HEAD_YAW] = self.home[HEAD_YAW] + 0.35 * np.sin(2.0 * np.pi * 1.5 * t)
            st = self.be.step(ctrl, n_substeps=1)
            qs.append(float(st.q[HEAD_YAW]))
            qds.append(float(st.qd[HEAD_YAW]))
        q = np.asarray(qs)
        qd = np.asarray(qds)
        fd = np.zeros_like(q)
        fd[1:] = (np.unwrap(q)[1:] - np.unwrap(q)[:-1]) / PHYS_DT
        sl = slice(10, None)
        if float(np.std(fd[sl])) < 0.05 or float(np.std(qd[sl])) < 0.05:
            self.fail(f"sinusoid segment not moving: std(fd)={np.std(fd[sl]):.4f} std(qd)={np.std(qd[sl]):.4f}")
        corr = float(np.corrcoef(qd[sl], fd[sl])[0, 1])
        self.assertGreater(corr, 0.9, f"qd vs Δq/dt corr={corr:.4f} (want >0.9)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
