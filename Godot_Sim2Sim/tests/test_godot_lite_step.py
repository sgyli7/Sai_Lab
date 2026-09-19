"""Headless Godot: ctrl_len error, lite step feet, send/recv split."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from sim2sim.backends.godot_backend import GodotBackend
from sim2sim.obs import DEFAULT_HOME
from sim2sim.runner import load_robot_cfg
from sim2sim.paths import sim2sim_root

ROOT = sim2sim_root()
CFG_PATH = ROOT / "robots/microduck.json"


class TestGodotLiteStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg = load_robot_cfg(CFG_PATH)
        spec = Path(cfg["godot_spec"])
        if not spec.is_file():
            raise unittest.SkipTest(f"missing spec {spec}")
        cls.cfg = cfg
        cls.home = np.asarray(cfg.get("home", DEFAULT_HOME), dtype=np.float64)
        cls.be = GodotBackend(spec, headless=True, current_limit_a=0.0, recv_timeout=30.0)

    @classmethod
    def tearDownClass(cls) -> None:
        be = getattr(cls, "be", None)
        if be is not None:
            be.close()

    def test_01_alive_and_ctrl_len(self) -> None:
        self.assertTrue(self.be.is_alive())
        st0 = self.be.reset(ctrl=self.home)
        t0 = float(st0.t)
        msg = self.be._client.call({"cmd": "step", "ctrl": [0.0, 0.0], "n_substeps": 4})
        self.assertFalse(msg.get("ok"))
        self.assertEqual(msg.get("cmd"), "step")
        self.assertEqual(msg.get("err"), "ctrl_len")
        self.assertEqual(int(msg.get("got")), 2)
        self.assertEqual(int(msg.get("want")), int(self.be.nu))
        st1 = self.be.step(self.home, n_substeps=1)
        self.assertGreater(float(st1.t), t0)

    def test_02_lite_step_feet_no_dump(self) -> None:
        self.be.reset(ctrl=self.home)
        self.be.send_step(self.home, n_substeps=4, report="lite")
        st = self.be.recv_step()
        raw = st.extra.get("raw") or {}
        self.assertNotIn("dump", raw)
        self.assertNotIn("wheels", raw)
        feet = st.extra.get("feet")
        self.assertIsInstance(feet, list)
        self.assertEqual(len(feet), 2)
        names = {f["name"] for f in feet}
        self.assertEqual(names, {"ankle_left", "ankle_right"})
        for f in feet:
            self.assertIn("contact", f)
            self.assertIn("n_contacts", f)
            self.assertIn("impulse", f)
            self.assertEqual(len(f["pos"]), 3)
            self.assertEqual(len(f["linvel"]), 3)
        self.assertEqual(len(raw["q"]), int(self.be.nu))
        self.assertIn("qd", raw)
        self.assertIn("tau", raw)
        self.assertIn("base_pos", raw)
        self.assertIn("base_quat", raw)
        self.assertIn("base_linvel", raw)
        self.assertIn("base_angvel_local", raw)
        for fat in (
            "dump",
            "wheels",
            "axis_dot",
            "body_names",
            "dbg_ang_world",
            "base_angvel_jolt",
            "applied",
            "missing",
            "tile_dy",
            "held",
            "held_order",
            "taps",
            "time_scale",
            "sole_n_on",
        ):
            self.assertNotIn(fat, raw, fat)

    def test_03_default_step_still_has_dump(self) -> None:
        self.be.reset(ctrl=self.home)
        st = self.be.step(self.home, n_substeps=1)
        raw = st.extra.get("raw") or {}
        self.assertIn("dump", raw)
        self.assertGreater(len(raw["dump"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
