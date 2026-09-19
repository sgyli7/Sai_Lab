"""Roller Godot must actually skate, not stand on glued wheels."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from sim2sim.paths import policies_dir, sim2sim_root
from sim2sim.policy import OnnxPolicy
from sim2sim.runner import load_robot_cfg, make_backend, run_rollout

ROLLER_ONNX = policies_dir() / "roller.onnx"
ROLLER_CFG = sim2sim_root() / "robots/microduck_roller.json"


class TestRollerGodotSkate(unittest.TestCase):
    @unittest.skipUnless(ROLLER_ONNX.is_file(), f"missing {ROLLER_ONNX}")
    def test_four_second_forward_covers_distance(self) -> None:
        cfg = dict(load_robot_cfg(ROLLER_CFG))
        cfg["schedule"] = [{"name": "fwd", "seconds": 4.0, "vel": [0.5, 0.0, 0.0]}]
        be = make_backend("godot", cfg, headless=True)
        try:
            hello = getattr(be, "_hello", {}) or {}
            self.assertIn("microduck_roller", str(hello.get("robot_scene", "")))
            self.assertIn("rollers", str(hello.get("window_title", "")))
            names = {b["name"] for b in json.loads(Path(cfg["godot_spec"]).read_text())["bodies"]}
            self.assertGreaterEqual(len(names & {"tire", "tire_2", "tire_3", "tire_4"}), 4)
            out = run_rollout(be, OnnxPolicy(ROLLER_ONNX), cfg)
        finally:
            be.close()
        pos = out["base_pos"]
        xy = float(np.linalg.norm(pos[-1, :2] - pos[0, :2]))
        zmin = float(pos[:, 2].min())
        self.assertGreater(zmin, 0.05, f"fell zmin={zmin:.3f}")
        self.assertGreater(xy, 0.8, f"wheels not rolling xy={xy:.3f}m")
