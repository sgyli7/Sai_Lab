"""Robot JSON paths expand from env, not hardcoded machine paths."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from sim2sim.paths import expand_cfg, sim2sim_root


class TestExpandCfg(unittest.TestCase):
    def test_env_placeholders(self) -> None:
        env = {
            "SIM2SIM_ROOT": "/tmp/sim2sim-root",
            "MICRODUCK_RL": "/tmp/rl",
            "MICRODUCK_POLICIES": "/tmp/pol",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = expand_cfg(
                {
                    "mjcf": "${MICRODUCK_RL}/scene.xml",
                    "godot_spec": "${SIM2SIM_ROOT}/godot/generated/microduck/robot_spec.json",
                    "policies": {"a": "${MICRODUCK_POLICIES}/a.onnx"},
                    "name": "microduck",
                }
            )
        self.assertEqual(cfg["mjcf"], "/tmp/rl/scene.xml")
        self.assertEqual(cfg["godot_spec"], "/tmp/sim2sim-root/godot/generated/microduck/robot_spec.json")
        self.assertEqual(cfg["policies"]["a"], "/tmp/pol/a.onnx")
        self.assertEqual(cfg["name"], "microduck")

    def test_committed_json_has_no_home_paths(self) -> None:
        root = sim2sim_root()
        text = (root / "robots/microduck.json").read_text()
        self.assertNotIn("/home/ethan", text)
        self.assertIn("${MICRODUCK_RL}", text)
        self.assertIn("${SIM2SIM_ROOT}", text)
        self.assertIn('"walk_godot": "${MICRODUCK_POLICIES}/Walk_Godot.onnx"', text)


if __name__ == "__main__":
    unittest.main()
