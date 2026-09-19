"""Godot project path must be the repo godot/, not site-packages."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from sim2sim.backends.godot_backend import _robot_user_args
from sim2sim.godot_proc import GODOT_PROJECT, sim2sim_root, _headless_core


class TestSim2simRoot(unittest.TestCase):
    def test_workers_cannot_expand_the_supervisor_cpu_affinity(self):
        with patch('os.sched_getaffinity',return_value={16,17,18,19}):
            self.assertEqual({_headless_core() for _ in range(8)},{16,17,18,19})

    def test_godot_project_is_repo_godot(self) -> None:
        root = sim2sim_root()
        self.assertTrue((root / "robots" / "microduck.json").is_file(), root)
        self.assertEqual(GODOT_PROJECT.resolve(), (root / "godot").resolve())
        self.assertTrue((GODOT_PROJECT / "main.tscn").is_file(), GODOT_PROJECT)

    def test_roller_spec_is_passed_to_godot(self) -> None:
        spec = GODOT_PROJECT / "generated/microduck_roller/robot_spec.json"
        if not spec.is_file():
            self.skipTest(f"generated roller spec missing: {spec}")
        args = _robot_user_args(spec)
        joined = " ".join(args)
        self.assertIn("generated/microduck_roller/robot_spec.json", joined)
        self.assertIn("generated/microduck_roller/robot.tscn", joined)


if __name__ == "__main__":
    unittest.main()
