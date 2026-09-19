"""compare_pair hard-fail rules (no Godot)."""

from __future__ import annotations

import unittest

import numpy as np

from sim2sim.compare import compare_pair


def _traj(xy: float = 1.2, z: float = 0.12, steps: int = 20) -> dict:
    t = np.linspace(0.02, 0.02 * steps, steps)
    pos = np.zeros((steps, 3))
    pos[:, 0] = np.linspace(0.0, xy, steps)
    pos[:, 2] = z
    quat = np.zeros((steps, 4))
    quat[:, 0] = 1.0
    return {
        "t": t,
        "q": np.zeros((steps, 14)),
        "base_pos": pos,
        "base_quat": quat,
        "action": np.zeros((steps, 14)),
    }


class TestXyStall(unittest.TestCase):
    def test_freeze_shuffle_is_hard_fail(self) -> None:
        cmp = compare_pair(_traj(1.256), _traj(0.137), {})
        self.assertTrue(any("godot_xy_stalled" in x for x in cmp["hard_fail"]))

    def test_matched_walk_is_ok(self) -> None:
        cmp = compare_pair(_traj(1.256), _traj(1.147), {})
        self.assertEqual(cmp["hard_fail"], [])

    def test_length_mismatch_is_hard_fail(self) -> None:
        cmp = compare_pair(_traj(1.2, steps=20), _traj(1.2, steps=19), {})
        self.assertTrue(any("length_mismatch" in x for x in cmp["hard_fail"]))
        joined = " ".join(cmp["hard_fail"])
        self.assertIn("20", joined)
        self.assertIn("19", joined)


if __name__ == "__main__":
    unittest.main()
