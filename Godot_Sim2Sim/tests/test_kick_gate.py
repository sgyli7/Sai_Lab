"""Kick gate soft/known-fail semantics (no Godot / no ONNX)."""

from __future__ import annotations

import unittest

import numpy as np

from sim2sim.skill_metrics import classify_kick_pair, fall_timeline, pair_rmse


def _traj(*, z: float = 0.12, tilt_hack: bool = False, steps: int = 50, action_peak: float = 1.0) -> dict:
    t = np.linspace(0.02, 0.02 * steps, steps)
    pos = np.zeros((steps, 3))
    pos[:, 2] = z
    if tilt_hack:
        # force gate fall via z
        pos[30:, 2] = 0.03
    quat = np.zeros((steps, 4))
    quat[:, 0] = 1.0
    q = np.zeros((steps, 14))
    q[:, 2] = np.linspace(0, 0.5, steps)
    act = np.zeros((steps, 14))
    act[10, 0] = action_peak
    return {
        "t": t,
        "q": q,
        "base_pos": pos,
        "base_quat": quat,
        "action": act,
    }


class TestKickGateSemantics(unittest.TestCase):
    def test_soft_godot_fall_is_known_fail(self) -> None:
        mj = fall_timeline(_traj(z=0.12))
        gd = fall_timeline(_traj(tilt_hack=True))
        self.assertFalse(mj["fell"])
        self.assertTrue(gd["fell"])
        gate = classify_kick_pair(mj, gd, known_fail=True)
        self.assertEqual(gate["status"], "KNOWN_FAIL")
        self.assertTrue(gate["known_fail"])
        self.assertEqual(gate["hard_fail"], [])

    def test_hard_mode_promotes_to_hard_fail(self) -> None:
        mj = fall_timeline(_traj(z=0.12))
        gd = fall_timeline(_traj(tilt_hack=True))
        gate = classify_kick_pair(mj, gd, known_fail=False)
        self.assertEqual(gate["status"], "HARD_FAIL")
        self.assertTrue(any("godot_fell" in x for x in gate["hard_fail"]))

    def test_mujoco_fall_is_always_hard(self) -> None:
        mj = fall_timeline(_traj(tilt_hack=True))
        gd = fall_timeline(_traj(z=0.12))
        gate = classify_kick_pair(mj, gd, known_fail=True)
        self.assertEqual(gate["status"], "HARD_FAIL")
        self.assertTrue(any("mujoco_fell" in x for x in gate["hard_fail"]))

    def test_both_stand_is_pass(self) -> None:
        mj = fall_timeline(_traj(z=0.12))
        gd = fall_timeline(_traj(z=0.12))
        gate = classify_kick_pair(mj, gd, known_fail=True)
        self.assertEqual(gate["status"], "PASS")

    def test_pair_rmse_and_action_peak(self) -> None:
        a = _traj(action_peak=1.5)
        b = _traj(action_peak=1.5)
        b["q"] = a["q"] + 0.1
        info = fall_timeline(a)
        self.assertAlmostEqual(info["action_peak"], 1.5)
        rmse = pair_rmse(a, b)
        self.assertGreater(rmse["q_rmse"], 0.05)


if __name__ == "__main__":
    unittest.main()
