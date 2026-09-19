"""Shared fallen / tilt predicate."""

from __future__ import annotations

import math
import unittest

import numpy as np

from sim2sim.fall import FallCriteria, fallen, fallen_mask, tilt_deg


def _quat_rx(deg: float) -> np.ndarray:
    a = math.radians(deg) * 0.5
    return np.array([math.cos(a), math.sin(a), 0.0, 0.0], dtype=np.float64)


class TestFall(unittest.TestCase):
    def test_defaults(self) -> None:
        c = FallCriteria()
        self.assertEqual(c.tilt_deg, 70.0)
        self.assertEqual(c.min_z, 0.055)

    def test_upright_high(self) -> None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(tilt_deg(q), 0.0, places=5)
        self.assertFalse(fallen(q, np.array([0.0, 0.0, 0.12])))

    def test_low_z(self) -> None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertTrue(fallen(q, np.array([0.0, 0.0, 0.05])))
        self.assertFalse(fallen(q, np.array([0.0, 0.0, 0.055])))

    def test_tilt_threshold(self) -> None:
        q69 = _quat_rx(69.0)
        q71 = _quat_rx(71.0)
        self.assertAlmostEqual(tilt_deg(q69), 69.0, places=4)
        self.assertAlmostEqual(tilt_deg(q71), 71.0, places=4)
        pos = np.array([0.0, 0.0, 0.12])
        self.assertFalse(fallen(q69, pos))
        self.assertTrue(fallen(q71, pos))
        self.assertTrue(fallen(_quat_rx(90.0), pos))

    def test_fallen_mask_matches_scalar(self) -> None:
        from sim2sim.coords import quat_rotate_inverse_wxyz

        down = np.array([0.0, 0.0, -1.0])
        quats = np.stack([_quat_rx(0.0), _quat_rx(71.0), _quat_rx(0.0)])
        pos = np.array([[0.0, 0.0, 0.12], [0.0, 0.0, 0.12], [0.0, 0.0, 0.04]])
        grav = np.stack([quat_rotate_inverse_wxyz(q, down) for q in quats])
        mask = fallen_mask(grav, pos)
        expected = np.array([fallen(q, p) for q, p in zip(quats, pos)])
        np.testing.assert_array_equal(mask, expected)


if __name__ == "__main__":
    unittest.main()
