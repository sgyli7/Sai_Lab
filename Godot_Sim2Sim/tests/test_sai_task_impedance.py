import unittest

import numpy as np

from sim2sim.sai_task_impedance import support_weights


class SupportWeightsTests(unittest.TestCase):
    def setUp(self):
        self.xy = np.array([[.12, .09], [.12, -.09], [-.12, .09], [-.12, -.09]])

    def test_symmetric_support_is_even(self):
        weights = support_weights(np.ones(4), self.xy)
        np.testing.assert_allclose(weights, .25, atol=2e-5)
        np.testing.assert_allclose(np.r_[weights.sum(), weights @ self.xy], [1., 0., 0.], atol=2e-5)

    def test_single_axle_keeps_finite_compressive_support(self):
        weights = support_weights(np.array([0., 0., 1., 1.]), self.xy)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(np.all(weights >= 0.))
        self.assertGreater(weights.sum(), .99)
        self.assertGreater(weights[2], .49)
        self.assertGreater(weights[3], .49)

    def test_unloaded_wheel_never_pulls(self):
        weights = support_weights(np.array([1., .05, .8, 0.]), self.xy)
        self.assertTrue(np.all(weights >= 0.))
        self.assertEqual(weights[3], 0.)
        self.assertLessEqual(weights.sum(), 1. + 1e-12)


if __name__ == "__main__":
    unittest.main()
