"""The inverse must preserve varying channels and use only exact TRAIN constants."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from experiments.lingbot_vla2_g1.constant_inverse import PointMassMeanStd
from experiments.lingbot_vla2_g1.data import MeanStd, OWN


class ConstantInverseTests(unittest.TestCase):
    def test_real_frozen_train_constant_channels(self):
        old = MeanStd(OWN / 'train_meanstd.json')
        new = PointMassMeanStd(OWN / 'train_meanstd.json')
        inputs = np.ones((50, 32), np.float32)
        prior = old.denormalize(inputs, 'actions')
        current = new.denormalize(inputs, 'actions')
        np.testing.assert_array_equal(np.flatnonzero(new.constant), np.arange(14, 28))
        self.assertGreater(float(abs(prior[:, new.constant]).max()), .9)
        np.testing.assert_array_equal(current[:, new.constant], 0)
        np.testing.assert_array_equal(current[:, ~new.constant], prior[:, ~new.constant])

    def test_state_and_input_normalization_unchanged(self):
        old = MeanStd(OWN / 'train_meanstd.json')
        new = PointMassMeanStd(OWN / 'train_meanstd.json')
        values = np.arange(64, dtype=np.float32).reshape(2, 32) / 100
        for kind in ('state', 'actions'):
            np.testing.assert_array_equal(new.normalize(values, kind), old.normalize(values, kind))
        np.testing.assert_array_equal(new.denormalize(values, 'state'), old.denormalize(values, 'state'))

    def test_low_variance_or_nonconstant_range_is_not_frozen(self):
        doc = json.loads((OWN / 'train_meanstd.json').read_text())
        doc['actions']['raw_std'][14] = 1e-12
        doc['actions']['max'][15] = 1e-12
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stats.json'
            path.write_text(json.dumps(doc))
            normalizer = PointMassMeanStd(path)
            self.assertFalse(normalizer.constant[14])
            self.assertFalse(normalizer.constant[15])
            self.assertTrue(normalizer.constant[16])


if __name__ == '__main__':
    unittest.main()
