import contextlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.lingbot_vla2_g1.data import MeanStd
from experiments.lingbot_vla2_g1.training import validate


class NormalizationTests(unittest.TestCase):
    def test_shared_std_and_local_scale_have_identical_units(self):
        values = np.arange(64, dtype=np.float32).reshape(2, 32) / 9
        mean = np.linspace(-1, 1, 32).tolist()
        std = [0., *np.linspace(.1, 3., 31).tolist()]
        with tempfile.TemporaryDirectory() as root:
            norms = []
            for key in ['std', 'scale']:
                path = Path(root) / (key + '.json')
                path.write_text(json.dumps({k: {'mean': mean, key: std} for k in ['state', 'actions']}))
                norm = MeanStd(path)
                normalized = norm.normalize(values, 'actions')
                np.testing.assert_allclose(norm.denormalize(normalized, 'actions'), values, atol=1e-6)
                norms.append(normalized)
            np.testing.assert_array_equal(*norms)

    def test_nonfinite_or_negative_norm_cannot_reach_model(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'norm.json'
            for bad in [float('nan'), -1., float('inf')]:
                path.write_text(json.dumps({k: {'mean': [0.] * 32, 'std': [bad] + [1.] * 31} for k in ['state', 'actions']}))
                with self.assertRaises(ValueError):
                    MeanStd(path)


class ValidationTests(unittest.TestCase):
    class Dataset:
        episodes = [None]

        def __init__(self, loss, count):
            self.loss, self.count = loss, count

        def validation_indices(self):
            return [0, 1]

        def sample(self, index):
            mask = np.arange(50) < (50 if index == 0 else self.count)
            return None, None, self.loss, mask

    class Runtime:
        model = torch.nn.Linear(1, 1)

        def loss(self, rgb, state, action, mask):
            return torch.tensor(action + .2), torch.tensor(action)

    def test_full_sources_equal_weight_independent_of_source_length(self):
        runtime = self.Runtime()
        with patch('torch.cuda.current_device', return_value=0), patch('torch.random.fork_rng', return_value=contextlib.nullcontext()), patch('torch.cuda.manual_seed_all'):
            result = validate(runtime, {'original': self.Dataset(2., 3), 'supplementary': self.Dataset(8., 47)}, time.monotonic() + 30, lambda: False)
        self.assertAlmostEqual(result['flow_loss'], 5.)
        self.assertAlmostEqual(result['total_loss'], 5.2, places=5)
        self.assertEqual(result['frames'], 150)
        self.assertEqual(result['chunks'], 4)
        self.assertTrue(result['complete'])
        self.assertTrue(runtime.model.training)

    def test_partial_validation_has_no_selection_metric(self):
        with patch('torch.cuda.current_device', return_value=0), patch('torch.random.fork_rng', return_value=contextlib.nullcontext()):
            result = validate(self.Runtime(), self.Dataset(2., 3), time.monotonic() - 1, lambda: False)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
