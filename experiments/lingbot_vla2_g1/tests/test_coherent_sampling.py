"""CPU numerical/protocol controls; none is a robot success result."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.lingbot_vla2_g1 import coherent_sampling as sampling
from experiments.lingbot_vla2_g1.data import ROOT, sha
from experiments.lingbot_vla2_g1.diagnostics.development_audit_v2 import sampling_evidence
from experiments.lingbot_vla2_g1.run_development_task_v5 import screen_eligible


class FakeModel:
    def __init__(self):
        self.noises = []

    def sample_actions(self, *, state, noise):
        self.noises.append(noise.clone())
        return noise.float() + state


class FakeNormalizer:
    def denormalize(self, value, kind):
        value = value.copy()
        value[:, 14:28] = 0
        return value


class Controls(unittest.TestCase):
    def test_two_actual_calls_are_antithetic_and_repeatable(self):
        model = FakeModel()
        runtime = SimpleNamespace(model=model, device=torch.device('cpu'), normalizer=FakeNormalizer(),
                                  inputs=lambda rgb, state: {'state': torch.tensor(float(np.mean(state)))})
        rgb = np.zeros((480, 640, 3), np.uint8)
        q = np.ones(32, np.float32) * .3
        first, predictions = sampling.infer(runtime, rgb, q)
        second, repeated = sampling.infer(runtime, rgb, q)
        self.assertEqual(len(model.noises), 4)
        self.assertTrue(torch.equal(model.noises[0], -model.noises[1]))
        self.assertTrue(torch.equal(model.noises[0], model.noises[2]))
        np.testing.assert_array_equal(first, predictions.mean(axis=0, dtype=np.float32))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(predictions, repeated)
        np.testing.assert_allclose(first[:, :14], .3, atol=1e-6)
        np.testing.assert_array_equal(first[:, 14:28], 0)

    def test_evidence_reconstructs_actual_mean_and_rejects_tampering(self):
        files = {'coherent_sampling_sha256': 'coherent_sampling.py', 'coherent_service_sha256': 'service_coherent_noise.py'}
        loaded = {'sampling_adapter': sampling.ADAPTER, 'common_noise_seed': sampling.SEED,
                  'predictions_per_query': 2, 'common_noise_float32_sha256': sampling.noise_identity()}
        config = {'frozen_files': {str(ROOT / 'experiments/lingbot_vla2_g1' / name): sha(ROOT / 'experiments/lingbot_vla2_g1' / name)
                                   for name in files.values()}}
        loaded.update({key: config['frozen_files'][str(ROOT / 'experiments/lingbot_vla2_g1' / name)] for key, name in files.items()})
        predictions = np.zeros((2, 50, 32), np.float32)
        predictions[0, :, :14] = 1
        predictions[1, :, :14] = -.5
        event = {'sampling_adapter': sampling.ADAPTER, 'common_noise_float32_sha256': sampling.noise_identity(),
                 'model_predictions32': predictions.tolist(), 'actions32': predictions.mean(0).tolist()}
        sampling_evidence(config, [loaded, event])
        bad = copy.deepcopy(event)
        bad['actions32'][0][0] += .001
        with self.assertRaisesRegex(ValueError, 'reconstruct'):
            sampling_evidence(config, [loaded, bad])
        with self.assertRaisesRegex(ValueError, 'identity'):
            sampling_evidence(config, [{**loaded, 'common_noise_seed': 43}, event])

    def test_early_source_drop_cannot_qualify_grasp_screen(self):
        value = {'grasped': True, 'fell': False, 'dropped': False, 'success': False}
        actual = {'physical_failure': True, 'termination_reason': 'source_object_dropped_predicate',
                  'completed_control_ticks': 738}
        self.assertFalse(screen_eligible(value, actual, {'ticks': 1000}))
        self.assertFalse(screen_eligible(value, {'completed_control_ticks': 999}, {'ticks': 1000}))
        self.assertTrue(screen_eligible(value, {'completed_control_ticks': 1000}, {'ticks': 1000}))
        self.assertTrue(screen_eligible({**value, 'success': True},
                        {'completed_control_ticks': 800, 'termination_reason': 'verified_task_success'}, {'ticks': 1000}))


if __name__ == '__main__':
    unittest.main()
