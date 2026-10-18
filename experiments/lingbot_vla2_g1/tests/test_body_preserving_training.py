"""CPU checks of native-mask weighting, padding, validation and gradients."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.lingbot_vla2_g1.body_preserving_training import (
    BODY_SLOTS, BodyPreservingRuntime, weighted_training_mask)
from experiments.lingbot_vla2_g1.contract import ACTION_MASK
from experiments.lingbot_vla2_g1.runtime import Runtime


def native_mask():
    return torch.as_tensor(np.broadcast_to(ACTION_MASK, (1, 50, 55)).copy(), dtype=torch.float32)


class BodyWeightTests(unittest.TestCase):
    def test_body_mapping_preserves_padding_unused_slots_and_original_mask(self):
        self.assertEqual(BODY_SLOTS, (30, 36, 37, 38))
        original = native_mask()
        original[:, -7:] = 0
        saved = original.clone()
        weighted = weighted_training_mask(original)
        self.assertTrue(torch.equal(original, saved))
        self.assertEqual(float(weighted.sum()), 43*44)
        self.assertTrue(torch.all(weighted[:, :43, list(BODY_SLOTS)] == 4))
        self.assertTrue(torch.all(weighted[:, -7:] == 0))
        self.assertTrue(torch.all(weighted[..., ~torch.as_tensor(ACTION_MASK)] == 0))
        self.assertNotEqual(weighted.data_ptr(), original.data_ptr())

    def test_invalid_shape_unused_slots_and_repeated_weighting_are_rejected(self):
        with self.assertRaises(ValueError):
            weighted_training_mask(torch.ones(50, 55))
        bad = native_mask()
        bad[..., 54] = 1
        with self.assertRaises(ValueError):
            weighted_training_mask(bad)
        with self.assertRaisesRegex(ValueError, 'double weighting'):
            weighted_training_mask(weighted_training_mask(native_mask()))

    def test_native_reduction_body_gradient_is_four_times_arm_and_unused_zero(self):
        parameter = torch.zeros(55, requires_grad=True)
        error = (parameter[None, None, :].expand(1, 50, 55) - 1).square()
        mask = weighted_training_mask(native_mask())
        loss = (error*mask).sum()/mask.sum()
        loss.backward()
        self.assertEqual(float(loss), 1.)
        for slot in BODY_SLOTS:
            self.assertAlmostEqual(float(parameter.grad[slot]/parameter.grad[0]), 4.)
        self.assertTrue(torch.all(parameter.grad[~torch.as_tensor(ACTION_MASK)] == 0))

    def test_validation_default_delegates_exact_original_runtime_loss(self):
        runtime = BodyPreservingRuntime.__new__(BodyPreservingRuntime)
        with patch.object(Runtime, 'loss', return_value=('native_total', 'native_flow')) as original, \
                patch('experiments.lingbot_vla2_g1.body_preserving_training.weighted_training_mask',
                      side_effect=AssertionError('validation must not reweight')):
            self.assertEqual(runtime.loss('rgb', 'q', 'a', 'mask'), ('native_total', 'native_flow'))
            original.assert_called_once_with('rgb', 'q', 'a', 'mask')
        self.assertIs(BodyPreservingRuntime.infer, Runtime.infer)

    def test_actual_CPU_training_step_receives_weighted_native_mask_without_mutating_inputs(self):
        class TinyNative(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.zeros(55))
                self.received = None

            def forward(self, actions, joint_mask):
                self.received = joint_mask.detach().clone()
                losses = (self.parameter[None, None, :] - actions).square()
                loss = (losses*joint_mask).sum()/joint_mask.sum()
                return loss, loss
        runtime = BodyPreservingRuntime.__new__(BodyPreservingRuntime)
        runtime.device = torch.device('cpu')
        runtime.model = TinyNative()
        inputs = {'actions': torch.ones(1, 50, 55), 'joint_mask': native_mask()}
        saved = inputs['joint_mask'].clone()
        runtime.inputs = lambda *args: dict(inputs)
        optimizer = torch.optim.SGD(runtime.model.parameters(), lr=.1)
        loss, _ = runtime.loss(None, None, None, None, train_weighted=True)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.equal(inputs['joint_mask'], saved))
        self.assertTrue(torch.all(runtime.model.received[..., list(BODY_SLOTS)] == 4))
        self.assertGreater(float(runtime.model.parameter[30]), float(runtime.model.parameter[0]))
        self.assertTrue(torch.all(runtime.model.parameter[~torch.as_tensor(ACTION_MASK)] == 0))
        self.assertFalse(torch.cuda.is_initialized())
