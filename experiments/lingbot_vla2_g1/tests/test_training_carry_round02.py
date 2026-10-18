"""CPU optimizer loop: restore origin, original-only validation, fallback selection."""
import contextlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from experiments.lingbot_vla2_g1 import training_carry_round02 as t
from experiments.pi05_g1_mobile import carry_prefix_data as c


class Samples:
    indices = [0]
    episodes = [{'id': 0}]

    def sample(self, index):
        return None, None, None, None


class Runtime:
    def __init__(self, **kwargs):
        self.arguments = kwargs
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(1.)
        self.model.config = types.SimpleNamespace(bias_update_speed=0.)
        self.parameter_elements = self.trainable_elements = 1

    def loss(self, *args):
        loss = self.model.weight.square().sum()
        return loss, loss


def metric(loss):
    return {'total_loss': loss, 'flow_loss': loss, 'frames': 9240,
            'chunks': 190, 'episodes': 10, 'complete': True, 'noise_seed': 42}


class TrainingTests(unittest.TestCase):
    def exercise(self, last):
        with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
            root = Path(folder)
            args = types.SimpleNamespace(output=root / 'run', hours=.2, max_steps=1,
                    norm_stats=t.OWN / 'train_meanstd.json', supplementary_manifest=None,
                    initial_delta=root / 'initial', carry_prefix_manifest=root / 'prefix')
            original, validation, prefix = Samples(), Samples(), Samples()
            stack.enter_context(patch.object(t, 'admit', return_value={
                'initial_delta': str(args.initial_delta), 'optimizer_resume': False,
                'carry_prefix_train_only': True, 'task_success_teacher': False}))
            datasets = stack.enter_context(patch.object(t, 'EpisodeDataset',
                                                        side_effect=[original, validation]))
            stack.enter_context(patch.object(c, 'CarryPrefix', return_value=prefix))
            runtimes = []
            def runtime(**kwargs):
                value = Runtime(**kwargs)
                runtimes.append(value)
                return value
            stack.enter_context(patch.object(t, 'Runtime', side_effect=runtime))
            valid = stack.enter_context(patch.object(t, 'validate',
                                                      side_effect=[metric(1.), last]))
            stack.enter_context(patch.object(t, 'memory_snapshot', return_value={}))
            stack.enter_context(patch.object(t.torch.cuda, 'manual_seed_all'))
            stack.enter_context(patch.object(t.torch.cuda, 'synchronize'))
            result = t.train(args)
            self.assertEqual(runtimes[0].arguments['delta'], args.initial_delta)
            self.assertEqual([call.args[2] for call in datasets.call_args_list],
                             ['train', 'validation'])
            self.assertTrue(all(call.args[1] is validation for call in valid.call_args_list))
            self.assertEqual(sum(result['training_draw_counts'].values()), 8)
            self.assertEqual(result['optimizer_steps'], 1)
            self.assertFalse(result['optimizer_resume'])
            self.assertLess(float(runtimes[0].model.weight.detach()), 1.)
            return result, str(args.initial_delta)

    def test_improvement_selects_new_parameters(self):
        result, _ = self.exercise(metric(.5))
        self.assertEqual(result['selected_step'], 1)
        self.assertTrue(result['selected_delta'].endswith('/selected'))

    def test_no_improvement_keeps_initial_checkpoint(self):
        result, initial = self.exercise(metric(1.1))
        self.assertEqual(result['selected_step'], 0)
        self.assertEqual(result['selected_delta'], initial)

    def test_incomplete_final_validation_cannot_select_new_checkpoint(self):
        result, initial = self.exercise(None)
        self.assertEqual(result['selected_delta'], initial)
        self.assertFalse(result['final_validation_complete'])


if __name__ == '__main__':
    unittest.main()
