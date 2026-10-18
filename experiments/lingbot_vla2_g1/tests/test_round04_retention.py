"""Real CPU optimizer/save/reload exercises for the previously discarded final."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file

from experiments.lingbot_vla2_g1 import training_task_round04 as training
from experiments.lingbot_vla2_g1.data import OWN


class Source:
    def __init__(self):
        self.indices = range(2); self.episodes = [{'id': 'CPU_toy_not_physical_data'}]
        self.phase_draw_counts = {'approach_grasp': 0, 'transport': 0, 'placement': 0}
        self.family_draw_counts = {'native201': 0, 'fresh_native2026101702': 0}
    def sample(self, index):
        return None, None, None, None


class ToyRuntime:
    def __init__(self, **kwargs):
        self.model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad(): self.model.weight.fill_(.25)
        self.model.config = SimpleNamespace(bias_update_speed=0)
        self.parameter_elements = self.trainable_elements = 1
    def loss(self, *args):
        loss = (self.model.weight - 1).square().mean()
        return loss, loss


def metric(flow):
    return {'flow_loss': flow, 'total_loss': flow, 'complete': True,
            'frames': 9240, 'chunks': 190, 'episodes': 10, 'noise_seed': 42}


class FinalRetentionTests(unittest.TestCase):
    def run_recipe(self, directory, final_flow):
        args = SimpleNamespace(output=Path(directory) / 'output', hours=.15, max_steps=1,
                               norm_stats=OWN / 'train_meanstd.json', initial_delta=OWN / 'training_round02/selected',
                               carry_prefix_manifest=None, new_carry_manifest=None, full_teacher_manifest=None)
        source = Source()
        # The actual optimizer and safetensors write/reload run on a real CPU
        # parameter. Only data, CUDA synchronization and expensive validation are
        # substituted; this proves checkpoint retention, not robot performance.
        with ExitStack() as stack:
            for name, value in {
                'admission': lambda args: {}, 'EpisodeDataset': lambda *a: source,
                'EqualCarryFamilies': lambda *a: source,
                'GraspBalancedPrefix': lambda *a: source, 'BalancedNewCarryPrefix': lambda *a: source,
                'FullScriptedSuccess': lambda *a: source, 'Runtime': ToyRuntime,
                'memory_snapshot': lambda *a: {'cuda_initialized': False},
            }.items(): stack.enter_context(patch.object(training, name, value))
            stack.enter_context(patch.object(training, 'validate', side_effect=[metric(.1), metric(.5), metric(final_flow), metric(.4)]))
            stack.enter_context(patch.object(torch.cuda, 'manual_seed_all', lambda *a: None))
            stack.enter_context(patch.object(torch.cuda, 'synchronize', lambda *a: None))
            return training.train(args)

    def test_worse_original_validation_still_saves_updated_final(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_recipe(directory, .2)
            self.assertEqual(result['selected_step'], 0)
            self.assertEqual(result['selected_delta'], str(OWN / 'training_round02/selected'))
            self.assertTrue(result['final_weights_saved'])
            self.assertEqual(result['physical_candidate_step'], 1)
            final = Path(result['final_delta'])
            self.assertGreater(float(load_file(str(final / 'trainable.safetensors'))['weight'][0, 0]), .25)
            self.assertFalse((final.parent / 'selected').exists())
            self.assertFalse(result['overall_robot_goal_completed'])

    def test_final_best_has_one_payload_and_two_explicit_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_recipe(directory, .05)
            self.assertEqual(result['selected_step'], 1)
            self.assertEqual(result['selected_delta'], result['final_delta'])
            self.assertEqual(len(list(Path(directory).rglob('trainable.safetensors'))), 1)
            metadata = json.loads((Path(result['final_delta']) / 'metadata.json').read_text())
            self.assertEqual(metadata['step'], 1)

    def test_failed_teacher_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps({'schema': 'lingbot_privileged_physical_full_success_TRAIN_episode_v1',
                                       'task_success_teacher': True, 'physical_outcome': {'success': False},
                                       'source_autonomous_model_policy': False, 'counts_as_LingBot_success': False}))
            with patch.object(training, 'admit_carry', lambda args: {}):
                with self.assertRaisesRegex(ValueError, 'successful physical TRAIN'):
                    training.admission(SimpleNamespace(full_teacher_manifest=path))


if __name__ == '__main__':
    unittest.main()
