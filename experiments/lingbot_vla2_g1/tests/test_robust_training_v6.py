"""Synthetic CPU regression: robustness copies and real optimizer/save flow."""
import json
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.lingbot_vla2_g1 import robust_training_v6 as robust
from experiments.lingbot_vla2_g1 import training_task_round06 as training
from experiments.lingbot_vla2_g1.data import sha


def validation(loss):
    return {'complete': True, 'episodes': 10, 'chunks': 190, 'frames': 9240,
            'noise_seed': 42, 'flow_loss': loss, 'total_loss': loss}


class TinySource:
    def __init__(self, *args, **kwargs):
        self.indices = range(8)
        self.episodes = [{'synthetic': True}]
        self.rgb = np.zeros((2, 2, 3), np.uint8)
        self.states = np.zeros((8, 32), np.float32)
        self.actions = np.zeros((8, 32), np.float32)
        self.mask = np.ones(50, bool)
        self.targets = np.zeros((50, 32), np.float32)
        self.phase_draw_counts = {'approach_and_grasp': 0, 'carry_remainder': 0}

    def sample(self, index):
        self.phase_draw_counts['approach_and_grasp'] += 1
        return self.rgb, self.states[index % 8], self.targets, self.mask


class TinyComplete(TinySource):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.phase_draw_counts = {'approach_grasp': 0, 'transport': 0, 'placement': 0}

    def sample(self, index):
        self.phase_draw_counts['transport'] += 1
        return self.rgb, self.states[index % 8], self.targets, self.mask


class TinyRuntime:
    def __init__(self, *args, **kwargs):
        self.model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(1.)
        self.model.config = SimpleNamespace(bias_update_speed=0)
        self.parameter_elements = self.trainable_elements = 1

    def loss(self, *args):
        value = self.model.weight.square().sum()
        return value, value


class RobustnessTests(unittest.TestCase):
    def test_seeded_noise_changes_only_declared_state_copy(self):
        state = np.arange(32, dtype=np.float32) / 100
        state[-1] = 0
        saved = state.copy()
        first, second = robust.StatePerturbation(42), robust.StatePerturbation(42)
        augmented = 0
        for _ in range(100):
            a = first.apply(state, 'full_scripted_TRAIN')
            b = second.apply(state, 'full_scripted_TRAIN')
            np.testing.assert_array_equal(a, b)
            self.assertFalse(np.shares_memory(a, state))
            np.testing.assert_array_equal(a[14:28], state[14:28])
            self.assertEqual(a[-1], 0)
            self.assertLessEqual(float(abs(a[:14] - state[:14]).max()), .120001)
            self.assertLessEqual(float(abs(a[28:31] - state[28:31]).max()), .030001)
            augmented += int(not np.array_equal(a, state))
            np.testing.assert_array_equal(state, saved)
        self.assertEqual(augmented, first.counts['full_scripted_TRAIN']['augmented'])
        self.assertGreater(augmented, 25)
        self.assertLess(augmented, 75)
        np.testing.assert_array_equal(first.apply(state, 'original80TRAIN'), saved)
        self.assertEqual(first.counts['original80TRAIN']['eligible'], 0)

    def test_mixture_preserves_pixels_targets_masks_and_reconciles_draws(self):
        sources = [TinySource(), TinySource(), TinyComplete(), TinyComplete()]
        mixture = robust.RecoveryTrainingMixture(*sources)
        random.seed(42)
        for _ in range(1000):
            rgb, _state, target, mask = mixture.sample(0)
            np.testing.assert_array_equal(rgb, sources[0].rgb)
            np.testing.assert_array_equal(target, sources[0].targets)
            np.testing.assert_array_equal(mask, sources[0].mask)
        self.assertEqual(sum(mixture.draw_counts.values()), 1000)
        for name, count in mixture.draw_counts.items():
            self.assertEqual(count, mixture.perturbation.counts[name]['draws'])
        self.assertGreater(mixture.draw_counts['full_scripted_TRAIN'], 250)
        self.assertGreater(mixture.draw_counts['recovery_scripted_TRAIN'], 450)
        self.assertLess(mixture.draw_counts['full_scripted_TRAIN'], 350)
        self.assertLess(mixture.draw_counts['recovery_scripted_TRAIN'], 550)
        self.assertEqual(mixture.perturbation.counts['original80TRAIN']['augmented'], 0)

    def test_successful_TRAIN_phase_sampler_uses_new_probabilities(self):
        sampler = robust.FullRecoveryBalanced.__new__(robust.FullRecoveryBalanced)
        sampler.states = np.zeros((300, 32), np.float32)
        sampler.actions = np.repeat(np.arange(300, dtype=np.float32)[:, None], 32, axis=1)
        sampler.frames = list(range(300))
        sampler.boundaries = {'approach_grasp': [0, 100], 'transport': [100, 200], 'placement': [200, 300]}
        sampler.phase_draw_counts = dict.fromkeys(sampler.boundaries, 0)
        random.seed(42)
        with patch.object(robust, 'raw_png', side_effect=lambda frame: frame):
            for _ in range(3000):
                frame, state, target, mask = sampler.sample(0)
                self.assertEqual(target[0, 0], frame)
                self.assertTrue(mask[0])
        for name, fraction in robust.RECOVERY_PHASE_PROBABILITIES.items():
            self.assertLess(abs(sampler.phase_draw_counts[name] / 3000 - fraction), .03)

    def test_real_CPU_optimizer_always_saves_new_final_and_truthful_selected_role(self):
        for improves in (False, True, 'intermediate'):
            with self.subTest(improves=improves), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ('norm.json', 'split.json', 'bf16_conversion_receipt.json',
                             'contract.json', 'runtime_overlay_receipt.json'):
                    (root / name).write_text('{}')
                args = SimpleNamespace(output=root / 'output', initial_delta=root / 'R4/final',
                    norm_stats=root / 'norm.json', hours=.55, max_steps=101 if improves == 'intermediate' else 3,
                    carry_prefix_manifest=root / 'carry.json', new_carry_manifest=root / 'new.json',
                    full_teacher_manifest=root / 'full.json', recovery_teacher_manifest=root / 'recovery.json', supplementary_manifest=None)
                provenance = {'synthetic': True, 'initial_total_optimizer_updates': 1800,
                    'initial_validation': validation(.167), 'source_best_reference_delta': str(root / 'R2/selected'),
                    'source_best_reference_validation': validation(.165)}
                source_calls = 0
                def checked(runtime, dataset, *unused):
                    nonlocal source_calls
                    if isinstance(dataset, training.DiagnosticTRAINView):
                        return {'synthetic': True, 'complete': True, 'flow_loss': .1}
                    source_calls += 1
                    if source_calls == 1:
                        return validation(.167)
                    return validation(.16 if improves and (improves != 'intermediate' or source_calls == 2) else .168)
                with patch.object(training, 'OWN', root), \
                     patch.object(training, 'PERMANENT_SOURCE_BEST', root / 'permanent/source_best'), \
                     patch.object(training, 'SPLIT', root / 'split.json'), \
                     patch.object(training, 'SPLIT_SHA', sha(root / 'split.json')), \
                     patch.object(training, 'admission', return_value=provenance), \
                     patch.object(training, 'MeanStd', return_value=None), \
                     patch.object(training, 'EpisodeDataset', TinySource), \
                     patch.object(training, 'GraspBalancedPrefix', TinySource), \
                     patch.object(training, 'BalancedNewCarryPrefix', TinySource), \
                     patch.object(training, 'FullTransportBalanced', TinyComplete), \
                     patch.object(training, 'FullRecoveryBalanced', TinyComplete), \
                     patch.object(training, 'Runtime', TinyRuntime), \
                     patch.object(training, 'validate', side_effect=checked), \
                     patch.object(training, 'memory_snapshot', return_value={}), \
                     patch.object(torch.cuda, 'manual_seed_all'), patch.object(torch.cuda, 'synchronize'):
                    result = training.train(args)
                self.assertEqual(result['optimizer_steps'], args.max_steps)
                self.assertEqual(result['model_weight_lineage_optimizer_updates'], 1800 + args.max_steps)
                self.assertFalse(result['overall_robot_goal_completed'])
                self.assertTrue(result['synthetic'])
                final = Path(result['final_delta'])
                self.assertTrue((final / 'trainable.safetensors').is_file())
                metadata = json.loads((final / 'metadata.json').read_text())
                self.assertEqual(metadata['step'], args.max_steps)
                self.assertEqual(metadata['model_weight_lineage_optimizer_updates'], 1800 + args.max_steps)
                if improves == 'intermediate':
                    self.assertEqual(result['selected_step'], 100)
                    self.assertEqual(result['selected_delta'], str(root / 'permanent/source_best'))
                    self.assertEqual(result['final_delta'], str(root / 'output/final'))
                    self.assertTrue((root / 'permanent/source_best/trainable.safetensors').is_file())
                    self.assertFalse((root / 'output/selected').exists())
                elif improves:
                    self.assertEqual(result['selected_delta'], str(final))
                    self.assertEqual(result['selected_step'], 3)
                else:
                    self.assertEqual(result['selected_delta'], str(root / 'R2/selected'))
                    self.assertNotEqual(final, args.initial_delta)
                    self.assertEqual(result['selected_step'], 0)

    def test_distinct_source_best_storage_keeps_actual_final_separate(self):
        source = robust.GlobalSourceBest('/synthetic/R2', validation(.165), '/synthetic/permanent/source_best')
        self.assertTrue(source.consider(100, validation(.164)))
        self.assertEqual(source.paths('/synthetic/R6', 240),
                         (Path('/synthetic/permanent/source_best'), Path('/synthetic/R6/final')))
        self.assertTrue(source.consider(240, validation(.163)))
        self.assertEqual(source.paths('/synthetic/R6', 240),
                         (Path('/synthetic/R6/selected'), Path('/synthetic/R6/selected')))


if __name__ == '__main__':
    unittest.main()
