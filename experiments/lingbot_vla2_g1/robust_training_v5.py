"""R5 TRAIN sampling and synthetic proprioceptive measurement perturbation.

No live observation, command, normalization, or acceptance rule changes.
"""
import copy
import math
from pathlib import Path
import random

import numpy as np

from experiments.pi05_g1_mobile.carry_prefix_data import raw_png
from .scripted_success_data import FullScriptedSuccess

SOURCE_PROBABILITIES = {'original80TRAIN': .1, 'carry_prefixes': .1, 'full_scripted_TRAIN': .8}
PHASE_PROBABILITIES = {'approach_grasp': .30, 'transport': .45, 'placement': .25}
AUGMENTATION = {'eligible_families': ['carry_prefixes', 'full_scripted_TRAIN'],
                'probability': .5, 'arm_raw_std_rad': .04, 'arm_clip_rad': .12,
                'waist_raw_std_rad': .01, 'waist_clip_rad': .03,
                'finger_state_and_padding_unchanged': True,
                'stored_data_RGB_targets_masks_and_normalization_unchanged': True,
                'synthetic_TRAIN_measurement_noise_only': True}


class StatePerturbation:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.counts = {name: {'draws': 0, 'eligible': 0, 'augmented': 0} for name in SOURCE_PROBABILITIES}

    def apply(self, state, family):
        state = np.asarray(state)
        if state.shape != (32,) or state.dtype != np.float32 or not np.isfinite(state).all() or state[31] != 0:
            raise ValueError('Raw finite effective32 float32 state required')
        counters = self.counts[family]
        counters['draws'] += 1
        result = state.copy()
        if family == 'original80TRAIN':
            return result
        counters['eligible'] += 1
        if self.rng.random() < .5:
            result[:14] += np.clip(self.rng.normal(0, .04, 14), -.12, .12).astype(np.float32)
            result[28:31] += np.clip(self.rng.normal(0, .01, 3), -.03, .03).astype(np.float32)
            counters['augmented'] += 1
        return result


class FullTransportBalanced(FullScriptedSuccess):
    def sample(self, index):
        value = random.random()
        phase = 'approach_grasp' if value < .30 else 'transport' if value < .75 else 'placement'
        first, end = self.boundaries[phase]
        index = random.randrange(first, end)
        self.phase_draw_counts[phase] += 1
        stop = min(index + 50, len(self.actions)); count = stop - index
        targets = np.repeat(self.actions[stop-1:stop], 50, axis=0)
        targets[:count] = self.actions[index:stop]
        return raw_png(self.frames[index]), self.states[index], targets, np.arange(50) < count


class RobustTaskTrainingMixture:
    def __init__(self, original, carry, complete, seed=42):
        self.sources = [original, carry, complete]
        self.indices = range(sum(len(source.indices) for source in self.sources))
        self.episodes = [episode for source in self.sources for episode in source.episodes]
        self.draw_counts = {name: 0 for name in SOURCE_PROBABILITIES}
        self.perturbation = StatePerturbation(seed)

    def sample(self, index):
        value = random.random()
        number = 0 if value < .1 else 1 if value < .2 else 2
        family = list(SOURCE_PROBABILITIES)[number]
        source = self.sources[number]
        rgb, state, actions, mask = source.sample(random.randrange(len(source.indices)))
        self.draw_counts[family] += 1
        return rgb, self.perturbation.apply(state, family), actions, mask


def require_complete_source_validation(value):
    if (value.get('complete') is not True or value.get('episodes') != 10
            or value.get('chunks') != 190 or value.get('frames') != 9240 or value.get('noise_seed') != 42
            or not math.isfinite(value['flow_loss'])):
        raise ValueError('Complete fixed original10 validation required')


class GlobalSourceBest:
    """A worse R4 initialization cannot replace the inherited R2 source best."""
    def __init__(self, inherited_delta, validation):
        require_complete_source_validation(validation)
        self.inherited_delta = Path(inherited_delta).absolute()
        self.validation = copy.deepcopy(validation)
        self.step = 0

    def consider(self, step, validation):
        if type(step) is not int or step <= 0:
            raise ValueError('A real positive new optimizer step required')
        require_complete_source_validation(validation)
        if validation['flow_loss'] < self.validation['flow_loss']:
            self.validation = copy.deepcopy(validation)
            self.step = step
            return True
        return False

    def paths(self, output, final_step):
        if type(final_step) is not int or final_step <= 0 or self.step > final_step:
            raise ValueError('Positive actual final update boundary required')
        selected = self.inherited_delta if self.step == 0 else Path(output) / 'selected'
        final = selected if self.step == final_step else Path(output) / 'final'
        return selected, final
