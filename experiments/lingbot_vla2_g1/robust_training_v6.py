"""R6 adds audited recovery demonstrations while retaining R5 augmentation."""
import copy
import random
from pathlib import Path

import numpy as np

from experiments.pi05_g1_mobile.carry_prefix_data import raw_png
from .robust_training_v5 import (StatePerturbation as PreviousPerturbation,
    GlobalSourceBest as PreviousSourceBest, AUGMENTATION as PREVIOUS_AUGMENTATION)
from .scripted_success_data import FullScriptedSuccess


SOURCE_PROBABILITIES = {'original80TRAIN': .1, 'carry_prefixes': .1,
                        'full_scripted_TRAIN': .3, 'recovery_scripted_TRAIN': .5}
RECOVERY_PHASE_PROBABILITIES = {'approach_grasp': .10, 'transport': .10, 'placement': .80}
AUGMENTATION = copy.deepcopy(PREVIOUS_AUGMENTATION)
AUGMENTATION['eligible_families'].append('recovery_scripted_TRAIN')
PERMANENT_SOURCE_BEST = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z/training_task_round06_source_best')


class GlobalSourceBest(PreviousSourceBest):
    def __init__(self, inherited_delta, validation, permanent_source_best):
        super().__init__(inherited_delta, validation)
        self.permanent_source_best = Path(permanent_source_best).absolute()

    def paths(self, output, final_step):
        selected, final = super().paths(output, final_step)
        if 0 < self.step < final_step:
            selected = self.permanent_source_best
        return selected, final


class StatePerturbation(PreviousPerturbation):
    def __init__(self, seed=42):
        super().__init__(seed)
        self.counts['recovery_scripted_TRAIN'] = {'draws': 0, 'eligible': 0, 'augmented': 0}


class FullRecoveryBalanced(FullScriptedSuccess):
    def __init__(self, manifest, normalizer):
        super().__init__(manifest, normalizer)
        if not self.boundaries['transport'][0] < 1700 < len(self.actions):
            raise ValueError('Actual physically successful recovery tail after1700 required')
        self.boundaries = copy.deepcopy(self.boundaries)
        self.boundaries['transport'][1] = 1700
        self.boundaries['placement'] = [1700, len(self.actions)]

    def sample(self, index):
        value = random.random()
        phase = 'approach_grasp' if value < .1 else 'transport' if value < .2 else 'placement'
        first, end = self.boundaries[phase]
        index = random.randrange(first, end)
        self.phase_draw_counts[phase] += 1
        stop = min(index + 50, len(self.actions)); count = stop - index
        targets = np.repeat(self.actions[stop-1:stop], 50, axis=0)
        targets[:count] = self.actions[index:stop]
        return raw_png(self.frames[index]), self.states[index], targets, np.arange(50) < count


class RecoveryTrainingMixture:
    def __init__(self, original, carry, complete, recovery, seed=42):
        self.sources = [original, carry, complete, recovery]
        self.indices = range(sum(len(source.indices) for source in self.sources))
        self.episodes = [episode for source in self.sources for episode in source.episodes]
        self.draw_counts = {name: 0 for name in SOURCE_PROBABILITIES}
        self.perturbation = StatePerturbation(seed)

    def sample(self, index):
        value = random.random()
        number = 0 if value < .1 else 1 if value < .2 else 2 if value < .5 else 3
        family = list(SOURCE_PROBABILITIES)[number]
        source = self.sources[number]
        rgb, state, actions, mask = source.sample(random.randrange(len(source.indices)))
        self.draw_counts[family] += 1
        return rgb, self.perturbation.apply(state, family), actions, mask
