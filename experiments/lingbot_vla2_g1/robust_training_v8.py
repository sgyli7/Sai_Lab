"""R8 preserves transport while sampling three audited TRAIN recoveries."""
import json
import random
from pathlib import Path

import numpy as np

from experiments.pi05_g1_mobile.carry_prefix_data import raw_png
from .data import sha
from .robust_training_v6 import StatePerturbation, GlobalSourceBest, AUGMENTATION
from .scripted_success_data import FullScriptedSuccess

SOURCE_PROBABILITIES = {'original80TRAIN': .1, 'carry_prefixes': .2,
                        'full_scripted_TRAIN': .25, 'recovery_scripted_TRAIN': .45}
RECOVERY_PHASE_PROBABILITIES = {'approach_grasp': .1, 'transport': .3,
                                'alignment': .3, 'lower_release': .3}
RECOVERY_FAMILY_PROBABILITIES = {'prior_R5_recovery04': 1/3, 'prior_R6_recovery08': 1/3, 'new_matched_R6_recovery09': 1/3}
PERMANENT_SOURCE_BEST = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z/training_task_round08_source_best')


def boundaries_from_events(declaration, events, grasp_boundary, length):
    version, prefix = declaration.get('collector_version'), declaration.get('initial_replay_controls')
    if (version, prefix) not in ((4, 1700), (8, 1800), (9, 1800)):
        raise ValueError('Only actual admitted recovery04/08/09 declarations are supported')
    actions = events[1:]
    if [e.get('control_tick') for e in actions] != list(range(0, length, 50)):
        raise ValueError('Every actual native query must be audited')
    if any(e.get('privileged_truth_used') is not (e['control_tick'] >= prefix) for e in actions):
        raise ValueError('Actual privileged query boundary differs from declaration')
    lowers = [e['control_tick'] for e in actions if e['phase'] == 'lower_held_box']
    releases = [e['control_tick'] for e in actions if e['phase'] == 'open_palms']
    if not lowers or not releases:
        raise ValueError('Actual alignment, lowering and release supervision required')
    lower, release = min(lowers), min(releases)
    if not 0 < grasp_boundary < prefix <= lower - 50 < release < length:
        raise ValueError('Positive independent alignment interval required')
    return {'approach_grasp': [0, grasp_boundary], 'transport': [grasp_boundary, prefix],
            'alignment': [prefix, lower], 'lower_release': [lower, length]}


class FullRecoveryBalanced(FullScriptedSuccess):
    def __init__(self, manifest, normalizer):
        super().__init__(manifest, normalizer)
        data = json.loads(Path(manifest).read_text())
        audit = json.loads(Path(data['independent_audit']).read_text())
        folder = Path(audit['result_path']).parent
        config, decoded = folder / 'config.json', folder / 'decoded.jsonl'
        for path in (config, decoded):
            if data['extra_frozen_files'].get(str(path)) != sha(path):
                raise ValueError('Phase boundaries must come from the independently audited collector')
        declaration = json.loads(config.read_text())['privileged_TRAIN_teacher']
        events = [json.loads(line) for line in decoded.read_text().splitlines()]
        self.boundaries = boundaries_from_events(declaration, events, self.boundaries['approach_grasp'][1], len(self.actions))
        self.phase_draw_counts = dict.fromkeys(RECOVERY_PHASE_PROBABILITIES, 0)

    def sample(self, index):
        value = random.random()
        phase = 'approach_grasp' if value < .1 else 'transport' if value < .4 else 'alignment' if value < .7 else 'lower_release'
        first, end = self.boundaries[phase]
        index = random.randrange(first, end)
        self.phase_draw_counts[phase] += 1
        stop = min(index + 50, len(self.actions)); count = stop - index
        targets = np.repeat(self.actions[stop-1:stop], 50, axis=0)
        targets[:count] = self.actions[index:stop]
        return raw_png(self.frames[index]), self.states[index], targets, np.arange(50) < count


class EqualRecoveryFamilies:
    def __init__(self, prior, intermediate, new):
        self.sources = [prior, intermediate, new]
        self.indices = range(sum(len(source.indices) for source in self.sources))
        self.episodes = [episode for source in self.sources for episode in source.episodes]
        self.family_draw_counts = dict.fromkeys(RECOVERY_FAMILY_PROBABILITIES, 0)

    @property
    def phase_draw_counts(self):
        return {name: sum(source.phase_draw_counts[name] for source in self.sources)
                for name in self.sources[0].phase_draw_counts}

    def sample(self, index):
        number = random.randrange(3)
        self.family_draw_counts[list(RECOVERY_FAMILY_PROBABILITIES)[number]] += 1
        source = self.sources[number]
        return source.sample(random.randrange(len(source.indices)))


class RecoveryTrainingMixture:
    def __init__(self, original, carry, complete, recovery, seed=42):
        self.sources = [original, carry, complete, recovery]
        self.indices = range(sum(len(source.indices) for source in self.sources))
        self.episodes = [episode for source in self.sources for episode in source.episodes]
        self.draw_counts = dict.fromkeys(SOURCE_PROBABILITIES, 0)
        self.perturbation = StatePerturbation(seed)

    def sample(self, index):
        value = random.random()
        number = 0 if value < .1 else 1 if value < .3 else 2 if value < .55 else 3
        family = list(SOURCE_PROBABILITIES)[number]
        source = self.sources[number]
        rgb, state, actions, mask = source.sample(random.randrange(len(source.indices)))
        self.draw_counts[family] += 1
        return rgb, self.perturbation.apply(state, family), actions, mask
