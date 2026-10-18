"""TRAIN-only phase-balanced sampling of the already admitted native201 prefix.

The boundary is the source's sampled bilateral-contact/lift event plus2seconds;
this label is used only for offline training sampling, never policy observation.
"""
import json
from pathlib import Path
import random

from experiments.pi05_g1_mobile.carry_prefix_data import CarryPrefix
from .data import sha

PREFIX_SHA = '3dab70a1de3b387197fabbbf73cb83af1db0eebe4e6c146f6d7577238d00b9df'


def phase_identity(manifest):
    path = Path(manifest)
    data = json.loads(path.read_text())
    source = data['evidence']['result']
    if sha(path) != PREFIX_SHA or sha(source['path']) != source['sha256']:
        raise ValueError('Original TRAIN-prefix/source identity changed')
    result = json.loads(Path(source['path']).read_text())
    first = result['acceptance']['first_grasp_control_tick']
    boundary = first + 100
    if first != 152 or data['length'] != 1396 or not 0 < boundary < data['length']:
        raise ValueError('Fixed TRAIN-only contact/lift phase boundary differs')
    return {'schema': 'lingbot_TRAIN_prefix_phase_sampling_v1', 'prefix_manifest_sha256': PREFIX_SHA,
            'first_contact_lift_tick': first, 'approach_and_stabilization_stop_exclusive': boundary,
            'prefix_length': data['length'], 'phase_probabilities': {'approach_and_grasp': .5, 'carry_remainder': .5},
            'actual_source_is_full_task_success': False, 'online_policy_phase_input': False}


class GraspBalancedPrefix(CarryPrefix):
    def __init__(self, manifest, normalizer):
        identity = phase_identity(manifest)
        super().__init__(manifest, normalizer)
        self.boundary = identity['approach_and_stabilization_stop_exclusive']
        self.phase_draw_counts = {'approach_and_grasp': 0, 'carry_remainder': 0}

    def sample(self, index):
        if random.random() < .5:
            phase = 'approach_and_grasp'
            frame = random.randrange(self.boundary)
        else:
            phase = 'carry_remainder'
            frame = random.randrange(self.boundary, len(self.states))
        self.phase_draw_counts[phase] += 1
        return super().sample(frame)
