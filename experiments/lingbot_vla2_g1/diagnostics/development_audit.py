"""Original independent physics audit plus an explicit new-service identity gate."""
import json
from pathlib import Path
import numpy as np

from experiments.lingbot_vla2_g1.acceptance_context import audit_case
from experiments.lingbot_vla2_g1.benchmark_pose_speed import decoded_evidence as original_decoded
from experiments.lingbot_vla2_g1.constant_inverse import PointMassMeanStd
from experiments.lingbot_vla2_g1.data import OWN


def decoded_evidence(config, result):
    command = config['policy_command']
    if command[2] == 'experiments.lingbot_vla2_g1.service':
        return original_decoded(config, result)
    if command[:3] != [str(OWN / 'venv/bin/python'), '-m', 'experiments.lingbot_vla2_g1.service_pointmass']:
        raise ValueError('Unknown development policy service')
    # The original checker verifies the identical JSONL protocol and weights.
    # Admit only this named variant, with additional actual load/code/output checks.
    equivalent = {**config, 'policy_command': [*command[:2], 'experiments.lingbot_vla2_g1.service', *command[3:]]}
    path, events = original_decoded(equivalent, result)
    loaded = events[0]
    source = str(OWN.parents[1] / 'experiments/lingbot_vla2_g1/constant_inverse.py')
    normalizer = PointMassMeanStd(loaded['norm_path'])
    if (loaded.get('inference_adapter') != 'original_train_exact_constant_point_mass_inverse_v1'
            or loaded.get('constant_inverse_sha256') != config['frozen_files'].get(source)
            or loaded.get('constant_action_indices') != np.flatnonzero(normalizer.constant).tolist()):
        raise ValueError('Actual inverse adapter identity differs')
    expected = np.asarray(normalizer.doc['actions']['mean'], np.float32)[normalizer.constant]
    for event in events[1:]:
        actions = np.asarray(event['actions32'], np.float32)
        if not np.all(actions[:, normalizer.constant] == expected):
            raise ValueError('Actual constant-channel action differs from original TRAIN mean')
    return path, events


def audit(config_path):
    config = json.loads(Path(config_path).read_text())
    checker = None if config['policy_id'] == 'gr00t_n16' else decoded_evidence
    return audit_case(config_path, decoded_checker=checker)
