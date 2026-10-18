"""Original physics audit plus independently reconstructed coherent sampling."""
import json
from pathlib import Path

import numpy as np

from experiments.lingbot_vla2_g1.acceptance_context import audit_case
from experiments.lingbot_vla2_g1.coherent_sampling import ADAPTER, SEED, noise_identity
from experiments.lingbot_vla2_g1.data import OWN, ROOT
from .development_audit import decoded_evidence as pointmass_decoded


def sampling_evidence(config, events):
    loaded = events[0]
    if (loaded.get('sampling_adapter') != ADAPTER or loaded.get('common_noise_seed') != SEED
            or loaded.get('predictions_per_query') != 2
            or loaded.get('common_noise_float32_sha256') != noise_identity()):
        raise ValueError('Fixed coherent sampling identity differs')
    for key, filename in [('coherent_sampling_sha256', 'coherent_sampling.py'),
                          ('coherent_service_sha256', 'service_coherent_noise.py')]:
        path = str(ROOT / 'experiments/lingbot_vla2_g1' / filename)
        if loaded.get(key) != config['frozen_files'].get(path):
            raise ValueError('Actual coherent sampling source differs')
    for event in events[1:]:
        predictions = np.asarray(event.get('model_predictions32'), np.float32)
        if (predictions.shape != (2, 50, 32) or not np.isfinite(predictions).all()
                or event.get('sampling_adapter') != ADAPTER
                or event.get('common_noise_float32_sha256') != loaded['common_noise_float32_sha256']
                or not np.array_equal(predictions.mean(axis=0, dtype=np.float32), np.asarray(event['actions32'], np.float32))):
            raise ValueError('Actual model predictions do not reconstruct the applied coherent mean')


def decoded_evidence(config, result):
    command = config['policy_command']
    if command[:3] == [str(OWN / 'venv/bin/python'), '-m', 'experiments.lingbot_vla2_g1.service_coherent_noise']:
        equivalent = {**config, 'policy_command': [*command[:2], 'experiments.lingbot_vla2_g1.service_pointmass', *command[3:]]}
        path, events = pointmass_decoded(equivalent, result)
        sampling_evidence(config, events)
        return path, events
    return pointmass_decoded(config, result)


def audit(config_path):
    config = json.loads(Path(config_path).read_text())
    return audit_case(config_path, decoded_checker=decoded_evidence)
