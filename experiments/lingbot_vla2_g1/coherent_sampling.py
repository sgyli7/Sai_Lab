"""Deterministic antithetic flow sampling; no task phase or world information."""
import contextlib
import hashlib
import sys

import numpy as np
import torch

from .contract import unpack_actions
from .diagnostics.precision_probe import noise_pair

SEED = 42
ADAPTER = 'coherent_antithetic_two_predictions_v1'


def common_noise():
    return noise_pair(SEED)[0]


def noise_identity():
    values = common_noise().float().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def infer(runtime, rgb, state):
    """Return the mean of two actual model predictions, with both as evidence."""
    inputs = runtime.inputs(rgb, state)
    noise = common_noise().to(runtime.device)
    predictions = []
    for sign in (1, -1):
        with torch.inference_mode(), torch.autocast(
                device_type=runtime.device.type, dtype=torch.bfloat16,
                enabled=runtime.device.type == 'cuda'), contextlib.redirect_stdout(sys.stderr):
            sampled = runtime.model.sample_actions(**inputs, noise=sign * noise)
        actions = runtime.normalizer.denormalize(unpack_actions(sampled[0].float().cpu().numpy()), 'actions')
        if actions.shape != (50, 32) or not np.isfinite(actions).all():
            raise ValueError('Finite same-checkpoint effective32 prediction required')
        predictions.append(actions)
    predictions = np.asarray(predictions, np.float32)
    return predictions.mean(axis=0, dtype=np.float32), predictions
