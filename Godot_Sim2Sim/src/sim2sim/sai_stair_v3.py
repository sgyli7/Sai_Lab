"""Dense-edge extension of the phase-free Sai stair policy contract."""
from __future__ import annotations

import numpy as np

from sim2sim.sai_stair_v2 import ACTION_SIZE, observation_numpy as observation_v2


DENSE_EDGE_SIZE = 138
OBSERVATION_SIZE = 242


def observation_numpy(state: dict, command, last_action) -> np.ndarray:
    """Append 20 mm edge samples while preserving every v2 feature index."""
    base = observation_v2(state, command, last_action)
    dense = np.asarray(state.get("terrain_edge_heights", []), dtype=float)
    ground = np.asarray(state.get("wheel_ground_heights", []), dtype=float)
    if dense.shape != (DENSE_EDGE_SIZE,) or ground.shape != (4,):
        raise ValueError("Invalid dense-edge Sai stair observation state")
    edge = np.clip((dense - float(np.mean(ground))) * 5., -2., 2.).astype(np.float32)
    result = np.concatenate((base, edge)).astype(np.float32)
    if result.shape != (OBSERVATION_SIZE,) or not np.isfinite(result).all():
        raise ValueError("Non-finite dense-edge Sai stair observation")
    return result
