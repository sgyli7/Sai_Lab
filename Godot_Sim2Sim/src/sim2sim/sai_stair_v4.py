"""High-obstacle event-memory extension of the dense-edge stair contract."""
from __future__ import annotations

import numpy as np

from sim2sim.sai_stair_v2 import ACTION_SIZE
from sim2sim.sai_stair_v3 import observation_numpy as observation_v3


OBSERVATION_SIZE = 243


def observation_numpy(state: dict, command, last_action) -> np.ndarray:
    base = observation_v3(state, command, last_action)
    latched = float(state.get("high_obstacle_latched", 0.))
    if latched not in (0., 1.):
        raise ValueError("Invalid high-obstacle event memory")
    return np.concatenate((base, np.array([latched], dtype=np.float32)))
