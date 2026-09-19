"""61D observation builder matching microduck_rl infer_policy --new-cmd-obs."""

from __future__ import annotations

import numpy as np

from sim2sim.backends import SimState
from sim2sim.coords import quat_rotate_inverse_wxyz

# STAND2 / HOME (matches infer_policy.DEFAULT_POSE)
DEFAULT_HOME = np.array(
    [
        0.0,
        -0.0873,
        -0.4579,
        -0.0049,
        0.4530,
        0.3491,
        0.3491,
        0.0,
        0.0,
        0.0,
        0.0873,
        0.4579,
        0.0049,
        -0.4530,
    ],
    dtype=np.float32,
)


def command_13(vel_xyz: np.ndarray, head: np.ndarray | None = None, body: np.ndarray | None = None) -> np.ndarray:
    cmd = np.zeros(13, dtype=np.float32)
    cmd[0:3] = np.asarray(vel_xyz, dtype=np.float32).reshape(3)
    if head is not None:
        cmd[3:7] = np.asarray(head, dtype=np.float32).reshape(4)
    if body is not None:
        cmd[7:13] = np.asarray(body, dtype=np.float32).reshape(6)
    return cmd


def build_obs(
    state: SimState,
    last_action: np.ndarray,
    command: np.ndarray,
    home: np.ndarray | None = None,
) -> np.ndarray:
    home = DEFAULT_HOME if home is None else np.asarray(home, dtype=np.float32)
    ang = np.asarray(state.base_angvel_local, dtype=np.float32).reshape(3)
    grav = quat_rotate_inverse_wxyz(state.base_quat_wxyz, np.array([0.0, 0.0, -1.0])).astype(np.float32)
    qrel = (np.asarray(state.q, dtype=np.float32).reshape(-1) - home).astype(np.float32)
    qd = np.asarray(state.qd, dtype=np.float32).reshape(-1)
    act = np.asarray(last_action, dtype=np.float32).reshape(-1)
    cmd = np.asarray(command, dtype=np.float32).reshape(-1)
    return np.concatenate([ang, grav, qrel, qd, act, cmd]).astype(np.float32)
