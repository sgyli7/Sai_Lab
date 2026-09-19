"""Phase-free Sai stair policy contract shared by training and deployment."""
from __future__ import annotations

import numpy as np


OBSERVATION_SIZE = 104
ACTION_SIZE = 16
STAND_HEIGHT = 0.2192
CROUCH_DROP = 0.035
WHEEL_RADIUS = 0.048
LEG_INDICES = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
WHEEL_INDICES = np.array([3, 7, 11, 15])
SIDES = np.array([1.0, -1.0, 1.0, -1.0])
FRONTS = np.array([1.0, 1.0, -1.0, -1.0])
LEG_SCALES = np.array([0.45, 0.70, 1.20] * 4)


def observation_numpy(state: dict, command, last_action) -> np.ndarray:
    """Build the exact deployable observation; no clock or gait phase is present."""
    columns = np.asarray(state["base_rotation_columns"], dtype=float)
    linear = np.asarray(state["base_linear_world"], dtype=float)
    angular = np.asarray(state["base_angular_world"], dtype=float)
    q = np.asarray(state["q"], dtype=float)
    v = np.asarray(state["v"], dtype=float)
    terrain = np.asarray(state["terrain_heights"], dtype=float)
    path = np.asarray(state["terrain_path_heights"], dtype=float)
    ground = np.asarray(state["wheel_ground_heights"], dtype=float)
    if q.size < 16 or v.size < 16 or terrain.shape != (24,) or path.shape != (15,) or ground.shape != (4,):
        raise ValueError("Invalid phase-free Sai stair observation state")
    body_linear = columns @ linear
    body_angular = columns @ angular
    local_ground = float(np.mean(ground))
    wheel_positions = np.asarray(state.get("wheel_positions", []), dtype=float)
    clearance = (wheel_positions[:, 2] - ground - WHEEL_RADIUS
                 if wheel_positions.shape == (4, 3) else np.zeros(4))
    result = np.concatenate([
        columns[:, 2], body_linear, body_angular, np.asarray(command, dtype=float),
        q[LEG_INDICES], .1 * v[LEG_INDICES], .1 * v[WHEEL_INDICES] * SIDES,
        np.asarray(last_action, dtype=float),
        np.clip((terrain - local_ground) * 5.0, -2.0, 2.0),
        np.clip((path - local_ground) * 5.0, -2.0, 2.0),
        np.clip((ground - local_ground) * 5.0, -1.0, 1.0),
        np.clip(clearance * 20.0, -1.0, 2.0),
        np.array([(float(state["base_position"][2]) - local_ground - STAND_HEIGHT) * 10.0]),
    ]).astype(np.float32)
    if result.shape != (OBSERVATION_SIZE,) or not np.isfinite(result).all():
        raise ValueError("Non-finite phase-free Sai stair observation")
    return result


def targets_numpy(action, command, crouch: float, *, wheel_residual_scale=6.0) -> np.ndarray:
    """Map direct normalized policy actions to joint positions and wheel velocities."""
    action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    command = np.asarray(command, dtype=float)
    if action.shape != (ACTION_SIZE,) or command.shape != (3,):
        raise ValueError("Invalid phase-free Sai stair action")
    target = np.zeros(16, dtype=float)
    target[LEG_INDICES] = action[LEG_INDICES] * LEG_SCALES
    target[0::4] = np.clip(target[0::4], -.45, .45)
    target[1::4] = np.clip(target[1::4], -.70, .70)
    target[2::4] = np.clip(target[2::4], -1.20, 1.20)
    target[WHEEL_INDICES] = SIDES * ((command[0] - command[1] * SIDES * .146) / WHEEL_RADIUS
                                     + wheel_residual_scale * action[WHEEL_INDICES])
    return target


def actions_from_targets_numpy(target, command, *, wheel_residual_scale=6.0) -> np.ndarray:
    """Invert the direct target mapping for phase-free policy distillation."""
    target = np.asarray(target, dtype=float)
    command = np.asarray(command, dtype=float)
    if target.shape != (ACTION_SIZE,) or command.shape != (3,) or wheel_residual_scale <= 0:
        raise ValueError("Invalid phase-free Sai stair target")
    action = np.zeros(ACTION_SIZE, dtype=np.float32)
    action[LEG_INDICES] = target[LEG_INDICES] / LEG_SCALES
    nominal = (command[0] - command[1] * SIDES * .146) / WHEEL_RADIUS
    action[WHEEL_INDICES] = (target[WHEEL_INDICES] / SIDES - nominal) / wheel_residual_scale
    return np.clip(action, -1., 1.)
