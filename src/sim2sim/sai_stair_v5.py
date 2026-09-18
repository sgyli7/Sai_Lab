"""Safety-bounded phase-free stair action contract.

The policy still decides every leg target.  This contract only exposes a
deployable subset of the mechanism range and rate-limits target motion to the
same envelope in training, MuJoCo evaluation and native Godot execution.
"""
from __future__ import annotations

import numpy as np

from sim2sim.sai_stair_v3 import OBSERVATION_SIZE, observation_numpy


ACTION_SIZE = 16
JOINT_ACTION_SCALES = np.tile(np.array([0.30, 0.60, 1.02, 1.0]), 4)
TARGET_SLEW_RAD_S = np.tile(np.array([3.0, 4.0, 6.0, 20.0]), 4)
SIDES = np.array([1.0, -1.0, 1.0, -1.0])
WHEEL_INDICES = np.array([3, 7, 11, 15])


def targets_numpy(action, command, previous_target, *, dt=0.02,
                  wheel_residual_scale=6.0, joint_action_scales=JOINT_ACTION_SCALES,
                  target_slew_rad_s=TARGET_SLEW_RAD_S) -> np.ndarray:
    action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    command = np.asarray(command, dtype=float)
    previous = np.asarray(previous_target, dtype=float)
    scales = np.asarray(joint_action_scales, dtype=float)
    slew = np.asarray(target_slew_rad_s, dtype=float)
    if action.shape != (16,) or command.shape != (3,) or previous.shape != (16,):
        raise ValueError("Invalid safe-residual stair action")
    if scales.shape == (3,):
        scales = np.tile(np.r_[scales, 1.0], 4)
    if slew.shape == (3,):
        slew = np.tile(np.r_[slew, 20.0], 4)
    if scales.shape != (16,) or slew.shape != (16,) or np.any(scales <= 0) or np.any(slew <= 0):
        raise ValueError("Invalid safe-residual stair envelope")
    desired = action * scales
    desired[WHEEL_INDICES] = SIDES * (
        (command[0] - command[1] * SIDES * 0.146) / 0.048
        + wheel_residual_scale * action[WHEEL_INDICES])
    return previous + np.clip(desired - previous, -slew * dt, slew * dt)
