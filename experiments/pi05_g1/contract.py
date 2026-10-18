"""Frozen G1/PI05 adapter. Dataset order is never simulator joint order.

The AGILE whole body controller owns legs and waist. Only 28 arm/hand
targets plus height/navigation pass from this policy to that controller.
"""
from __future__ import annotations

import numpy as np

SCHEMA = "pi05_g1_static_v1"
ARM_SUFFIXES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
HAND_SUFFIXES = ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")
GROUPS = {
    **{f"{side}_arm": tuple(f"{side}_{s}_joint" for s in ARM_SUFFIXES) for side in ("left", "right")},
    **{f"{side}_hand": tuple(f"{side}_hand_{s}_joint" for s in HAND_SUFFIXES) for side in ("left", "right")},
}
UPPER_NAMES = tuple(name for group in ("left_arm", "right_arm", "left_hand", "right_hand") for name in GROUPS[group])
WAIST_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
STATE_NAMES = UPPER_NAMES + WAIST_NAMES + ("padding",)
ACTION_NAMES = UPPER_NAMES + ("base_height_command", "navigate_vx", "navigate_vy", "navigate_yaw_rate")
DATASET_NAMES = (
    *(f"{side}_{s}_joint" for side in ("left", "right") for s in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")),
    *WAIST_NAMES, *GROUPS["left_arm"], *GROUPS["left_hand"], *GROUPS["right_arm"], *GROUPS["right_hand"],
)
PROMPT = "Pick up the apple and place it on the plate."
CONTROL_HZ = 50
ACTION_HORIZON = 50


def _vector(value, size: int, name: str):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name}: expected finite shape {(size,)}, got {result.shape}")
    return result


def _indices(names, wanted):
    if len(names) != 43 or len(set(names)) != 43 or set(names) != set(DATASET_NAMES):
        raise ValueError("G1 joint names must match the complete 43 named joints")
    return [list(names).index(name) for name in wanted]


def encode_state(joint_positions, joint_names=DATASET_NAMES):
    q = _vector(joint_positions, 43, "joint_positions")
    return np.concatenate((q[_indices(joint_names, UPPER_NAMES + WAIST_NAMES)], np.zeros(1, np.float32)))


def encode_action(joint_targets, base_height_command, navigate_command, joint_names=DATASET_NAMES):
    q = _vector(joint_targets, 43, "joint_targets")
    height = np.asarray(base_height_command, dtype=np.float32).reshape(-1)
    return np.concatenate((q[_indices(joint_names, UPPER_NAMES)], _vector(height, 1, "base_height"), _vector(navigate_command, 3, "navigate")))


def encode_row(row):
    return {
        "state": encode_state(row["observation.state"]),
        "actions": encode_action(row["action"], row["teleop.base_height_command"], row["teleop.navigate_command"]),
    }


def decode_to_source(action32, measured_joint_state43, source_joint_names):
    """Return 43 named joint targets + nav3 + height1 + torsoRPY3.

    Measured leg/waist values only fill entries which AGILE replaces; they
    are not model actions and are never substituted for measured feedback.
    """
    action = _vector(action32, 32, "action32")
    result = np.zeros(50, dtype=np.float32)
    result[:43] = _vector(measured_joint_state43, 43, "measured joints")
    result[_indices(source_joint_names, UPPER_NAMES)] = action[:28]
    result[43:46] = action[29:32]
    result[46] = action[28]
    return result


def normalize(values, stats):
    """PI05 quantile transform; constant channels are exactly zero."""
    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
    if values.shape[-1] != 32 or lo.shape != (32,) or hi.shape != (32,) or not np.isfinite(values).all():
        raise ValueError("Invalid normalization shapes or values")
    span = hi - lo
    return np.where(span > 1e-6, 2 * (values - lo) / np.maximum(span, 1e-6) - 1, 0).astype(np.float32)


def denormalize(values, stats):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
    if values.shape[-1] != 32 or lo.shape != (32,) or hi.shape != (32,) or not np.isfinite(values).all():
        raise ValueError("Invalid denormalization shapes or values")
    span = hi - lo
    return np.where(span > 1e-6, (values + 1) * .5 * span + lo, lo).astype(np.float32)
