"""Shared fallen / tilt predicate for play, compare, and training termination."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim2sim.coords import quat_rotate_inverse_wxyz

_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)


@dataclass(frozen=True)
class FallCriteria:
    tilt_deg: float = 70.0
    min_z: float = 0.055


def _projected_gravity(base_quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse_wxyz(base_quat_wxyz, _DOWN)


def tilt_deg(base_quat_wxyz: np.ndarray) -> float:
    """Tilt from vertical in degrees (0 = upright)."""
    grav = _projected_gravity(base_quat_wxyz)
    c = float(np.clip(-float(grav[2]), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def tilt_threshold(tilt_deg: float = 70.0) -> float:
    """Projected-gravity z above this value is past ``tilt_deg`` from vertical."""
    return -math.cos(math.radians(float(tilt_deg)))


def fallen_mask(
    grav: np.ndarray,
    pos: np.ndarray,
    *,
    tilt_deg: float = 70.0,
    min_z: float = 0.055,
) -> np.ndarray:
    """Batched ``fallen`` from already-computed projected gravity and trunk position."""
    g = np.asarray(grav, dtype=np.float64).reshape(-1, 3)
    p = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    return (g[:, 2] > tilt_threshold(tilt_deg)) | (p[:, 2] < float(min_z))


def fallen(
    base_quat_wxyz: np.ndarray,
    base_pos: np.ndarray,
    *,
    tilt_deg: float = 70.0,
    min_z: float = 0.055,
) -> bool:
    """True if projected-gravity z > -cos(tilt) (lean past tilt_deg) or trunk z < min_z."""
    grav = _projected_gravity(base_quat_wxyz)
    return bool(fallen_mask(grav.reshape(1, 3), np.asarray(base_pos).reshape(1, 3), tilt_deg=tilt_deg, min_z=min_z)[0])
