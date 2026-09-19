"""Coordinate helpers: MuJoCo Z-up (wxyz) ↔ Godot Y-up (xyzw).

Godot RigidBody3D local axes are aligned with the MuJoCo inertial principal
axes (same XYZ meaning). Only *world* quantities are rotated:

    p_godot = R @ p_mujoco
    R = [[1, 0, 0],
         [0, 0, 1],
         [0,-1, 0]]   # (x, y, z)_m → (x, z, -y)_g

    R_godot = R @ R_mujoco     # local coords are *not* conjugated
"""

from __future__ import annotations

import numpy as np

# (x, y, z)_mujoco → (x, z, -y)_godot
R_M2G = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
R_G2M = R_M2G.T


def m2g_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([v[0], v[2], -v[1]], dtype=np.float64)


def g2m_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([v[0], -v[2], v[1]], dtype=np.float64)


def m2g_mat(rm: np.ndarray) -> np.ndarray:
    """MuJoCo body rotation (local→world) to Godot basis columns."""
    rm = np.asarray(rm, dtype=np.float64).reshape(3, 3)
    return R_M2G @ rm


def g2m_mat(rg: np.ndarray) -> np.ndarray:
    rg = np.asarray(rg, dtype=np.float64).reshape(3, 3)
    return R_G2M @ rg


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] → 3×3 rotation (local→world)."""
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mat_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    """3×3 rotation → unit quaternion [w, x, y, z]."""
    r = np.asarray(r, dtype=np.float64).reshape(3, 3)
    t = float(np.trace(r))
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (r[2, 1] - r[1, 2]) * s
        y = (r[0, 2] - r[2, 0]) * s
        z = (r[1, 0] - r[0, 1]) * s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if q[0] < 0:
        q = -q
    return q / n


def m2g_quat_wxyz(q_m: np.ndarray) -> np.ndarray:
    """MuJoCo world quat → Godot world quat (still wxyz)."""
    return mat_to_quat_wxyz(m2g_mat(quat_wxyz_to_mat(q_m)))


def g2m_quat_wxyz(q_g: np.ndarray) -> np.ndarray:
    return mat_to_quat_wxyz(g2m_mat(quat_wxyz_to_mat(q_g)))


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_rotate_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vec by inverse of [w,x,y,z] (matches infer_policy.PolicyInference)."""
    w = float(quat[0])
    xyz = np.asarray(quat[1:4], dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    t = np.cross(xyz, v) * 2.0
    return v - w * t + np.cross(xyz, t)


def basis_from_z(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.float64).reshape(3)
    n = np.linalg.norm(z)
    if n < 1e-12:
        raise ValueError("zero hinge axis")
    z = z / n
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def fmt_f(v: float) -> str:
    return f"{float(v):.10g}"
