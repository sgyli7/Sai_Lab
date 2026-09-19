"""Shared skill / kick metrics (gate fall, play fall, peaks)."""

from __future__ import annotations

import numpy as np

from sim2sim.compare import metrics_one
from sim2sim.coords import quat_rotate_inverse_wxyz

FALL_Z = 0.08
FALL_TILT = 55.0
PLAY_FALL_GRAV_Z = -0.35
PLAY_FALL_RESET_S = 0.8


def tilt_deg_from_quat(quat: np.ndarray) -> np.ndarray:
    """Body-z vs world-z tilt in degrees. quat shape (N,4) wxyz."""
    q = np.asarray(quat)
    x, y = q[:, 1], q[:, 2]
    gz_body = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(gz_body, -1.0, 1.0)))


def fall_timeline(traj: dict) -> dict:
    """Enrich metrics_one with gate/play fall times and peaks used by kick gate."""
    pos = np.asarray(traj["base_pos"])
    quat = np.asarray(traj["base_quat"])
    t = np.asarray(traj["t"])
    z = pos[:, 2]
    tilt = tilt_deg_from_quat(quat)

    grav_z = np.empty(len(quat), dtype=np.float64)
    for i in range(len(quat)):
        g = quat_rotate_inverse_wxyz(quat[i], np.array([0.0, 0.0, -1.0]))
        grav_z[i] = float(g[2])

    gate_mask = (z < FALL_Z) | (tilt > FALL_TILT)
    gate_t = float(t[gate_mask.argmax()]) if gate_mask.any() else None

    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.02
    acc = 0.0
    play_t = None
    for i in range(len(t)):
        if grav_z[i] > PLAY_FALL_GRAV_Z or z[i] < 0.055:
            acc += dt
            if acc >= PLAY_FALL_RESET_S:
                play_t = float(t[i])
                break
        else:
            acc = 0.0

    q = np.asarray(traj["q"])
    qerr = np.max(np.abs(q - q[0:1]), axis=1) if len(q) else np.asarray([])
    motion_i = int(np.argmax(qerr > 0.15)) if len(qerr) and (qerr > 0.15).any() else None

    m = metrics_one(traj)
    return {
        **m,
        "fall_t_gate": gate_t,
        "fall_t_play": play_t,
        "tilt_at_1s": float(tilt[np.argmin(np.abs(t - 1.0))]) if len(t) else None,
        "z_at_1s": float(z[np.argmin(np.abs(t - 1.0))]) if len(t) else None,
        "tilt_at_2s": float(tilt[np.argmin(np.abs(t - 2.0))]) if len(t) else None,
        "z_at_2s": float(z[np.argmin(np.abs(t - 2.0))]) if len(t) else None,
        "xy_final": [float(pos[-1, 0]), float(pos[-1, 1])] if len(pos) else [0.0, 0.0],
        "q_max_abs_delta": float(qerr.max()) if len(qerr) else 0.0,
        "first_motion_t": float(t[motion_i]) if motion_i is not None else None,
        "action_peak": float(np.max(np.abs(traj["action"]))) if "action" in traj else 0.0,
    }


def pair_rmse(mj: dict, gd: dict) -> dict:
    n = min(len(mj["t"]), len(gd["t"]))
    q_rmse = float(np.sqrt(np.mean((np.asarray(mj["q"])[:n] - np.asarray(gd["q"])[:n]) ** 2)))
    xy_rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (np.asarray(mj["base_pos"])[:n, :2] - np.asarray(gd["base_pos"])[:n, :2]) ** 2,
                    axis=1,
                )
            )
        )
    )
    return {"q_rmse": q_rmse, "xy_rmse": xy_rmse}


def classify_kick_pair(mj_info: dict, gd_info: dict, *, known_fail: bool = True) -> dict:
    """Decide gate status for one kick skill without poisoning walk gate semantics.

    Soft / known-fail (default): Godot falls while MuJoCo stands → KNOWN_FAIL, not hard.
    Hard mode (known_fail=False): same asymmetry → HARD_FAIL.
    Infra FAIL: MuJoCo fell (unexpected for kick), or both stood but q_rmse insane is warn only.
    """
    hard: list[str] = []
    known: list[str] = []
    warn: list[str] = []

    if mj_info.get("fell"):
        hard.append("mujoco_fell_unexpected")
    if gd_info.get("fell") and not mj_info.get("fell"):
        msg = (
            f"godot_fell_mujoco_stood(fall_t_gate={gd_info.get('fall_t_gate')})"
        )
        if known_fail:
            known.append(msg)
        else:
            hard.append(msg)
    if gd_info.get("fell") and mj_info.get("fell"):
        warn.append("both_fell")
    if (not gd_info.get("fell")) and (not mj_info.get("fell")):
        # Future green path
        pass

    if hard:
        status = "HARD_FAIL"
    elif known:
        status = "KNOWN_FAIL"
    else:
        status = "PASS"
    return {"status": status, "hard_fail": hard, "known_fail": known, "warnings": warn}
