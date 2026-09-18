"""Trace-level safety gates for Sai locomotion candidates.

Task completion is deliberately absent from this module.  A controller must
first keep the measured mechanism inside a conservative deployable envelope;
task metrics are evaluated only after this gate passes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, radians
from typing import Iterable

import numpy as np


HAA = np.array([0, 4, 8, 12])
NON_WHEEL = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
JOINT_LIMITS = np.tile(np.array([0.45, 0.70, 1.20, np.inf]), 4)


@dataclass(frozen=True)
class SafetyLimits:
    hard_stop_utilization: float = 0.98
    max_hard_stop_fraction: float = 0.01
    max_hard_stop_run_s: float = 0.10
    haa_soft_stop_utilization: float = 0.95
    max_haa_soft_stop_fraction: float = 0.05
    max_haa_soft_stop_run_s: float = 0.20
    action_saturation: float = 0.98
    max_action_saturation_fraction: float = 0.08
    max_action_saturation_run_s: float = 0.30
    max_target_slew_rad_s: float = 12.0
    max_measured_speed_rad_s: float = 12.0
    min_upright: float = cos(radians(15.0))
    max_low_step_support_loss_run_s: float = 0.06
    max_high_step_support_loss_run_s: float = 0.50


def _leg_q(sample: dict) -> np.ndarray | None:
    if len(sample.get("q", [])) >= 16:
        return np.asarray(sample["q"][:16], dtype=float)
    # MuJoCo free-joint qpos is xyz + quaternion, followed by the 16 leg axes.
    if len(sample.get("qpos", [])) >= 23:
        return np.asarray(sample["qpos"][7:23], dtype=float)
    return None


def _target(sample: dict) -> np.ndarray | None:
    value = sample.get("target_leg")
    if value is None:
        value = sample.get("controller_command", {}).get("target_leg")
    return np.asarray(value, dtype=float) if value is not None and len(value) >= 16 else None


def _active(samples: list[dict]) -> list[dict]:
    stair = [row for row in samples
             if row.get("controller_stage", row.get("stage")) == "stairs"]
    if stair:
        return stair
    moving = [row for row in samples if abs(float(row.get("command", 0.0)
                                                if not isinstance(row.get("command"), list)
                                                else row["command"][0])) > 0.015]
    return moving or samples


def _run_windows(rows: list[dict], mask: np.ndarray) -> list[dict]:
    if not len(mask):
        return []
    times = np.asarray([float(row.get("time", i * 0.02)) for i, row in enumerate(rows)])
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.02
    windows = []
    start = None
    for i, active in enumerate(np.r_[mask, False]):
        if active and start is None:
            start = i
        elif not active and start is not None:
            windows.append({"start_s": float(times[start]), "end_s": float(times[i - 1]),
                            "duration_s": float((i - start) * dt)})
            start = None
    return sorted(windows, key=lambda row: row["duration_s"], reverse=True)


def assess_motion_safety(samples: Iterable[dict], riser_m: float = 0.0,
                         limits: SafetyLimits = SafetyLimits(),
                         ignored_action_indices: tuple[int, ...] = ()) -> dict:
    rows = _active(list(samples))
    if not rows:
        return {"passed": False, "checks": {"has_active_trace": False},
                "metrics": {}, "limits": asdict(limits), "windows": {}}

    q_rows = [_leg_q(row) for row in rows]
    has_q = all(row is not None for row in q_rows)
    q = np.stack(q_rows) if has_q else np.empty((0, 16))
    utilization = np.abs(q) / JOINT_LIMITS if has_q else np.empty((0, 16))
    hard_mask = np.any(utilization[:, NON_WHEEL] >= limits.hard_stop_utilization, axis=1) if has_q else np.ones(len(rows), bool)
    haa_mask = np.any(utilization[:, HAA] >= limits.haa_soft_stop_utilization, axis=1) if has_q else np.ones(len(rows), bool)

    actions = [row.get("policy_action") for row in rows]
    has_actions = all(value is not None and len(value) >= 16 for value in actions)
    action = np.asarray(actions, dtype=float) if has_actions else np.empty((0, 16))
    action_columns = np.asarray(
        [i for i in range(16) if i not in ignored_action_indices], dtype=int
    )
    screened_action = action[:, action_columns] if has_actions else action
    action_mask = (np.any(np.abs(screened_action) >= limits.action_saturation, axis=1)
                   if has_actions else np.ones(len(rows), bool))

    targets = [_target(row) for row in rows]
    has_targets = all(value is not None for value in targets)
    target = np.stack(targets) if has_targets else np.empty((0, 16))
    times = np.asarray([float(row.get("time", i * 0.02)) for i, row in enumerate(rows)])
    dt = np.diff(times)
    slew = (np.abs(np.diff(target[:, NON_WHEEL], axis=0)) / dt[:, None]
            if has_targets and len(rows) > 1 and np.all(dt > 0) else np.empty((0, len(NON_WHEEL))))

    velocities = [row.get("v") for row in rows]
    has_velocity = all(value is not None and len(value) >= 16 for value in velocities)
    velocity_peak = (float(np.max(np.abs(np.asarray(velocities, dtype=float)[:, NON_WHEEL])))
                     if has_velocity else None)

    upright = np.asarray([float(row.get("upright", np.nan)) for row in rows])
    has_upright = bool(np.isfinite(upright).all())

    supported = []
    for row in rows:
        if "wheels_supported" in row:
            supported.append(int(row["wheels_supported"]))
        elif len(row.get("wheel_positions", [])) == 4 and len(row.get("wheel_ground_heights", [])) == 4:
            gaps = [position[2] - ground - 0.048
                    for position, ground in zip(row["wheel_positions"], row["wheel_ground_heights"])]
            supported.append(sum(gap <= 0.006 for gap in gaps))
        else:
            supported = []
            break
    support_mask = np.asarray(supported) < 2 if supported else np.zeros(len(rows), bool)

    windows = {
        "hard_stop": _run_windows(rows, hard_mask)[:5],
        "haa_soft_stop": _run_windows(rows, haa_mask)[:5],
        "action_saturation": _run_windows(rows, action_mask)[:5],
        "fewer_than_two_supports": _run_windows(rows, support_mask)[:5],
    }
    longest = lambda name: windows[name][0]["duration_s"] if windows[name] else 0.0
    support_limit = (limits.max_high_step_support_loss_run_s if riser_m > 0.04
                     else limits.max_low_step_support_loss_run_s)
    metrics = {
        "active_samples": len(rows),
        "hard_stop_frame_fraction": float(np.mean(hard_mask)),
        "hard_stop_longest_run_s": longest("hard_stop"),
        "haa_soft_stop_frame_fraction": float(np.mean(haa_mask)),
        "haa_soft_stop_longest_run_s": longest("haa_soft_stop"),
        "action_saturated_entry_fraction": (float(np.mean(np.abs(screened_action) >= limits.action_saturation))
                                               if has_actions else None),
        "action_saturation_longest_run_s": longest("action_saturation"),
        "target_slew_peak_rad_s": float(np.max(slew)) if slew.size else None,
        "measured_joint_speed_peak_rad_s": velocity_peak,
        "minimum_upright": float(np.min(upright)) if has_upright else None,
        "support_loss_longest_run_s": longest("fewer_than_two_supports") if supported else None,
    }
    checks = {
        "has_joint_positions": has_q,
        "hard_stop_fraction": has_q and metrics["hard_stop_frame_fraction"] <= limits.max_hard_stop_fraction,
        "hard_stop_run": has_q and metrics["hard_stop_longest_run_s"] <= limits.max_hard_stop_run_s,
        "haa_soft_stop_fraction": has_q and metrics["haa_soft_stop_frame_fraction"] <= limits.max_haa_soft_stop_fraction,
        "haa_soft_stop_run": has_q and metrics["haa_soft_stop_longest_run_s"] <= limits.max_haa_soft_stop_run_s,
        "has_policy_actions": has_actions,
        "action_saturation": has_actions and metrics["action_saturated_entry_fraction"] <= limits.max_action_saturation_fraction,
        "action_saturation_run": has_actions and metrics["action_saturation_longest_run_s"] <= limits.max_action_saturation_run_s,
        "has_targets": has_targets,
        "target_slew": has_targets and metrics["target_slew_peak_rad_s"] is not None and metrics["target_slew_peak_rad_s"] <= limits.max_target_slew_rad_s,
        "measured_joint_speed": (not has_velocity or velocity_peak <= limits.max_measured_speed_rad_s),
        "upright": has_upright and metrics["minimum_upright"] >= limits.min_upright,
        "support": bool(supported) and metrics["support_loss_longest_run_s"] <= support_limit,
    }
    return {"passed": bool(all(checks.values())), "checks": checks, "metrics": metrics,
            "limits": asdict(limits), "windows": windows}
