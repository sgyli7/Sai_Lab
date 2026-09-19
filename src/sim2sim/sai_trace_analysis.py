"""Turn a Sai motion trace into aligned 3D-plus-time diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


JOINT_NAMES = [f"{corner}_{axis}" for corner in ("fl", "fr", "rl", "rr")
               for axis in ("haa", "hip", "knee", "wheel")]
LIMITS = np.tile(np.array([.45, .70, 1.20, np.inf]), 4)


def _windows(mask: np.ndarray, time: np.ndarray) -> list[dict]:
    padded = np.r_[False, mask, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    return [{"start_s": float(time[a]), "end_s": float(time[b - 1]),
             "duration_s": float(time[b - 1] - time[a] + (np.median(np.diff(time)) if len(time) > 1 else 0.))}
            for a, b in zip(starts, stops)]


def analyze_trace(trace: list[dict]) -> dict:
    def command_value(row, key, default=None):
        value = row.get(key)
        if value is None:
            value = row.get("controller_command", {}).get(key, default)
        return value

    rows = [row for row in trace
            if row.get("q") is not None and command_value(row, "target_leg") is not None]
    if len(rows) < 2:
        raise ValueError("Trace needs at least two aligned control samples")
    time = np.asarray([row["time"] for row in rows], dtype=float)
    dt = float(np.median(np.diff(time)))
    q = np.asarray([row["q"][:16] for row in rows], dtype=float)
    v = np.asarray([row["v"][:16] for row in rows], dtype=float)
    target = np.asarray([command_value(row, "target_leg") for row in rows], dtype=float)
    torque = np.asarray([row.get("motor_torque", [np.nan] * 16) for row in rows], dtype=float)
    has_torque = bool(np.isfinite(torque).any())
    rotation = np.asarray([
        row["base_rotation"] if row.get("base_rotation") is not None
        else np.asarray(row.get("base_rotation_columns", np.eye(3)), dtype=float).T
        for row in rows
    ], dtype=float)
    roll = np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2])
    pitch = np.arcsin(np.clip(-rotation[:, 2, 0], -1., 1.))
    body_linear = np.asarray([row.get("base_linear_world", [0., 0., 0.]) for row in rows], dtype=float)
    body_angular = np.asarray([row.get("base_angular_world", [0., 0., 0.]) for row in rows], dtype=float)
    body_accel = np.gradient(body_linear, dt, axis=0)
    body_angular_accel = np.gradient(body_angular, dt, axis=0)
    if all(row.get("cargo_filtered_acceleration") is not None for row in rows):
        cargo_accel = np.asarray([row["cargo_filtered_acceleration"] for row in rows], dtype=float)
        cargo_jerk = np.asarray([row.get("cargo_jerk", [0., 0., 0.]) for row in rows], dtype=float)
    else:
        cargo_velocity = np.asarray([row.get("object_linear_world", [0., 0., 0.]) for row in rows], dtype=float)
        raw_cargo_accel = np.gradient(cargo_velocity, dt, axis=0)
        cargo_accel = np.zeros_like(raw_cargo_accel)
        alpha = dt / (.015 + dt)
        for index in range(1, len(cargo_accel)):
            cargo_accel[index] = cargo_accel[index - 1] + alpha * (
                raw_cargo_accel[index] - cargo_accel[index - 1]
            )
        cargo_jerk = np.gradient(cargo_accel, dt, axis=0)
    has_normal_force = all(row.get("wheel_normal_forces") is not None for row in rows)
    normal_force = np.asarray([row.get("wheel_normal_forces", [0., 0., 0., 0.]) for row in rows], dtype=float)
    support = ((normal_force > .05).sum(axis=1) if has_normal_force else
               np.asarray([row.get("wheels_supported", 0) for row in rows], dtype=int))
    intensity = np.asarray([command_value(row, "skill_intensity", 0.) for row in rows], dtype=float)
    policy_action = np.asarray([row.get("policy_action", [0.] * 16) for row in rows], dtype=float)
    commanded_lift_m = np.maximum(policy_action[:, 4:8], 0.) * np.clip(intensity[:, None], 0., 1.) * .080
    commanded_lift = commanded_lift_m > .003
    commanded_lift_onset = commanded_lift & ~np.vstack(
        [np.zeros((1, 4), dtype=bool), commanded_lift[:-1]])
    wheel = np.asarray([row["wheel_positions"] for row in rows], dtype=float)
    ground = np.asarray([row["wheel_ground_heights"] for row in rows], dtype=float)
    gap = wheel[:, :, 2] - ground - .048
    airborne = gap > .015
    lift = airborne & ~np.vstack([np.zeros((1, 4), dtype=bool), airborne[:-1]])
    landing = ~airborne & np.vstack([np.zeros((1, 4), dtype=bool), airborne[:-1]])
    utilization = np.divide(np.abs(q), LIMITS, out=np.zeros_like(q), where=np.isfinite(LIMITS))

    events = {
        "skill_active": _windows(intensity > .10, time),
        "support_below_three": _windows(support < 3, time),
        "support_below_two": _windows(support < 2, time),
        "tilt_over_10_deg": _windows((np.abs(roll) > np.deg2rad(10)) | (np.abs(pitch) > np.deg2rad(10)), time),
        "cargo_accel_over_5": _windows(np.linalg.norm(cargo_accel, axis=1) > 5., time),
        "joint_over_85pct_limit": _windows((utilization[:, np.isfinite(LIMITS)] > .85).any(axis=1), time),
        "torque_over_7_5Nm": (_windows(np.nanmax(np.abs(torque[:, np.isfinite(LIMITS)]), axis=1) > 7.5, time)
                               if has_torque else []),
        "wheel_lifts": [{"time_s": float(time[i]), "wheel": int(j), "gap_m": float(gap[i, j])}
                        for i, j in zip(*np.nonzero(lift))],
        "commanded_wheel_lifts": [
            {"time_s": float(time[i]), "wheel": int(j),
             "commanded_lift_m": float(commanded_lift_m[i, j])}
            for i, j in zip(*np.nonzero(commanded_lift_onset))],
        "uncommanded_airborne_onsets": [
            {"time_s": float(time[i]), "wheel": int(j), "gap_m": float(gap[i, j])}
            for i, j in zip(*np.nonzero(lift & ~commanded_lift))],
        "wheel_landings": [{"time_s": float(time[i]), "wheel": int(j), "normal_force_N": float(normal_force[i, j])}
                           for i, j in zip(*np.nonzero(landing))],
    }
    joint_summary = {}
    for index, name in enumerate(JOINT_NAMES):
        joint_summary[name] = {
            "position_abs_peak": float(np.max(np.abs(q[:, index]))),
            "target_abs_peak": float(np.max(np.abs(target[:, index]))),
            "tracking_error_abs_p95": float(np.quantile(np.abs(target[:, index] - q[:, index]), .95)),
            "speed_abs_peak": float(np.max(np.abs(v[:, index]))),
            "torque_abs_peak": (float(np.nanmax(np.abs(torque[:, index])))
                                if np.isfinite(torque[:, index]).any() else None),
        }
    return {
        "schema_version": 1,
        "samples": len(rows), "control_dt_s": dt, "duration_s": float(time[-1] - time[0]),
        "body": {
            "roll_abs_peak_deg": float(np.rad2deg(np.abs(roll).max())),
            "pitch_abs_peak_deg": float(np.rad2deg(np.abs(pitch).max())),
            "linear_accel_rms": float(np.sqrt(np.mean(np.sum(body_accel**2, axis=1)))),
            "angular_accel_rms": float(np.sqrt(np.mean(np.sum(body_angular_accel**2, axis=1)))),
        },
        "payload": {
            "accel_rms": float(np.sqrt(np.mean(np.sum(cargo_accel**2, axis=1)))),
            "accel_peak": float(np.linalg.norm(cargo_accel, axis=1).max()),
            "jerk_rms": float(np.sqrt(np.mean(np.sum(cargo_jerk**2, axis=1)))),
        },
        "support": {
            "mean_wheels": float(support.mean()), "minimum_wheels": int(support.min()),
            "below_three_fraction": float(np.mean(support < 3)),
            "below_two_fraction": float(np.mean(support < 2)),
            "normal_force_peak_N": normal_force.max(axis=0).tolist() if has_normal_force else None,
        },
        "skill": {
            "intensity_mean": float(intensity.mean()), "intensity_p95": float(np.quantile(intensity, .95)),
            "intensity_peak": float(intensity.max()),
            "wheel_lift_count": lift.sum(axis=0).tolist(),
            "physical_airborne_onset_count": lift.sum(axis=0).tolist(),
            "commanded_lift_count": commanded_lift_onset.sum(axis=0).tolist(),
            "uncommanded_airborne_onset_count": (lift & ~commanded_lift).sum(axis=0).tolist(),
        },
        "joints": joint_summary,
        "events": events,
    }


def analyze_file(trace_path: Path, output: Path | None = None) -> dict:
    trace = json.loads(Path(trace_path).read_text())
    if isinstance(trace, dict):
        trace = trace.get("rows", trace.get("samples", []))
    report = analyze_trace(trace)
    destination = output or Path(trace_path).with_name("motion-4d.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report
