"""Reproduce Sai interactive steering, crouch drift, and leg jitter in native Jolt."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "idle": [],
    "W": ["W"],
    "A": ["A"],
    "WA": ["W", "A"],
    "W_shift": ["W", "Shift"],
    "pick": ["G"],
    "cancel": ["G"],
    "pick_complete": ["G"],
}
LEG_INDICES = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
WHEEL_INDICES = [3, 7, 11, 15]
WHEEL_SIDES = np.asarray([1.0, -1.0, 1.0, -1.0])


def _yaw(row: dict) -> float:
    columns = row["base_rotation_columns"]
    return math.atan2(float(columns[0][1]), float(columns[0][0]))


def evaluate(trace: dict, case: str) -> dict:
    rows = trace["rows"]
    active = [r for r in rows if 1.5 <= float(r["time"]) <= 3.15]
    if len(active) < 60:
        return {"passed": False, "checks": {"captured_50hz_state": False}, "samples": len(active)}
    yaws = np.unwrap(np.asarray([_yaw(r) for r in active]))
    positions = np.asarray([r["base_position"] for r in active], dtype=float)
    leg_v = np.asarray([[r["v"][i] for i in LEG_INDICES] for r in active], dtype=float)
    wheel_v = np.asarray([[r["v"][i] for i in WHEEL_INDICES] for r in active], dtype=float) * WHEEL_SIDES
    yaw_delta = float(yaws[-1] - yaws[0])
    lateral = float(positions[-1, 1] - positions[0, 1])
    leg_rms = float(np.sqrt(np.mean(leg_v * leg_v)))
    leg_p95 = float(np.percentile(np.abs(leg_v), 95))
    wheel_side_bias = float(abs(wheel_v[:, [0, 2]].mean() - wheel_v[:, [1, 3]].mean()) * .048)
    checks = {
        "captured_50hz_state": len(active) >= 60,
        "upright": min(float(r["upright"]) for r in active) > .97,
        "leg_jitter_rms_below_0_20": leg_rms < .20,
        "leg_jitter_p95_below_0_40": leg_p95 < .40,
    }
    if case in ("W", "W_shift"):
        checks.update({
            "straight_yaw_drift_below_0_05rad": abs(yaw_delta) < .05,
            "straight_lateral_drift_below_0_04m": abs(lateral) < .04,
            "wheel_side_bias_below_0_04m_s": wheel_side_bias < .04,
        })
    elif case in ("A", "WA"):
        checks["left_input_turns_left"] = yaw_delta > (.15 if case == "A" else .35)
    elif case in ("pick", "cancel", "pick_complete"):
        history = [event for session in trace.get("grab_sessions", []) for event in session.get("history", [])]
        checks["pick_input_started_object_interaction"] = any(event.get("event") == "start" for event in history)
        if case == "cancel":
            checks["cancel_input_stopped_interaction"] = any(event.get("event") == "cancel" for event in history)
        if case == "pick_complete":
            deliveries = [event for session in trace.get("grab_sessions", []) for event in session.get("deliveries", [])]
            checks.update({
                "object_attached_to_gripper": any(event.get("event") == "attach" for event in history),
                "object_released_from_gripper": any(event.get("event") == "release" for event in history),
                "object_delivered_to_cargo": len(deliveries) == 1 and bool(deliveries[0].get("success")),
            })
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "samples": len(active),
        "yaw_delta_rad": yaw_delta,
        "lateral_delta_m": lateral,
        "leg_velocity_rms_rad_s": leg_rms,
        "leg_velocity_p95_rad_s": leg_p95,
        "wheel_side_bias_m_s": wheel_side_bias,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for case in args.cases:
        directory = output / case
        directory.mkdir(parents=True, exist_ok=True)
        events = []
        for key in CASES[case]:
            events.extend(({"at": 1.0, "key": key, "pressed": True},
                           {"at": 3.35, "key": key, "pressed": False}))
        if case == "cancel":
            events.extend(({"at": 2.0, "key": "X", "pressed": True},
                           {"at": 2.1, "key": "X", "pressed": False}))
        events.sort(key=lambda event: event["at"])
        scene = "workshop" if case in ("pick", "cancel", "pick_complete") else "science_station"
        task = "sort" if case in ("pick", "cancel", "pick_complete") else "drive"
        plan = {"seconds": 50.0 if case == "pick_complete" else 4.0, "events": events}
        if case == "pick_complete":
            plan["finish_on_grab"] = True
        if scene == "science_station":
            plan["initial_position"] = [10.0, 2.0]
        plan_path = directory / "plan.json"
        plan_path.write_text(json.dumps(plan) + "\n")
        command = [str(ROOT / "run-native.sh"), "--headless", "--fast-check",
                   "--scene", scene, "--robot", "sai", "--task", task,
                   "--plan", str(plan_path), "--output", str(directory)]
        completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), timeout=90)
        trace = directory / "sai-native-trace.json"
        if completed.returncode or not trace.is_file():
            result = {"passed": False, "exit_code": completed.returncode,
                      "checks": {"native_run_and_trace": False}}
        else:
            result = evaluate(json.loads(trace.read_text()), case)
        reports[case] = result
        print(case, json.dumps(result, separators=(",", ":")), flush=True)
    (output / "acceptance.json").write_text(json.dumps(reports, indent=2) + "\n")
    return 0 if all(report["passed"] for report in reports.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
