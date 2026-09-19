#!/usr/bin/env python3
"""Run the accepted stair profile through Godot-native ONNX/Jolt courses."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from evaluate_sai_stair_ppo_mujoco import lift_events
from sim2sim.sai_motion_safety import assess_motion_safety


ROOT = Path(__file__).resolve().parents[1]


def cargo_accel_rms(samples: list[dict]) -> float | None:
    rows = [sample for sample in samples if len(sample.get("object_linear_world", [])) == 3
            and abs(float(sample.get("command", [0.])[0])) > .015]
    if len(rows) < 3:
        return None
    filtered = np.zeros(3); values = []
    previous = np.asarray(rows[0]["object_linear_world"], dtype=float)
    previous_time = float(rows[0]["time"])
    for row in rows[1:]:
        now = float(row["time"]); dt = now - previous_time
        velocity = np.asarray(row["object_linear_world"], dtype=float)
        if dt > 0:
            acceleration = (velocity - previous) / dt
            filtered += (acceleration - filtered) * dt / (.015 + dt)
            values.append(float(np.dot(filtered, filtered)))
        previous = velocity; previous_time = now
    return float(np.sqrt(np.mean(values))) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--godot", default=os.environ.get("GODOT") or "godot")
    parser.add_argument("--cases", nargs="+", default=["up20", "up40", "up60"])
    parser.add_argument("--physics-hz", type=int, choices=(100, 200, 250, 500, 1000, 2000), default=500)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    profile_path = ROOT / "src/sim2sim/assets/sai/upstream/experimental" / f"{args.profile_name}.json"
    profile = json.loads(profile_path.read_text())
    expected_contract = profile["contract"]
    expected_observation_size = int(profile["observation_size"])
    plan = ROOT / "docs/workshop-hub-20260912/plans/stairs.json"
    rows = []
    checks = {}
    for case in args.cases:
        destination = args.output / case
        command = [sys.executable, "-m", "sim2sim.workshop", "--godot-bin", args.godot,
            "--runtime-dir", str(args.runtime), "--robot", "sai", "--task", case,
            "--sai-physics-hz", str(args.physics_hz),
            "--sai-stair-profile", args.profile_name, "--allow-quarantined-stair-profile",
            "--headless", "--fast-check",
            "--plan", str(plan), "--output", str(destination)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=240)
        (args.output / f"{case}.log").write_text(result.stdout + result.stderr)
        task_path = destination / "task.json"
        if result.returncode != 0 or not task_path.is_file():
            checks[case + "/process"] = False
            rows.append({"case": case, "returncode": result.returncode})
            continue
        task = json.loads(task_path.read_text())
        samples = task.get("samples", [])
        active = [sample for sample in samples
                  if sample.get("controller_stage") in ("stairs", "task_skill")]
        events = lift_events(samples)
        minimum = min((float(sample.get("upright", 0.)) for sample in samples), default=0.)
        row = {"case": case, "returncode": result.returncode,
            "cleared_at_s": task.get("cleared_at"), "duration_s": task.get("duration_s"),
            "min_upright": minimum, "active_samples": len(active), "lift_events": events,
            "backend": task.get("controller_backend"), "cargo_accel_rms": cargo_accel_rms(samples),
            "cargo_retained": (bool(samples) and bool(samples[-1].get("cargo_inside", False))
                               and bool(samples[-1].get("cargo_supported", False))),
            "motion_safety": assess_motion_safety(
                samples, riser_m=float(task.get("riser", 0.0)),
                ignored_action_indices=((15,) if expected_contract == "sai-task-space-skills-v6" else ()),
            )}
        rows.append(row)
        checks[case + "/process"] = True
        checks[case + "/backend"] = task.get("controller_backend") == "godot-native-onnxruntime"
        checks[case + "/completed"] = float(task.get("cleared_at", -1.)) >= 0. and minimum > .6
        checks[case + "/cargo"] = row["cargo_retained"] and row["cargo_accel_rms"] is not None
        checks[case + "/contract"] = bool(active) and all(
            sample.get("contract_id") == expected_contract
            and len(sample.get("policy_observation", [])) == expected_observation_size
            and len(sample.get("policy_action", [])) == 16 for sample in active)
        checks[case + "/premature"] = events["premature_events"] == 0
        checks[case + "/redundant"] = events["redundant_events"] <= 4
        checks[case + "/motion_safety"] = row["motion_safety"]["passed"]
    summary = {"passed": all(checks.values()), "checks": checks, "courses": rows,
        "physics_hz": args.physics_hz, "controller_hz": 50,
        "definition": f"Godot 4/Jolt, {args.physics_hz} Hz physics, 50 Hz in-process native ONNX; every candidate must pass the continuous joint, target, action, upright and support safety envelope before completion and lift-event metrics are considered; stair samples must use the phase-free {expected_observation_size}x16 contract."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"passed": summary["passed"], "checks": checks}), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
