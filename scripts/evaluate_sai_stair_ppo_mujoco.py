#!/usr/bin/env python3
"""Paired CPU-MuJoCo acceptance for a trained phase-free stair actor."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
from sai_agent.paths import resource_root
from sim2sim.sai_controller import MotionController
from sim2sim.sai_motion_safety import assess_motion_safety
from sim2sim.sai_trace_analysis import analyze_trace
import sai_loaded_mujoco as loaded


@contextmanager
def controller(profile: Path | None):
    original = loaded.CompliantController
    class Production(MotionController):
        def __init__(self, root, parameters=None):
            if parameters is not None:
                raise ValueError("Paired stair acceptance does not accept control parameter overrides")
            super().__init__(root, stair_profile=profile)
    loaded.CompliantController = Production
    try:
        yield
    finally:
        loaded.CompliantController = original


def lift_events(trace: list[dict], stair_start=.45, tread=.18) -> dict:
    gaps = np.array([[p[2] - g - .048 for p, g in zip(row["wheel_positions"], row["wheel_ground_heights"])] for row in trace])
    wheel_x = np.array([[p[0] for p in row["wheel_positions"]] for row in trace])
    airborne = gaps > .015
    rising = airborne & ~np.vstack([np.zeros((1, 4), bool), airborne[:-1]])
    action = np.array([row.get("policy_action", [0.] * 16) for row in trace], dtype=float)
    intensity = np.array([row.get("skill_intensity", 0.) for row in trace], dtype=float)
    commanded_height = np.maximum(action[:, 4:8], 0.) * np.clip(intensity[:, None], 0., 1.) * .080
    commanded = commanded_height > .003
    commanded_rising = commanded & ~np.vstack([np.zeros((1, 4), bool), commanded[:-1]])
    premature = rising & (stair_start - wheel_x > .10)
    commanded_premature = commanded_rising & (stair_start - wheel_x > .10)
    physical_edge_counts = np.zeros((4, 4), dtype=int)
    commanded_edge_counts = np.zeros((4, 4), dtype=int)
    for sample, wheel in zip(*np.nonzero(rising & ~premature)):
        edge = int(np.floor((wheel_x[sample, wheel] - stair_start) / tread) + 1)
        if 0 <= edge < 4:
            physical_edge_counts[wheel, edge] += 1
    for sample, wheel in zip(*np.nonzero(commanded_rising & ~commanded_premature)):
        edge = int(np.floor((wheel_x[sample, wheel] - stair_start) / tread) + 1)
        if 0 <= edge < 4:
            commanded_edge_counts[wheel, edge] += 1
    physical_redundant = np.maximum(physical_edge_counts - 1, 0)
    commanded_redundant = np.maximum(commanded_edge_counts - 1, 0)
    uncommanded_airborne = rising & ~commanded
    return {"rising_events": int(rising.sum()), "premature_events": int(commanded_premature.sum()),
            "redundant_events": int(commanded_redundant.sum()),
            "commanded_rising_events": int(commanded_rising.sum()),
            "commanded_redundant_events": int(commanded_redundant.sum()),
            "uncommanded_airborne_events": int(uncommanded_airborne.sum()),
            "physical_redundant_events": int(physical_redundant.sum()),
            "wheel_edge_counts": commanded_edge_counts.tolist(),
            "physical_wheel_edge_counts": physical_edge_counts.tolist(),
            "max_clearance_m": float(gaps.max())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1601, 1602, 1603])
    parser.add_argument("--cases", nargs="+", default=["up20", "up40", "up60"])
    parser.add_argument("--clamped", action="store_true", help="Close the physical cargo clamp in both paired runs")
    args = parser.parse_args();args.output.mkdir(parents=True, exist_ok=False)
    pairs = []
    for case in args.cases:
        for seed in args.seeds:
            reports = []
            for label, profile in [("baseline", None), ("candidate", args.profile)]:
                out = args.output / f"{case}-{seed}-{label}"
                with controller(profile):
                    report = loaded.run(case, seed, .1, None, out, clamped=args.clamped)
                trace = json.loads((out / "trace.json").read_text())
                report["lift_events"] = lift_events(trace)
                report["motion_safety"] = assess_motion_safety(trace, riser_m=float(case[2:]) / 1000.)
                report["motion_4d"] = analyze_trace(trace)
                (out / "motion-4d.json").write_text(json.dumps(report["motion_4d"], indent=2) + "\n")
                reports.append(report)
            pairs.append({"case": case, "seed": seed, "baseline": reports[0], "candidate": reports[1]})
    checks = {}
    for pair in pairs:
        name = f"{pair['case']}-{pair['seed']}";base = pair["baseline"]["metrics"];candidate = pair["candidate"]["metrics"]
        baseline_completed = bool(base["completed"] and not base["cargo_lost"] and base["min_upright"] > .6)
        checks[name + "/completed"] = bool(candidate["completed"] and not candidate["cargo_lost"] and candidate["min_upright"] > .6)
        checks[name + "/premature"] = pair["candidate"]["lift_events"]["premature_events"] == 0
        checks[name + "/redundant"] = pair["candidate"]["lift_events"]["redundant_events"] <= 4
        comfort_limit = base["cargo_accel_rms"] * 1.15 if baseline_completed else 3.25
        time_limit = max(base["duration"] * 1.5, base["duration"] + 3.) if baseline_completed else 25.
        checks[name + "/comfort"] = candidate["cargo_accel_rms"] <= comfort_limit
        checks[name + "/time"] = candidate["duration"] <= time_limit
        checks[name + "/motion_safety"] = pair["candidate"]["motion_safety"]["passed"]
        pair["acceptance_limits"] = {"baseline_completed": baseline_completed,
            "cargo_accel_rms": comfort_limit, "duration_s": time_limit}
    summary = {"passed": all(checks.values()), "checks": checks, "pairs": pairs,
        "definition": f"Same articulated CPU MuJoCo, {'physically clamped' if args.clamped else 'free'} 100 g tray payload, 50 Hz policy; candidate must first pass the continuous joint, target, action, upright and support safety envelope, then clear without premature or excessive repeated lifts, preserve cargo RMS within 15%, and finish near baseline time. Where the baseline fails, absolute limits are 3.25 m/s^2 cargo RMS and 25 s."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"passed": summary["passed"], "checks": checks}), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
