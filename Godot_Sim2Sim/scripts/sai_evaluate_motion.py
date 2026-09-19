"""Measure steady flat-ground rolling quality from actual 50 Hz Sai telemetry.

Use a constant drive command and omit acceleration/crouch transitions from the
selected window. Wheel spin is deliberately excluded from leg joint motion.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(samples, start=2.0, end=5.0, expected_command=None):
    rows = [r for r in samples if start <= r["time"] <= end]
    if len(rows) < 50:
        raise ValueError("Need at least one second of 50 Hz steady-drive samples")
    times = np.array([r["time"] for r in rows])
    if not np.allclose(np.diff(times), .02, atol=1e-5):
        raise ValueError("Quality window must be one uninterrupted 50 Hz session")
    legs = [i for i in range(16) if i % 4 != 3]
    velocities = np.array([r["v"] for r in rows])[:, legs]
    positions = np.array([r["q"] for r in rows])[:, legs]
    result = {
        "samples": len(rows),
        "leg_velocity_rms_rad_s": float(np.sqrt(np.mean(velocities ** 2))),
        "leg_position_max_peak_to_peak_rad": float(np.ptp(positions, axis=0).max()),
        "leg_step_rms_rad": float(np.sqrt(np.mean(np.diff(positions, axis=0) ** 2))),
    }
    checks = {"leg_velocity_rms_below_0_5": result["leg_velocity_rms_rad_s"] < .5}
    if expected_command is not None:
        checks["requested_command_held"] = bool(np.allclose(
            [r["command"] for r in rows], expected_command, atol=1e-6))
    if "wheel_positions" in rows[0]:
        # In a level rolling stance, wheel centres are at the 48 mm radius.
        heights = np.array([r["wheel_positions"] for r in rows])[:, :, 2]
        result["max_wheel_lift_m"] = float(np.max(heights) - .048)
        checks["wheel_lift_below_12mm"] = result["max_wheel_lift_m"] < .012
        result["mean_wheels_supported"] = float(np.mean([r["wheels_supported"] for r in rows]))
    result["checks"] = checks
    result["passed"] = all(checks.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--start", type=float, default=2.)
    parser.add_argument("--end", type=float, default=5.)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "WZ", "CROUCH"),
                        help="Require this command throughout the quality window; catches lost replay keys")
    args = parser.parse_args()
    if args.trace.suffix == ".jsonl":
        samples = [json.loads(line)["state"] for line in args.trace.read_text().splitlines()]
    else:
        samples = json.loads(args.trace.read_text())["samples"]
    result = evaluate(samples, args.start, args.end, args.command)
    output = json.dumps(result, indent=2) + "\n"
    print(output, end="")
    if args.out:
        args.out.write_text(output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
