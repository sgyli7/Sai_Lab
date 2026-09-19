"""Verify the deployed Sai input, policy and physical motion in both game scenes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "W": (["W"], [.5, 0., 0.]),
    "W_shift": (["W", "Shift"], [.5, 0., 1.]),
    "SA": (["S", "A"], [-.5, -.45, 0.]),
    "SD": (["S", "D"], [-.5, .45, 0.]),
    "SA_shift": (["S", "A", "Shift"], [-.5, -.45, 1.]),
    "SD_shift": (["S", "D", "Shift"], [-.5, .45, 1.]),
    "WA": (["W", "A"], [.5, .45, 0.]),
    "A": (["A"], [0., .45, 0.]),
}


def evaluate(rows, expected):
    moving = [r for r in rows if 2. <= r["state"]["time"] <= 2.98]
    stopped = [r for r in rows if 4. <= r["state"]["time"] <= 4.98]
    obs = np.array([r["command"]["policy_observation"] for r in moving])
    vx, wz = float(obs[:, 3].mean()), float(obs[:, 8].mean())
    legs = [i for i in range(16) if i % 4 != 3]
    rms = float(np.sqrt(np.mean(np.array([r["state"]["v"] for r in moving])[:, legs] ** 2)))
    lift = float(np.max([p[2] for r in moving for p in r["state"]["wheel_positions"]]) - .048)
    policy_hash = hashlib.sha256((ROOT / "src/sim2sim/assets/sai/flat-motion-v1.onnx").read_bytes()).hexdigest()
    checks = {
        "sustained_input": len(moving) >= 49 and all(np.allclose(r["state"]["command"], expected) for r in moving),
        "physical_drive": abs(vx - expected[0]) < .07,
        "physical_turn": abs(wz) < .05 if expected[1] == 0 else wz * (1 if expected[1] > 0 else -1) > .2,
        "leg_velocity_rms_below_0_5": rms < .5,
        "wheel_lift_below_12mm": lift < .012,
        "upright": min(r["state"]["base_rotation_columns"][2][2] for r in rows) > .98,
        "released_and_stopped": len(stopped) >= 49 and all(r["state"]["command"] == [0., 0., 0.] for r in stopped)
            and max(abs(r["command"]["policy_observation"][3]) for r in stopped) < .03
            and max(abs(r["command"]["policy_observation"][8]) for r in stopped) < .03,
        "deployed_policy": all(r["command"].get("flat_policy_id") == "sai-flat-motion-v1"
                               and r["command"].get("flat_policy_sha256") == policy_hash
                               and r["command"].get("driving_profile") == "sai-driving-20260913" for r in rows),
        "terrain_path_connected": all(len(r["state"].get("terrain_path_heights", [])) == 15 for r in rows),
        "flat_actor_selected": all(r["command"].get("stage") != "stairs" for r in moving),
    }
    return dict(checks=checks, passed=all(checks.values()), actual_forward_m_s=vx,
                actual_yaw_rad_s=wz, leg_velocity_rms_rad_s=rms, max_wheel_lift_m=lift,
                policy_sha256=policy_hash)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, help="Reuse an isolated, already imported runtime")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--scenes", nargs="+", choices=["science_station", "workshop"], default=["science_station", "workshop"])
    args = parser.parse_args()
    output = args.output.resolve()
    reports = {}
    for scene in args.scenes:
        for case in args.cases:
            name = f"{scene}-{case}"
            directory = output / name
            directory.mkdir(parents=True, exist_ok=True)
            keys, expected = CASES[case]
            plan = dict(seconds=5., events=[dict(at=t, key=k, pressed=down)
                        for t, down in [(1., True), (3., False)] for k in keys])
            if scene == "science_station":
                plan["initial_position"] = [10., 2.]
            plan_path = directory / "plan.json"
            plan_path.write_text(json.dumps(plan))
            run = directory / "run"
            command = [sys.executable, str(ROOT / "scripts/run_science_check.py"), "--scene", scene,
                       "--robot", "sai", "--headless", "--fast-check", "--plan", str(plan_path),
                       "--runtime-dir", str((args.runtime_dir or output / "runtime").resolve()), "--output", str(run)]
            child = subprocess.run(command, cwd=ROOT, env=dict(os.environ, SIM2SIM_ROOT=str(ROOT),
                                   PYTHONPATH=str(ROOT / "src")), timeout=240)
            if child.returncode == 0:
                rows = [json.loads(line) for line in (run / "sai-trace.jsonl").read_text().splitlines()]
                result = evaluate(rows, expected)
                result["checks"]["scene"] = json.loads((run / "hub.json").read_text())["scene"] == scene
                result["passed"] = all(result["checks"].values())
            else:
                result = dict(passed=False, exit_code=child.returncode)
            reports[name] = result
            print(name, json.dumps(result), flush=True)
            (output / "acceptance.json").write_text(json.dumps(reports, indent=2) + "\n")
    return 0 if all(r["passed"] for r in reports.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
