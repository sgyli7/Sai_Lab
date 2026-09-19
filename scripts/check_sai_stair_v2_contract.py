#!/usr/bin/env python3
"""Check Python and Godot-native ONNX inference for a phase-free Sai actor."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from sai_agent.paths import resource_root
from sim2sim.sai_controller import MotionController


ROOT = Path(__file__).resolve().parents[1]


def state(case: str, x: float, step: float) -> dict:
    terrain = [0.] * 9 + [step] * 15
    path = [0.] * 6 + [step] * 9
    return dict(fixture_id=case, robot_id="Sai_Agent_001", physics_owner="Godot/Jolt", time=1.25,
        q=np.linspace(-.04, .04, 25).tolist(), v=np.linspace(.08, -.06, 25).tolist(), command=[.16, 0., 0.],
        terrain_heights=terrain, terrain_path_heights=path,
        terrain_edge_heights=[0.] * 69 + [step] * 69,
        wheel_ground_heights=[0., 0., 0., 0.],
        wheel_positions=[[x + .15, .146, .049], [x + .15, -.146, .049],
                         [x - .15, .146, .049], [x - .15, -.146, .049]],
        base_position=[x, 0., .2192], base_rotation_columns=np.eye(3).tolist(),
        base_linear_world=[.11, 0., .002], base_angular_world=[.01, -.02, .005],
        stair_course=True, experimental_profile=True, arm_gravity_bias=[0.] * 6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--godot", default=os.environ.get("GODOT") or shutil.which("godot"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    actor = args.profile.parent / profile["actor"]
    controller = MotionController(resource_root(), stair_profile=args.profile, suspension_profile="off")
    cases = []
    for item in [state("approach", -.20, .02), state("edge", -.06, .04),
                 state("high-edge", -.06, .06), state("high-settle", .08, 0.)]:
        case_id = item.pop("fixture_id")
        result = controller.command(item)
        cases.append({"id": case_id, "state": item, "expected": {key: result[key] for key in
            ("policy_observation", "policy_action", "target_leg", "wheel_speed", "effective_crouch",
             "cargo_target_rad", "stage", "stair_profile", "contract_id")}})
    fixture = {"schema_version": 1, "groups": [{"profile": "res://sai_policy/experimental/profile.json", "cases": cases}]}
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sai-stair-v2-native-") as directory:
        runtime = Path(directory) / "runtime"
        subprocess.run(["cp", "--reflink=auto", "-a", str(args.runtime), str(runtime)], check=True)
        target = runtime / "sai_policy/experimental"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(actor, target / actor.name)
        deployed = dict(profile, actor=actor.name)
        (target / "profile.json").write_text(json.dumps(deployed, indent=2) + "\n")
        shutil.copy2(ROOT / "godot/sai/native_controller.gd", runtime / "sai/native_controller.gd")
        shutil.copy2(ROOT / "godot/sai/terrain_suspension.gd", runtime / "sai/terrain_suspension.gd")
        native_binary = ROOT / "godot/native/bin/libmicroduck_policy.linux.debug.arm64.so"
        if native_binary.is_file():
            shutil.copy2(native_binary, runtime / "native/bin/libmicroduck_policy.linux.debug.arm64.so")
        shutil.copy2(ROOT / "godot/tests/sai_native_contract_probe.gd", runtime / "tests/sai_native_contract_probe.gd")
        fixture_path = Path(directory) / "fixture.json"
        fixture_path.write_text(json.dumps(fixture, separators=(",", ":")))
        result = subprocess.run([args.godot, "--headless", "--path", str(runtime),
            "--script", "res://tests/sai_native_contract_probe.gd", "--", f"--fixtures={fixture_path}"],
            capture_output=True, text=True, timeout=120)
        log = result.stdout + result.stderr
        (args.output / "godot.log").write_text(log)
        rows = [line.split("SAI_NATIVE_CONTRACT_RESULT ", 1)[1] for line in log.splitlines()
                if "SAI_NATIVE_CONTRACT_RESULT " in line]
        if len(rows) != 1:
            raise RuntimeError("Godot did not emit a single native contract result")
        report = json.loads(rows[0]); report["returncode"] = result.returncode
        report["profile"] = profile["id"]; report["onnx_sha256"] = profile["onnx_sha256"]
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return 0 if result.returncode == 0 and report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
