#!/usr/bin/env python3
"""Fixed-state Python-oracle versus in-process Godot Sai contract check."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation
from sai_agent.paths import resource_root
from sim2sim.sai_controller import MotionController

ROOT = Path(__file__).resolve().parents[1]


def state(case: str, time_s: float, command: list[float], *, yaw=0.0, roll=0.0,
          terrain=None, stair_course=False, wheel_ground=None) -> dict:
    rotation = Rotation.from_euler("xyz", [roll, 0.0, yaw]).as_matrix()
    q = np.linspace(-0.08, 0.08, 25)
    v = np.linspace(0.12, -0.09, 25)
    heights = np.zeros(24) if terrain is None else np.asarray(terrain, dtype=float)
    result = dict(
        fixture_id=case, robot_id="Sai_Agent_001", physics_owner="Godot/Jolt", time=time_s,
        q=q.tolist(), v=v.tolist(), command=command, terrain_heights=heights.tolist(),
        base_position=[0.31, -0.07, 0.2192], base_rotation_columns=rotation.T.tolist(),
        base_linear_world=[0.16, -0.03, 0.01], base_angular_world=[0.02, -0.04, 0.17],
        stair_course=stair_course, terrain_path_heights=[0.0] * 15,
        terrain_edge_heights=[0.0] * 138, arm_gravity_bias=[0.0] * 6,
    )
    if wheel_ground is not None:
        result["wheel_ground_heights"] = wheel_ground
    return result


def expected(controller: MotionController, item: dict) -> dict:
    result = controller.command(item)
    return {key: result[key] for key in (
        "policy_observation", "policy_action", "target_leg", "wheel_speed",
        "effective_crouch", "stage", "stair_profile", "contract_id")}


def group(profile: Path | None, cases: list[dict]) -> dict:
    controller = MotionController(resource_root(), stair_profile=profile)
    return dict(
        profile="" if profile is None else "res://sai_policy/experimental/" + profile.name,
        cases=[dict(id=item.pop("fixture_id"), state=item, expected=expected(controller, item)) for item in cases],
    )


def fixtures(extra_profile: Path | None = None) -> dict:
    stair = np.repeat([0.0, 0.0, 0.0, 0.04, 0.04, 0.04, 0.04, 0.04], 3)
    policies = resource_root() / "policies" / "experimental"
    groups = [
        group(None, [
            state("flat_heading", .20, [.18, .22, 0.0], yaw=.11, roll=.04),
            state("flat_crouch", .22, [.14, -.16, 1.0], yaw=.10, roll=-.03),
            state("park_filter", .24, [0.0, 0.0, 0.0], yaw=.09),
            state("terrain_suspension", .44, [.18, 0.0, 0.0], yaw=.03, roll=.04,
                  wheel_ground=[0.0, .006, -.004, .002]),
            state("stairs_default", .64, [.16, .0, 0.0], terrain=stair, stair_course=True),
        ]),
        group(policies / "ascent60.json", [
            state("stairs_ascent60", .82, [.16, .0, 0.0], terrain=stair, stair_course=True),
        ]),
        group(policies / "descent60.json", [
            state("stairs_descent60", .90, [.16, .0, 0.0], yaw=.08, terrain=stair, stair_course=True),
        ]),
    ]
    if extra_profile is not None:
        groups.append(group(extra_profile, [
            state("task_space_v6", .72, [.16, .0, 0.0], terrain=stair,
                  stair_course=True, wheel_ground=[0.0, 0.0, 0.0, 0.0]),
        ]))
    return dict(schema_version=1, groups=groups)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", default=os.environ.get("GODOT") or shutil.which("godot"))
    parser.add_argument("--runtime", type=Path, default=ROOT / "results/workshop-hub/runtime")
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args(argv)
    if not args.godot or not (args.runtime / "project.godot").is_file():
        raise SystemExit("Prepare the runtime first: ./run-workshop.sh --prepare-only")
    shutil.copy2(ROOT / "godot/tests/sai_native_contract_probe.gd", args.runtime / "tests/sai_native_contract_probe.gd")
    shutil.copy2(ROOT / "godot/sai/native_controller.gd", args.runtime / "sai/native_controller.gd")
    shutil.copy2(ROOT / "godot/sai/task_space_impedance.gd", args.runtime / "sai/task_space_impedance.gd")
    native_binary = ROOT / "godot/native/bin/libmicroduck_policy.linux.debug.arm64.so"
    if native_binary.is_file():
        shutil.copy2(native_binary, args.runtime / "native/bin/libmicroduck_policy.linux.debug.arm64.so")
    if args.profile is not None:
        profile = json.loads(args.profile.read_text())
        destination = args.runtime / "sai_policy/experimental"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.profile, destination / args.profile.name)
        shutil.copy2(args.profile.parent / profile["actor"], destination / profile["actor"])
    with tempfile.TemporaryDirectory(prefix="sai-native-contract-") as directory:
        path = Path(directory) / "fixtures.json"
        path.write_text(json.dumps(fixtures(args.profile), separators=(",", ":")))
        result = subprocess.run([args.godot, "--headless", "--path", str(args.runtime),
                                 "--script", "res://tests/sai_native_contract_probe.gd", "--",
                                 f"--fixtures={path}"], text=True, timeout=120)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
