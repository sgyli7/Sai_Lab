"""Sai physics/control timebase configuration for generated Godot runtimes.

The upstream Sai Godot adapter was authored for one fixed 2 kHz runtime and
therefore encoded 40 ticks per policy step and 0.5 ms per physics tick.  The
workshop supports a measured physics-rate sweep, so generated copies are
patched to derive both quantities from the engine rate while preserving the
published 50 Hz policy contract.
"""
from __future__ import annotations

from pathlib import Path
import math
import re


CONTROLLER_HZ = 50
DEFAULT_PHYSICS_HZ = 1000
SUPPORTED_PHYSICS_HZ = (100, 200, 250, 500, 1000, 2000)


def validate_physics_hz(physics_hz: int) -> int:
    physics_hz = int(physics_hz)
    if physics_hz not in SUPPORTED_PHYSICS_HZ:
        raise ValueError(
            f"Sai physics rate must be one of {SUPPORTED_PHYSICS_HZ}, got {physics_hz}"
        )
    if physics_hz % CONTROLLER_HZ:
        raise ValueError(
            f"Sai physics rate {physics_hz} cannot schedule {CONTROLLER_HZ} Hz exactly"
        )
    return physics_hz


def max_physics_steps_per_frame(physics_hz: int) -> int:
    """Allow real time down to 20 rendered FPS without an unbounded catch-up loop."""
    return max(8, math.ceil(validate_physics_hz(physics_hz) / 20))


def configure_project_timebase(project: Path, physics_hz: int) -> None:
    physics_hz = validate_physics_hz(physics_hz)
    text = project.read_text()
    replacements = {
        r"common/physics_ticks_per_second=\d+": f"common/physics_ticks_per_second={physics_hz}",
        r"common/max_physics_steps_per_frame=\d+": (
            f"common/max_physics_steps_per_frame={max_physics_steps_per_frame(physics_hz)}"
        ),
    }
    for pattern, replacement in replacements.items():
        text, count = re.subn(pattern, replacement, text, count=1)
        if count != 1:
            raise ValueError(f"Missing Godot timebase setting {pattern!r} in {project}")
    project.write_text(text)


def patch_generated_runtime(runtime: Path) -> None:
    """Remove 2 kHz assumptions from a disposable runtime made by sai_agent."""
    main_path = runtime / "main.gd"
    robot_path = runtime / "robot.gd"
    main = main_path.read_text()
    robot = robot_path.read_text()

    replacements = {
        "if robot.tick%40==0:": "if robot.is_control_tick():",
        '"physics_hz":2000': '"physics_hz":Engine.physics_ticks_per_second',
        "robot.tick*.0005": "robot.sim_time_seconds()",
        "robot.tick*0.0005": "robot.sim_time_seconds()",
    }
    if "if robot.tick%40==0:" in main:
        for old, new in replacements.items():
            main = main.replace(old, new)
    elif "if robot.is_control_tick():" not in main:
        raise ValueError(f"Unknown Sai main timebase contract in {main_path}")

    marker = "var initial_ground_height := 0.0\n"
    methods = '''var initial_ground_height := 0.0
const CONTROLLER_HZ := 50

func control_decimation() -> int:
\tvar physics_hz := Engine.physics_ticks_per_second
\tassert(physics_hz >= CONTROLLER_HZ and physics_hz % CONTROLLER_HZ == 0,
\t\t"Sai physics rate must be an integer multiple of the 50 Hz controller")
\treturn physics_hz / CONTROLLER_HZ

func is_control_tick() -> bool:
\treturn tick % control_decimation() == 0

func sim_time_seconds() -> float:
\treturn float(tick) / float(Engine.physics_ticks_per_second)
'''
    if "func control_decimation()" not in robot:
        if robot.count(marker) != 1:
            raise ValueError(f"Unknown Sai robot timebase contract in {robot_path}")
        robot = robot.replace(marker, methods, 1)
    robot = robot.replace('"time":tick*0.0005', '"time":sim_time_seconds()')

    main_path.write_text(main)
    robot_path.write_text(robot)
