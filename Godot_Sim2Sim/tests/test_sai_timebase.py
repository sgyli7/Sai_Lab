from pathlib import Path

import pytest

from sim2sim.sai_timebase import (
    DEFAULT_PHYSICS_HZ,
    configure_project_timebase,
    max_physics_steps_per_frame,
    patch_generated_runtime,
    validate_physics_hz,
)


def test_supported_rates_preserve_exact_50_hz_control() -> None:
    assert DEFAULT_PHYSICS_HZ == 1000
    assert [validate_physics_hz(rate) // 50 for rate in (100, 200, 250, 500, 1000, 2000)] == [2, 4, 5, 10, 20, 40]
    assert max_physics_steps_per_frame(500) == 25
    with pytest.raises(ValueError):
        validate_physics_hz(60)


def test_generated_adapter_uses_engine_timebase(tmp_path: Path) -> None:
    (tmp_path / "main.gd").write_text(
        'if robot.tick%40==0:\n\tpass\nvar a={"physics_hz":2000}\nprint(robot.tick*.0005)\n'
    )
    (tmp_path / "robot.gd").write_text(
        'var initial_ground_height := 0.0\nfunc state():\n\treturn {"time":tick*0.0005}\n'
    )
    patch_generated_runtime(tmp_path)
    assert "robot.is_control_tick()" in (tmp_path / "main.gd").read_text()
    assert "Engine.physics_ticks_per_second" in (tmp_path / "main.gd").read_text()
    robot = (tmp_path / "robot.gd").read_text()
    assert "func control_decimation()" in robot
    assert '"time":sim_time_seconds()' in robot


def test_project_configuration(tmp_path: Path) -> None:
    project = tmp_path / "project.godot"
    project.write_text(
        "common/physics_ticks_per_second=2000\ncommon/max_physics_steps_per_frame=100\n"
    )
    configure_project_timebase(project, 250)
    assert "common/physics_ticks_per_second=250" in project.read_text()
    assert "common/max_physics_steps_per_frame=13" in project.read_text()
