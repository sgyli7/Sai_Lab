import json
import re
from pathlib import Path

from agenticrobot_bridge.upstream import load_upstream_lock


ROOT = Path(__file__).parents[1]


def test_policy_scenarios_cover_every_locked_policy_and_both_robot_variants() -> None:
    lock = load_upstream_lock(ROOT / "upstream.lock.json")
    raw = json.loads((ROOT / "config" / "policy-scenarios.json").read_text(encoding="utf-8"))

    assert raw["schemaVersion"] == 1
    assert raw["physicsTimestepSeconds"] == 0.005
    assert raw["policyDecimation"] == 4
    assert len(raw["defaultJointPositionRad"]) == 14
    assert tuple(raw["scenarios"]) == lock.policies
    assert {scenario["robotVariant"] for scenario in raw["scenarios"].values()} == {
        "legged",
        "roller",
    }


def test_each_scenario_has_a_finite_duration_command_and_acceptance_metric() -> None:
    raw = json.loads((ROOT / "config" / "policy-scenarios.json").read_text(encoding="utf-8"))

    for name, scenario in raw["scenarios"].items():
        assert scenario["durationSeconds"] > 0, name
        assert scenario["actionScale"] > 0, name
        assert scenario["commandProgram"], name
        assert scenario["acceptance"], name
        assert all(len(step["command"]) == 13 for step in scenario["commandProgram"]), name


def test_phase_scenarios_pin_runtime_timing_contracts() -> None:
    raw = json.loads((ROOT / "config" / "policy-scenarios.json").read_text(encoding="utf-8"))

    ground_pick = raw["scenarios"]["alpha_ground_pick.onnx"]["commandProgram"][0]
    roller_crouch = raw["scenarios"]["roller_crouch.onnx"]["commandProgram"][0]

    assert ground_pick["phasePeriodSeconds"] == 4.0
    assert ground_pick["phaseEnd"] == 0.8
    assert roller_crouch["phasePeriodSeconds"] == 5.0
    assert roller_crouch["phaseEnd"] == 0.7


def test_policy_command_state_exposes_deterministic_and_fixed_time_skill_triggers() -> None:
    source = (
        ROOT
        / "TuanjieProject"
        / "Assets"
        / "MicroDuck"
        / "Runtime"
        / "PolicyCommandState.cs"
    ).read_text(encoding="utf-8")

    assert "public void TriggerSkill(float nowSeconds)" in source
    assert re.search(
        r"public void TriggerSkill\(\)\s*\{\s*TriggerSkill\(Time\.fixedTime\);\s*\}",
        source,
    )
