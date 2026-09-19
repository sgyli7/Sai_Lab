import json
from pathlib import Path

import pytest

from agenticrobot_bridge.mujoco_rollout import (
    run_all_policy_scenarios,
    run_policy_scenario,
)


ROOT = Path(__file__).parents[1]
POLICY_NAMES = tuple(
    json.loads((ROOT / "config" / "policy-scenarios.json").read_text(encoding="utf-8"))[
        "scenarios"
    ]
)


def test_headless_rollout_writes_200hz_trace_and_50hz_policy_frames(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "walking.jsonl"

    result = run_policy_scenario(
        ROOT,
        "alpha_walking.onnx",
        trace_path,
        duration_seconds=0.04,
    )

    assert result.physics_steps == 8
    assert result.policy_steps == 2
    assert result.passed is False
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert records[0]["recordType"] == "header"
    frames = [record for record in records if record["recordType"] == "frame"]
    assert len(frames) == 8
    assert [frame["policyStep"] for frame in frames] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert all(len(frame["observation"]) == 61 for frame in frames)
    assert all(len(frame["rawAction"]) == 14 for frame in frames)
    assert all(len(frame["targetPositionRad"]) == 14 for frame in frames)
    assert records[-1]["recordType"] == "result"
    assert records[-1]["passed"] is False


def test_rollout_suite_writes_deterministic_report_with_walking_calibration(
    tmp_path: Path,
) -> None:
    report = run_all_policy_scenarios(ROOT, tmp_path)

    assert report.passed is True
    assert [result.policy_name for result in report.results] == list(POLICY_NAMES)
    report_path = tmp_path / "rollout-report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload == report.to_dict()
    assert payload["summary"] == {"passed": 9, "failed": 0, "total": 9}
    assert payload["calibrations"]["alpha_walking.onnx"] == {
        "actionScale": 1.1,
        "scope": "headless MuJoCo plain-XML behavior baseline",
        "productionRobotdActionScale": 0.9,
        "officialInferPolicyDefaultActionScale": 1.0,
        "reason": (
            "Empirically required to cross the configured 0.03 m gait threshold "
            "with the plain-XML contact model."
        ),
    }
    assert all(Path(result["tracePath"]).is_file() for result in payload["results"])


@pytest.mark.parametrize("policy_name", POLICY_NAMES)
def test_official_policy_passes_its_full_headless_scenario(
    policy_name: str,
    tmp_path: Path,
) -> None:
    result = run_policy_scenario(ROOT, policy_name, tmp_path / f"{policy_name}.jsonl")

    assert result.passed, result.to_dict()
