from pathlib import Path

import pytest

from agenticrobot_bridge.policy_audit import (
    PolicyAuditError,
    audit_policy_bundle,
    discover_policy_paths,
)
from agenticrobot_bridge.upstream import load_upstream_lock


ROOT = Path(__file__).parents[1]


def test_policy_discovery_matches_every_locked_policy_name() -> None:
    lock = load_upstream_lock(ROOT / "upstream.lock.json")

    policies = discover_policy_paths(ROOT / ".cache" / "upstream", lock)

    assert tuple(policies) == lock.policies
    assert all(path.is_file() for path in policies.values())


def test_policy_discovery_rejects_unlocked_onnx_files(tmp_path: Path) -> None:
    lock = load_upstream_lock(ROOT / "upstream.lock.json")
    policy_directory = tmp_path / "microduck" / "policies"
    policy_directory.mkdir(parents=True)
    for name in lock.policies:
        (policy_directory / name).touch()
    (policy_directory / "surprise.onnx").touch()

    with pytest.raises(PolicyAuditError, match="Unexpected unlocked policies: surprise.onnx"):
        discover_policy_paths(tmp_path, lock)


def test_locked_policy_bundle_has_expected_shapes_and_finite_inference() -> None:
    lock = load_upstream_lock(ROOT / "upstream.lock.json")

    report = audit_policy_bundle(ROOT / ".cache" / "upstream", lock)

    assert tuple(policy.name for policy in report.policies) == lock.policies
    assert all(policy.input_name == "obs" for policy in report.policies)
    assert all(policy.input_shape == (1, 61) for policy in report.policies)
    assert all(policy.output_name == "actions" for policy in report.policies)
    assert all(policy.output_shape == (1, 14) for policy in report.policies)
    assert all(policy.output_is_finite for policy in report.policies)
    assert all(len(policy.sha256) == 64 for policy in report.policies)

    structured = report.to_dict()
    assert structured["policyCount"] == 9
    assert structured["allFinite"] is True
    assert [policy["name"] for policy in structured["policies"]] == list(lock.policies)
