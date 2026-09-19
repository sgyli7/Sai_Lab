import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from agenticrobot_bridge.onnx_compat import OnnxCompatibilityError, convert_policy_bundle
from agenticrobot_bridge.upstream import UpstreamLock, load_upstream_lock


ROOT = Path(__file__).parents[1]
UPSTREAM_ROOT = ROOT / ".cache" / "upstream"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_locked_bundle_converts_to_deterministic_opset9_with_parity_artifacts(
    tmp_path: Path,
) -> None:
    lock = load_upstream_lock(ROOT / "upstream.lock.json")
    originals = UPSTREAM_ROOT / "microduck" / "policies"
    original_hashes = {name: _sha256(originals / name) for name in lock.policies}

    first = convert_policy_bundle(UPSTREAM_ROOT, lock, tmp_path / "first")
    second = convert_policy_bundle(UPSTREAM_ROOT, lock, tmp_path / "second")

    assert second.to_dict() == first.to_dict()
    assert first.target_opset == 9
    assert first.fixture_names == ("zeros", "ramp", "seeded")
    assert tuple(policy.name for policy in first.policies) == lock.policies
    assert all(policy.max_abs_error <= 1e-5 for policy in first.policies)
    assert all(policy.fixture_count == 3 for policy in first.policies)

    for name in lock.policies:
        first_path = tmp_path / "first" / "policies" / name
        second_path = tmp_path / "second" / "policies" / name
        assert first_path.read_bytes() == second_path.read_bytes()
        converted = onnx.load(first_path, load_external_data=False)
        assert [(item.domain, item.version) for item in converted.opset_import] == [("", 9)]
        assert _sha256(originals / name) == original_hashes[name]

    fixtures = json.loads((tmp_path / "first" / "parity-fixtures.json").read_text())
    report = json.loads((tmp_path / "first" / "compatibility-report.json").read_text())
    assert fixtures["inputShape"] == [1, 61]
    assert fixtures["outputShape"] == [1, 14]
    assert [
        (case["policyName"], case["fixtureName"]) for case in fixtures["cases"]
    ] == [
        (policy_name, fixture_name)
        for policy_name in lock.policies
        for fixture_name in first.fixture_names
    ]
    assert all(
        len(case["input"]) == 61
        and len(case["expectedOutput"]) == 14
        and np.isfinite(case["input"]).all()
        and np.isfinite(case["expectedOutput"]).all()
        for case in fixtures["cases"]
    )
    assert all("expectedOutputs" not in case for case in fixtures["cases"])
    assert report == first.to_dict()


def test_conversion_rejects_an_operator_outside_the_proven_policy_family(
    tmp_path: Path,
) -> None:
    policy_directory = tmp_path / "upstream" / "microduck" / "policies"
    policy_directory.mkdir(parents=True)
    graph = helper.make_graph(
        [helper.make_node("Relu", ["obs"], ["actions"])],
        "unsafe",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 61])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.save(model, policy_directory / "unsafe.onnx")
    lock = UpstreamLock(schema_version=1, repositories={}, policies=("unsafe.onnx",))

    with pytest.raises(OnnxCompatibilityError, match="unsupported operator.*Relu"):
        convert_policy_bundle(tmp_path / "upstream", lock, tmp_path / "converted")


def test_conversion_rejects_unsafe_attributes_on_an_allowed_operator(
    tmp_path: Path,
) -> None:
    policy_directory = tmp_path / "upstream" / "microduck" / "policies"
    policy_directory.mkdir(parents=True)
    weights = numpy_helper.from_array(np.ones((14, 61), dtype=np.float32), "weights")
    bias = numpy_helper.from_array(np.zeros(14, dtype=np.float32), "bias")
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["obs", "weights", "bias"], ["actions"], transB=2)],
        "unsafe_attribute",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
        [weights, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.save(model, policy_directory / "unsafe.onnx")
    lock = UpstreamLock(schema_version=1, repositories={}, policies=("unsafe.onnx",))

    with pytest.raises(OnnxCompatibilityError, match="unsafe attribute.*transB"):
        convert_policy_bundle(tmp_path / "upstream", lock, tmp_path / "converted")
