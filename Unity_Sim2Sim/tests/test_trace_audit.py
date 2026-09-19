import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from agenticrobot_bridge.trace_audit import TraceAuditError, audit_rollout_trace


def _write_slice_policy(path: Path) -> None:
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([14], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    steps = numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps")
    graph = helper.make_graph(
        [helper.make_node("Slice", ["obs", "starts", "ends", "axes", "steps"], ["actions"])],
        "trace-test-policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
        [starts, ends, axes, steps],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def _trace(policy_name: str, action_delta: float = 0.0) -> str:
    observation = np.linspace(-1.0, 1.0, 61, dtype=np.float32)
    action = observation[:14].copy()
    action[3] += action_delta
    header = {
        "recordType": "header",
        "schemaVersion": 1,
        "producer": "tuanjie",
        "policyName": policy_name,
        "policySha256": "",
        "scene": "MicroDuckMvp.unity",
        "sceneSha256": "",
        "role": "stand",
        "robotVariant": "legged",
        "physicsTimestepSeconds": 0.005,
        "policyDecimation": 4,
        "observationSize": 61,
        "actionSize": 14,
        "servoNames": [f"servo{index}" for index in range(14)],
        "passiveWheelNames": [],
    }
    frame = {
        "recordType": "frame",
        "physicsStep": 0,
        "policyStep": 0,
        "timeSeconds": 0.005,
        "rootPosition": [0.0, 0.0, 0.125],
        "rootQuaternionWxyz": [1.0, 0.0, 0.0, 0.0],
        "rootLinearVelocity": [0.0, 0.0, 0.0],
        "rootAngularVelocity": [0.0, 0.0, 0.0],
        "jointPositionRad": [0.0] * 14,
        "jointVelocityRadPerSecond": [0.0] * 14,
        "passiveWheelVelocityRadPerSecond": [],
        "observation": observation.tolist(),
        "rawAction": action.tolist(),
        "targetPositionRad": [0.0] * 14,
        "command": [0.0] * 13,
        "contacts": [],
    }
    result = {"recordType": "result", "passed": True}
    return "\n".join(json.dumps(record) for record in (header, frame, result)) + "\n"


def test_audits_tuanjie_trace_schema_and_recomputes_policy_output(tmp_path: Path) -> None:
    policy_directory = tmp_path / "policies"
    policy_directory.mkdir()
    policy_path = policy_directory / "test.onnx"
    _write_slice_policy(policy_path)
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(_trace(policy_path.name), encoding="utf-8")

    report = audit_rollout_trace(
        trace_path,
        policy_directory,
        expected_producer="tuanjie",
        max_action_error=1e-6,
    )

    assert report.producer == "tuanjie"
    assert report.policy_name == "test.onnx"
    assert report.frame_count == 1
    assert report.policy_step_count == 1
    assert report.max_abs_action_error == pytest.approx(0.0)
    assert report.passed is True


def test_rejects_trace_when_recorded_action_does_not_match_policy(tmp_path: Path) -> None:
    policy_directory = tmp_path / "policies"
    policy_directory.mkdir()
    _write_slice_policy(policy_directory / "test.onnx")
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(_trace("test.onnx", action_delta=0.01), encoding="utf-8")

    with pytest.raises(TraceAuditError, match="action parity"):
        audit_rollout_trace(trace_path, policy_directory, max_action_error=1e-5)


def test_rejects_non_finite_or_wrong_width_trace_data(tmp_path: Path) -> None:
    policy_directory = tmp_path / "policies"
    policy_directory.mkdir()
    _write_slice_policy(policy_directory / "test.onnx")
    trace_path = tmp_path / "trace.jsonl"
    records = [json.loads(line) for line in _trace("test.onnx").splitlines()]
    records[1]["observation"] = [0.0] * 60
    trace_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    with pytest.raises(TraceAuditError, match="observation.*61"):
        audit_rollout_trace(trace_path, policy_directory)
