"""Schema and policy-output audit for MuJoCo/Tuanjie rollout JSONL traces."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


OBSERVATION_SIZE = 61
ACTION_SIZE = 14
COMMAND_SIZE = 13


class TraceAuditError(RuntimeError):
    """Raised when a rollout trace breaks its schema or inference contract."""


@dataclass(frozen=True, slots=True)
class TraceAuditReport:
    """Machine-readable evidence from one cross-simulator trace audit."""

    trace_path: Path
    trace_sha256: str
    producer: str
    policy_name: str
    frame_count: int
    policy_step_count: int
    max_abs_action_error: float
    tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "tracePath": self.trace_path.as_posix(),
            "traceSha256": self.trace_sha256,
            "producer": self.producer,
            "policyName": self.policy_name,
            "frameCount": self.frame_count,
            "policyStepCount": self.policy_step_count,
            "maxAbsActionError": self.max_abs_action_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


def audit_rollout_trace(
    trace_path: str | Path,
    policy_directory: str | Path,
    *,
    expected_producer: str | None = None,
    max_action_error: float = 1e-5,
) -> TraceAuditReport:
    """Validate a JSONL trace and recompute every distinct policy-step output."""

    path = Path(trace_path).resolve()
    policy_root = Path(policy_directory).resolve()
    if not path.is_file():
        raise TraceAuditError(f"Trace file does not exist: {path}")
    if not math.isfinite(max_action_error) or max_action_error < 0:
        raise ValueError("max_action_error must be finite and non-negative")

    records = _read_records(path)
    header = records[0]
    _require(header.get("recordType") == "header", "Trace line 1 must be a header")
    _require(header.get("schemaVersion") == 1, "Unsupported trace schemaVersion")
    _require(header.get("observationSize") == OBSERVATION_SIZE, "Header observationSize must be 61")
    _require(header.get("actionSize") == ACTION_SIZE, "Header actionSize must be 14")
    _require(_positive_number(header.get("physicsTimestepSeconds")), "Header physics timestep must be positive")
    _require(isinstance(header.get("policyDecimation"), int) and header["policyDecimation"] > 0,
             "Header policyDecimation must be positive")
    _require_array(header.get("servoNames"), ACTION_SIZE, "Header servoNames", numeric=False)
    _require(isinstance(header.get("passiveWheelNames"), list), "Header passiveWheelNames must be an array")

    producer = header.get("producer", "unknown")
    _require(isinstance(producer, str) and producer, "Header producer must be a non-empty string")
    if expected_producer is not None and producer != expected_producer:
        raise TraceAuditError(
            f"Trace producer is {producer!r}; expected {expected_producer!r}"
        )

    policy_name = header.get("policyName")
    _require(isinstance(policy_name, str) and policy_name, "Header policyName is missing")
    if Path(policy_name).name != policy_name:
        raise TraceAuditError("Header policyName must be a file name, not a path")
    policy_path = (policy_root / policy_name).resolve()
    try:
        policy_path.relative_to(policy_root)
    except ValueError as exc:
        raise TraceAuditError("Resolved policy escaped policy_directory") from exc
    _require(policy_path.is_file(), f"Policy file does not exist: {policy_path}")

    declared_hash = header.get("policySha256", "")
    actual_policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    if declared_hash:
        _require(
            isinstance(declared_hash, str) and declared_hash.lower() == actual_policy_hash,
            "Header policySha256 does not match the audited policy",
        )

    try:
        session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise TraceAuditError(f"Could not load policy {policy_name}: {exc}") from exc
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    _require(len(inputs) == 1 and tuple(inputs[0].shape) == (1, OBSERVATION_SIZE),
             "Policy must expose one [1,61] input")
    _require(len(outputs) == 1 and tuple(outputs[0].shape) == (1, ACTION_SIZE),
             "Policy must expose one [1,14] output")

    frames: list[dict[str, Any]] = []
    saw_result = False
    for line_number, record in enumerate(records[1:], start=2):
        record_type = record.get("recordType")
        if record_type == "result":
            _require(line_number == len(records), "A result record is only allowed as the final line")
            saw_result = True
            continue
        _require(not saw_result and record_type == "frame", f"Trace line {line_number} must be a frame")
        _validate_frame(record, header, line_number)
        frames.append(record)
    _require(frames, "Trace must contain at least one frame")

    previous_physics_step = -1
    previous_policy_step = -1
    previous_time = -math.inf
    policy_steps: list[dict[str, Any]] = []
    for frame in frames:
        physics_step = frame["physicsStep"]
        policy_step = frame["policyStep"]
        time_seconds = float(frame["timeSeconds"])
        _require(physics_step > previous_physics_step, "physicsStep must increase strictly")
        _require(policy_step >= previous_policy_step, "policyStep must not decrease")
        _require(time_seconds > previous_time, "timeSeconds must increase strictly")
        if policy_step != previous_policy_step:
            policy_steps.append(frame)
        previous_physics_step = physics_step
        previous_policy_step = policy_step
        previous_time = time_seconds

    maximum_error = 0.0
    for frame in policy_steps:
        observation = np.asarray(frame["observation"], dtype=np.float32).reshape(1, -1)
        recorded = np.asarray(frame["rawAction"], dtype=np.float32).reshape(1, -1)
        expected = session.run([outputs[0].name], {inputs[0].name: observation})[0]
        error = float(np.max(np.abs(expected - recorded)))
        if not math.isfinite(error):
            raise TraceAuditError("Policy action parity produced a non-finite error")
        maximum_error = max(maximum_error, error)

    if maximum_error > max_action_error:
        raise TraceAuditError(
            f"Policy action parity error {maximum_error:.9g} exceeds {max_action_error:.9g}"
        )

    return TraceAuditReport(
        trace_path=path,
        trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        producer=producer,
        policy_name=policy_name,
        frame_count=len(frames),
        policy_step_count=len(policy_steps),
        max_abs_action_error=maximum_error,
        tolerance=max_action_error,
        passed=True,
    )


def write_trace_audit_report(report: TraceAuditReport, destination: str | Path) -> Path:
    """Write one deterministic audit report for orchestration artifacts."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceAuditError(f"Trace line {line_number} is malformed: {exc}") from exc
        _require(isinstance(record, dict), f"Trace line {line_number} must be a JSON object")
        records.append(record)
    _require(len(records) >= 2, "Trace must contain a header and at least one frame")
    return records


def _validate_frame(frame: dict[str, Any], header: dict[str, Any], line_number: int) -> None:
    prefix = f"line {line_number}"
    _require(isinstance(frame.get("physicsStep"), int) and frame["physicsStep"] >= 0,
             f"{prefix} physicsStep must be a non-negative integer")
    _require(isinstance(frame.get("policyStep"), int) and frame["policyStep"] >= 0,
             f"{prefix} policyStep must be a non-negative integer")
    _require(_finite_number(frame.get("timeSeconds")), f"{prefix} timeSeconds must be finite")
    for field, width in (
        ("rootPosition", 3),
        ("rootQuaternionWxyz", 4),
        ("rootLinearVelocity", 3),
        ("rootAngularVelocity", 3),
        ("jointPositionRad", ACTION_SIZE),
        ("jointVelocityRadPerSecond", ACTION_SIZE),
        ("passiveWheelVelocityRadPerSecond", len(header["passiveWheelNames"])),
        ("observation", OBSERVATION_SIZE),
        ("rawAction", ACTION_SIZE),
        ("targetPositionRad", ACTION_SIZE),
        ("command", COMMAND_SIZE),
    ):
        _require_array(frame.get(field), width, f"{prefix} {field}")
    _require(isinstance(frame.get("contacts"), list), f"{prefix} contacts must be an array")


def _require_array(value: Any, width: int, label: str, *, numeric: bool = True) -> None:
    _require(isinstance(value, list) and len(value) == width, f"{label} must contain {width} values")
    if numeric:
        _require(all(_finite_number(item) for item in value), f"{label} must contain only finite numbers")
    else:
        _require(all(isinstance(item, str) and item for item in value), f"{label} must contain names")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive_number(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise TraceAuditError(message)
