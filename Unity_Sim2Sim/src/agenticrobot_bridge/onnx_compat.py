"""Deterministic Barracuda-compatible copies of the locked ONNX policies."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from .policy_audit import discover_policy_paths
from .upstream import UpstreamLock


TARGET_OPSET = 9
ALLOWED_OPERATORS = frozenset({"Sub", "Div", "Gemm", "Elu"})


class OnnxCompatibilityError(RuntimeError):
    """Raised when a policy cannot be converted with proven numerical parity."""


@dataclass(frozen=True, slots=True)
class PolicyCompatibilityResult:
    """Parity evidence for one converted policy."""

    name: str
    original_sha256: str
    converted_sha256: str
    fixture_count: int
    max_abs_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "originalSha256": self.original_sha256,
            "convertedSha256": self.converted_sha256,
            "fixtureCount": self.fixture_count,
            "maxAbsError": self.max_abs_error,
        }


@dataclass(frozen=True, slots=True)
class PolicyCompatibilityReport:
    """Structured evidence for a complete compatibility bundle."""

    target_opset: int
    fixture_names: tuple[str, ...]
    policies: tuple[PolicyCompatibilityResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "targetOpset": self.target_opset,
            "fixtureNames": list(self.fixture_names),
            "policies": [policy.to_dict() for policy in self.policies],
        }


def convert_policy_bundle(
    upstream_root: str | Path,
    lock: UpstreamLock,
    output_directory: str | Path,
    *,
    policy_subdirectory: str | Path = "policies",
) -> PolicyCompatibilityReport:
    """Emit opset-9 policy copies plus deterministic parity fixtures and report."""

    source_paths = discover_policy_paths(upstream_root, lock)
    output_root = Path(output_directory)
    policy_relative_path = Path(policy_subdirectory)
    if policy_relative_path.is_absolute() or ".." in policy_relative_path.parts:
        raise OnnxCompatibilityError("Policy output subdirectory must stay within its bundle")
    policy_output_root = output_root / policy_relative_path
    fixtures = _parity_fixtures()
    expected_outputs: dict[str, dict[str, list[float]]] = {
        fixture_name: {} for fixture_name, _ in fixtures
    }

    converted_bytes: dict[str, bytes] = {}
    results: list[PolicyCompatibilityResult] = []
    for name, source_path in source_paths.items():
        model = onnx.load(source_path, load_external_data=False)
        _validate_policy_family(model, name)
        _rewrite_default_opset(model, name)
        try:
            onnx.checker.check_model(model, full_check=True)
        except onnx.checker.ValidationError as exc:
            raise OnnxCompatibilityError(f"Converted {name} failed ONNX validation: {exc}") from exc

        serialized = model.SerializeToString(deterministic=True)
        converted_bytes[name] = serialized
        fixture_errors, original_outputs = _parity_evidence(
            source_path.read_bytes(), serialized, fixtures, name
        )
        for (fixture_name, _), output in zip(fixtures, original_outputs, strict=True):
            expected_outputs[fixture_name][name] = output.reshape(-1).tolist()
        max_abs_error = max(fixture_errors, default=0.0)
        if not np.isfinite(max_abs_error) or max_abs_error > 1e-5:
            raise OnnxCompatibilityError(
                f"{name} conversion parity error {max_abs_error:.9g} exceeds 1e-5"
            )
        results.append(
            PolicyCompatibilityResult(
                name=name,
                original_sha256=_sha256_bytes(source_path.read_bytes()),
                converted_sha256=_sha256_bytes(serialized),
                fixture_count=len(fixtures),
                max_abs_error=max_abs_error,
            )
        )

    report = PolicyCompatibilityReport(
        target_opset=TARGET_OPSET,
        fixture_names=tuple(name for name, _ in fixtures),
        policies=tuple(results),
    )

    policy_output_root.mkdir(parents=True, exist_ok=True)
    for name, serialized in converted_bytes.items():
        destination = policy_output_root / name
        if destination.resolve() == source_paths[name].resolve():
            raise OnnxCompatibilityError(f"Refusing to overwrite original policy {name}")
        destination.write_bytes(serialized)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "parity-fixtures.json").write_text(
        _fixtures_json(
            fixtures,
            expected_outputs,
            tuple(policy.name for policy in report.policies),
        ),
        encoding="utf-8",
        newline="\n",
    )
    (output_root / "compatibility-report.json").write_text(
        _json_document(report.to_dict()), encoding="utf-8", newline="\n"
    )
    return report


def _validate_policy_family(model: onnx.ModelProto, name: str) -> None:
    for node in model.graph.node:
        if node.domain != "":
            raise OnnxCompatibilityError(
                f"{name} uses unsupported node domain {node.domain!r} on {node.op_type}"
            )
        if node.op_type not in ALLOWED_OPERATORS:
            raise OnnxCompatibilityError(
                f"{name} uses unsupported operator {node.op_type!r}"
            )
        seen_attributes: set[str] = set()
        for attribute in node.attribute:
            if attribute.name in seen_attributes or not _attribute_is_safe(node.op_type, attribute):
                raise OnnxCompatibilityError(
                    f"{name} has unsafe attribute {attribute.name!r} on {node.op_type}"
                )
            seen_attributes.add(attribute.name)


def _attribute_is_safe(operator: str, attribute: onnx.AttributeProto) -> bool:
    if operator in {"Sub", "Div"}:
        return False
    if operator == "Elu":
        return (
            attribute.name == "alpha"
            and attribute.type == onnx.AttributeProto.FLOAT
            and math.isfinite(attribute.f)
        )
    if operator == "Gemm":
        if attribute.name in {"alpha", "beta"}:
            return attribute.type == onnx.AttributeProto.FLOAT and math.isfinite(attribute.f)
        if attribute.name in {"transA", "transB"}:
            return attribute.type == onnx.AttributeProto.INT and attribute.i in {0, 1}
    return False


def _rewrite_default_opset(model: onnx.ModelProto, name: str) -> None:
    default_imports = [item for item in model.opset_import if item.domain in ("", "ai.onnx")]
    if len(model.opset_import) != 1 or len(default_imports) != 1:
        imports = [(item.domain, item.version) for item in model.opset_import]
        raise OnnxCompatibilityError(f"{name} has unsupported opset imports: {imports}")
    source_opset = default_imports[0]
    if source_opset.version != 18:
        raise OnnxCompatibilityError(
            f"{name} default opset is {source_opset.version}, expected locked version 18"
        )
    source_opset.domain = ""
    source_opset.version = TARGET_OPSET


def _parity_fixtures() -> tuple[tuple[str, np.ndarray], ...]:
    rng = np.random.default_rng(20_260_903)
    return (
        ("zeros", np.zeros((1, 61), dtype=np.float32)),
        ("ramp", np.linspace(-1.0, 1.0, 61, dtype=np.float32).reshape(1, 61)),
        ("seeded", rng.uniform(-1.0, 1.0, size=(1, 61)).astype(np.float32)),
    )


def _parity_evidence(
    original: bytes,
    converted: bytes,
    fixtures: tuple[tuple[str, np.ndarray], ...],
    name: str,
) -> tuple[list[float], list[np.ndarray]]:
    try:
        original_session = ort.InferenceSession(original, providers=["CPUExecutionProvider"])
        converted_session = ort.InferenceSession(converted, providers=["CPUExecutionProvider"])
        original_input = original_session.get_inputs()[0].name
        converted_input = converted_session.get_inputs()[0].name
        errors: list[float] = []
        original_outputs: list[np.ndarray] = []
        for _, observation in fixtures:
            original_output = original_session.run(None, {original_input: observation})[0]
            converted_output = converted_session.run(None, {converted_input: observation})[0]
            errors.append(float(np.max(np.abs(original_output - converted_output))))
            original_outputs.append(original_output.astype(np.float32, copy=False))
        return errors, original_outputs
    except Exception as exc:
        raise OnnxCompatibilityError(f"Parity inference failed for {name}: {exc}") from exc


def _fixtures_json(
    fixtures: tuple[tuple[str, np.ndarray], ...],
    expected_outputs: dict[str, dict[str, list[float]]],
    policy_names: tuple[str, ...],
) -> str:
    observations = {
        name: observation.reshape(-1).tolist() for name, observation in fixtures
    }
    return _json_document(
        {
            "schemaVersion": 1,
            "inputShape": [1, 61],
            "outputShape": [1, 14],
            "cases": [
                {
                    "fixtureName": fixture_name,
                    "policyName": policy_name,
                    "input": observations[fixture_name],
                    "expectedOutput": expected_outputs[fixture_name][policy_name],
                }
                for policy_name in policy_names
                for fixture_name, _ in fixtures
            ],
        }
    )


def _json_document(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
