"""Discovery and validation of the pinned MicroDuck ONNX policy bundle."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from .upstream import UpstreamLock


class PolicyAuditError(RuntimeError):
    """Raised when the vendored policy bundle violates its locked contract."""


EXPECTED_INPUT_SHAPE = (1, 61)
EXPECTED_OUTPUT_SHAPE = (1, 14)


@dataclass(frozen=True, slots=True)
class PolicyAuditResult:
    """Validated metadata and smoke-inference result for one ONNX policy."""

    name: str
    sha256: str
    input_name: str
    input_shape: tuple[int, ...]
    input_type: str
    output_name: str
    output_shape: tuple[int, ...]
    output_type: str
    output_is_finite: bool
    sample_output_min: float
    sample_output_max: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "name": self.name,
            "sha256": self.sha256,
            "input": {
                "name": self.input_name,
                "shape": list(self.input_shape),
                "type": self.input_type,
            },
            "output": {
                "name": self.output_name,
                "shape": list(self.output_shape),
                "type": self.output_type,
                "isFinite": self.output_is_finite,
                "sampleMin": self.sample_output_min,
                "sampleMax": self.sample_output_max,
            },
        }


@dataclass(frozen=True, slots=True)
class PolicyAuditReport:
    """Structured result for the complete locked policy bundle."""

    policies: tuple[PolicyAuditResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report suitable for build artifacts."""

        return {
            "schemaVersion": 1,
            "policyCount": len(self.policies),
            "allFinite": all(policy.output_is_finite for policy in self.policies),
            "policies": [policy.to_dict() for policy in self.policies],
        }


def discover_policy_paths(
    upstream_root: str | Path,
    lock: UpstreamLock,
) -> dict[str, Path]:
    """Resolve every locked policy from the fixed MicroDuck policies directory."""

    policy_directory = Path(upstream_root) / "microduck" / "policies"
    policies = {name: policy_directory / name for name in lock.policies}
    missing = [name for name, path in policies.items() if not path.is_file()]
    if missing:
        raise PolicyAuditError(f"Missing locked policies: {', '.join(missing)}")
    expected_names = set(policies)
    unexpected = sorted(
        path.name for path in policy_directory.glob("*.onnx") if path.name not in expected_names
    )
    if unexpected:
        raise PolicyAuditError(f"Unexpected unlocked policies: {', '.join(unexpected)}")
    return policies


def audit_policy_bundle(
    upstream_root: str | Path,
    lock: UpstreamLock,
) -> PolicyAuditReport:
    """Validate and execute every policy named by the upstream lock."""

    paths = discover_policy_paths(upstream_root, lock)
    results = tuple(_audit_policy(name, path) for name, path in paths.items())
    return PolicyAuditReport(policies=results)


def _audit_policy(name: str, path: Path) -> PolicyAuditResult:
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise PolicyAuditError(f"Unable to load {name}: {exc}") from exc

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise PolicyAuditError(
            f"{name} must expose one input and one output; got {len(inputs)} and {len(outputs)}"
        )

    input_meta = inputs[0]
    output_meta = outputs[0]
    input_shape = tuple(input_meta.shape)
    output_shape = tuple(output_meta.shape)
    if input_shape != EXPECTED_INPUT_SHAPE:
        raise PolicyAuditError(
            f"{name} input shape is {input_shape}, expected {EXPECTED_INPUT_SHAPE}"
        )
    if output_shape != EXPECTED_OUTPUT_SHAPE:
        raise PolicyAuditError(
            f"{name} output shape is {output_shape}, expected {EXPECTED_OUTPUT_SHAPE}"
        )
    if input_meta.type != "tensor(float)" or output_meta.type != "tensor(float)":
        raise PolicyAuditError(
            f"{name} must use float tensors; got {input_meta.type} and {output_meta.type}"
        )

    observation = np.zeros(EXPECTED_INPUT_SHAPE, dtype=np.float32)
    try:
        sample_output = session.run([output_meta.name], {input_meta.name: observation})[0]
    except Exception as exc:
        raise PolicyAuditError(f"Inference failed for {name}: {exc}") from exc

    output_is_finite = bool(np.isfinite(sample_output).all())
    if not output_is_finite:
        raise PolicyAuditError(f"{name} produced NaN or infinite output for a zero observation")

    return PolicyAuditResult(
        name=name,
        sha256=_sha256(path),
        input_name=input_meta.name,
        input_shape=input_shape,
        input_type=input_meta.type,
        output_name=output_meta.name,
        output_shape=output_shape,
        output_type=output_meta.type,
        output_is_finite=output_is_finite,
        sample_output_min=float(np.min(sample_output)),
        sample_output_max=float(np.max(sample_output)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as policy_file:
        for chunk in iter(lambda: policy_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
