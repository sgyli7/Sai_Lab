"""Machine-readable acceptance gates used by ``run-mvp.ps1``.

This script intentionally stays a thin adapter over the public bridge APIs.  The
PowerShell runner remains responsible for orchestration and external processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
import onnxruntime as ort

from agenticrobot_bridge.mujoco_rollout import run_all_policy_scenarios
from agenticrobot_bridge.policy_audit import audit_policy_bundle
from agenticrobot_bridge.trace_audit import audit_rollout_trace, write_trace_audit_report
from agenticrobot_bridge.upstream import load_upstream_lock


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_audit(project_root: Path, output: Path) -> int:
    lock = load_upstream_lock(project_root / "upstream.lock.json")
    report = audit_policy_bundle(project_root / ".cache" / "upstream", lock).to_dict()
    _write_json(output, report)
    return 0 if report["policyCount"] == 9 and report["allFinite"] else 1


def _mujoco_rollouts(project_root: Path, output_directory: Path) -> int:
    report = run_all_policy_scenarios(project_root, output_directory)
    return 0 if report.passed and len(report.results) == 9 else 1


def _training_cache(project_root: Path, output: Path) -> int:
    log_root = (
        project_root
        / ".cache"
        / "upstream"
        / "microduck_rl"
        / "logs"
        / "rsl_rl"
        / "velocity"
    )
    rejected: list[str] = []
    upstream_root = project_root / ".cache" / "upstream" / "microduck_rl"
    lock = load_upstream_lock(project_root / "upstream.lock.json")
    expected_commit = lock.repositories["microduck_rl"].commit
    actual_commit = subprocess.run(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(upstream_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if actual_commit != expected_commit or dirty:
        raise ValueError(
            "PPO cache source is not the clean locked microduck_rl commit: "
            f"HEAD={actual_commit}, expected={expected_commit}, dirty={bool(dirty)}"
        )
    candidates = sorted(log_root.glob("*_mvp-smoke"), reverse=True)
    for candidate in candidates:
        agent_path = candidate / "params" / "agent.yaml"
        env_path = candidate / "params" / "env.yaml"
        expected_checkpoints = [candidate / f"model_{iteration}.pt" for iteration in range(5)]
        event_files = list(candidate.glob("events.out.tfevents.*"))
        if not agent_path.is_file() or not env_path.is_file():
            rejected.append(f"{candidate.name}: missing parameter files")
            continue
        agent = agent_path.read_text(encoding="utf-8")
        environment = env_path.read_text(encoding="utf-8")
        checks = {
            "maxIterationsIsFive": bool(
                re.search(r"(?m)^max_iterations:\s*5\s*$", agent)
            ),
            "saveIntervalIsOne": bool(
                re.search(r"(?m)^save_interval:\s*1\s*$", agent)
            ),
            "runNameIsMvpSmoke": bool(
                re.search(r"(?m)^run_name:\s*mvp-smoke\s*$", agent)
            ),
            "numEnvsIs64": bool(
                re.search(r"(?m)^\s{2}num_envs:\s*64\s*$", environment)
            ),
            "fiveCheckpointsPresent": all(path.is_file() for path in expected_checkpoints),
            "tensorboardEvidencePresent": bool(event_files),
            "checkpointsNonEmpty": all(
                path.is_file() and path.stat().st_size > 100_000
                for path in expected_checkpoints
            ),
            "tensorboardEvidenceNonEmpty": bool(event_files)
            and all(path.stat().st_size > 0 for path in event_files),
        }
        if not all(checks.values()):
            failed = ", ".join(name for name, passed in checks.items() if not passed)
            rejected.append(f"{candidate.name}: {failed}")
            continue
        report = {
            "schemaVersion": 1,
            "passed": True,
            "task": "Mjlab-Velocity-Flat-MicroDuck",
            "numEnvs": 64,
            "maxIterations": 5,
            "checkpointIteration": 4,
            "checkpoint": str(expected_checkpoints[-1].resolve()),
            "checkpointBytes": expected_checkpoints[-1].stat().st_size,
            "checkpointSha256": _sha256(expected_checkpoints[-1]),
            "runDirectory": str(candidate.resolve()),
            "upstreamCommit": actual_commit,
            "agentConfigSha256": _sha256(agent_path),
            "environmentConfigSha256": _sha256(env_path),
            "eventFiles": [
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(event_files)
            ],
            "checks": checks,
        }
        _write_json(output, report)
        return 0
    details = "; ".join(rejected) if rejected else "no mvp-smoke run directories"
    raise FileNotFoundError(f"No valid exact 64-env x 5-iteration PPO cache: {details}")


def _onnx_attest(training_report: Path, onnx_path: Path, output: Path) -> int:
    training = json.loads(training_report.read_text(encoding="utf-8"))
    if training.get("passed") is not True:
        raise ValueError("PPO training report is not passing")
    checkpoint = Path(training["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PPO checkpoint is missing: {checkpoint}")
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != training.get("checkpointSha256"):
        raise ValueError("PPO training report checkpoint hash no longer matches its bytes")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Exported ONNX is missing: {onnx_path}")
    attestation = {
        "schemaVersion": 1,
        "passed": True,
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "exporter": "microduck_rl/scripts/export.py",
        "upstreamCommit": training.get("upstreamCommit"),
        "checkpoint": str(checkpoint),
        "checkpointSha256": checkpoint_hash,
        "onnx": str(onnx_path.resolve()),
        "onnxSha256": _sha256(onnx_path),
    }
    _write_json(output, attestation)
    return 0


def _validate_onnx_attestation(
    training_report: Path,
    onnx_path: Path,
    attestation_path: Path,
) -> dict[str, Any]:
    training = json.loads(training_report.read_text(encoding="utf-8"))
    if not attestation_path.is_file():
        raise FileNotFoundError(f"ONNX source attestation is missing: {attestation_path}")
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    checkpoint = Path(training["checkpoint"]).resolve()
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != training.get("checkpointSha256"):
        raise ValueError("Current checkpoint hash does not match the training report")
    if attestation.get("checkpointSha256") != checkpoint_hash:
        raise ValueError("ONNX attestation checkpoint hash does not match current checkpoint hash")
    if Path(attestation.get("checkpoint", "")).resolve() != checkpoint:
        raise ValueError("ONNX attestation checkpoint path does not match")
    if Path(attestation.get("onnx", "")).resolve() != onnx_path.resolve():
        raise ValueError("ONNX attestation output path does not match")
    if attestation.get("onnxSha256") != _sha256(onnx_path):
        raise ValueError("ONNX attestation hash does not match current ONNX bytes")
    if (
        attestation.get("schemaVersion") != 1
        or attestation.get("passed") is not True
        or attestation.get("exporter") != "microduck_rl/scripts/export.py"
        or attestation.get("upstreamCommit") != training.get("upstreamCommit")
    ):
        raise ValueError("ONNX attestation metadata is invalid")
    return attestation


def _onnx_export(
    training_report: Path,
    onnx_path: Path,
    attestation_path: Path,
    output: Path,
) -> int:
    training = json.loads(training_report.read_text(encoding="utf-8"))
    if training.get("passed") is not True:
        raise ValueError("PPO training report is not a passing exact-cache report")
    checkpoint = Path(training["checkpoint"]).resolve()
    if not checkpoint.is_file() or checkpoint.name != "model_4.pt":
        raise FileNotFoundError(f"Validated PPO checkpoint is missing: {checkpoint}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Official PPO ONNX export was not found: {onnx_path}")
    attestation = _validate_onnx_attestation(training_report, onnx_path, attestation_path)

    model = onnx.load(onnx_path, load_external_data=False)
    onnx.checker.check_model(model)
    metadata = {item.key: item.value for item in model.metadata_props}
    source_checkpoint = Path(metadata.get("run_path", "")).resolve()
    if source_checkpoint != checkpoint:
        raise ValueError(
            "ONNX run_path metadata does not identify the validated model_4.pt checkpoint"
        )
    if metadata.get("action_scale") != "1.0":
        raise ValueError("Official PPO ONNX export must declare action_scale=1.0")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "obs" or inputs[0].shape != [1, 61]:
        raise ValueError("Official PPO ONNX input contract must be obs [1,61]")
    if len(outputs) != 1 or outputs[0].name != "actions" or outputs[0].shape != [1, 14]:
        raise ValueError("Official PPO ONNX output contract must be actions [1,14]")
    generator = np.random.default_rng(20260903)
    for _ in range(100):
        observation = generator.standard_normal((1, 61), dtype=np.float32)
        actions = session.run(["actions"], {"obs": observation})[0]
        if actions.shape != (1, 14) or not np.isfinite(actions).all():
            raise ValueError("Official PPO ONNX produced invalid randomized inference output")

    report = {
        "schemaVersion": 1,
        "passed": True,
        "sourceCheckpoint": str(checkpoint),
        "sourceCheckpointSha256": training["checkpointSha256"],
        "onnx": str(onnx_path.resolve()),
        "sha256": _sha256(onnx_path),
        "attestation": str(attestation_path.resolve()),
        "attestationSha256": _sha256(attestation_path),
        "upstreamCommit": attestation["upstreamCommit"],
        "irVersion": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "input": {"name": "obs", "shape": [1, 61]},
        "output": {"name": "actions", "shape": [1, 14]},
        "actionScale": 1.0,
        "randomizedFiniteRuns": 100,
    }
    _write_json(output, report)
    return 0


def _is_result(result: str | None, expected: str) -> bool:
    return bool(result) and result.startswith(expected)


def _tuanjie_results(
    platform: str,
    results_path: Path,
    output: Path,
    *,
    required_tests: Sequence[str] = (),
    allowed_skipped_tests: Sequence[str] = (),
    not_before_utc: datetime | None = None,
) -> int:
    if not results_path.is_file():
        raise FileNotFoundError(f"Tuanjie {platform} test results were not found: {results_path}")
    root = ET.parse(results_path).getroot()
    if root.tag != "test-run":
        raise ValueError(f"Unexpected Tuanjie test result root: {root.tag}")
    result = root.attrib.get("result")
    total = int(root.attrib.get("total", "0"))
    passed = int(root.attrib.get("passed", "0"))
    failed = int(root.attrib.get("failed", "0"))
    skipped = int(root.attrib.get("skipped", "0"))
    declared_platforms = {
        property_.attrib.get("value")
        for property_ in root.findall(".//property[@name='platform']")
    }
    cases = root.findall(".//test-case")
    passed_cases = {
        case.attrib.get("fullname", "")
        for case in cases
        if _is_result(case.attrib.get("result"), "Passed")
    }
    failed_cases = {
        case.attrib.get("fullname", "")
        for case in cases
        if _is_result(case.attrib.get("result"), "Failed")
    }
    skipped_cases = {
        case.attrib.get("fullname", "")
        for case in cases
        if _is_result(case.attrib.get("result"), "Skipped")
    }
    inconclusive_cases = {
        case.attrib.get("fullname", "")
        for case in cases
        if not any(
            _is_result(case.attrib.get("result"), known)
            for known in ("Passed", "Failed", "Skipped")
        )
    }
    required = set(required_tests)
    allowed_skips = set(allowed_skipped_tests)
    modified_utc = datetime.fromtimestamp(results_path.stat().st_mtime, timezone.utc)
    source_fresh = not_before_utc is None or modified_utc >= not_before_utc
    checks = {
        "resultPassed": result == "Passed",
        "nonEmptySuite": total > 0,
        "noFailures": failed == 0 and not failed_cases,
        "leafAccountingMatches": (
            len(cases) == total
            and len(passed_cases) == passed
            and len(failed_cases) == failed
            and len(skipped_cases) == skipped
            and not inconclusive_cases
        ),
        "platformMatches": not declared_platforms or platform in declared_platforms,
        "requiredTestsPassed": required <= passed_cases,
        "onlyAllowedTestsSkipped": skipped_cases <= allowed_skips,
        "sourceFresh": source_fresh,
    }
    report = {
        "schemaVersion": 1,
        "passed": all(checks.values()),
        "platform": platform,
        "source": str(results_path.resolve()),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        "leafSummary": {
            "total": len(cases),
            "passed": len(passed_cases),
            "failed": len(failed_cases),
            "skipped": len(skipped_cases),
            "inconclusive": len(inconclusive_cases),
        },
        "requiredTests": sorted(required),
        "passedRequiredTests": sorted(required & passed_cases),
        "skippedTests": sorted(skipped_cases),
        "allowedSkippedTests": sorted(allowed_skips),
        "sourceModifiedUtc": modified_utc.isoformat(),
        "notBeforeUtc": not_before_utc.isoformat() if not_before_utc else None,
        "checks": checks,
    }
    _write_json(output, report)
    if not report["passed"]:
        failed_checks = ", ".join(name for name, value in checks.items() if not value)
        raise ValueError(f"Tuanjie {platform} test gate failed: {failed_checks}")
    return 0


_OFFICIAL_POLICY_NAMES = {
    "alpha_walking.onnx",
    "alpha_stand.onnx",
    "alpha_sitstand.onnx",
    "alpha_ground_pick.onnx",
    "ball_kick_left.onnx",
    "ball_kick_right.onnx",
    "roller.onnx",
    "roller_crouch.onnx",
    "roulade.onnx",
}
_COMPOUND_SCENARIO_NAMES = {
    "stand-roulade-stand-live-hot-swap",
    "roller-crouch-roller-live-hot-swap",
}


def _native_behavior(
    source: Path,
    output: Path,
    *,
    not_before_utc: datetime | None,
) -> int:
    if not source.is_file():
        raise FileNotFoundError(f"Native MuJoCo behavior report was not found: {source}")
    document = json.loads(source.read_text(encoding="utf-8"))
    scenarios = document.get("scenarioResults")
    compounds = document.get("compoundResults")
    if not isinstance(scenarios, list):
        scenarios = []
    if not isinstance(compounds, list):
        compounds = []
    results = scenarios + compounds
    modified_utc = datetime.fromtimestamp(source.stat().st_mtime, timezone.utc)

    def valid_result(item: object) -> bool:
        if not isinstance(item, dict):
            return False
        checks = item.get("checks")
        return (
            item.get("passed") is True
            and item.get("finiteState") is True
            and isinstance(item.get("physicsStepCount"), int)
            and item["physicsStepCount"] > 0
            and isinstance(item.get("policyTickCount"), int)
            and item["policyTickCount"] > 0
            and isinstance(checks, list)
            and bool(checks)
            and all(isinstance(check, dict) and check.get("passed") is True for check in checks)
        )

    hot_swap_jumps = [
        item.get("metrics", {}).get("hotSwapStateJump")
        for item in compounds
        if isinstance(item, dict) and isinstance(item.get("metrics"), dict)
    ]
    checks = {
        "schemaMatches": document.get("schemaVersion") == 1,
        "rootPassed": document.get("passed") is True,
        "authorityMatches": document.get("physicsAuthority")
        == "official MuJoCo 3.12 native runtime",
        "nativeVersionMatches": (
            document.get("nativeVersion") == 3012000
            and document.get("nativeVersionString") == "3.12.0"
        ),
        "runtimeMatches": (
            isinstance(document.get("policyRuntime"), str)
            and document["policyRuntime"].startswith("Barracuda 3.0.1")
            and "Sentis" not in document["policyRuntime"]
        ),
        "timingMatches": (
            document.get("physicsTimestepSeconds") == 0.005
            and document.get("policyDecimation") == 4
        ),
        "exactPolicySet": (
            len(scenarios) == len(_OFFICIAL_POLICY_NAMES)
            and {item.get("name") for item in scenarios if isinstance(item, dict)}
            == _OFFICIAL_POLICY_NAMES
        ),
        "exactCompoundSet": (
            len(compounds) == len(_COMPOUND_SCENARIO_NAMES)
            and {item.get("name") for item in compounds if isinstance(item, dict)}
            == _COMPOUND_SCENARIO_NAMES
        ),
        "allResultsFinite": bool(results) and all(valid_result(item) for item in results),
        "hotSwapStateContinuous": (
            len(hot_swap_jumps) == 2
            and all(isinstance(value, (int, float)) and abs(value) <= 1e-12 for value in hot_swap_jumps)
        ),
        "sourceFresh": not_before_utc is None or modified_utc >= not_before_utc,
    }
    gate = {
        "schemaVersion": 1,
        "passed": all(checks.values()),
        "source": str(source.resolve()),
        "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sourceModifiedUtc": modified_utc.isoformat(),
        "notBeforeUtc": not_before_utc.isoformat() if not_before_utc else None,
        "physicsAuthority": document.get("physicsAuthority"),
        "nativeVersion": document.get("nativeVersion"),
        "nativeVersionString": document.get("nativeVersionString"),
        "policyRuntime": document.get("policyRuntime"),
        "policyCount": len(scenarios),
        "compoundCount": len(compounds),
        "checks": checks,
    }
    _write_json(output, gate)
    if not gate["passed"]:
        failures = ", ".join(name for name, passed_ in checks.items() if not passed_)
        raise ValueError(f"Native behavior gate failed: {failures}")
    return 0


def _smoke_checks(smoke: dict[str, Any]) -> dict[str, bool]:
    return {
        "schemaMatches": smoke.get("schemaVersion") == 1,
        "smokePassed": smoke.get("passed") is True,
        "nativeSceneLoaded": smoke.get("scene") == "MicroDuckNativeMvp",
        "nativeVersionMatches": (
            smoke.get("nativeVersion") == 3012000
            and smoke.get("nativeVersionString") == "3.12.0"
        ),
        "backendMatches": smoke.get("backend")
        == "MuJoCo 3.12 + Barracuda 3.0.1 CPU",
        "policyTicked": isinstance(smoke.get("policyTicks"), int)
        and smoke["policyTicks"] >= 1,
        "tensorContractMatches": (
            smoke.get("observationCount") == 61
            and smoke.get("actionCount") == 14
            and smoke.get("targetCount") == 14
        ),
        "allFinite": smoke.get("allFinite") is True,
        "noFault": smoke.get("fault") == "",
    }


def _locked_native_binary(
    upstream_lock: Path,
    lock_key: str,
) -> tuple[dict[str, Any], Path, str]:
    if not upstream_lock.is_file():
        raise FileNotFoundError(f"Upstream lock was not found: {upstream_lock}")
    lock_document = json.loads(upstream_lock.read_text(encoding="utf-8"))
    native_lock = lock_document.get("nativeBinaries", {}).get(lock_key)
    if not isinstance(native_lock, dict):
        raise ValueError(f"Upstream lock has no {lock_key} native binary contract")
    expected_native_hash = str(native_lock.get("sha256", "")).lower()
    if len(expected_native_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_native_hash
    ):
        raise ValueError("Locked MuJoCo native SHA-256 is invalid")
    project_path = native_lock.get("projectPath")
    if not isinstance(project_path, str) or not project_path:
        raise ValueError("Locked MuJoCo native projectPath is invalid")
    lock_root = upstream_lock.parent.resolve()
    source_native = (lock_root / project_path).resolve()
    if not source_native.is_relative_to(lock_root):
        raise ValueError("Locked MuJoCo native projectPath escapes the project root")
    if not source_native.is_file():
        raise FileNotFoundError(f"Locked source MuJoCo native was not found: {source_native}")
    return native_lock, source_native, expected_native_hash


def _player_smoke(
    build_directory: Path,
    smoke_path: Path,
    output: Path,
    upstream_lock: Path,
) -> int:
    data_directory = build_directory / "AgenticRobotGame_Data"
    player_candidates = [
        build_directory / "TuanjiePlayer.dll",
        build_directory / "UnityPlayer.dll",
    ]
    player = next((path for path in player_candidates if path.is_file()), player_candidates[0])
    required_files = [
        build_directory / "AgenticRobotGame.exe",
        player,
        data_directory / "Plugins" / "x86_64" / "mujoco.dll",
        data_directory / "Managed" / "Mujoco.Runtime.dll",
        data_directory / "Managed" / "MicroDuck.MujocoRuntime.dll",
    ]
    for path in required_files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Required Windows player file is missing or empty: {path}")
    if not smoke_path.is_file():
        raise FileNotFoundError(f"Windows player smoke report was not found: {smoke_path}")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if not upstream_lock.is_file():
        raise FileNotFoundError(f"Upstream lock was not found: {upstream_lock}")
    lock_document = json.loads(upstream_lock.read_text(encoding="utf-8"))
    native_lock = lock_document.get("nativeBinaries", {}).get("mujocoWindowsX64")
    if not isinstance(native_lock, dict):
        raise ValueError("Upstream lock has no mujocoWindowsX64 native binary contract")
    expected_native_hash = str(native_lock.get("sha256", "")).lower()
    if len(expected_native_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_native_hash
    ):
        raise ValueError("Locked MuJoCo native SHA-256 is invalid")
    project_path = native_lock.get("projectPath")
    if not isinstance(project_path, str) or not project_path:
        raise ValueError("Locked MuJoCo native projectPath is invalid")
    lock_root = upstream_lock.parent.resolve()
    source_native = (lock_root / project_path).resolve()
    if not source_native.is_relative_to(lock_root):
        raise ValueError("Locked MuJoCo native projectPath escapes the project root")
    if not source_native.is_file():
        raise FileNotFoundError(f"Locked source MuJoCo DLL was not found: {source_native}")
    built_native = data_directory / "Plugins" / "x86_64" / "mujoco.dll"
    source_native_hash = _sha256(source_native)
    built_native_hash = _sha256(built_native)
    checks = {
        **_smoke_checks(smoke),
        "nativeSourceMatchesLock": source_native_hash == expected_native_hash,
        "builtNativeMatchesLock": built_native_hash == expected_native_hash,
        "builtNativeMatchesSource": built_native_hash == source_native_hash,
    }
    bundle_paths = sorted(
        (path for path in build_directory.rglob("*") if path.is_file()),
        key=lambda path: str(path.relative_to(build_directory)).casefold(),
    )
    files = [
        {
            "path": str(path.resolve()),
            "relativePath": path.relative_to(build_directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in bundle_paths
    ]
    report = {
        "schemaVersion": 1,
        "passed": all(checks.values()),
        "buildDirectory": str(build_directory.resolve()),
        "smoke": str(smoke_path.resolve()),
        "smokeSha256": hashlib.sha256(smoke_path.read_bytes()).hexdigest(),
        "requiredFileCount": len(required_files),
        "fileCount": len(files),
        "files": files,
        "nativeBinary": {
            "version": native_lock.get("version"),
            "distribution": native_lock.get("distribution"),
            "url": native_lock.get("url"),
            "archiveSha256": native_lock.get("archiveSha256"),
            "archiveMember": native_lock.get("archiveMember"),
            "expectedSha256": expected_native_hash,
            "sourcePath": str(source_native),
            "sourceSha256": source_native_hash,
            "builtPath": str(built_native.resolve()),
            "builtSha256": built_native_hash,
        },
        "checks": checks,
    }
    _write_json(output, report)
    if not report["passed"]:
        failures = ", ".join(name for name, passed_ in checks.items() if not passed_)
        raise ValueError(f"Windows player smoke gate failed: {failures}")
    return 0


def _player_smoke_macos(
    build_app: Path,
    smoke_path: Path,
    output: Path,
    upstream_lock: Path,
) -> int:
    required_files = [
        build_app / "Contents" / "MacOS" / "AgenticRobotGame",
        build_app / "Contents" / "Frameworks" / "TuanjiePlayer.dylib",
        build_app / "Contents" / "PlugIns" / "mujoco.dylib",
        build_app / "Contents" / "Resources" / "Data" / "Managed" / "Mujoco.Runtime.dll",
        build_app
        / "Contents"
        / "Resources"
        / "Data"
        / "Managed"
        / "MicroDuck.MujocoRuntime.dll",
    ]
    for path in required_files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Required macOS player file is missing or empty: {path}")
    if not smoke_path.is_file():
        raise FileNotFoundError(f"macOS player smoke report was not found: {smoke_path}")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    native_lock, source_native, expected_native_hash = _locked_native_binary(
        upstream_lock,
        "mujocoMacOSUniversal2",
    )
    built_native = build_app / "Contents" / "PlugIns" / "mujoco.dylib"
    source_native_hash = _sha256(source_native)
    built_native_hash = _sha256(built_native)
    checks = {
        **_smoke_checks(smoke),
        "nativeSourceMatchesLock": source_native_hash == expected_native_hash,
        "builtNativeMatchesLock": built_native_hash == expected_native_hash,
        "builtNativeMatchesSource": built_native_hash == source_native_hash,
        "noWindowsNativeDll": not any(build_app.rglob("mujoco.dll")),
    }
    bundle_paths = sorted(
        (path for path in build_app.rglob("*") if path.is_file()),
        key=lambda path: str(path.relative_to(build_app)).casefold(),
    )
    files = [
        {
            "path": str(path.resolve()),
            "relativePath": path.relative_to(build_app).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in bundle_paths
    ]
    report = {
        "schemaVersion": 1,
        "passed": all(checks.values()),
        "buildApp": str(build_app.resolve()),
        "smoke": str(smoke_path.resolve()),
        "smokeSha256": hashlib.sha256(smoke_path.read_bytes()).hexdigest(),
        "requiredFileCount": len(required_files),
        "fileCount": len(files),
        "files": files,
        "nativeBinary": {
            "version": native_lock.get("version"),
            "distribution": native_lock.get("distribution"),
            "url": native_lock.get("url"),
            "archiveSha256": native_lock.get("archiveSha256"),
            "archiveMember": native_lock.get("archiveMember"),
            "expectedSha256": expected_native_hash,
            "sourcePath": str(source_native),
            "sourceSha256": source_native_hash,
            "builtPath": str(built_native.resolve()),
            "builtSha256": built_native_hash,
        },
        "checks": checks,
    }
    _write_json(output, report)
    if not report["passed"]:
        failures = ", ".join(name for name, passed_ in checks.items() if not passed_)
        raise ValueError(f"macOS player smoke gate failed: {failures}")
    return 0


def _trace_parity(trace_path: Path, policy_directory: Path, output: Path) -> int:
    report = audit_rollout_trace(
        trace_path,
        policy_directory,
        expected_producer="tuanjie",
        max_action_error=1e-5,
    )
    write_trace_audit_report(report, output)
    return 0 if report.passed else 1


def _evidence(kind: str, source: Path, output: Path) -> int:
    if not source.is_file():
        raise FileNotFoundError(f"{kind.title()} evidence was not found: {source}")
    raw = source.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{kind} evidence is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{kind} evidence root must be an object")
    if document.get("schemaVersion") != 1 or document.get("kind") != kind:
        raise ValueError(f"{kind} evidence has the wrong schema or kind")
    checks = document.get("checks")
    if (
        document.get("passed") is not True
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise ValueError(f"{kind} evidence does not prove every required check passed")
    if kind == "codely":
        required_checks = {
            "bridgeInstalled",
            "redTestObserved",
            "greenTestPassed",
            "fullEditModeSuitePassed",
            "runtimeSourceChanged",
        }
        if not required_checks <= checks.keys() or not all(checks[name] is True for name in required_checks):
            raise ValueError("Codely evidence is missing one or more required RED/GREEN checks")
    items = document.get("evidence")
    if not isinstance(items, list) or not items or not all(isinstance(item, str) and item for item in items):
        raise ValueError(f"{kind} evidence must cite at least one non-empty evidence item")
    referenced_evidence: list[dict[str, Any]] = []
    codely_red_observed = False
    codely_green_observed = False
    codely_full_green_observed = False
    for item in items:
        referenced = Path(item)
        if not referenced.is_absolute():
            referenced = source.parent / referenced
        referenced = referenced.resolve()
        if not referenced.is_file():
            raise FileNotFoundError(
                f"{kind} referenced evidence file was not found: {referenced}"
            )
        referenced_evidence.append(
            {
                "path": str(referenced),
                "sha256": hashlib.sha256(referenced.read_bytes()).hexdigest(),
            }
        )
        if kind == "codely" and referenced.suffix.lower() == ".xml":
            result_root = ET.parse(referenced).getroot()
            result = result_root.attrib.get("result", "")
            total = int(result_root.attrib.get("total", "0"))
            path_parts = {part.casefold() for part in referenced.parts}
            if "red" in path_parts and result.startswith("Failed"):
                codely_red_observed = True
            if "green" in path_parts and result == "Passed" and total > 0:
                codely_green_observed = True
                codely_full_green_observed |= total >= 20
    if kind == "codely" and not (
        codely_red_observed and codely_green_observed and codely_full_green_observed
    ):
        raise ValueError(
            "Codely evidence must include a failed RED XML, a passing GREEN XML, "
            "and a passing full EditMode XML with at least 20 tests"
        )
    gate = {
        "schemaVersion": 1,
        "kind": kind,
        "passed": True,
        "source": str(source.resolve()),
        "sourceSha256": hashlib.sha256(raw).hexdigest(),
        "referencedEvidence": referenced_evidence,
        "checks": checks,
    }
    if kind == "codely":
        gate["verifiedTestEvidence"] = {
            "redFailed": codely_red_observed,
            "greenPassed": codely_green_observed,
            "fullEditModePassed": codely_full_green_observed,
        }
    _write_json(output, gate)
    return 0


def _parse_utc(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC timestamps must include an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    policy = subparsers.add_parser("policy-audit")
    policy.add_argument("--project-root", required=True, type=Path)
    policy.add_argument("--output", required=True, type=Path)
    rollouts = subparsers.add_parser("mujoco-rollouts")
    rollouts.add_argument("--project-root", required=True, type=Path)
    rollouts.add_argument("--output-dir", required=True, type=Path)
    training = subparsers.add_parser("training-cache")
    training.add_argument("--project-root", required=True, type=Path)
    training.add_argument("--output", required=True, type=Path)
    exported = subparsers.add_parser("onnx-export")
    exported.add_argument("--project-root", required=True, type=Path)
    exported.add_argument("--training-report", required=True, type=Path)
    exported.add_argument("--onnx", required=True, type=Path)
    exported.add_argument("--attestation", required=True, type=Path)
    exported.add_argument("--output", required=True, type=Path)
    attest = subparsers.add_parser("onnx-attest")
    attest.add_argument("--training-report", required=True, type=Path)
    attest.add_argument("--onnx", required=True, type=Path)
    attest.add_argument("--output", required=True, type=Path)
    test_results = subparsers.add_parser("tuanjie-results")
    test_results.add_argument("--platform", required=True, choices=("EditMode", "PlayMode"))
    test_results.add_argument("--input", required=True, type=Path)
    test_results.add_argument("--output", required=True, type=Path)
    test_results.add_argument("--required-test", action="append", default=[])
    test_results.add_argument("--allowed-skipped-test", action="append", default=[])
    test_results.add_argument("--not-before-utc")
    native_behavior = subparsers.add_parser("native-behavior")
    native_behavior.add_argument("--input", required=True, type=Path)
    native_behavior.add_argument("--output", required=True, type=Path)
    native_behavior.add_argument("--not-before-utc")
    player_smoke = subparsers.add_parser("player-smoke")
    player_smoke.add_argument("--build-directory", required=True, type=Path)
    player_smoke.add_argument("--input", required=True, type=Path)
    player_smoke.add_argument("--output", required=True, type=Path)
    player_smoke.add_argument("--upstream-lock", required=True, type=Path)
    player_smoke_macos = subparsers.add_parser("player-smoke-macos")
    player_smoke_macos.add_argument("--build-app", required=True, type=Path)
    player_smoke_macos.add_argument("--input", required=True, type=Path)
    player_smoke_macos.add_argument("--output", required=True, type=Path)
    player_smoke_macos.add_argument("--upstream-lock", required=True, type=Path)
    trace = subparsers.add_parser("trace-parity")
    trace.add_argument("--trace", required=True, type=Path)
    trace.add_argument("--policy-directory", required=True, type=Path)
    trace.add_argument("--output", required=True, type=Path)
    evidence = subparsers.add_parser("evidence")
    evidence.add_argument("--kind", required=True, choices=("codely", "trace-parity"))
    evidence.add_argument("--input", required=True, type=Path)
    evidence.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "policy-audit":
        return _policy_audit(arguments.project_root.resolve(), arguments.output.resolve())
    if arguments.command == "mujoco-rollouts":
        return _mujoco_rollouts(
            arguments.project_root.resolve(), arguments.output_dir.resolve()
        )
    if arguments.command == "training-cache":
        return _training_cache(arguments.project_root.resolve(), arguments.output.resolve())
    if arguments.command == "onnx-export":
        return _onnx_export(
            arguments.training_report.resolve(),
            arguments.onnx.resolve(),
            arguments.attestation.resolve(),
            arguments.output.resolve(),
        )
    if arguments.command == "onnx-attest":
        return _onnx_attest(
            arguments.training_report.resolve(),
            arguments.onnx.resolve(),
            arguments.output.resolve(),
        )
    if arguments.command == "tuanjie-results":
        return _tuanjie_results(
            arguments.platform,
            arguments.input.resolve(),
            arguments.output.resolve(),
            required_tests=arguments.required_test,
            allowed_skipped_tests=arguments.allowed_skipped_test,
            not_before_utc=_parse_utc(arguments.not_before_utc),
        )
    if arguments.command == "native-behavior":
        return _native_behavior(
            arguments.input.resolve(),
            arguments.output.resolve(),
            not_before_utc=_parse_utc(arguments.not_before_utc),
        )
    if arguments.command == "player-smoke":
        return _player_smoke(
            arguments.build_directory.resolve(),
            arguments.input.resolve(),
            arguments.output.resolve(),
            arguments.upstream_lock.resolve(),
        )
    if arguments.command == "player-smoke-macos":
        return _player_smoke_macos(
            arguments.build_app.resolve(),
            arguments.input.resolve(),
            arguments.output.resolve(),
            arguments.upstream_lock.resolve(),
        )
    if arguments.command == "trace-parity":
        return _trace_parity(
            arguments.trace.resolve(),
            arguments.policy_directory.resolve(),
            arguments.output.resolve(),
        )
    if arguments.command == "evidence":
        return _evidence(arguments.kind, arguments.input.resolve(), arguments.output.resolve())
    raise AssertionError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
