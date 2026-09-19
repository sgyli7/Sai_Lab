from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "mvp-gates.py"
SPEC = importlib.util.spec_from_file_location("mvp_gates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MVP_GATES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MVP_GATES)


def _write_test_results(path: Path, *, required_result: str = "Passed") -> None:
    passed = 1 if required_result == "Passed" else 0
    failed = 1 if required_result == "Failed" else 0
    path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<test-run result="{'Passed' if failed == 0 else 'Failed'}" total="2"
          passed="{passed}" failed="{failed}" skipped="1" inconclusive="0">
  <test-suite>
    <properties><property name="platform" value="PlayMode" /></properties>
    <test-case fullname="Example.Required" result="{required_result}" />
    <test-case fullname="Example.CalibrationDiagnostic" result="Skipped" />
  </test-suite>
</test-run>
""",
        encoding="utf-8",
    )


def test_tuanjie_gate_requires_named_tests_and_allows_only_declared_diagnostics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "results.xml"
    output = tmp_path / "report.json"
    _write_test_results(source)

    result = MVP_GATES._tuanjie_results(
        "PlayMode",
        source,
        output,
        required_tests=("Example.Required",),
        allowed_skipped_tests=("Example.CalibrationDiagnostic",),
        not_before_utc=datetime.now(timezone.utc) - timedelta(seconds=5),
    )

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["checks"]["requiredTestsPassed"] is True
    assert report["checks"]["onlyAllowedTestsSkipped"] is True
    assert report["leafSummary"] == {
        "total": 2,
        "passed": 1,
        "failed": 0,
        "skipped": 1,
        "inconclusive": 0,
    }


@pytest.mark.parametrize(
    ("required", "allowed", "message"),
    [
        (("Missing.Test",), ("Example.CalibrationDiagnostic",), "requiredTestsPassed"),
        (("Example.Required",), (), "onlyAllowedTestsSkipped"),
    ],
)
def test_tuanjie_gate_fails_closed_for_missing_or_unapproved_leaf_tests(
    tmp_path: Path,
    required: tuple[str, ...],
    allowed: tuple[str, ...],
    message: str,
) -> None:
    source = tmp_path / "results.xml"
    _write_test_results(source)

    with pytest.raises(ValueError, match=message):
        MVP_GATES._tuanjie_results(
            "PlayMode",
            source,
            tmp_path / "report.json",
            required_tests=required,
            allowed_skipped_tests=allowed,
            not_before_utc=None,
        )


def test_tuanjie_gate_rejects_stale_xml_even_when_it_says_passed(tmp_path: Path) -> None:
    source = tmp_path / "results.xml"
    _write_test_results(source)
    old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()
    os.utime(source, (old_timestamp, old_timestamp))

    with pytest.raises(ValueError, match="sourceFresh"):
        MVP_GATES._tuanjie_results(
            "PlayMode",
            source,
            tmp_path / "report.json",
            required_tests=("Example.Required",),
            allowed_skipped_tests=("Example.CalibrationDiagnostic",),
            not_before_utc=datetime.now(timezone.utc) - timedelta(seconds=5),
        )


def _native_behavior_payload() -> dict[str, object]:
    policy_names = [
        "alpha_walking.onnx",
        "alpha_stand.onnx",
        "alpha_sitstand.onnx",
        "alpha_ground_pick.onnx",
        "ball_kick_left.onnx",
        "ball_kick_right.onnx",
        "roller.onnx",
        "roller_crouch.onnx",
        "roulade.onnx",
    ]

    def result(name: str) -> dict[str, object]:
        return {
            "name": name,
            "policySequence": [name],
            "physicsStepCount": 4,
            "policyTickCount": 1,
            "finiteState": True,
            "passed": True,
            "metrics": {"finiteState": 1.0},
            "checks": [{"name": "finiteState", "passed": True}],
            "failure": "",
        }

    return {
        "schemaVersion": 1,
        "physicsAuthority": "official MuJoCo 3.12 native runtime",
        "nativeVersion": 3012000,
        "nativeVersionString": "3.12.0",
        "policyRuntime": "Barracuda 3.0.1 CSharp",
        "physicsTimestepSeconds": 0.005,
        "policyDecimation": 4,
        "passed": True,
        "scenarioResults": [result(name) for name in policy_names],
        "compoundResults": [
            {
                **result("stand-roulade-stand-live-hot-swap"),
                "policySequence": [
                    "alpha_stand.onnx",
                    "roulade.onnx",
                    "alpha_stand.onnx",
                ],
                "metrics": {"finiteState": 1.0, "hotSwapStateJump": 0.0},
            },
            {
                **result("roller-crouch-roller-live-hot-swap"),
                "policySequence": [
                    "roller.onnx",
                    "roller_crouch.onnx",
                    "roller.onnx",
                ],
                "metrics": {"finiteState": 1.0, "hotSwapStateJump": 0.0},
            },
        ],
    }


def _write_native_lock(root: Path, native_bytes: bytes) -> Path:
    source = root / "TuanjieProject" / "Assets" / "Plugins" / "x86_64" / "mujoco.dll"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(native_bytes)
    lock = root / "upstream.lock.json"
    lock.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "nativeBinaries": {
                    "mujocoWindowsX64": {
                        "version": "3.12.0",
                        "distribution": "official MuJoCo PyPI wheel",
                        "url": "https://files.pythonhosted.org/mujoco.whl",
                        "archiveSha256": "a" * 64,
                        "archiveMember": "mujoco/mujoco.dll",
                        "sha256": MVP_GATES._sha256(source),
                        "projectPath": (
                            "TuanjieProject/Assets/Plugins/x86_64/mujoco.dll"
                        ),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return lock


def test_native_behavior_gate_proves_exact_authority_policies_and_hot_swaps(
    tmp_path: Path,
) -> None:
    source = tmp_path / "native.json"
    output = tmp_path / "gate.json"
    source.write_text(json.dumps(_native_behavior_payload()), encoding="utf-8")

    result = MVP_GATES._native_behavior(
        source,
        output,
        not_before_utc=datetime.now(timezone.utc) - timedelta(seconds=5),
    )

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["policyCount"] == 9
    assert report["compoundCount"] == 2
    assert all(report["checks"].values())


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda payload: payload.update({"physicsAuthority": "PhysX"}), "authorityMatches"),
        (lambda payload: payload.update({"policyRuntime": "Sentis"}), "runtimeMatches"),
        (lambda payload: payload["scenarioResults"].pop(), "exactPolicySet"),
        (
            lambda payload: payload["compoundResults"][0].update({"finiteState": False}),
            "allResultsFinite",
        ),
    ],
)
def test_native_behavior_gate_rejects_shallow_pass_markers(
    tmp_path: Path,
    mutation,
    failed_check: str,
) -> None:
    payload = _native_behavior_payload()
    mutation(payload)
    source = tmp_path / "native.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=failed_check):
        MVP_GATES._native_behavior(source, tmp_path / "gate.json", not_before_utc=None)


def test_player_smoke_gate_requires_native_policy_tick_and_complete_bundle(
    tmp_path: Path,
) -> None:
    build = tmp_path / "Windows64"
    data = build / "AgenticRobotGame_Data"
    plugin = data / "Plugins" / "x86_64" / "mujoco.dll"
    managed = data / "Managed"
    for path in (
        build / "AgenticRobotGame.exe",
        build / "TuanjiePlayer.dll",
        plugin,
        managed / "Mujoco.Runtime.dll",
        managed / "MicroDuck.MujocoRuntime.dll",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not-empty")
    smoke = tmp_path / "smoke.json"
    smoke.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "passed": True,
                "scene": "MicroDuckNativeMvp",
                "nativeVersion": 3012000,
                "nativeVersionString": "3.12.0",
                "backend": "MuJoCo 3.12 + Barracuda 3.0.1 CPU",
                "policyTicks": 3,
                "observationCount": 61,
                "actionCount": 14,
                "targetCount": 14,
                "allFinite": True,
                "fault": "",
            }
        ),
        encoding="utf-8",
    )

    lock = _write_native_lock(tmp_path, b"not-empty")
    result = MVP_GATES._player_smoke(
        build, smoke, tmp_path / "bundle.json", lock
    )

    assert result == 0
    report = json.loads((tmp_path / "bundle.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["fileCount"] == 5
    assert all(len(item["sha256"]) == 64 for item in report["files"])


def test_player_smoke_gate_rejects_an_exe_without_native_bundle(tmp_path: Path) -> None:
    build = tmp_path / "Windows64"
    build.mkdir()
    (build / "AgenticRobotGame.exe").write_bytes(b"launcher")
    (build / "TuanjiePlayer.dll").write_bytes(b"player")
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"schemaVersion": 1, "passed": True}), encoding="utf-8")
    lock = _write_native_lock(tmp_path, b"not-empty")

    with pytest.raises(FileNotFoundError, match="mujoco.dll"):
        MVP_GATES._player_smoke(build, smoke, tmp_path / "bundle.json", lock)


def test_player_smoke_gate_rejects_native_dll_not_bound_to_the_locked_source(
    tmp_path: Path,
) -> None:
    build = tmp_path / "Windows64"
    data = build / "AgenticRobotGame_Data"
    managed = data / "Managed"
    for path, contents in (
        (build / "AgenticRobotGame.exe", b"launcher"),
        (build / "TuanjiePlayer.dll", b"player"),
        (data / "Plugins" / "x86_64" / "mujoco.dll", b"tampered-native"),
        (managed / "Mujoco.Runtime.dll", b"bridge"),
        (managed / "MicroDuck.MujocoRuntime.dll", b"runtime"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    smoke = tmp_path / "smoke.json"
    smoke.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "passed": True,
                "scene": "MicroDuckNativeMvp",
                "nativeVersion": 3012000,
                "nativeVersionString": "3.12.0",
                "backend": "MuJoCo 3.12 + Barracuda 3.0.1 CPU",
                "policyTicks": 3,
                "observationCount": 61,
                "actionCount": 14,
                "targetCount": 14,
                "allFinite": True,
                "fault": "",
            }
        ),
        encoding="utf-8",
    )
    lock = _write_native_lock(tmp_path, b"trusted-native")

    with pytest.raises(ValueError, match="builtNativeMatchesLock"):
        MVP_GATES._player_smoke(build, smoke, tmp_path / "bundle.json", lock)


def _passing_smoke_payload() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "passed": True,
        "scene": "MicroDuckNativeMvp",
        "nativeVersion": 3012000,
        "nativeVersionString": "3.12.0",
        "backend": "MuJoCo 3.12 + Barracuda 3.0.1 CPU",
        "policyTicks": 3,
        "observationCount": 61,
        "actionCount": 14,
        "targetCount": 14,
        "allFinite": True,
        "fault": "",
    }


def _write_macos_native_lock(root: Path, native_bytes: bytes) -> Path:
    source = root / "TuanjieProject" / "Assets" / "Plugins" / "macOS" / "mujoco.dylib"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(native_bytes)
    lock = root / "upstream.lock.json"
    lock.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "nativeBinaries": {
                    "mujocoMacOSUniversal2": {
                        "version": "3.12.0",
                        "distribution": "official MuJoCo GitHub release DMG",
                        "url": "https://github.com/google-deepmind/mujoco/macos.dmg",
                        "archiveSha256": "b" * 64,
                        "archiveMember": "libmujoco.3.12.0.dylib",
                        "sha256": MVP_GATES._sha256(source),
                        "projectPath": "TuanjieProject/Assets/Plugins/macOS/mujoco.dylib",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return lock


def _write_macos_bundle(app: Path, *, dylib_bytes: bytes) -> None:
    files = {
        app / "Contents" / "MacOS" / "AgenticRobotGame": b"player",
        app / "Contents" / "Frameworks" / "TuanjiePlayer.dylib": b"engine",
        app / "Contents" / "PlugIns" / "mujoco.dylib": dylib_bytes,
        app
        / "Contents"
        / "Resources"
        / "Data"
        / "Managed"
        / "Mujoco.Runtime.dll": b"bridge",
        app
        / "Contents"
        / "Resources"
        / "Data"
        / "Managed"
        / "MicroDuck.MujocoRuntime.dll": b"runtime",
    }
    for path, contents in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)


def test_macos_player_smoke_gate_requires_native_policy_tick_and_complete_bundle(
    tmp_path: Path,
) -> None:
    app = tmp_path / "AgenticRobotGame.app"
    _write_macos_bundle(app, dylib_bytes=b"trusted-native")
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps(_passing_smoke_payload()), encoding="utf-8")
    lock = _write_macos_native_lock(tmp_path, b"trusted-native")

    result = MVP_GATES._player_smoke_macos(
        app, smoke, tmp_path / "bundle.json", lock
    )

    assert result == 0
    report = json.loads((tmp_path / "bundle.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["fileCount"] == 5
    assert all(len(item["sha256"]) == 64 for item in report["files"])
    assert report["checks"]["noWindowsNativeDll"] is True


def test_macos_player_smoke_gate_rejects_an_app_without_native_dylib(
    tmp_path: Path,
) -> None:
    app = tmp_path / "AgenticRobotGame.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "MacOS" / "AgenticRobotGame").write_bytes(b"player")
    (app / "Contents" / "Frameworks").mkdir(parents=True)
    (app / "Contents" / "Frameworks" / "TuanjiePlayer.dylib").write_bytes(b"engine")
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps(_passing_smoke_payload()), encoding="utf-8")
    lock = _write_macos_native_lock(tmp_path, b"trusted-native")

    with pytest.raises(FileNotFoundError, match="mujoco.dylib"):
        MVP_GATES._player_smoke_macos(app, smoke, tmp_path / "bundle.json", lock)


def test_macos_player_smoke_gate_rejects_native_dylib_not_bound_to_the_locked_source(
    tmp_path: Path,
) -> None:
    app = tmp_path / "AgenticRobotGame.app"
    _write_macos_bundle(app, dylib_bytes=b"tampered-native")
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps(_passing_smoke_payload()), encoding="utf-8")
    lock = _write_macos_native_lock(tmp_path, b"trusted-native")

    with pytest.raises(ValueError, match="builtNativeMatchesLock"):
        MVP_GATES._player_smoke_macos(app, smoke, tmp_path / "bundle.json", lock)


def test_onnx_attestation_binds_export_bytes_to_checkpoint_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_4.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps(
            {
                "passed": True,
                "checkpoint": str(checkpoint),
                "checkpointSha256": MVP_GATES._sha256(checkpoint),
                "upstreamCommit": "5946fd9cdbc58956424420153e51975af3b30d77",
            }
        ),
        encoding="utf-8",
    )
    onnx = tmp_path / "policy.onnx"
    onnx.write_bytes(b"onnx-v1")
    attestation = tmp_path / "attestation.json"

    assert MVP_GATES._onnx_attest(training, onnx, attestation) == 0
    proof = json.loads(attestation.read_text(encoding="utf-8"))
    assert proof["checkpointSha256"] == MVP_GATES._sha256(checkpoint)
    assert proof["onnxSha256"] == MVP_GATES._sha256(onnx)

    checkpoint.write_bytes(b"checkpoint-v2")
    with pytest.raises(ValueError, match="checkpoint hash"):
        MVP_GATES._validate_onnx_attestation(training, onnx, attestation)
