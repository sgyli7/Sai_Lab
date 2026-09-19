#!/usr/bin/env python3
"""macOS MVP orchestrator. Windows PowerShell remains the Windows entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
STAGES: tuple[str, ...] = (
    "bootstrap",
    "python-tests",
    "policy-audit",
    "mujoco-rollouts",
    "ppo-smoke",
    "onnx-export",
    "robot-manifest",
    "tuanjie-editmode",
    "tuanjie-playmode",
    "trace-parity",
    "codely-proof",
    "windows-build",
    "macos-build",
    "macos-environment-acceptance",
    "training-prep",
)
CUDA_SKIP_REASON = (
    "requires CUDA (microduck_rl/mjlab Warp); not available on Apple Silicon"
)
CUDA_IMPORT_MODULES = frozenset({"warp", "mjlab"})
LOCKED_EDITOR = "2022.3.62t14"
DEFAULT_TUANJIE = Path(
    "/Applications/Tuanjie/Hub/Editor/2022.3.62t14/Tuanjie.app/Contents/MacOS/Tuanjie"
)
TUANJIE_TIMEOUT = 1800
CODELY_MISSING_REASON = "artifacts/mvp/codely/proof.json is not present"
WINDOWS_BUILD_REASON = "Windows player build is not applicable on macOS"
EDITMODE_REQUIRED = (
    "AgenticRobot.MicroDuck.Tests.AuthoritativeMujocoPolicyBehaviorTests."
    "AllOfficialPoliciesAndLiveSameModelHotSwapsMeetBehaviorContracts",
    "AgenticRobot.MicroDuck.Tests.BarracudaParityTests."
    "AllOfficialPoliciesMatchOnnxRuntimeReferenceFixtures",
    "AgenticRobot.MicroDuck.Tests.OfficialMujocoPluginTests."
    "NativePluginLoadsAndStepsTheOfficialMicroDuckScene",
    "AgenticRobot.MicroDuck.Tests.OfficialMujocoPrefabImporterTests."
    "ImportsAllExpandedOfficialScenesAsCompleteDeterministicPrefabs",
    "AgenticRobot.MicroDuck.Tests.DemoSceneBuilderTests."
    "CreatesPlayableNativeSceneWithOfficialMuJoCoAndBarracudaControls",
    "AgenticRobot.MicroDuck.Tests.PolicyRuntimeContractTests."
    "OutOfRangeFiniteActionFailsClosedAndClearsEveryAction",
)
PLAYMODE_REQUIRED = (
    "AgenticRobot.MicroDuck.Tests.NativeMujocoScenePlayModeTests."
    "KeyboardPolicySwitchesRunAfterMujocoSceneLateUpdate",
    "AgenticRobot.MicroDuck.Tests.NativeMujocoScenePlayModeTests."
    "NativeStandPolicyKeepsTheVisibleDuckUprightForFourSeconds",
    "AgenticRobot.MicroDuck.Tests.NativeMujocoScenePlayModeTests."
    "NativeWalkingPolicyMovesForwardWhileRemainingUprightForSixSeconds",
    "AgenticRobot.MicroDuck.Tests.NativeMujocoScenePlayModeTests."
    "NativeSceneRunsBarracudaPolicyAndRecreatesOnlyForRobotVariantChanges",
    "AgenticRobot.MicroDuck.Tests.NativeMujocoScenePlayModeTests."
    "NativeRobotRemainsInsideAUsableCameraFrame",
    "AgenticRobot.MicroDuck.Tests.OfficialPolicyPlayModeTests."
    "GeneratedSceneTicksAllNinePoliciesThroughBarracudaAndArticulation",
    "AgenticRobot.MicroDuck.Tests.OfficialPolicyPlayModeTests."
    "HotSwappingRollerPoliciesPreservesLiveRigAndControlState",
    "AgenticRobot.MicroDuck.Tests.KickBallPlayModeTests."
    "BothKickPoliciesStrikeTheBallAndRemainStanding",
)
PLAYMODE_ALLOWED_SKIPPED = (
    "AgenticRobot.MicroDuck.Tests.StrictPolicyBehaviorPlayModeTests."
    "OfficialPoliciesMeetMuJoCoDerivedSustainedAndCompoundBehaviorContracts",
    "AgenticRobot.MicroDuck.Tests.OfficialPolicyPlayModeTests."
    "AnkleStepResponseMatchesTheMuJoCoArmatureDynamics",
    "AgenticRobot.MicroDuck.Tests.OfficialPolicyPlayModeTests."
    "DiagnosticHomeDrivesHoldTheRobotWithoutPolicyForTwoSeconds",
)
HF_VERIFY_SCRIPT = r"""
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort

hf_path, walking_path = sys.argv[1], sys.argv[2]
model = onnx.load(hf_path)
session = ort.InferenceSession(hf_path, providers=["CPUExecutionProvider"])
inp = session.get_inputs()[0]
out = session.get_outputs()[0]
inp_shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
out_shape = [dim if isinstance(dim, int) else 1 for dim in out.shape]
if inp_shape[-1] != 61 or out_shape[-1] != 14:
    raise SystemExit(f"unexpected IO shape {inp_shape} -> {out_shape}")
rng = np.random.default_rng(0)


def check(values):
    output = session.run(None, {inp.name: values.astype(np.float32)})[0]
    if output.shape[-1] != 14 or not np.isfinite(output).all():
        raise SystemExit("hf sidecar produced a non-finite or wrong-width action")


check(np.zeros((1, 61), np.float32))
check(np.linspace(-1, 1, 61, dtype=np.float32)[None, :])
check(rng.normal(size=(1, 61)).astype(np.float32))
hf_ops = [node.op_type for node in model.graph.node]
hf_params = sum(1 for _ in model.graph.initializer)
walking_ops = []
walking_params = None
if walking_path:
    walking = onnx.load(walking_path)
    walking_ops = [node.op_type for node in walking.graph.node]
    walking_params = sum(1 for _ in walking.graph.initializer)
print(json.dumps({
    "ok": True,
    "inputWidth": 61,
    "outputWidth": 14,
    "hfOps": hf_ops,
    "hfInitializers": hf_params,
    "walkingOpsEqual": hf_ops == walking_ops,
    "walkingInitializerCount": walking_params,
    "walkingInitializerEqual": walking_params == hf_params,
}))
"""
IMPORT_ONE_SCRIPT = r"""
import json
import sys

name = sys.argv[1]
try:
    __import__(name)
    print(json.dumps({"ok": True}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
"""
TORCH_DEVICE_SCRIPT = r"""
import json
import torch

print(json.dumps({
    "cuda": bool(torch.cuda.is_available()),
    "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
}))
"""
WANDB_LIST_SCRIPT = r"""
import json
import os
import sys

run_id = sys.argv[1]
candidates = [
    f"pollen-robotics/mjlab_microduck/{run_id}",
    f"pollen-robotics/microduck/{run_id}",
    f"pollen-robotics/microduck_rl/{run_id}",
]
try:
    import wandb
except Exception as exc:
    print(json.dumps({"status": "failed", "error": f"wandb import failed: {exc}"}))
    raise SystemExit(0)

api = wandb.Api()
errors = []
for path in candidates:
    try:
        run = api.run(path)
        files = [item.name for item in run.files()]
        print(json.dumps({
            "status": "listed",
            "runPath": path,
            "files": files,
            "hasModel9999": "model_9999.pt" in files,
        }))
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as exc:
        errors.append(f"{path}: {exc}")
print(json.dumps({
    "status": "failed",
    "runId": run_id,
    "error": " ; ".join(errors),
}))
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def venv_python(root: Path) -> Path:
    candidate = root / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else Path(sys.executable)


def resolve_uv() -> str:
    found = shutil.which("uv")
    if found:
        return found
    homebrew = Path("/opt/homebrew/bin/uv")
    if homebrew.is_file():
        return str(homebrew)
    return "uv"


def with_local_bin(env: dict[str, str] | None = None) -> dict[str, str]:
    merged = os.environ.copy() if env is None else env.copy()
    local_bin = str(Path.home() / ".local" / "bin")
    merged["PATH"] = local_bin + os.pathsep + merged.get("PATH", "")
    return merged


def resolve_tuanjie(explicit: str | None, *, must_exist: bool) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists() or path.is_absolute():
            path = path.resolve()
    else:
        env = os.environ.get("TUANJIE_EDITOR", "").strip()
        if env:
            path = Path(env).expanduser()
            if path.exists() or path.is_absolute():
                path = path.resolve()
        else:
            path = DEFAULT_TUANJIE
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"Tuanjie editor not found: {path}")
    return path


def editor_version_from_binary(tuanjie: Path) -> str:
    # .../Editor/<ver>/Tuanjie.app/Contents/MacOS/Tuanjie
    try:
        return tuanjie.parents[3].name
    except IndexError:
        return "unknown"


def restore_project_version(root: Path) -> None:
    subprocess.run(
        ["git", "checkout", "--", "TuanjieProject/ProjectSettings/ProjectVersion.txt"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def try_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        completed = run_command(command, cwd=cwd, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "returncode": None,
            "timedOut": True,
            "timeoutSeconds": timeout,
            "stdout": stdout[-500:],
            "stderr": stderr[-500:],
            "error": f"timed out after {timeout} seconds",
        }
    except OSError as exc:
        return {
            "returncode": None,
            "timedOut": False,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
        }
    return {
        "returncode": completed.returncode,
        "timedOut": False,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


def run_tuanjie(
    tuanjie: Path,
    args: Sequence[str],
    *,
    cwd: Path,
    log_file: Path,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [str(tuanjie), *args]
    print(f"tuanjie: {' '.join(command)}", file=sys.stderr, flush=True)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            timeout=TUANJIE_TIMEOUT,
        )
    finally:
        restore_project_version(cwd)
    if completed.returncode != 0:
        raise RuntimeError(f"Tuanjie exited {completed.returncode}; see {log_file}")


class StageError(RuntimeError):
    pass


def load_lock(root: Path) -> dict[str, Any]:
    return json.loads((root / "upstream.lock.json").read_text(encoding="utf-8"))


def required_test_args(flag: str, names: Sequence[str]) -> list[str]:
    arguments: list[str] = []
    for name in names:
        arguments.extend([flag, name])
    return arguments


def planned_commands(root: Path, artifacts: Path, tuanjie: Path) -> dict[str, list[list[str]]]:
    python = posix(venv_python(root))
    helper = posix(root / "scripts" / "mvp-gates.py")
    tuanjie_project = posix(root / "TuanjieProject")
    test_root = posix(artifacts / "tuanjie")
    tuanjie_bin = posix(tuanjie)
    return {
        "bootstrap": [[posix(resolve_uv()), "sync", "--frozen", "--python", "3.12"]],
        "python-tests": [
            [
                python,
                "-m",
                "pytest",
                "--cov=agenticrobot_bridge",
                f"--cov-report=json:{posix(artifacts / 'python' / 'coverage.json')}",
                "--cov-fail-under=80",
            ],
            [python, "-m", "ruff", "check", "src", "tests", "scripts"],
        ],
        "policy-audit": [
            [
                python,
                helper,
                "policy-audit",
                "--project-root",
                posix(root),
                "--output",
                posix(artifacts / "policy-audit.json"),
            ]
        ],
        "mujoco-rollouts": [
            [
                python,
                helper,
                "mujoco-rollouts",
                "--project-root",
                posix(root),
                "--output-dir",
                posix(artifacts / "mujoco"),
            ]
        ],
        "ppo-smoke": [],
        "onnx-export": [],
        "robot-manifest": [
            [
                python,
                "-m",
                "agenticrobot_bridge.tuanjie_assets",
                "--project-root",
                posix(root),
                "--tuanjie-project",
                tuanjie_project,
            ],
            [
                tuanjie_bin,
                "-batchmode",
                "-nographics",
                "-quit",
                "-projectPath",
                tuanjie_project,
                "-executeMethod",
                "AgenticRobot.MicroDuck.Editor.DemoSceneBuilder.CreateAllSceneAssets",
                "-logFile",
                f"{test_root}/prefab-import.log",
            ],
        ],
        "tuanjie-editmode": [
            [
                tuanjie_bin,
                "-batchmode",
                "-nographics",
                "-projectPath",
                tuanjie_project,
                "-runTests",
                "-testPlatform",
                "EditMode",
                "-testResults",
                f"{test_root}/editmode-results.xml",
                "-logFile",
                f"{test_root}/editmode.log",
            ],
            [
                python,
                helper,
                "tuanjie-results",
                "--platform",
                "EditMode",
                "--input",
                f"{test_root}/editmode-results.xml",
                "--output",
                f"{test_root}/editmode-report.json",
                "--not-before-utc",
                "<stage-start-utc>",
                *required_test_args("--required-test", EDITMODE_REQUIRED),
            ],
            [
                python,
                helper,
                "native-behavior",
                "--input",
                posix(artifacts / "tuanjie-native-mujoco-policy-behavior.json"),
                "--output",
                f"{test_root}/native-behavior-report.json",
                "--not-before-utc",
                "<stage-start-utc>",
            ],
        ],
        "tuanjie-playmode": [
            [
                tuanjie_bin,
                "-batchmode",
                "-projectPath",
                tuanjie_project,
                "-runTests",
                "-testPlatform",
                "PlayMode",
                "-testResults",
                f"{test_root}/playmode-results.xml",
                "-logFile",
                f"{test_root}/playmode.log",
            ],
            [
                python,
                helper,
                "tuanjie-results",
                "--platform",
                "PlayMode",
                "--input",
                f"{test_root}/playmode-results.xml",
                "--output",
                f"{test_root}/playmode-report.json",
                "--not-before-utc",
                "<stage-start-utc>",
                *required_test_args("--required-test", PLAYMODE_REQUIRED),
                *required_test_args("--allowed-skipped-test", PLAYMODE_ALLOWED_SKIPPED),
            ],
        ],
        "trace-parity": [
            [
                tuanjie_bin,
                "-batchmode",
                "-nographics",
                "-quit",
                "-projectPath",
                tuanjie_project,
                "-executeMethod",
                "AgenticRobot.MicroDuck.Editor.TraceBatchExporter.ExportOnePolicyBatch",
                "-logFile",
                f"{test_root}/trace-export.log",
            ],
            [
                python,
                helper,
                "trace-parity",
                "--trace",
                posix(artifacts / "traces" / "tuanjie-alpha_stand.jsonl"),
                "--policy-directory",
                posix(
                    root
                    / "TuanjieProject"
                    / "Assets"
                    / "MicroDuck"
                    / "Generated"
                    / "Policies"
                    / "Barracuda"
                ),
                "--output",
                posix(artifacts / "traces" / "trace-parity.json"),
            ],
        ],
        "codely-proof": [],
        "windows-build": [],
        "macos-build": [
            [
                tuanjie_bin,
                "-batchmode",
                "-nographics",
                "-quit",
                "-projectPath",
                tuanjie_project,
                "-executeMethod",
                "AgenticRobot.MicroDuck.Editor.DemoSceneBuilder.BuildMacOS",
                "-logFile",
                f"{test_root}/macos-build.log",
            ],
            [
                posix(root / "Builds" / "macOS" / "AgenticRobotGame.app" / "Contents" / "MacOS" / "AgenticRobotGame"),
                "-batchmode",
                "-nographics",
                "-logFile",
                f"{test_root}/macos-player.log",
                "-microduckSmokeReport",
                f"{test_root}/player-smoke.json",
            ],
            [
                python,
                helper,
                "player-smoke-macos",
                "--build-app",
                posix(root / "Builds" / "macOS" / "AgenticRobotGame.app"),
                "--input",
                f"{test_root}/player-smoke.json",
                "--output",
                f"{test_root}/macos-build-report.json",
                "--upstream-lock",
                posix(root / "upstream.lock.json"),
            ],
        ],
        "macos-environment-acceptance": [
            [
                python,
                posix(root / "scripts" / "run-visual-acceptance-macos.py"),
                "--player",
                posix(root / "Builds" / "macOS" / "AgenticRobotGame.app"),
                "--output-directory",
                posix(artifacts / "environment-acceptance"),
            ]
        ],
        "training-prep": [],
    }


def ensure_upstream_repo(root: Path, name: str, repo: Mapping[str, Any]) -> dict[str, Any]:
    path = root / ".cache" / "upstream" / name
    git_dir = path / ".git"
    if not git_dir.exists():
        if path.exists() and any(path.iterdir()):
            raise StageError(f"Refusing to clone into non-empty non-Git path: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        clone = run_command(["git", "clone", str(repo["url"]), str(path)], cwd=root, timeout=300)
        if clone.returncode != 0:
            raise StageError(clone.stderr or clone.stdout or f"git clone {name} failed")
    origin = run_command(["git", "-C", str(path), "remote", "get-url", "origin"], cwd=root)
    if origin.returncode != 0 or origin.stdout.strip() != str(repo["url"]):
        raise StageError(f"upstream {name} origin does not match lock")
    head = run_command(["git", "-C", str(path), "rev-parse", "HEAD"], cwd=root)
    if head.returncode != 0:
        raise StageError(f"upstream {name} HEAD could not be read")
    if head.stdout.strip() != str(repo["commit"]):
        fetch = run_command(["git", "-C", str(path), "fetch", "--tags", "origin"], cwd=root, timeout=180)
        if fetch.returncode != 0:
            raise StageError(fetch.stderr or fetch.stdout or f"git fetch {name} failed")
        checkout = run_command(
            ["git", "-C", str(path), "checkout", "--detach", str(repo["commit"])],
            cwd=root,
        )
        if checkout.returncode != 0:
            raise StageError(checkout.stderr or checkout.stdout or f"git checkout {name} failed")
        head = run_command(["git", "-C", str(path), "rev-parse", "HEAD"], cwd=root)
    porcelain = run_command(["git", "-C", str(path), "status", "--porcelain"], cwd=root)
    if head.stdout.strip() != str(repo["commit"]) or porcelain.stdout.strip():
        raise StageError(f"upstream {name} is not at the locked clean commit")
    return {"commit": head.stdout.strip(), "clean": True, "url": str(repo["url"])}


def stage_bootstrap(ctx: dict[str, Any]) -> dict[str, Any]:
    root: Path = ctx["root"]
    artifacts: Path = ctx["artifacts"]
    tuanjie: Path = ctx["tuanjie"]
    uv = resolve_uv()
    completed = run_command([uv, "sync", "--frozen", "--python", "3.12"], cwd=root, timeout=600)
    if completed.returncode != 0:
        raise StageError(completed.stderr or completed.stdout)
    python = venv_python(root)
    versions = run_command(
        [
            str(python),
            "-c",
            "import mujoco,onnxruntime;print(mujoco.__version__,onnxruntime.__version__)",
        ],
        cwd=root,
    )
    if versions.returncode != 0:
        raise StageError(versions.stderr or versions.stdout)
    mujoco_ver, ort_ver = versions.stdout.strip().split()
    lock = load_lock(root)
    native = lock["nativeBinaries"]["mujocoMacOSUniversal2"]
    dylib = (root / native["projectPath"]).resolve()
    if not dylib.is_file():
        raise StageError(f"mujoco.dylib is missing: {dylib}")
    dylib_hash = sha256_file(dylib)
    if dylib_hash != native["sha256"]:
        raise StageError(f"mujoco.dylib hash mismatch: {dylib_hash}")
    dylib_arch = run_command(["lipo", "-archs", str(dylib)], cwd=root).stdout.strip()
    editor_arch = run_command(["lipo", "-archs", str(tuanjie)], cwd=root).stdout.strip()
    editor_version = editor_version_from_binary(tuanjie)
    version_matches = editor_version == LOCKED_EDITOR
    upstream = {
        name: ensure_upstream_repo(root, name, repo)
        for name, repo in lock["repositories"].items()
    }
    payload = {
        "uvSync": "passed",
        "mujoco": mujoco_ver,
        "onnxruntime": ort_ver,
        "dylibSha256": dylib_hash,
        "dylibArchitectures": dylib_arch,
        "editorArchitectures": editor_arch,
        "editorVersion": editor_version,
        "lockedEditorVersion": LOCKED_EDITOR,
        "editorMatchesLock": version_matches,
        "upstream": upstream,
    }
    write_json(artifacts / "bootstrap.json", payload)
    return {"bootstrap": payload}


def stage_python_tests(ctx: dict[str, Any]) -> dict[str, Any]:
    root: Path = ctx["root"]
    artifacts: Path = ctx["artifacts"]
    python = str(venv_python(root))
    coverage = artifacts / "python" / "coverage.json"
    coverage.parent.mkdir(parents=True, exist_ok=True)
    pytest = run_command(
        [
            python,
            "-m",
            "pytest",
            "--cov=agenticrobot_bridge",
            f"--cov-report=json:{coverage}",
            "--cov-fail-under=80",
        ],
        cwd=root,
        timeout=600,
    )
    if pytest.returncode != 0:
        raise StageError((pytest.stdout or pytest.stderr or "")[-4000:])
    ruff = run_command([python, "-m", "ruff", "check", "src", "tests", "scripts"], cwd=root)
    if ruff.returncode != 0:
        raise StageError(ruff.stdout or ruff.stderr)
    return {}


def stage_policy_audit(ctx: dict[str, Any]) -> dict[str, Any]:
    root: Path = ctx["root"]
    artifacts: Path = ctx["artifacts"]
    python = str(venv_python(root))
    helper = str(root / "scripts" / "mvp-gates.py")
    completed = run_command(
        [
            python,
            helper,
            "policy-audit",
            "--project-root",
            str(root),
            "--output",
            str(artifacts / "policy-audit.json"),
        ],
        cwd=root,
        timeout=180,
    )
    if completed.returncode != 0:
        raise StageError(completed.stderr or completed.stdout)
    return {}


def stage_mujoco_rollouts(ctx: dict[str, Any]) -> dict[str, Any]:
    root: Path = ctx["root"]
    artifacts: Path = ctx["artifacts"]
    python = str(venv_python(root))
    helper = str(root / "scripts" / "mvp-gates.py")
    completed = run_command(
        [
            python,
            helper,
            "mujoco-rollouts",
            "--project-root",
            str(root),
            "--output-dir",
            str(artifacts / "mujoco"),
        ],
        cwd=root,
        timeout=600,
    )
    if completed.returncode != 0:
        raise StageError(completed.stderr or completed.stdout)
    return {}


def stage_robot_manifest(ctx: dict[str, Any]) -> dict[str, Any]:
    root: Path = ctx["root"]
    artifacts: Path = ctx["artifacts"]
    tuanjie: Path = ctx["tuanjie"]
    python = str(venv_python(root))
    assets = run_command(
        [
            python,
            "-m",
            "agenticrobot_bridge.tuanjie_assets",
            "--project-root",
            str(root),
            "--tuanjie-project",
            str(root / "TuanjieProject"),
        ],
        cwd=root,
        timeout=180,
    )
    if assets.returncode != 0:
        raise StageError(assets.stderr or assets.stdout)
    log = artifacts / "tuanjie" / "prefab-import.log"
    run_tuanjie(
        tuanjie,
        [
            "-batchmode",
            "-nographics",
            "-quit",
            "-projectPath",
            str(root / "TuanjieProject"),
            "-executeMethod",
            "AgenticRobot.MicroDuck.Editor.DemoSceneBuilder.CreateAllSceneAssets",
            "-logFile",
            str(log),
        ],
        cwd=root,
        log_file=log,
    )
    return {}


def _tuanjie_results(
    ctx: dict[str, Any],
    *,
    platform: str,
    results: Path,
    report: Path,
    started: str,
    required: Sequence[str],
    allowed_skipped: Sequence[str] = (),
) -> None:
    python = str(venv_python(ctx["root"]))
    helper = str(ctx["root"] / "scripts" / "mvp-gates.py")
    command = [
        python,
        helper,
        "tuanjie-results",
        "--platform",
        platform,
        "--input",
        str(results),
        "--output",
        str(report),
        "--not-before-utc",
        started,
    ]
    command.extend(required_test_args("--required-test", required))
    command.extend(required_test_args("--allowed-skipped-test", allowed_skipped))
    completed = run_command(command, cwd=ctx["root"], timeout=60)
    if completed.returncode != 0:
        raise StageError(completed.stderr or completed.stdout)


def stage_tuanjie_editmode(ctx: dict[str, Any]) -> dict[str, Any]:
    root: Path = ctx["root"]
    artifacts: Path = ctx["artifacts"]
    tuanjie: Path = ctx["tuanjie"]
    started = utc_now()
    results = artifacts / "tuanjie" / "editmode-results.xml"
    if results.exists():
        results.unlink()
    log = artifacts / "tuanjie" / "editmode.log"
    run_tuanjie(
        tuanjie,
        [
            "-batchmode",
            "-nographics",
            "-projectPath",
            str(root / "TuanjieProject"),
            "-runTests",
            "-testPlatform",
            "EditMode",
            "-testResults",
            str(results),
            "-logFile",
            str(log),
        ],
        cwd=root,
        log_file=log,
    )
    _tuanjie_results(
        ctx,
        platform="EditMode",
        results=results,
        report=artifacts / "tuanjie" / "editmode-report.json",
        started=started,
        required=EDITMODE_REQUIRED,
    )
    python = str(venv_python(root))
    helper = str(root / "scripts" / "mvp-gates.py")
    native = run_command(
        [
            python,
            helper,
            "native-behavior",
            "--input",
            str(artifacts / "tuanjie-native-mujoco-policy-behavior.json"),
            "--output",
            str(artifacts / "tuanjie" / "native-behavior-report.json"),
            "--not-before-utc",
            started,
        ],
        cwd=root,
        timeout=60,
    )
    if native.returncode != 0:
        raise StageError(native.stderr or native.stdout)
    return {"notBeforeUtc": started}


def stage_tuanjie_playmode(ctx: dict[str, Any]) -> dict[str, Any]:
    root: Path = ctx["root"]
    artifacts: Path = ctx["artifacts"]
    tuanjie: Path = ctx["tuanjie"]
    started = utc_now()
    results = artifacts / "tuanjie" / "playmode-results.xml"
    if results.exists():
        results.unlink()
    log = artifacts / "tuanjie" / "playmode.log"
    run_tuanjie(
        tuanjie,
        [
            "-batchmode",
            "-projectPath",
            str(root / "TuanjieProject"),
            "-runTests",
            "-testPlatform",
            "PlayMode",
            "-testResults",
            str(results),
            "-logFile",
            str(log),
        ],
        cwd=root,
        log_file=log,
    )
    _tuanjie_results(
        ctx,
        platform="PlayMode",
        results=results,
        report=artifacts / "tuanjie" / "playmode-report.json",
        started=started,
        required=PLAYMODE_REQUIRED,
        allowed_skipped=PLAYMODE_ALLOWED_SKIPPED,
    )
    return {"notBeforeUtc": started}


def stage_trace_parity(ctx: dict[str, Any]) -> dict[str, Any]:
    root: Path = ctx["root"]
    artifacts: Path = ctx["artifacts"]
    tuanjie: Path = ctx["tuanjie"]
    log = artifacts / "tuanjie" / "trace-export.log"
    run_tuanjie(
        tuanjie,
        [
            "-batchmode",
            "-nographics",
            "-quit",
            "-projectPath",
            str(root / "TuanjieProject"),
            "-executeMethod",
            "AgenticRobot.MicroDuck.Editor.TraceBatchExporter.ExportOnePolicyBatch",
            "-logFile",
            str(log),
        ],
        cwd=root,
        log_file=log,
    )
    python = str(venv_python(root))
    helper = str(root / "scripts" / "mvp-gates.py")
    completed = run_command(
        [
            python,
            helper,
            "trace-parity",
            "--trace",
            str(artifacts / "traces" / "tuanjie-alpha_stand.jsonl"),
            "--policy-directory",
            str(
                root
                / "TuanjieProject"
                / "Assets"
                / "MicroDuck"
                / "Generated"
                / "Policies"
                / "Barracuda"
            ),
            "--output",
            str(artifacts / "traces" / "trace-parity.json"),
        ],
        cwd=root,
        timeout=60,
    )
    if completed.returncode != 0:
        raise StageError(completed.stderr or completed.stdout)
    return {}


def stage_codely(ctx: dict[str, Any]) -> dict[str, Any]:
    proof = ctx["artifacts"] / "codely" / "proof.json"
    if not proof.is_file():
        return {
            "status": "not-applicable",
            "reason": CODELY_MISSING_REASON,
        }
    python = str(venv_python(ctx["root"]))
    helper = str(ctx["root"] / "scripts" / "mvp-gates.py")
    output = ctx["artifacts"] / "codely" / "gate.json"
    completed = run_command(
        [
            python,
            helper,
            "evidence",
            "--kind",
            "codely",
            "--input",
            str(proof),
            "--output",
            str(output),
        ],
        cwd=ctx["root"],
        timeout=60,
    )
    if completed.returncode != 0:
        raise StageError(completed.stderr or completed.stdout)
    return {}


def _run_player_smoke(binary: Path, smoke: Path, player_log: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    smoke.parent.mkdir(parents=True, exist_ok=True)
    return run_command(
        [
            str(binary),
            "-batchmode",
            "-nographics",
            "-logFile",
            str(player_log),
            "-microduckSmokeReport",
            str(smoke),
        ],
        cwd=cwd,
        timeout=120,
    )


def stage_macos_build(ctx: dict[str, Any]) -> dict[str, Any]:
    root: Path = ctx["root"]
    artifacts: Path = ctx["artifacts"]
    tuanjie: Path = ctx["tuanjie"]
    log = artifacts / "tuanjie" / "macos-build.log"
    run_tuanjie(
        tuanjie,
        [
            "-batchmode",
            "-nographics",
            "-quit",
            "-projectPath",
            str(root / "TuanjieProject"),
            "-executeMethod",
            "AgenticRobot.MicroDuck.Editor.DemoSceneBuilder.BuildMacOS",
            "-logFile",
            str(log),
        ],
        cwd=root,
        log_file=log,
    )
    app = root / "Builds" / "macOS" / "AgenticRobotGame.app"
    binary = app / "Contents" / "MacOS" / "AgenticRobotGame"
    if not binary.is_file():
        raise StageError(f"macOS player binary was not produced: {binary}")
    smoke = artifacts / "tuanjie" / "player-smoke.json"
    player_log = artifacts / "tuanjie" / "macos-player.log"
    if smoke.exists():
        smoke.unlink()
    player = _run_player_smoke(binary, smoke, player_log, root)
    if player.returncode != 0 and not smoke.is_file():
        killed = player.returncode in (-9, 137)
        codesign = run_command(
            ["codesign", "--force", "--deep", "--sign", "-", str(app)],
            cwd=root,
            timeout=60,
        )
        player = _run_player_smoke(binary, smoke, player_log, root)
        if player.returncode != 0 and not smoke.is_file():
            detail = player.stderr or player.stdout or f"exit {player.returncode}"
            raise StageError(
                f"macOS player smoke exited {player.returncode} after codesign "
                f"(killed={killed}, codesign={codesign.returncode}): {detail}"
            )
    python = str(venv_python(root))
    helper = str(root / "scripts" / "mvp-gates.py")
    gate = run_command(
        [
            python,
            helper,
            "player-smoke-macos",
            "--build-app",
            str(app),
            "--input",
            str(smoke),
            "--output",
            str(artifacts / "tuanjie" / "macos-build-report.json"),
            "--upstream-lock",
            str(root / "upstream.lock.json"),
        ],
        cwd=root,
        timeout=60,
    )
    if gate.returncode != 0:
        raise StageError(gate.stderr or gate.stdout)
    return {}


def stage_environment_acceptance(ctx: dict[str, Any]) -> dict[str, Any]:
    root: Path = ctx["root"]
    artifacts: Path = ctx["artifacts"]
    python = str(venv_python(root))
    completed = run_command(
        [
            python,
            str(root / "scripts" / "run-visual-acceptance-macos.py"),
            "--player",
            str(root / "Builds" / "macOS" / "AgenticRobotGame.app"),
            "--output-directory",
            str(artifacts / "environment-acceptance"),
        ],
        cwd=root,
        timeout=240,
        env=with_local_bin(),
    )
    if completed.returncode != 0:
        raise StageError(completed.stdout or completed.stderr)
    return {}


def _verify_hf_policy(python: Path, hf_onnx: Path, walking: Path) -> dict[str, Any]:
    result = try_command(
        [
            str(python),
            "-c",
            HF_VERIFY_SCRIPT,
            str(hf_onnx),
            str(walking) if walking.is_file() else "",
        ],
        cwd=hf_onnx.parent,
        timeout=60,
    )
    if result["timedOut"] or result["returncode"] != 0:
        return {
            "status": "failed",
            "error": result.get("error") or (result["stderr"] or result["stdout"])[-800:],
        }
    try:
        payload = json.loads(result["stdout"])
    except json.JSONDecodeError as exc:
        return {"status": "failed", "error": str(exc)}
    payload["status"] = "passed"
    return payload


def _list_wandb_checkpoint(python: Path, cwd: Path) -> dict[str, Any]:
    if not os.environ.get("WANDB_API_KEY"):
        return {"status": "blocked", "reason": "blocked: needs wandb login"}
    result = try_command(
        [str(python), "-c", WANDB_LIST_SCRIPT, "yr25mna4"],
        cwd=cwd,
        timeout=60,
        env=os.environ.copy(),
    )
    if result["timedOut"] or result["returncode"] != 0:
        return {
            "status": "failed",
            "reason": result.get("error")
            or (result["stderr"] or result["stdout"] or "wandb listing failed")[-800:],
        }
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"status": "failed", "reason": result["stdout"][-800:]}


def _checkpoint_status(root: Path, artifacts: Path, python: Path) -> dict[str, Any]:
    search_roots = (
        root / ".cache" / "artifacts",
        artifacts / "training",
        root / ".cache" / "hf",
    )
    found: list[str] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        found.extend(posix(path) for path in search_root.rglob("*.pt") if path.is_file())
    if not found:
        return {"status": "blocked", "reason": "blocked: no checkpoint"}
    validator = root / "scripts" / "validate-training-runtime.py"
    training_report = artifacts / "training" / "ppo-smoke.json"
    if not training_report.is_file():
        return {
            "status": "blocked",
            "reason": "blocked: no checkpoint",
            "ptFiles": found,
            "validator": "skipped: no ppo-smoke.json training report",
        }
    output = artifacts / "training" / "ppo-runtime.json"
    result = try_command(
        [
            str(python),
            str(validator),
            "--training-report",
            str(training_report),
            "--output",
            str(output),
        ],
        cwd=root,
        timeout=120,
    )
    if result["timedOut"] or result["returncode"] != 0:
        return {
            "status": "failed",
            "reason": result.get("error")
            or (result["stderr"] or result["stdout"] or "validate-training-runtime failed")[-800:],
            "ptFiles": found,
        }
    return {"status": "passed", "ptFiles": found, "report": posix(output)}


def _probe_upstream_imports(probe_python: Path, cwd: Path) -> dict[str, Any]:
    mods: dict[str, Any] = {}
    for name in ("mujoco", "torch", "warp", "mjlab"):
        result = try_command(
            [str(probe_python), "-c", IMPORT_ONE_SCRIPT, name],
            cwd=cwd,
            timeout=90,
        )
        if result["timedOut"]:
            mods[name] = result["error"]
            continue
        if result["returncode"] != 0:
            mods[name] = (result["stderr"] or result.get("error") or "import failed")[-400:]
            continue
        try:
            payload = json.loads(result["stdout"])
        except json.JSONDecodeError as exc:
            mods[name] = str(exc)
            continue
        mods[name] = True if payload.get("ok") else payload.get("error")
    devices = {"cuda": False, "mps": False}
    if mods.get("torch") is True:
        device_result = try_command(
            [str(probe_python), "-c", TORCH_DEVICE_SCRIPT],
            cwd=cwd,
            timeout=30,
        )
        if device_result["returncode"] == 0 and not device_result["timedOut"]:
            try:
                devices = json.loads(device_result["stdout"])
            except json.JSONDecodeError:
                devices = {"cuda": False, "mps": False, "error": "invalid torch device probe"}
        else:
            devices = {
                "cuda": False,
                "mps": False,
                "error": device_result.get("error") or device_result["stderr"][-200:],
            }
    return {"mods": mods, "cuda": devices.get("cuda"), "mps": devices.get("mps"), "devices": devices}


def stage_training_prep(ctx: dict[str, Any]) -> dict[str, Any]:
    root: Path = ctx["root"]
    artifacts: Path = ctx["artifacts"]
    python = venv_python(root)
    notes: dict[str, Any] = {
        "blocked": [],
        "completed": [],
        "skipped": [],
        "failed": [],
    }
    try:
        audit_path = artifacts / "policy-audit.json"
        policies: Any = []
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            policies = audit.get("policies") or audit
            notes["completed"].append("policy-audit.json reused")
        else:
            notes["failed"].append("policy-audit.json missing; policy hashes not reused")
        rl = root / ".cache" / "upstream" / "microduck_rl"
        uv = resolve_uv()
        sync = try_command([uv, "sync", "--frozen"], cwd=rl, timeout=600)
        notes["microduckRlUvSync"] = {
            "returncode": sync["returncode"],
            "timedOut": sync["timedOut"],
            "stderr": (sync["stderr"] or "")[-500:],
            "stdout": (sync["stdout"] or "")[-200:],
            "error": sync.get("error"),
        }
        if sync["timedOut"] or sync["returncode"] not in (0,):
            notes["failed"].append("microduck_rl uv sync --frozen failed")
        rl_python = rl / ".venv" / "bin" / "python"
        probe_python = rl_python if rl_python.is_file() else python
        notes["imports"] = _probe_upstream_imports(probe_python, root)
        notes["importPython"] = posix(probe_python)
        _record_import_probe_outcomes(notes)
        hf_dir = root / ".cache" / "hf" / "microduck-rough-walk-e"
        env = with_local_bin()
        hf = shutil.which("hf", path=env["PATH"])
        if not hf:
            install = try_command(
                [uv, "tool", "install", "huggingface_hub"],
                cwd=root,
                timeout=180,
                env=env,
            )
            notes["hfInstall"] = {
                "returncode": install["returncode"],
                "stderr": (install["stderr"] or "")[-400:],
                "error": install.get("error"),
            }
            env = with_local_bin()
            hf = shutil.which("hf", path=env["PATH"]) or str(Path.home() / ".local" / "bin" / "hf")
            if install["timedOut"] or install["returncode"] not in (0,):
                notes["failed"].append("uv tool install huggingface_hub failed")
        download = try_command(
            [
                hf or "hf",
                "download",
                "RemiFabre/microduck-rough-walk-e",
                "policy.onnx",
                "manifest.json",
                "--local-dir",
                str(hf_dir),
            ],
            cwd=root,
            timeout=180,
            env=env,
        )
        sidecar: dict[str, Any] = {
            "downloadReturncode": download["returncode"],
            "timedOut": download["timedOut"],
        }
        onnx_path = hf_dir / "policy.onnx"
        walking = (
            root
            / "TuanjieProject"
            / "Assets"
            / "MicroDuck"
            / "Generated"
            / "Policies"
            / "Original"
            / "alpha_walking.onnx"
        )
        if download["returncode"] == 0 and onnx_path.is_file():
            sidecar["verify"] = _verify_hf_policy(python, onnx_path, walking)
            if sidecar["verify"].get("status") == "passed":
                notes["completed"].append("hf sidecar downloaded and verified")
            else:
                notes["failed"].append("hf sidecar ONNX verify failed")
        else:
            notes["failed"].append("hf download failed or policy.onnx missing")
            sidecar["stderr"] = (download["stderr"] or download.get("error") or "")[-400:]
        wandb = _list_wandb_checkpoint(python, root)
        notes["wandb"] = wandb
        if wandb.get("status") == "blocked":
            notes["blocked"].append(wandb.get("reason") or "blocked: needs wandb login")
        elif wandb.get("status") != "listed":
            notes["failed"].append("wandb listing failed")
        checkpoint = _checkpoint_status(root, artifacts, python)
        notes["checkpoint"] = checkpoint
        if checkpoint.get("status") == "blocked":
            notes["blocked"].append(checkpoint.get("reason") or "blocked: no checkpoint")
        elif checkpoint.get("status") == "failed":
            notes["failed"].append("validate-training-runtime failed")
        notes["policies"] = policies
        notes["hfSidecar"] = sidecar
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        notes["failed"].append(str(exc))
    write_json(artifacts / "training" / "macos-training-prep.json", notes)
    return training_prep_stage_outcome(notes)


def _record_import_probe_outcomes(notes: dict[str, Any]) -> None:
    notes.setdefault("skipped", [])
    notes.setdefault("completed", [])
    notes.setdefault("failed", [])
    mods = notes.get("imports", {}).get("mods", {})
    if not isinstance(mods, dict):
        notes["failed"].append("upstream import probe did not return a module map")
        return
    for name, value in mods.items():
        if value is True:
            notes["completed"].append(f"{name} import ok")
            continue
        detail = value if isinstance(value, str) and value else repr(value)
        if name in CUDA_IMPORT_MODULES:
            notes["skipped"].append(f"{name} import: {detail}; {CUDA_SKIP_REASON}")
            continue
        notes["failed"].append(f"{name} import probe failed: {detail}")


def training_prep_stage_outcome(notes: Mapping[str, Any]) -> dict[str, Any]:
    failed = [item for item in notes.get("failed", []) if item]
    extra: dict[str, Any] = {"trainingPrep": notes}
    if failed:
        extra["status"] = "failed"
        extra["reason"] = "; ".join(str(item) for item in failed)
        return extra
    extra["status"] = "passed"
    return extra


HANDLERS = {
    "bootstrap": stage_bootstrap,
    "python-tests": stage_python_tests,
    "policy-audit": stage_policy_audit,
    "mujoco-rollouts": stage_mujoco_rollouts,
    "robot-manifest": stage_robot_manifest,
    "tuanjie-editmode": stage_tuanjie_editmode,
    "tuanjie-playmode": stage_tuanjie_playmode,
    "trace-parity": stage_trace_parity,
    "codely-proof": stage_codely,
    "macos-build": stage_macos_build,
    "macos-environment-acceptance": stage_environment_acceptance,
    "training-prep": stage_training_prep,
}


def fixed_stage_outcome(name: str, artifacts: Path) -> dict[str, str] | None:
    if name in ("ppo-smoke", "onnx-export"):
        return {"status": "skipped", "reason": CUDA_SKIP_REASON}
    if name == "windows-build":
        return {"status": "not-applicable", "reason": WINDOWS_BUILD_REASON}
    if name == "codely-proof" and not (artifacts / "codely" / "proof.json").is_file():
        return {"status": "not-applicable", "reason": CODELY_MISSING_REASON}
    return None


def run_stages(
    *,
    root: Path,
    artifacts: Path,
    tuanjie: Path,
    selected: Sequence[str],
    dry_run: bool,
) -> dict[str, Any]:
    started = utc_now()
    results: list[dict[str, Any]] = []
    ctx = {"root": root, "artifacts": artifacts, "tuanjie": tuanjie}
    plan = planned_commands(root, artifacts, tuanjie)
    for name in selected:
        stage_started = time.perf_counter()
        item: dict[str, Any] = {"name": name, "commands": plan.get(name, [])}
        print(f"starting {name}", file=sys.stderr, flush=True)
        try:
            fixed = fixed_stage_outcome(name, artifacts)
            if dry_run:
                if fixed:
                    item.update(fixed)
                else:
                    item["status"] = "passed"
            elif fixed and name in ("ppo-smoke", "onnx-export", "windows-build"):
                item.update(fixed)
            else:
                extra = HANDLERS[name](ctx)
                status = extra.pop("status", "passed") if extra else "passed"
                item["status"] = status
                if extra:
                    item.update(extra)
                if status != "passed" and "reason" not in item:
                    raise StageError(f"{name} ended {status} without reason")
        except (
            StageError,
            OSError,
            RuntimeError,
            ValueError,
            json.JSONDecodeError,
            subprocess.TimeoutExpired,
        ) as error:
            item["status"] = "failed"
            item["reason"] = str(error)
        item["durationSeconds"] = round(time.perf_counter() - stage_started, 3)
        results.append(item)
        print(
            f"{name}: {item['status']} {item.get('reason', '')} ({item['durationSeconds']}s)",
            file=sys.stderr,
            flush=True,
        )
        if item["status"] == "failed" and not dry_run:
            break
    failed = any(item["status"] == "failed" for item in results)
    report = {
        "passed": not failed,
        "startedAt": started,
        "finishedAt": utc_now(),
        "tuanjiePath": posix(tuanjie),
        "editorVersion": editor_version_from_binary(tuanjie),
        "lockedEditorVersion": LOCKED_EDITOR,
        "editorMatchesLock": editor_version_from_binary(tuanjie) == LOCKED_EDITOR,
        "stages": results,
    }
    return report


def selected_stages(requested: Sequence[str] | None) -> list[str]:
    if not requested:
        return list(STAGES)
    wanted = set(requested)
    return [name for name in STAGES if name in wanted]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-stages", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--stage", action="append", choices=list(STAGES))
    parser.add_argument("--tuanjie-path", type=Path)
    parser.add_argument("--artifacts-root", type=Path, default=ROOT / "artifacts" / "mvp")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_stages:
        payload = list(STAGES)
        print(json.dumps(payload) if args.json else "\n".join(payload))
        return 0

    artifacts = args.artifacts_root.expanduser().resolve()
    tuanjie = resolve_tuanjie(
        str(args.tuanjie_path) if args.tuanjie_path else None,
        must_exist=not args.dry_run,
    )
    selected = selected_stages(args.stage)
    if not args.dry_run:
        artifacts.mkdir(parents=True, exist_ok=True)
        report_path = artifacts / "mvp-report.json"
        if report_path.exists():
            report_path.unlink()
    report = run_stages(
        root=ROOT,
        artifacts=artifacts,
        tuanjie=tuanjie,
        selected=selected,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        write_json(artifacts / "mvp-report.json", report)
    if args.json:
        text = json.dumps(report, indent=2, ensure_ascii=False)
    else:
        text = json.dumps(
            {
                "passed": report["passed"],
                "stages": [
                    {
                        "name": item["name"],
                        "status": item["status"],
                        "reason": item.get("reason"),
                    }
                    for item in report["stages"]
                ],
            },
            ensure_ascii=False,
        )
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
