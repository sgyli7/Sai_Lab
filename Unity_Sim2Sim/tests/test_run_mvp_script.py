import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="run-mvp.ps1 is the Windows entry point and needs powershell.exe",
)


ROOT = Path(__file__).parents[1]
EXPECTED_STAGES = [
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
    "environment-acceptance",
]


def _write_nunit_result(path: Path, *, result: str, total: int, passed: int, failed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<test-run result="{result}" total="{total}" passed="{passed}" '
        f'failed="{failed}" skipped="0" inconclusive="0" />\n',
        encoding="utf-8",
    )


def _complete_codely_checks() -> dict[str, bool]:
    return {
        "bridgeInstalled": True,
        "redTestObserved": True,
        "greenTestPassed": True,
        "fullEditModeSuitePassed": True,
        "runtimeSourceChanged": True,
    }


def test_mvp_runner_exposes_the_complete_ordered_gate_list() -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-ListStages",
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == EXPECTED_STAGES


def test_list_stages_is_read_only_and_preserves_an_existing_run_report(
    tmp_path: Path,
) -> None:
    report = tmp_path / "mvp-report.json"
    sentinel = '{"passed":true,"sentinel":"keep"}\n'
    report.write_text(sentinel, encoding="utf-8")

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-ListStages",
            "-Json",
            "-Stage",
            "policy-audit",
            "-ArtifactsRoot",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == EXPECTED_STAGES
    assert report.read_text(encoding="utf-8") == sentinel


def test_dry_run_json_exposes_every_stage_as_machine_readable_commands() -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-DryRun",
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schemaVersion"] == 1
    assert payload["selectedStage"] == "all"
    assert payload["dryRun"] is True
    assert payload["forceTraining"] is False
    assert [stage["name"] for stage in payload["stages"]] == EXPECTED_STAGES
    assert all(stage["commands"] for stage in payload["stages"])
    assert all(command["executable"] for stage in payload["stages"] for command in stage["commands"])
    assert all(isinstance(command["arguments"], list) for stage in payload["stages"] for command in stage["commands"])


def test_codely_gate_fails_closed_when_independent_evidence_is_missing(tmp_path: Path) -> None:
    missing_evidence = tmp_path / "not-created-by-the-runner.json"
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-Stage",
            "codely-proof",
            "-CodelyEvidencePath",
            str(missing_evidence),
            "-ArtifactsRoot",
            str(tmp_path / "artifacts"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode != 0
    assert "Codely evidence was not found" in (completed.stdout + completed.stderr)
    assert not missing_evidence.exists()


def test_single_policy_audit_stage_executes_and_writes_structured_evidence(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-Stage",
            "policy-audit",
            "-ArtifactsRoot",
            str(tmp_path),
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["passed"] is True
    assert [stage["name"] for stage in result["stages"]] == ["policy-audit"]
    assert result["stages"][0]["status"] == "passed"
    report = json.loads((tmp_path / "policy-audit.json").read_text(encoding="utf-8"))
    assert report["policyCount"] == 9
    assert report["allFinite"] is True


def test_codely_gate_accepts_but_does_not_replace_external_proof(tmp_path: Path) -> None:
    proof = tmp_path / "proof.json"
    _write_nunit_result(
        tmp_path / "red" / "target.xml",
        result="Failed(Child)",
        total=1,
        passed=0,
        failed=1,
    )
    _write_nunit_result(
        tmp_path / "green" / "target.xml",
        result="Passed",
        total=1,
        passed=1,
        failed=0,
    )
    _write_nunit_result(
        tmp_path / "green" / "full.xml",
        result="Passed",
        total=20,
        passed=20,
        failed=0,
    )
    original = {
        "schemaVersion": 1,
        "kind": "codely",
        "passed": True,
        "checks": _complete_codely_checks(),
        "evidence": ["red/target.xml", "green/target.xml", "green/full.xml"],
    }
    proof.write_text(json.dumps(original), encoding="utf-8")
    artifact_root = tmp_path / "artifacts"

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-Stage",
            "codely-proof",
            "-CodelyEvidencePath",
            str(proof),
            "-ArtifactsRoot",
            str(artifact_root),
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["passed"] is True
    assert json.loads(proof.read_text(encoding="utf-8")) == original
    gate = json.loads((artifact_root / "codely" / "gate.json").read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert gate["source"] == str(proof.resolve())


def test_codely_gate_rejects_a_proof_that_cites_missing_session_evidence(
    tmp_path: Path,
) -> None:
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "codely",
                "passed": True,
                "checks": _complete_codely_checks(),
                "evidence": ["missing-session.log"],
            }
        ),
        encoding="utf-8",
    )
    artifact_root = tmp_path / "artifacts"

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-Stage",
            "codely-proof",
            "-CodelyEvidencePath",
            str(proof),
            "-ArtifactsRoot",
            str(artifact_root),
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode != 0
    assert "referenced evidence file was not found" in completed.stdout
    assert not (artifact_root / "codely" / "gate.json").exists()


def test_single_mujoco_stage_runs_all_nine_scenarios(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-Stage",
            "mujoco-rollouts",
            "-ArtifactsRoot",
            str(tmp_path),
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "mujoco" / "rollout-report.json").read_text())
    assert report["passed"] is True
    assert report["summary"] == {"passed": 9, "failed": 0, "total": 9}


def test_ppo_stage_reuses_only_a_valid_exact_training_cache(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-Stage",
            "ppo-smoke",
            "-ArtifactsRoot",
            str(tmp_path),
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    if result["stages"][0]["status"] != "cached":
        repeated = subprocess.run(
            completed.args,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert repeated.returncode == 0, repeated.stdout + repeated.stderr
        assert json.loads(repeated.stdout)["stages"][0]["status"] == "cached"
    report = json.loads((tmp_path / "training" / "ppo-smoke.json").read_text())
    assert report["passed"] is True
    assert report["task"] == "Mjlab-Velocity-Flat-MicroDuck"
    assert report["numEnvs"] == 64
    assert report["maxIterations"] == 5
    assert Path(report["checkpoint"]).name == "model_4.pt"
    assert report["checkpointBytes"] > 100_000
    assert len(report["checkpointSha256"]) == 64
    assert report["eventFiles"]
    assert all(item["bytes"] > 0 and len(item["sha256"]) == 64 for item in report["eventFiles"])
    assert len(report["agentConfigSha256"]) == 64
    assert len(report["environmentConfigSha256"]) == 64


def test_onnx_export_stage_reuses_only_an_official_contract_valid_cache(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-Stage",
            "onnx-export",
            "-ArtifactsRoot",
            str(tmp_path),
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    if result["stages"][0]["status"] != "cached":
        repeated = subprocess.run(
            completed.args,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert repeated.returncode == 0, repeated.stdout + repeated.stderr
        assert json.loads(repeated.stdout)["stages"][0]["status"] == "cached"
    report = json.loads((tmp_path / "training" / "onnx-export.json").read_text())
    assert report["passed"] is True
    assert report["input"] == {"name": "obs", "shape": [1, 61]}
    assert report["output"] == {"name": "actions", "shape": [1, 14]}
    assert report["randomizedFiniteRuns"] == 100
    assert len(report["sha256"]) == 64
    assert len(report["sourceCheckpointSha256"]) == 64
    assert len(report["attestationSha256"]) == 64


def test_single_stage_and_force_training_are_reflected_in_dry_run() -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-Stage",
            "ppo-smoke",
            "-ForceTraining",
            "-DryRun",
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["forceTraining"] is True
    assert [stage["name"] for stage in payload["stages"]] == ["ppo-smoke"]
    train = next(
        command
        for command in payload["stages"][0]["commands"]
        if command["label"] == "train exact PPO smoke"
    )
    assert train["arguments"][train["arguments"].index("--env.scene.num-envs") + 1] == "64"
    assert train["arguments"][train["arguments"].index("--agent.max-iterations") + 1] == "5"


def test_explicit_missing_tuanjie_editor_is_rejected_even_in_dry_run(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "Tuanjie.exe"
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-Stage",
            "tuanjie-editmode",
            "-TuanjiePath",
            str(missing),
            "-DryRun",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode != 0
    assert "Tuanjie editor does not exist" in (completed.stdout + completed.stderr)


def test_tuanjie_dry_run_uses_pinned_editor_test_validation_and_real_build_entry() -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-DryRun",
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    stages = {stage["name"]: stage for stage in payload["stages"]}
    assert "2022.3.62t14" in payload["tuanjieEditor"]
    robot_import = stages["robot-manifest"]["commands"][1]
    assert "AgenticRobot.MicroDuck.Editor.DemoSceneBuilder.CreateAllSceneAssets" in (
        robot_import["arguments"]
    )
    assert any(
        path.endswith("Generated\\Scenes\\MicroDuckMvp.unity")
        for path in stages["robot-manifest"]["artifacts"]
    )
    assert any(
        path.endswith("Generated\\Scenes\\MicroDuckNativeMvp.unity")
        for path in stages["robot-manifest"]["artifacts"]
    )
    assert sum(
        "Generated\\MuJoCo" in path and path.endswith(".prefab")
        for path in stages["robot-manifest"]["artifacts"]
    ) == 3
    for name, platform in (("tuanjie-editmode", "EditMode"), ("tuanjie-playmode", "PlayMode")):
        commands = stages[name]["commands"]
        assert platform in commands[0]["arguments"]
        assert commands[1]["arguments"][1] == "tuanjie-results"
    assert any(
        path.endswith("tuanjie-native-mujoco-policy-behavior.json")
        for path in stages["tuanjie-editmode"]["artifacts"]
    )
    build = stages["windows-build"]["commands"][0]
    assert "AgenticRobot.MicroDuck.Editor.DemoSceneBuilder.BuildWindows64" in build["arguments"]
    assert any(path.endswith("Builds\\Windows64\\AgenticRobotGame.exe") for path in stages["windows-build"]["artifacts"])
    environment = stages["environment-acceptance"]
    assert any("run-visual-acceptance.ps1" in argument for argument in environment["commands"][0]["arguments"])
    assert any(path.endswith("latest-report.json") for path in environment["artifacts"])
    assert any(path.endswith("latest-validation.json") for path in environment["artifacts"])


def test_tuanjie_commands_are_waited_for_instead_of_trusting_the_detached_launcher() -> None:
    script = (ROOT / "scripts" / "run-mvp.ps1").read_text(encoding="utf-8")

    assert "Start-Process" in script
    assert "-PassThru" in script
    assert ".WaitForExit(1800000)" in script
    assert "timed out after 30 minutes" in script
    assert "Tuanjie.exe" in script


def test_bootstrap_plan_names_both_immutable_upstream_commits() -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-Stage",
            "bootstrap",
            "-DryRun",
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    commands = json.loads(completed.stdout)["stages"][0]["commands"]
    serialized = json.dumps(commands)
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    for repository in lock["repositories"].values():
        assert repository["url"] in serialized
        assert repository["commit"] in serialized


def test_clean_generated_dry_run_is_workspace_scoped_and_preserves_codely_proof() -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-CleanGenerated",
            "-DryRun",
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    clean = payload["clean"]
    assert clean["requested"] is True
    assert clean["performed"] is False
    root = ROOT.resolve()
    assert clean["targets"]
    assert all(Path(path).resolve().is_relative_to(root) for path in clean["targets"])
    assert str((ROOT / ".cache" / "upstream").resolve()) not in clean["targets"]
    assert str((ROOT / ".cache" / "artifacts").resolve()) not in clean["targets"]
    assert str(
        (ROOT / "TuanjieProject" / "Assets" / "MicroDuck" / "Generated" / "MuJoCo").resolve()
    ) in clean["targets"]
    assert clean["preserved"] == [str((ROOT / "artifacts" / "mvp" / "codely").resolve())]


def test_trace_stage_exports_a_real_tuanjie_tick_then_audits_barracuda_output() -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-Stage",
            "trace-parity",
            "-DryRun",
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    stage = json.loads(completed.stdout)["stages"][0]
    assert len(stage["commands"]) == 2
    assert "AgenticRobot.MicroDuck.Editor.TraceBatchExporter.ExportOnePolicyBatch" in stage["commands"][0]["arguments"]
    audit_arguments = stage["commands"][1]["arguments"]
    assert audit_arguments[1] == "trace-parity"
    policy_directory = audit_arguments[audit_arguments.index("--policy-directory") + 1]
    assert policy_directory.endswith("Generated\\Policies\\Barracuda")
    assert any(path.endswith("tuanjie-alpha_stand.jsonl") for path in stage["artifacts"])


def test_clean_generated_rejects_external_targets_before_even_a_dry_run(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-CleanGenerated",
            "-ArtifactsRoot",
            str(tmp_path),
            "-DryRun",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode != 0
    assert "Refusing to clean a path outside the workspace" in (
        completed.stdout + completed.stderr
    )


def test_clean_generated_rejects_arbitrary_workspace_artifact_roots() -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-CleanGenerated",
            "-ArtifactsRoot",
            str(ROOT / "src"),
            "-DryRun",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode != 0
    assert "fixed default artifacts root" in (completed.stdout + completed.stderr)


def test_clean_generated_rejects_dependency_broken_single_stage_runs() -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-CleanGenerated",
            "-Stage",
            "tuanjie-playmode",
            "-DryRun",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode != 0
    assert "only supported with -Stage all" in (completed.stdout + completed.stderr)


def test_tuanjie_plans_require_identity_checked_native_tests_and_player_smoke() -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-DryRun",
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    stages = {stage["name"]: stage for stage in json.loads(completed.stdout)["stages"]}
    edit_validation = stages["tuanjie-editmode"]["commands"][1]["arguments"]
    play_validation = stages["tuanjie-playmode"]["commands"][1]["arguments"]
    assert any(
        argument.endswith(
            "AuthoritativeMujocoPolicyBehaviorTests."
            "AllOfficialPoliciesAndLiveSameModelHotSwapsMeetBehaviorContracts"
        )
        for argument in edit_validation
    )
    assert any(
        argument.endswith(
            "NativeMujocoScenePlayModeTests."
            "NativeSceneRunsBarracudaPolicyAndRecreatesOnlyForRobotVariantChanges"
        )
        for argument in play_validation
    )
    assert any(
        argument.endswith(
            "NativeMujocoScenePlayModeTests."
            "NativeRobotRemainsInsideAUsableCameraFrame"
        )
        for argument in play_validation
    )
    assert any(
        argument.endswith(
            "NativeMujocoScenePlayModeTests."
            "NativeStandPolicyKeepsTheVisibleDuckUprightForFourSeconds"
        )
        for argument in play_validation
    )
    assert any(
        argument.endswith(
            "NativeMujocoScenePlayModeTests."
            "NativeWalkingPolicyMovesForwardWhileRemainingUprightForSixSeconds"
        )
        for argument in play_validation
    )
    assert any(
        argument.endswith(
            "NativeMujocoScenePlayModeTests."
            "KeyboardPolicySwitchesRunAfterMujocoSceneLateUpdate"
        )
        for argument in play_validation
    )
    play_command = stages["tuanjie-playmode"]["commands"][0]
    assert "-nographics" not in play_command["arguments"]
    assert any(
        path.endswith("artifacts\\visual-acceptance\\playmode-camera.png")
        for path in stages["tuanjie-playmode"]["artifacts"]
    )
    assert any(command["arguments"][1] == "native-behavior" for command in stages["tuanjie-editmode"]["commands"])
    assert any(command["arguments"][1] == "player-smoke" for command in stages["windows-build"]["commands"])
    build_artifacts = stages["windows-build"]["artifacts"]
    assert any(path.endswith("AgenticRobotGame_Data\\Plugins\\x86_64\\mujoco.dll") for path in build_artifacts)
    assert any(path.endswith("player-smoke.json") for path in build_artifacts)


def test_bootstrap_checks_upstream_cleanliness_even_when_head_is_already_locked() -> None:
    script = (ROOT / "scripts" / "run-mvp.ps1").read_text(encoding="utf-8")

    cleanliness = script.index('status", "--porcelain"')
    commit_branch = script.index("if ($actualCommit -ne [string]$repository.commit)")
    assert cleanliness < commit_branch


def test_training_runtime_validator_never_executes_checkpoint_pickle_payloads() -> None:
    source = (ROOT / "scripts" / "validate-training-runtime.py").read_text(
        encoding="utf-8"
    )

    assert "weights_only=True" in source
    assert "weights_only=False" not in source


def test_native_mujoco_binary_has_immutable_source_and_build_gate_contract() -> None:
    lock = json.loads((ROOT / "upstream.lock.json").read_text(encoding="utf-8"))
    native = lock["nativeBinaries"]["mujocoWindowsX64"]

    assert native["url"].startswith("https://files.pythonhosted.org/")
    assert len(native["archiveSha256"]) == 64
    assert native["archiveMember"] == "mujoco/mujoco.dll"
    assert len(native["sha256"]) == 64
    assert native["projectPath"].endswith("Assets/Plugins/x86_64/mujoco.dll")

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run-mvp.ps1"),
            "-Stage",
            "windows-build",
            "-DryRun",
            "-Json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    commands = json.loads(completed.stdout)["stages"][0]["commands"]
    validation_arguments = commands[-1]["arguments"]
    assert "--upstream-lock" in validation_arguments
    assert str((ROOT / "upstream.lock.json").resolve()) in validation_arguments
