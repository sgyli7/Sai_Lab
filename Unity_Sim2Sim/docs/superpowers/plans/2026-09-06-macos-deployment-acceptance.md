# macOS Deployment Acceptance Implementation Plan

> **For agentic workers:** execute this plan on the user's MacBook Air M5. Do not weaken gates, invent hashes, or merge to `main`.

**Goal:** Add a macOS Sim2Sim acceptance path that is fully decoupled from Windows (`run-mvp.ps1`, `run-visual-acceptance.ps1`, `BuildWindows64`) and actually run it on this machine.

**Architecture:** Keep the existing Windows PowerShell runner and DLL gate untouched. Add a native `mujoco.dylib` plugin, a Tuanjie macOS player build, an in-player tour that drives the same APIs as the Windows key sequence, and a stdlib `run-mvp-macos.py` orchestrator with explicit `passed` / `cached` / `skipped` / `not-applicable` / `failed` states.

**Tech Stack:** Tuanjie `2022.3.62t14` locked (`1.10.2`); this machine currently has `2022.3.62t12` (`1.10.0`, arm64) as a recorded deviation. Official MuJoCo 3.12.0 universal2 dylib, Barracuda 3.0.1 CPU, Python 3.12 via `uv`, native MuJoCo 3.12 + ONNX Runtime 1.29.0.

---

## Constraints

- Do not change `scripts/run-mvp.ps1`, `scripts/run-visual-acceptance.ps1`, `scripts/validate-environment-acceptance.py`, `mujoco.dll` bytes, `BuildWindows64`, the Windows branch of `_player_smoke`, `config/policy-scenarios.json`, any test threshold, required/allowed-skipped test lists, `Packages/manifest.json`, `packages-lock.json`, or `ProjectVersion.txt`.
- Do not change physics parameters, weaken assertions, or skip failures to manufacture a pass.
- Distinguish `failed` / `skipped` (with reason) / `not-applicable`. Platform-unsupported items must never be recorded as passed.
- Do not write placeholder hashes. If a digest cannot be obtained, omit it and report a blocker.
- One conventional commit per logical change on `feature/macos-deployment-acceptance`. Do not create a PR or merge `main`. Do not commit `.cache/`, `artifacts/`, `.venv/`, `Builds/`, `Library/`.
- Do not start long training. `ppo-smoke` / `onnx-export` must not execute.
- Tuanjie: if Hub contains `2022.3.62t14`, use it. If only `t12`, finish Python first, then open the project with t12. After every Tuanjie command: `git checkout -- TuanjieProject/ProjectSettings/ProjectVersion.txt`. Annotate `editor 2022.3.62t12 ≠ locked t14` in every report.
- Tuanjie binary: `TUANJIE="/Applications/Tuanjie/Hub/Editor/<ver>/Tuanjie.app/Contents/MacOS/Tuanjie"`. Timeout 1800 s. Logs under `artifacts/mvp/tuanjie/*.log`. License/Hub login or Codely package resolve failure is a user-action blocker, not something to bypass.

## File map

- Create: `docs/macos-deployment.md`, `scripts/run-mvp-macos.py`, `scripts/run-visual-acceptance-macos.py`
- Create: `TuanjieProject/Assets/Plugins/macOS/mujoco.dylib` + `.meta`
- Create: `TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoPlayerAcceptanceTour.cs`
- Create: `TuanjieProject/Assets/MicroDuck/Tests/EditMode/MujocoPlayerAcceptanceTourTests.cs`
- Create: `tests/test_run_mvp_macos_script.py`, `tests/test_run_visual_acceptance_macos.py`
- Modify: `upstream.lock.json` (`nativeBinaries.mujocoMacOSUniversal2`)
- Modify: `TuanjieProject/Assets/Plugins/x86_64/mujoco.dll.meta` (`Standalone: OSXUniversal` enabled 0 only)
- Modify: `TuanjieProject/Assets/MicroDuck/Editor/DemoSceneBuilder.cs` (`CreateMacOSBuildOptions` / `BuildMacOS`, tour component)
- Modify: `scripts/mvp-gates.py` (`player-smoke-macos`, shared `_smoke_checks`)
- Modify: `tests/test_run_mvp_script.py` (win32 skipif), `tests/test_mvp_gates.py`, `tests/test_upstream_lock.py`
- Modify: `README.md` (macOS section)

---

## Win / Mac stage对照

| Stage | Windows (`run-mvp.ps1`) | macOS (`run-mvp-macos.py`) |
| --- | --- | --- |
| bootstrap | uv sync, upstream, Windows DLL hash, CUDA probe | uv sync, upstream, dylib hash vs lock, lipo vs editor, version vs lock |
| python-tests | pytest + ruff | same |
| policy-audit | nine ONNX | same |
| mujoco-rollouts | nine headless | same |
| ppo-smoke | CUDA train or cache | `skipped`: requires CUDA (microduck_rl/mjlab Warp); not available on Apple Silicon |
| onnx-export | official `cuda:0` export | `skipped`: same CUDA reason |
| robot-manifest | assets + CreateAllSceneAssets | same Tuanjie args |
| tuanjie-editmode | required ×6 + native-behavior | same required tests and args |
| tuanjie-playmode | required ×7, allowed-skipped ×3, graphics | same; no `-nographics` |
| trace-parity | ExportOnePolicyBatch + 1e-5 | same |
| codely-proof | fail closed if missing | `not-applicable` unless `artifacts/mvp/codely/proof.json` exists |
| windows-build | BuildWindows64 + player-smoke | `not-applicable` |
| macos-build | n/a | BuildMacOS + player-smoke-macos |
| environment-acceptance | `run-visual-acceptance.ps1` SendInput | in-player tour + `run-visual-acceptance-macos.py` |
| training-prep | n/a (training is ppo-smoke) | inventory only: HF sidecar, W&B blocked without key, no `.pt` |

---

## Phase 0 — Inventory

Write `artifacts/mvp/macos-inventory.json` (not committed): `sw_vers`, `uname -m`, `hw.memsize`, `xcode-select -p`, brew/uv/git/python3/ffmpeg/hf, Tuanjie versions and `lipo -archs`, MacStandaloneSupport presence, git status.

Missing ffmpeg/hf: `brew install ffmpeg`, `uv tool install huggingface_hub`.

This machine: arm64, 24 GiB, only Tuanjie `2022.3.62t12` (arm64), MacStandaloneSupport present.

## Phase 1 — Branch + Python

- `git checkout -b feature/macos-deployment-acceptance`
- `uv sync --frozen --python 3.12`
- Verify `mujoco==3.12.0`, `onnxruntime==1.29.0`, `libmujoco.3.12.0.dylib` universal2
- Clone `.cache/upstream/{microduck,microduck_rl}` at locked SHAs, porcelain empty
- Baseline pytest (Windows script tests fail without `powershell.exe`), policy-audit, mujoco-rollouts, ruff

## Phase 2 — Plan + pytest platform mark

- This document.
- `tests/test_run_mvp_script.py`: module `pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="run-mvp.ps1 is the Windows entry point and needs powershell.exe")`.
- Confirm that file is fully skipped, remaining tests pass, coverage ≥80%. Separate commits.

## Phase 3 — macOS native library

- Download official `mujoco-3.12.0-macos-universal2.dmg` + `.sha256`, `shasum -a 256 -c`
- Mount, copy `libmujoco.3.12.0.dylib` → `TuanjieProject/Assets/Plugins/macOS/mujoco.dylib`
- Record `lipo -archs` (`x86_64 arm64`), codesign, xattr (no quarantine), sha256
- New `mujoco.dylib.meta` (Any 0; Editor OSX AnyCPU 1; OSXUniversal AnyCPU 1; Win/Win64/Linux64 0). After first Tuanjie open, commit the editor-generated meta.
- `mujoco.dll.meta`: `Standalone: OSXUniversal` enabled 0 only
- `upstream.lock.json` add `mujocoMacOSUniversal2` with real hashes
- Tests: both native entries, 64-hex sha256, file exists and matches

## Phase 4 — Tuanjie Editor gates

Use the same method names, required tests, and flags as `run-mvp.ps1`. Timeout 1800 s. Restore `ProjectVersion.txt` after each editor run.

1. `tuanjie_assets` then `CreateAllSceneAssets`. Require generated prefabs/scenes, `asset-report.json` passed, no `DllNotFoundException: mujoco`.
2. EditMode XML + `tuanjie-results` required ×6 + `native-behavior`.
3. PlayMode without `-nographics`, required ×7, allowed-skipped ×3.
4. `TraceBatchExporter.ExportOnePolicyBatch` + `trace-parity`.

Mac-only adapter fixes if logs show a first exception. Do not change assertions or thresholds.

### EditMode required tests (copy of `$EditModeRequiredTests`)

- `AgenticRobot.MicroDuck.Tests.AuthoritativeMujocoPolicyBehaviorTests.AllOfficialPoliciesAndLiveSameModelHotSwapsMeetBehaviorContracts`
- `AgenticRobot.MicroDuck.Tests.BarracudaParityTests.AllOfficialPoliciesMatchOnnxRuntimeReferenceFixtures`
- `AgenticRobot.MicroDuck.Tests.OfficialMujocoPluginTests.NativePluginLoadsAndStepsTheOfficialMicroDuckScene`
- `AgenticRobot.MicroDuck.Tests.OfficialMujocoPrefabImporterTests.ImportsAllExpandedOfficialScenesAsCompleteDeterministicPrefabs`
- `AgenticRobot.MicroDuck.Tests.DemoSceneBuilderTests.CreatesPlayableNativeSceneWithOfficialMuJoCoAndBarracudaControls`
- `AgenticRobot.MicroDuck.Tests.PolicyRuntimeContractTests.OutOfRangeFiniteActionFailsClosedAndClearsEveryAction`

### PlayMode required / allowed-skipped (copy of `$PlayModeRequiredTests` / `$PlayModeAllowedSkippedTests`)

Required:

- `...NativeMujocoScenePlayModeTests.KeyboardPolicySwitchesRunAfterMujocoSceneLateUpdate`
- `...NativeStandPolicyKeepsTheVisibleDuckUprightForFourSeconds`
- `...NativeWalkingPolicyMovesForwardWhileRemainingUprightForSixSeconds`
- `...NativeSceneRunsBarracudaPolicyAndRecreatesOnlyForRobotVariantChanges`
- `...NativeRobotRemainsInsideAUsableCameraFrame`
- `...OfficialPolicyPlayModeTests.GeneratedSceneTicksAllNinePoliciesThroughBarracudaAndArticulation`
- `...OfficialPolicyPlayModeTests.HotSwappingRollerPoliciesPreservesLiveRigAndControlState`
- `...KickBallPlayModeTests.BothKickPoliciesStrikeTheBallAndRemainStanding`

Allowed skipped:

- `...StrictPolicyBehaviorPlayModeTests.OfficialPoliciesMeetMuJoCoDerivedSustainedAndCompoundBehaviorContracts`
- `...OfficialPolicyPlayModeTests.AnkleStepResponseMatchesTheMuJoCoArmatureDynamics`
- `...OfficialPolicyPlayModeTests.DiagnosticHomeDrivesHoldTheRobotWithoutPolicyForTwoSeconds`

## Phase 5 — macOS build + smoke

- `CreateMacOSBuildOptions()` → `Builds/macOS/AgenticRobotGame.app`, `BuildTarget.StandaloneOSX`
- `BuildMacOS` menu: CreateAllSceneAssets → ConfigureBuildSettings → `SetPlatformSettings("OSXUniversal","Architecture","ARM64"|`"x64"`) → BuildPlayer
- EditMode test `CreatesMacOSBuildOptionsAtTheRepositoryBuildPath`
- Headless smoke via existing `MujocoPlayerSmokeProbe`: `passed`, nativeVersion `3012000`, backend `MuJoCo 3.12 + Barracuda 3.0.1 CPU`, policyTicks≥3, 61/14/14, allFinite
- If `Killed: 9`, `codesign --force --deep --sign -` then retry before hashing
- `mvp-gates.py player-smoke-macos`: required files from measured layout; shared `_smoke_checks` reused by Windows without behavior change; hash vs `mujocoMacOSUniversal2`
- Three Python tests: pass / missing dylib FileNotFoundError / tampered dylib fail
- Separate C# and Python commits

## Phase 6 — In-player environment tour

- `MujocoPlayerAcceptanceTour` (`[DefaultExecutionOrder(1001)]`), inert unless `-microduckTourReport` / `-microduckTourFrames`
- Event table copies `run-visual-acceptance.ps1` `$events` (same names, times, order)
- Keys map to APIs: digits → `SwitchPolicy` (same as `MujocoKeyboardPolicyInput`), R reset, C cycle, Tab toggle, F focus, T next terrain, W hold `SetTwist(0.2,0,0)`, orbit/zoom/FreeFly share camera APIs
- Capture PNG every 1/8 s for 24 s, then numeric segment: walking 6 s on `flat_plaza` (`WalkingMinForwardMeters=0.03`, `WalkingMinUprightDot=0.25` from `alpha_walking`), reset ≤0.05 m, hot-swap 1→2→1 / 1→7 / 7→8 / 8→1, tensors 61/14 finite
- `scripts/run-visual-acceptance-macos.py` launches player, ffmpeg (same flags as ps1), copies terrain/camera PNGs, calls unmodified `validate-environment-acceptance.py`, asserts `interactionChecks.passed`
- Static frames are not a pass. Separate C# and Python commits.

## Phase 7 — Orchestrator + docs

- `scripts/run-mvp-macos.py` stages listed in the Win/Mac table, including `training-prep`
- CLI: `--list-stages`, `--dry-run --json`, `--stage`, `--tuanjie-path`, `--artifacts-root`, `--json`
- `passed` only when no `failed`
- Docs: `docs/macos-deployment.md` + README macOS section
- Full `python3 scripts/run-mvp-macos.py`, then pytest coverage ≥80% and ruff

## Phase 8 — Close-out

- Clean `git status` for intended files
- `git log --oneline main..HEAD`
- `git push -u origin feature/macos-deployment-acceptance`
- No PR, no merge to `main`

---

## User-action items

- Tuanjie Hub / license login if the editor blocks batchmode
- Codely package `cn.tuanjie.codely.bridge` resolve failure
- Install locked editor `2022.3.62t14` if a t12/t14 behavioral difference appears
- W&B login (`WANDB_API_KEY`) for `yr25mna4` checkpoint listing
- CUDA GPU machine for official PPO/ONNX regeneration

## Recorded deviation (this machine)

`editor 2022.3.62t12 ≠ locked t14`. Hub lists only `2022.3.62t12`; resolved app is `/Users/liyunxuan/Workspace/Software/Tuanjie/1.10.0/Editor/Tuanjie.app` (arm64). `ProjectVersion.txt` must stay `2022.3.62t14` / `1.10.2`.
