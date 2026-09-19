# MicroDuck-Unity-Sim2Sim

MicroDuck 的 **Unity / 团结引擎版 Sim2Sim** 工程：在团结引擎
(`2022.3.62t14`, `1.10.2`) 中运行并验证官方 MuJoCo 模型与 ONNX 策略。

这是 [MicroDuck-Godot-Simi2Sim](https://github.com/sgyli7/MicroDuck-Godot-Simi2Sim)
的 Unity / 团结引擎对应实现，不是 Godot/Jolt 项目。项目以 MuJoCo 的原生
运行时作为权威物理后端；团结引擎负责场景、输入、渲染、Windows 构建和
Barracuda ONNX 推理，同时保留 PhysX/ArticulationBody 路径用于校准与比较。
由于 MicroDuck 3D 模型的许可证限制，本仓仅面向非商业验证与研究使用。

The accepted runtime architecture is:

```text
Tuanjie input and rendering
          |
          v
61-value policy observation -> Barracuda 3.0.1 -> 14 servo targets
          ^                                      |
          |                                      v
     official MuJoCo 3.12 native state <- 200 Hz physics
                         policy inference: 50 Hz
```

MuJoCo is the physics authority in the playable MVP. Tuanjie owns the scene,
input, rendering, and build. Barracuda is the ONNX runtime. **Sentis is not a
project dependency.** The separate PhysX/ArticulationBody importer remains a
calibration and comparison path only; it is not used to judge policy behavior.

See [the native runtime architecture](docs/native-mujoco-architecture.md) for
the ownership boundaries and generated assets.

## Locked stack

| Component | Locked version | Purpose |
| --- | --- | --- |
| Tuanjie Engine | `2022.3.62t14` (`1.10.2`) | Editor, input, rendering, tests, Windows build |
| Codely Bridge | `1.0.78` | Editor automation; development-time only |
| `org.mujoco` | `3.12.0`, commit `13827e9ee56f097f57acf69ae52b078f9839682d` | Official Unity importer and native physics runtime |
| Barracuda | `3.0.1`, commit `eefb8cb01e6c6a3a9f839e93bd5d0338f0b5a2f5` | CPU ONNX inference (`61D -> 14D`) |
| MicroDuck | commit `9f7eaad1008fffd90ef871a33a18aecd066b51a9` | Robot source and nine official ONNX policies |
| `microduck_rl` | commit `5946fd9cdbc58956424420153e51975af3b30d77` | MJCF models, PPO, and export source |

Repository pins live in `upstream.lock.json`, `TuanjieProject/Packages/manifest.json`,
and `TuanjieProject/Packages/packages-lock.json`.

## Reproducible MVP runner

To **revalidate this delivered workspace and its verified PPO/ONNX cache**, the
prerequisites are a Windows x64 machine, Git, `uv`, and Tuanjie
`2022.3.62t14`. The full run also expects the independently captured Codely
RED/GREEN proof described below. Python environments and the two pinned
upstream repositories are then bootstrapped by the runner.

A first training/export with no `.cache`, or an intentional `-ForceTraining`
run, additionally requires an NVIDIA CUDA-capable GPU, a working NVIDIA
driver, and enough disk space for the CUDA Torch environment. The official
export command is locked to `cuda:0`; CPU-only machines can validate and run
the delivered ONNX policies but cannot currently regenerate that PPO/export
cache. `.cache` is deliberately not source-controlled, while
`-CleanGenerated` preserves the already-validated local cache.

The built player loads the official MuJoCo DLL, which uses the Microsoft Visual
C++ 2015-2022 x64 runtime. This machine already has it; install that redistributable
before copying the build to a clean Windows machine.

The Windows DLL is copied from MuJoCo's official 3.12.0 PyPI wheel. Its wheel
URL, wheel SHA-256, archive member, extracted DLL SHA-256, and project path are
locked in `upstream.lock.json`. Bootstrap checks the installed and project DLLs;
the final build gate separately requires the packaged DLL to have the same hash.

From PowerShell at the repository root:

```powershell
# Inspect every command and expected artifact without changing the workspace.
.\scripts\run-mvp.ps1 -DryRun
.\scripts\run-mvp.ps1 -DryRun -Json

# Run the complete ordered acceptance pipeline. A valid 64 x 5 PPO cache and
# its official ONNX export are verified and reused automatically.
.\scripts\run-mvp.ps1

# Delete and rebuild regenerable editor/build outputs. Pinned upstream sources,
# the PPO/ONNX cache, and Codely evidence are deliberately preserved.
.\scripts\run-mvp.ps1 -CleanGenerated

# Run one independently repeatable gate, or deliberately retrain PPO.
.\scripts\run-mvp.ps1 -Stage policy-audit -Json
.\scripts\run-mvp.ps1 -Stage ppo-smoke -ForceTraining
```

The ordered gates are `bootstrap`, `python-tests`, `policy-audit`,
`mujoco-rollouts`, `ppo-smoke`, `onnx-export`, `robot-manifest`,
`tuanjie-editmode`, `tuanjie-playmode`, `trace-parity`, `codely-proof`, and
`windows-build`, followed by `environment-acceptance`. The final gate opens the
fresh Player for a 24-second automated seven-terrain/five-camera capture, closes
it, and fails unless the report, Player hash, log, PNGs, and MP4 validate. It
therefore requires an interactive Windows desktop. List all gates with:

```powershell
.\scripts\run-mvp.ps1 -ListStages
```

Tuanjie is discovered from the supported Hub locations. Override it when
needed with `-TuanjiePath 'D:\path\to\Tuanjie.exe'`.

## macOS

Windows PowerShell 入口保持不变。Apple Silicon 上的对等链路是
`scripts/run-mvp-macos.py`：同一组 Python / Tuanjie gate，加上 `BuildMacOS`
与 in-player 巡演，并且把 CUDA 训练、Windows 构建和缺席的 Codely proof 显式记为
`skipped` / `not-applicable`。前置条件、阶段对照、产物路径和用户操作项见
[macOS 部署与验收](docs/macos-deployment.md)。

```bash
python3 scripts/run-mvp-macos.py --json
open Builds/macOS/AgenticRobotGame.app
```

编辑器打开 `TuanjieProject/Assets/MicroDuck/Generated/Scenes/MicroDuckNativeMvp.unity`。
覆盖编辑器路径时使用 `--tuanjie-path` 或 `TUANJIE_EDITOR`；默认锁定
`/Applications/Tuanjie/Hub/Editor/2022.3.62t14/Tuanjie.app/Contents/MacOS/Tuanjie`。

The important outputs are:

- `artifacts/mvp/mvp-report.json`: ordered pass/fail report and stage timings;
- `artifacts/mvp/mujoco`: nine headless MuJoCo rollout traces and verdicts;
- `artifacts/mvp/tuanjie`: editor/player test XML, reports, and logs;
- `artifacts/mvp/tuanjie/player-smoke.json`: proof that the freshly built player
  loaded MuJoCo 3.12.0 and completed finite Barracuda policy ticks;
- `artifacts/mvp/tuanjie/windows-build-report.json`: SHA-256 manifest for the
  complete Windows bundle plus its smoke-test verdict;
- `artifacts/mvp/environment-acceptance/latest-report.json`: latest structured
  Player tour with named terrain and camera captures;
- `artifacts/mvp/traces`: Tuanjie-to-ONNX policy trace parity evidence;
- `.cache/artifacts/microduck-ppo-64x5.onnx`: official export from the validated
  `64`-environment, `5`-iteration PPO smoke checkpoint;
- `Builds/Windows64/AgenticRobotGame.exe`: Windows x64 player.

Any failed gate returns a non-zero exit code. The run report identifies the
first failed stage. Tuanjie XML is accepted only when the named native/import,
Barracuda, behavior, and scene E2E tests appear in a fresh result file. The
native behavior report must contain all nine policies and both live hot-swap
scenarios; a shallow `passed: true` marker is rejected.

### Codely evidence

Codely is a development harness, not a player dependency. The gate is
intentionally read-only: it validates an independently captured
`artifacts/mvp/codely/proof.json` and never invents or replaces proof. A full
run fails fast when that file or one of its referenced RED/GREEN artifacts is
missing. Use `-CodelyEvidencePath` only when the proof is stored elsewhere.

## Play the native scene

Generate the assets with `run-mvp.ps1` (or the Tuanjie menu command
`MicroDuck/Create Native MuJoCo MVP Scene`), then open:

```text
TuanjieProject/Assets/MicroDuck/Generated/Scenes/MicroDuckNativeMvp.unity
```

The Windows build starts from this native scene. The overlay shows the active
policy, backend, policy-tick count, and current fault/health state.

The ready-to-run player is `Builds/Windows64/AgenticRobotGame.exe`; keep its
entire sibling directory together rather than copying the executable alone.

Policy slots:

| Key | Policy | Robot |
| --- | --- | --- |
| `1` | walking | legged |
| `2` | stand | legged |
| `3` | sit / stand | legged |
| `4` | ground pick | legged |
| `5` | left kick | legged + ball |
| `6` | right kick | legged + ball |
| `7` | roller locomotion | roller |
| `8` | roller crouch | roller |
| `9` | roulade | legged |

Controls:

| Keys | Command |
| --- | --- |
| `W` / `S` | forward / backward |
| `A` / `D` | left / right |
| `Q` / `E` | yaw left / right |
| `I` / `K` | neck pitch |
| Arrow keys | head pitch and yaw |
| `J` / `L` | head roll |
| `Page Up` / `Page Down` | body height |
| `Z` / `X` | body roll |
| `,` / `.` | body pitch |
| `Space` | toggle sit/stand, or start the selected phase-based skill |
| `R` | reset the active robot |
| `T` / `Shift+T` | next / previous terrain and reset on its verified spawn |
| `C` | cycle Side, Rear, Top, and Showcase camera presets |
| `Tab` | toggle Follow / FreeFly camera mode |
| `F` | focus the camera on MicroDuck |
| Follow: left-drag / wheel | orbit / zoom |
| FreeFly: right-drag, `WASD`, `Q`/`E`, wheel | look, move, and change speed |

The world contains the central safety plaza, all four upstream terrain families
(pyramid stairs, seeded random grid, pyramid slope, and 2°/11°/20° roller
slope), plus rock/stepping-stone and regular stair/bridge extensions. The HUD
labels training-matched versus extension terrain. All robot/terrain contacts
remain native MuJoCo contacts; decorative alpine scenery has no hidden PhysX
collision.

To create a fresh review bundle and leave the exact verified Player open on the
flat plaza with the stand policy selected:

```powershell
.\scripts\run-visual-acceptance.ps1 -KeepPlayerOpen
```

Same-robot policy changes use a state-preserving hot swap. Switching between
legged and roller selects the matching imported MuJoCo model.

## ONNX compatibility and trace scope

All nine original ONNX files remain preserved and hash-recorded. The asset
pipeline audits their exact `float[1,61] -> float[1,14]` contract and produces
Barracuda copies in `Assets/MicroDuck/Generated/Policies/Barracuda`. Those
copies only lower the declared ONNX opset to the tested Barracuda-compatible
value; fixtures verify output parity and record both hashes in
`compatibility-report.json`.

The `trace-parity` gate exports a real Tuanjie/Barracuda policy tick and checks
the action against ONNX Runtime within `1e-5`. This proves the policy I/O loop;
it is not a general-purpose round-trip exporter for arbitrary Tuanjie scenes.

The stricter sustained-behavior oracle runs on official native MuJoCo. The old
PhysX armature/hold/behavior comparisons remain explicit diagnostics and are
reported as intentional skips in the default PlayMode suite; they do not gate
the native MVP.

## Licensing boundary

The MicroDuck and `microduck_rl` software repositories include Apache-2.0
licenses, while the upstream README identifies the 3D model files as Creative
Commons BY-SA-NC. Generated models, meshes, and prefabs therefore remain
restricted to internal, non-commercial validation in this repository. Review
attribution, share-alike obligations, and commercial rights before publishing
or shipping any derived model asset. Provenance and hashes are copied to
`TuanjieProject/Assets/MicroDuck/Generated/Licenses`.

NWH Vehicle Physics is separately licensed and deliberately not copied into
this repository. It is deferred to a later conventional-vehicle slice; see
[the NWH assessment](docs/nwh-assessment.md).
