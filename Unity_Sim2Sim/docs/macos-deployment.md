# macOS 部署与验收

与 Windows 完全解耦的 MicroDuck Sim2Sim 本机链路。Windows 入口仍是
`scripts/run-mvp.ps1`；macOS 只使用本页命令，不要调用 `.ps1`、
`BuildWindows64` 或 `mujoco.dll`。

锁定编辑器为 Tuanjie `2022.3.62t14`（团结引擎 `1.10.2`）。本机若同时装有
`t12`，一律使用 t14。

## 前置条件

- macOS arm64（本任务在 MacBook Air M5 上验收）
- Tuanjie Hub 已安装 `2022.3.62t14`，并带 **Mac Build Support**
- 团结引擎**个人版许可证**已在本机登录（`batchmode` 否则会在编辑器许可处失败）
- `uv`（Homebrew：`brew install uv`）
- `ffmpeg`（巡演录像；`brew install ffmpeg`）
- `hf`（Hugging Face CLI；缺失时编排脚本会执行 `uv tool install huggingface_hub`）
- Xcode Command Line Tools（`lipo`、`codesign`）

可选覆盖编辑器路径：

```bash
export TUANJIE_EDITOR="/Applications/Tuanjie/Hub/Editor/2022.3.62t14/Tuanjie.app/Contents/MacOS/Tuanjie"
```

解析顺序：`--tuanjie-path` > 环境变量 `TUANJIE_EDITOR` > 上述默认路径。

## 一条命令入口

在仓库根目录：

```bash
python3 scripts/run-mvp-macos.py
python3 scripts/run-mvp-macos.py --json
python3 scripts/run-mvp-macos.py --list-stages --json
python3 scripts/run-mvp-macos.py --dry-run --json
python3 scripts/run-mvp-macos.py --stage bootstrap --stage python-tests --json
```

Gate 一律调用 `.venv/bin/python scripts/mvp-gates.py`。不要启动训练：
`ppo-smoke` / `onnx-export` 固定为 skipped。

手动打开场景（编辑器）：

```text
TuanjieProject/Assets/MicroDuck/Generated/Scenes/MicroDuckNativeMvp.unity
```

手动打开已构建 Player：

```bash
open Builds/macOS/AgenticRobotGame.app
```

## Win / Mac 阶段对照

| Stage | Windows (`run-mvp.ps1`) | macOS (`run-mvp-macos.py`) |
| --- | --- | --- |
| bootstrap | uv sync、上游 SHA、Windows DLL hash、CUDA probe | uv sync、上游 SHA、`mujoco.dylib` hash 对锁、`lipo` 对照 dylib/编辑器、编辑器版本对锁 |
| python-tests | pytest + ruff | 相同 |
| policy-audit | 九个 ONNX | 相同 |
| mujoco-rollouts | 九个 headless | 相同 |
| ppo-smoke | CUDA 训练或 cache | **skipped**：`requires CUDA (microduck_rl/mjlab Warp); not available on Apple Silicon` |
| onnx-export | 官方 `cuda:0` 导出 | **skipped**：同上 |
| robot-manifest | assets + `CreateAllSceneAssets` | 相同 executeMethod |
| tuanjie-editmode | required ×6 + `native-behavior` | 相同 required-test 与 `--not-before-utc` |
| tuanjie-playmode | required ×8，allowed-skipped ×3，真实图形 | 相同；无 `-nographics` |
| trace-parity | `ExportOnePolicyBatch` + 1e-5 | 相同 |
| codely-proof | 缺 proof 则 fail-closed | **not-applicable**（仅当 `artifacts/mvp/codely/proof.json` 存在才跑 evidence gate） |
| windows-build | `BuildWindows64` + player-smoke | **not-applicable**：Windows player 构建在 macOS 上不适用 |
| macos-build | n/a | `BuildMacOS` + 无头 smoke + `player-smoke-macos` |
| environment-acceptance | `run-visual-acceptance.ps1` SendInput | `macos-environment-acceptance`：in-player tour + `run-visual-acceptance-macos.py` |
| training-prep | n/a（训练在 ppo-smoke） | 仅库存：策略哈希、HF 旁证、无 key 则 W&B blocked、无 `.pt` 则 checkpoint blocked；`warp`/`mjlab` 记 skipped（CUDA）。库存 `failed` 会让该 stage 失败，不再写成 passed |

非 `passed` 的状态必须带 `reason`。平台不支持项记 `skipped` 或 `not-applicable`，绝不记 `passed`。`mvp-report.json` 的 `passed` 仅当没有任何 `failed`。

Tuanjie 每次命令超时 1800 s，日志在 `artifacts/mvp/tuanjie/`。命令结束后脚本会
`git checkout -- TuanjieProject/ProjectSettings/ProjectVersion.txt`，保持锁定
`2022.3.62t14` / `1.10.2`。

## 用户操作项

- 在 Tuanjie Hub 登录个人版许可证；否则 `batchmode` 无法跑 EditMode/PlayMode/构建
- 若 Codely 包 `cn.tuanjie.codely.bridge` 解析失败，这是用户侧阻塞，不是跳过条件
- 官方 PPO / ONNX 再生需要 NVIDIA CUDA 机器；Apple Silicon 不能跑通训练闭环
- 列出 W&B run `yr25mna4` 的 `model_9999.pt` 需要设置 `WANDB_API_KEY`（或 `wandb login`）
- `hostile-terrain` 上游提交 `6cd45fc` 与锁定的 `microduck_rl` `5946fd9` 不是同一提交，未接入本验收

## 产物路径

| 产物 | 路径 |
| --- | --- |
| 总报告 | `artifacts/mvp/mvp-report.json` |
| bootstrap | `artifacts/mvp/bootstrap.json` |
| pytest coverage | `artifacts/mvp/python/coverage.json` |
| 策略审计 | `artifacts/mvp/policy-audit.json` |
| MuJoCo rollouts | `artifacts/mvp/mujoco/` |
| EditMode / PlayMode / smoke | `artifacts/mvp/tuanjie/` |
| trace-parity | `artifacts/mvp/traces/trace-parity.json` |
| macOS Player | `Builds/macOS/AgenticRobotGame.app` |
| 巡演报告与画面 | `artifacts/mvp/environment-acceptance/`（`latest-report.json`、session 目录内 mp4 / 7 地形 PNG / 5 相机 PNG） |
| 训练准备库存 | `artifacts/mvp/training/macos-training-prep.json` |
| HF 旁证（不进基线目录） | `.cache/hf/microduck-rough-walk-e/` |
