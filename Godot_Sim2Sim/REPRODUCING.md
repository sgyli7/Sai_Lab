# 研究结果的获取、运行与复测

本轮结果见 [RESEARCH_RESULT_20260910.md](RESEARCH_RESULT_20260910.md)。
源码与模型分开交付：Git 保存代码、文档、模型清单与校验值；ONNX、训练权重、
日志和生成场景按既有 `.gitignore` 留在仓库外。模型尚未全面达到 MuJoCo 动作质量。

## 外部依赖

- Python 3.12，`uv sync --extra train` 安装训练和测试依赖。以后使用 `.venv/bin/python`
  或 `uv run --no-sync`，避免卸掉已安装的 train extra。
- Godot 4.7.2；以 `GODOT` 指定可执行文件。
- `microduck_rl` 的 MJCF 与网格，研究参考提交
  `5946fd9cdbc58956424420153e51975af3b30d77`。设置 `MICRODUCK_RL`，不在本仓重新发布网格。
- 原版九个 factory ONNX 放在 `MICRODUCK_POLICIES` 指向的目录：
  `alpha_stand.onnx`、`alpha_walking.onnx`、`alpha_sitstand.onnx`、
  `alpha_ground_pick.onnx`、`ball_kick_left.onnx`、`ball_kick_right.onnx`、
  `roulade.onnx`、`roller.onnx`、`roller_crouch.onnx`。
  完整旧模型 A/B 还需要九个 `*_Godot.onnx`。本次另附包含这 18 个原始快照的
  `MicroDuck_sim2sim_20260910_baselines.zip`，可按下文校验并解压。仅播放候选包不需要 factory 权重。

在本仓根目录执行，环境变量请指向自己机器上的资源：

```bash
export SIM2SIM_ROOT="$PWD"
export MICRODUCK_RL="$HOME/Projects/microduck_rl"
export MICRODUCK_POLICIES="$HOME/Projects/MicroDuck/policies"
export GODOT="$HOME/.local/bin/godot"
uv sync --extra train
.venv/bin/python -m sim2sim.research.setup scenes
```

`scenes` 生成标准、有球、修正脚底的有球及轮滑四个场景，并执行 Godot 资源导入。
已有完整场景保持原样；不完整的生成目录会明确报错。核验本轮物理时应在新的检出目录
生成场景，避免继承旧生成物。该准备步骤不训练，也不修改原版权重。

## 获取并运行本轮九模型包

包名：`MicroDuck_sim2sim_20260910_candidates.zip`，49,835,120 字节。
获取方式是使用随本轮结果交付的本地 ZIP，再复制到其他机器；尚未登记远端下载地址。
本机文件在 `results/research_20260910/`，Git clone 不会下载它。
可核验的来源与九模型哈希见 [artifacts/research_20260910.json](artifacts/research_20260910.json)。
压缩包对应实验提交 `0fe88910dc65185c1922d006ff0dd230fa89a967`；本次收尾只增加准备、
加载和复测入口，不重写该包或已封存的实验结果。

把 ZIP 放到上述相对目录后，从仓库根目录运行：

```bash
sha256sum -c artifacts/research_20260910.sha256
unzip -n results/research_20260910/MicroDuck_sim2sim_20260910_candidates.zip \
  -d results/research_20260910
export BUNDLE="$PWD/results/research_20260910/delivery_v3"
.venv/bin/python -m sim2sim.research.bundle "$BUNDLE"
.venv/bin/python -m sim2sim.research.bundle "$BUNDLE" --play
# 轮滑：
.venv/bin/python -m sim2sim.research.bundle "$BUNDLE" --play --roller
```

`bundle` 逐项核对九个 ONNX 及 manifest 的哈希、61→14 维度和有限推理输出，
从包所在位置解析模型。它不使用历史 `bank.json` 里的原机器绝对路径，不改动封存文件。
原包 `play.sh` 保留当时机器路径；迁移机器后使用上面的入口。
右踢 KR06 和前滚需要元数据声明的时间输入，右踢还需要相对起始航向。
当前播放代码提供这些量，不应将它们接到全零命令的旧控制器。

## 初始化新实验及测试

如需复用本轮完全相同的原版／旧模型对照，将配套的
`MicroDuck_sim2sim_20260910_baselines.zip` 放入 `results/research_20260910/`：

```bash
sha256sum -c artifacts/research_20260910_baselines.sha256
unzip -n results/research_20260910/MicroDuck_sim2sim_20260910_baselines.zip \
  -d results/research_20260910/references
export MICRODUCK_POLICIES="$PWD/results/research_20260910/references/baseline"
(cd "$MICRODUCK_POLICIES" && sha256sum -c "$SIM2SIM_ROOT/artifacts/research_20260910_reference_policies.sha256")
```

该配套包包含 18 个原始 ONNX；不包含可选的历史 local-ppo checkpoint，相关测试会明确跳过。
也可以提供自己的 factory 目录进行新实验，但不同权重的结果不能冒充本轮历史对照。

使用新的目录创建时间预算与 factory 快照，不延长或重写上一轮 `session.json`：

```bash
export SIM2SIM_RESEARCH_DIR="$PWD/results/research_local"
.venv/bin/python -m sim2sim.research.setup init \
  --out "$SIM2SIM_RESEARCH_DIR" --policies "$MICRODUCK_POLICIES" --hours 8
.venv/bin/python -m unittest discover -s tests -v
```

目录已存在时 `init` 拒绝覆盖。九个 factory 必须齐全，存在的旧 Godot 模型也会被快照；
需要完整 A/B 时加 `--require-previous` 强制要求九个旧模型。`SIM2SIM_RESEARCH_DIR`
必须在启动 Python 前设置；未设置时保留原有 `results/research_20260910` 兼容路径。
默认路径中的旧预算已经结束，不应用来开始下一轮训练。

测试包括原生 Godot 和真实 ONNX 检查；需要上述依赖与场景。
少量历史 rsl_rl checkpoint 测试在缺少其独立 checkpoint／local-ppo ONNX 时会明确跳过，
可用 `SIM2SIM_TEST_CHECKPOINT` 指定配对的 checkpoint。
跳过数量必须与通过数量一起报告，不能把缺资源的运行称为全部集成测试通过。

短训练闭环示例（新目录中两次迭代，不是复现原实验的 66 份训练记录）：

```bash
.venv/bin/python -m sim2sim.research.train \
  --skill standing --name smoke_standing --variant residual \
  --iterations 2 --minutes 2 --envs 2 --steps 32 --epochs 1 --minibatch 64 \
  --eval-seeds 1 --eval-seed-start 40000 --eval-entry both
```

`--variant anchor` 额外需要教师轨迹，先运行：

```bash
.venv/bin/python -m sim2sim.research.setup references --skill standing --seeds 3
```

该步骤使用真实 MuJoCo 采样并保留完整回合，供 anchor 的离线观测约束使用。
`plain`、`residual` 不需要预先生成这份数据。带中途翻滚课程或动作参考的实验会另行
采集 source 轨迹。历史蒸馏脚本中专用的父 checkpoint 和中间数据仍需原实验归档；
初始化新目录不会凭空恢复这些文件，也不保证重新训练得到逐字节相同的 checkpoint。

## 在新目录复测候选

单技能和连续操作可以直接复测，无需运行全部旧模型 A/B：

```bash
.venv/bin/python -m sim2sim.research.evaluate \
  --skill kick_right --onnx "$BUNDLE/models/KickRight_Godot.onnx" \
  --scene-robot microduck_ball_stand_fix --entry both \
  --seed-start 5000 --seeds 3 --out "$SIM2SIM_RESEARCH_DIR/recheck_right"
.venv/bin/python -m sim2sim.research.bundle "$BUNDLE" \
  --write-bank "$SIM2SIM_RESEARCH_DIR/recheck_bank.json"
.venv/bin/python -m sim2sim.research.play_sequence \
  "$SIM2SIM_RESEARCH_DIR/recheck_bank.json" --seed 5000 \
  --out "$SIM2SIM_RESEARCH_DIR/recheck_sequence.json"
```

具备完整 18 个原版／旧模型快照时，可以运行全部固定协议：

```bash
.venv/bin/python -m sim2sim.research.holdout "$BUNDLE" \
  --out "$SIM2SIM_RESEARCH_DIR/recheck_all" --seed-start 5000 --seeds 3 --workers 4
```

完整复测输出目录必须不存在；候选包原有 `holdout/` 保持封存。
1000–1029、2000–2029、3000–3029 已在本轮观察过，不再是下一轮未见测试集。
以上 5000 起的示例在用于调试后同样成为已知样本；真正选模前应另行登记开发和最终种子。

压缩包包含模型、摘要、配置索引与演示，完整训练权重、父模型链及逐回合数组仍在本地
实验归档。下一轮复现实验入口与复测候选，并不等于已完整复刻本轮全部训练历史。
