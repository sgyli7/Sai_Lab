# 复核与继续研究

本轮四候选均未通过，不安装到默认 MD。先读 [RESULT.md](RESULT.md) 与[论文 / 源码研究](../research_forward_gait_20260914.md)。不要因训练目录存在而重复启动，需同时检查完成标记和监督器终态。

## 本机证据

- 原始会话：`/home/ethan/Projects/MicroDuck/sim2sim/results/forward_gait_20260914/`。
- 本轮集成工作树：`/home/ethan/Projects/MicroDuck-SpeedControls/`。
- CPU Python：`/home/ethan/Projects/MicroDuck/sim2sim/.venv/bin/python`。
- CUDA Python：`/home/ethan/Projects/microduck_rl/.venv/bin/python`，GB10、PyTorch 2.9.1 + CUDA 12.9。
- Godot：`/home/ethan/.local/bin/godot`，4.7.2 / Jolt 5.5，原生 ORT 1.29.0。所有物理执行均 200 Hz，决策 50 Hz。

`training_completed.json` 确认四个固定终点，每个 8388608 条 transition；`evaluation_completed.json` 确认 5 组 GPU、6 组原生回放。各原生 `suite/summary.json` 指向逐回合完整 `trace.json`，GPU 的 `trajectories.npz` 包含状态、动作、观测、请求、技能选择和足部监测。`outcome_audit.json` 为真实原生动作抽样与失败分类。

训练权重、原始轨迹与运行时压缩基底未上传 Git；因此全新机器不能仅凭本次小型提交原样恢复旧 checkpoint。Git 中的 `evidence/` 包含协议、机器汇总、固定配置、模型与输入 SHA、最终指标和监督记录，`harness/` 保存实际编排脚本。完整本机会话的哈希清单保存在 closeout。获取对应本地资产后再运行新实验；缺文件应失败，不能改用其他模型冒充原始起点。

## 离线重新汇总

下面只读取已保存轨迹，不启动游戏或训练：

```bash
cd /home/ethan/Projects/MicroDuck/sim2sim
env PYTHONPATH=/home/ethan/Projects/MicroDuck-SpeedControls/src \
  .venv/bin/python /home/ethan/Projects/MicroDuck-SpeedControls/scripts/summarize_forward_gait.py \
  results/forward_gait_20260914
```

它重新生成 `analysis.json` 和图表，并核对两域控制配置、132000 帧请求和技能选择。`audit_outcomes.py` 另用真实原生观测做动作抽样，需原 ONNX 和 CPU ORT。统计有明确边界：净位移基于躯干原点，不是全身重心；足步长由踝部惯性中心的同足落地距离 / 2 估计；同时双脚落地不计为交替，窗口内跌倒不生成有效稳态汇总。

## 恢复原生冻结运行时

13 份临时运行时目录归档后移除。`baseline_runtime.tar.gz` 保存完整基底但排除 `.godot` 缓存；每个 `native_LABEL/runtime_overlay/` 保存相对基底的模型、manifest、部署配置和 fixture 差异。压缩包及覆盖文件 SHA 见 `closeout/runtime_archive.json`。

恢复到一个**新的、独占**目录：展开基底，覆盖所选 overlay，然后用 `sim2sim.standalone.suite.runtime_inputs` 校验该组 `suite/runtime_snapshot.json` 的 `input_sha256`。如需导入资源，重建 `.godot` 缓存，再做原生自检。不要向桌面维修站的运行时复制实验权重。

实际评测命令见 `harness/run_evaluation.py`，其中绝对路径需映射到新目录；原始 GPU 初态与按键程序来自旧 `sprint_input_diagnosis_20260914/ceiling_gate/baseline/suite/summary.json`。只取其零速度初态与按键，显式使用本轮 control 的 0.45 上限，不继承旧 0.30 速度；新原生输入与它逐帧核验。完整状态投影仍不包含全部求解器内部状态。

## 精确训练来源和续训

四组正式训练使用的完整核心源码在本地 `frozen_code/`，其 SHA 见 `training_code_sha256.json`；Git 中的 `frozen_training_code/` 只保存相对基准提交改变的源码覆盖文件，其余捕获文件均与 `b512981` 相同。把覆盖文件应用到该提交的新 checkout，保持同一相对目录结构，再核对全部捕获文件、依赖与原始 actor、critic、教师 replay 哈希；不能直接在部分覆盖目录内运行。编排见 `harness/run_training.py` / `evidence/training_base_args.json`，不要直接运行其指向已存在结果的输出路径。

正式训练冻结后修复了暖启动续训来源记录与空样本统计，因此当前提交的训练脚本 SHA 不等于四组旧训练脚本 SHA。不能在旧实验下悄悄换源码续训；若要继续，应建新会话、登记来源。当前版本已通过 `warm_resume_start` 的 5 次迭代与 `warm_resume_finish` 的第 6 次迭代检查，新的物理回合从完整 checkpoint 开始，不声称恢复求解器的精确中间接触状态。

首次错误教师范围方案、独立 Warp 世界位级一致性检查、归档字典范围不一致的失败均保留；详见结果第 4 节。协议命名残留有独立 errata，原始文件未事后重写。

## 资源与下一轮边界

新训练 / 评测必须通过 `sim2sim.research.budget.start_supervised` 启动，指定新会话、有限 timeout 和收尾预留。CUDA 训练最多 4 核 CPU 辅助，原生评测 2 worker / 2 核，nice 10；正式评测期间无竞争训练。采样完成立即关掉所属 Godot，按 PID 起始时间 / 自建进程组清理，不能批量误杀其他人的任务。

新 SAC 路线只是研究建议，尚未实现 replay / Q 学习栈。先补正确的 transition、终止 / 超时和跨技能 bootstrap，再做固定预算对照。继续使用开发种子，最终 `929200–929229` 不用于调参；不得将本轮四个失败模型换成默认，或把研究成功收尾称为跑步质量达标。
