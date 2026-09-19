# 固定 Jolt 试验：复现与恢复

本轮从已保存的反馈候选出发，试验目录为 `results/jolt_learning_20260911/`。它不替换默认模型、上一轮发布包或上一轮保留集结果。以下命令从 `sim2sim` 仓库根目录执行。

## 环境与冻结材料

本机为 Ubuntu 24.04 ARM64、GB10、Python 3.12.14、Godot 4.7.2、PyTorch 2.9.1+cu129、ONNX 1.22.0、ORT 1.29.0。本轮 R0–R3 的冻结训练器使用 CPU 上的演员／critic 和真实 Jolt 采样；后续工程修复新增 CUDA 学习路径，其 smoke 和计时不混入这 27 个质量终点，见 [GPU 与资源](GPU_AND_RESOURCES.md)。离线源域参照使用本机 MuJoCo 3.12.0；它不进入游戏运行时。

`reproduction_environment.json` 保存实际版本、源码快照与脚本 SHA。模型来源与实际训练命令分别记录在各次 `config.json`、系列 recipe 和原子完成记录中。物理脚本、生成机器人、机器人配置及独立评分器相对 `a020052` 未改动；200 Hz 物理、50 Hz 决策。

| 路径（相对试验目录） | 用途 |
|---|---|
| `baseline_candidate/summary.json` | 184 条九技能开发基线及每条原始轨迹路径 |
| `development_cases/`、`development_case_index.json` | 固定命令、初态和 SHA；开发种子 917000–917007 |
| `task_anchor.onnx`、`task_initial.onnx` | 扩展到 68D 的原锚点与零增量策略 |
| `code_r0/src`、`code_r1/src`、`code_r2/src`、`code_r3/src` | 各系列实际读取的 Python 源码快照 |
| `runtime_r0/`、`runtime_r1/` | 原生验收运行时快照；R2、R3 复用 R1 游戏运行时 |
| `runs/<name>/` | 配置、更新日志、完整 checkpoint、ONNX 和导出数值检查 |
| `candidate_evaluations/<name>/` | 独立候选模型、准备工程、72 条轮滑回放及严格配对结果 |
| `teacher_replay_complete.json`、`teacher_replay.npz` | 只含成功教师回放的制动观测，含来源／内容 SHA |
| `teacher_training_suite/` | 112 条教师回合，包含未收录的失败尝试 |
| `sampling_gap_traces/` | 均值／探索噪声诊断的原始物理轨迹 |
| `active_budget.json` | 工作区间并集、活动进程与开始／完成事件 |

文档内的 `TEACHER_REPLAY.json` 是审计副本。训练读取试验目录的原始 manifest；其中相对路径 `teacher_replay.npz` 相对于该 manifest 所在目录解析。不能将文档副本当成数据文件已搬到文档目录。

原始轨迹、运行时、模型、checkpoint 和快照保留在本机忽略目录，没有全部放入 Git。完整迁移须复制该试验目录及上一轮 `results/research_20260911/delivery/candidate_models/`、`final_candidate/control.json`；只克隆代码仓不会获得这些权重与证据。绝对路径出现在 recipe 和 summary 中，换机器需明确重定位并保存转换记录，不能冒称字节级重跑原配置。

## 查看结果，不重启试验

```bash
.venv/bin/python results/jolt_learning_20260911/summarize_pilot.py
.venv/bin/python -m sim2sim.research.budget \
  --directory results/jolt_learning_20260911 status
```

聚合只读取 `completed: true` 的原生回放结果；训练奖励、目录存在和通用 `brake` 内评都不构成晋级证据。每个系列须有 `*_completed.json` 才表示全部计划终点完成。`*_progress.json` 可用于检查已完成成员，不能把未出现的成员默认视作失败或成功。

R0、R1、R2 的早期快照在完整跑满 64 次更新后仍写 `status=time_limit`；应同时核验 `iterations=64`、`samples=262144` 和导出检查。此状态语义已修正，后续记录为 `iteration_limit`。R1／R2 的旧候选 sidecar 可能仍继承 `obs_len=61`，实际 ONNX 和 `deployment.json` 均为 68D，原生回放读取实际契约；当前 `candidate.stage` 已同步 sidecar 字段。

## 复测一个已完成候选

以 R1 完整状态、种子 71 的 1M 终点为例。输出必须是不存在的新目录：

```bash
.venv/bin/python -m sim2sim.standalone.candidate evaluate \
  --incumbent results/research_20260911/delivery/candidate_models \
  --skill roller \
  --model results/jolt_learning_20260911/runs/r1_full_s71_1m/final.onnx \
  --cases results/jolt_learning_20260911/development_cases \
  --baseline results/jolt_learning_20260911/baseline_candidate/summary.json \
  --out results/jolt_learning_recheck_s71 \
  --workers 8
```

该命令只评轮滑；本轮行走控制配置不影响这些轮滑回放。系列驱动脚本还明确传入旧 `control.json` 与冻结 `runtime_r1`，完整逐字复现以保存的系列脚本为准。质量评测在同对训练均停止后进行；性能延迟测量另须停止竞争资源的训练，不能把并行采样时的耗时当成正式推理性能。

## 恢复与重新训练

先检查 `active_budget.json` 的活动进程、PID 起始时间及相应日志，避免启动重复任务。旧会话完成后不直接重跑系列脚本：它们使用 `exist_ok=False` 保护已有目录，而且不会按目录存在自动恢复。

重训应另建试验目录，复制冻结来源、recipe、数据及所需源码快照，并将 `SIM2SIM_RESEARCH_DIR`、`SIM2SIM_ROOT`、`PYTHONPATH` 指向会话、仓库根目录和 Python 源码快照。新长任务通过 `research.budget start` 启动独立监督（前台 `run` 也可用），默认四核和 nice 10，保存新名字、新账本和有限超时；原始记录保持不变。实际训练进程从 `SIM2SIM_ROOT/godot` 启动物理服务，原生验收另读 `runtime_r0`／`runtime_r1`；不能只还原 Python 代码而允许根目录物理实现变化。训练配置内 `physics` 指纹记录实际机器人与物理版本。

恢复训练必须使用最后完整 `latest.pt`，保留演员、critic、优化器、随机数和环境回合计数，并从新物理回合开始。不能换目标、任务掩码、动作权限、教师模式或数据后还标记为同一次续训。R1／R2 追加 192 次更新，计为 786,432 个新样本、累计 1,048,576；R3 为 64+192 两段，同样累计 1,048,576，不能重复累加前缀样本。

## 独立工程包检查

`contract_export_package/` 是零增量 68D 工程包。`contract_export_clean/` 保存 Ubuntu 24.04 ARM64、禁网、无 Python／训练仓库的容器环境记录和 23 条回放。动作与旧包逐条完全一致；它不表示学习改善或最终模型通过。

本次容器核验使用本机缓存的 `ubuntu:24.04`，实际观测到的镜像摘要为 `ubuntu@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254`；复测应固定该摘要。随机／边界九模型自检与真实观测 shadow 是分别保存的证据。`contract_switch_shadow.json` 另覆盖复位、机器人切换及蹲起。

宿主锁屏期间不执行全局键盘注入，因此这组检查不称作本轮真实键盘验收。若后续候选达到晋级条件，必须冻结后重新做完整包、真实输入与新保留集验收。

## 冻结候选与 release 文件

开发集唯一满足改善和无回退条件的终点为 `r3_replay_kl_s73_1m`，冻结记录见 `FINAL_FREEZE.json`。新保留集 918000–918029 在冻结后使用；验收执行期间不再选择参数。

首次 `final_validation/` 因 `runtime_r1` 里的 release 扩展仍是旧 61D 版本，在模型加载自检时停止，尚未生成或使用保留集。该快照的 debug 扩展已支持 68D，所以开发回放正常；两个二进制不能互相代替验证。`release_extension_recovery.json` 记录 SHA 与重试原因。`final_validation_retry/` 仅将验收工程的 release 文件换成此前 68D 工程包验证过的版本，模型、控制和物理未改，原快照及失败目录保持原样。打包器现强制执行实际导出文件的九模型自检。

复位／换机器人命令带只作控制契约检查，不能送进任务成功评分器。实际第二次驱动误将它交给严格任务评分器，评分器正确拒绝带复位的轨迹；保留该失败并使用同一次成功的原生执行轨迹做 Python shadow，没有重跑物理、放宽评分或把复位算成任务成功。`finish_frozen_validation.py` 记录恢复步骤，`ENGINEERING_SWITCH_RECOVERY.json` 保存原因。`reproduction/` 同时保留两次实际驱动与修正后的后续复现版本，不能互称同一个脚本。
