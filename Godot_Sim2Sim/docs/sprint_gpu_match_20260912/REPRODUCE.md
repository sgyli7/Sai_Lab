# 复算、恢复与依赖

工作目录 `/home/ethan/Projects/MicroDuck/sim2sim`。本轮已经封存，不要因为训练目录存在就重新启动。先读取 `RESULT.json`、`results/sprint_gpu_match_20260912/PHASE.md` 和 `active_budget.json`，核对每个任务的完成或中断记录。

## 本机输入

- S05：`results/sprint_20260912/runs/s05_native_handoff/final.onnx`，SHA `27ebbf83d63e5e59f125ceb158414cf7d2574990676d73c75b9c9870bcefed6e`。
- 归一化模板：`results/sprint_20260912/gpu_proxy/jolt_torque_512/final.onnx`；S05已是组合图，不能直接按单支MLP解析。
- 控制：`results/sprint_joint_20260912/delivery/control.json`。
- 冻结原生基底：`results/sprint_exit_20260912/state_contract/suite/runtime`。本轮每个 `native_*/suite/runtime` 保留其真实模型和完整代码快照。
- GPU依赖环境：`/home/ethan/Projects/microduck_rl/.venv/bin/python`；环境版本见 `evidence/environment.json`。该外部仓库只作只读依赖，没有改动。普通 `.venv` 用于原生Godot评测、CPU数值验证和报告。

大模型、checkpoint、原始trace/NPZ、完整runtime在本机results内，没有伪称已包含于Git。异机复现必须先恢复这些输入以及MJCF/mesh依赖。`ARCHIVE_SHA256.json` 校验Git内的文本证据和图表，`evidence/candidate_freeze.json` 校验实际模型。

## 冻结实验

`harness/gpu_pair_graph_protocol.json` 和 `harness/cat_pair_protocol.json` 保留完整命令。四个正式终点均128×64×512、训练种子951101、只按固定终点评测；前者只比较脚底，后者只比较约束概率。每个训练目录的 `code/` 是当时源文件，逐项校验其 `config.json` 中的code SHA。不要拿后续CaT版本的训练代码冒充先前图调度实验。

`train_source` 是raw调度下主动中断的第16轮，不是最终候选；`run_gpu_pair.py` 已过时。Graph重启另用 `train_source_graph` 与 `train_jolt_graph`。旧响应/控制探针要使用当时的world快照：Graph迁移后增加了跨Torch/Warp流的显式同步，旧脚本直接 `mw.forward` 的调用不应与新world混用。旧反例和失败尝试完整保留，不以当前代码重跑覆盖。

CaT冒烟、续训、正值奖励控制、整数时钟回归均有独立日志。约束仅折扣学习回报，不增加物理复位；`--constraints positive` 和 `cat` 两臂具有相同计算路径及非负奖励前提。

## 可直接复算的只读证据

原始路径仍存在时，可以运行：

```bash
.venv/bin/python results/sprint_gpu_match_20260912/native_comparison.py
.venv/bin/python results/sprint_gpu_match_20260912/audit_resources.py
```

第一项复算开发集成功/失败与同种子回退，第二项只核对登记的进程组/启动时间和容器挂载所有权。不要将其他任务的Godot、RobotDesign、Sai或workshop进程当成本轮残留清掉。

`gpu_native_starts.py` 把原生t=0、零速度的姿态投影到GPU广义坐标，保留相同按键，再用原评分器检查；它不能把任意接触状态解释为完全等价。`handoff_audit.py` 对六条新失败核对当时实际运行技能及Python影子结果。回放材料足够复算，不必为了阅读结论重跑训练。

## 新实验

必须使用新的会话、输出目录与明确预算。开发925000–925015已用；最终926000–926049、前轮924000–924049仍未用，不能拿来继续调参。示例结构（需填写新路径，不是覆盖本轮的命令）：

```text
.venv/bin/python -m sim2sim.research.budget --directory NEW_SESSION start
  --timeout SECONDS --reserve 1800 --label NAME --cpus 4 --nice 10 --
  env PYTHONPATH=/home/ethan/Projects/MicroDuck/sim2sim/src
  /home/ethan/Projects/microduck_rl/.venv/bin/python scripts/sprint_gpu_train.py
  --output NEW_SESSION/RUN --feet jolt --cuda-graphs --constraints cat
  --iterations 128 --envs 512 --steps 64 --seed NEW_TRAIN_SEED
```

GPU不可用时明确失败，不回退CPU。禁止同时进行争用资源的训练和正式性能测量；结束后检查监督器终态与资源清理。五份已删除staging的canonical源及恢复命令在 `evidence/closeout/prepared_cleanup.json`，只有逐文件相同的副本才被删除。
