# 复现与后续校正入口

先读 [RESULT.md](RESULT.md)。封存会话 `results/sprint_stop_state_20260912/`，基准提交 `66608b329b3c86662a3657decdba0b60285c9c0e`。不要在旧会话中重复运行固定输出的脚本。新实验使用独立工作树、目录和有效时长账本；所有长任务经 `research.budget start` 限四核、nice 10、超时和退出清理。

## 输入与代码

训练源权重 S05、标准化模板、教师数据及固定控制配置沿用上一轮。逐项路径/哈希和完整命令见 `evidence/training_protocol.json`、`evidence/train_joint_fd/config.json`。本轮正式训练是 CUDA 采样与 CUDA 学习；GPU Python 为 `/home/ethan/Projects/microduck_rl/.venv/bin/python`，外部项目只读。新工作树须将 `PYTHONPATH` 指向其 `src`，不能误导入原工作树。原生工具使用本仓 `.venv` 和 Godot 4.7.2 / ORT 1.29.0。

`harness/training_code/` 保存正式训练的精确代码；续训前覆盖到隔离工作树并核验配置中的全部 `code_sha256`。`harness/full_observer_prototype/` 是被拒绝的完整角速度版本，不能与正式 `joint_fd` 混用。当前生产默认仍是 `solver`。

`harness/runtime_patches/` 保存各个已清理工程的独有脚本。按 `evidence/closeout/runtime_cleanup.json` 先复制 canonical 工程，再覆盖对应补丁，才可恢复原来的诊断/探索环境。普通验收工程不应用这些补丁。原始大文件路径与 SHA256 在 `evidence/local_artifacts.json`，没有随 Git 提交权重或轨迹正文。

## 执行链

1. `reproduce.py` 用冻结的上轮最佳模型复现 927008；只有正确的原生停止失败才视为红灯。先修正新会话路径，保留旧失败尝试。
2. `build_probe.py` 与 `run_probe.py` 捕获 9 秒前状态，对比连续运行和两次冷恢复。`roundtrip_probe.py` 是同进程诊断，不可作为玩法修复；严格检查恢复瞬间 61 维输入、控制历史和动作。
3. `validate_native_observer.py` 的完整角速度门槛本轮应失败，不能改成通过。`check_observer_projection.py` 记录其误差；`validate_joint_contract.py` 对已采集的 200 Hz 轨迹验证关节观测与实际 PD 力矩。
4. `run_joint_fd_training.py` 使用冻结命令，先 8 轮 smoke 再正式 256 轮。禁止改变源权重、种子、教师或预算后仍声称是同一对照。导出误差必须小于 1e-5。
5. `native_compare.py` 使用旧轮 `native_eval.py` 的完整 263 条原生课程，再调用旧轮 `compare.py` 做配对。旧工具是只读依赖；输出必须改到新会话。`gpu_compare.py` 是诊断，不能替代原生判据。
6. `target_sampler_probe.py` 构造隔离探索工程，先无噪声单例再四核批量。原始失败输出在 `target_sampler/`，成功输出在 `target_sampler_retry/`。必须确认 `completed.json` 和监督器终态后才能运行 `validate_target_samples.py`；后者在 CUDA 上重建均值、分布和采样对数概率，检查评分器拒绝探索数据。

完整 argv、返回码和超时在 `evidence/active_budget.json`；日志在 `evidence/supervisors/`。恢复工作必须另建物理回合，不能把冷恢复当作精确接触状态重放。

## 交给下一轮的训练数据

`results/sprint_stop_state_20260912/target_sampler_retry/samples.npz`：`obs`、实际 `action`、行为 `mean`、`old_logprob`、`sprint`、`actor_mask`、`episode`、`time`。对应完整原生轨迹和课程在 `target_sampler_retry/completed.json` 的 rows 中，顺序就是 episode 索引。噪声种子逐条记录，普通槽始终 S05，加速槽均值为本轮 a402791 模型。两条无噪声对照不进入随机策略 PPO 采样集合；据 rows 中的 sigma 区分整个 episode，不能仅以加速标志区分。

尚未提供后状态奖励、约束回报、critic 值或新的策略更新。下一轮需从原始轨迹按动作后状态重建，包含最终状态并在最后一帧终止；不要把旧 `old_logprob` 用于不同权重、不同方差或不同动作变换。只更新所声明的随机加速样本，普通段用于任务回报与冻结控制对照。使用 GPU 学习，保留教师和原生全量门禁；这些是待实现的工作，不是本轮已验证的训练成果。

最终种子 **928000–928049 未用**。候选尚未满足开发门，不能开始最终选模或重打包。
