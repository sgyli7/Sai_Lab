# 复现与恢复

先读 [RESULT.md](RESULT.md)。本目录为封存研究证据，不是生产模型发布。大模型、checkpoint 和原始轨迹保留在本机 `results/sprint_joint_gpu_20260912/`；路径、大小、SHA256 见 `evidence/local_artifacts.json`。Git 包含配置、代码快照、课程和精简逐用例指标，不能凭空重建缺失的源权重或运行资源。

## 代码与运行环境

基准提交 `1c80048dc71b1c7ef608080b3409fc15c3e6a9e2`，实验分支 `codex/sprint-exit-20260912`。恢复到隔离工作树，不覆盖其他任务的工作区。按目标实验覆盖相应快照，并逐项核验 `config.json` 内的 `code_sha256`：

| 实验 | 代码快照 | 固定命令 |
|---|---|---|
| 分离/共用速度目标 | `harness/frozen_code/` | `evidence/training_protocol.json` |
| 共用命令跟踪 | `harness/tracking_code/` | `evidence/tracking_protocol.json` |
| 分离命令跟踪 | `harness/tracking_code/` | `evidence/tracking_split_protocol.json` |
| 补齐普通课程 | `harness/paired_code/` | `evidence/complete_courses_protocol.json` |
| 被拒绝转向闭环 | 另覆盖 `harness/feedback_code/` 的三份控制源码 | `evidence/feedback_protocol.json` |

生产源码已撤回最后一行控制原型；只拷贝反馈运行脚本而不覆盖隔离工程的控制源码，不能复现该实验。恢复旧 checkpoint 也不能使用不同代码哈希的当前训练脚本。

原机 Linux ARM64 / NVIDIA GB10。GPU Python 为 `/home/ethan/Projects/microduck_rl/.venv/bin/python`，PyTorch 2.9.1+cu129、MuJoCo 3.10、mujoco-warp 3.8.1、Warp 1.12、Python 3.12.14。外部 `microduck_rl` 是只读依赖；将环境中的 `PYTHONPATH` 指向隔离工程的 `src`，避免意外导入原工作树。训练图中 anchor 与 learner 在 CUDA；导出对照使用 CPU ORT 是数值检查，不代表 CPU 训练。游戏扩展使用 ORT 1.29.0 CPU。

固定输入包括：

- 初始 S05：`results/sprint_20260912/runs/s05_native_handoff/final.onnx`，SHA `27ebbf83d63e5e59f125ceb158414cf7d2574990676d73c75b9c9870bcefed6e`。
- 标准化模板：`results/sprint_20260912/gpu_proxy/jolt_torque_512/final.onnx`。
- 控制：`results/sprint_joint_20260912/delivery/control.json`；普通 0.25、加速 0.30、角速度 0.8，路径前视 0.20。
- 冻结原生运行工程：`results/sprint_exit_20260912/state_contract/suite/runtime`；模型银行：`results/research_20260911/delivery/candidate_models`。
- `robots/microduck_ball_stand_fix.json`、实际脚底凸包和惯量映射；哈希/数据见训练配置及 `evidence/closeout/defaults_and_physics.json`。
- 教师数据 `results/sprint_joint_gpu_20260912/teacher/observations.npz`；哈希与逐源来源见教师 manifest。

## 执行纪律

为新实验建独立目录和有效时长账本，不把已封存目录当成可继续写的输出。所有长任务必须由 `python -m sim2sim.research.budget --directory NEW_SESSION start --timeout SECONDS --reserve 1800 --cpus 4 --nice 10 -- COMMAND` 启动。根据实际剩余预算设置有界 timeout；收尾任务才使用 reserve 0。不要直接重复运行归档 orchestrator，它们固定写入旧目录。

`evidence/*protocol.json` 保存完整 argv，`evidence/active_budget.json` 保存实际监督命令及终态。先核验命令中的源文件和快照，只将实验输出、会话位置及工作树路径映射到新目录；输入资产保持只读。续训先确认上次任务已退出、检查 checkpoint/config/hash，从最后完整 checkpoint 开始新物理回合，保留旧失败尝试。路径存在不是成功证据。

## 原生与 GPU 对照

`harness/native_eval.py` 构造八种真实键盘课程与普通配对，另外读取旧六条失败及最小失败。263 条原生结果均须完成。`harness/compare.py` 额外核验课程哈希、模型范围、每类普通速度保留和基准成功损失；不能用其余不完整汇总代替。

`harness/gpu_native_starts_post.py` 与 `harness/post_score.py` 是正式跨域诊断入口：取同一原生初态投影和输入带，按动作后物理状态评分，必须有最终状态。`gpu_native_starts.py`、`matched_*` 是保留的旧动作前诊断，不作为本轮最终跨域数字。

停止定位用 `harness/stop_state_probe.py`，恢复 927008 的 9 秒前控制历史，比较 Jolt 轨迹继续段、Jolt 状态投到 GPU、GPU 状态冷恢复。完整轨迹在本地 manifest 索引中；投影不含接触缓存和全部非理想连杆状态，不能据此声称两个引擎的内部状态相同。

清理过的 `prepared` 目录可按 `evidence/closeout/prepared_cleanup.json` 的 restore argv 从保留的 `suite/runtime` 恢复。先逐文件确认 canonical 文件仍与记录一致；原始轨迹和失败尝试没有删除。

测试模式冻结于 `evidence/test_protocol.json`，运行器为 `harness/run_checks.py`。本轮 284 项中 282 通过、2 可选跳过。归档校验使用 `ARCHIVE_SHA256.json`；大文件另用 `evidence/local_artifacts.json`。最终种子 **928000–928049 未使用**，只有开发门全通过后才允许一次性最终评测。
