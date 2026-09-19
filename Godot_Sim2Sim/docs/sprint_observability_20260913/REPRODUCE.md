# 复现信息诊断和同动作 GPU 回放

先读 [RESULT.md](RESULT.md)。本轮没有合格的新动作策略或安装包。会话为 `results/sprint_observability_20260913/`；恢复前读取账本和任务终态，不重复运行已完成的输出目录。

## 版本与数据

40 epochs 实验使用 `harness/predictor_v1/` 下的源码快照，协议在 `evidence/predictor/protocol.json`。640 epochs 实验使用提交中的 `scripts/sprint_observability.py` 和 `src/sim2sim/research/observability.py`，精确副本也保存在 `harness/predictor_v2/`；协议在 `evidence/predictor_long/protocol.json`。第二版增加了无新信息对照的去留条件，不能拿它假装重现第一版原始初筛。

从 `32097eb` 创建隔离 checkout，覆盖对应完整快照，核对协议中的源码哈希。输入来自之前冻结的 `sprint_target_gpu_20260912` 三组训练回合及六组原生候选，以及 `sprint_stop_state_20260912/native_joint_fd`；逐文件、逐回合来源在 `evidence/dataset.json.gz`。本地原始结果和 Python 环境不随 Git 发布；缺失时应明确列出，不重新生成数据并沿用旧哈希。路径迁移须记录映射，保留封存证据原样。

## 运行顺序

新建独立有效工时会话和输出目录。所有长命令经 `python -m sim2sim.research.budget --directory <session> start` 启动，明确超时、至少 1200 秒收尾预留、`--cpus 4 --nice 10`。实际命令和终态均在 `evidence/active_budget.json`。

1. `harness/reproduce.py` 驱动五条真实 Jolt 回放；实际打分失败是需要保留的诊断结果，删去前缀的通过不算策略改善。
2. `scripts/sprint_observability.py freeze --output <new-output>` 写入协议和源码哈希；第二版延长对照另加 `--epochs 640`。第一版请使用原始快照，固定 40 epochs。
3. `... extract --output <new-output>` 从已有轨迹提取因果特征，四个 CPU 工作进程；完整种子隔离由脚本强制检查。第二组复用第一组不可变的 `data/`，并复制对应 `dataset.json`，不要重新抽取或随机切分相邻行。
4. `... learn --output <new-output>` 使用 GPU Python `/home/ethan/Projects/microduck_rl/.venv/bin/python`，设置 `PYTHONPATH=<checkout>/src`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=4`。CUDA 不可用会失败。两个种子、五种输入、固定终点均完成后才进行测试统计；不存在按验证集挑检查点或通用续训入口。
5. `... analyze --output <new-output>` 计算完整开发测试、移动和命令变化窗口的指标；`harness/targeted_analysis.py` 另做已知失败段的事后统计，不能用来继续挑本轮终点。
6. `harness/action_tape_probe.py` 从 16 个原生零速度初态开始，按完全相同的实际动作运行两种 CUDA 速度路径。验证初始观测、实际控制目标、上一动作，再比较物理状态；它不使用策略网络，不替代原生游戏验收。
7. `harness/run_checks.py` 按 `evidence/test_protocol.json` 的清单回归。必须移除 `SIM2SIM_RESEARCH_DIR`，避免新研究目录覆盖测试原本使用的封存 baseline；保留 `SIM2SIM_ACTIVE_BUDGET_DIR` 监督。

第一版和第二版各自保存模型及预测结果。20 个研究模型只预测三个物理变化量，不能放进 61→14 的游戏推理槽。两次实验复用了开发测试集，最终 928000–928049 从未进入这些脚本。

## 退出和核验

完整源码、协议、分析和学习曲线存档在本目录，原始 NumPy 数据、预测器权重和轨迹保留本地。核对 `evidence/local_artifacts.json.gz`、默认模型／物理哈希和账本 job_finished；目录存在不等于成功。失败测试尝试单独保存。

停止并清理本会话拥有的子进程，确认没有残留 Godot 或学习进程，再停止 agent 心跳。不得终止其他项目的机器人查看器、Sai／workshop、远程桌面或用户应用。当前工作树有其他任务修改，提交仅显式列出的本轮文件，不推送远端。
