# 复现接触与关节约束标定

先读 [RESULT.md](RESULT.md)。本轮不提供晋级策略。账本封存在 `evidence/active_budget.json`；恢复须新建会话，不因输出目录存在就当实验完成，也不覆盖失败尝试。

## 版本与输入

从 `60ea7d7f8e46b43bcc0384ab49bdf59ad99f8e06` 创建隔离 checkout。接触试验使用 `harness/contact_base/scripts/sprint_gpu_world.py`；正格式连接试验使用 `harness/compliance_probe/` 完整源码；负格式试验使用 `harness/compliance_direct_probe/`。当前模块后来只整理了格式，原始精确版本及协议哈希均已存档。失败接触试验分别在 `harness/contact_probe_attempt_01/`、`..._02/`。

输入为 `results/sprint_stop_state_20260912/native_joint_fd/suite/summary.json` 及其 16 条 `sprint_alternate` 原生轨迹；路径、种子和 SHA256 见 `evidence/manifold_probe.json`。依赖机器人 JSON、原始 MJCF、Jolt 脚资源和 `results/sprint_joint_20260912/delivery/control.json`，均沿用起点。原生轨迹与 Python 环境不随 Git 分发；缺失时明确报错，不能重新模拟后沿用旧哈希。最终 928000–928049 不进入此流程。

GPU 环境为 `/home/ethan/Projects/microduck_rl/.venv/bin/python`，本轮 Torch 2.9.1+cu129、MuJoCo 3.10、MuJoCo Warp 3.8.1、Warp 1.12，GB10。CPU测试使用项目 `.venv/bin/python`。设置 `PYTHONPATH=<checkout>/src`、`OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=4`。精确监督命令、超时、退出状态见封存账本；所有长命令经有效时长监督器，四核、nice 10、至少 1200 秒收尾保留。

## 步骤

1. 新建研究会话和输出目录。这里的诊断脚本故意固定原会话路径，并检查 `SIM2SIM_ACTIVE_BUDGET_DIR`；复现时将 harness 中该路径整体映射到新会话，同时保存改后的源码哈希和映射说明，不原地修改封存目录。
2. 接触原始尝试若需审计，分别使用对应失败源码；正常重复诊断由 `parity_probe.py`、`parity_probe_strong.py` 完成。正式四配置使用最终 `contact_probe.py`。第一次失败发生在采样前，第二次只完成三个配置；不能与最终结果拼成一轮完整试验。
3. `manifold_probe.py` 读取同帧原生姿态，仅做正向运动学，生成 `manifold_probe.json` 和 `manifold_residuals.npz`。后两组连接试验都依赖该文件。
4. `compliance_probe.py` 使用正格式快照的 `GpuWorld` 和模块，完成五个固定配置。切换至负格式完整快照后，`compliance_direct_probe.py` 完成另外五个固定配置。两组均使用最终 `contact_probe.initialize`，不推理新策略。
5. `component_analysis.py` 只分析已冻结结果，标记为事后定位；不得将它当作预注册门禁或继续筛选本轮参数。
6. `run_checks.py` 根据 `evidence/test_protocol.json` 执行原有回归加三项新测试。它在导入项目前移除 `SIM2SIM_RESEARCH_DIR`，避免覆盖测试的旧 baseline，同时保留有效时长监督标记。
7. `audit_resources.py` 检查进程开始时间、进程组、会话环境标记与监督终态，只清理自己拥有的任务。其他 Sai／workshop、用户应用和远程桌面不属于本轮。

本轮输出约 12.6 MB 的 18 个 NumPy 文件，清单在 `evidence/local_artifacts.json`。正式 JSON、源码、失败尝试和原始日志的 gzip 副本入库。去留结果为三个完整试验的 `eligible` 均空；没有续训模型可加载，也没有新原生性能成绩或安装包。
