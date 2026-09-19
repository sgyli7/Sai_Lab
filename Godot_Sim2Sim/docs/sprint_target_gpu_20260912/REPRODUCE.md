# 复现本轮目标引擎校正

先读 [RESULT.md](RESULT.md)。原始数据在 `results/sprint_target_gpu_20260912/`；该会话封存后不能重复启动同名输出。所有长任务使用有效工时监督器，四核、nice 10、显式超时。GPU Python 为 `/home/ethan/Projects/microduck_rl/.venv/bin/python`，`PYTHONPATH` 指向本仓 `src`；原生评分使用本仓 `.venv`。不要改动其他任务的 Sai／workshop 工作。

## 精确版本

本轮有三个实现快照，不能用当前文件假装复现旧实验：

| 训练输出 | 源码快照 | 协议 |
|---|---|---|
| `train` | `harness/training_code` | `evidence/protocol.json` |
| `train_warm` | `harness/warm_prefix_code` | `evidence/warm_prefix_protocol.json` |
| `train_initial_teacher` | `harness/initial_teacher_code` | `evidence/initial_teacher_protocol.json` |

使用基准提交 `b20a33d` 的隔离 checkout，覆盖对应快照中的全部文件，再核验协议的 `source_code_sha256`。补齐本地模型、Python 环境及已核验的 canonical Godot runtime；固定绝对路径如需迁移，另记清晰映射，不修改封存证据。冻结普通源、模板和 a402791 的路径见训练 `config.json`；最后一组教师是 S05 公共 anchor 加冻结的 a402791 增量，不能将审计里的公共 anchor SHA 误认成完整教师。

所有实际命令、监督器 PID／启动时间、超时和返回码在 `evidence/active_budget.json`。每组 12 批、四个复位种子、seed 955101，固定导出第 6／12 批；只有第三组传入 `--retention-reference initial`，后两组使用 `--curriculum warm_prefix`。恢复后须新建目录及物理回合，保留旧失败尝试，不依据目录存在判断完成。

## 核验顺序

1. `sprint_native_correct.py --audit-only` 在既有的 16 条随机轨迹上重建动作后奖励、控制状态、critic 输入和有限任务终止，验证初始均值与 native a402791 一致。两条无噪声全程对照不作为随机 PPO 数据。
2. `native_evaluate.py`、`native_warm_evaluate.py`、`native_initial_teacher_evaluate.py` 调用封存的 263 用例评测器，分别比较 S05、上一轮最佳和本轮起点。不能修改评分阈值、模型的普通槽或物理后再沿用同一实验名。
3. `coverage_audit.py` 是训练轨迹的诊断统计。`noise_probe.py` 固定起点与三种初态，比较 0／0.005／0.02 动作噪声，所有输出依然标记探索数据。`prefix_probe.py` 检查前 7 秒及探索入口与确定性回放完全一致，随后验证随机段似然和前段屏蔽。按实际时间顺序选用对应源码快照，勿混淆后来增加的前段功能。
4. `teacher_pair_audit.py` 检查两组在首次 actor 更新前的 96 条轨迹一致，排除不同初态／控制作为教师对照的混杂因素。
5. `run_checks.py` 的用例清单在 `test_protocol.json`，包括动作后奖励、终止、跌倒后帧、确定性前段排除及教师冻结的回归。

采样通过独立原生进程完整回放。`native_sprint.dataset` 的 `valid` 用于 critic，`mask` 只包含有噪声的加速动作，`sprint` 单独保留真实技能选择；不要把确定性前段误当成普通 S05 动作，也不要给其赋高斯行为概率。`done` 包含物理跌倒和有限课程结束，最后一个真实动作的响应来自 `summary.final_raw`。探索轨迹不具备验收资格。

## 数据与恢复

`evidence/local_artifacts.json` 列出本地原始轨迹、各批 ONNX、三个 `latest.pt` 和重建数据的哈希。checkpoint 含学习状态，但当前 CLI 不支持通用断点续训；如需增加恢复入口，应从整批检查点启动新物理回合、恢复优化器／随机状态，并先证明分段与连续执行等价。不能直接重跑原输出目录或将未完成批次算成完整实验。

`evidence/closeout/runtime_cleanup.json` 记录清理的每个临时运行目录。恢复时先复制对应 canonical runtime，再按 `restore` 列表覆盖已保存的 GDScript／部署配置；训练运行目录中的最后模型来自第 11 批，因为它用于采集第 12 批。不要用最终第 12 批模型冒充该采样源。六个验收 canonical runtime 均保留，差异脚本在 `harness/runtime_patches`，权重只保留本地路径及哈希。

`freeze_candidate.py` 未执行，未产生 `delivery` 或新包，最终种子 928000–928049 仍封存。准备脚本未核定为完整交付流程，不能把它的存在当成包装或键盘验收证据。
