# 历史试验驱动与诊断脚本

这些是本机本次实际使用的驱动／分析脚本副本，路径绑定于 `results/jolt_learning_20260911/`。它们用于审计和建立新会话时参考，不是可在已完成目录上重复运行的通用命令。不要直接执行来覆盖原记录。

每组训练的冻结 Python 代码、运行时、权重与原始轨迹仍位于本机结果目录；详细限制和复测入口见 [复现说明](../REPRODUCE.md)。所有系列保留完整源模型、训练参数、样本预算与配对评分。`prepare_teacher_replay.py` 已把执行后曾补写的 data／anchor_sha256 字段纳入生成逻辑，现有原始数据文件与来源 SHA 不变。

`validate_frozen_candidate_attempt1.py` 是 release 自检停止的首次驱动；`validate_frozen_candidate_attempt2.py` 是实际完成最终配对／干净环境后，因误用任务评分器处理复位命令带而停止的第二次驱动。`finish_frozen_validation.py` 复用其原生轨迹完成工程 shadow。`validate_frozen_candidate.py` 是纠正工程命令带处理后的复现参考，不声称已用该文件重新跑过全部保留集。计时、监督及一次性恢复脚本中的历史 PID 也须按新会话重建，不能原样作为新会话的进程身份。
