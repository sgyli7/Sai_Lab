# G1 π0.5 长距离搬运：冻结 60 场检查点

本提交保存截至 2026-10-18 的实验代码、已有测试源码、结果摘要和实际执行收据。
**资料交付已完成并核验；机器人完整任务仍未达标，整体目标未完成。**
用户要求先提交 GitHub 再暂停；新增 GPU、示教、训练和测试预算尚未批准。

## 实际结果

| 策略 | 完整任务成功 | 4m / 60° 携物阶段合格 |
| --- | ---: | ---: |
| GR00T N1.6 | 4/20 | 20/20 |
| π0.5 微调，完整 VAL 选中 745 步 | 1/20 | 15/20 |
| 官方 π0.5 基座，同一 G1 适配链路 | 0/20 | 0/20 |

60/60 正式场通过独立证据核验，120 路 head/observer MP4 为 50fps 仿真原速。
每策略 20 case = 5 个真实箱体 XY 位置 × 4 个种子，不能称为 20 个独立布局。
冻结完整门槛为 16/20，所有策略均未达到；未降低门槛、未边评边调。
完整任务要求抓取、长距离携物、持物转向、到目标、释放并稳定至少 2 秒，
且通过容纳、箱体支撑、与机器人分离、低速和站立条件。

携物判据中的 held 是 50Hz 双手 contact AND 箱体 actor 抬升 >3cm 的代理，
可跨采样间断，不证明全路线持续安全握持。`hand_contact_gap` 包含抬升不足，
不能直接解释为手接触丢失。AABB gap 为 None 表示未评估，不是实测重叠。
独立 fall/drop 与源端 world-z 提前失败分别报告；缺失源字段保留 UNKNOWN。

## 配对速度与不确定性

- GR00T/微调仅 1 个共同完整成功 case：x_minus1043，34.42 vs 49.06 仿真秒。
  这一对差 14.64 秒，不能推广为全部 20 case 的整体速度排名。
- 共同携物合格 15 对，微调−GR00T 的仿真时间配对中位差 +3.52 秒；
  墙钟中位差 +46.342838279 秒。仅比较达到相同终点的同一 case。
- 含基座的共同完整成功或携物样本均为空，速度 N/A；快失败不排第一。
- Wilson、McNemar 和按 5 个实际位置聚类的统计保留在结果 JSON 中。
  不宣称最终 winner、总体架构胜负或完整任务训练收益已证实。
- 同步推理暂停未编码进原速视频；没有证明墙钟 1× 或实机能力。

## 数据、模型与官方案例边界

100 条完整去重源导出按 seed42 固定拆分 80/10/10，仅 TRAIN 拟合归一化。
“文件完整”不等于 100 条已独立证明严格释放放稳成功的示教；
转换器末帧 reward/done 是写入约定，原 80 TRAIN 的严格任务覆盖仍未知。
新增 TRAIN 为单一家族 native201 的 1396 步携物前缀，没有新增完整 5m 放置监督。

微调从官方 pi05_base 初始化，BF16 全参、batch1、累积8、LR1e-5，
实际完成 745 更新，使用完整原 10 VAL /190 chunks 选择；留出离线 400 次推理已完成。
部署仅使用同期 head RGB、自身关节和 prompt，不读取物体/目标真值。
32 动作为 28 个上肢绝对关节角 + 高度 + vx/vy/yaw，50Hz 接 Homie，正式物理 200Hz。
手部 14 GT 通道恒0、vy 误差变差、raw vx 限幅、单前缀覆盖窄、预训练配方差异均保留披露。

相同 G1/Homie 下，官方原生策略的原场景 CPU 和 CUDA 兼容路径各实测成功一次；
没有证据锁定错误 G1 型号或唯一根因。原约 1.963m 的位置接近判据不同于本轮
5m 释放稳定判据。微调在原短任务的一次诊断也失败，不能仅用任务更长解释低分。
这些适配和验收边界应由执行者事先解释，不能归咎于用户修改前提。

## 代码与证据入口

- `acceptance_pose_speed.py`、`audit_episodes_v3.py`：冻结物理验收与独立审计。
- `make_formal_segment_candidate_v3.py`、`round03_driver_allocated.py`：有限批次与原监督流程。
- `training_carry_memory_v3.py`：本轮训练；`benchmark_v3_notes.py`、
  `transport_report_v2_notes.py`：最终统计与报告。
- [冻结结果与逐文件来源清单](checkpoints/frozen60_20261018/checkpoint_manifest.json)
- [60场实际命令](checkpoints/frozen60_20261018/executed_commands_full60.json)
- [逐场结果与统计](checkpoints/frozen60_20261018/benchmark_summary.json)
- [携物和配对统计](checkpoints/frozen60_20261018/transport_statistics.json)
- [120路录像索引](checkpoints/frozen60_20261018/video_index120.json)
- [独立永久归档审计](checkpoints/frozen60_20261018/independent_archive_audit.json)

Python 模块从仓库根目录运行。`../pi05_g1`、`../pi05_g1_round02` 和
`../pi05_g1_round03_mobile` 保留历史实现/数据准备依赖，`../vla_resources` 是共享 FIFO。
历史 README 属于各自轮次，不能当作本轮完整成功声明。
本次仅保存已有测试源码，没有执行新的测试、GPU 推理或训练。

## 本地完整交付、备份与续接

Git 中的 JSON/CSV 为原件逐字节复制，实际命令、源路径和 SHA 保留；
这是代码及小证据检查点，**不是**完整数据、模型、依赖环境或视频的 Git 分发。
Git 单独检出不足以恢复所有训练输入与运行环境。

完整实验：`/home/ethan/Projects/Sai_Lab/.scratch/pi05_g1/round03_mobile/`。
本地可读总入口：
`/home/ethan/ProjectBackups/2026-10-17/g1_pi05_mobile_full60_execution_20261018T045400Z/delivery_index_zh.html`。
永久备份：
`/home/ethan/ProjectBackups/2026-10-17/g1_pi05_mobile_full60_20261018T045400Z/`。
独立审计通过 57,128 文件、1,545 个无损 gzip 文本、1,401 个共享 inode 硬链接；
全部存储与解压原始 SHA 通过。硬链接与原件共享存储，不是独立物理磁盘副本。
大文本的原路径需通过原归档器恢复，本次未执行恢复。所有原件保留。

旧 12 GPU 小时硬上限及各阶段额度不重置。已收费 40194.56960293202 秒，
训练分项仅余 0.318256675 秒；不能用总余额自行重训或挪用分项额度。
[下一阶段方案](checkpoints/frozen60_20261018/next_performance_plan_UNAPPROVED.json)
请求最多新增 14 GPU 小时，先最多 3.5 小时验证合格完整示教，再决定能否进入训练。
此方案未批准、未执行，不保证达到 16/20。后续有 token 后先读实际 Rootstate 与新的人类授权，
保持旧实验、权重、录像、评分和备份不可变；正式 TEST 轨迹不得训练。
