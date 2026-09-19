# 九项针对性开发实验

以下表格固定记录每项首轮训练与原生开发对照，全部使用 915000–915007。它不是最终保留种子验收，也不把小幅均值变化自动当作换模依据。更晚的制动搜索和控制实验见执行记录及最终报告。

| 技能 | 配对任务成功（旧→实验） | 针对性指标（旧→实验） | 去留 |
|---|---:|---|---|
| 站立 | 8/8 → 7/8 | 位移 mm：1.485 → 6.326 | 保留旧模型 |
| 行走 | 20/56 → 22/56 | 直行松键偏航 °：6.219 → 3.341 | 保留旧模型 |
| 坐起 | 8/8 → 8/8 | 动作变化率：0.01895 → 0.01889 | 保留旧模型 |
| 捡地 | 8/8 → 8/8 | 最低嘴部高度 mm：19.74 → 19.59 | 保留旧模型 |
| 左踢 | 8/8 → 8/8 | 身体偏航 °：29.08 → 31.38 | 保留旧模型 |
| 右踢 | 8/8 → 8/8 | 身体偏航 °：25.26 → 24.72 | 保留旧模型 |
| 前滚 | 7/8 → 7/8 | 恢复后朝向误差 °：15.29 → 14.73 | 保留旧模型 |
| 轮滑 | 0/56 → 0/56 | 主动制动通过：0/56 → 0/56 | 保留旧模型 |
| 轮滑蹲起 | 8/8 → 8/8 | 恢复后倾角 °：1.947 → 2.051 | 保留旧模型 |

行走首轮残差虽然改善基础直行，但丢失旧版长直行成功配对，未采用。后续独立控制候选“松键后 0.5 秒稳定朝向、gain=6”在 56 条开发回放中为 20/56→23/56，未丢失旧成功，移动速度至少保留 100%；模型仍为旧模型。该控制的最终去留以冻结后的验收为准。

坐起动作率仅改善约 0.3%，但位移增加约 47%；捡地深度改善约 0.15 mm，却增加约 52% 位移。右踢与前滚的平均朝向虽略改善，配对结果混合，所以保留旧模型。逐项原始原因如下：

- **站立 / stand_drift_residual_v1_attempt_01**：Standing drift, heading and action rate worsened; one paired success was lost. 原始比较：`results/research_20260911/candidate_evaluations/stand_drift_residual_v1_attempt_01/comparison.json`。
- **行走 / walk_velocity_residual_v1_attempt_01**：Lost an incumbent successful long-forward case at development seed 915004; basic straight-case improvement does not override paired regression. 原始比较：`results/research_20260911/candidate_evaluations/walk_velocity_residual_v1_attempt_01/comparison.json`。
- **坐起 / sit_smooth_residual_v1_attempt_01**：Action rate improved only 0.3%, while displacement increased 47% and yaw drift increased 13%. No convincing quality improvement. 原始比较：`results/research_20260911/candidate_evaluations/sit_smooth_residual_v1_attempt_01/comparison.json`。
- **捡地 / pick_reach_residual_v1_attempt_01**：Jaw depth changed by only 0.15 mm while displacement increased 52% and action rate increased 3.7%. No convincing quality improvement. 原始比较：`results/research_20260911/candidate_evaluations/pick_reach_residual_v1_attempt_01/comparison.json`。
- **左踢 / kick_left_spin_residual_v1_attempt_01**：Targeted body spin worsened (29.08 to 31.38 degrees) despite all eight kicks succeeding. 原始比较：`results/research_20260911/candidate_evaluations/kick_left_spin_residual_v1_attempt_01/comparison.json`。
- **右踢 / kick_right_heading_residual_v1_attempt_01**：Mean yaw improves only 0.53 degrees with mixed paired outcomes (3/8 worse); no clear improvement. 原始比较：`results/research_20260911/candidate_evaluations/kick_right_heading_residual_v1_attempt_01/comparison.json`。
- **前滚 / roll_landing_residual_v1_attempt_01**：Mean final heading improves only 0.56 degrees with mixed paired outcomes; the same one of eight roll cases still fails. 原始比较：`results/research_20260911/candidate_evaluations/roll_landing_residual_v1_attempt_01/comparison.json`。
- **轮滑 / roller_brake_residual_v1_final**：No active brake case passed the fixed stop/stand/distance/no-reverse gates. The candidate avoids many falls but continues moving. 原始比较：`results/research_20260911/candidate_evaluations/roller_brake_residual_v1_final/comparison.json`。
- **轮滑蹲起 / crouch_stable_residual_v1_attempt_01**：All eight cases pass on both sides, but posture score slightly decreases and final tilt increases; no clear improvement. 原始比较：`results/research_20260911/candidate_evaluations/crouch_stable_residual_v1_attempt_01/comparison.json`。

训练参数、预算、代码/模型来源位于各运行目录的 `config.json`，原生套件保留完整轨迹、case SHA 和模型 SHA。全体实验索引位于 `results/research_20260911/experiment_decisions_index.json`；实际工时不按这些并行任务预算相加，而使用统一账本。
