# 13 precision 原生 witness 数值误差独立核查

结论：6.290 s 的停机来自原生 Newton 求解器按既定 cost 改进量结束后的有限收敛残差。独立重建与冻结失败候选的加速度、约束力逐位一致；没有发现漏力、驱动力映射错误或 held POST／未下发候选的状态混用。本证据不属于硬件能力、真实打滑或拖运失败证据，也不撤销历史 FAIL。

本单两次相同 scratch `mj_forward`（第二次补全 solver 统计），另一次仅运动学和点 Jacobian。零积分、零 LP、零训练，未修改源、求解设置、验收门或实际状态。

## 精确输入与残差

输入为 `coherent_native_soft_13_precision_online1000/attempt_001/true_terminal_poststate.json` 的 q/v 与 `metadata.drive_FF_refresh.actual_blended_native_witness.total_tau18`，即实际 POST 同点上尚未下发的下一 drive 候选。其原始字段名包含 `actual_dispatched`，但这次命令因门拒绝而没有实际下发。没有使用 held-ctrl NPZ 的加速度或约束力。新 `MjData` 使用默认零 warmstart，和原 witness 一致。

MJB SHA256：`3344e5a2dcf86f85640e9b4fda2ac0fd6a65b8b18802e5888f61824fa40c8528`。完整 M 包含源 armature；49 个速度自由度、93 个约束行、22 个接触记录。

`r = M qacc + bias − passive − B actuator_force − Jᵀ lambda`。

| 项目 | 独立读回 |
|---|---:|
| 与冻结候选 qacc／lambda 最大差 | 0／0 |
| 最大残差：root/ty | −1.39980606022e−5 N |
| 次大：root/tx | −7.48629951330e−6 N |
| 最大转动行：right_yaw_input_rotor | +6.75902871072e−6 N·m |
| 与原生 `M qacc − qfrc_smooth − qfrc_constraint` 差 | ≤1.4211e−14 |
| B actuator_force − qfrc_actuator | 0 |
| Jᵀ lambda − qfrc_constraint | ≤1.4211e−14 |
| qfrc_applied／xfrc_applied | 0／0 |

所以这是原生求解梯度；不是公式漏项。把不同自由度的 N 与 N·m 混为一个绝对量，也不能单靠最大值判断物理后果。该候选 equality 加速度关系残差 2.48e−16、contact KKT 8.88e−16、干摩擦模式 3.55e−15、摩擦圆超限 0、源限幅映射误差 0。

## 为什么原生求解合法停止

原源：Newton、100 次上限、tolerance=1e−8，实际仅两次迭代。最终归一化 gradient=1.322938742734e−7；最终归一化 cost improvement=3.798726907613e−9，后者小于 tolerance，符合官方实现的停止分支。`||r||₂ / trace(M) = 1.322938742671e−7`，与 island gradient 一致（差约6.3e−18）。

这属于配置允许的有限收敛误差，不是仅仅 float64 最后几位舍入：本状态保守点积舍入量级约4.73e−11，远小于1.40e−5。不能把原生 tolerance 直接当成有单位的广义力绝对误差。

## 有单位的保守预算

单独 `M⁻¹r` 只是冻结约束力的代数修正，不是真正带接触最优解误差。采用原生加速度目标的 M 强凸性，令 `rho = sqrt(rᵀ M⁻¹ r) = 2.48132002767e−4`，则同一 q/v 下的精确最优加速度误差满足 `||e||_M ≤ rho`。对点 Jacobian：

`||Jp e||₂ ≤ sqrt(lambda_max(Jp M⁻¹ Jpᵀ)) rho`。

以下使用这个保守界。双足检查各自全部63个真实 sole 网格顶点，四杆远销取源 equality 的真实局部销位，未虚构几何。

| 点域 | 加速度误差界 m/s² | 1 ms 局部位移预算 | 5 ms 局部位移预算 |
|---|---:|---:|---:|
| 左足全部顶点最坏点 | 3.94566e−4 | 0.197 nm | 4.932 nm |
| 右足全部顶点最坏点 | 3.92708e−4 | 0.196 nm | 4.909 nm |
| 真实嘴部 TCP | 4.84046e−4 | 0.242 nm | 6.051 nm |
| 轻四杆近销 | 2.10900e−4 | 0.105 nm | 2.636 nm |
| 轻四杆远销 | 5.57981e−3 | 2.790 nm | 69.748 nm |
| 轻四杆 COM | 2.78703e−3 | 1.394 nm | 34.838 nm |

最坏机器人关节为轻四杆 coupler，角加速度误差界0.140348 rad/s²，对应1／5 ms局部角误差预算0.0702／1.754 µrad。原始最大14 µN并不等价于所有关节都只有微小角加速度，因为轻件惯量很小，必须做本项投影。

时间换算为 `0.5 dt²` 的零初始误差、冻结 M/J 局部 Taylor 预算。瞬时加速度界来自强凸性；时间预算不等于变化接触、移动 M/J 或闭环全程的严格积分误差上界，更不能当实机误差证明。[官方原生凸目标依据](https://mujoco.readthedocs.io/en/stable/computation/#algorithms)。

## 供 Owner 批准的一项规范建议

将“公式身份检查”“原生数值求解预算”“实际物理验收”分开：保持有限性、驱动源上限、接触摩擦与模式检查，以及实际滑移／腾空／夹持门不变；原生广义力 residual 使用上述有单位投影预算，而不是事后把统一1e−6改成能过本点的数。

一个可追溯的待批准分配是：取既有最细1 mm整足腾空判据的0.1%，将数值误差分配为每5 ms驱动周期最多1 µm的局部材料点位移预算，同时报告每拍 rho／广义残差行名／solver统计与累计诊断。这个分配来自既有几何验收尺度，不来自当前14 µN数值；本点最敏感的真实四杆远销约70 nm，低于该分配约14倍。该分配仍需 Owner 明确采用，不能声称它已经替代原门或保证长期累计误差。

当前无需据此更换电机、改碰撞摩擦或改硬件。旧门产生的 FAIL 和未下发事实保留；后续若采用新数值规范，应另封候选与依据，不能热改本条记录。
