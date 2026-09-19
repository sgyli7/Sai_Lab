# MuJoCo ↔ Godot/Jolt Sim2Sim

> 2026-09-10 更新：本文保留原有映射与训练历史。最新研究修复、九模型选择和质量缺口见
> [实验报告](RESEARCH_RESULT_20260910.md)，准备与复测入口见 [REPRODUCING.md](REPRODUCING.md)。
> 新研究采用独立实验目录；右踢 KR06／前滚的时间输入与候选模型契约以新报告和 manifest 为准。

Python 是唯一控制器。MuJoCo 编译后的 `MjModel` 是模型真源；Godot 4.7.2 + 内置 Jolt 是第二个物理后端。ONNX policy 只在 Python 里跑。

## 一键

```bash
export PATH="$HOME/.local/bin:$PATH"
export DISPLAY="${DISPLAY:-:1}"
cd /path/to/MicroDuck-Godot-Simi2Sim
./run.sh
```

需要：Godot 4.7.2（`~/.local/bin/godot`）、Python 3.12+（uv）、`microduck_rl` 的 `scene.xml`、仓库内 ONNX。

窗口里自己开：

```bash
cd /path/to/MicroDuck-Godot-Simi2Sim
uv run sim2sim-play            # alpha_walking
uv run sim2sim-play --local-ppo
uv run sim2sim-play --roller   # scene_rollers.xml + roller.onnx
```

**按住 W/↑ 才会走**（松手 = idle，`vx=0`）。时间流速滑条只改 Python 墙钟睡眠，不改 `dt` / `decimation`；不是 TimeScale 把 ONNX 弄傻的。窗口里按 **6**（或底部「轮滑 6」）会 `execv` 重启进程，走路脚+技能 ⇄ 轮滑机，和 MuJoCo `infer_policy` 的 6 一样（不能热切换 XML）。

换机器人：复制 [`robots/microduck.json`](robots/microduck.json)，改 `mjcf` / `home` / obs 维数（若不是 61D 还要改 `obs.py`）。

## 映射（已验证的事实，不是假设）

| 量 | MuJoCo | Godot/Jolt |
|---|---|---|
| 世界系 | Z-up, quat wxyz | Y-up；协议仍用 MuJoCo 系，GDScript 做 `(x,y,z)→(x,z,-y)` |
| Body 节点 | body 系 + 惯性系 `ipos/iquat` | RigidBody3D 对齐**惯性主轴系**（COM=0，`inertia` 对角） |
| tscn `Transform3D` | — | 9 个浮点是 basis **行**：`x.x,y.x,z.x, x.y,y.y,z.y, x.z,y.z,z.z`。按轴拼接会转置旋转；reset 会覆盖 body 世界位姿，但 **CollisionShape 局部变换不会**，脚垫被转置后会提早 1–2 tick 碰地并产生角冲量 |
| 碰撞 mesh（非脚） | mesh convex hull | `ConvexPolygonShape3D`，hull 顶点 cap 128（`jaw_soft` cap 512） |
| 脚碰撞 | `*_foot_collision` mesh（凸包约 5000 顶点） | STAND 2 mm 支撑带的 **共面 16 边形棱柱**（向上挤出 8 mm）。AABB 假边角和角球会滚；整脚 128 点凸包会撑在顶点上把 `local_ppo` 摔倒；轮廓 sphere / toe-cap 改变不了 t=0.92 的面接触 |
| 碰撞层 | `contype` / `conaffinity`（OR） | `collision_layer` / `mask`。Microduck 上每组 contype==conaffinity，两者等价 |
| 自碰 | 默认不生成 parent–child contact | `add_collision_exception_with` 只排除 parent 和 grandparent（2-hop hull 仍重叠）。排除整条 ancestor 会丢掉 trunk↔foot |
| 关节 | hinge，`q` 是相对 q=0 的 twist | body 系 `R_rel(q)=Rot(axis_parent,q)@R_rel(0)` 抽角；reset 后 **rebake** `node_a/node_b` |
| 执行器 | `<position kp kv forcerange>` | `τ=clamp(kp(e)−kv qd, forcerange) − damping·qd − μ·tanh(qd/0.05)`，`apply_torque`。`qd` 是运动学差分（见下节），不是 Jolt hinge 速度 |
| `qd` / `base_angvel_local` | `qvel` / 体轴 `cvel` | 每物理 tick 姿态有限差分 + 1-tick EMA。**不用** Jolt `RigidBody3D.angular_velocity`（含 Baumgarte，idle 会报 gyro_z≈+0.22）。`base_linvel` 仍读 Jolt |
| `armature` | 关节空间 `I += A nnᵀ` | `I += diag(A nnᵀ)`，主轴再 floor 到 `max/10`（满各向异性或 cond≳80 会让 Jolt `\|qd\|` 爆炸；各向同性 `I+=(A,A,A)` 会把摆腿惯量抬高一个数量级、run 变慢） |
| `frictionloss` | 关节库仑摩擦 | PD 里平滑库仑（Jolt hinge friction 未绑定） |
| 地面 | `plane` z=0，μ=1 | 40×0.02×40 `BoxShape3D`，顶面 y=0，μ=1（不用 `WorldBoundaryShape3D`） |
| pin（仅 C1） | `sim2sim_pin_base` weld 钉住当前位姿 | `RigidBody3D.freeze` 基座。**禁止**在 `mj_step` 后回写 freejoint：那是自由落体，关节不受重力，C1 会假绿成 0.0013 |
| 读数时刻 | `mj_step` 后 `xpos/cvel` 仍是上一步，需再 `mj_forward` | 报告步进后姿态；两端对齐到 1e-8（自由落体 tick 1–8） |
| 控制频率 | dt=0.005，decimation=4 → 50 Hz | `physics_ticks_per_second=200` + TCP lockstep。**禁止**在两条 Python 命令之间把机器人 `RigidBody3D.freeze` 成静体：Jolt 会清零速度，每个 control step 从静止重启（跌落 `vz` 卡在 `g·Δt=0.196`，C3 `z_min≈0.11`，走路 xy≈0.14 m）。正确做法：在当帧 `_physics_process` 里自旋等到下一条命令，物理步不推进 |
| 力矩上限 | 可选 `current_limit_a=1.75` → `±kt I` | 同样，`set_tau_limit`（kt=0.3660，limit≈0.6405 Nm） |
| 重力 | 9.81 | `default_gravity=9.81` |

### 运动学速度（Godot）

Jolt `RigidBody3D.angular_velocity` 带着约束修正（Baumgarte）速度：站立、姿态静止时曾报 `gyro_z ≈ +0.22 rad/s`，`hip_yaw` / `head_yaw` `qd` ±0.24。不能当 MuJoCo `cvel` / `qvel` 用。

`physics_server.gd` 每个物理 tick（dt=0.005）用姿态有限差分：

- `qd = wrap(q_new − q_old) / dt`（`wrapf(..., −π, π)`）
- `ω = axis_angle(R_new · R_oldᵀ) / dt`
- 1-tick EMA：`KIN_VEL_TAU=0.005`，`α = 1 − e^(−dt/τ)`（dt=τ 时 α≈0.632），压掉 200 Hz 模态。不用 decimation 窗（20 ms 延迟会拖死 PD）。

用于 obs 的 `base_angvel_local` / `qd`，以及 PD 阻尼和 Coulomb。`base_linvel` 仍读 Jolt。full（非 lite）reply 另给 `base_angvel_jolt`（体轴、未滤波的 Jolt ω）。

修后 idle：gyro ≈ (−0.0007, −0.0018, +0.0010），qd ≈ 0；驱动铰上 qd vs Δq/dt **corr>0.9**。S3 spike RMSE **0.00283 未变**。单测：[`tests/test_godot_kinematic_vel.py`](tests/test_godot_kinematic_vel.py)。

## 明确不映射（不做伪等效）

- Jolt `RigidBody3D.angular_velocity`（Baumgarte 修正速度；观测与 PD 用运动学差分，见上节）
- 软限位 `solref/solimp`（Jolt 限位是硬约束）
- 接触软约束 vs Baumgarte / `penetration_slop`。MuJoCo 默认 `solref=(0.02,1)` / `solimp=(0.9,0.95,0.001,0.5,2)`；Jolt 全局 `penetration_slop=0.0002`，无 per-geom 柔度。

  **solref/solimp 扫描（`python -m sim2sim.mj_solref_sweep`，2.5 s，均为独立诊断入口，不参与门禁）**：

  | 档位 | solref | solimp width | 首次穿透(mm) | tilt@0.92(°) | z_min | 结论 |
  |---|---|---|---|---|---|---|
  | S0 XML默认 | (0.02,1) | 1e-3 | **7.6** | 81° | 0.040 | 基线锚点 |
  | S1 timeconst=2dt | (0.01,1) | 1e-3 | 2.5 | 74° | 0.044 | 穿透降但z_min升 |
  | S2 timeconst=dt  | (0.005,1) | 1e-3 | 2.5 | 74° | 0.044 | 与S1相同 |
  | S3 紧 solimp | (0.005,1) | 1e-5 | 2.4 | 69° | 0.048 | 穿透降、z_min反升 |
  | S4 直接刚度 | (-1e6,-1e4) | 1e-5 | — | —  | 0.111 | 数值不稳定（弹起） |

  **副分支确认**：穿透从 7.6 mm 降到 2.4 mm，但 MuJoCo z_min 从未降到 0.031——反而**升高**。全程 hip_n=0（MuJoCo 机器人从不髋着地）。**C3 缺口的根因不是穿透量，而是两端的最终状态**：MuJoCo 在 81° 时形成**头+双脚三脚架**（jaw x≈+0.12，脚仍着地 foot_n=2），机器人在 z≈0.040–0.048 处稳定；Godot 中 jaw 在 t=1.26 掀起脚（脚被弹离），随后髋着地，z_min≈0.031。差异来源是 **MuJoCo 接触 reduction 机制**：0.737 kg 全脚凸包对平面第一下仅报 2–4 个 contact 点（外侧脊），四点面变成两点线，俯仰被解锁，整个倒塌轨迹不同；Jolt 对凸包对盒子报出全垫 4 点流形，俯仰被锁，机器人只能缓慢前倾到 jaw 接地后再髋着地。这是引擎接触点数/reduction 算法的结构性差异，不是 solref 单一参数。

  **Jolt 侧反向闭环**：唯一能让 Jolt 达到 4 mm 量级穿透的旋钮是全局 `penetration_slop`。实测 `penetration_slop=0.002`（~1.7 mm）剥离更慢（t=0.92 仅 23.6°），1.5 s 截断的 z_min=0.0485 是假齐（当时 vz=−0.30、脚已离，t=1.60 髋着地，稳态 0.032），且会毁掉步态。已改回 0.0002。

  **结论**：C3 差异（MJ 0.040 vs GD 0.031）是 MuJoCo contact reduction（mesh-plane 只报外侧脊 2–4 点）vs Jolt 凸包-盒子全垫 4 点流形的引擎结构性差异，属于「明确不映射」范畴。其余映射（质量、惯量、armature、PD、摩擦、几何）均正确。

  C3 时间线（HOME PD 跌落）：MuJoCo **t=0.11 首次接触是全脚 convex hull 外侧缘最低两点**（x≈−0.009 与 +0.022，y 在外侧，dist **−4.2 mm**），之后全程只有 2–4 个 contact（reduction）。hull 最低 0.05 mm 是 ~6 mm 宽外侧脊。t=0.80 只剩脚尖（x≈+0.028，46°），t=0.92 已 81°/jaw 且脚仍着地，t=1.50 仍是头+脚、z=0.046。Godot 全垫：踝 z 从 t=0.12 到 t=0.92 **钉在 0.010**。2 mm 带 3D hull 仍被收成 35 mm 面；改成顶点球后紧 slop 能在 t=0.40 收到踵线，体重随即把 0.1 mm 脊压回整垫。

  单凸包微楔几度后重新变成面，帮不上俯仰剥离。侧向收窄也不改矢状力臂（16 mm 能把 C3 拉到 0.048 的是**前后**中带，不是左右变窄）。

  本轮排除（两体 weld 已能做到 0.1 mm 级误差，问题不在关节松）：
  - **共面两体鞋跟**（前掌 xmin=-0.006 + 独立鞋跟刚体）：岛能分开，t=1.26 前掌收到脚尖、鞋跟 n=0，但 t=0.92 仍 ~25°（全长支撑仍在），C3 z_min **0.032**。
  - **鞋跟 +1 mm**（同一套 sibling+rebake FixedConstraint）：C3 z_min **0.041**（更接近 0.046），着地时焊误差回到 ~1 mm，walk Δyaw +77°，run vx **−0.028**。抬高量和地面抢约束，刚焊也救不了步态。
  - **全垫 + 独立脚尖体**（xmin=0.025，+3 mm，mass=0.002）：t=1.26 出现 **脚尖+jaw 三脚架**（脚垫 n=0、脚尖 x≈+0.034），run 世界 vx **0.208 vs 0.202**（不腰斩）。t=1.34 脚仍被掀，C3 仍 **0.032**。脚尖 mass 提到与脚相同（0.03）焊更差、C1 变坏、run 变慢。`position_steps=10` 让 t=1.26 三脚架消失且 `local_ppo` 摔倒，已改回 6。
  - **同体后掌+脚尖两块凸包**：cshape 能隔离（t=1.26 只剩 shape 1），但毒化全垫 4 点流形，run vx **0.073** 腰斩。
  - **`velocity_steps=64`**（全垫）：t=1.26 脚已离地，`local_ppo` 摔倒。已改回 32。
  - **两体高度差**（鞋跟 zmin、前掌抬 2 mm）：FixedConstraint 被接触拉开 **~2.5 mm**（约等于抬高量），t=0.12 两岛仍同时着地，错高被焊“吃掉”。内建 Jolt 没有 per-joint solver override。
  - **同体短鞋跟（xmax=-0.008，zmin）+ 前掌抬 1.5 mm**：第一次做出 MuJoCo 式剥离——t=0.12 只有鞋跟、t=0.40 tilt **8.7° vs MJ 9.5°**、t=0.80 只剩脚尖。t=0.92 仍只到 50°，随后掉髋，C3 z_min **0.032**。两块 compound 即使抬高只有 0.25 mm，`local_ppo` run 仍摔倒。步态需要单块全垫的 4 点流形。
  - **3 条共面横向 capsule**（同体、r=1.5 mm）：C3 z_min 仍 **0.032**，但 t=1.26 出现 **脚尖+jaw 三脚架**（16 边形此刻脚已离）。`local_ppo` 不倒，run 世界 vx **0.187 vs 0.203**（不腰斩），xy_rmse 0.149。未采用（C3 仍掉髋）。
  - **capsule 鞋跟更低 0.5–1.5 mm**：t=0.12 只有鞋跟、t=0.92 只剩脚尖 **45°**，`local_ppo` **摔倒**（站立只踩到踵线）。
  - **共面 capsule 脚尖 +4 mm**：t=1.26 脚尖 x≈+0.035 对齐 MJ，run 世界 vx **0.043** 腰斩。
  - **同体/两体脚尖 bumper**（STAND 时抬高 4–8 mm）：高倾角几乎不产生接触；同体毒化步态，两体焊还让 C1 变差、run 摔倒。
  - **2 条共面 capsule**：左右脚都粘到 t=0.92，`local_ppo` 摔倒。
  - **单条矢状 capsule**（跟–尖一线、r=1.5 mm，对应 MJ C3 首次两点）：t=0.12 已是 2 点、x 跨满 [-0.016,+0.030]，踝 z 钉在 0.010 到 t=0.92（tilt **29.6°**）。t=1.26 才收到 1 点脚尖、tilt 79°，随后髋着地，z_min **0.032**。`local_ppo` **摔倒**，run 世界 vx **0.057**。μ=1 下两点跟–尖线仍然锁俯仰；4→2 不够。
  - **16 边形 + `WorldBoundaryShape3D` 地面**：仍是 4 点整面，t=0.92 tilt **30.8°**，z_min **0.031**。凸包对平面和凸包对盒子一样锁。已改回 40×0.02×40 盒子。
  - **单凸包 16 mm 平底 + 矢状四分之一圆**（r≈16 mm，横向满宽）：hull 侧影不是 49 mm 平面（flat 近地点 ~26 mm）。C3 z_min **0.040**，但 t=0.92 仍只 **17°**（踝 z 钉 0.010、三点 x 跨 18 mm）；t=1.26 才走到脚尖 x≈+0.022 / 53°。切线斜率 0 + slop 0.2 mm 把有效面拉长。`local_ppo` 不倒但走不动，run 世界 vx **0.033 vs 0.203**，xy_rmse 0.795。
  - **16 mm 平底 + 两端 2 mm 线性切角**：切角从边沿就是 ~7°。C3 **站住**（z_min **0.108**，t=1.50 tilt 10°）。t=0.40 接触扩到 x≈−0.017（接到鞋跟）。walk 世界 vx **0.127 vs 0.122**，run 摔倒（Δyaw −198°）。浅切角是摇杆，会接住 HOME 跌落。
  - **16 mm 平底 + 只留脚尖 2 mm 切角**（鞋跟悬崖）：C3 z_min **0.0462** 是假齐——t=0.12 无接触（缺鞋跟，踝 z=0.006 穿地），随后向后摔，t=1.26 jaw 在 **x≈−0.25**（MJ 是 **+0.24**），脚已离。`local_ppo` **摔倒**，walk vx 变负。C3 只看 z_min 会误判；必须看 jaw 的 x 符号和脚是否还着地。
  - **地板 μ=0**（诊断）：脚会仰（t=0.92 tilt **54°**、踝 z 升到 0.020），但整脚在冰上后滑，仍掉髋，z_min **0.030**。μ=0.5 仍四点粘住且 `local_ppo` 摔倒。MuJoCo 脚-地 `condim=3`（无扭转/滚动摩擦）；Jolt 四点各向同性 μ=1 会锁俯仰。降 μ 不是可用插值。几乎平坦的摇杆（矢状矢高 ≲2 mm）半径必大于躯干 COM 高度，会自稳站住；不稳定摇杆又太弯，走不了。
  - **16 mm 踵垫 host + 1 mm sibling 脚尖**（FixedConstraint）：C3 时间线是目前最接近 MJ 的一次——t=0.80 tilt **46.8° vs MJ 46°**（只剩脚尖岛着地），t=0.92 **73.5°** 脚尖仍着地、髋未碰。但踝 z 只到 **0.017 vs MJ 0.038**，t=1.26 髋仍着地，z_min **0.031**。t=0.12 踵垫 n=0、脚尖先着地（焊已拉开 ~2 mm）。`local_ppo` **摔倒**（xy 0.064，walk vx 负）。短踵垫能仰、sibling 脚尖补不上步态。已改回全垫 16 边形。
  - **髋凸包并未变肥**：STAND 时 128-hull vs mesh zmin 差 <0.05 mm。MJ C3 t=1.50 髋 mesh 仍离地 **~14 mm**（ncon=0）。Godot 全垫 t=1.26 踝 COM 已到 **0.030**（对齐 MJ），但脚 **n=0**（倾倒把脚弹离），随后髋着地。C3 差的是倾倒后脚没留在地上，不是髋几何。
  - **全垫倾倒时间线**（dense dump）：t=1.00 四点已收到 6 mm 脚尖面；t=1.14–1.24 是 **2 点线**（cxspan 0.3 mm，x=0.031），踝在升、tilt 57°→79°；**t=1.26 jaw n=1 把踝 vz 从 0.06 打到 0.26**，脚离地，t=1.44 髋着地。
  - **全垫 + 垫前缘共面 sibling capsule**：能跟到 jaw 碰上那一帧（t=1.28 仍 n_pad=2 且 n_toe=2），但垫前缘和 capsule 仍差 **2 mm** 力臂，冲击后踝 vz **0.35**，z_min **0.031**。
  - **host 裁到 x≤0.022 + 前缘 capsule**：t=1.22 垫已卸载、只剩 capsule 线；t=1.24 jaw 仍把这条 μ=1 线绊飞。
  - **16 mm 踵 host + 远端 capsule（mass=0.03）**：焊不再拉 7 mm。t=1.22 tilt **81°、踝 z 0.026、n_toe=2**（已是 MJ 三脚架缺 jaw）；t=1.24 jaw n=1 仍绊飞。jaw hull 从 cap 512 升到 8192，第一下仍是 n=2、impulse≈0.013。μ=1 的脚尖线接不住 jaw 的水平冲击。已改回全垫 16 边形、sibling mass 0.002、jaw cap 512。
  - **全垫 + mesh 脚尖 sibling 横向 capsule**（lip 在 STAND 时 +2.6 mm，x≈0.033，对应 MJ t=0.92 接触）：t=0.40–0.92 仍是垫子四点；t=1.26 出现 **jaw+垫前缘+capsule 三脚架**（踝 z **0.027**，jaw x **+0.26**）。**60 ms 后 jaw 撞击把脚弹飞**（t=1.32 踝 z 升到 0.040、n=0），髋仍着地，z_min **0.031**。焊在 t=1.26 只拉开 ~0.2–1 mm，不是主因。
  - **裁短 host 垫到 x≤0.028**：t=1.10 垫前缘和 capsule 同时着地，但 t=1.26 脚已经离地。短垫倾倒更猛，capsule 单独接不住。
  - **同体第二块 capsule**（无 sibling）：Jolt 流形被 16 边形吃掉，倾倒过程 capsule **零接触**（cshape 全是 0）。已证实「同体第二块即使抬高也不参加接触」。
  - **全垫 + sibling 两点 sphere**（同一 lip）：比 capsule 更早丢接触，t=1.26 几乎 n=0。点比线更接不住 jaw 冲击。
  - **`condim=3` CoP Coulomb**（地面+全身 engine μ=0，切向力打在接触形心、法向仍是四点垫）：t=0.92 仍 **32.5° / 四点面**（踝 z 0.011）。俯仰是支撑多边形的法向锁，不是扭转摩擦。t=1.22 出现 jaw+脚三脚架（tilt 81.5°、踝 z 0.027），t=1.24 脚仍被掀，z_min **0.031**。已关。
  - **C3 第一下是 `top_head_shell`（g49）不是下巴**：MJ t=0.92 单点 x≈0.278、穿透 **1.7 mm**（solref=0.02）；20 ms 后 `jaw`+`bottom_head_shell` 铺开成 ~87 mm 宽的下巴垫。Godot 512-hull 第一下穿透约 0.2 mm。把 g49 换成龙骨上 r=2 mm 横向 capsule（下巴 hull 全去掉，cshape 全是 0）**照样 t=1.26 绊飞脚**，z_min **0.028**（头更穿）。μ=1 的线在晚到的冲击下接不住，圆不圆、是不是下巴都不是根因。已改回三块 hull。
  - **`baumgarte=0.15`（复测 2.5 s）**：t=1.26–1.28 脚尖仍 2 点着地，但 **t=1.32 jaw n=1 仍把脚掀离**，t=1.50 髋着地，z_min **0.0315**。旧记录「t=1.26 脚尖仍着地、xy_rmse 0.331」没写 jaw 晚 60 ms 照样绊。已改回 0.35。
  - **`penetration_slop=0.002`**（对齐 MJ 第一下 ~1.7 mm）：剥离更慢（t=0.92 仅 23.6°）。C3 1.5 s 的 z_min **0.0485 是假齐**——当时 vz=−0.30、脚已离；t=1.60 髋着地，稳态 **0.032**。不要用 1.5 s 截断当闭合。已改回 0.0002。
  - **MJ C3 t=0.11 第一下就是全脚 convex hull 外侧缘最低两点**（左脚 `pos≈(0.022, 0.053)` 与 `(-0.009, 0.053)`，dist **−4.2 mm**，`condim=3` / `solref=0.02`）。hull 最低 0.05 mm 是一条 ~6 mm 宽的外侧脊，不是 36 mm 整面；2 mm 带按定义跨 2 mm 高，把它压成 16 边形会抹掉这条脊。
  - **2 mm 带真 3D hull**（不压平，~1060 点，`FOOT_COLLISION=band_hull`）：Godot 仍是 3 点、矢状跨 **35 mm**（0.1 mm camber < slop 0.2 mm），t=0.92 仍 26°；t=1.00 才收到脚尖；t=1.28 脚还在，**t=1.32 jaw 照样掀脚**，z_min **0.031**。已改回全垫 16 边形。
  - **同一组 hull 顶点改成球**（`band_spheres`，r=1 mm，n≈1060/脚）：凸包对地是面，球是点。默认 slop 下 t=0.12 已是 **n=64（上报帽）整垫 39 mm**。`penetration_slop=0.00005` 时 t=0.40 能收到踵线（跨 5 mm，像 MJ），但体重把 0.1 mm 脊压平，t=0.80 又是整垫 38 mm / 15°；t=1.44 jaw 仍掀脚，z_min **0.031**。Jolt 没有 MJ 那种「穿透 4 mm 仍只留 2–4 个 contact」的 reduction。已改回 16 边形和 slop=0.0002。
  - **3 个包络球（MJ 站立/走路每脚 1–3 点）**：静态 3 球 t=0.92 已收到单点脚尖（35°，MJ 此时是 81°），随后脚离地早于 jaw。每步 snap 到当前最低 mesh 顶点后，**t=1.32 出现 jaw+脚三脚架**（tilt 77°、踝 z 0.024、jaw x +0.22），但硬 jaw 冲击仍把脚掀掉，t=1.60 髋着地，z_min **0.030**。
  - **迁移 3 球 + `top_head_shell` 2 mm 龙骨 capsule**：第一下冲量 0.009–0.022（hull 是 0.17）。t=1.32 仍是三脚架；t=1.34 脚 n=0。随后最低 0.4 mm 带沿外侧缘以 ~2 mm/tick 爬回鞋跟（不是瞬移）；world-x 滞后会因「同一组脚尖顶点随俯仰在世界 x 上后移」而解锁。把 <8 mm 簇 pin 在鞋跟上会绕踵线摔倒（踝穿地）。脚尖 pin 能把三脚架留到 jaw，20 ms 后仍被掀。4 mm 半径不改第一下高度（底仍在 mesh zmin）。
  - **r=4 mm 迁移球**（对齐 MJ 脚第一下 dist=−4.2 mm）：t=1.32 仍三脚架，t=1.34 踝跳了 5 mm，球底贴 mesh 够不着地。`speculative_contact_distance=0.002` 把 t=1.34 jaw 冲量抬到 **0.082**，更糟。Jolt 的 shape margin 是先缩小再加壳，**不会**变成 solref 那种 1.7 mm 软穿透。C3 剩下的是硬法向速度约束，不是接触点数。已改回 16 边形 + 三块 jaw hull，`speculative=0`。
  - **jaw-only Generic6DOF Y 弹簧垫**（脚仍撞静态盒）：垫会动。6 mm Y 硬限位在 t=1.26 被撞穿，t=1.28 仍掀脚。去掉限位后垫沉到 −13 mm，硬 jaw↔垫接触仍在同一拍把脚掀掉。把垫减到 10 g 只让头把垫打穿（冲量更大）。刚体垫改变的是 J，不是约束硬度。
  - **jaw-only SoftBody 布**：无重力锚定的布会先垂 7 cm；只松开 C3 着地点 12 cm 补丁后布仍被凸包打穿（jaw z→−0.03，z_min **0.013**）。CCD + `soft_body_point_radius=3 mm` + 下方 3 mm 硬背板：t=1.26 就撞上背板，脚 n=0，z_min 回到 **0.031**。Godot 4.7 内置 Jolt **没有** per-geom 接触弹簧；引擎层垫/布都映射不成 MJ `solref=[0.02,1]`。已撤回层拆分和垫，植物仍是全垫 16 边形。
  - **jaw hull → 轻 bumper + Generic6DOF 弹簧**（硬接触离开 `jaw_soft` 运动链）：角锁 + k=1500 时 t=1.32 bumper 被拉开 **17 mm**，脚仍掀，z_min **0.031**。k_ang=2（几乎无角刚度）脚能留到第一下，但 hull 脱开、头穿地，z_min **0.011**（头已不在链上，不算 C3）。k_ang=25/80 ± 线性硬限位仍在同一拍掀脚，稳态 z **0.030–0.031**、髋着地。刚体岛只改接触质量，角锁把偏心冲量原样拷回脖子。
  - **jaw hull 关掉 + 半固定 SoftBody 球**（想让第一下变成软点对地）：Jolt 下节点变换/pin 都没把球跟到头上，球留在世界原点搅脚（t=0.12 脚 n=0），头无碰撞穿地，z_min **0.012**。与地面布同类：要么打穿，要么后面还得一块硬背板。已撤回，hull 仍在 `jaw_soft`。
  - **jaw 不撞地板 + 首次 hull 穿地时 Generic6DOF 法向弹簧**（层拆分 floor=1 / robot=2 / jaw=4，避免单向 mask 仍生成接触）：弹簧在 t=1.24、tilt 79°、vn≈0.36 接上（MJ 是 t=0.92 / 81° / 低 vn）。k=2500 仍在 20 ms 掀脚，hull 再沉 5–27 mm；k=700 过阻尼仍掀脚，hull 沉 **30 mm** 后髋着地，z_min **0.023**。晚到的高 vn 第一下：硬则掀脚，软则掉髋。已撤回层拆分和关节。
  - **短踵 sibling + 前掌抬 1.5 mm + Generic6DOF 三轴线性弹簧**（替代会被拉开 2.5 mm 的 0-limit hinge）：k=3000 与 k=30000 都在 t=0.12 被冲击压成两岛同时着地。k=3e4 把 t=0.92 tilt 从全垫 31° 拉到 **56°**（MJ 81°），t=1.00 脚尖仍着地，t=1.14 jaw 碰上时脚已 n=0，z_min **0.031**。两端 t=0.80 的关节 |Δq| 只有 0.11，差的是 4 点脚掌没让整机仰过去。已撤回，植物仍是全垫 16 边形。
  - **MJ 第一下两点（外侧脊 x≈−0.009/+0.022）做成 sibling 球 + 6DOF 线性弹簧**：t=0.12 接触点对齐 MJ。球-地仍是硬约束，串联弹簧挡不住弹跳（k=2e3/5e3、mass 0.002/0.015 时 t=0.40 vz>+0.7），t=0.80 已髋+头，z_min **0.030**，jaw x 只有 +0.09。刚性两点 31 mm 线和软点球都会锁/弹，差的是 MJ 的 solref 穿透，不是点数。已撤回。
  - **16 mm 踵 sibling + 3 mm 前掌抬 + 6DOF 线性弹簧（k=1e5）**：冲击后 t=0.40 只剩踵（错高保住了）。t=0.92 tilt **58°**（全垫 31°，1.5 mm/k=3e4 时 56°，MJ 81°），t=1.00 已 77° / vz≈−0.34，t=1.14 jaw n=6 掀脚，髋着地，z_min **0.0315**。c=2000 几乎不变。短垫能剥，但碰头时 vn 更高。已撤回。
  - **独立弹簧地砖（把地板 solref 离散化）**：关掉静盒，16×16×30 mm 砖（k=2500/1.2e4，mass 0.08/0.5）。第一下整块冲量打在 1–2 块砖上，t=0.12 砖沉 **13–15 mm**（当时 6DOF 锚在原点偏移处，出力为零）。单块 0.5 m / 2 kg 垫在负载下弹簧留不住。**COM 对齐重砖网格**（每块砖自己的锚，弹簧是真的）：k=2000 / mass=1.5 整机跟着砖掉到 −0.6 m；k=20000 / mass=0.4 时 t=0.12 仍沉 **15.5 mm**（与 k 无关），随后回弹把机器人弹到 **z=2.5 m**。一块大板能静沉 4 mm，是因为冲量由同一块质量承担；网格第一下集中在 1–2 块。系列质量-弹簧改接触质量，约束仍是硬的。已撤回静盒。
  - **32 mm 矢状裁剪**（对齐 MJ 第一下 x∈[−0.009,+0.023]）：t=0.92 仍是 4 点面 / tilt **29°**（全垫 31°）。t=1.24 短暂脚尖+jaw 三脚架，20 ms 后掀脚，髋着地，z_min **0.0312**。32 mm 仍是硬长垫，不是 16 mm 与 48 mm 之间的可用插值。
  - **xmin=-0.006 2.5 s 复核**：z_min **0.0464** 但是**后仰**（t=0.92 tilt 仅 1.5°，jaw x **−0.25**）。旧记录 0.050 不是 MJ 那种前倾头+脚。已改回全垫。
  - **`body_pair_contact_cache_enabled=false`**：t=0.92 tilt **30.9°**（全垫 30.8°），t=1.26 仍掀脚，z_min **0.0312**。关掉接触缓存不改 4 点面。已改回默认。
  - **脚底 SoftBody 橡胶垫**（顶面 pin 到踝、关掉刚体 16 边形，stiffness=0.7）：运行时 `SoftBody3D.new()` 进不了 Jolt physics space。放进 `main.tscn` 后 pin 成功，但垫打穿——t=0.12 vz=−1.19（全垫 −0.65），t=1.14 已髋+jaw，z_min **0.031**。与地砖/布同类：Jolt SoftBody 映射不成 `solref=[0.02,1]`。已撤回。
  - **单块整板 Y 弹簧地**（关掉静盒碰撞，一块 4 m RigidBody + Generic6DOF，`gravity_scale=0`，mass=0.15，k_y=2000）。弹簧本身是真的：`godot/spikes/spring_y_test.gd` 里 1 kg / k=2000 静沉 **4.9 mm**（=mg/k）。C3 上板静沉 ~4 mm，t=0.92 仍是 4 点面 / tilt **32.4°**（全垫 30.8°）。t=1.26 jaw 那一帧 vz≈0（静地板是 +0.26 掀脚），但脚仍离，t=1.42 髋着地，z_min **0.0266**。共享 Y 柔度不解锁脚-板俯仰。
  - **整板俯仰铰**（平移 k=1e5 锁死，`angular_spring_z=0.25`，自定义 I_z=0.008）：板会仰（t=1.24 板 pitch **34°**），但脚没有跟着贴面，机器人从倾斜板穿下去，z_min **−8**。世界铰链代替不了接触剥离；步态上 CoP 力臂随 xy 增长，更不能用。已撤回静盒。
  - **共面踵/尖两片叶子**（host 无地面碰撞，n_host=0，各一片 16 边形，split x=0.008，6DOF k_y=900 / k_pitch=1，mass=0.02）：第一下弹簧压进 ~3 mm，但 t=0.80 两片仍是共面 4+4 点 / tilt **22°** / 踝 z=0.010。t=0.92 踵卸载、tilt 仍 **31°**，剩下 ~21 mm 脚尖面；t=1.26 jaw 掀脚，z_min **0.0311** 髋着地。两块硬共面 + Y 弹簧在俯仰上等价刚体全垫。已撤回。
  - **6 颗独立 Y 弹簧球**（3×2 点在 2 mm 包络上，host 不碰地，k_y=180 / k_xz=2000，mass=0.01）：点接触会按踵→中→尖剥，t=1.26 短暂 **jaw+脚尖** 三脚架，但 t=0.92 tilt 只有 **27°**（全垫 31°，MJ 81°）。剩下 4 个点仍是支撑多边形，差分下沉 <0.5 mm，买不到 MJ 第一下 4.2 mm 的早期剥离。z_min **0.0318** 髋着地。已撤回。
  - **两根独立横向 capsule 轨**（踵/尖各一条，host 不碰地，6DOF 锚在 capsule 上而不是踝 COM，k_y=250，世界 Z 俯仰解开）：t=0.12 两轨都着地（x≈−0.016/+0.030）。解开俯仰就是活板门——踝 COM 穿到 z≈−0.02，轨还在接触。剥离方向是**后仰**（t=1.14 脚尖卸载，t=0.80 起 jaw x<0）。t=0.92 tilt **7.5°**。z_min **0.0454**（t=2.28）是后脑勺（jaw x **−0.24**），与 xmin=-0.006 同类假齐。锁俯仰则回到支撑多边形；解开俯仰则穿地。脚上 6DOF 映射不成 `solref`。已撤回。
  - **场景 SoftBody 地板**（连续介质，区别于刚性砖）。`spikes/soft_floor_test.tscn`：Jolt **全 pin = 无 Rigid 接触**（12 cm 板也自由落体）；只 pin 网格时 12×12 cm 板能坐下（默认 point radius 1 cm）。脚大小 45×20 mm 板：16 mm 网格 / r=20 mm / 轴锁 / 只 pin 边框 / BoxMesh 气垫（pressure=40）都是**先碰到再打穿或弹飞**。整板 Y 弹簧已证明共享柔度不解锁俯仰；SoftBody 在脚尺度上还保不住 4 mm 差分。未接入 `main.tscn`。
  - **HeightMap 蛋盒地板**（A=1.2 mm，λ=30 mm，1 cm 格，打破共面；不是挤出正弦所以不是俯仰圆柱）。spike 里 45×20 mm 板能坐（y≈0.0107）。C3：站距 ~0.10 m 让一只脚在峰、一只脚在谷，谷脚从薄三角面穿过去（右踝 z≈**−0.016**）。t=0.12 已只有左脚 n=24；t=0.80 tilt **85.7°**、jaw n=7；t=0.92 髋着地，jaw x 只有 **+0.03**（MJ +0.24）。z_min **0.0354** 是单脚活板门，不是头+双脚。已撤回静盒。
  - **同体 16 mm 踵 + 22 mm 间隙 + 10 mm 脚尖抬 1.5 mm**：t=0.12 右脚只有踵（span 14 mm）；t=0.92 tilt **48°**（全垫 31°，MJ 81°）、只剩脚尖 span 4 mm、jaw x **+0.17**、脚仍着地。t=1.14 脚已离、jaw x +0.23；t=1.26 髋着地，z_min **0.0311**。剩下的 10 mm 面撑不到 81° 还接着头。**sibling 小脚尖 + 每帧焊复位**：t=0.80 已只剩脚尖 / 58°；t=0.92 **83° 脚已 n=0**，随后髋着地，z_min **0.0304**。短剩余面剥得太快。已改回全垫 16 边形。
  - **大平面 + 5 mm SoftBody**（`spikes/soft_floor_fine_test.tscn`，0.6 m、14884 点）。12 cm 板能坐是因为 ~4 cm 格 + 8 cm pin 对 12 cm 物体够密。45×20 mm 脚：8 cm 点阵 / 20 mm 点阵 / 40 mm 线网 / 把 total_mass 加到 400 kg 都是先碰到再打穿或弹飞。Jolt SoftBody 不能收到脚尺度还保接触。未接入 `main.tscn`。
  - **6 条同体 5 mm 梳状板**（3 mm 缝）。包络窗裁会变成线并回退全垫。世界 XY 矩形（n_host=6）t=0.12 只踩脚尖，jaw x<0，z_min **0.0386** 是后脑勺。**脚掌平面 4 条**（~8 mm / 4 mm 缝，n_host=4）：t=0.12 是踵条（cx −0.017/−0.009），jaw x>0；中间条把 t=0.92 锁在 **33°**（空档 2 岛才 48°）；t=1.24 jaw 掀脚，z_min **0.0317**。已改回全垫 16 边形。
  - **4.2 mm 抛物线摇杆**（矢高对齐 MJ 第一下 solref 穿透）。**中掌顶点**：t=0.12 中掌线（cx 0.006），往后滚到踵，jaw x<0，z_min **0.0470** 是后脑勺。**踵顶点**：t=0.12 踵（cx −0.017/−0.012）；t=0.40 tilt **10.8° vs MJ 9.5°**；t=0.80 **49.2° vs MJ 46°** 且只剩脚尖；t=0.92 **74.7° vs MJ 81°** 脚尖棱还在（span 0.4 mm）、jaw x +0.23。t=1.00 jaw n=4 掀脚，t=1.14 髋着地，z_min **0.0316**。前倾时间线是目前最接近 MJ 的一次，硬脚尖棱仍接不住 jaw。
  - **踵摇杆 + 同体第二块前缘**（n_host=2；并进同一 hull 会填平凸轮，左脚 t=0.40 就踩唇）。**4 mm 前伸板**：t=0.80 已全在唇上（csh=1），t=0.92 仅 **70° / 0.5 mm 棱**。**2 mm 竖直墙**：躯干 74° ≠ 脚俯仰（踝只升 ~12 mm → ~25°），墙仍与地成 65°，t=0.92 仍是 0.7 mm 棱。**30° 三角楔**：t=0.80 垫+楔混合；t=0.92 已滚到前尖（cx 0.047），jaw 冲量 **0.114**（摇杆-only 的约 10 倍）掀得更狠，z_min **0.0308**。**R=12 mm 十二面四分之一圆**：t=0.80 span 14 mm；t=0.92 仍收成 0.5 mm 棱（cx 0.041），jaw 冲量 **0.116**，z_min **0.0310**。多出来的前缘几何让躯干仰得更晚，jaw 打得更重；Jolt 凸包对盒子在该脚俯仰下仍退化成线。
  - **踵摇杆 + 整板 Y 弹簧地**（k=2000，mass=0.15，6DOF 锚在重合 COM）。凸轮仍剥（t=0.80 55° 只剩脚尖），但板沉 4–8 mm，jaw 在 **t=0.92** 碰上（n=9，冲量 **0.242**，摇杆-only 约 0.012）。t=1.00 脚已离，髋砸在下沉的板上，z_min **0.0264**（zrel 0.032）。早剥离 + 会动的地板让头更早、更重地撞上。
  - **同脚四角单边 Y 弹簧**（脚-地硬碰撞排除；6DOF 锚在静地板上、偏移 COM 有效：`spikes/spring_y_offset_test.tscn` 1 kg / k=2000 静沉 4.9 mm）。8 角 k=250：t=0.12 全压 **8 mm**；t=0.92 仍 **28° / 6 角**（均匀下沉，不是 MJ 那种早剥）；t=1.00 收到 4 角后才加速到 t=1.14 **65°**。无 Coulomb，jaw x 只有 **+0.10**（MJ +0.24）。t=1.20 jaw 冲量 0.17 把剩下的弹簧卸掉，髋着地，z_min **0.0305**。Jolt 6DOF 在 k=0 时 XZ 阻尼不出力。四角弹簧仍是支撑多边形。已改回硬 16 边形 + 静盒。
  - **同脚两点外侧脊单边 Y 弹簧**（MJ C3 第一下矢状对，k=500×4，脚-地硬碰撞排除）。t=0.12 沉 **7.7 mm** / 4 开；t=0.92 仍 **4 开 / 32° / 踝 z 0.000**（脚贴地不仰，躯干靠关节折叠，接触点滑到 cx −0.07）；t=1.00 才卸踵；t=1.20 jaw+两只脚尖弹簧随后被掀，z_min **0.0309**，jaw x **+0.09**。矢状两点 Y 弹簧既抵抗俯仰又没有 Coulomb，变不成脚掌剥离。已撤回。
  - **踵凸轮 + jaw 迁移单边 Y 弹簧**（脚仍是硬凸轮 μ=1；jaw-floor 硬碰撞排除，k=2000，每帧跟最低 hull 顶点）。凸轮剥离时间线仍在：t=0.92 **74.7° / 0.4 mm 脚尖 / jaw x +0.23**，此时 jpad 还有 **+24 mm**。没有三脚架窗口：t=1.00 脚已 n=0、tilt 87°，弹簧才第一次压进 8.8 mm（vn≈0.40）；随后头穿到 jpad **−47 mm**，t=1.14 髋着地，z_min **0.0274**。凸轮到不了 MJ 那种 81°+脚+头重叠；头弹簧接不住已经滚过 0.4 mm 棱的机体。已改回平 16 边形 + 静盒。
  - **踵切圆弧摇杆 R=90 mm**（n_outline=32，φ≈32° / 趾抬高 ~14 mm）。t=0.12 只剩踵（span 1.5 mm）；t=0.40 已 **15.4°**（4.2 mm 抛物线 10.8°）；t=0.80 **86° / jaw n=10 / 脚已 n=0** / 冲量 0.040；z_min **0.0308**（t=0.96 髋）。更长滚程是更高的香蕉，不是更长的三脚架窗口；单段踝切圆不能同时做到 4.2 mm 那么温和又有 32° 弧。已撤回。
  2 mm STAND 带底面沿矢状各 4 mm bin 的最低点起伏只有 **0.00–0.16 mm**（< slop 0.20 mm）。单凸包放大外侧脊超过 slop = 硬两点线，已排除。

  同 RigidBody 再加任何 CollisionShape（即使从不着地）会让前掌 4 点在 t=0.40 无法收到后沿，C3 回到髋着地。三块错高条能按 cshape 剥开，但最低条若是长鞋跟则 C3 太稳、run 摔倒；最低条若是剪过的前掌则与单独 xmin=-0.006 一样偏航。单凸包脚尖 +3 mm 能让 t=1.26 仍咬住，jaw 冲击后仍掉髋，且 run vx 从 0.24 降到 0.16。`xmin=-0.006` 2.5 s z_min **0.0464 是后仰**（jaw x −0.25），不是 MJ 前倾。圆柱摇杆会让 HOME PD **站住**（z_min~0.11）。稀疏 sphere 能把 C3 拉到 0.046 但 run 腰斩。Jolt 动态 body 上的 `ConcavePolygonShape3D` 等价整脚凸包，`local_ppo` 会倒。 |
- 摩擦组合规则（MuJoCo max vs Godot-Jolt min）；两端 μ 都设成 1
- BAM 电压模型；推理侧与 `infer_policy.py` 一样用 XML PD
- Jolt 无关节空间 armature 槽；`diag(A nnᵀ)+cond floor` 是稳定近似，不是 `A nnᵀ`
- `Generic6DOFJoint3D` 弹簧：线性弹簧在 Godot 4.7.2 Jolt **有效**（spike：1 kg / k=2000 静沉 4.9 mm）。不能用 6DOF 复现 XML `kp=0.55` 铰，保持 hinge + `apply_torque`。整板 Y/俯仰弹簧地、脚上接触点弹簧轨、场景 SoftBody 地板、HeightMap 蛋盒也映射不成脚-地 `solref`（锁俯仰=支撑多边形，解开俯仰=活板门；SoftBody 全 pin 无接触，脚尺度打穿；蛋盒谷脚穿面）。XML 积分器是 **Euler**，C1 稳态差不是隐式/显式问题

Jolt 小尺度（`project.godot`）：`penetration_slop=0.0002`、`speculative_contact_distance=0.0`、`collision_margin_fraction=0.001`、`baumgarte=0.35`、`velocity_steps=32`、`position_steps=2`、`use_enhanced_internal_edge_removal=false`、`allow_sleep=false`。内边消除会把共面脚掌粘成慢前倾；关掉后 run vx 也不再腰斩。`baumgarte=0.5` 会把 xy_rmse 抬过 warn。`position_steps=10` 会弄丢三脚架并摔倒；从 6 降到 2 不改 C3，但 `local_ppo` xy_rmse 从 0.216 降到 **0.093**、run 世界 vx **0.205 vs 0.203**。

Godot stdout 必须写文件而不是 `PIPE`，否则 Jolt 警告填满管道会和 TCP lockstep 死锁。

窗口目视：`./run.sh` 在有 `DISPLAY` 时会试一次非 headless。本机 `DISPLAY=:1` 上 Vulkan/GB10 能起来，但 physics server 等 TCP，12s 后 timeout 并继续。门禁以 `--headless` 为准。

## 自测（`./run.sh`，exit 0）

| 项 | 结果 |
|---|---|
| Godot `--headless --version` | `4.7.2.stable.official.ed1daf0bf` |
| S1 lockstep | ticks=40，dt=0.005，200 Hz |
| S2 惯量 + 层 | ω 保持 (0,5,0)；异层 0 contact，同层 1 |
| S3 单铰 PD | vs MuJoCo RMSE **0.0028 rad**，两端 q_end≈0.37075 |
| obs vs `infer_policy` | max_abs_err=0，一步后 **1.46e-11** |
| Reset 后 Godot `q` vs HOME | max abs **2e-7 rad** |
| HOME-hold 自由落体 | tick 1–8 的 z/vz 对齐到 **1e-8**；两端 tick 9 接触 |
| C1 单关节阶跃（weld/freeze pin） | MuJoCo abs_err **0.0371**；Godot **0.0386**。旧 0.0013 是回写 freejoint 的自由落体假象。其它关节两端都垂 ~0.05 rad（`left_hip_roll` MJ 0.047 / GD 0.052，重力 + kp=0.55）；GD−MJ max Δq **0.009**，不是 Godot 独有 |
| C2 STAND 纯 PD | **两端都倒**（XML kp=0.55 撑不住）。MJ z_final≈0.043；GD≈0.032 |
| C3 跌落 | MJ z_min **0.046**；GD **0.031**（全垫 16 边形 2.5 s 探针 **0.0310**，t=1.26 jaw 掀脚后髋着地，jaw x≈+0.22）。t=0.80 关节 |Δq|=0.11，GD 仍是 4 点整垫 / tilt 22°（MJ 已 46° 只剩脚尖）。**32 mm 矢状裁剪**仍 4 点 / 29°，z_min 0.0312。**xmin=-0.006 的 0.0464 是后仰**（jaw x −0.25），不是 MJ C3。弹簧地砖打穿。整板 Y 弹簧地 t=0.92 仍 32°，z_min 0.027。俯仰铰地板穿地 z_min −8。共面踵/尖叶子 t=0.92 仍 31°，z_min 0.0311。6 弹簧球 t=0.92 仅 27°，z_min 0.0318。两根弹簧 capsule 轨解开俯仰后后仰穿地，z_min 0.0454 是后脑勺。COM 对齐重砖网格：k=2000 跟着砖掉下去；k=20000 弹到 2.5 m。场景 SoftBody 地板：全 pin 无接触；12 cm 板能坐；脚尺度打穿/弹飞。5 mm 密网格大平面同样打穿。HeightMap 蛋盒：谷脚穿面，t=0.80 已 86°，z_min 0.0354 是单脚活板门。同体 16 mm 踵+间隙+抬高小脚尖：t=0.92 48° 后脚离，z_min 0.031。sibling 小脚尖过冲到 83° 脚已离，z_min 0.030。6 条世界 XY 梳状板后仰，z_min 0.0386 是后脑勺。脚掌平面 4 条 t=0.92 仅 33°，z_min 0.0317。4.2 mm 抛物线摇杆：中掌顶点后仰 z_min 0.0470 是后脑勺；踵顶点前倾 t=0.80 49° vs MJ 46°、t=0.92 75° 脚还在，jaw 仍掀脚 z_min 0.0316。踵摇杆+同体前缘（前伸板/竖直墙/30°楔/四分之一圆）t=0.92 仍收成 0.5 mm 棱，jaw 冲量可到 0.116，z_min 0.031。踵摇杆+整板 Y 弹簧地：板沉 8 mm 让 jaw 在 t=0.92 以冲量 0.242 碰上，z_min 0.0264。同脚四角单边 Y 弹簧：t=0.92 仍 28°，jaw x +0.10，z_min 0.0305。同脚两点外侧脊 Y 弹簧：t=0.92 仍 32° 脚贴地，jaw x +0.09，z_min 0.0309。踵凸轮+jaw 迁移 Y 弹簧：t=0.92 仍 75° 脚在，t=1.00 脚已离、头穿 −47 mm，z_min 0.0274。踵切圆弧 R=90 mm：t=0.80 已 86° 脚离，z_min 0.0308。`penetration_slop=0.002` 的 1.5 s z_min 0.048 是假齐 |
| `alpha_walking` | 两端都不倒。MJ xy=1.256 m；GD xy=0.972 m；q_rmse=0.125；xy_rmse=0.261；z_rmse=0.002；HARD FAIL: none |
| `local_ppo` | 两端都不倒。MJ xy=1.814 m；GD xy=1.859 m；q_rmse=0.156；xy_rmse=0.160；cadence 2.67 vs 2.58；HARD FAIL: none |

Lockstep 回归（已修）：曾把两条命令之间的机体 `freeze` 成静体。Jolt 清零速度后，每个 20 ms control step 从静止掉 `½g(Δt)²`。症状：C3 `z_min≈0.113`（其实没倒下）、`alpha_walking` GD xy **0.137 m**、HUD 里看起来「傻站着」。修复：`physics_server.gd` 自旋等待下一条命令，不再 freeze-as-static。`compare_pair` 增加 `godot_xy_stalled` HARD FAIL，避免「不倒 + xy_rmse<2.5」把瘫走判绿。

`local_ppo` 分相位（本次 `./run.sh` 的 `results/*.npz`）：

| 相位 | 量 | MuJoCo | Godot |
|---|---|---|---|
| walk | 机体系 fwd | 0.129 | 0.151 |
| walk | 世界 vx | 0.121 | 0.144 |
| walk | 4 s Δyaw | +31° | +30° |
| run | 机体系 fwd | 0.324 | **0.281** |
| run | 世界 vx | 0.203 | **0.138** |
| run | 4 s Δyaw | +32° | **+29°** |
| cadence | 左膝 Hz | 2.67 | 2.58 |

`alpha_walking` run：世界 vx 0.168 vs 0.136，GD Δyaw **−106° vs −14°**（走得动、航向更散）。

修前 GD 更接近 MJ 的部分数字（alpha xy 1.147 / q_rmse 0.123、local_ppo xy_rmse **0.093** / run vx 0.207）是假象：Jolt Baumgarte 角速度污染了 obs 的 `base_angvel_local`/`qd`，并给 PD 阻尼和 Coulomb 一项常偏摩擦。门禁仍是 `SIM2SIM_RUN: done`。

旧 box+角球：run 世界 vx 0.104 vs 0.203。仅共面 16 边形 + 内边消除：vx 0.146。关掉内边消除、`baumgarte=0.35`、全垫 16 边形后 run vx 不再腰斩；运动学速度修复后 local_ppo run 世界 vx **0.203 vs 0.138**（修前 0.202 vs 0.207）。`armature` cond 从 10 调到 15/30 会倒。圆柱摇杆会让 C3 站住（z_min~0.11），不要用。降 μ / 减 velocity_steps / 减 baumgarte 都不能在 C3≈0.046 和步态之间两头齐。单凸包 16 mm 平底加切角/圆角：浅则接住跌落，陡则走时踩不到脚尖；C3 的 z_min 必须核对 jaw 的 x 符号。交付植物仍是全垫 16 边形 + `position_steps=2`。

产物：`sim2sim/results/`（`calib_*.json`、`*.npz`、`*_compare/report.md`、`metrics.json`、`compare.png`）。

## Alpha baseline（Godot，干净 eval）

`sim2sim-eval-walk` 自比（A=B=`alpha_walking.onnx`），3 seeds × 8 s，运动学速度修复后。这是后续 A/B 的「before / alpha」一侧。修前 turn 上报的「±0.22」是 gyro 偏置，不是真在转。

| 条件 | 结果 |
|---|---|
| idle | mean_wz −0.0000；yaw drift **0.45° / 8 s** |
| walk_015 | mean_vx +0.0000 |
| walk_025 | mean_vx +0.1295 |
| run_040 | mean_vx +0.2175 / mean_wz −0.604 |
| turn_l（cmd +0.8） | mean_wz +0.0003 |
| turn_r（cmd −0.8） | mean_wz −0.0007 |
| back_020 / strafe_l / strafe_r | mean_v ≈ 0 |

CLI 默认是 5 seeds × 10 s；上表是这次 3×8 s 的数。

## 训练循环（Godot/Jolt finetune）

在 Godot/Jolt 上对 `alpha_walking` 做 PPO 微调。Python 仍是唯一控制器；ONNX / rsl_rl actor 只在 Python 里跑。

### 架构（`src/sim2sim/train/`）

| 模块 | 职责 |
|---|---|
| [`vec_env.py`](src/sim2sim/train/vec_env.py) | `GodotVecEnv`：rsl_rl `VecEnv`，N 个 headless lockstep worker，`report: "lite"`；先对全部 worker `send_step`，再用 `select` 按就绪序 `recv` |
| [`rewards.py`](src/sim2sim/train/rewards.py) | 向量化走路奖励。lite step 没有 whole-body angmom，**跳过** `angular_momentum` |
| [`commands.py`](src/sim2sim/train/commands.py) | 13-D twist（`command_13`）；head/body 固定 0 |
| [`reset_poses.py`](src/sim2sim/train/reset_poses.py) | 一个 MuJoCo companion 做 FK：home + yaw 随机 + 关节噪声；Godot `reset` 的 `ctrl` 仍是 HOME |
| [`onnx_import.py`](src/sim2sim/train/onnx_import.py) | 从 MLP ONNX 恢复 rsl_rl actor。goal 要求 max_abs **&lt;1e-5**。代码里 `PARITY_FAIL_ABS` 现为 `2e-4`（本轮改过，不是该 goal 阈值）。近零方差 command 维 clamp `_std>=0` |
| [`runner.py`](src/sim2sim/train/runner.py) | `sim2sim-train`。三选一：`--init-onnx` / `--init-checkpoint` / `--resume`（都不给则用 yaml `init_onnx`）。critic warmup：冻 actor MLP+std；ONNX/checkpoint 初始化还会把 actor `EmpiricalNormalization` 冻死（`until=count`，整段 run 不再更新）。日志：`train.log` / `metrics.jsonl` / tfevents / `params/{env,agent}.yaml` / `params/git.txt` / `params/init_check.json` |
| [`ppo_finetune.py`](src/sim2sim/train/ppo_finetune.py) | rsl_rl PPO 子类，把 mean KL 写入 `loss_dict` |
| [`export.py`](src/sim2sim/train/export.py) | `sim2sim-export`：normalizer fold 进图，schema-2 manifest sidecar |
| [`eval_walk.py`](src/sim2sim/train/eval_walk.py) | `sim2sim-eval-walk` A/B。12 条件：`idle`、`walk_015`、`walk_025`、`run_040`、`back_020`、`strafe_l`/`strafe_r`、`turn_l`/`turn_r`、`walk_turn`、`walk_push`、`game_seq`。主指标 `vel_err_1s_rmse` / `yaw_err_1s_rmse`，另有 `fell`；逐条件 win 计数；verdict `improved` / `regressed` / `mixed`（过半条件更好 **且** `walk_push` 摔倒率不升 → improved）。死区：vel 0.01 m/s 或 5%，yaw-rate 0.02 rad/s 或 5% |
| [`bench.py`](src/sim2sim/train/bench.py) | `sim2sim-bench-godot`：零动作 lite lockstep 吞吐 |
| [`manifest.py`](src/sim2sim/train/manifest.py) | schema-2 sidecar（`obs_len=61`，`action_len=14`） |

### 配置 [`configs/walk_godot.yaml`](configs/walk_godot.yaml)

| 块 | 要点 |
|---|---|
| env | `num_envs: 8`（`--num-envs` 覆盖）、`episode_s: 20`、`headless: true`、`device: cpu`、`recv_timeout_s: 10`、`max_faults_per_step: 8`。yaml `faults_jsonl` 默认 `results/faults.jsonl`；`sim2sim-train` 改写到 run 目录 |
| reset | `yaw_range: ±π`，`joint_noise_rad: 0.05` |
| obs_noise | uniform；gyro 0.03 / grav 0.01 / q 0.001 / qd 0.25 |
| commands | resample 3–8 s；vx ±0.4、vy ±0.3、wz ±1；`standing_frac` 初值 0.02；`turn_in_place_frac: 0.15` |
| curriculum | 线性 `[start, end, start_iter, end_iter]`：`standing_frac` 0.02→0.25 @0–1000；`action_rate_l2` −0.1→−1.0 @0–1000；`head_pose_bias` 0→−3 @300–1000（microduck_rl 阶梯表压到 1000 iter） |
| pushes | 3–6 s 间隔，xy 速度 ≤0.3（`nudge`） |
| rewards | track_lin/ang、upright、air_time（mjlab：当前 air_time ∈ (0.125, 0.3) 时每步每脚 +1）、pose_legs、foot_clearance/swing、action_rate、foot_slip、body_ang_vel、head_pose_*、dof_pos_limits；`scale_by_dt: true` |
| termination | `tilt_deg: 70`，`min_z: 0.055`（与 [`fall.py`](src/sim2sim/fall.py) 相同） |
| ppo | lr 3e-4 adaptive，`desired_kl: 0.015`，`init_std: 0.18`，`critic_warmup_iters: 100`，`unfreeze_learning_rate: 3e-5`（清 actor Adam + 第一个 minibatch lr=0），MLP `[512,256,128]` ELU，obs_normalization |
| train | `samples_per_iter: 2048`，`save_interval: 50`，`max_iterations: 3000`，`log_root: logs/walk_godot` |

相对 microduck_rl 的偏差：reset 关节噪声 **±0.05**（那边是 0）；head/body command 固定 0；无 `angular_momentum`（lite 没有整机角动量）。`standing_frac` / `action_rate_l2` / `head_pose_bias` 已对齐源课程表的起止值，只是把阶梯压成线性并在 iter 1000 到终值（那边是 1500–2000）。

### 入口

| 命令 | 作用 |
|---|---|
| [`scripts/train_walk_godot.sh`](scripts/train_walk_godot.sh) | `exec .venv/bin/sim2sim-train`，SIGINT/SIGTERM 能进 runner 的 checkpoint-on-signal。**不要 kill `uv run` 包装进程**：信号到不了 Python |
| [`scripts/walk_godot_smoke.sh`](scripts/walk_godot_smoke.sh) | 闭环：ONNX 训 3 iter → resume → export → godot runner → play bank load → eval；约 25 s |
| `sim2sim-eval-walk` | A/B（`--a` / `--b`） |
| `sim2sim-bench-godot` | 吞吐 |
| `sim2sim-export` | checkpoint → ONNX + `.manifest.json` |
| `sim2sim-onnx-import` | ONNX → actor `.pt`（可选） |

安装一次：`uv sync --extra train`。之后用 `uv run --no-sync`。裸 `uv sync` 会卸掉 extra。[`run.sh`](run.sh) 用 `uv sync --inexact`，避免把门禁依赖装回去时拆掉 torch/rsl_rl。

### 与 play 共用的语义

同一套 `build_obs` 61-D、同一套 `fallen`（`fall.py`：tilt 70° 或 z<0.055）、同一套 `physics_server.gd` PD。reset = MuJoCo 采样的 home 位姿 + yaw 随机；reset 后 `last_action` 清零。parity：[`tests/test_vec_env_obs_parity.py`](tests/test_vec_env_obs_parity.py)（相对 `build_obs` max_abs 0 / `<1e-9`）。

### 故障处理（相对 play）

worker timeout / crash → 记 `faults.jsonl`、respawn；单步故障数 > `max_faults_per_step` 则 abort。Godot 子进程 `PR_SET_PDEATHSIG=SIGTERM`（父死子死）。**所有 headless** spawn（train / eval / bench / compare，不单是训练）用临时 overlay：`worker_pool/max_threads=1`，进程 pin 到 `nproc−2` 个核之一。play **窗口**（非 headless）不走 overlay、不 pin。

### 吞吐

16 worker、零动作 lite：bench **3200–3700** control steps/s（此前 1328；缩放约 10×/16）。训练 collect 约 **2700 samples/s**（`torch`/`OMP` 线程钉在 2，避免和钉核 Godot 抢；此前 torch 默认 20 线程时 ~1400）。冻住的 alpha + `init_std=0.18` 探索噪声，在训练分布里大约每 2–8 s 摔一次（消融：探索噪声为主，随机 command 其次；obs/reset 路径已 bit-identical）。

### A/B（alpha vs Walk_Godot.onnx）

来源：`sim2sim-eval-walk --seeds 5 --seconds 10 --workers 8`（计划规格），报告 [`results/walk_godot_eval/report.md`](results/walk_godot_eval/report.md)（`results/` gitignore，数字抄在这里）。A = `alpha_walking.onnx`，B = `policies/Walk_Godot.onnx`（walk2，iter 3000，从 `model_700.pt` 续训；`*.onnx` gitignore）。**120/120 未摔倒**。

主指标（1 s 滑动跟踪误差；越小越好）与结算后平均速度：

| 条件 | 主指标 | alpha | Walk_Godot | Δ | wins | B mean_v |
|---|---|---:|---:|---:|---|---|
| idle | yaw_drift_deg | 0.45° | 57.6° | +57.1 | 0/5 | vx −0.034，wz +0.090（仍在迈步） |
| walk_015 | vel_err_1s | 0.150 | **0.106** | −0.044 | 5/5 | vx **+0.044**（alpha 0） |
| walk_025 | vel_err_1s | **0.123** | 0.163 | +0.040 | 0/5 | vx 0.087 vs alpha 0.131 |
| run_040 | vel_err_1s | **0.184** | 0.265 | +0.081 | 0/5 | vx 0.135 vs 0.218；B yaw 稳（wz −0.07 vs alpha **−0.59**，Δyaw −41° vs −321°） |
| back_020 | vel_err_1s | 0.200 | **0.097** | −0.103 | 5/5 | vx **−0.104**（alpha 0） |
| strafe_l | vel_err_1s | 0.177 | **0.159** | −0.018 | 5/5 | |
| strafe_r | vel_err_1s | 0.200 | **0.166** | −0.034 | 5/5 | |
| turn_l | yaw_err_1s | 0.797 | **0.054** | −0.743 | 5/5 | wz **+0.851**（cmd +0.8；alpha 0） |
| turn_r | yaw_err_1s | 0.797 | **0.057** | −0.740 | 5/5 | wz **−0.747**（cmd −0.8；alpha 0） |
| walk_turn | vel_err_1s | **0.101** | 0.141 | +0.040 | 0/5 | B yaw 更好（0.020 vs 0.251） |
| walk_push | fell | 0 | 0 | 0 | tie | 两边都不倒 |
| game_seq | vel_err_1s | **0.172** | 0.189 | +0.017 | 0/5 | B yaw 更好 |

VERDICT: **mixed**（B 主指标 6 胜 / 5 负 / 1 平）。闭环和「确实改善」要分开说：

- 训练闭环已跑通（采样 → 更新 → 存盘 → 续训 → ONNX → play 加载）。
- Godot 上 **yaw 跟踪是实质改善**：alpha 在 ±0.8 转向指令下 wz≈0，B 跟到 ±0.75–0.85；run 不再以 −0.6 rad/s 自旋。walk_015 / 后退 / 侧移，alpha 几乎不动，B 会动。
- 代价：idle 停不住（cadence ~1.8 Hz，10 s 漂 58°）；0.25–0.40 m/s 直线跟踪比 alpha 慢。这是 Godot 物理上的新步态，不是 MuJoCo 轨迹复现。
- 计划「多数条件 B 显著优于 A 且摔倒不升 → 称改善」：**未达到**（6/12 主指标，摔倒持平）。

加载：`sim2sim-play` 默认优先 `Walk_Godot.onnx`（没有则回退 `alpha_walking.onnx`）。`--walking` 仍可强制路径。`sim2sim-export` 默认写 `policies/Walk_Godot.onnx` + sidecar。

训练日志：`logs/walk_godot/2026-09-07_03-03-34_walk2/`（从 `…_walk/model_700.pt` resume；iter 1100 之后 falls≈0，air_time≈0.025，kl_max 全程 <0.1）。

### 其余 8 个技能（Godot continue-train）

对齐需求、已有门禁、导出清单、相对 &lt;1e-5 的实测偏差：见 [HANDOFF.md](HANDOFF.md)。下面只列接线，不当任务质量验收。

8 个 factory ONNX 都是 61-D / 14-D、512-256-128 ELU。play 命令与 `PlayBrain.command_13` 一致：

| 技能 | 配置 | init ONNX | 导出文件 | play 命令 |
|---|---|---|---|---|
| standing | `configs/stand_godot.yaml` | `alpha_stand.onnx` | `Stand_Godot.onnx` | 全 0 |
| sitstand | `configs/sitstand_godot.yaml` | `alpha_sitstand.onnx` | `Sitstand_Godot.onnx` | cmd[0]=sit flag |
| ground_pick | `configs/pick_godot.yaml` | `alpha_ground_pick.onnx` | `GroundPick_Godot.onnx` | `(cos 2πφ, sin 2πφ)` |
| kick_left / right | `configs/kick_*_godot.yaml` | `ball_kick_*.onnx` | `KickLeft/Right_Godot.onnx` | 全 0 |
| roulade | `configs/roulade_godot.yaml` | `roulade.onnx` | `Roulade_Godot.onnx` | 全 0 |
| roller | `configs/roller_godot.yaml` | `roller.onnx` | `Roller_Godot.onnx` | twist；`microduck_roller.json` |
| roller_crouch | `configs/roller_crouch_godot.yaml` | `roller_crouch.onnx` | `RollerCrouch_Godot.onnx` | 全 0（`--roller` 的 standing 槽） |

```bash
./scripts/train_skills_godot.sh
FORCE=1 SKILL=stand ./scripts/train_skills_godot.sh
uv run --no-sync sim2sim-play            # 有 *_Godot.onnx 则优先加载
uv run --no-sync sim2sim-play --roller
```

`sim2sim-eval-skill` 是本轮加的脚本，不是对齐的验收。技能效果对照用 factory ONNX + 上表「仓库已有门禁」/ MuJoCo `infer_policy` / 窗口，见 HANDOFF。

日志：`logs/train_skills_godot.out`。ONNX gitignore。

## 目录

```
sim2sim/
  src/mjcf2godot/      # 转换器
  src/sim2sim/         # backends, obs, policy, runner, calib, compare, train/
  godot/               # Godot 4.7 工程
  robots/microduck.json
  configs/walk_godot.yaml, stand_godot.yaml, sitstand_godot.yaml, pick_godot.yaml,
  configs/kick_left_godot.yaml, kick_right_godot.yaml, roulade_godot.yaml,
  configs/roller_godot.yaml, roller_crouch_godot.yaml
  scripts/train_walk_godot.sh, train_skills_godot.sh, walk_godot_smoke.sh
  HANDOFF.md           # 8 技能 continue-train 对齐需求与交接
  run.sh
```

## 协议（TCP，行分隔 JSON，MuJoCo 系）

`hello` / `reset{bodies,ctrl}` / `step{ctrl,n_substeps,report,timing}` / `pin` / `nudge` / `set_tau_limit` / `close`

`bodies[].pos` 是惯性系 COM（`xipos`），`quat` 是 `ximat`。Python 的 `GodotBackend` 用 spec 里的 `ipos/iquat` 转回 body 系，对齐 `infer_policy` 的 `xquat` + IMU。线协议姿态四元数键是 `base_quat`（wxyz）；`SimState.base_quat_wxyz` 是转完 body 系之后的字段。

`step` 的 `report: "lite"` 只在 `cmd=="step"` 时生效（`reset` 始终 full）。lite 回复（`physics_server.gd` `_send_state`）：

| 键 | 含义 |
|---|---|
| `ok`, `cmd`, `t` | 应答头 |
| `q`, `qd`, `tau` | 执行器序；`qd` 为运动学差分 |
| `base_pos` | 惯性系 COM |
| `base_quat` | 惯性系 wxyz |
| `base_linvel` | Jolt 线速度，MuJoCo 系 |
| `base_angvel_local` | 运动学 ω，体轴 |
| `feet` | `[{name, contact, n_contacts, impulse, pos, linvel}, ...]` |
| `timing` | 可选（请求 `timing: true`）：`phys_usec`, `pd_usec`, `wait_usec`, `wait_iters`, `prev_send_usec`, `json_bytes` |

full 另含 `base_angvel_jolt`（体轴、未滤波的 Jolt ω）、`dbg_ang_world`、`dump`、`axis_dot`、`applied` / `missing` 等。lite **没有** `base_angvel_jolt` / `dump`。

## Kick headless gate（独立子门禁，默认 soft）

Godot `kick_left` / `kick_right` 当前会倒（单足支撑植物发散，known-fail）。**不挂进** `./run.sh` 的 HARD FAIL，避免弄坏 walk `SIM2SIM_RUN` 绿语义。

```bash
cd /path/to/MicroDuck-Godot-Simi2Sim
./run_kick_gate.sh              # soft：记录 KNOWN_FAIL，exit 0
uv run sim2sim-kick-gate        # 同上
uv run sim2sim-kick-gate --mode hard   # 植物修好后才用
uv run python -m sim2sim.contact_timeline --also-right   # 接触时间线诊断（只读）
```

产物：`results/kick_gate/`、`results/skill_repro/contact_timeline/`。单测：`uv run python -m unittest tests.test_kick_gate`。
