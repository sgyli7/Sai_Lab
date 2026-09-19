# 九个候选模型的内嵌部署契约审计

日期：2026-09-11。审计代码固定在 `da5022170a6da02260826627d574a0c56a18f6c7`；权重为冻结的 `results/research_20260910/delivery_v3/models`，逐个核对模型和 manifest 的 SHA-256，均与 [候选包索引](../artifacts/research_20260910.json) 一致。本文只做源码、模型结构和现有 Python ORT 的读取验证，没有修改权重、训练、实现原生插件或运行新的物理评测。

结论：**现有策略算法可以随 ONNX 图迁移到内嵌 ONNX Runtime，不因为调用语言由 Python 改为 C++ 而需要重训。** 现有 Python 本来就调用 ORT CPU；原生 C++ API 可加载同类模型。因此首个验证应保留原始图、精度和控制契约，检验原生执行是否等价。它仍是一项尚待原型验证的部署判断，并不是已完成的 Godot 原生运行证明。[当前策略包装](../src/sim2sim/policy.py)、[ORT C++ 官方文档](https://onnxruntime.ai/docs/get-started/with-cpp.html)

**“独立运行”要搬走整个控制循环，不能只把 ONNX 文件放进 Godot。** 当前技能状态、命令生成、61 维观测、上次动作、动作缩放、重置初始位姿和机器人切换都含 Python 工作；其中首次重置还调用 MuJoCo 做正向运动学。[交互入口](../src/sim2sim/play.py)、[观测构造](../src/sim2sim/obs.py)

## 实物模型清单

全部模型具有单个输入 `obs: float32[1,61]`、单个输出 `actions: float32[1,14]`，固定 batch 为 1，标准 ONNX domain（空 domain，即 `ai.onnx`），opset 18；没有外部权重文件。九图共 **18,784,420 字节，约 17.91 MiB**。这只是 ONNX 文件大小，不含 ORT 库、运行内存或场景资源。以下结果来自本次逐文件 `onnx.load`、checker、类型/形状检查和对应 manifest；文件身份由 [候选包索引](../artifacts/research_20260910.json) 固定。

| 模型 | 字节数 | 节点数 / IR | 计算类型 | 候选来源与命令契约 |
| --- | ---: | --- | --- | --- |
| Stand_Godot | 793,705 | 9 / 8 | FLOAT | 保留工厂站立；13 维命令全零 |
| Walk_Godot | 7,476,774 | 119 / 8 | FLOAT + DOUBLE | 蒸馏与多层适配组合；身体前进、侧移、转向命令；自身处理 idle |
| Sitstand_Godot | 793,881 | 9 / 8 | FLOAT | 保留上一版 Godot 策略；命令首位为坐下标志 |
| GroundPick_Godot | 793,685 | 9 / 8 | FLOAT | 保留工厂策略；4 秒 cos/sin 相位 |
| KickLeft_Godot | 1,117,978 | 51 / 8 | FLOAT | k10 iteration 81 学习残差；命令全零，**没有模型时间输入** |
| KickRight_Godot | 1,244,005 | 88 / 8 | FLOAT | kr06 iteration 43；镜像后的适配左踢之上再学习残差；5 秒进度及技能起始相对朝向 |
| Roulade_Godot | 4,977,022 | 91 / 10 | FLOAT + DOUBLE | 两个神经专家及学习残差；5 秒进度，**没有相对朝向输入** |
| Roller_Godot | 793,685 | 9 / 8 | FLOAT | 保留工厂轮滑；声明推行/滑行/刹车与相对朝向命令，详见后文 |
| RollerCrouch_Godot | 793,685 | 9 / 8 | FLOAT | 保留工厂轮滑下蹲；5 秒 cos/sin 相位 |

INTEGER 张量用于索引，未列作上表的浮点计算类型。当前九个图都不声明 `sim2sim_yaw_memory`；因此可选的 Python `YawDriftMemory` 并不是这批交付物的运行依赖。不能因为源码支持它，就为当前模型额外增加该特征。[可选记忆包装](../src/sim2sim/policy_memory.py)、[元数据读取](../src/sim2sim/policy.py)

算子审计（次数为图节点数；五个 9 节点模型结构相同，权重不同）：

| 图 | 算子及次数 |
| --- | --- |
| Stand、Sitstand、GroundPick、Roller、RollerCrouch | Div×1、Elu×3、Gemm×4、Sub×1 |
| Walk | Add×9、Cast×4、Clip×3、Concat×3、Constant×9、Div×6、Elu×21、Gemm×29、Identity×3、MatMul×3、Max×1、Min×1、Mul×4、Relu×1、Slice×10、Sub×9、Tanh×3 |
| KickLeft | Add×3、Clip×3、Constant×9、Div×4、Elu×9、Gemm×13、Mul×3、Sub×4、Tanh×3 |
| KickRight | Add×4、Clip×5、Constant×21、Div×6、Elu×11、Gather×2、Gemm×16、Identity×2、Mul×10、Slice×1、Sub×6、Tanh×4 |
| Roulade | Add×4、Cast×2、Clip×4、Constant×15、Div×7、Elu×16、Gemm×22、Mul×8、Slice×2、Sub×9、Tanh×2 |

初始化张量类型也已逐个检查：五个普通图各有 FLOAT×10；左踢 FLOAT×34；右踢 FLOAT×46、INT64×2；行走 FLOAT×44、DOUBLE×36、INT64×23；前滚 FLOAT×42、DOUBLE×18、INT64×3。行走和前滚的 DOUBLE 路径实际经过归一化、Gemm、Elu 和两个网络结果相减，不是可忽略的标签。源码解释了这样做是为了避免两个接近的 FLOAT32 网络输出相减时，舍入误差掩盖很小的学习增量。[增量网络及 DOUBLE 原因](../src/sim2sim/research/models.py)

因此不能照搬“改低 opset 声明即可导入”的简化办法，也不能默认换成全 FLOAT32、FLOAT16 或 GPU 执行会保留现有精度。前滚的 **IR 10** 还说明：只说运行时支持 opset 18 不足以判断兼容；官方兼容表把 opset 和 IR 分列。[ORT 兼容文档](https://onnxruntime.ai/docs/reference/compatibility.html)

## 图外必须保留的控制契约

观测的确切布局如下，索引从 0 开始；不是一个不带语义的任意 61 维向量。[build_obs](../src/sim2sim/obs.py)

| 观测索引 | 内容 |
| --- | --- |
| 0–2 | 机身坐标系角速度 |
| 3–5 | 世界负 Z 单位重力经机身姿态逆变换后的方向 |
| 6–19 | 14 个关节角减去精确 home |
| 20–33 | 14 个关节角速度 |
| 34–47 | 上一控制周期的原始策略动作 |
| 48–60 | 当前技能的 13 维命令 |

1. **坐标、符号与传感器定义。** TCP 协议采用 MuJoCo Z-up、四元数 wxyz。Godot 刚体基座使用惯性 COM 坐标系，Python 的 `inertial_to_body` 用 robot spec 的 `ipos/iquat` 恢复机身原点和朝向。关节数组按 actuator 顺序排列；关节角由相对姿态重建，角速度来自姿态差分与时间常数 0.005 秒的 EMA。基座陀螺也使用姿态差分，而不是直接读取 Jolt solver 的 `angular_velocity`。原生实现若省掉任一变换或换回 solver 速度，即使 ORT 输出对齐，也已换了观测分布。[Godot 状态解析](../src/sim2sim/backends/godot_backend.py)、[姿态差分与状态输出](../godot/physics_server.gd)
2. **动作含义与执行器。** 输出是 14 个位置偏移，当前入口执行 `ctrl = home + action * scale`，然后 Godot 在每个物理步重新计算 PD 力矩、限幅和阻尼/摩擦。输出不是力矩，也不是要写给动画骨骼的角度。现有图已包含归一化和学习适配，不应在原生端再归一化一次。[交互控制](../src/sim2sim/play.py)、[_apply_pd](../godot/physics_server.gd)、[组合图导出](../src/sim2sim/research/models.py)
3. **精确配置来源。** `play.py` 从机器人配置读取 home、action_scale、dt、decimation；`PolicyBundle` 虽也读取模型/manifest 的 action_scale，但入口实际用机器人配置的 scale。当前九份 manifest 与两个机器人配置的 scale 都是 1.0，因此没有实际缩放冲突。模型旧元数据中的 home 仅保留三位小数，不可代替机器人配置中更精确的 home。原生资源要明确配置优先级并校验一致性。[策略包装](../src/sim2sim/policy.py)、[带球机器人配置](../robots/microduck_ball.json)、[轮滑机器人配置](../robots/microduck_roller.json)
4. **技能与跨周期状态。** `PlayBrain` 管理输入按下顺序、斜向归一化、加减速斜坡、站走迟滞、坐下/起身、技能互斥及计时。上一动作在普通技能切换时保留，只有机器人重置/异常恢复等路径清零。行走 manifest 的 `use_stand_policy=false` 还决定 idle 继续由行走模型承担；不能拿九个网络分别独立运行后任意切换。[输入与技能脑](../src/sim2sim/play_input.py)、[切换和重置](../src/sim2sim/play.py)
5. **有时间输入不等于循环相位。** 只有右踢与前滚声明 `sim2sim_time_input_s=5`，由 `obs[48]=clip(elapsed/5,0,1)` 输入，技能刚触发的首次动作必须得到 0。右踢另用 `obs[49:51]` 表示相对于技能起始朝向的横向轴 sin/cos；横向轴方案避免前滚式倒置时前向轴朝向歧义。左踢虽然由技能脑运行 5 秒，输入仍全零。捡地/轮滑下蹲则是各自 4/5 秒的 cos/sin 相位。[时间契约](../src/sim2sim/policy_time.py)、[推进和触发顺序](../src/sim2sim/play_input.py)、[入口注入时间与朝向](../src/sim2sim/play.py)
6. **完整启动与机器人切换。** `ensure_godot_scene` 可现场调用 Python 转换 MJCF 与 Godot 导入；`capture_home_poses` 现场启动 MuJoCo、设置 home 后读取各 body 位姿；切换步行/轮滑当前会关闭后端并 `os.execv` 重启 Python。独立包需要预先导出场景/spec、准备可复现的 reset 位姿，并在引擎中完成换机器人与上下文重置。踢球时放球位置也由 Python 根据当前机器人朝向计算。这些均不能由一个“ONNX 推理节点”自动替代。[交互入口相关函数](../src/sim2sim/play.py)

## 200 Hz / 50 Hz 的实际调度

机器人配置是 dt=0.005 秒、decimation=4，策略每 20 毫秒**仿真时间**计算一次，控制目标在中间 4 个物理步内保持。现有 Godot 在 `_physics_process` 入口先更新上一积分步的姿态差分速度；4 步结束后发送状态，并阻塞当前物理 tick 等待 Python 新命令。它不是“Godot 连续向前跑，Python 延迟地回传旧状态的动作”的异步设计。[配置](../robots/microduck_ball.json)、[锁步循环](../godot/physics_server.gd)

`project.godot` 的 `max_physics_steps_per_frame=1` 不是可见窗口的最终设置：`_ready` 将可见模式改为 16、关闭 vsync，`_physics_process` 又设为 32；无头模式为 1。Python 启动器两种模式都会传 `--fixed-fps 200`，交互入口通过 wall-clock 等待控制可见速度。滑条修改等待时长，不修改物理 dt；窗口帧率与策略仿真时间不能混为一谈。[Godot 项目设置](../godot/project.godot)、[运行时覆盖](../godot/physics_server.gd)、[启动参数](../src/sim2sim/godot_proc.py)、[wall_dt](../src/sim2sim/play_input.py)

因此移除 TCP 可减少跨进程封装/传输，并简化打包，但**本次证据不能证明 TCP 是动作质量差的原因**。原生调度应保留“先读取新物理状态，再在控制边界推理，然后连续积分 4 步”的顺序；渲染卡顿、追赶帧和加速模式也要明确边界，不能默认每个渲染帧推一次模型。源码还明确记录了逐次 freeze/unfreeze 会损失 Jolt 接触 warmstart；现有锁步已经避免这种做法，迁移时不应重新引入。[锁步中等待与历史 warmstart 注释](../godot/physics_server.gd)

## 轮滑语义需要澄清，当前不能定为控制错误

交付 manifest 将轮滑声明为 `roller_throttle_heading_error`；当前 play 用通用 `PlayBrain` 的 `self.vel[2]` 直接给模型，UI/变量名称把它称作转向速度，没有目标航向累积或按实时姿态求误差。[play 的 ROLLER_LIMITS 与加载分支](../src/sim2sim/play.py)、[command_13](../src/sim2sim/play_input.py)

但固定上游 `microduck_rl@5946fd9cdbc58956424420153e51975af3b30d77` 的 `RelativeHeadingVelocityCommand` 明确说：推理时允许用户直接给第三维，持续常数输入会产生近似持续转向。所以不能仅凭变量叫 velocity，就断言当前按键路径一定违反源模型用法。训练时实际代码计算的是 `wrap(target_yaw-current_yaw)`；该类更早的 docstring 对符号存在相反的旧说明，执行公式应优先。工厂 ONNX 元数据仅写 `command_names=twist`，也不足以还原训练 checkpoint 的全部命令配置。[上游直接输入说明与实际公式](https://github.com/pollen-robotics/microduck_rl/blob/5946fd9cdbc58956424420153e51975af3b30d77/src/mjlab_microduck/tasks/mdp.py#L4576-L4623)、[上游轮滑命令配置](https://github.com/pollen-robotics/microduck_rl/blob/5946fd9cdbc58956424420153e51975af3b30d77/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py#L540-L553)

本仓库另有 `roller_contract=True` 的研究路径，会显式计算目标减当前朝向；对应单测检查了实际误差随朝向变化。普通 play 的轮滑单测检查推行范围和禁用侧移，没有证明目标航向跟踪；普通 `evaluate.episode` 也未自动启用这一研究路径。这说明**按键持续转向**和**跟踪一个世界目标航向**是两个不同的验收任务，不是已经证明某条必须被修复。[研究轮滑命令](../src/sim2sim/research/roller_tasks.py)、[研究路径](../src/sim2sim/research/world.py)、[评估入口](../src/sim2sim/research/evaluate.py)、[play 单测](../tests/test_play_input.py)、[研究单测](../tests/test_research.py)

迁移基线应保持当前直接输入行为，并把第三维明确称为相对朝向命令，不能直接把数值解释成物理角速度。它不构成阻断迁移的问题。如果产品以后需要“转到并保持一个世界目标航向”，再把目标航向控制作为独立功能设计与验收，不能混入推理框架替换后把行为变化归因给 ORT。上述测试只阅读了源码，本次未重新运行物理试验，也没有确认轮滑质量差的因果来源。

## 已验证与后续等价门槛

本次实际执行了以下只读审计：ONNX 1.22.0 checker 对九个图全部通过；ONNX Runtime 1.29.0、CPUExecutionProvider、intra/inter-op 线程均设为 1，九图各执行 16 条固定随机种子 `20260911` 的合成输入（第一条以零向量为底，其余为正态随机数；随后全部统一重力为 `[0,0,-1]`、命令清零），共 144 次单样本推理，输出均为有限数和 14 维；九图传入 batch=2 均被 `InvalidArgument` 拒绝。模型及 manifest 共 18 个 hash 全部吻合索引。这是加载/结构冒烟检查，**不是 16 条真实运动轨迹，不是 C++ 对 Python parity，也不是性能基准**。

建议原型与实施票据采用以下分层门槛，本文未执行这些新原型：

1. **原生模型执行对齐。** 用同一批哈希锁定的原始九图、CPU EP 和单样本契约，对比 Python 基准与目标原生平台；覆盖随机、真实录制 obs、右踢时间/朝向边界、前滚专家过渡区和非零动作。沿用现有 `<1e-5` 最大绝对动作误差作为初始门槛；失配先查精度/核/优化，不放宽阈值掩盖。当前训练导出使用的就是单样本门槛，batch 训练结果另列。[现有 parity 实现](../src/sim2sim/research/models.py)
2. **控制特征对齐。** 对同一状态快照和输入事件序列，逐项比较 61D obs、所选技能、时间、heading、last_action 与 ctrl；包括切换技能、reset、左右踢、站走 idle、轮滑换机器人。网络相等不能替代这一层。
3. **闭环与导出验收。** 同一 Jolt 参数、初始条件和输入序列下，比较 Python 基准与原生的成功率、轨迹/接触和现有运动质量指标；覆盖不同渲染帧率、卡顿、暂停/重置，并在无 Python、无训练仓库的干净目标机启动导出包。不要承诺跨平台浮点物理逐帧完全一致；但应先定位短程状态差异，再检查长期行为是否回归。
4. **并发与性能。** 当前图没有动态 batch，先不要把多机器人输入拼成 `[N,61]`。可先保持每只机器人的动作/相位状态独立、单样本调用，再评测复用 session 与线程策略。现有 play 直接采用 ORT 默认线程配置；官方文档说明 CPU 默认会使用多个线程及线程池，自旋也影响资源占用。新进程内这些线程会与 Jolt、渲染共享资源，需要量测目标机的推理 P50/P95/P99、控制期限、内存和帧耗时，而不是仅看平均 infer_ms。[当前 session 创建](../src/sim2sim/policy.py)、[ORT 线程管理](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)

保持原始图进行上述验证不需要重新训练。若之后为了移动端、GPU、批量机器人或体积改写 DOUBLE 子图、量化、蒸馏或替换物理引擎，则是另外的模型/控制变更，应重新建立数值和闭环证据；不能把当前 Python ORT 加载通过视为这些路线也已通过。
