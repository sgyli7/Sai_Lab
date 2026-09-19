# 训练与部署物理架构调查：先决定要消除哪一种误差

调查日期：2026-09-11。依据官方论文、版本发布、依赖清单和源码；未安装新框架、未训练、未修改冻结物理。版本日期采用 GitHub `published_at`（UTC），不把网页抓取时间当发布时间。下文“建议”均为工程推断，尚非本机实验结论。

## 结论

**目前不应因 GPU 模拟器更新而立即迁移训练。首先明确产品是否确实需要 Jolt 主导机器人动力学。** 若主要诉求是独立应用和接近原版 MuJoCo 的动作，Godot 负责界面、渲染，内置原生 MuJoCo 负责机器人与可交互世界，是应认真评估的另一条路线。它仍可无 Python、无 TCP。若 Jolt 是长期硬约束，则必须把它当训练的目标环境；在另一个模拟器大量训练后直接迁移，不会自动消除现有域差异。

用户参考的 Unity 项目实际上采用这种呈现与物理分离方式：MuJoCo 3.12 是物理权威，Barracuda 只做 ONNX 推理。其原版动作表现不能作为“已经解决 MuJoCo→游戏物理引擎迁移”的证据。[参考项目说明](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim)、[固定版本控制循环](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoPolicyStepper.cs)、[本地既有审计](research_unity_embedded_20260911.md)

## 已存在的结构差异，不只是增益数值

本地 [physics_server.gd](../godot/physics_server.gd) 的 358–386 行将 MuJoCo 关节空间 armature 近似加入子刚体的对角惯量，并将三个惯量下限限制为最大值的 1/10；这同时影响整条链运动，并非保留原广义质量矩阵。1444–1480 行以显式 PD 扭矩、饱和、阻尼和 `tanh` 摩擦替代原执行器，被动轮另有处理。这些是**实际代码中的已知近似**，不是本次证明的失败主因；应先用自由空间关节响应和无控制滑行定位其贡献。

接触也不具有参数的一一映射。MuJoCo 支持软接触、不同摩擦锥、切向／扭转／滚动摩擦；Jolt 默认按单个滑动摩擦系数组合接触。不能通过复制一个 `friction` 数字宣称等价。[MuJoCo 接触模型](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact)、[Jolt 材料与惯量](https://github.com/jrouwe/JoltPhysics/blob/master/Docs/Architecture.md#friction-and-restitution)

Godot 还明确说明：内置 Jolt 不支持部分关节柔性／ERP 属性；`get_contact_impulse()` 是预估值，多接触情况下不保证准确。因此本地接触数量可用于接触判定，但不能未经验证将该冲量当成 MuJoCo 求解后的接触力标签。脚本 1556、1569、1689 行确实读取此 API；是否影响训练由具体奖励消费者决定。[Godot 官方限制](https://docs.godotengine.org/en/stable/tutorials/physics/using_jolt_physics.html#contact-impulses)

## 截止调查时可用的栈

以下版本分别核验，不表示所有最新版可组成同一个兼容环境；尤其 Newton 与 Isaac Lab 仍需按所选发布的依赖锁定组合。

| 栈与已核版本 | 对本项目的用途 | 关键边界 |
|---|---|---|
| MuJoCo CPU **3.13.0，09-09**；现有基线为 3.12.0 | 原版参照、CPU 多环境、未来原生部署 | 官方发布 Linux aarch64 原生包。3.13 新增 `discrete` 积分器，升级会引入新动力学变量，参照实验先锁 3.12。 |
| MuJoCo Warp **3.13.0，09-09** | NVIDIA GPU 批量采样，可直连 PyTorch | 追求批量吞吐，不适合据此推断单机器人低延迟；不是自动获得可微训练。 |
| mjlab **1.6.0，08-09** | PyTorch/RSL-RL、速度跟踪与动作模仿，最接近现有训练代码 | **该发布锁 MuJoCo/MJWarp `~=3.11.0`**，不能直接与最新 3.13 或现有 3.12 混装。 |
| MuJoCo Playground **0.2.0，03-16** | MJX-JAX/MJX-Warp 训练；直接复用对应上游任务时有优势 | 需对齐 JAX、Brax、模型特性；自身 CUDA extra 仍指向 CUDA12，不能作为 GB10 配置保证。 |
| Newton **1.6.0，09-10** + Isaac Lab **3.0.0-beta2.patch1，07-02** | 多后端、传感／复杂场景／柔性物体生态 | Newton 是框架，必须明确选 MJWarp、Kamino 等具体 solver；Isaac Lab 3.0 仍是 beta，现已支持无需 Isaac Sim 的 kit-less Newton 路径。 |

版本及依赖依据：[MuJoCo 发布](https://github.com/google-deepmind/mujoco/releases/tag/3.13.0)、[MJWarp 发布](https://github.com/google-deepmind/mujoco_warp/releases/tag/v3.13.0)、[mjlab 发布依赖](https://github.com/mujocolab/mjlab/blob/v1.6.0/pyproject.toml)、[Playground 发布依赖](https://github.com/google-deepmind/mujoco_playground/blob/v0.2.0/pyproject.toml)、[Newton 发布](https://github.com/newton-physics/newton/releases/tag/v1.6.0)、[Isaac Lab 发布](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2.patch1)。MuJoCo 3.13 变更标题标注 09-08，实际 GitHub 发布时间为 09-09；两者不要混淆。

MJX-JAX 与 MJWarp 都支持自由根、转动关节，故轮子不因“无限转动”就不可表达；但**能加载轮子不等于轮地接触等价**。MJWarp 覆盖更多 MuJoCo 特性，仍有求解器例外及部分接触点数差异；MJX-JAX 的 cylinder／mesh 等碰撞组合受限。迁移前必须逐项核对本机器人实际几何、`condim`、摩擦损失、积分器、传感器，以及批量 contact/constraint 缓冲容量；溢出不得忽略。[特性对照](https://github.com/google-deepmind/mujoco/blob/3.13.0/doc/mjx.rst)、[MJWarp 使用与限制](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html)

mjlab 原生暴露 MuJoCo 状态并提供 manager 式命令、奖励、事件以及电机模型，适合作为训练基础设施；论文明确将通用跨模拟器可移植性列为非目标。其导出器可导出 CPU ONNX，部署端不需要跟着安装训练框架；但新观测、历史状态和归一化仍要重新定义本项目契约。[mjlab 论文，2026-01](https://arxiv.org/abs/2601.22074)、[1.6.0 导出源码](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/rl/runner.py)

## GB10 上“可以安装”和“已经验证”要分开

Warp 1.17.0 官方有 Linux aarch64 的 CUDA12／13 wheel；JAX 官方支持 Linux aarch64 NVIDIA GPU。这提供底层可行性，但 mjlab 1.6.0 的 `required-environments` 只列 Linux x86_64 和 macOS arm64，**本次没有取得它在 GB10 上完成 MicroDuck 训练的证明**。Isaac Lab 当前安装文档明确讨论 Spark，并要求相应 CUDA13/PyTorch 构建，列出 ARM 扩展源码编译等限制；不能因本项目旧 Torch CUDA12.9 能用就认定整个新栈兼容。[Warp 发布文件](https://github.com/NVIDIA/warp/releases/tag/v1.17.0)、[JAX 支持表](https://docs.jax.dev/en/latest/installation.html#supported-platforms)、[Isaac Lab 安装要求](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/setup/installation/index.html)

GB10 的 128GB 为 CPU/GPU 共享内存，官方带宽 273GB/s。容量充足不代表能复现 RTX PRO 6000 上的采样吞吐；统一内存也不会消除 Python/TCP 序列化、调度和同步。候选栈只应先做隔离环境的模型装载、有限步测试和 16/64/256/1024 环境吞吐曲线，分开记编译时间、有效控制步、内存、接触溢出、CPU↔GPU 同步；本次未运行这些测试。[硬件规格](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)

保留 Jolt 时优先复用现有 headless 多进程训练；Jolt 本身是 CPU 多线程物理库。只有实测采样被 Godot 场景／协议开销主导，才值得考虑 C++ 批量接口；单独重写裸 Jolt 环境又会产生“裸 Jolt↔Godot Jolt”的校验成本。[Jolt 架构](https://github.com/jrouwe/JoltPhysics/blob/master/Docs/Architecture.md#multi-threaded-access)

## 工业案例能教什么，不能证明什么

2026-03 NVIDIA 披露 Skild AI 用 Isaac Lab/Newton 训练 GPU 机架装配，并为其场景选 SDF 与 hydroelastic 接触，绕过 MJWarp 默认接触管线；Samsung/Lightwheel 段落包含未来使用和仿真演示。它说明工业团队会针对接触类型选择、标定模型，**不是所有任务统一换一个更快 solver**，也没有公开足以验证这些产线成功率的完整数据。[NVIDIA 工业案例原文](https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics/)

Isaac Lab 官方给出 Newton↔PhysX 四类足式机器人迁移示例，并专门使用关节顺序映射。这是可复现跨模拟器检验的工程方向，不能外推为本项目 Jolt 轮滑已受支持；更不能以不同硬件、批量和基线得到的数百倍宣传数字预测本机收益。[官方双向迁移](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/sim-to-sim.html)

## 推荐的分叉决策

1. **先查归因，不先训练。** 用全新开发种子，统一命令语义、初态、动作后处理，对照原策略＋MuJoCo、原策略＋Jolt、候选＋Jolt；另测关节阶跃、自由滑行、短推进刹停。原版也失败的场景属于任务／策略能力问题，不能全算 sim2sim 损失。既有 holdout 保持封存。
2. **若原版显著更好，且产品不要求 Jolt 动态交互**：优先小范围验证 Godot＋原生 MuJoCo。沿用 C++ ORT，只增加 MJCF/资产装载、`mj_step`、状态与渲染同步；地面、球等所有影响机器人的碰撞由 MuJoCo 统一处理，避免一边 Jolt 一边 MuJoCo 的双向耦合。工程成本主要在世界表示和资源生命周期，而非推理。[原生仿真 API](https://mujoco.readthedocs.io/en/stable/programming/simulation.html)
3. **若 Jolt 必须保留**：最终选模以 Jolt 实际闭环为准。MuJoCo/MJWarp 可提供教师动作／参考轨迹和廉价预训练，随后需目标域适配或蒸馏；不要把“更快在错误域训练”当作解决接触差距。随机化范围先由两域响应差异支持，不靠放宽物理掩盖失败。
4. **只有确认目标和任务定义后，才选择加速框架。** 现有 PyTorch 路线先评估隔离的 mjlab/MJWarp；若重用 Playground 的具体任务可节省更多重写，则评估 Playground。暂不为单机器人平地九技能引入完整 Isaac Sim；Newton 多物理能力在明确需要柔性物体或复杂交互时再付成本。

许可层面，Jolt 为 MIT；MuJoCo、MJWarp、Warp、mjlab、Newton 为 Apache-2.0，Isaac Lab 的组件许可证需随所用组合保留。物理／训练库许可证不改变本项目机器人模型资产原有条款。无论采用哪个训练框架，最终可选择 Godot/Jolt 或 Godot/原生 MuJoCo 加小策略部署，训练框架不必进入最终包；这不意味着 Newton 的 Python API 本身已成为 Godot 原生扩展。[MuJoCo](https://github.com/google-deepmind/mujoco/blob/3.13.0/LICENSE)、[Jolt](https://github.com/jrouwe/JoltPhysics/blob/master/LICENSE)、[Newton](https://github.com/newton-physics/newton/blob/v1.6.0/LICENSE.md)、[Isaac Lab](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2.patch1/LICENSE)
