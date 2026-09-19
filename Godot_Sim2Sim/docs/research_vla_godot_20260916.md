# VLA 接入 Godot / Sai 抓取链路调研

调研日期：2026-09-16。范围：判断当前项目是否已经使用 VLA，解释现有抓取为何准确，并评估将 VLA 实际接入 Godot 的可行路径。外部事实只引用论文、作者项目页、官方仓库或官方文档；本轮未下载模型、未训练、未修改运行时代码，也没有把候选模型在本机上实际跑通。

## 结论

**当前抓取没有使用 VLA。** 它是“已训练的底盘运动策略 + 确定性的任务状态机/轨迹 + MuJoCo 运动学求解 + Godot/Jolt 近距离物理夹持”的混合控制。目标物位置来自 Godot 场景刚体的精确坐标，而不是由相机图像识别；文字按键提示也不进入模型。它之所以能准确抓，是因为系统已经知道目标三维中心，并把机械臂末端轨迹重定向到该点，只有末端真正靠近或发生接触后才建立辅助约束。[当前抓取控制器](../src/sim2sim/workshop_grab.py)、[Godot 夹持与验收](../godot/hub/scene_grab.gd)、[Sai 的 Godot 桥接](../godot/hub/sai.gd)

**把 VLA 接进本项目可行，但首选架构是独立 GPU 推理服务，不是把完整 3B/7B Transformer 塞进现有 Godot ONNX 扩展。** Godot 负责渲染 RGB、采集机器人状态和推进 2000 Hz Jolt；VLA 服务以约 5–10+ Hz 产生未来动作块；已有 50 Hz 控制层异步消费动作块并继续负责限位、IK、轮式底盘和失联停车。项目已经有 Godot 与 Python 的 TCP 控制边界，因此无需重造整个控制系统，只需增加图像/时间戳/动作块协议和新的策略适配器。[Workshop 运行入口](../src/sim2sim/workshop.py)、[Sai 50 Hz 控制入口](../godot/hub/sai.gd)

当前主机为 Linux aarch64、NVIDIA GB10、约 128 GiB 统一内存。它具备本地 VLA 原型的硬件基础。NVIDIA 对当前 GR00T N1.7 的官方实测表列出 DGX Spark 为 PyTorch eager 7.9 Hz、TensorRT 10.1 Hz；但这只是同类平台的官方结果，**不是本项目已经测得的性能**。[GR00T N1.7 硬件建议](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md)

## 当前方案如何做到准确抓取

当前闭环可以概括为：

1. Godot 的 `scene_grab.gd` 直接读取被选刚体的 `global_transform` 和 `grasp_center`，得到精确三维目标；这属于仿真特权状态，不是视觉估计。
2. `WorkshopController` 先用目标相对底盘的位置做自动靠近，并要求底盘落在约 20–27 cm 的抓取窗口且停稳。
3. 机械臂执行已有 `Task.path()` 的分段轨迹。控制器把原轨迹末端点重定向到目标和车载货仓，使用 MuJoCo Jacobian 做 40 次阻尼位置 IK，并裁剪关节到模型限位。控制器只调用正向运动学/Jacobian，不推进另一套物理；实际物理仍在 Godot/Jolt。
4. Godot 只有在夹爪末端距离物体中心小于 26 mm，或已经接触且小于 60 mm 时，才创建 `PinJoint3D`。因此物体不会从远处瞬移进手中；辅助约束只负责补偿仿真夹爪对不同小物件的接触不稳定。
5. 放置时还要满足目标到货仓释放点小于 25 mm。最终成功由真实刚体边界、支撑接触、是否曾夹住/释放以及抬升高度共同判定。

这套设计的优点是稳定、可解释、容易做物理验收；局限是泛化来自人工写好的几何和轨迹。换物体形状、遮挡、相机视角或未知位置时，它不会像 VLA 那样从像素和语言推断动作。保留 `PinJoint3D` 也意味着当前结果属于“游戏辅助抓取”，不能作为纯接触抓取能力的证明。

## 候选模型的一手资料对比

| 候选 | 模型输入与输出 | 官方速度/硬件信息 | 微调数据与接口 | 许可与判断 |
| --- | --- | --- | --- | --- |
| **GR00T N1.7（同时参考 N1/N1.5）** | RGB 视频序列 `uint8[B,T,H,W,3]`、状态 `float32[B,T,D]`、语言；输出按 embodiment 定义的连续动作字典 `float32[B,T,D]`，单位已经反归一化。基础 N1.7 动作块长 40，可只执行前 8/16 步后重规划。N1/N1.5 使用 Eagle VLM 表征视觉语言、DiT/flow matching 生成连续动作；N1.5 冻结 VLM，并加入 FLARE 未来表征目标。[Policy API](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md)、[N1.5 官方项目页](https://research.nvidia.com/labs/gear/gr00t-n15/) | 当前 N1.7 约 3B；官方最低推理配置为一张 16 GB+ GPU。DGX Spark 实测 7.9 Hz eager / 10.1 Hz TensorRT；动作执行与相机建议约 30 FPS，通过动作块和异步推理解耦。[硬件表](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md)、[部署指南](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/real_world_deployment.md) | 官方 `NEW_EMBODIMENT`、LeRobot v2 风格数据、LIBERO/SimplerEnv/RoboCasa 仿真，以及 ZMQ PolicyServer/Client。指南建议窄任务至少约 100 个有效 episode，30 个可能勉强可用，200+ 更稳；采集 20–50 Hz。[仓库](https://github.com/NVIDIA/Isaac-GR00T)、[部署指南](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/real_world_deployment.md) | 代码 Apache-2.0，权重为 NVIDIA Open Model License。与本机 GB10 和服务化部署最匹配，是较完整的目标方案；仍必须为 Sai 训练新 embodiment，不能期待对陌生机构零样本直接工作。 |
| **OpenVLA-OFT / OFT+** | 可用第三视角图、腕部图、proprio 和语言；并行解码连续动作块，使用 L1 回归，OFT+ 可用 FiLM 加强语言条件。官方 LIBERO 配置为 7D、chunk 8；ALOHA 为三路图像、14D 双臂、chunk 25。[官方仓库](https://github.com/moojink/openvla-oft)、[论文](https://arxiv.org/abs/2502.19645) | 推理约需 15.9–18.0 GB。作者项目的训练使用 8×A100/H100 80 GB、50k–150k 步、约 1–2 天；论文报告相对原 OpenVLA 的动作生成吞吐提高约 26 倍。[项目 FAQ](https://openvla-oft.github.io/) | 官方有 LIBERO 和真实 ALOHA 样例、HTTP 服务。ALOHA 文档以 25 Hz、1 秒动作块为默认；可改 50 Hz / 50 步。自有数据主要走 RLDS。[ALOHA 指南](https://github.com/moojink/openvla-oft/blob/main/ALOHA.md) | 代码 MIT，但它继承 OpenVLA/Llama 2，分发权重时还要核对基础模型条款。它最贴近现有抓放任务，可作强基线；7B 依赖栈和推理成本高于 GR00T N1.7/SmolVLA。 |
| **OpenVLA 7B 原版** | 单张 224×224 RGB + 英文任务；输出单步 7D 末端增量与夹爪，每一维离散成 256 个 action token。预训练使用约 970k 条真实机器人轨迹。[论文](https://arxiv.org/abs/2406.09246)、[官方仓库](https://github.com/openvla/openvla) | 官方仓库提供 bf16/量化加载和 REST 服务。原版是逐 token、单步输出；官方已经建议优先尝试 OFT 以获得动作块和高频控制。 | 支持 RLDS、LoRA、WidowX/Franka 和 LIBERO。官方 LoRA 示例称至少约 27 GB 显存可通过减小 batch 使用；全量微调示例为 8×A100。[微调说明](https://github.com/openvla/openvla#fine-tuning-openvla-via-lora) | 代码 MIT；README 明确说明 OpenVLA 权重基于 Llama 2，受 Llama Community License 约束。适合复现或语义基线，不建议作为本项目首个闭环控制版本。 |
| **Octo Base / Small** | 多 RGB、语言或目标图，可加入 proprio；历史窗口 2，扩散头默认一次预测 4 个连续动作。可替换观察和动作头适配新机构。[官方仓库](https://github.com/octo-models/octo)、[论文](https://arxiv.org/abs/2405.12213) | 官方仓库在 RTX 4090 上列 Base 93M 为 13 inference it/s，Small 27M 为 17 it/s；一次推理可产生 4 步，不能把 `it/s × 4` 直接当闭环重规划频率。 | 提供 Gym wrapper、模拟 ALOHA 与真实 WidowX 示例、三种冻结/微调模式。预训练约 800k 条轨迹。 | 代码与权重 MIT。模型轻，适合较便宜的接口验证；官方 JAX/CUDA 安装说明较旧，当前 aarch64/GB10 需要额外兼容性验证。 |
| **SmolVLA 450M** | 多视角图像、状态、可选语言，flow matching 输出连续动作块。[模型卡](https://huggingface.co/lerobot/smolvla_base) | 约 450M，可在消费级硬件甚至 CPU 上推理；官方异步方案让策略服务提前生成下一动作块，避免动作队列耗尽。[项目介绍](https://huggingface.co/blog/smolvla)、[异步推理文档](https://github.com/huggingface/lerobot/blob/main/docs/source/async.mdx) | LeRobot 直接提供训练、记录、PolicyServer/RobotClient 和 Real-Time Chunking。 | LeRobot 代码 Apache-2.0；具体 checkpoint 分发还应读取其模型卡及上游 VLM 条款。它是最低成本的端到端数据/协议 PoC 候选。 |
| **RT-2** | 相机 + 指令，输出 `terminate + 6DoF 末端增量 + gripper` 的离散 token。训练关键是机器人数据与原始网页数据共同微调。[论文](https://robotics-transformer2.github.io/assets/rt2.pdf) | 55B 模型部署在多 TPU 云服务，论文报告 1–3 Hz；5B 约 5 Hz。 | 没有公开可用的主模型、训练代码或 Godot/ROS 适配器。 | 可作为 VLA 概念和语义泛化参考，无法作为本项目可实施的开源基线。 |

OpenVLA-OFT 和 GR00T 都没有官方 Godot 插件；这不构成根本障碍。两者官方都把“环境/机器人适配器 + 独立策略服务”当正常部署形式：OpenVLA 提供 REST，OpenVLA-OFT 的 ALOHA 示例采用服务端/客户端，GR00T 提供 ZMQ PolicyServer/Client。Godot 只是新的环境客户端。

## 推荐的项目架构

```text
Godot / Jolt 2000 Hz
  ├─ 前视 SubViewport RGB（建议再加腕部 RGB）
  ├─ q / qd / 底盘与末端状态 / gripper
  └─ task text + episode/timestamp
             │  5–10+ Hz，二进制帧或 msgpack
             ▼
Python VLA adapter ─── GPU PolicyServer
             │  future action chunk + generation timestamp
             ▼
50 Hz action queue / RTC
  ├─ 相对末端位姿 + gripper → 限速 → IK → arm targets
  ├─ base twist → 已验证 Sai 运动策略 / wheel controller
  └─ 超时、碰撞、关节限位、人工取消 → stop / safe pose
```

动作空间建议先定义为**语义清晰的低维连续控制**，例如相对末端 `Δxyz + Δrotation + gripper`，必要时增加 `base_vx/base_wz`，再由现有确定性层转换为 22 个关节/轮组目标。直接让 VLA 输出 22 路电机目标会把底盘平衡、机械臂运动学和语义抓取同时压进一个数据问题，所需示范更多，也更难定位故障。

现有 JSON-line 通道不适合直接放图像：当前 Sai 服务对接收缓存设了 1 MiB 上限，base64 还会增加约三分之一体积和额外拷贝。推荐保留控制 JSON 或改成长度前缀的二进制/msgpack；RGB 可传 JPEG/PNG 字节，或让本机共享内存环形缓冲区只传帧序号。GR00T 的 Godot 侧最省事方案是“Godot → 本项目 Python adapter → 官方 ZMQ client”，Godot 无需引入 ZeroMQ 扩展。

动作块必须异步消费。假设 VLA 每 100 ms 重规划一次，Godot 的 50 Hz 控制队列仍能按 20 ms 一步执行；新块到达时用时间戳丢弃过期结果，并在重叠区做短窗融合或 RTC。不能让 2000 Hz 物理线程同步等待 GPU 推理，否则一次抖动就会冻结物理闭环。GR00T 官方明确区分约 10 Hz 的推理频率和约 30 FPS 的动作执行频率；LeRobot 也提供相同目的的异步推理和 RTC。[GR00T 频率说明](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md)、[LeRobot RTC](https://github.com/huggingface/lerobot/blob/main/docs/source/rtc.mdx)

## 建议的实施顺序与验收门槛

### 0. 先证明协议，不先训练

为 Godot 加两路稳定相机和时间同步日志；把现有确定性抓取器的动作录成动作块。用“回放策略”经过同一服务接口重新执行，先证明 RGB/状态/动作尺度、坐标系、重置和异步队列无误。GR00T 官方也建议先用 ReplayPolicy 验证环境，回放失败通常意味着重置、观察预处理或动作空间不一致。[GR00T Policy API](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md)

### 1. 做可归因的 VLA PoC

先选一个窄任务：“抓指定颜色/类型的单个物件并放入蓝色货仓”。用现有稳定控制器和人工纠正采集至少 100 个有效 episode，随机化目标位置、物体组合、背景、光照和相机扰动，并保留独立的场景种子/物体组合作为 holdout。数据至少包含：

- 统一时间戳的前视/腕部 RGB；
- 语言指令，且不同物体的措辞有可控变体；
- 机器人状态、相对末端动作、夹爪动作、底盘指令；
- episode 成败、碰撞、取消和恢复标记。

若优先验证管线，SmolVLA 的训练/服务成本最低；若优先验证当前 GB10 上较完整的目标栈，使用 GR00T N1.7 `NEW_EMBODIMENT`。OpenVLA-OFT 适合做 LIBERO 风格抓放的强对照。

### 2. 分三档移除特权信息

1. **VLA 选目标、现有轨迹执行**：VLA 根据图像/语言输出对象类别或技能参数，保留确定性 IK 和近距离约束。最快看到语言/视觉价值。
2. **VLA 输出末端动作块**：关闭策略输入中的 Godot 目标坐标；保留限速、IK 和近距离 `PinJoint3D`。这能证明视觉闭环动作，但仍属于游戏辅助夹持。
3. **纯接触验收**：移除 `PinJoint3D`，只允许真实双指接触和摩擦完成抓取。只有这一档能声称 VLA 自身完成了物理抓取。

### 3. 必须记录的指标

- 每种物体和每条指令的成功率，训练分布与 holdout 分开；
- 是否实际抬起、真实双指接触时长、最终货仓支撑、掉落/碰撞率；
- 推理 p50/p95/p99、动作队列欠载率、过期动作比例、Godot 帧耗时；
- 图像打乱、语言打乱、关闭特权坐标三项消融，防止模型实际忽略视觉或语言；
- 中途取消、切机器人、重置、推理服务断线时能否在一个控制周期内停止。

## 风险与决策

最大风险不是“Godot 能否调用模型”，而是**数据契约和机构差异**。这些基础模型见过的主要是机械臂/人形双臂，Sai 是带轮腿底盘和 6 轴臂的新机构；官方资料也没有承诺陌生机构零样本可用。必须微调并冻结清晰的 action/state schema。

第二个风险是把辅助夹持误算作 VLA 精度。若模型仍收到 `target_m`，或远距离自动吸附，视觉策略即使失效也可能完成任务。VLA 评测应关闭这些特权输入，并把辅助约束作为单独实验变量。

第三个风险是控制延迟。动作块能掩盖平均延迟，但过长的开环窗口会降低对碰撞、滑动和掉落的响应。首轮可从 50 Hz 执行、8–16 步执行 horizon、约 5–10 Hz 重规划开始，具体值只能由本机闭环基准决定。

综合判断：**项目现在没有 VLA；技术上能够接入。建议先用 SmolVLA 或 ReplayPolicy 打通真实的视觉/动作块管线，再以 GR00T N1.7 作为当前 GB10 上的主候选；OpenVLA-OFT 保留为抓放性能对照。** RT-2 没有可用公开实现，不进入实施路线。达到“关闭 Godot 目标坐标、VLA 由 RGB+语言产生动作、holdout 成功、断线安全”这四项后，才能把功能标记为实际 VLA 控制。
