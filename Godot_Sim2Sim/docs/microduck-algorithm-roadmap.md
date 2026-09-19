# MicroDuck 自动作业算法路线与研究审计

日期：2026-09-19。目标是从 A 点自主发现轻物、靠近、用可动下嘴抓住、起身、带载行走到 B 点、安全放下，并在 Godot 中本地推理。本文是**研究和验收设计**，不是模型训练结果。任何 `_Pickable.onnx` 只有在能追溯到真实优化步骤、独立评估与本地部署时才算新策略。

## 1. 仓库现状：算法尚未具备抓运放能力

- 上游 [`robot_allcollisions.xml`](/home/ethan/Projects/microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml) 的 `jaw_soft` 有 `head_roll`，但下嘴网格没有独立 hinge/actuator；`head_camera` 已在 MJCF 中，Godot 当前 MicroDuck 控制链却没有把相机帧送入策略。MuJoCo 对关节、执行器和接触传感器的定义见[官方 MJCF 参考](https://mujoco.readthedocs.io/en/latest/XMLreference.html)。
- 当前 `GroundPick` 的[奖励](../src/sim2sim/research/rewards.py)只测嘴尖靠地、向下、回站、脚支撑和头碰撞；[成功判据](../src/sim2sim/research/evaluate.py)也不测物体接触、离地、运输或放置。[任务定义](../src/sim2sim/research/tasks.py)是四秒相位动作。其旧模型没有目标位姿、相机、下嘴关节输入，14 维动作不能驱动新关节。
- [`pickable_bundle.py`](../src/sim2sim/pickable_bundle.py)逐字节复制九个 ONNX；哈希相同不代表新训练。现有 Godot 嘴脚本使用 PinJoint 辅助持物，旧按 G 基线探针曾报告未夹住、升高不足；这些是**失败基线**，不能算交付。
- 该机器本地 `nvidia-smi` 当前仅列出 **1 张 NVIDIA GB10**。不能把未核实的第二张 GPU、训练进程、训练步数或性能写成事实。远程主机若可用须单独列出设备、进程和日志。

## 2. 推荐总架构：分层闭环，先物理后视觉

```text
前置相机 RGB(+可获取的深度) + 本体/嘴关节/接触
          ↓ 视觉定位器：目标类别、可抓点、相对位姿与置信度
任务管理器：搜索/对位 → 抓取 → 确认持物 → 负载运输 → 放下/核验
          ↓ 目标与阶段命令
50 Hz 抓取策略（14 身体动作 + 1 嘴动作）或带载行走策略
          ↓
MuJoCo: 主训练与留出评估  →  Godot/Jolt: 本地推理与跨仿真验收
```

层级划分并非把动作写死。抓取策略必须学习根据目标相对位置和嘴接触反馈修正下蹲、开合、抬升；运输策略必须根据负载扰动稳定行走。任务管理器只处理目标选取、失败重试、导航目标与阶段切换，并用物理事件触发，不用纯时间脚本强行宣称抓取成功。工业级多阶段 pick/place 方案常把 approach、grasp、lift、transport、release 分为有明确成功条件的阶段；[MoveIt Task Constructor 官方教程](https://github.com/moveit/moveit2_tutorials/blob/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.rst)展示这种组合思想。[Isaac Lab 官方任务教程](https://isaac-sim.github.io/IsaacLab/develop/source/setup/tutorial.html)给出更贴近本题的 `home → pregrasp → grasp → lift → reorient → transport → insert → release` 物理证据课程、相机 actor 与特权 critic。这里借鉴的是方法，不是直接移植其 SO-101 策略。

### 两个有区别的物理目标

1. **纯接触夹持**：上下嘴碰撞面与摩擦真实承重，不加约束。优点是物理解释直接；缺点是单侧活动下嘴、微小碰撞面和不同形状物体可能不可达或滑脱。必须先做静态可夹域、力矩与摩擦敏感性测量。[MuJoCo 接触模型](https://mujoco.readthedocs.io/en/latest/computation/)支持测真实接触与力；其[接触参数规则](https://mujoco.readthedocs.io/en/latest/modeling.html)说明摩擦和求解参数会影响结果。
2. **Sai 式接触后辅助连接**：允许约束协助承重，但只在嘴合拢、实际接触、抓点在可达域内时建立，并在释放、超载或安全异常时解除。MuJoCo 可用运行时切换的[`equality/connect`](https://mujoco.readthedocs.io/en/latest/XMLreference.html#equality-connect)；Godot 可用 [PinJoint3D](https://docs.godotengine.org/en/stable/classes/class_pinjoint3d.html)。两边必须用相同触发/断开规则，不得隔空吸附或先瞬移目标。按质量和约束力/冲量设断开阈值，并保留连接前后 site gap 与对象位移；无限强的连接会掩盖夹持失败。MuJoCo 另有[接触驱动 adhesion actuator](https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator-adhesion)；它与点关节的受力语义不同，不能混用后假设跨引擎等价。

用户允许“像 SaiRobot 一样”吊起轻物，故辅助连接是可比较的工程分支；报告必须标明采用哪一种物理目标，分别测试，不把辅助连接结果称为纯夹力抓取。

## 3. MuJoCo 主实验：可复现的研究顺序

| 阶段 | 实验与冻结的证据 | 不满足时处理 |
| --- | --- | --- |
| 机构与可达性 | 复制原 MJCF 派生 pickable 模型；将下嘴几何移到独立轻质 body；测量 hinge 轴/限位/质量/碰撞、actuator 力矩；中性姿态轮廓与原模型重合；对球、瓶、箱扫描可达夹持位姿和净空。 | 若目标尺寸超开口或嘴撞地，先修几何/目标集合；不能靠奖励优化解决机构不可能性。 |
| 状态抓取 teacher | 从原 `GroundPick` 身体动作初始化**可训练**权重，新增目标/嘴/接触观测与第 15 个嘴动作；目标位置、速度、形状、质量随机化，保持原 14 关节次序；critic 可见更完整状态；保存梯度步、优化器、随机种子、环境哈希和检查点。 | 若无接触或抬升，把 `reach→bite→lift→stand` 分层统计，调整采样与物理，禁止用嘴尖触地奖励代替目标状态。 |
| 长程任务 | 从已验证持物状态开始训练/评估带载步行与放置，再将 reset 课程扩展到原始地面状态；A→B 至少有转向、速度变化与避障留出组。阶段切换只依据接触、物体高度/相对位姿、稳定时间。 | 若旧 `Walk_Godot.onnx` 带载不稳，训练载荷条件残差或独立运输策略；不能只替换 `GroundPick` 后假设八模型兼容。 |
| 视觉替代 | 在训练期用物体真值完成控制上限；然后逐步换成前置相机推断的目标相对位姿与置信度，加入遮挡、像素延迟、光照、纹理、相机外参误差；teacher 仍可给 critic 特权状态。 | 若视觉学生失败，区分定位误差、控制误差、遮挡失效，不在评估时回退读取场景全知坐标。 |
| 跨仿真 | 固定已训练模型/策略逻辑，按同一初始条件比较 MuJoCo 和 Godot 的接触、嘴角、对象轨迹、掉落、关节/力矩；只针对测出的差异随机化或校准。 | Godot 视觉外观通过不代表动力学通过；单一演示片段不代表总体成功。 |

**观测合同**应先从部署时真实可取得的变量定义，而非随意指定 76 维：原 61D 本体观测、目标抓点在头/底盘坐标系的 3D 相对位置和方向、目标尺寸/类别、估计速度与置信度、嘴角和角速度、接触/夹持状态、阶段与 B 目标。质量若 Godot 不会提供给真实 actor，就只给 critic 或使用估计值/载荷响应。时戳、单位、坐标变换、缺失目标掩码、归一化、图像分辨率必须版本化。抓取 actor 输出身体 14 维及嘴动作；导航/运输 actor 可沿用 14 维，只在带载留出集通过时。训练时把前一策略的输出当初始化并产生**新的梯度更新和新权重**，不等于复制原 ONNX。

**课程与随机化**：从居中单物、固定质量接触开始；接着变相对位置、朝向、球/瓶/箱、质量、质心、尺寸、表面摩擦、接触求解参数、地面摩擦、执行器延迟、感知误差；保留交叉组合 holdout。MuJoCo Playground 的[官方库](https://github.com/google-deepmind/mujoco_playground)提供 MJX/Warp GPU 学习、操纵与视觉示例；[官方随机化代码](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/berkeley_humanoid/randomize.py)具体展示质量/摩擦等采样。迁移到 MJX/Warp 前先跑本项目带嘴、接触/约束、图像任务的语义和速度对照；GPU 加速不是自动成立。当前 PyTorch PPO 可先做单机可信基线。

**奖励/终止只围绕真实物体事件**：目标接近、上下接触、闭合后保持、离地净高、回站、带载前进、落入 B 区；跌倒、头撞地、掉物、过大冲量/扭矩、嘴/地穿透、错误对象、超时作为惩罚或终止。独立评估代码使用对象自由刚体状态与[MuJoCo contact sensor](https://mujoco.readthedocs.io/en/latest/XMLreference.html#sensor-contact)，不要复用训练奖励来判定成功。对“夹住但不能走”“走到 B 但掉物”等给出单独失败原因。

## 4. 前置相机：定位与控制如何接上

原 MJCF 已定义 `head_camera`，先在两引擎测相机内参/外参、投影、视野、图像颜色/深度和延迟。MuJoCo 的[官方 Python 渲染接口](https://mujoco.readthedocs.io/en/stable/python.html)可生成离屏相机数据；Godot [SubViewport](https://docs.godotengine.org/en/stable/classes/class_subviewport.html)可承载机器人相机，[Viewport 官方文档](https://docs.godotengine.org/en/stable/classes/class_viewport.html)说明应等待 `frame_post_draw` 再读图。[ViewportTexture 文档](https://docs.godotengine.org/en/stable/classes/class_viewporttexture.html)特别提醒图像复制开销，不能无基准地每个 2000 Hz 物理步读回整帧。Godot 深度路径需单独实现/量测，不能把 RGB 画面误报为 RGB-D。

可选定位路线：

| 路线 | 所需输入/数据 | 算力与延迟（公开值或需实测） | 对 MicroDuck 的判断 |
| --- | --- | --- | --- |
| 训练期真值目标状态 | 仿真自由刚体位姿；无图像数据 | 最低；用于明确抓取机构和控制上限 | **仅 teacher/物理消融**。部署时不能偷偷读 Godot 对象坐标充当视觉。 |
| 小型实例分割或关键点网络 + 地面/相机几何 | MuJoCo/Godot 带像素标签的合成数据、相机标定；若有深度可反投影 | 训练量、CPU/GPU 占用和 p95 延迟须在本机测 | 已知球/瓶/箱的优先部署候选。球的朝向物理上不唯一，主要需求是中心/半径；瓶/箱可增加抓取方向。前嘴遮挡时必须依靠接触/本体状态维持估计。 |
| 现成 6D 姿态估计 | [MegaPose 官方实现](https://github.com/megapose6d/megapose6d)要求 RGB、相机内参、物体网格、2D 框，深度可选；[FoundationPose 官方实现](https://github.com/hewu2008/FoundationPose)需要 CAD 或参考图，提供姿态跟踪 | 需要独立部署/延迟基准，现有 Godot native 扩展不支持直接加载 | 未知位置和复杂旋转目标时可对照；小物像素数、遮挡和与 Godot 渲染差异要实测。仅“找到姿态”仍不能替代接触控制。 |
| 端到端视觉动作学生 | 带相机图、姿态/接触/动作对齐的成功与失败轨迹；teacher 数据或在线视觉 RL | 数据/训练成本最高，推理依网络大小 | teacher 达标后做；[非对称 actor-critic 原论文](https://openai.com/index/asymmetric-actor-critic-for-image-based-robot-learning/)让图像 actor 与特权状态 critic 协同训练；[Isaac Lab 官方实例](https://isaac-sim.github.io/IsaacLab/develop/source/setup/tutorial.html)也使用 camera actor/特权 critic/teacher 蒸馏。 |

定位输出应包含置信度和过期时间。若目标离开视野、误差超阈或接触前估计不稳定，任务管理器停止继续低头、重新观察/对位。视觉失效时不能切换到训练期真值。仅单目 RGB 的绝对尺度取决于标定、已知物体大小、地面约束或额外深度；不是“装了相机就自动得到精确 6D”。

**视觉数据合同与防泄漏**：每条样本须记录相机 RGB 原帧、内参、帧时间戳、与策略周期的对齐关系、可由关节编码器/IMU 推导的相机姿态、本体观测、目标可见掩码/像素遮挡比和训练标签；真值自由刚体位姿、质量、精确尺寸只能生成监督标签或进入训练期 critic，不能进入部署 actor。按完整 episode、物体纹理/外形和场景随机种子切分集合，不对同一轨迹逐帧随机分割。训练与留出均要包含物体尚远、头/嘴遮挡、接触后视线丢失、错误对象、目标出画面、空场景、失败轨迹。不可见时输出低置信度/未知并停止接近或按有时效的状态估计短时跟踪，不得拿真值补洞。相机外参、画幅/FOV、光照、材质、背景、曝光/噪声与采样延迟需要受控随机化，并用 Godot 渲染场景作独立域测试；不同相机标定版本不能混进一个数据集。开放箱的几何中心位于空腔，目标应是可夹的壁/边缘候选抓点及接近方向，不能把中心点距离拟合得很准就算抓取定位成功。

## 5. VLA 决策：可研究，但先证明需要它

| 算法候选 | 官方证据与数据要求 | 本地推理 / 部署代价 | 适用决定 |
| --- | --- | --- | --- |
| 状态 teacher + 几何视觉前端 + 小 ONNX 策略 | 本仓库已有 14D PPO/ONNX 管线可复用，需新抓运放任务；视觉前端需合成标注与 holdout | 策略可继续在 Godot 的 ONNX Runtime GDExtension 内执行；真实 p95 要测。新增多输入图像网络需扩展接口。 | **第一主线**：物理问题和视觉问题可分别诊断，闭集轻物无需语言规划。 |
| ACT / Diffusion Policy 模仿学习 | [ACT 原作者项目](https://tonyzhaozh.github.io/aloha/)报告每任务约 50 段示范的实验；[Diffusion Policy 官方实现](https://github.com/real-stanford/diffusion_policy)训练视觉动作序列；这些结果不保证转到鸭嘴 | 需要自动专家或人工示教、动作块回放、闭环延迟测试；可导出或改写小网络本地推理 | 如果 PPO 探索抓取效率差，用接触可行的 teacher/脚本产生**真实物理成功**轨迹，再做行为克隆和闭环 DAgger；不能用瞬移脚本作为成功示教。[DAgger 原论文](https://proceedings.mlr.press/v15/ross11a.html)。 |
| SmolVLA-450M | [Hugging Face 原作者说明](https://huggingface.co/blog/smolvla)为 450M 参数、公开数据和微调配方，可用单消费级 GPU；仍须本体/动作 schema 适配和 MicroDuck 轨迹 | 需本机测视频预处理、动作块推理、负载、掉帧；当前单输入 ONNX 扩展不能直接跑完整模型 | 当物体类别/语言指令扩展，且已有足够抓运放示范时作为首个 VLA 基线；不能零样本宣称能控制 15 轴新机构。 |
| OpenVLA-OFT 7B / GR00T N1.7 ~3B | [OpenVLA-OFT 作者项目](https://openvla-oft.github.io/)给出动作块及相对原模型更低延迟；[GR00T 官方 GB10/DGX Spark 表](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md)报 N1.7 在同类平台 7.9 Hz eager、10.1 Hz TensorRT（单相机、四去噪步）及微调最低 40GB GPU；这是**厂商基准，不是本机测量** | 需要异步推理和低层 50 Hz 稳定器；若走本机独立服务，属于本地计算但不属于 Godot 进程内 native 推理。若严格要求进程内，须单独移植模型/预处理/运行时。 | 更广泛语义作业的备选研究；以当前闭集 A→B 基线为先。10 Hz 高层动作不能直接替代 50 Hz 嘴部接触反馈。 |

资源量级也必须分清**来源环境**：状态 PPO 需要在线物理步，样本数/每秒步数必须由新 MuJoCo 场景实际 benchmark；小视觉网络需要合成标注帧，帧数与 p95 延迟要由相机精度目标反推。LeRobot [SmolVLA 官方教程](https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx)建议特定任务从约 50 条示范开始，报告同一 SO-100 数据集 25 条效果不足，20k 微调步在单 A100 上约 4 小时；这既不证明 50 条足够覆盖 MicroDuck 的三类物体和搬运，也不代表 GB10 训练耗时。NVIDIA [GR00T 官方部署指南](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/real_world_deployment.md)建议至少约 100 条有效完整 episode、200+ 通常更稳定，20–50 Hz 采集；跨本体适配的数据采集和推理仍是额外成本。两类 VLA 的模型参数、GB10 频率和教程数据量应只作为实验预算起点，最终方案必须用本机实际吞吐、显存、p95 推理时间和全链留出成功率决策。

**VLA 启用门槛**：在固定相机/目标类别的视觉定位 + 小策略已做基线后，若未知类别/遮挡/语言条件成为主要失败原因，再收集经过物理验证的轨迹微调 VLA；比较同一 holdout 的全链成功率、p95 延迟、算力与失败种类。仅换大模型不能修正碰撞网格、铰链力矩、观察盲区或带载步态。Godot 原生推理优先小 ONNX 模型；[ONNX Runtime C++ API](https://onnxruntime.ai/docs/get-started/with-cpp.html)支持进程内会话，GPU provider 的依赖与运行细节见[官方 CUDA EP 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。现有 [`policy_runtime.cpp`](../native/src/policy_runtime.cpp)只接受单张量/单输出且白名单形状不含新 15 维动作，必须修改并做 Python/Godot ONNX 数值逐步对齐。当前 [`driver.gd`](../godot/standalone/driver.gd)也需按技能路由不同观测/动作；八个原技能可保留原名与原模型，但**每个带新嘴质量/碰撞后的能力要独立验证**。

## 6. 严格验收与防自欺记录

在优化前冻结测试协议和阈值。建议从 ≥0.5 m 的 A→B 路径（含一次转向）和球/瓶/箱 × 多质量/位置/纹理/摩擦的分层留出集开始，物体实际离地 ≥0.05 m 且持续 ≥1 s，回站时机器人不跌倒；路径中不能掉物、撞到障碍或超限力矩；B 区内释放并稳定落下才算单次成功。数值阈值是**待按实际比例确认的建议**，不能在看完失败结果后调低。报告每层样本数、全链与分阶段成功率、置信区间、掉落/碰撞/电机超限、失败视频。成功演示之外还要保留随机种子和失败样本。

每个 episode 必须按时序记录 `detected → aligned → true_contact → grasp_confirmed → lifted → upright → A_to_B_progress → released → landed_in_B`，并保存 RGB 帧、目标/机器人/嘴轨迹、接触力、策略输入输出、阶段切换原因、模型/场景哈希。训练启动、日志、优化器 checkpoint、验证 seed、ONNX 导出与 MuJoCo/Godot 数值一致性是**必要证据**；仅 ONNX 文件名/不同 SHA、漂亮视频或手工按 H/B 不是证据。Godot 最终测试需关闭外部策略服务和训练时的真值目标坐标，只运行随包本地推理，完整录下从自动启动到落在 B 区的连续视频。

**当前判定：未完成。** 首个研究可证伪问题是“具有真实下嘴 hinge 和给定抓点的 MicroDuck，在 MuJoCo 能否对现有轻物完成接触、夹持、抬升和回站？”通过后才扩大到带载 A→B 与视觉；任一阶段若失败，应保留失败数据并修正该阶段，而非将其他阶段的视觉展示当成训练结果。

## 7. 2026-09-19 独立监督记录

本仓库已有真正经过 GPU 反向传播的特权状态行为克隆：[`train-bottle-v1/manifest.json`](../results/microduck-pickable/train-bottle-v1/manifest.json)记录 20 条训练、5 条验证成功示范、140 个 epoch、源/数据/产物 SHA 和 ONNX 数值对齐；新 ONNX 与源模型的 SHA 不同。它的观测含 MuJoCo 真值物体位置、速度、尺寸与质量，且只有窄瓶成功示范，不能据此宣称视觉作业或原尺寸物体的能力。

本次审计时源 `GroundPick` SHA-256 为 `ffbf5109982ff999b0ba53afe86b9ae731bbec679d67fb7f8ab4c52152c88872`，训练 ONNX 为 `e596f8b0c55f73f462f0765f44a98530d4879e73bcc0cef36307a5a7d2d3d29c`，示范数据为 `8fc8be549b91cd8e4b7c3d9a37199ed790e5dee9d1364fb2d5ebbd036f872a0f`。这些值和以下数值是本地 `results/` 的审计快照；该目录当前被 Git 忽略，正式交付须另外封存 manifest、命令、模型和留出 episode 证据，不能只依赖本页的相对链接。

独立检查同一批 100 个新种子的 [`eval-bottle-v1`](../results/microduck-pickable/eval-bottle-v1/summary.json) 与 [`eval-bottle-fixed-source`](../results/microduck-pickable/eval-bottle-fixed-source/summary.json)：行为克隆成功 18 次，原 `GroundPick` 身体动作加固定 0.46 s 闭嘴成功 34 次；配对样本中前者独赢 9、后者独赢 25，McNemar 双侧精确检验约 `p=0.009`。这是探索性单场景对照，不能代表所有条件，但足以阻止直接替换默认策略。新模型在更远目标上常更早闭嘴，可能将物体推离；应按接触前相对位置和目标移动量做分层诊断。`0.10–0.14 m` 更远目标的 [`eval-bottle-v1-far`](../results/microduck-pickable/eval-bottle-v1-far/summary.json) 为 0/100。行为克隆的验证损失只是动作拟合度，不能作为任务成功的选择指标。

初版 [`mission-bottle-v1`](../results/microduck-pickable/mission-bottle-v1/summary.json) 在 50 个新种子上有 5 次抓起、5 次运输/释放、4 次最终放置成功，证明窄瓶上的物理阶段管线存在可行轨迹；成功起点集中在约 `x=0.060–0.066 m`。该 pilot 使用全知物体/机器人坐标与无承重上限的辅助点连接，尚未使用前置相机，也没有在 Godot 加载 76→15 ONNX。现有 35 mm 嘴净开口对原 70 mm 球不可达；原尺寸瓶和开放箱也要单独测可行抓点与力学，不能从 16 mm 窄瓶外推。下一次验收先冻结多位置、多质量、多原尺寸形状和负对照，报告真实力/约束冲量、每阶段失败、视觉替代误差与 Godot 进程内推理延迟。

另有原尺寸 36×78 mm、8 g 瓶的接触式吸附探索：旧 `GroundPick` 身体策略加常开 1.0 N 总力上限的 MuJoCo adhesion actuator，在早期软接触参数下 `x=0.05–0.11 m` 的 100 个新种子有 73 次按吸附接触与稳持定义的成功。独立审计又以与吸附开关无关的末态连续 1 s 离地 ≥8 cm、机器人直立、嘴尖到抓点 ≤5 cm 为条件配对重放，吸附开/关为 71/100 和 10/100。关吸附时不少瓶可暂时卡在上下嘴壳上，但最终掉落；故吸附带来留持收益。修改吸附垫的 MuJoCo 接触硬度后，同种子瓶的成功降到 57/100；双吸附垫的原 70 mm 球与原尺寸箱分别在另两组 100 个合法起点得到 37/100、45/100。成功轨迹仍有明显嘴壳/物体穿透，尤其球在抬起前可达约 5–6 mm；这些仍是机构可行性研究。此审计阈值为事后定义，只能用于诊断，正式效果须在新留出集合上预先固定成功、最大穿透与承载条件，并进一步完成带载运输与释放；这也不是新 `GroundPick_Pickable` 的训练结果。

### 在线优化决策与数据划分

新一轮抓取优化应保留已有 34/100 的固定源策略作为对照。第一阶段仅优化 `neck/head/yaw` 有界残差和嘴的接触/相对位姿触发，而非重新探索全部 15 维动作；在当前已知几何可行区域以条件交叉熵搜索或其他低维黑箱优化取得成功轨迹，再蒸馏为目标条件策略。第二阶段用在线残差强化学习或失败轨迹聚合修正接触后的状态分岔，必要时才增加动作维度。[Residual RL 原论文](https://arxiv.org/abs/1812.03201)为基控制器叠加可学习校正提供先例；[DAgger 原论文](https://proceedings.mlr.press/v15/ross11a.html)指出仅模仿专家访问的状态会受分布偏移影响；[QT-Opt 原论文](https://proceedings.mlr.press/v87/kalashnikov18a.html)展示闭环抓取与动作采样优化的价值。这些论文只支持方法选择，不能代替本项目成败测量。

现有 25 条成功搜索轨迹来自 1000 次随机候选，`head` 偏置范围约 `-0.394..+0.083 rad`、闭嘴时间约 `0.315..0.744 s`；逐时刻 MSE/BCE 模仿会把多种可行解平均成不可行控制。下一轮至少冻结三个互不重叠的种子/初始条件集合：优化集可反复探索；验证集仅供选择 checkpoint 与超参数；测试集在算法定型后只评一次。之前用于多次诊断的 `991301` 集已是开发集，不能再称为独立测试。按目标距离与横向偏差分层采样，成对报告新策略与固定源策略在同一 reset 下的抓取成功率、首碰、对象被推走、峰值接触/连接力、掉落与跌倒。行为克隆损失只能辅助监视，最终模型由留出闭环成功率和安全约束决定；后续再扩展完整尺寸与视觉输入。
