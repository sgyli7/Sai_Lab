# MicroDuck 嘴部抓取与 A→B 搬运：技术研究记录

日期：2026-09-19。本文是设计与证据审计，**不是训练结果或验收报告**。目标是只重新训练或续训 `GroundPick_Godot_Pickable.onnx`；其余八个技能保留原模型，验证与新嘴部机构的兼容性。若原步行策略带载失稳，应先量化失败，再决定反馈适配是否足够，不能把未验证的搬运能力写成已完成。

## 已证实的现状

1. **当前 `_Pickable` 没有新训练。** [`src/sim2sim/pickable_bundle.py`](../src/sim2sim/pickable_bundle.py) 明确逐字节复制九个 ONNX。工作坊运行包内 `GroundPick_Godot.onnx` 与 `GroundPick_Godot_Pickable.onnx` 的 SHA-256 同为 `ffbf5109982ff999b0ba53afe86b9ae731bbec679d67fb7f8ab4c52152c88872`。[`godot/tests/microduck_pickable_probe.gd`](../godot/tests/microduck_pickable_probe.gd) 先把目标物体移动到嘴前，再按 H；这不能证明自然场景按 G 自动捡取。
2. **原 GroundPick 只学了低头触地再站起。** [`src/sim2sim/research/tasks.py`](../src/sim2sim/research/tasks.py) 定义四秒相位；[`src/sim2sim/research/rewards.py`](../src/sim2sim/research/rewards.py) 的 GroundPick 奖励是嘴尖离地高度、向下姿态、回站、双脚接地和头部防撞；[`src/sim2sim/research/evaluate.py`](../src/sim2sim/research/evaluate.py) 的成功条件也只有这些量。评估中没有物体接触、夹持、吊升或搬运。
3. **原策略既看不到目标，也不能驱动嘴。** [`src/sim2sim/obs.py`](../src/sim2sim/obs.py) 与 [`godot/standalone/policy_contract.gd`](../godot/standalone/policy_contract.gd) 定义 61 维输入：3 角速度、3 重力、14 关节位置、14 关节速度、14 上次动作、13 命令；无物体位置或嘴关节状态。14 维输出仅对应现有电机。[`godot/standalone/driver.gd`](../godot/standalone/driver.gd) 直接要求 GroundPick 输出 14 维；[`native/src/policy_runtime.cpp`](../native/src/policy_runtime.cpp) 也只准许 MicroDuck 61/68→14。两个物体位姿不同、机器人状态相同的情形会给现有网络相同输入，因此网络不可能按目标位姿调整抓取；新增嘴电机也不可能由原 14 个动作直接控制。
4. **原 MJCF 下嘴并非关节。** 上游 [`robot_allcollisions.xml`](/home/ethan/Projects/microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml) 的 `jaw_soft` 通过 `head_roll` 连接整个头；`jaw`、`bottom_head_shell` 的视觉/碰撞 geom 都固定在 `jaw_soft`，14 个 position actuator 中没有下嘴电机。`mouth_tip` site 是测量位置，不能开合嘴。MuJoCo [关节和 freejoint 定义](https://mujoco.readthedocs.io/en/latest/XMLreference.html) 指出：没有 joint 的子 body 焊接在父 body，hinge 才增加转动自由度；[position actuator](https://mujoco.readthedocs.io/en/3.2.2/XMLreference.html) 才能给该自由度施加位置控制。
5. **当前 Godot 嘴部是脚本改网格并用点关节辅助抓住。** [`godot/hub/microduck_beak.gd`](../godot/hub/microduck_beak.gd) 把 `vis_unnamed_55_55`、`vis_unnamed_57_57` 挂到 `Node3D` 枢轴，启用 `AnimatableBody3D` 碰撞体；满足距离阈值后创建连接物体和整颗头 `jaw_soft` 的 `PinJoint3D`。这不是受力控制的独立嘴关节，也不证明上下嘴的实际夹持。Godot [PinJoint3D](https://docs.godotengine.org/en/stable/classes/class_pinjoint3d.html) 是两刚体间的单点约束；要称为“辅助抓取”，必须公开该机制及其触发条件。
6. **坐标错误有可计算的证据；本轮已经修正可见脚本。** 导入器 [`src/sim2sim/coords.py`](../src/sim2sim/coords.py) 将 MuJoCo 世界 `(x,y,z)` 映射为 Godot 世界 `(x,z,-y)`，但 Godot 刚体的局部轴是 MuJoCo **惯性主轴**，并非世界轴；[`src/mjcf2godot/convert.py`](../src/mjcf2godot/convert.py) 以 `xipos/ximat` 建立刚体。`jaw_soft` 的上游 `mouth_tip` 是 MJCF body 局部 `(-0.00809334,0,-0.0777383)` 米；根据运行包 [`robot_spec.json`](../results/workshop-hub/runtime/generated/microduck_ball_stand_fix/robot_spec.json) 的 `ipos/iquat`，用 `R_iquatᵀ (p_body - ipos)` 换算到 Godot `jaw_soft` 惯性局部，为 `(-0.01440896,-0.00309805,-0.04465586)` 米。先前 [`microduck_beak.gd`](../godot/hub/microduck_beak.gd) 的 `TIP_LOCAL=(-0.015,-0.045,0.003)` 和 Z 轴开合让嘴歪斜；本轮已把 TIP 改为上述惯性局部坐标、枢轴改到中线并绕局部 Y 轴开合。已用 [`microduck_beak_visual_probe.gd`](../godot/tests/microduck_beak_visual_probe.gd) 捕获正面/侧面/斜视角的旧开嘴图（`/tmp/microduck-beak-current/`）与修正后图（`/tmp/microduck-beak-hinge-corrected/`）；这只验证外观，仍未建立受训练控制的真实嘴关节，也未验证载荷。
7. **现有 Godot 输入不是要求的流程，G 单键基线失败。** [`godot/hub/hub.gd`](../godot/hub/hub.gd) 对 MicroDuck 的 G 仍只触发旧捡地技能，H 才调用嘴抓取，B 用于挑选物件；[`godot/atelier/loose_props.gd`](../godot/atelier/loose_props.gd) 提供真实的 6–15 g 瓶、球、箱。本轮 [`microduck_g_pick_probe.gd`](../godot/tests/microduck_g_pick_probe.gd) 将未冻结球作为**初始条件**放在嘴下地面、不选目标、只按 G，输出 `attached=false`、最近距离 `0.0524326749 m`、最大升高 `0.00804 m`、`selection=0`、`events=[]`。该探针还不是默认散件位置的完整测试；它明确说明仅按 G 尚未实现抓取。

## 推荐的训练与控制合同（待实施和实测）

**先修模型和物理。** 以原 `scene_ball.xml` 所引用的 `robot_allcollisions.xml` 派生独立 Pickable MJCF，保持原 14 个关节及其顺序；把下嘴网格及碰撞 geom 从 `jaw_soft` 移入新刚体，在实测对称轴和铰链点上加第 15 个 hinge、限位、合理质量/惯量和有限力矩 position actuator。将上下嘴接触面与可见网格逐一核对；中性角的轮廓必须与原模型重合。Godot 侧要由相同几何生成或构造真实 `RigidBody3D + HingeJoint3D`，校验左右对称、铰链轴、限位和接触；不能只旋转两个视觉网格。Godot [HingeJoint3D 文档](https://docs.godotengine.org/en/stable/classes/class_hingejoint3d.html) 给出角限位和电机配置，Godot [3D 坐标约定](https://docs.godotengine.org/en/stable/tutorials/3d/introduction_to_3d.html) 明确 Y 向上、单位为米。铰链点、转轴、质量和 20 g 上限目前都是**需量测/试验的参数**，不可当作已验证设计值。

**新 GroundPick 模型采用版本化观测与动作。** 推荐示例 `pickup_v1` 为 `float32[1,76] → float32[1,15]`：保留前 61 维原观测和 4 秒相位编码，再加入机器人局部目标抓取点 3 维、物体尺寸 3 维、相对速度 3 维、嘴角/角速度 2 维、上下嘴与目标接触标志 2 维、质量 1 维、已夹持状态 1 维，共 15 个附加量；这里的维数是**设计建议**，在两模拟器都能稳定提供这些量后才能冻结。前 14 个动作仍是原关节目标，第 15 个是嘴角目标。可从原 GroundPick ONNX 的前 61→14 子网络初始化，增加零初始化的本体残差和新嘴动作头；保存真实优化器检查点、训练步数、种子、环境哈希与导出数值校验。现有 [`src/sim2sim/research/models.py`](../src/sim2sim/research/models.py)、[`src/sim2sim/research/world.py`](../src/sim2sim/research/world.py)、[`src/sim2sim/workshop_assets.py`](../src/sim2sim/workshop_assets.py) 都硬编码 61/14，因此需要独立的新任务实现，不能直接运行旧 GroundPick 训练脚本后宣称学会抓取。

**物体与奖励。** MuJoCo 训练场景应放带 freejoint 的真实刚体：球、瓶、空箱，随机化相对位置/朝向、半径、质量、重心、摩擦、恢复系数与轻微初速度；范围先由 Godot 现有物件的实际尺寸和 6–15 g 质量确定，再逐步拓展并留出未训练组合。先检验可达域和身体/嘴部净空，再做课程训练：固定居中目标 → 多目标位姿 → 多形状/质量 → 不确定接触。奖励建议分阶段给接近抓取点、正确开嘴、双面接触、稳定夹住、离地高度、回站和抗摆动；惩罚头撞地、脚离地、跌倒、物体穿透、超扭矩、动作抖动。终局奖励必须依赖独立的实际物体状态与接触/保持，不能仅依赖嘴尖高度。MuJoCo [碰撞与 contact 数据](https://mujoco.readthedocs.io/en/latest/computation/) 可以独立测物体和嘴的真实接触。

**辅助点关节的取舍。** 若任务允许 Sai 式辅助夹持，应在训练与 Godot 测试中使用同一公开规则：只有嘴/物体近距离且报告真实接触、嘴正在闭合时才激活，松嘴/超载/异常位移即释放；禁止先按距离吸附远处物体。激活后点关节可能使两者不再接触，因此不能简单用“每帧接触消失”作为释放条件。MuJoCo 可用可切换的 [equality/connect 约束](https://mujoco.readthedocs.io/en/stable/programming/simulation.html) 实现对应的单点连接，Godot 用 `PinJoint3D`。这只能证明“受物理约束的辅助抓取”，**不能**报告为纯摩擦力闭合。若要求纯嘴夹持，两侧碰撞、摩擦和真实接触力必须单独训练与评估。Godot [RigidBody3D 接触监测](https://docs.godotengine.org/en/stable/classes/class_rigidbody3d.html) 需要启用 `contact_monitor` 与足够的 `max_contacts_reported`。

**Godot 本地运行。** 仅将 `ground_pick` 槽替换为真正新导出的 `GroundPick_Godot_Pickable.onnx`，八个原技能继续按原 61/14 合同加载；部署记录模型/源检查点/训练环境 SHA-256、合同版本、输入顺序、归一化、动作映射、50 Hz 推理与 200 Hz 物理。扩展 [`native/src/policy_runtime.cpp`](../native/src/policy_runtime.cpp) 的尺寸白名单和 [`godot/standalone/driver.gd`](../godot/standalone/driver.gd) 的分技能观测/动作路由；核对 Python ONNX Runtime 与 Godot 内嵌 ONNX Runtime 对真实轨迹的逐步动作误差。不能有训练进程或 Python/TCP 在场外替 Godot 决策；本地模型推理本身仍可使用现有 GDExtension 的 ONNX Runtime [C++ 接口](https://onnxruntime.ai/docs/get-started/with-cpp.html)。

**完整 A→B 闭环。** G 应自动选择嘴前/脚边可达的最近轻物体，记录目标与抓取点，执行靠近、低头、开嘴、接触、闭嘴、吊起、回站。确认稳定夹持后，切到原步行模型，由场景导航/速度反馈到达 B；保持嘴闭合并用载荷位置、头部姿态、跌倒/摆动监测调整步速。到达 B 后在安全高度张嘴释放，确认物体落在目标区且无机器人碰撞。原 `Walk_Godot.onnx` 是否能稳定带载是**待测假设**；不通过时要报告并改进反馈/动作适配，再次测八模型兼容性。单个 GroundPick 政策不负责几米路程的导航，也不能以“低头成功”代替搬运成功。

## 反作弊验收协议（实施前冻结阈值）

- **模型真实性：** 新 ONNX 哈希必须不同于源文件，并能追溯到实际训练检查点、日志与导出；比较不是充分条件，还要做新模型对旧模型的 A/B 物理评估。八个原 ONNX 不改名、不复制充数。
- **用户流程：** 从未手选目标的自然场景只按 G；禁用 B/H、脚本把物体放到嘴前、夹持前瞬移/冻结物体、无真实接触的定距吸附。记录输入事件、目标原始位姿、每帧物体位姿、接触、关节角、模型输出、夹持状态和机器人状态。
- **完整事件序列：** 明确记录 `selected → reached → physical_contact → closed/attached → lifted ≥ H → stood ≥ T → walked A→B ≥ D → released → landed_in_B`，任何丢物、跌倒、回站失败或超时都记失败。`H/T/D` 与 B 区边界在评估开始前冻结，建议先按实物尺度设计，再做可达性验证，不事后降低门槛。
- **泛化与安全：** 按形状（球/瓶/箱）、质量、目标相对位置/朝向、地面摩擦和随机种子分层统计全链成功率与每阶段失败率；训练种子与物体组合不得进入最终留出集。测最大摆幅、载荷掉落、机器人倾角、脚滑、碰撞冲量、超限扭矩和释放后物体稳定时间。须报告每一层样本数与置信区间，不能只展示成功片段。
- **可见复核：** 用真正的 Godot 可视渲染录下原始连续视频，包含按 G、弯腰、接触、吊升、回站、从 A 到 B 行走和释放；正面、侧面、斜视角另录中性/开嘴/闭嘴，确认无歪嘴或网格错位。同步保存原始轨迹与模型哈希，并用视频逐帧复核失败与成功样本；无头模式数值探针不能替代此项。

## 当前结论

现有九个 `_Pickable` 是原模型副本；现有 GroundPick 训练目标不含物体；嘴没有原生自由度；仅按 G 的物理基线没有抓到球。本轮修正了 Godot 嘴的可见局部坐标和开合轴，但**外观修正不等于抓取训练**。可行路线是**先修可动嘴的双模拟器物理，再为 GroundPick 建立带目标/嘴状态的新合同并真实训练，最后以 Godot 本地推理完成 G 一键抓取与 A→B 搬运的独立留出集评估**。截至本记录，没有证据表明上述新训练或完整搬运已经完成。
