# Unity 独立部署中可复用的策略与控制机制

调研日期：2026-09-11。Unity 参考仓库固定在 [`082e5dcfc7ed66f9dabe607454294259c4252d7b`](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/commit/082e5dcfc7ed66f9dabe607454294259c4252d7b)，提交时间 2026-09-07 00:37:42 +08:00。本地 Godot 对照基线为 `da5022170a6da02260826627d574a0c56a18f6c7`。这是源码调研与模型文件核对，没有运行团结引擎或重新构建其 Player。

## 结论

**算法与部署思路共通，值得借鉴；应迁移推理边界和完整控制契约，不能把 Barracuda 或 Unity 控制代码原样放进 Godot。** 参考项目已将策略推理放在 Player 内：C# 从物理状态拼出观测，调用 Barracuda，得到动作，再写入内嵌 MuJoCo。Python 承担导出、资产生成和验收编排，读到的可玩运行循环没有调用 Python 或 TCP。[推理实现](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Runtime/BarracudaPolicyRuntime.cs#L13-L67)、[物理到策略闭环](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoPolicyStepper.cs#L170-L215)。

**“推理内嵌”和“更换物理后端”是两个独立决策。** 该 Unity 项目的可玩物理权威是 MuJoCo 3.12，不是 Unity PhysX；Barracuda 是神经网络执行器，不是改善接触、驱动器或机器人动力学的算法。若 Godot 保持 Jolt，内嵌等价推理主要解决独立运行及进程通信依赖，不能由此推断现有动作质量会达到参考项目。此为根据双方运行循环作出的架构推断。[MuJoCo 状态读写](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoPolicyStepper.cs#L170-L215)、[只构建 native 场景](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Editor/DemoSceneBuilder.cs#L209-L225)。

## 已核实的运行契约

| 边界 | Unity 源码事实 | 对 Godot 的意义 |
| --- | --- | --- |
| 推理后端 | `BarracudaPolicyRuntime` 实现窄接口 `Evaluate(observation, actionDestination)`；默认 `CSharpBurst` CPU，持有长期 worker，每次创建并释放输入 Tensor | 可保留相同接口思想，Godot 的实现换成适合其导出的推理库；worker、Tensor、`NNModel` 和 Unity 包绑定需重写 |
| 观测 | float32 `[1,61]`：局部角速度 3、投影重力 3、相对 home 关节角 14、关节速度 14、上一原始动作 14、命令 13 | 与本项目 `obs.py` 的布局相同，但坐标基、关节顺序、时间命令含义也必须相同 |
| 动作 | float32 `[1,14]` 原始动作，先计算 `home + action × role_scale`，再滤波，写入 MuJoCo 的 14 个控制通道 | 外部维度一致不代表动作缩放、后处理和控制器一致 |
| 时序 | 物理步长 0.005 秒；每 4 次 MuJoCo 控制回调推理一次，使用 MuJoCo 仿真时间；其余步重新施加上次目标 | 应保留 200 Hz 物理、50 Hz 控制的确定先后关系，不能把推理直接系在渲染帧率上 |
| 模型绑定 | 控制关节由 MuJoCo actuator transmission 的 joint ID 解析；轮滑另有 4 个被动物理关节 | 推理库不会处理关节映射或轮滑模型生命周期 |

来源：[接口](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Runtime/IPolicyRuntime.cs#L5-L10)、[worker](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Runtime/BarracudaPolicyRuntime.cs#L13-L67)、[观测布局](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Runtime/PolicyObservationBuilder.cs#L31-L68)、[目标滤波](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Runtime/PolicyTargetFilter.cs#L7-L39)、[物理步长](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Editor/DemoSceneBuilder.cs#L136-L159)、[控制回调](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoDemoController.cs#L420-L460)、[关节绑定](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoPolicyStepper.cs#L14-L96)、本地 [obs.py](../src/sim2sim/obs.py)。

## 不能盲目照搬的控制行为

Unity 的目标滤波首步直接使用目标，此后头部索引 5–8 使用 `alpha=0.5`，其余关节 `alpha=0.7`。本项目 Python 可玩循环直接将 `home + action × scale` 交给 Godot 后端；因此移植推理时额外照搬这个滤波会改变闭环，必须单独证明其必要性。[Unity 滤波](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Runtime/PolicyTargetFilter.cs#L26-L39)、本地 [play.py](../src/sim2sim/play.py) 的动作提交段。

Unity 命令状态对官方 kick / roulade 保持零命令；ground-pick 和 roller-crouch 才写 cos/sin 相位，周期分别为 4 秒和 5 秒。它没有读取本项目新候选图的 `policy_time` / heading 元数据。本项目可玩循环会为相应候选传入动作时间与相对起始朝向，所以即使都是 61 维，新模型也不能直接配 Unity 旧 `PolicyCommandState`。[命令构造](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Runtime/PolicyCommandState.cs#L87-L132)、[策略参数表](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Runtime/PolicyCatalog.cs#L54-L67)、本地 [play.py](../src/sim2sim/play.py) 的 `time_command` 段。

同机器人热切换的可复用做法，是保留物理状态、上一原始动作和目标滤波状态，替换推理 worker；命令相位和 sit 状态则在选择策略时清除。腿式/轮滑切换使用另一个模型，并重建、重新绑定 MuJoCo 场景。复位同时清控制历史、清 MuJoCo data、写 home、设置根高度和初始姿态，并执行前向计算。Godot 独立版本需要自己的等价状态生命周期，而不仅是一个 ONNX 调用。[热切换](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoDemoController.cs#L202-L256)、[连续状态](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Runtime/MicroDuckControlLoop.cs#L6-L30)、[控制复位](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Runtime/MicroDuckControlLoop.cs#L96-L112)、[物理复位](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoPolicyStepper.cs#L112-L168)。

## ONNX 兼容性与验证强度

源码转换器不是通用 ONNX 转换器：只接受默认域、opset 18、`Sub / Div / Gemm / Elu` 四类算子及经过限制的属性，然后把声明改成 opset 9，执行 ONNX 校验和原图/副本的 ORT 对比。它不证明任意 residual、expert、Cast、Slice、时间输入等组合图可通过同一办法进入 Barracuda。[允许集合](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/src/agenticrobot_bridge/onnx_compat.py#L20-L21)、[验证及 opset 改写](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/src/agenticrobot_bridge/onnx_compat.py#L146-L193)。

本次直接读取 Git 中的九组 Original/Barracuda 模型并用本机 ONNX Runtime 1.29.0 CPU 执行了原项目全部 27 个 fixture：每份原图均为 float32 `[1,61] → [1,14]`、opset 18、9 个节点、上述四类算子；18 个模型文件的 SHA-256 均与提交内报告一致，原图对兼容副本最大绝对误差为 0，参考 fixture 输出也在 `1e-5` 内。**这是本次重跑的 ORT 对比，不是 Barracuda 对比，也不是新 Godot 九候选兼容性证明。** 可复查输入与哈希：[compatibility-report.json](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Generated/Policies/Barracuda/compatibility-report.json)、[parity-fixtures.json](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Generated/Policies/Barracuda/parity-fixtures.json)、[ORT 比较方法](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/src/agenticrobot_bridge/onnx_compat.py#L196-L225)。

上游的证据边界还需分层看：

- **静态模型对比测试**：九模型各 3 个 fixture，Barracuda 显式选 `CSharp`，阈值 `1e-5`；而可玩默认 worker 为 `CSharpBurst`。因此不能把这个测试等同于所有发布后端、所有真实观测的严格证明。[BarracudaParityTests](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Tests/EditMode/BarracudaParityTests.cs#L19-L72)。
- **trace-parity**：实际 exporter 打开旧 `MicroDuckMvp.unity` 校准场景，用 `MicroDuckDemoController` 的 stand 策略只导出 1 tick；它不是 native 场景九技能长序列逐步对照。[TraceBatchExporter](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Editor/TraceBatchExporter.cs#L23-L76)、[两个场景常量](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Editor/DemoSceneBuilder.cs#L17-L20)。
- **持续行为测试**：另有 native MuJoCo、Barracuda CSharp 的九技能和两条同模型热切换测试；指标是动作行为门槛，并非与 Python 每一步完全相同。[AuthoritativeMujocoPolicyBehaviorTests](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Tests/EditMode/AuthoritativeMujocoPolicyBehaviorTests.cs#L17-L120)。
- **Player smoke**：代码要求 MuJoCo 3.12.0、指定后端名、至少 3 个策略 tick、61/14/14 缓冲均有限；该门槛证明基本装载和运行，不证明动作质量。[MujocoPlayerSmokeProbe](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoPlayerSmokeProbe.cs#L18-L20)、[通过条件](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoPlayerSmokeProbe.cs#L99-L118)。

本次 clone 不包含 `artifacts/` 或 `Builds/`，它们也明确被 Git 忽略。因此上述后三类是已检查的测试实现与要求，不能写成“本次已通过”的结果。[忽略规则](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/.gitignore#L9-L26)。

## 独立发布的实际范围

参考项目锁定的是 Unity 包 `com.unity.barracuda` 和 `org.mujoco`；没有 Sentis 依赖。直接复用其 `NNModel` 导入器、`Unity.Barracuda` worker、Unity 组件及场景生命周期，需要 Unity 环境，不是 Godot 可直接加载的组件。[依赖清单](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Packages/manifest.json#L1-L6)、[Barracuda API 依赖](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Runtime/BarracudaPolicyRuntime.cs#L1-L38)。

源码提供 Windows x64 和 macOS 构建入口，均只包含 native 场景。Windows 包含 Player、MuJoCo DLL 和托管桥接程序集，README 声明需 VC++ 2015–2022 x64 runtime；Mac 包包含 Player dylib、MuJoCo dylib 和程序集，其原生库锁文件列出 x86_64/arm64 两架构。没有由这些证据建立 Linux、移动端或 Web 的已验收承诺。[构建目标](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/TuanjieProject/Assets/MicroDuck/Editor/DemoSceneBuilder.cs#L209-L284)、[Windows 发布文件校验](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/scripts/mvp-gates.py#L545-L563)、[Mac 发布文件校验](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/scripts/mvp-gates.py#L642-L658)、[原生库锁定](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/upstream.lock.json#L13-L32)、[Windows 运行库说明](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/blob/082e5dcfc7ed66f9dabe607454294259c4252d7b/README.md#L63-L65)。

这里的“独立运行”可理解为随应用分发全部模型与平台原生依赖，最终用户不需 Python 服务；并不要求最后只有一个可执行文件。该判断来自实际发布文件清单。

## 对下一阶段决策的建议

1. 将训练/导出留在 Python，把发布运行时的观测构造、命令状态、推理、动作后处理和复位/切换迁入 Godot。以当前冻结候选的实际契约为准；Unity 代码提供架构参照。
2. 先决定 Godot 推理扩展路线和目标平台，使用候选原始 ONNX 验证；不要先按 Unity 的 opset-9 转换器改图。需要改图时，按新算子与数据类型重新证明数值等价。
3. 后续原型应分别通过：真实观测和边界输入的九模型推理对照、相同 Jolt 初态/命令下完整控制序列回放、模型切换/复位/轮滑切换、无 Python 环境的导出包启动。这是基于参考项目验证分层提出的建议，尚未实施或验收。
4. 是否内嵌 MuJoCo 应保持独立决策。若目标包含缩小物理域差距，可另评估它；仅为消除 Python/TCP，现有证据不要求更换 Jolt。
