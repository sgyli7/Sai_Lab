# Godot 内嵌 ONNX 推理路线调研

调研日期：2026-09-11。范围：独立导出应用的推理框架、原生依赖与模型兼容性；未实现插件、未执行导出包性能实验。桌面架构、移动端/Web 和机器人数量仍由路线图中的产品决策确定。

## 结论

**可行，首个验证候选建议为：保留现有 GDScript/Jolt，通过一个小型 C++ GDExtension 内嵌 ONNX Runtime 1.29.0 CPU，直接加载现有九个 ONNX。** 这是基于现有 Python 已调用同一 ORT CPU 后端、模型算子和引擎扩展接口得出的工程判断，并非已验证的交付承诺。Godot 的 GDExtension 可以加载原生共享库，无需把扩展编译进引擎；ORT 提供独立于 Python 的 C/C++ API。[Godot GDExtension](https://docs.godotengine.org/en/stable/tutorials/scripting/gdextension/what_is_gdextension.html)、[ORT C API](https://onnxruntime.ai/docs/get-started/with-c.html)

“内嵌推理”改变部署边界，不改变策略训练算法，也不会自然消除 Jolt 与 MuJoCo 的动力学差距。共同的数学对象是观测到动作的 ONNX 图；需要迁移的应用行为还包括观测构造、动作缩放、上一帧动作、技能时钟、相对航向、复位与机器人切换。仅替换 Python 中的 `InferenceSession` 调用，仍不足以得到独立运行版本。当前控制契约与模型审核见[九模型运行契约调研](research_embedded_policy_contract_20260911.md)。

## 候选方案

| 路线 | 独立桌面导出 | 当前项目适配代价 | 当前模型风险 | 本轮判断 |
| --- | --- | --- | --- | --- |
| 自有 C++ GDExtension + ORT CPU | 引擎、扩展、ORT 原生库、模型随包交付；无需 Python | 新增窄原生边界；保持 GDScript 项目 | 保留 Float64 内部节点；锁定 ORT、线程与优化选项后做数值比对 | 首个验证候选 |
| Godot .NET/C# + ORT CPU NuGet | Godot 支持桌面 .NET 导出，已编译游戏随包携带所需 .NET 部分 | 需要 .NET 版编辑器/导出模板、C# 项目及目标 RID 原生依赖验证 | 同一 ORT 内核；不能把 NuGet 成功还原当作目标架构已打包成功 | 团队愿意采用 C# 时的合理备选 |
| 现成 Godot ONNX 插件 | 已有社区原生扩展与跨平台构建 | 可以借鉴，但必须审查版本、PCK 读取、释放资源和导出路径 | 当前查到的较新候选锁定 ORT 1.20.1；不是本项目已经验证的 1.29.0 | 作为代码参考或对照原型，不能直接认定完成交付 |
| Barracuda / Sentis 直接放入 Godot | 官方部署对象为 Unity 运行时 | 不是 Godot 可直接加载的通用推理 SDK | 数据类型、Unity 依赖与导入转换都要重新处理 | 不建议作为 Godot 路线 |

C# 路线同样可以满足“不安装 Python、不启动额外服务”。Godot 官方说明，运行已编译游戏所需的 .NET 部分会被带入，开发编译才需要额外安装 SDK；项目可使用 NuGet。这里不把“内嵌”限制为单个可执行文件，实际分发物可以是包含本地库的应用目录或 `.app`。[Godot C# 基础](https://docs.godotengine.org/en/stable/tutorials/scripting/c_sharp/c_sharp_basics.html)

## Barracuda、Sentis 与算法的关系

Barracuda 官方定位是 Unity 神经网络推理包，后继为 Sentis。本次查到的官方文档为 **Sentis 2.6.1**，包标识仍是 `com.unity.ai.inference`；从 2.4 起显示名由 Inference Engine 改回 Sentis，所以“Barracuda → Sentis → Inference Engine”不能当作当前名称的完整描述。[Barracuda 介绍](https://docs.unity.cn/Packages/com.unity.barracuda@3.0/manual/index.html)、[Sentis 2.6 概览](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/index.html)

Sentis 的官方工作流、后端和 API 面向 Unity；它支持 Unity 平台，不等于支持任意宿主引擎。其张量接口列出 `Tensor<float>` 与 `Tensor<int>`，且明确存在导入时的数据类型转换。因此，当前 Walk/Roulade 的图内 Float64 不能依据“外部输入输出都是 float32”就认定在 Sentis 中保持原精度。[Sentis 张量](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/tensor-fundamentals.html)、[支持模型及转换限制](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/supported-models.html)

可借鉴 Unity 项目的模型资产管理、固定控制周期、数值对照和独立包验收思路；Godot 中最直接的等价物是内嵌 ORT。无需仅为去掉 TCP 重新训练模型，也无需先把复合策略压成另一种网络。以上是部署结构判断，实际动作是否等价仍由模型与闭环验证决定。

## 版本、架构与平台边界

本地项目当前 Python 环境与 `uv.lock` 为 **ORT 1.29.0**，主机为 Linux aarch64；策略明确选择 CPU。官方最新版本已到 1.29.1（2026-09-10 发布），但本轮建议先与已用版本对齐，以免同时引入运行时升级变量。[项目依赖锁](../uv.lock)、[ORT 1.29.0 发布](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0)、[ORT 1.29.1 发布](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.1)

下表“有包”仅表示官方 ORT 1.29.0 release 中存在对应 CPU 原生发行物，不代表整个 Godot 应用已通过测试；C# 还必须检查所选 NuGet 版本的运行时资产和目标 RID。

| 平台/架构 | ORT 1.29.0 官方 CPU 原生包 | GDExtension 路线剩余工作 |
| --- | --- | --- |
| Windows x86_64 | `onnxruntime-win-x64-1.29.0.zip` | 同架构扩展、依赖部署、干净系统导出包验证 |
| Windows ARM64 | `onnxruntime-win-arm64-1.29.0.zip` | ARM64 Godot/扩展与目标机器验证，不能拿 x64 验证代替 |
| Linux x86_64 | `onnxruntime-linux-x64-1.29.0.tgz` | 锁定目标发行版最低 ABI、库搜索路径及导出验证 |
| Linux aarch64 | `onnxruntime-linux-aarch64-1.29.0.tgz` | 可在当前主机优先验证；依然需要真实导出包 |
| macOS Apple Silicon | `onnxruntime-osx-arm64-1.29.0.tgz` | arm64 扩展/框架、应用内依赖路径及签名验证 |
| macOS Intel | 本次 release 资产未见 x64 包 | 不承诺开箱即用；另行确定发行物或自行构建并实机测试 |

发行物清单来自[1.29.0 官方发布 API](https://api.github.com/repos/microsoft/onnxruntime/releases/tags/v1.29.0)。ORT 构建文档仍提供 macOS x86_64、arm64 以及双架构构建方法，因此“本次发布没有 Intel 包”不等于“算法不支持 Intel Mac”；自行构建的维护和验证成本要单独计入。[ORT 原生构建](https://onnxruntime.ai/docs/build/inferencing.html)

Godot GDExtension 按平台和架构选库，并通过 `[dependencies]` 指定要导出的依赖。扩展需固定 godot-cpp/API 与引擎、导出模板的组合；官方说明旧 minor 目标通常可用于新 minor，反向不保证。不能仅因插件写着“4.6+”就替代本项目 4.7 的导出测试。[`.gdextension` 文件](https://docs.godotengine.org/en/stable/tutorials/scripting/gdextension/gdextension_file.html)、[GDExtension 构建与兼容说明](https://docs.godotengine.org/en/stable/tutorials/scripting/cpp/gdextension_cpp_example.html)

移动端和 Web 应作为独立分支：

- **Android/iOS**：ORT 有专门的原生发行与构建路径；Godot 扩展仍需为目标架构编译并接入平台包。Android 可用 AAR/插件导出方式；iOS 要验证其原生链接与 Xcode 导出。Godot 官方 C# 平台页仍将 Android/iOS 标为实验性支持，不能把桌面 C# 结论直接扩大。[ORT 安装矩阵](https://onnxruntime.ai/docs/install/)、[Godot Android 插件](https://docs.godotengine.org/en/stable/tutorials/platform/android/android_plugin.html)、[Godot C# 平台支持](https://docs.godotengine.org/en/stable/tutorials/scripting/c_sharp/index.html)
- **Web**：不能使用桌面 DLL/SO。可以另研 ORT Web/WASM 与 Godot JavaScript 桥接，或为 Web 编译扩展，但本轮没有证明二者组合可直接使用。Godot Web 扩展需要专用编译和相应跨源隔离设置；官方文档仍不支持 Godot 4 C# Web 导出。当前项目 Forward+ 渲染也不是官方 Web 支持的 Compatibility 路径。[Godot Web 导出](https://docs.godotengine.org/en/stable/tutorials/export/exporting_for_web.html)、[ORT Web 部署](https://onnxruntime.ai/docs/tutorials/web/deploy.html)

## 九模型兼容性意味着什么

本轮独立模型审核得到：九个模型均为固定 `float32[1,61] → float32[1,14]`，均为 opset 18；Roulade 是 IR 10，其余 IR 8；没有自定义算子域和外部权重文件。Walk/Roulade 分别有 36/18 个 DOUBLE initializer，并存在实际 Float64 运算与 Cast；不能只改模型版本标签或统一降精度。详见[九模型运行契约调研](research_embedded_policy_contract_20260911.md)。

ORT 1.29.0 完整 CPU kernel 表对 Add、Sub、Mul、Div、Gemm、MatMul、Relu、Tanh、Clip 等列出 double 支持。因此 **只接收 float32 输入输出的薄包装器可以承载图内 Float64 运算**，只要包装器交给完整 ORT 执行、没有改写图或裁掉类型内核。这是优先保留 ORT CPU 的主要理由；不能推导为任何 GPU、CoreML、移动精简包也覆盖同一组类型。[ORT 1.29.0 CPU 算子与类型表](https://github.com/microsoft/onnxruntime/blob/v1.29.0/docs/OperatorKernels.md)

模型 opset 与 IR 要分别检查：官方兼容表中 ORT 1.14 已列 opset 18，但只列 IR 8；Roulade IR 10 要求更高的兼容组合。社区插件的 1.20.1 在版本上覆盖 opset 18/IR 10，仍不能以版本号代替本项目图的加载和数值测试。[ORT 模型版本兼容](https://onnxruntime.ai/docs/reference/compatibility.html)

初版使用完整 CPU 包更容易保持可解释的差异。若之后压缩体积，ORT 可按算子与数据类型裁剪；基本 minimal build 不支持直接加载 ONNX，需要转换为 ORT 格式。裁剪清单必须来自全部九模型及真实类型需求，之后重做一致性验证。[ORT 定制构建限制](https://onnxruntime.ai/docs/build/custom.html)

## 现成插件的实际证据

审查了 `DynamicDevices/godot-onnx-loader` 的固定提交 `ed6f269dbe913e23e24ffe12cd8df91d218ce4b2`，以及 `joemarshall/godot_onnx_extension` 的固定提交 `4ff6d2a2d840c710322a50f53f47745d6fe4ab8d`。它们是维护者自己的源码，能证明已有实现路径，但不构成本项目质量背书。

较新的 DynamicDevices 候选有明确 float32 张量接口、模型元数据读取、平台依赖声明；CI 配置覆盖 Linux x64、Windows x64、macOS arm64 的原生/Godot headless smoke，锁定 ORT 1.20.1。这比仅有教程片段更可评估，但 CI 中的 fixture 不是 MicroDuck 九模型。[固定 README](https://github.com/DynamicDevices/godot-onnx-loader/blob/ed6f269dbe913e23e24ffe12cd8df91d218ce4b2/README.md)、[固定 CI](https://github.com/DynamicDevices/godot-onnx-loader/blob/ed6f269dbe913e23e24ffe12cd8df91d218ce4b2/.github/workflows/ci.yml)

有两处需要先处理：

1. `OnnxLoader.cpp` 把 `res://` 交给 `globalize_path`，C 侧随后用 `fopen` 读取。它虽然调用 `CreateSessionFromArray`，字节来源仍是普通磁盘文件。由源码推断：**不能据此认定模型仅存在 PCK 中时可用**；需支持 Godot `FileAccess` 读取资源字节，或者明确把模型放在分发目录中的真实路径。[路径处理](https://github.com/DynamicDevices/godot-onnx-loader/blob/ed6f269dbe913e23e24ffe12cd8df91d218ce4b2/src/OnnxLoader.cpp#L43)、[原生文件读取](https://github.com/DynamicDevices/godot-onnx-loader/blob/ed6f269dbe913e23e24ffe12cd8df91d218ce4b2/src/onnx_runtime.c#L534)
2. C 源码中 `ONNX_LOADER_SKIP_SESSION_RELEASE` 默认打开，注释说明用于绕过 Godot/ORT 的 `ReleaseSession` invalid-free。它属于需调查的资源释放约束；高频切换/重新加载模型不能带着这个默认项直接进入生产方案。[释放规避逻辑](https://github.com/DynamicDevices/godot-onnx-loader/blob/ed6f269dbe913e23e24ffe12cd8df91d218ce4b2/src/onnx_runtime.c#L328)

joemarshall 的固定 CI 列出 Linux/Windows x64 和 Android arm64 构建，可以参考 Android 接入；该固定版本不足以证明当前 Godot 4.7、ORT 1.29.0、PCK 导出或 MicroDuck 模型均可用。[固定构建流程](https://github.com/joemarshall/godot_onnx_extension/blob/4ff6d2a2d840c710322a50f53f47745d6fe4ab8d/.github/workflows/build.yml)

## 打包、调度与验证门槛

建议把扩展边界限制在“加载已校验模型、读取契约、执行、释放和报告错误”，将机器人业务状态留在引擎控制层。模型文件通过 Godot 资源接口读成字节再传给 ORT；导出配置要显式包含 `.onnx`、manifest 等非资源文件。原生库通过平台依赖声明导出。macOS 应把新增依赖计入应用签名；Linux/Windows 则验证实际的库查找和运行时依赖。[Godot FileAccess](https://docs.godotengine.org/en/stable/classes/class_fileaccess.html)、[非资源导出过滤器](https://docs.godotengine.org/en/stable/tutorials/export/exporting_projects.html)、[macOS 导出](https://docs.godotengine.org/en/stable/tutorials/export/exporting_for_macos.html)

线程不能照搬 ORT 默认值：默认 intra-op 线程数与物理核数有关，每个 session 又可有自己的线程池。九个常驻模型、多只机器人和 Jolt 同时运行时，默认配置可能争抢资源。首轮测量可从每次同步调用、intra-op=1、顺序执行开始，再比较有限线程数与共享线程池；这只是实验起点，不是已证明最快的设置。[ORT 线程管理](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)

当前模型 batch 固定为 1。多只机器人可以依次执行同一模型，也可以研究受控并发，但每只机器人的上一动作、时间和参考航向必须分开存储。不能只把输入扩为 `[N,61]` 就宣称支持批处理；动态 batch 需要检查整张复合图并单独验证。异步推理引入的动作延迟也会改变闭环，应先保持现有 200 Hz 物理/50 Hz 控制语义，再评估吞吐优化。[九模型运行契约调研](research_embedded_policy_contract_20260911.md)

后续原型应回答四件具体的事：

1. 在真实导出应用里从交付模型包加载全部九模型；没有 Python、虚拟环境、TCP 服务、仓库绝对路径或在线下载也能启动和切换。
2. 固定模型 hash、ORT 版本、执行后端和选项，对随机输入与真实观测逐项比较 14 维输出；建议沿用现有 `1e-5` 严格阈值，任何例外先分析原因。
3. 使用同一物理与控制契约验证连续技能、复位、长时间运行；单步输出相近不足以证明控制时序和闭环行为相同。
4. 在目标设备的 release 导出包记录冷启动、模型切换、内存、控制计算的 p50/p95/p99、物理/渲染帧耗时，以及目标机器人数量下的抖动。20 ms 控制周期不是可全部占用的推理预算，也不能据桌面单机器人结果承诺移动端或百机器人表现。

在这些证据出现前，应把“自有 GDExtension + ORT CPU”标记为领先候选，把具体插件复用、GPU、批处理、移动/Web 作为后续决策。独立运行目标本身已不受 Python 训练工具的限制。
