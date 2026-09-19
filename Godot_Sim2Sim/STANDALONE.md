# Linux ARM64 独立运行

独立入口在 Godot 进程内完成九技能选择、观测生成和 ONNX 推理。发布目录包含 Godot ARM64 可执行文件、PCK、C++ GDExtension 和 ONNX Runtime 1.29.0；运行时不需要 Python、训练仓库、TCP 服务或网络。验收与模型去留见 [本轮结果](docs/overnight_20260911/RESULT.md)，开发过程见 [执行记录](docs/overnight_20260911/PROGRESS.md)。冻结轮滑候选的最终主动制动为 180/210、零跌倒，尚未通过全部硬门槛。

2026-09-13 行走加速已通过最终平地验收并接入主游戏默认操作，见 [加速结果](docs/sprint_joint_identification_20260913/RESULT.md)。最新独立包为 `dist/MicroDuck-ARM64-20260913-sprint-reversal-trial.tar.gz`，最终加速 400/400、普通 400/400、零跌倒，长直行快 20.4%；宿主 OS 键盘已通过。该包与主游戏默认资源的其他八技能来源分别记录，旧轮滑限制仍存在。

## 使用发布包

解压后保留目录内全部文件，在 Ubuntu 24.04 ARM64 上执行：

```bash
sha256sum -c SHA256SUMS
./MicroDuck.arm64
```

| 输入 | 操作 |
|---|---|
| W / A / S / D | 移动与转向；轮滑 S 为主动制动 |
| 左 Shift + W | 当前默认行走加速，可同时 A/D；松 Shift 恢复普通行走，右 Shift 和轮滑不启用 |
| 空格 | 中性命令 |
| 6 | 切换步行／轮滑机器人 |
| 7 | 步行机器人显式站立 |
| Y | 坐下／站起；轮滑时为蹲起 |
| G | 捡地 |
| K / L | 左踢／右踢 |
| R | 前滚 |
| 0 | 复位当前机器人 |
| F8 | 暂停／继续 |
| Escape | 退出 |

九个模型为站立、行走、坐起、捡地、左踢、右踢、前滚、轮滑、轮滑蹲起。界面根据机器人模式显示可用技能。模型加载、校验或推理失败会报错并结束，程序不会静默换模型，也不会在验收过程中自动复位。

## 构建与导出

构建需要 Linux ARM64、C++17 编译器、CMake、Git、Godot 4.7.2 stable、Ubuntu 的 `fonts-noto-cjk` 和项目 Python 环境。版本与下载校验固定在 [native/dependencies.json](native/dependencies.json)。已晋级的普通行走 / 加速模型对随源码提交于 `src/sim2sim/assets/microduck_sprint_v1/`；其余八技能、生成资源和发布包不提交 Git，须按复现说明另行准备完整资源及 sidecar。

在仓库根目录运行：

```bash
uv sync --extra train
.venv/bin/python native/bootstrap.py --jobs 8
.venv/bin/python -m sim2sim.standalone.prepare --models /absolute/path/to/models --fixture-count 256
.venv/bin/python -m sim2sim.standalone.package dist/my-arm64-build --archive
```

本轮离线参考环境为 Python 3.12.14、ONNX Runtime 1.29.0、ONNX 1.22.0、MuJoCo 3.12.0 和 PyTorch 2.9.1。重建时核对实际安装版本；原生 SDK 与模板按 `native/dependencies.json` 固定，Python 参考环境也应使用同版本 ORT 生成最终数值夹具。

中文界面使用随包附带的 Noto Sans CJK SC，字体文件和原始许可随资源一起打包，并在启动时核验校验值与必需字形；不依赖目标机器安装中文字体。

导出目标目录必须不存在，避免覆盖旧验收产物。`prepare` 用 Python/MuJoCo 预生成机器人初始位姿和 Godot 资源，发布后由引擎读取。使用 `--control-config /absolute/control.json` 指定已经评测的控制候选；缺省保持原控制契约。正式打包应给 `prepare` 增加 `--real-traces`，用冻结模型的真实轨迹补充九技能数值夹具。真实输入只接收模型 SHA 与该技能相同的轨迹，并记录轨迹校验值；最终验收须确认每项都有真实观测覆盖。

对已经冻结的工程目录导出时，给 `package` 增加 `--project /absolute/frozen-project`。它读取该目录中的模型、控制和机器人资源，并把源码与机器人资源校验记入 `build.json`。同一候选的自检、回放、性能测量和最终分发应使用同一个包目录。

行走加速使用 `prepare --sprint /absolute/Sprint_Godot.onnx` 显式加入第十个策略，随附 manifest，保持普通 61 维观测 / 14 维动作。普通九技能仍从 `--models` 读取。仅加入模型不表示质量验收通过；实验来源与物理对照见 [行走加速记录](docs/sprint_20260912/EXPERIMENTS.md)。Python 开发入口对应 `sim2sim-play --sprint ...`。`sprint_vmax_x` / `sprint_vmax_ang` 属于 `control_config.walk.twist_limits`，必须使用与模型一起评测的控制配置，不能用未核验的默认速度运行实验模型。

2026-09-12 试验包为 `dist/MicroDuck-ARM64-20260912-sprint-trial.tar.gz`，使用已冻结模型和控制。长直行较配对普通 W 快约 32%，但完整最终集仅 206/240，单向长测在 222.48 秒走出地板边缘（续轮已诊断，不等于场内策略失稳），未晋级；宿主实际键盘仍待验收。包内其他九模型沿用上一轮候选包，默认工程未覆盖。详见 [冻结结果](docs/sprint_20260912/RESULT.md)。

联合行走续轮的试验包为 `dist/MicroDuck-ARM64-20260912-joint-trial/`，普通行走和加速显式共用S05，分别请求0.25/0.30m/s，配合0.20m路径前视。新最终集231/240、零跌倒，同种子旧包211/240；存在9个原有成功丢失，未晋级。长直行0.2256m/s，比同包普通W快16.8%。其他八模型沿用前一试验包，工作区默认不覆盖；详见 [联合行走报告](docs/sprint_joint_20260912/RESULT.md)。

扩展分别链接 godot-cpp 的 `template_debug` 和 `template_release`，由 `.gdextension` 特征选择。不要把调试绑定库重命名后用于发布模板：两种构建的分配器布局不同。当前 bootstrap 校验 SDK 提交、ORT 下载和匹配模板，生成资源仍使用项目现有导入流程。

## 验证和复现

发布程序内置数值自检，直接比较打包模型与离线 Python 生成的期望动作：

```bash
./MicroDuck.arm64 --headless -- --self-test
./MicroDuck.arm64 --headless --fixed-fps 200 -- --replay=/absolute/case.json --trace=/absolute/trace.json
```

回放 JSON 使用 `mode`、`seconds`、`segments`；每段以 `at` 秒定义 `held`、`taps` 和可选 `order`。测试入口支持 `--seed`、`--roller`、`--render-fps`。物理固定 200 Hz，每四次物理积分后生成下一条观测，策略固定 50 Hz；渲染帧率不改变这个采样顺序。完整轨迹记录请求命令、控制后命令、模型声明的完整观测、上次动作、当前动作和执行目标。

训练环境中的评分、原生回放和 Python 影子对照入口：

```bash
.venv/bin/python -m sim2sim.standalone.suite /absolute/cases --out /absolute/new-suite --executable /absolute/package/MicroDuck.arm64
.venv/bin/python -m sim2sim.standalone.score /absolute/trace.json /absolute/case.json
.venv/bin/python -m sim2sim.standalone.replay shadow /absolute/trace.json --project /absolute/suite/runtime
```

影子对照必须指向轨迹对应的模型／控制快照。新套件保存独立运行快照及模型校验，不把工作区后续改动混入评测。`--resume` 只复用已经完整保存、校验一致的回合；目录存在不代表实验完成。schema 2 同时校验脚本、配置、原生库和模型，旧 schema 1 套件须保留原记录并创建新目录，不能按新协议续跑。发布包的 `build.json`、`models.json` 和 `SHA256SUMS` 记录构建来源、模型和文件校验；验收报告另行记录性能及任务质量。

干净环境验证只挂载发布目录，使用固定 Ubuntu ARM64 镜像、`--network none`、只读根文件系统及临时 `/tmp`。宿主窗口键盘检查是单独的验收项；锁屏时暂停该检查，窗口定向事件与真实全局键盘投递分别记录。

`suite` 支持 `--container-image ubuntu@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254`，须同时指定 `--executable`，且镜像已经存在本机。每个回合只挂载发布目录、该回放 JSON 和输出目录；容器验证无 Python、无训练仓库、禁网。评分在容器外离线完成。超时会清理该回合专用容器，失败记录保留。

## 演示录制

在有图形显示的宿主机上，使用同一个导出包录制声明的命令回放：

```bash
.venv/bin/python -m sim2sim.standalone.demo /absolute/package/MicroDuck.arm64 /absolute/replay.json --out /absolute/new-demo
```

输出保留 Godot 直接生成的 `capture.avi`、便于播放的 `demo.mp4`、逐周期 `trace.json`、命令与校验记录。录制不使用全局键盘注入，不能替代宿主键盘验收。最终制动演示须和相应原始轨迹的评分一并查看。

独立发布配置启用日志逐行刷新，让启动和错误状态及时到达监督器；发布版默认会缓冲日志，这一行为见 [Godot ProjectSettings 文档](https://docs.godotengine.org/en/4.5/classes/class_projectsettings.html#class-projectsettings-property-application-run-flush-stdout-on-print)。性能采样以实际 READY/RESULT 生命周期界定稳态区间，避免将启动或释放阶段的内存下降当成运行趋势。

## 训练和恢复

Python/TCP 保留为离线训练工具；其世界、物理脚本、坐标转换和策略契约与独立入口共享。有效工时采用已确认区间的并集，后台任务计时，并行不重复计时。恢复时先核对 PID、日志、检查点和完成标记，再决定重试；不按断线后的墙钟时间自动补记。

```bash
.venv/bin/python -m sim2sim.research.budget --directory /absolute/session status
.venv/bin/python -m sim2sim.research.budget --directory /absolute/session run --timeout 1900 --reserve 5400 --label experiment -- .venv/bin/python -m sim2sim.standalone.train_trial /absolute/recipe.json --name unique_attempt
```

训练在完整 PPO 更新后原子写入 `latest.pt`。`train_trial --resume /absolute/latest.pt` 在新的尝试名称下恢复参数、优化器、随机生成器和计数，开启新的物理回合。失败尝试保留；仅有训练奖励或 `completed.json` 不能触发模型替换，必须检查配对任务成功、回退和制动硬门槛。

## 实验性状态输入契约

只有明确声明 `sim2sim_brake_state_input=planar_com_velocity_height_v1` 的轮滑候选使用新状态。obs[58:61] 分别为沿当前朝向的平移 COM 速度、侧向 COM 速度（m/s）和躯干 body 高度（m）。坐标计算使用双精度，最后转为 float32；观测总维度仍为 61。

一个输入屏蔽节点让原 ONNX 锚点的这三个位置继续读取零，只有新增残差可见状态。原图节点及内部 DOUBLE 张量保持不变；残差仅在负油门启用，推进和中性动作保留原图。训练、Python Play、离线对照和 GDScript 同步解释声明，未知声明或部署／模型不一致会失败。该契约是否有效以配对实验为准，不能仅凭新增信息推断模型改善。

新一轮路线试验还支持显式声明 `sim2sim_roller_task_input=brake_markov_68_v1` 的 **68 维轮滑模型**。在已有速度／高度输入之上，追加制动计时、连续低速时长、0.2 秒速度历史与左右轮接地，共七维；具体索引及因果采样规则见 [任务学习协议](docs/jolt_learning_20260911/PLAN.md)。旧 61 维模型继续按原契约执行。68 维模型必须同时声明对应元数据，且部署的 `obs_dim`、sidecar 的 `obs_len` 与模型一致。

该任务状态在物理复位和离开轮滑策略时清除。同一仿真时刻的重复读取不会推进计时；没有录制轨迹时也必须提供轮接地传感信息。原生扩展、Python 交互和离线对照均支持这一契约；缺少所需传感信息会使该次运行失败。68 维部署能力本身不构成模型质量改善，候选仍需通过原有配对验收。
