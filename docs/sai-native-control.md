# Sai 001 本地 ONNX 控制

Sai 的默认驾驶链路现已完全位于 Godot 进程内：Jolt 采样真实刚体和关节状态，GDScript 构造 82 维观测，原生 GDExtension 调用 ONNX Runtime，随后在同一物理帧生成 16 维腿/轮目标。正常启动不创建 Python 控制进程、TCP 监听端口、NumPy/SciPy 或 MuJoCo 运行时。

## 启动

首次安装或资源变化后准备一次自包含运行目录：

```bash
./run-workshop.sh --prepare-only
```

之后直接启动 Godot：

```bash
./run-native.sh --choose-scene
./run-native.sh --scene science_station --robot sai
./run-native.sh --scene workshop --robot sai --task up40
```

桌面入口 `Robot_Godot_Sim2Sim` 也调用 `run-native.sh`。`run-workshop.sh` 仍保留为开发准备入口和显式 Python 参考实现入口。

场景抓取、机械臂 IK 和 cargo 自动任务同样由本地控制器执行：

```bash
./run-native.sh --scene workshop --robot sai --task cargo18
```

自由驾驶时用 B 选择物件、G 抓取入仓、X 取消。任务菜单中的 cargo18/cargo25 运行原版固定抓取路径、夹紧机构和 legacy crawl；不会启动 Python。`--sai-controller python` 仍保留为显式数值参考入口。

## 固定契约

物理积分默认 1000 Hz，策略严格保持 50 Hz；1000 Hz 已通过负载凹凸地面、转向和 20/40 mm 台阶对照。可审计的频率扫描保留 2000/1000/500/250/200/100 Hz 档位，策略调度和时间戳按实际物理频率计算。500 Hz 暂不作为默认值，因为旧货舱皮带约束在该频率失去双侧接触。详见 [频率审计](sai-physics-frequency-audit-20260918.md)。

82 维观测布局如下：

| 范围 | 内容 |
|---|---|
| 0–2 | 机体旋转矩阵的重力方向 |
| 3–5 | 机体坐标系线速度 |
| 6–8 | 机体坐标系角速度 |
| 9–11 | 前进、转向、有效下蹲命令 |
| 12–23 | 12 个腿关节位置 |
| 24–35 | 12 个腿关节速度 × 0.1 |
| 36–39 | 四轮速度、方向归一化 × 0.1 |
| 40–55 | 上一帧 16 维动作 |
| 56–57 | 步态相位 sin/cos |
| 58–81 | 8×3 地形高度扫描 |

动作经过 `[-1, 1]` 裁剪和停车归零，再进入与 Python 参考实现相同的 `targets` / `targets_stairs`、下蹲渐变和 heading hold。稠密轮迹扫描用于决定平地或楼梯 actor；20/40 mm 上下阶使用 `stairs-dev40`，60 mm 实验任务按 profile 选择对应 actor 和速度、抬腿、缩放、最小下蹲、相位、车道与转向限制。

机械臂和轮腿的重力/支撑项不再建立一套并行 MuJoCo FK：每个铰链直接使用 Godot/Jolt 中实际父体朝向、子体质心、质量和世界重力。默认平地策略恢复为已验收的 `flat-motion-v1`，并与其绑定的 `suspension-v2` 一起执行；四轮接触权重、支撑阻抗和 contact-following 下阶状态机沿用原参数。场景抓取沿用原固定路径、停稳靠近、点目标 IK、夹持/释放时序和双货仓槽位，cargo 使用原 `legacy-crawl57` 权重。

## 模型与校验

打包源位于 `src/sim2sim/assets/sai/upstream/`，准备后位于 `res://sai_policy/`：

| 模型 | SHA-256 | 用途 |
|---|---|---|
| `flat-motion-v1.onnx` | `094adb4484b3d812dfb8a056491beda34a23fd0fa6f463b7784854ebc354d53d` | 防抖平地驾驶 |
| `stairs-dev40.onnx` | `0eca6930e7ccd3201023ce9dd0ce4b9cb0476a9490020da89802cac598b2e38d` | 默认上下阶 |
| `experimental/ascent60.onnx` | `501ad1f8e9b6131bc83b9e35012e6f1ad7bcaf7e5869352eefb636440f1e44f4` | 60 mm 实验上阶 |
| `experimental/descent60.onnx` | `821b2cededa8e204a850384b990828d8ac749a07580b6e4ddc563fd437089d57` | 60 mm 实验下阶 |

启动时逐个校验哈希、输入 `[1,82]` 和输出 `[1,16]`。不匹配会以非零状态退出，不回退到网络服务。

Linux ARM64 原生库与 Worlds 发布包可复现构建：

```bash
uv run --no-sync python native/bootstrap.py --jobs 4
./run-workshop.sh --prepare-only
godot --headless --path results/workshop-hub/runtime \
  --export-release "Linux ARM64 Worlds" dist/Robot_Godot_Sim2Sim.arm64
```

`native/bootstrap.py` 固定 Godot 4.7 对应的 godot-cpp 提交与 ONNX Runtime 1.29.0 校验和，生成 ARM64 debug/release 库；打包 preset 会包含本地 ONNX、JSON sidecar、GDExtension 和 `$ORIGIN` 下的 ONNX Runtime 动态库。需要双精度引擎时使用 bootstrap 的 `--double-engine` 与 `--double-godotcpp` 参数生成对应库。

## 验收记录（2026-09-15/16）

```bash
python scripts/check_sai_native_contract.py
./run-native.sh --headless --fast-check --scene science_station --robot sai \
  --plan tests/fixtures/sai_native_smoke.json --output results/sai-native-direct-smoke
```

- Python oracle 对本地 GDExtension：七个固定案例全部通过，含非平地悬挂；观测与动作最大绝对误差 `1.11e-16`，腿目标与轮速最大绝对误差 `3.93e-7`。
- 交互运动门禁全部通过：空闲腿速 RMS `0.004 rad/s`；W 偏航 `0.038 rad`、横移 `0.029 m`；W+A 左转 `0.559 rad`；Shift+W 偏航 `0.035 rad`、横移 `0.033 m`。
- 原生场景抓取完整经过开始、夹持、抬起、释放与入仓，抬升 `0.316 m`；cargo18/cargo25 均夹紧货物并真实接触三段障碍，运输 `2.69/2.60 m`，无出仓或松夹采样。
- 20/40 mm 上下阶四组均清除台阶；下阶轨迹实际进入 `descending` 接触跟随分支。
- 无 Python 直跑：进程链只有 `run-native.sh → Godot`。
- Linux ARM64 已生成 single/double 的 debug/release 四套扩展；构建默认最多使用 4 个 CPU 核。

机器可读的完整摘要见 [`sai-native-control-evidence-20260916.json`](sai-native-control-evidence-20260916.json)。

这些是冻结模型推理、原逻辑移植和物理验收，不是训练。60 mm 仍是实验 profile，不扩大解释为未知地形泛化保证。
