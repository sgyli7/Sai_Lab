# 小小维修站：三种机器人，共用一个游戏窗口

首次执行 `./run-workshop.sh --prepare-only` 准备资源，之后用 `./run-native.sh` 启动维修站。F5 / F6 / F7（或左上按钮）在 MicroDuck、MD 轮滑版和 Sai 001 之间切换；窗口和维修站保留，六件可移动场景物件的实例与位置也保留。切换约有短暂加载停顿。机器人位置按相应任务复位，不是在场上同时控制三台机器人。

Sai 沿用 [PR #4](https://github.com/sgyli7/Robot_Godot_Sim2Sim/pull/4) 的 v0.1.0-alpha.3 固定提交 `5d2ab070dcdbae61a8d750a3f812e77ad2568f26`。SO101、四组轮腿、蓝色货仓、碰撞、质量、关节和 ONNX 保持发布版定义。新增的是维修站材质、任务位置和会话管理。

## 应用菜单入口

Linux 应用菜单统一使用 **小小维修站**。模型和原生库准备好后，在当前项目执行一次：

```bash
python3 scripts/install_workshop_desktop.py
```

该入口通过 `scripts/run-workshop-desktop.sh` 调用同一份 `run-workshop.sh`，默认进入 MicroDuck，F5/F6/F7 切换三种机器人。重复点击会唤起已有窗口，不会重复启动游戏。窗口和任务栏使用统一的名称、图标及应用标识。

安装器将旧的「MicroDuck · 小小维修站」和「MicroDuck · 最新策略维修站」快捷方式移出应用菜单，备份到 `~/.local/state/robot-godot-workshop/legacy-launchers/`；若旧图标已固定到收藏栏，也会迁移到新入口。原项目文件保留。移动项目后重新运行安装器即可更新路径。

启动日志默认在 `~/.local/state/robot-godot-workshop/desktop.log`，启动失败时会显示通知。设置 `XDG_STATE_HOME` 时，日志与备份使用对应状态目录。[本机入口验证](workshop-desktop-20260913.json)。

## 操作

| 输入 | 操作 |
|---|---|
| F5 / F6 / F7 | MicroDuck / 轮滑版 / Sai，游戏内加载 |
| W / S、A / D | Sai 前后、转向；MicroDuck 原有移动；轮滑 S 为制动命令 |
| 左 Shift + W（MD） | 默认行走加速，可同时 A/D 转向；松 Shift 恢复普通行走，轮滑不启用 |
| 按住 Shift / 松开 | Sai 下蹲 / 恢复 |
| R | Sai 复位；MicroDuck 保留前滚技能 |
| 0 | 当前机器人与松散场景物件复位 |
| Y / G / K / L | MicroDuck 坐起 / 捡地 / 左踢 / 右踢；轮滑 Y 为蹲起 |
| B / G / X（Sai） | 选择物件 / 抓取入仓 / 取消并松开 |
| B（MicroDuck） | 切换踢击目标 |
| 右键拖动 / 滚轮 | 旋转跟随视角 / 缩放 |
| Tab | 跟随 / 维修站观景 |
| Escape | 退出 |

选中 Sai 后，本地模式提供自由驾驶、场景物件选择与抓取入仓、18/25mm 夹紧运输、20/40mm 上下阶和 60mm 实验上下阶；台阶任务需按 W 驾驶。上述控制均在 Godot 进程内运行，结束后保留画面，可选任务或按 R 重新运行。

```bash
./run-native.sh
./run-native.sh --robot sai --task up60
./run-native.sh --robot sai --task cargo18
```

运送区放在主通道侧面；20/40mm 检修台在院场外侧，60mm 实验台在右侧。阶高、180mm 踏面、四级台阶以及货物障碍尺寸沿用发布测试，不缩放机器人或任务几何。

## 首次准备

当前已验证的平台是 Ubuntu 24.04 Linux ARM64、Godot 4.7.2 / 内置 Jolt。需要 Python 3.12、uv、C++17 / CMake 和 Noto CJK 字体。原生 MicroDuck 扩展目前提供 Linux ARM64 构建入口；其他平台需适配后验证。

先按 [复现说明](../REPRODUCING.md) 准备 MicroDuck MJCF 和选定的完整九模型包（含 `.manifest.json`），再执行：

```bash
export MICRODUCK_RL=/absolute/path/to/microduck_rl
export GODOT=/absolute/path/to/godot
uv sync
uv run --no-sync python native/bootstrap.py --jobs 2
uv run --no-sync python -m sim2sim.workshop_assets --models /absolute/path/to/nine-models
uv run --no-sync python scripts/refine_robot_normals.py
./run-workshop.sh
```

MD 默认安装随源码固定的 S05 普通行走 + a402 加速模型对，自动校验机器人资产、模型和控制配置；不替换其他八技能或 Sai。平地最终加速 400/400、普通 400/400、零跌倒，长直行快 20.4%；接入前后九技能配对 30/69→42/69，无旧成功丢失。柜体碰撞与台阶仍可能跌倒，旧轮滑限制保留；完整证据见 [加速报告](sprint_joint_identification_20260913/RESULT.md)。

准备器用独立 `.venv-sai` 安装锁定的 Sai 包并生成场景；已有训练环境不会被同步或启动。正常游戏由 `run-native.sh` 直接启动 Godot：MicroDuck 与 Sai 模型均在 Godot 进程中由 ONNX Runtime 1.29.0 推理，Jolt 负责刚体、接触和电机受力积分；Sai 的抓取路径、IK、夹紧与 cargo crawl 同样在进程内执行。Python/MuJoCo 仅保留为数值 oracle。Sai 的 82D/16D 契约、模型哈希和测试结果见 [本地控制说明](sai-native-control.md)。

## 物理与验收

MicroDuck 保持 200Hz 物理 / 50Hz 策略；Sai 经负载凹凸地面和 20/40 mm 台阶实测后采用 1000Hz 物理 / 50Hz 策略。Jolt 在创建物理空间时读取求解器设置，因此切换机器人会暂停模拟、更新对应设置并重建物理空间，再迁移同一批场景节点。没有调用游戏重启或场景树重载。Sai 的策略调度由物理频率动态计算，不再依赖写死的 40 个物理 tick。

[2026-09-12 验收、限制与截图](workshop-hub-20260912/RESULT.md) · [15 秒 PV](media/sai-workshop-15s.mp4) · [原片、轨迹与复现资料](https://github.com/sgyli7/Robot_Godot_Sim2Sim/releases/tag/workshop-hub-20260912)

录制使用 Godot 原生视口图像和逐帧墙钟时间戳；原片按这些时间戳编码。15 秒 PV 只剪辑和加速，倍率写入画面和 [剪辑清单](media/sai-workshop-15s.json)。没有位姿动画、搬动被抓物体或更改仿真时间尺度。

新版 PV 使用两个固定机位，前三段为近景，最后运输段直接切远景。位置、朝向、焦距逐帧写入 `hub.json`；编码脚本检查每个机位内只有一个相机变换。[新版原片与轨迹](https://github.com/sgyli7/Robot_Godot_Sim2Sim/releases/tag/workshop-pv-stable-20260912)。

```bash
./run-workshop.sh --robot sai --task cargo18 --plan docs/workshop-hub-20260912/plans/movie-stable.json --record --output results/new-movie
# 在具有 PyAV / Pillow 和 ffmpeg 的媒体环境中：
python scripts/build_sai_workshop_pv.py results/new-movie
```
