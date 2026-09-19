# MicroDuck · 小小维修站

2026-09-10 选定九模型包接入原创维修站场景。Python 运行 ONNX，Godot/Jolt 提供真实物理和画面。

[观看 1080p PV](media/microduck-service-bay-pv.mp4) · [完整原片与轨迹](https://github.com/sgyli7/MicroDuck-Godot-Simi2Sim/releases/tag/atelier-20260910-short) · [实验报告](../RESEARCH_RESULT_20260910.md)

![维修站九项策略剪辑](media/microduck-service-bay-15s.gif)

2026-09-11 背景已扩展为有碰撞的机械小院，见 [新画面、通行检查与前后对照](workshop_atmosphere.md)。上方九技能剪辑保留原版场景与固定演示协议。

## 本次展示了什么

| 能力 | 选定模型 | 这次固定初态演示 |
|---|---|---|
| 站立 | 原版 alpha_stand | 10 秒稳定站立 |
| 行走 | WD05 / iteration 90 | 前进、转向、停止，保持站立 |
| 坐起 | 既有 Sitstand_Godot | 12 秒坐下、保持、起身 |
| 低头拾取 | 原版 alpha_ground_pick | 完整低头与恢复动作；没有新增抓取物品任务 |
| 左踢 | K10 / iteration 81 | 左脚触球，球向前运动，保持站立 |
| 右踢 | KR06 / iteration 43 | 右脚触球，球向前运动，保持站立 |
| 前滚 | R11 / iteration 145 | 倒置、前滚一周、自主恢复站立 |
| 轮足 | 原版 roller | 推进与滑行可见；刹车未通过，有反向滑动 |
| 下蹲滑行 | 原版 roller_crouch | 下蹲、滑行、起身 |

八项通过各自的单次演示协议；轮足连续推进／滑行／刹车未通过。这不是九项都已训练成功，也不是跨种子成功率评估。
本次统一 seed 61000，关闭初态随机扰动，技能前由站立／轮足策略稳定 1 秒，之后按原定义运行完整技能周期。
轮足下蹲沿用原任务的初始 0.3 m/s 速度。没有技能中途重置、位姿动画、瞬移恢复或视频加速。
前滚统计中的 `fell=true` 表示经历了倒置姿态，应结合有序前滚事件与最终站立判定，而非单看这个通用标志。

20 秒、1920×1080 的 PV 与约 13 秒首页 GIF 已提供；三组动作特写也都短于 15 秒。
PV 与 GIF 均为动作精选：开场直接翻滚，删去等待与保持段，并适当裁切远景，让机器人动作更清楚。完整周期保存在原片 ZIP。画面上的轮足说明同步标记刹车问题。
原片按实际采集时间戳编码，30 FPS 的成片会重复未产生新画面的帧，不会把低帧率伪装成加速动作。视频无声。

## 运行最新模型的维修站

使用独立检出目录；不要在正在训练的目录里同步依赖、导入资源或启动预览。准备 MJCF、九模型包与四个生成场景见 [复现说明](../REPRODUCING.md)。

```bash
# 在自己的独立检出目录中
export MICRODUCK_RL=/path/to/microduck_rl
uv sync --extra train --extra media
.venv/bin/python -m sim2sim.research.setup scenes
# 可选：只生成视觉法线；不改碰撞或原网格的顶点／三角形
.venv/bin/python scripts/refine_robot_normals.py
# 导入新生成的法线及本仓库原创美术资源
"${GODOT:-godot}" --headless --path godot --editor --import --quit

# --bundle 指向解压后的、含 bundle.json 与 models/ 的 delivery_v3
./scripts/play_atelier.sh --bundle /path/to/delivery_v3
./scripts/play_atelier.sh --bundle /path/to/delivery_v3 --roller
./scripts/play_atelier.sh --bundle /path/to/delivery_v3 --tour
```

默认包位置 `bundles/delivery_v3`，不随 Git clone 下载。入口先核验九个 ONNX 与 manifest、61→14 输入输出及有限推理，再调用当前 `sim2sim.play`。
原生 W/S/A/D、Q/E、空格、1–6 技能／轮足切换、0 重置及滚轮缩放保留；6 切换机器人后保留维修站场景和模型包路径。
右踢、前滚使用 manifest 声明的时间输入，右踢还使用相对起始航向。没有回接旧版控制脚本。

游玩默认 30 FPS，低优先级、最多两个可用逻辑核。同一个检出目录只允许一个预览。
Git worktree 默认观察原检出目录的训练进程；也可设置 `SIM2SIM_TRAINING_ROOT=/path/to/training/sim2sim`。
无法可靠比较训练吞吐、连续两个一分钟窗口下降超过 5% 或出现资源压力时，预览停止。
不要把同时运行有画面预览理解为 GPU 零竞争。所有缓存、日志和生成文件保存在独立检出目录。

## 复现采集与剪辑

采集依赖 `train` 与 `media` extra；剪辑另需 FFmpeg 的 libass、libx264、palettegen、paletteuse。
每段连接成功后最多采集 50 秒，含启动的外层上限 55 秒，策略最多运行 24 个模拟秒。实际本次最长技能 12 秒。

```bash
source scripts/showcase_env.sh
# 以下步骤顺序执行；有其他训练时指定 SIM2SIM_TRAINING_ROOT
for skill in standing walking sitstand ground_pick kick_left kick_right roulade roller roller_crouch; do
  nice -n 10 taskset -c 18,19 .venv/bin/python scripts/capture_showcase.py \
    --skill "$skill" --tag "04_$skill"
done
# 将 18,19 换成本机两个可用逻辑核
nice -n 10 taskset -c 18,19 .venv/bin/python scripts/verify_showcase_physics.py --prefix 04
nice -n 10 taskset -c 18,19 .venv/bin/python scripts/build_showcase_media.py --prefix 04
```

每段保存模型 hash、真实轨迹、原始帧的时间戳、渲染后端、丢帧记录、运行日志和资源监测。输出目录已存在则拒绝覆盖。
[剪辑清单](media/edit.json) 给出原片 hash、截取范围和章节；[运行摘要](showcase_runs.json) 给出九次实际结果。
完整证据及原片在 Release 附件中。没有发布 ONNX、NC 机器人网格或本地生成缓存。

## 隔离与视觉迭代

当前物理服务器和选定模型来自 `da50221` 的最新研究代码，工作区为独立 Git worktree。
原 `main.tscn` 及默认入口保持原定义；维修站通过可选 scene 参数接入。
新版维修站默认启用固定设施碰撞与六个轻质量动态刚体，中央维修坪仍为原高度平面。
使用 B 切换踢击目标，0 同时归位物件；静态台阶改为可选 `--steps`。详见 [动态物件](loose_props.md)。
使用 `scripts/play_atelier.sh --flat` 返回历史无道具碰撞演示模式。详见 [碰撞与闪烁修复](workshop_contacts.md)。

1. 资源导入缓存缺失的首次采集弃用，补齐独立缓存。
2. 九项完整录制，检查动作、相机和模型协议。
3. 发现长距离轮足路线经过装饰物，开放右侧出口、连同货箱标签和机械细节一起移动。
4. 重新采集最终版本，逐帧检查主要动作，制作真实速度的章节 PV 和 GIF。
5. 根据反馈将 59 秒 PV 缩至约 20 秒、33 秒 GIF 缩至约 13 秒；所有动图验证多帧变化与 15 秒时长上限，首页另提供原始 GIF 直达播放。

以下是历史平地演示的验证，适用于 `--flat` 和原九技能 PV，不用于宣称新版障碍场景轨迹相同。

与原平地 headless 世界复跑相同九项，合计 3,300 个控制步的动作、关节、接触、底座运动及任务遥测逐项完全一致。
详见 [轨迹等价结果](showcase_physics_equivalence.json)。步行与轮足独立入口各进行了 12 秒限时启动，均加载候选包并报告 50 Hz 控制循环；限时退出的 KeyboardInterrupt 属于测试主动中断，确认没有遗留 Godot 进程。
已有 38 项键盘控制测试与 8 项模型包／准备测试通过。最终 MP4 与四个 GIF 全帧解码通过，时间戳严格递增；尺寸、时长及校验值见 [媒体清单](media/manifest.json)。
这是这九条轨迹的等价检查，不代表吞吐基准或所有可能输入的形式化证明。
不在共享训练窗口声称达到 1080p 60 FPS；本次按 30 FPS 采集上限制作。

## 素材来源

- 维修站几何、shader、图标、服务单与画面排版：本项目原创，Apache-2.0。
- 机器人设计、MJCF 与网格：Pollen Robotics [microduck_rl](https://github.com/pollen-robotics/microduck_rl)，使用本地参考快照；网格不随本仓重新发布。
- Noto Sans CJK：SIL Open Font License，字体和完整许可位于 `godot/atelier/fonts/`。
- 莫比乌斯式细线机械表达是美术方向；未复制用户参考壁纸、第三方游戏画面、音乐或付费素材。
