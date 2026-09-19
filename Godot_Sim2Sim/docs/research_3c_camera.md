# MicroDuck sim2sim — 3C 控制与镜头优化调研

## 现状事实（读码确认，非猜测）

### 输入链路
- Godot `physics_server.gd::_sample_held()` 每渲染帧采物理键 → `_held_now`
- 每个 `step` 响应带 `held/taps/time_scale` 回传 → Python `play.py:261`
  → 下一拍 `brain.tick(held, taps, dt_ctrl)`
- `PlayBrain._set_loco()`：`held_twist()` **直接查表输出满幅值**：
  - fwd → vx = +0.3（约一个控制周期内从 0 阶跃到 0.3 m/s）
  - left → yaw = +1.5 rad/s 阶跃
- standing↔walking 切换阈值 `switch_threshold=0.05`，单点判断、无迟滞

### 镜头
`_follow_camera()`（每帧）：
```
cam.look_at_from_position(look + _cam_offset, look, Vector3.UP)
```
- `_cam_offset` 固定 (0.65, 0.42, 0.65)：相机钉死在世界固定象限
- 鸭子 A/D 原地转 180°：期望机位不变但相对鸭子反了，观感像穿模/背面
- **完全无平滑**：起跳、被推、reset 时相机硬瞬移
- 无交互：鼠标环绕、滚轮缩放都没有

### 其它 3C 毛病
1. W+A（前进+左转）→ vx=0.3 且 ω=1.5 同时满幅，轨迹夸张
2. W+S 同按 → vx=0 且 policy 切 standing，鸭子"犹豫"
3. A+D 同按 → ω=0 但 policy 仍 walking
4. taps 最多滞后一个控制周期 20 ms，可接受，不动
5. `strafe_l` 用 +vmax_y、`strafe_r` 用 vmin_y，与 Q/E 语义一致，OK

## 优化设计（游戏行业标准做法）

### A. 输入平滑 ramp——治"一冲一冲"
`PlayBrain` 内加逐轴一阶 ramp：按住以 accel 升满幅，松开以 decel 回零，
反向按 decel 过零。默认 accel=12 /s^2、decel=20 → 0→0.3 m/s 约 25 ms。
命令仍是 twist（对 ONNX 透明）。

### B. 命令整形——治"W+A 太浪、W+S 犹豫"
- 对角输入合成长度归一到单轴满幅（斜向不超速）
- W+S / A+D 冲突：后按优先（newest-wins），消灭 0 歧义
- stand↔walk 迟滞：>0.10 走、<0.03 停，防阈值抖动

### C. 轨道相机 orbit-follow——治"镜头难受"
重写 `_follow_camera()` 为 Cinemachine 风格最小实现：
- 状态 yaw/pitch/dist + 目标点**指数平滑**：
  `pos = target + orbit(yaw,pitch)*dist`，阻尼跟随 base
- yaw 死区 ±35°：鸭子小幅转向镜头不动，大转向柔和跟上
- 右键拖拽自由环绕，松手 1.5 s 恢复自动跟随
- 滚轮缩放 0.35–2.5 m
- reset 相机快速吸附而非瞬移
- 帧率无关插值 `1-exp(-k*dt)`，兼容窗口 60/144 Hz 与 headless fixed-fps

### 验收自动化
1. `test_play_input.py` 全绿 + 新增 ramp/整形/迟滞用例
2. headless Godot 相机几何集成测试场景：print PASS/FAIL
3. 真窗口 play.py：协议注入 held → 走位
4. `screenshot` 协议截 4 张：走前 / 左转 90°后 / 右键环绕 / 缩放，
   vision_analyze 自审（鸭子在画面内、无穿地、构图），迭代到没问题

## Validation notes (this session, GB10 desktop `DISPLAY=:1`)

Vulkan device creation fails (-3) on this host; windowed Godot runs via the
`SIM2SIM_FORCE_GL=1` → `--rendering-driver opengl3` fallback, with
`SIM2SIM_DISPLAY_DRIVER=x11` pinning the X11 backend.

* Full suite: `PYTHONPATH=src python -m unittest discover` → **55 tests OK**
  (1 skip = this visual item).
* Headless Godot integration `test_camera_follow.py` (real TCP): 7/7 —
  behind-duck framing, teleport & reset snap-glide (~0.4 s, no teleport),
  yaw deadzone, big-turn trail, held_order echo, zoom clamp.
* Windowed render capture (desktop :1, GL): protocol `screenshot` produced
  two real 1280×720 PNGs (home spawn; far corner at -3,-3 yaw 45°). Pixel
  forensics: fully rendered scene, yellow duck body centered in both —
  orbit framing confirmed. No HUD text: correct, `_build_hud()` only latches
  in run mode (`physics_server.gd:1632`), which static captures never enter.
* Real play loop windowed 65 s: lockstep ≈49.9 Hz, godot_step 6.6 ms, no
  step warnings.
* `--disable-vsync` in the windowed spawn args segfaults Godot at GL init
  on this NVIDIA 580.xx driver; the flag now sits behind
  `SIM2SIM_DISABLE_VSYNC=1` (off by default).
* End-to-end key-driven play session (windowed, ONNX walking policy):
  a new protocol `key` command feeds `Input.parse_input_event` — the real
  input pipeline (physical-key tracking for held sampling, `_unhandled_input`
  for taps) — bypassing the X server entirely, which sidesteps the host's
  synthetic-event limitation. Sequence captured in 6× 1280×720 PNGs
  (`shots/viz_*.png`): spawn → hold W 3 s (walks ~2 m, camera trails behind,
  orbit yaw damps to -0.13 rad) → add A 2 s (turn ~130°, camera yaw follows
  to 2.33 rad) → SPACE idle (held gains idle, stop within ~0.4 m, camera
  settles) → re-press (resumes) → reset (camera snap-glides back). held and
  press_order echoed correctly through the real step acks throughout.

Known environment limitation (host-level, not fixed here): on this
GNOME/mutter Wayland-session X server, OS-level synthetic keys (XTEST /
XSendEvent) never reach Godot — a human typing into the real window is
unaffected, and automated key capture works via the in-process `key`
protocol command above (validated in the end-to-end session).
