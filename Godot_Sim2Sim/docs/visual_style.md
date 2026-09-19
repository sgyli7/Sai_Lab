# MicroDuck 可视化材质与色调

有窗口的原 `sim2sim-play` 入口默认使用 MicroDuck 石墨灰、黄色、紫色配色，细描边与克制排线。原场景中的天空、环境光、主光和棋盘地面改为协调的中性色；棋盘尺寸、UV、场景几何、相机、UI、碰撞与控制均保持原定义。没有引入维修站场景、视觉法线替换网格或旧模型。

```bash
uv run sim2sim-play
SIM2SIM_VISUAL_STYLE=legacy uv run sim2sim-play   # 原材质与色调
SIM2SIM_VISUAL_INK=0 uv run sim2sim-play          # 新配色与着色，关闭描边
```

## 训练边界

`physics_server.gd` 只增加一个启动时的非 headless 条件加载；无 `preload`，没有增加每步或每帧回调。Godot headless 不加载新增样式脚本或 shader，MuJoCo 训练不读取这组 Godot 资源。因此不会给无画面训练增加 shader 绘制开销。首次编辑器资源扫描、可视化评测或同时打开游玩窗口与纯训练不同，不能把这些情况宣称为零成本。

没有更改项目默认场景、Forward+ 后端、抗锯齿、阴影分辨率、线程数、ONNX 设置、模型路径或研究代码。研究溯源记录中的 `physics_server.gd` 文件哈希会因新增入口而变化；物理计算部分保持原样。

机器人部件名称来自独立美术工程的已验证映射，按编译后机器人场景 SHA-256 匹配。未知的新网格版本回退至主体已有的部件分类，避免将旧网格编号错误地套到新模型；所有原始网格及碰撞资源保持不变。

## 验证

- Godot 4.7.2 / NVIDIA GB10，原 Vulkan Forward+ 后端实际加载并绘制成功。
- headless 实机检查：新样式脚本及三个 shader 均未进入资源缓存。
- 材质切换前后，43 个物理／相机节点的序列化属性完全一致。
- 与合并前提交 `0fe8891` 比较：100 个 headless 控制步样本的时间、关节位置／速度、底座位置／姿态最大差为 0。这是状态等价检查，不是吞吐基准。
- 证据保存在 `results/visual_style_merge/`，摘要见 [visual_style_validation.json](visual_style_validation.json)。

集成探针为 `godot/tests/visual_style_probe.gd`。设置 `STYLE_PROBE_DIR` 为已存在的输出目录，以 `godot --path godot --script res://tests/visual_style_probe.gd` 运行；添加 `--headless` 检查训练资源隔离，添加环境变量 `STYLE_PROBE_DEFAULT=1` 检查默认有窗口入口。探针应由外部 `timeout 30` 限时，截图保持原引擎输出。

## 来源

着色器与配色提取自本地 MicroDuck-Atelier 的原创实现（提交 `e6a2411`），沿用本项目 Apache-2.0 许可。此次未复制第三方机器人网格、字体、壁纸或付费资源。
