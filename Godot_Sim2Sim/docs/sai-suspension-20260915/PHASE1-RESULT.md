> 状态修订：用户要求进一步验证载货柔顺性。以下为第一阶段有限范围结果，未合入主线，默认启用已撤回；不代表载货悬挂目标完成。

# Sai 起伏地面悬挂：训练与部署结果

2026-09-15。第一阶段试验覆盖科学站／维修站的 Sai 001 自由驾驶控制器：连续地形滚动并逐轮调整悬挂，真实台阶保留抬腿动作。候选参数在有限空载范围通过 MuJoCo 和 Godot/Jolt 独立验收。不是宣称所有地形的全局最优策略。

## 原因与实施

原控制器把轮迹扫描的总高度差超过 4 mm 当成台阶。因此缓坡也会将 0.5 m/s 命令压至 0.12 m/s，并叠加脚本定义的 55 mm 周期抬轮。先运行真实控制器回归，连续坡面用例失败，再实现和验证修复。

新控制器保留原始 82D 观测、flat/stairs ONNX、轮速、航向控制及物理参数，增加 20 mm 间距近场扫描，用局部阶跃相对邻近坡度判断真正边缘。在滚动状态，根据逐轮地面相对高度与实际车身姿态补偿腿 IK。行程范围 -25/+20 mm，每 20 ms 最大变化 3 mm，IK 下伸长度约束在 120–195 mm。公共地面升降由真实轮地接触带动车身，悬挂补偿只处理差异，不强制固定世界高度。

本轮训练是**有界悬挂参数的动力学优化**，不是 PPO 或神经网络重训：三组固定训练种子，15 个候选、45 个完整关节物理回合，三轮有界交叉熵参数搜索，用时 30.62 秒；模型构造、分析、原生评估和录像时间不含在内。最优候选地形增益 1.07769、姿态增益 0.41255、滤波常数 0.08415 秒。随后冻结，未根据留出结果继续调参。[原始候选记录](training-trials.jsonl)保留全部参数、得分和分地形指标。

## 独立验证

六组留出地形 × 三种控制方式 × 两个引擎，共 36 个回合。下表为三组未参与训练的起伏地面在原生 Jolt 中的均值；所有组请求速度均为 0.5 m/s。

| 指标 | 原控制器 | 仅修分类 | 分类＋训练悬挂 |
|---|---:|---:|---:|
| 实际前速（m/s） | 0.146 | 0.485 | 0.485 |
| 俯仰/横滚角速度 RMS（rad/s） | 0.304 | 0.196 | 0.129 |
| 竖向加速度 RMS（m/s²） | 3.779 | 0.210 | 0.155 |
| 超过 3 mm 容差的离地量 RMS（mm） | 26.19 | 3.26 | 0.78 |
| 四轮同时接触占比 | 1.0% | 19.5% | 52.7% |

悬挂相对“仅修分类”在不降低平均速度的同时，角摆动下降约 34%，离地误差下降约 76%，竖向加速度下降约 26%。MuJoCo 中总体方向一致；其中一个起伏种子的竖向加速度略升，因此不宣称每个指标在每个地形均改善。接触由引擎实际接触统计，离地量由轮心高度减逐轮地面和轮半径计算；不是要求每时每刻四轮都接触。

此外：

- 平地输出/运动无可测退化，连续纵坡、横坡通过。
- 20/40 mm 上下台阶四项全部通过原有完整阶梯验收器；测试没有强制 `stair_course`，实际自由驾驶探测器触发了跨阶。
- 另六项原生回合覆盖不同初始朝向、倒车、转向、低姿行驶、蹲下再起身和每回合停步。
- 科学站真实碰撞地面（出生点 x=-18,z=5，同一输入）原版 0.137 m/s、持续台阶模式；候选 0.504 m/s、全程滚动，2–6 秒窗口四轮持续接触。两组测试均为 449 个控制采样。不是对用户视频逐帧重放，也不代表科学站所有位置。
- [51 项物理验收条件](acceptance.json)全部通过；[12 项单元/控制器回归](units.log)通过；构建产物包含原 ONNX、几何模块、悬挂模块与训练参数。
- [科学站实际回放](science-station.mp4)按记录的模拟时间重建到 30 FPS，无动作生成或插帧。原始 266 帧，8.93 秒模拟跨度，8.91 秒墙钟跨度，丢弃 2 个捕获请求。可见回放在退出时有一次 Godot `scenario_remove_viewport_visibility_mask` 渲染清理警告；进程正常结束，完整控制轨迹已保存，没有控制脚本报错。

## 使用与复算

第一阶段候选需通过实验配置显式加载 `src/sim2sim/assets/sai/suspension-v1.json`，无需环境变量。已有运行中的进程需重新启动才能加载新的 Python 控制器。上游包单独的诊断入口未改版；本次交付针对用户视频中的项目场景入口。

从项目根目录，以安装 Sai extra 的 Python 运行：

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_sai_*.py'
PYTHONPATH=src python scripts/sai_suspension_experiment.py --train --out results/sai-suspension-retrain
PYTHONPATH=src python scripts/validate_sai_suspension.py --profile results/sai-suspension-retrain/candidate.json --out results/sai-heldout-cpu
PYTHONPATH=src python scripts/validate_sai_suspension.py --native --profile results/sai-suspension-retrain/candidate.json --out results/sai-heldout-jolt
PYTHONPATH=src python scripts/accept_sai_suspension.py results/sai-suspension-20260915
```

完整原始轨迹和日志位于项目 `results/sai-suspension-20260915`；[来源校验](provenance.json)、[阶梯验收](stairs-acceptance.json)、[实施前方案](PLAN.md)、[一手研究](RESEARCH.md)在此目录。

边界：新密集扫描仍是仿真碰撞射线；没有实现真实相机高度估计。此次不覆盖高台跌落、传感器缺测、极窄障碍、负载随机化和 60 mm 实验台阶的全量复验。悬挂行程有限；不能推断任意障碍可按 0.5 m/s 通过。
