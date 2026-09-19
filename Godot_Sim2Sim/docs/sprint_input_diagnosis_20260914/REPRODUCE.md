# 复现

研究根目录 `results/sprint_input_diagnosis_20260914/`，起点源码 `6b43936`（桌面工作区另有维修站/Sai UI 修改，精确相关源码在 harness 中）。发布集成基于 `4cb6831`。实际模型 a402 / S05 的 SHA256 见 `evidence/model_control_validation.json`。

`harness/probe.py` 是本轮实际运行的反馈环：在已准备的桌面 runtime 中用 InputEventKey 分别发 W、左 Shift+W、S，测位置与模型选择；它含“后退不快于前进”的检查，当前默认会明确失败。诊断保留了这个失败，不把它改成成功。运行时用唯一新输出名：

```bash
.venv/bin/python results/sprint_input_diagnosis_20260914/probe.py --label unique-new-label --project /absolute/prepared-runtime
```

该脚本只用于本轮已冻结资产，不在其他任务正在使用的 runtime 原地执行。较大的原始轨迹与运行时副本不重复提交 Git；`evidence/local_artifacts.json` 给出保留原件的 SHA256、大小与路径。模型、场景、控制契约缺失时先还原，不替换为别的默认模型。

`harness/gait_analysis.py` 复现全身重心与足端统计；`harness/ceiling_accept.py` 是事前固定的 0.45 命令反事实与既有评分器对照。不要在封存结果目录重新执行这些脚本，先复制到独立可复现目录并核验依赖。相关 Python 模块、固定模型来源与既有重建说明见仓库 `REPRODUCING.md` 和上一轮加速报告。

`host_baseline` 是实际 X11 系统按键的三段分开测量；`continuous_keys` 是引擎事件持续 W、Shift 切换与复位。`host_transition` 是窗口失焦后中止的额外系统按键尝试，不是通过记录。`ui_equivalence` 中正反向模型输出与原始回放完全一致；后退速度仍较高是已保留的诊断结论。首个 HUD 工具参数装配失败也保留在监督日志中。
