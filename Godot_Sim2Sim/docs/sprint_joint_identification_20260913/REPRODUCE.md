# 复现与运行

本轮可恢复记录位于 `results/sprint_joint_identification_20260913/`。`RESEARCH.md` 解释诊断和失败，`RESULT.md`/`RESULT.json` 给出最终采用结论。不要根据文件夹存在就当作任务完成：以各阶段 `completed.json`、监督器终态和封存账本为准。

## 主游戏默认运行

```bash
./run-workshop.sh
```

MD 模式默认左 Shift + W 加速，可同时 A/D 转向，松开 Shift 恢复普通行走；右 Shift 和轮滑不触发。维修站 UI 显示操作及“加速行走”状态。`workshop_assets` 的默认准备及 `workshop.prepare` 启动流程安装随源码提交的 `src/sim2sim/assets/microduck_sprint_v1/` 固定模型和控制；显式实验参数仍走独立资源准备。安装器只替换 walking、增加 sprint，保留其他八技能和轮滑控制，并拒绝不同的机器人物理资产。MD 推理仍在 Godot 内，三机器人游戏里既有 Sai Python 控制器保持原状。

已准备的本地工程可单独安装并检查默认资源：

```bash
.venv/bin/python -m sim2sim.default_sprint godot
```

首次完整准备仍按 [主游戏说明](../workshop-hub.md)，需要原生扩展、机器人资源和其他八技能。该模型对不包含整个主游戏的全部依赖。

## 独立运行

本地包目录：`dist/MicroDuck-ARM64-20260913-sprint-reversal-trial/`。归档为同名 `.tar.gz`。这是加速专项候选；普通九技能沿用已核验的既有候选集合，旧轮滑质量限制没有因此消失。

```bash
cd /home/ethan/Projects/MicroDuck/sim2sim/dist/MicroDuck-ARM64-20260913-sprint-reversal-trial
sha256sum -c SHA256SUMS
./MicroDuck.arm64
```

只需 Ubuntu 24.04 ARM64 及图形运行依赖。模型、原生推理扩展和 ONNX Runtime 都在包内；不需要 Python、训练仓库、网络或 TCP 控制服务。左 Shift + W 为行走加速，A/D 在加速中转向；右 Shift 不触发。0 复位、F8 暂停、6 切换轮滑，轮滑忽略加速修饰键。可重复原生数值自检：

```bash
./MicroDuck.arm64 --headless -- --self-test
```

正式失败复现和修复回放：

```bash
./MicroDuck.arm64 --headless --fixed-fps 200 -- \
  --replay=/home/ethan/Projects/MicroDuck/sim2sim/results/sprint_joint_identification_20260913/integrated_dev/cases/sprint_alternate_927001.json \
  --trace=/tmp/microduck-sprint-reversal-replay.json
```

18 秒演示及其原始 AVI、轨迹和评分位于会话 `demo/`。录像是实际导出程序的声明命令回放；`host_keyboard/result.json` 则是单独向本进程所属 X11 窗口注入 OS 键盘事件的验证，二者不可混称人工试玩。

## 模型与冻结契约

加速模型为此前 GPU 训练的 a402，普通行为 S05；完整 SHA256、侧车和九技能来源在包内 `models.json`。本轮训练代理全部被拒绝，无新策略权重。控制只启用 `walk.twist_limits.sprint_yaw_reversal_s=0.2`，默认值 0 可复现旧输入契约。

`candidate_freeze.json` 在最终评测前记录模型、全部运行时输入文件校验和、控制与种子。开发 927000–927015、最终 928000–928049。最终种子已使用，不再用于候选选择或调参。新研究必须声明新的保留集。

生产接入的 263 条开发验收与 807 条最终验收由 `harness/integrated_eval.py` 完成。首次清单断言失败保存在 `integrated_dev_attempt_01/`；`integrated_attempts.json` 解释了修正。该脚本硬编码本轮结果目录，不能在已封存目录原地重跑；在独立复现 checkout 中还原依赖资产与旧结果，再使用空目录执行。

## 研究证据与依赖

研究起点是 `d357400b56df25d5f7a7482efeccafe9245d6249`。几何、GPU 代理及首个 yaw 原型须使用该起点的 Python 输入实现；当前生产版本的 PlayBrain 已增加关键字参数，不能直接与旧原型混用。`harness/base/` 保存相关起点文件，`harness/` 保存各次精确源码与失败快照。

主要来源包括：

- `results/sprint_stop_state_20260912/native_joint_fd/suite/`：旧 a402 原生基线、模型和原始病例。
- `results/sprint_contact_calibration_20260913/`：同动作 GPU 响应和旧脚位置残差；由对应上一轮文档复现。
- `results/sprint_joint_20260912/final_engineering/nine_new/`：普通九技能的旧成功集合。
- 训练用 GPU 环境 `/home/ethan/Projects/microduck_rl/.venv/bin/python`；项目 `.venv/bin/python` 用于 Jolt 评测、数值参照和打包。CUDA/ORT/Godot 版本与包内 `build.json`、监督日志对应。

`evidence/local_artifacts.json` 列出原始轨迹、NPZ 和媒体的绝对路径、大小与 SHA256；大文件不重复提交 Git。可独立读取的小记录直接存 JSON，大摘要压缩为 `.json.gz`；病例原文集中在 `evidence/replay_cases.tar.gz`。检查旧原始数据校验值后才可复用，缺失依赖不能以空数据或新种子替代。`evidence/closeout/source_validation.json` 检查各诊断源码快照及冻结模型/物理文件。

长任务一律使用有效工时监督器，四核、nice 10、有限超时和退出清理。开发阶段保留最后 20 分钟用于收尾。恢复不能重启旧的已完成训练；先查 `active_budget.json` 活动任务、进程启动标识与最后完整检查点。退出审计只处理本轮所属进程，不处理其他任务的 Sai/workshop 或 MuJoCo 查看器。

## 回归检查

`evidence/test_protocol.json` 列出 306 项主回归；另外新增导出快照 1 项和默认安装 3 项，共 310 项，308 通过、2 项可选跳过。合并源码核心结果见 `evidence/closeout/merged_tests.json`，四项额外检查见 `evidence/closeout/focused_tests.json`。测试导入前去掉 `SIM2SIM_RESEARCH_DIR`，避免把新会话目录误当旧测试 fixture；保留 `SIM2SIM_ACTIVE_BUDGET_DIR` 监督标记。

完整产品证据还包括：原型/生产逐帧一致、Python 原生影子、无 Python/无仓库/禁网容器、九技能配对回归、30/144 FPS 生命周期、实际导出程序的宿主键盘、长时间场内操作、推理与内存记录。每个结论只适用于其实际运行的病例和模型集合。
