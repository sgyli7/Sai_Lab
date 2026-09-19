# 本轮交付与复现

## 运行独立包

最终构建目录为 `dist/MicroDuck-ARM64-20260911-candidate/`，参考目录为 `dist/MicroDuck-ARM64-20260911-incumbent/`，各有同名 `.tar.gz`。两者均包含全部九模型。candidate 带已冻结的行走控制和实验轮滑策略；incumbent 保留旧模型与原控制。模型去留以 [RESULT.md](RESULT.md) 为准。

在 Ubuntu 24.04 ARM64 上解压，进入相应目录运行：

```bash
sha256sum -c SHA256SUMS
./MicroDuck.arm64
./MicroDuck.arm64 --headless -- --self-test
```

运行不需要本项目、Python、MuJoCo、训练服务或网络。W/A/S/D 或方向键操作，6 切换机器人，7 显式站立，Y 坐起/蹲起，G 捡地，K/L 踢球，R 前滚，0 复位，F8 暂停，Esc 退出；界面也提供按钮。轮滑短推进制动仍有未通过用例，不能把实验包当作已完成质量验收的版本。

## 证据目录

原始会话根目录为 `results/research_20260911/`：

| 内容 | 相对会话根目录的路径 |
|---|---|
| 实际工时、全部任务起止及中断 | `active_budget.json` |
| 最初协议、旧模型校验 | `protocol.json` |
| 候选冻结和冻结时点审核 | `final_freeze.json`、`freeze_timing_audit.json` |
| 690 条最终输入及 SHA | `final_cases/`、`final_cases_sha256.json` |
| 最终旧版 / 候选完整轨迹与评分 | `final_holdout_incumbent/`、`final_holdout_candidate/` |
| 最终配对和组件去留 | `final_holdout_comparison.json`、`final_quality_decision.json` |
| 标称平地回放 | `final_nominal_cases/`、`final_nominal_incumbent/`、`final_nominal_candidate/` |
| 原始制动前后曲线、CSV | `final_brake_comparison/` |
| 制动和九技能连续演示 | `final_demo_brake/`、`final_demo_nine_skills/` |
| 干净环境及数值自检 | `final_numeric_incumbent/`、`final_numeric_candidate/` |
| 性能与长期运行 | `final_performance_headless/`、`final_performance_fps_30/`、`final_performance_fps_144/`、`final_realtime_soak/` |
| 宿主输入及其适用范围 | `final_host_desktop_state.json`、`final_host_input/` |
| 异常模型 / 动作检查 | `final_native_failures/` |
| 九项首轮比较及全部去留索引 | `nine_skill_first_round_comparison.json`、`experiment_decisions_index.json` |
| 可搬运模型银行及来源审计 | `delivery/` |
| 提交前最终完整测试（晚于证据包冻结） | `final_unit_suite.log`、`final_unit_suite_result.json`、`final_unit_suite_sources.json` |

每个质量回合的 `completed.json` 只表示运行和评分完整；任务是否成功看 `task_metrics.success`，主动制动看 `brake_success` 和 `brakes`。轮滑通用跟踪评分与专用制动门槛是不同指标，不混用。失败记录、旧评分版本、崩溃尝试均保留，不用目录存在代替完成标记。

## 重放最终输入

以下命令在训练工作区执行，Python 仅作为离线评分器；真正回放仍由独立程序完成。输出必须使用新的目录：

```bash
.venv/bin/python -m sim2sim.standalone.suite \
  results/research_20260911/final_cases \
  --out results/replay-final-candidate \
  --workers 8 \
  --executable dist/MicroDuck-ARM64-20260911-candidate/MicroDuck.arm64 \
  --container-image ubuntu@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254
```

固定 Docker 镜像需预先可用。套件以只读根目录、禁网、无 Python 的容器运行，只挂载发布包、一条回放和该条输出目录。宿主离线评分使用当前仓库的 MuJoCo 元数据及 `physical_tasks_v8`；物理积分发生在容器里的 Jolt，不在 Python 评分器中。

单个原始制动用例可直接重放，无需 Python：

```bash
/absolute/package/MicroDuck.arm64 --headless --fixed-fps 200 -- \
  --replay=/absolute/roller_brake_3s_nominal.json \
  --trace=/absolute/new-trace.json
```

最终 916000–916029 已经使用，重放用于复核本轮结果，不能再次当作下一轮的未见测试集。

## 重建和恢复

构建步骤见 [STANDALONE.md](../../STANDALONE.md)。使用 `delivery/candidate_models/` 或 `delivery/incumbent_models/` 的九个 ONNX 及 sidecar；候选还需传入 `final_candidate/control.json`。数值夹具应加入对应模型的 `final_candidate/integration_trace.json` 或 `final_baseline/integration_trace.json`，确保九项 `real_count` 都达到 64。

```bash
.venv/bin/python -m sim2sim.standalone.prepare \
  --models results/research_20260911/delivery/candidate_models \
  --control-config results/research_20260911/final_candidate/control.json \
  --fixture-count 256 \
  --real-traces results/research_20260911/final_candidate/integration_trace.json
.venv/bin/python -m sim2sim.standalone.package dist/rebuilt-candidate --archive
```

`build.json` 记录构建时的 Git 基点和包含未提交文件的源码 SHA；收尾提交另由交付清单关联。不要仅根据基点提交号猜测构建内容。重建后的导出时间戳、容器输出路径和包校验可能不同，应重新执行数值与回放检查；冻结的分发包本身则按完整 SHA 验证。

统一账本记录确认工作区间的并集，断线但监督任务仍运行的时间计入；完全停工的空档不计。队列支持完整更新后的原子 checkpoint 恢复和有界重试。本轮已经冻结并使用最终种子，不应恢复搜索或继续调参；后续研究必须创建新会话和新的保留种子。旧失败尝试只保留证据，不自动重跑。

证据包另外保存 20 个 `latest.pt` 完整训练检查点和训练配置引用的 ONNX 锚点。换工作区时，按 `delivery/recovery_models/path_mapping.json` 重定位 `source`、`template`、`entry_source`，并验证 SHA；不要仅复制残差 checkpoint 而遗漏其原模型。训练仍需要本项目和构建依赖，独立应用本身不需要这些文件。证据包内的工时是打包开始时的快照，最终收尾工时另见交付回执和工作区原始账本。
