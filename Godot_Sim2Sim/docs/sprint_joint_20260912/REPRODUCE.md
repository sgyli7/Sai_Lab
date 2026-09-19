# 冻结候选与复现

本轮仅供试验，未晋级；已知开发及最终失败均不能通过重选参数掩盖。最终种子922000–922029已使用，旧920000–920029继续封存。另建会话和新种子才可进行后续优化。

## 使用同一已验收包

发布目录为 `dist/MicroDuck-ARM64-20260912-joint-trial/`，归档路径和SHA256见 `RESULT.json` 的 archive 项。

```bash
cd dist/MicroDuck-ARM64-20260912-joint-trial
sha256sum -c SHA256SUMS
./MicroDuck.arm64
./MicroDuck.arm64 --headless -- --self-test
```

左Shift+W加速，仅行走；A/D可同时转向，右Shift和轮滑不加速。其他操作见根目录STANDALONE.md及包内README.txt。锁屏尚未完成的宿主OS键盘验收不能由输入回放替代。

## 原始验收证据

本地会话 `results/sprint_joint_20260912/` 保留：

- `delivery/freeze.json`、`PROTOCOL.frozen.md`、`control.json`：冻结来源和控制；`delivery/runtime` 是带原生扩展和模型的工程快照。
- `delivery/final_cases/`、`final_suite/`：新包480条保留集配对输入和原始轨迹；`final_baseline/` 为旧包相同种子的480条对照。
- `final_engineering/`：独立包普通九技能69条回归、30/144 FPS、复位与六次换机器人、暂停/反复加载、场内长测和性能。
- `runs/j02_joint_short`、`j03_replay_teacher`、`j04_online_teacher`、`j05_native_exit`：完整配置、每轮指标、最终ONNX、导出误差和checkpoint。正式样本分别393216、262144、262144、262144，全部学习器CUDA。
- `teacher/`：训练种子931201–931204的成功观测库与来源哈希；`teacher_pair_matched_audit.json`记录教师约束两组来源/初始actor/参数匹配。
- `joint_contract/`、`lookahead_contract/`、`native_exit_contract/`：训练/原生控制链的500/500/700行对照。
- `boundary_probe/`：旧长回放越过40米地板边缘的四秒位置对照。
- `active_budget.json`、`supervisors/`、`closeout/`：实际工时、超时和进程组监督、包完整性与资源退出记录。

Git保存代码、协议和紧凑报告；体积较大的模型、轨迹、checkpoint及发布包按项目既有规则留在本地交付目录，不包含在源码提交中。不要清掉会话后再声称checkpoint可恢复。

复跑同一个冻结包不会改变模型，但必须使用新的输出目录。例如在仓库根目录：

```bash
.venv/bin/python -m sim2sim.standalone.suite \
  results/sprint_joint_20260912/delivery/final_cases \
  --out results/joint_reproduction_new \
  --executable dist/MicroDuck-ARM64-20260912-joint-trial/MicroDuck.arm64 \
  --container-image ubuntu@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254
```

自动运行应外套新会话的 `research.budget start`，限定四核16–19、nice10、超时；不恢复已收尾的旧会话来隐含追加预算。正式性能测量前停止同会话训练和回放，不终止其他项目的进程。

`prepared_copy_removed.json`只表示该试验的暂存工程与`suite/runtime`逐文件比较相同后去重，权威快照、模型和轨迹仍在；按记录中的rehydrate命令可以恢复副本。目录存在不是任务完成，恢复以completed/result、模型SHA和运行配置一致为准。崩溃后从完整checkpoint开启新物理回合，不能续接半条轨迹。
