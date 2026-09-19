# 本轮证据复算与恢复

工作目录：`/home/ethan/Projects/MicroDuck/sim2sim`。本轮已完成，先读取 `RESULT.json`、`results/sprint_exit_20260912/active_budget.json`、各 `completed.json` 和资源审计；目录存在不是完成标志，也不要重新启动已完成的训练。

## 保留的内容

- `harness/` 是本轮 22 个脚本与两个命令配置的原样存档，SHA256 见 `harness/SHA256.json`。脚本内部仍指向本轮路径，是证据复现入口，不是允许覆盖封存结果的调度器。
- `motion_pair_protocol.json` 含两条完整训练命令。`run_motion_pair.py` 依次执行固定训练与原生评测，`motion_pair_audit.py` 核对只有 actor 的三维 mask 不同。
- `motion_native.py` 运行 128 加速、128 普通、1 最小回放；原生 runtime、summary、attempt trace 保留在每个候选的 `suite/`。
- `state_contract.py`、`state_shadow.py`、`verify_state_candidates.py` 分别检查包装不改变 anchor、真实控制组成、checkpoint/ONNX/原生数值。
- `torque_comparison.py`、`torque_envelope.py`、`prepare_dynamics_probe.py`、`gpu_dynamics_probe.py` 保留公式、状态投影与 CUDA 物理对照。
- `baseline_all.py` **未执行**；`lateral_basis_command.json` 对应 H6 **未执行**，所需生产原型已撤回。`frame_probe.py`、`speed_budget_probe.py` 的代码原型也已撤回，不能以当前生产代码重跑后称为原实验。对应原型在本地 `projection_prototype/`、`speed_budget_prototype/`、`lateral_basis_prototype/`。

大模型、冻结 runtime、输入轨迹、原始 BAM 源码不进入 Git，保存在本机 `results/sprint_exit_20260912/`；Git 提供实现、报告、脚本和校验值。异机复现必须先恢复这些依赖，不能只 clone 后把缺少输入误判成训练失败。脚本还依赖前轮 joint-trial 包和已存在的本地 microduck_rl 环境。

## 只读检查

```bash
.venv/bin/python -m unittest discover -s tests -p test_policy_state.py
.venv/bin/python results/sprint_exit_20260912/audit_resources.py
```

资源审计只读取已登记任务的进程组/启动时间与容器挂载，不能在本轮仍有监督任务时当作收尾成功。其他任务在共享工作区活动不表示本轮进程遗留。

## 新实验而非覆盖旧证据

先建新的有效工时会话和开发种子；将脚本输出路径、输入模型、配置及命令中的 session 路径统一改为新目录，再启动。当前 924000–924049 未使用，可作为未来冻结后的保留集；不能先拿它调 GPU 接触参数。旧 920000/922000 最终集不能复用调参。

每个耗时构建、训练、仿真或验证都使用预算监督器（以下是结构示例，需填入新会话与已确定的脚本路径）：

```text
.venv/bin/python -m sim2sim.research.budget --directory NEW_SESSION start
  --timeout 600 --reserve 1800 --label gpu_response --cpus 4 --nice 10 --
  env PYTHONPATH=/home/ethan/Projects/MicroDuck/sim2sim/src
  /home/ethan/Projects/microduck_rl/.venv/bin/python NEW_SESSION/gpu_dynamics_probe.py
```

上述排版不是可直接粘贴的一行 shell 命令。实际命令记录在本轮预算 events/supervisor 日志中。GPU 物理使用 microduck_rl 的 MuJoCo/Warp CUDA 环境；普通本地 `.venv` 用于离线 JSON、ONNX 和 Jolt 入口。不能因未安装 CUDA 依赖静默回退 CPU。

严格顺序：确认源轨迹与 physics/spec SHA → 准备投影快照 → 首次跌倒前筛选 → GPU 短时响应 → 独立轨迹对照 → 决定是否进入训练。`prepare_dynamics_probe.py` 使用 `exist_ok=False` 防止复写初始证据。CPU/GPU 同模型接触并不逐位等价，不把它的 qvel 差直接当 action parity；模型推理门槛仍为最大绝对误差 <1e-5。

删除的 staging 副本只在 `prepared_duplicate_cleanup.json` 中记录为 identical 的六项；各条包含 canonical runtime 和恢复命令。保留真实导出包与 canonical runtime 就足够读轨迹和复算评分，无需为了看结果重新复制数 GB 工程。
