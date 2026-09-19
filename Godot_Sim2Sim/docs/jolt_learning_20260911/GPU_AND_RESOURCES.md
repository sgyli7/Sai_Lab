# GPU 学习与后台资源修复

## 为什么先前消耗 CPU

R0–R3 使用冻结的历史训练器：真实 Godot/Jolt 物理采样、ONNX 锚点推理、演员和 critic 的梯度更新均在 CPU 上。GB10 和 CUDA 实际可用，网络更新留在 CPU 是训练器的实现缺口，不是没有 GPU 或必须用 CPU 才能保留原模型精度。

2026-09-12 凌晨检查到 32 个 Godot 都属于仍在进行的两个训练任务，每个 16 个环境，没有查到本任务的孤儿 Godot。这些 headless 进程没有桌面窗口。随后将该训练进程组的所有线程限制在 CPU 16–19、nice 10；区间采样合计约 329% 单核占用，即这台 20 核机器总容量的 16.45%。系统中其他项目的 MuJoCo／f3d 和桌面服务未被停止。

## 当前实现

`research.train --learner-device auto` 在 CUDA 可用时默认使用 GPU 更新演员和 critic，显式 `cuda` 不可用或运行失败会报错。`cpu` 可用于固定对照。Jolt 采样和冻结 ORT 锚点仍在 CPU；采样期间使用上一次完整更新同步出的 CPU 策略／critic，整批样本送入 GPU，再同步下一轮采样器。

68D 演员内部保留 float64、critic 保留 float32，禁用 TF32；原 ONNX 锚点没有近似重实现，也没有量化。导出时复制学习增量到 CPU，保留原 ONNX 图并照常做 `1e-5` 门禁。checkpoint 保存模型、优化器及 CPU／CUDA 随机数状态，设备迁移记录在配置中；恢复仍从新物理回合开始，不宣称跨设备逐位相同的训练轨迹。

这不是 GPU 物理仿真。只把网络迁到 CUDA 无法消除 Jolt、逐步 Python 编排和通信成本。游戏运行时继续保持 Godot/Jolt 和 CPU ORT。若后续需要大规模 GPU 物理采样，应独立评估离线训练域与 Jolt 校验之间的差距，不应直接替换游戏物理。

## 验证及限制

- 实际 GPU 参数更新与 CPU 采样器同步的回归测试通过；ORT 会话保持共享且未被复制。
- `cuda_integration_smoke_retry` 完成 192 个转移、3 次 CUDA 更新，`cuda_resume_smoke` 从其 checkpoint 再完成 128 个转移、2 次更新；两次导出动作误差为 0，通过严格门禁。另从 CPU 计时试验 checkpoint 迁到 CUDA 完成 4,096 转移、一次更新，优化器恢复和导出门禁同样通过。它们只验证工程路径，不参与选模，其通用 `brake` 内评不是正式键盘制动验收。
- `gpu_network_microbenchmark.json` 的单网络更新中位数为 CPU 28.00 ms、CUDA 2.79 ms。CPU 当时与旧训练争用四核，且不含物理采样，不能用这个约十倍数字声称整轮训练十倍加速。完整流水线的串行对照另存 `pipeline_profile_complete.json`，结果见下表。
- 第一次同时启动 GPU 单测和小训练时 CUDA 分配失败。GB10 使用统一内存，当时 Linux 有大量可回收页缓存；记录保留在原失败目录／日志。仅对本项目已完成的 2,146 个大轨迹文件调用 `POSIX_FADV_DONTNEED`（8.73 GiB 文件范围，无内容删除），可用空闲内存随后增加。改为串行启动后重试通过。缓存回收和并发方式都变了，不能据此唯一证明故障原因。没有修改驱动、全局内存参数或清理其他项目。

完整流水线使用相同初始模型、训练种子、四个 Jolt 环境、每更新 4,096 转移、minibatch 1,024、两轮优化；CPU 后 CUDA 串行运行，每侧六次更新，排除前两次预热。测量时没有其他本项目训练或回放任务，桌面服务仍正常运行。只有一组计时对照，不宣称统计显著性。

| 中位数／总耗时 | CPU 学习 | CUDA 学习 |
|---|---:|---:|
| 每轮采样 | 5.417 s | 5.472 s |
| 每轮学习 | 0.126 s | 0.196 s |
| 每轮采样器同步 | <0.001 ms | 0.602 ms |
| 完整更新 | 5.551 s | 5.678 s |
| 整个进程（含初始化、评测与导出） | 47.47 s | 55.69 s |

当前小网络／小批量路径没有获得 GPU 整轮加速；约 97% 的稳态 CPU 版本时间在采样。这个区间包含 Jolt、Python 编排、TCP 和 CPU ORT，还没有拆出各自占比，不能说 97% 都是 Jolt 内核计算，也不能保证单改通信就能消除它。算子追踪另显示，每五次教师损失抽样触发 20 次动态 `nonzero`，并有频繁小 kernel 启动。该追踪与原生验证并行，只用于定位调用结构，不作吞吐数字。可缓存固定教师分组并减少设备同步，但这里的最大收益受采样占比限制，不值得将 GPU 利用率本身作为训练目标。下一轮大规模加速须先处理采样组织，或引入独立验证的 GPU 离线训练域；现有 CUDA 路径是可用能力，不能包装成已提速成果。

统一内存的排查依据为 [NVIDIA DGX Spark 已知问题](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html) 和 [NVIDIA 内存不足说明](https://nvidia.custhelp.com/app/answers/detail/a_id/5776)。实际 CUDA 运算和训练已通过；本机 PyTorch 的架构范围警告不等价于这些实测不支持。

## 监督和收尾

旧监督进程曾随应用中断消失，而独立进程组中的训练仍在运行。已依据同一 PID 起始时间和连续 checkpoint 时间恢复有效时长，证据在 `supervision_recovery_audit.json`；没有把完全停工时间计入工作。

新长任务使用 `research.budget start`：监督进程本身独立运行，关闭启动终端后仍记录心跳、执行超时与预算截止。日志和监督 PID 在会话的 `supervisors/` 及账本中。前台 `run` 继续可用。两种入口默认限制四核、nice 10，Godot 绑定只从父进程允许的核心中选择，不能自行扩展到整机。容器回放也显式传入同一 `--cpuset-cpus`，不会因 Docker 守护进程另起进程而丢失限制。

监督任务正常结束、报错、超时或收到中断时，均检查自身创建的进程组，先 TERM、必要时 KILL，并记录剩余进程；PID 起始时间不同则拒绝误杀。正常退出后遗留子进程、无关任务不受影响、默认资源限制以及启动器退出后监督仍完成，均有实测回归。不可捕获的 supervisor SIGKILL 或机器故障仍须通过账本核对恢复，不承诺进程监督无法失效。

长任务示例（另建会话，不向封存会话添加训练）：

```bash
.venv/bin/python -m sim2sim.research.budget --directory results/new_session init
.venv/bin/python -m sim2sim.research.budget --directory results/new_session start \
  --timeout 1800 --reserve 5400 --cpus 4 --nice 10 --label experiment \
  -- .venv/bin/python -m sim2sim.research.train \
  --name new_run --skill roller --learner-device cuda --minutes 20
```

这只是资源监督调用形式；实际重训必须补齐冻结来源、控制契约、课程及教师配置，以保存的 recipe 为准。
