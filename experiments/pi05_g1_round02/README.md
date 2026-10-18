# G1 苹果取放：Round02 综合 benchmark

本文是交付骨架与复现指引。**最终成绩、最终模型选择和完整录像索引均待实测补入；TODO 不是结果。** 第一轮代码、数据划分、模型、录像及统计继续保留，本轮新产物放在 `.scratch/pi05_g1/round02/`。

## 比较对象与用途

| 主策略 ID | 身份 | 输入及执行接口 |
| --- | --- | --- |
| `gr00t_n17` | 本机固定 revision 的原 G1 GR00T N1.7，完整五图 ONNX 导出，原生固定指令 `move the apple to the plate` | 原生图内图像预处理、实测自身关节、原生 40 步动作块 |
| `pi05_round01` | 第一轮固定 `step_001200` 权重重新评测 | **显式覆盖为本轮 norm v2**，保留第一轮权重，原生 50 步动作块 |
| `pi05_round02` | 从相同第一轮权重继续训练，依据完整混合验证集选择的 checkpoint；本地step0也有资格，累计更新数包含继承 | 与 `pi05_round01` 使用同一 norm v2，原生 50 步动作块 |

两种 π 的指令均为 `Pick up the apple and place it on the plate.`；输入是同期头部 RGB、自身关节状态和指令。物体真值、接触、成功事件用于验收和数据审计，不作为模型观测。下肢和平衡继续由现有全身控制器负责。

所有策略使用最终冻结的同一物理环境、相机、observer、控制器、实际动作限制 B、50Hz 控制和 **40 tick 重规划间隔**。π 每块只执行前40步，再用新观测重规划。动作仍为双臂14、双手14、身高1、底盘3，共32维；上肢限制与硬件限制相交。不得把 GR00T 原生提示词差异隐去。

两 π 共用 `norm_stats_v2.json` 的决定见 `comparator_adapter_rule.json`。它仅修复训练集 vx/vy 稀疏非恒定目标被旧分位数归一化折叠的问题，其余变换保持原值。第一轮权重的训练元数据仍记录旧 norm；本轮推理覆盖必须显式披露。新旧 π 的主要差异是新增训练配方；若有收益，不能单独归因于 fresh 数据、phase 采样或某一项变化。

第一轮 **2/20 仅为历史记录**，当时使用旧分位数限制/协议，不能复用为本轮分数。开发 cadence 对照亦不计入最终分数。可选 `official_pi05_base` 是相同本轮接口下的完整20次第四政策参考；是否执行只依据预算和预计时长，不能依据已出成绩择样或把部分执行当完整参考。

## 冻结样本、选择与成功定义

最终20次为5个苹果 XY 偏移 `(0,0), (+.02,0), (-.02,0), (0,+.02), (0,-.02)` 米，各重复 seeds40..43。它们是 **5个物理位置×4个随机种子**，不是20个独立物理场景。逐 case 比较实际初始状态 hash；最终初态收据包括命名关节位置/速度、root pose、COM/link速度和物体姿态/速度，不包含 PhysX 隐藏求解器缓存。

最终 checkpoint 只由原始21条完整验证轨迹和合格 fresh 验证轨迹的声明加权 loss 选择，不读取 final20 或 offline test outcomes。工厂读取当前run显式 `candidate_training_dirs.json`，复核 best pointer 对应全部声明run的完整checkpoint验证分数全局最小值。恢复run的本地step0可选，但需计入继承更新数：例如继承本轮step200后local0的累计本轮更新数是200，不是0。被中止run的完整候选保留，不伪造其训练正常结束；未持久化的已执行更新单列在失败/恢复收据中。递归权重链必须回到原第一轮step1200比较器。teacher 的整轨迹 split、去重、失败记录和路由必须在采集前冻结；失败不回填、不挪 split。12次采集现已全部完成并通过独立CPU标签/动作/图像链审计：train 9次中7条成功，validation 3次中3条成功，最终接收7条train和3条validation。`gr00t_grid_04_seed204` 已抓取并产生源候选，但未满足冻结稳定释放窗；`gr00t_grid_08_seed208` 未满足手部接触抬升抓取代理，也无源候选。两条失败原始记录全部保留并拒收入训练，不补采、不改split。证据为 `teacher_all_twelve_audit_summary.json` 及其绑定的train9/val3独立审计；这些teacher成功率不是最终20次benchmark分数。

成功使用冻结 `pi05_g1_placement_v2`：

- 抓取代理必须先于源任务候选时刻：苹果抬高超过3cm，并至少两个命名手部 body 有真实接触。该规则是 lift/contact proxy，不能证明 force closure。
- 释放后满足连续101个50Hz样本、共2秒稳定窗；全部原始支撑、足迹、与机器人分离及站立检查通过，期间不跌倒、不掉落。
- 窗内 actor origin 全部两两距离不超过5mm、四元数角度不超过5度。物理任务时长6秒，独立稳定审核最多延长2秒。
- 模型评测必须绑定同版验收、同接口的有界/无界 replay 双 gate。启动成功或 loss 下降均不能替代物理任务成功。

任务可用门槛仍为至少16/20成功；不足则报告实际分数。主要“失误率”为 `(20-success)/20`，仅在20个物理任务完整执行后计算。互斥失败阶段按以下优先级：`fell`、`dropped`、`success`、`missed_grasp`、`grasp_without_placement`、`unstable_or_unreleased_placement`。原始 timeout、掉落、跌倒等标记另列。基础设施失败/缺失单列 incomplete，不能记为0分；零次底线掉落不等于没有失误。

统计并列给出每政策 Wilson95%区间、配对 conditional McNemar、按20次条件试验的 paired bootstrap、按5个位置的 cluster bootstrap，以及位置均值差的 exact cluster sign-flip 双侧 p。5组只有32种符号，最小双侧 p 为2/32=0.0625；此字段不改变16/20门槛或原结论规则。

## 时间、速度与录像口径

- 成功任务 sim time：`qualified_stability_window.end_tick / 50`。失败 source candidate 只作诊断，不是完成时刻。
- 实际任务 wall time：成功稳定窗末帧 `monotonic_ns - row0.monotonic_ns`，包含首模型冷加载 RPC 和推理暂停；排除 frame0 前的模拟器初始化及完成后的清理。另报进程启动至成功的 wall time。
- cold startup、首 RPC/model load、warm RPC 分开报告；warm RPC 是主要推理服务速度指标，包含 IPC、预处理和推理。
- 扣除 RPC 暂停的执行 wall time 仍含物理计算、渲染和录像开销，不能当作实体机器人1×能力。
- launcher wall time 包含初始化、整个任务与清理。只能匹配真实原始 launch 收据，不能编造 task-only wall time。
- 配对速度仅比较双方都成功、实际初态匹配的 case；失败/超时更快不算速度胜。各自成功样本的边际完成时长另列，样本群体可能不同。

每case保存原始 head 和全身 observer 录像，均为50fps物理时间回放。带 wall 字幕的50fps衍生视频仍会压缩暂停；代表 case 可额外导出真正按 wall clock 重复帧的25fps视频，其中初始化无图像部分用空白占位、推理暂停重复最后真实观测。后者不含末帧之后的清理。observer v1 被遮挡的开发构图不作为最终全身录像。

原始目录：`EVALROOT/<policy>/<case>/`。外置衍生目录：`EVALROOT/video_exports/<policy>/<case>/`。导出 manifest 绑定原 result/trace SHA，不改写原始结果。最终索引：**TODO：填入所有原速 head/observer 路径、代表 wall-clock 路径及 SHA。**

## 复现命令模板

以下模板从仓库根目录执行。将所有 `TODO_...` 替换为经审核路径/预算；每次输出目录、收据文件都使用新名称。文档中的命令不会自动执行。GPU命令必须通过同一轮预算监督，8小时累计含失败占用；不得重置 ledger 绕过预算。训练、batch 与模拟器会启动 GPU，其余下述工具仅做 CPU 处理。

```bash
ROUND02_PY=/home/ethan/Projects/Sai_Lab/.scratch/pi05_g1/venv/bin/python
ROUND02_ROOT=/home/ethan/Projects/Sai_Lab/.scratch/pi05_g1/round02
ROUND02_TRAINING_DIR="$ROUND02_ROOT/TODO_new_training_run"
ROUND02_EVALROOT="$ROUND02_ROOT/TODO_new_final_evaluation"
ROUND02_LEDGER="$ROUND02_ROOT/TODO_authorized_round_ledger.json"
export PYTHONPATH=/home/ethan/Projects/Sai_Lab/.scratch/pi05_g1/openpi/src:/home/ethan/Projects/Sai_Lab
export OPENPI_DATA_HOME=/home/ethan/Projects/Sai_Lab/.scratch/pi05_g1/venv/openpi-cache
export JAX_PLATFORMS=cpu
```

这些环境值已核对 `.scratch/pi05_g1/ENVIRONMENT.md`、训练 CPU admission 记录和冻结 policy 子进程实现。`JAX_PLATFORMS=cpu` 仅约束 JAX；π 的 PyTorch 训练/推理仍使用 CUDA。GPU 命令不要沿用 CPU conversion 的 `CUDA_VISIBLE_DEVICES=''`。模拟器为独立 policy 子进程显式重建 `PYTHONPATH`（含仓库、openpi/src 和 openpi-client/src），并清除 Isaac/Kit 的 Python 与动态库污染；主机训练仍需上面这些导出值。

依赖版本及修补记录见 `.scratch/pi05_g1/ENVIRONMENT.md` 和 `environment_probe.json`，固定 source Docker image/build、资产与源脚本路径见冻结 launcher、协议及 gate 原始 `config.json`/运行收据。当前 launcher 的 owned image 为 `sai-pi05-g1-source:6.0-frozen`，源镜像为 `nvcr.io/nvidia/isaac-sim:6.0.0-dev2`（ARM64 manifest `sha256:9bc527c60c1cdde532b05859b955e866e6115ac5bf2fe22eaa24daa6ea8783e6`）；标签不能替代原收据里的实际镜像/build身份。运行前需确保这些绝对路径、隔离环境、固定镜像及资产与目标本机对应，不能仅复制命令后默认可移植。

### 1. 审核并转换教师示范（CPU）

仅在被路由的12次采集已结束、实际成功及验收证据审计后运行。不读取正在写入的目录。选择记录必须在采集之前冻结；不得在两种 teacher family 间看成绩择优。

```bash
"$ROUND02_PY" -m experiments.pi05_g1_round02.audit_episodes \
  "$ROUND02_ROOT/TODO_completed_teacher_root" \
  --output "$ROUND02_ROOT/TODO_teacher_physics_audit.json"

"$ROUND02_PY" -m experiments.pi05_g1_round02.teacher_dataset \
  --results-root "$ROUND02_ROOT/TODO_completed_teacher_root" \
  --cases "$ROUND02_ROOT/TODO_frozen_selected_teacher_cases.json" \
  --routing-receipt "$ROUND02_ROOT/TODO_frozen_route_receipt.json" \
  --acceptance-frozen "$ROUND02_ROOT/../acceptance_v2_frozen.json" \
  --output "$ROUND02_ROOT/TODO_new_teacher_dataset"
```

产物 `manifest.json` 记录实际 train/validation、所有拒收原因、NPZ/视频 hash、逐监督帧原始无损 PNG 双 hash、抓取/候选/稳定窗事件。监督为 row k 实测 state/RGB 对 row k+1 实际 applied action，末帧无下一动作不监督；训练优先读取原PNG，不从有损MP4生成输入。事件只用于采样审计，不进模型输入。两边均至少1条合格fresh轨迹才能启用fresh源。

### 2. 训练（GPU，预算监督）

先核对冻结 data config 的 manifest 路径就是已审计的转换产物；本模板不能改已冻结配置。实际时长和步数以本轮授权及训练收据为准。

```bash
"$ROUND02_PY" -m experiments.pi05_g1_round02.budget \
  --ledger "$ROUND02_LEDGER" --stage training --max-seconds TODO_training_stage_seconds -- \
  "$ROUND02_PY" -m experiments.pi05_g1_round02.training \
  --warmstart "$ROUND02_ROOT/../training_run02/step_001200" \
  --norm-stats "$ROUND02_ROOT/norm_stats_v2.json" \
  --data-config "$ROUND02_ROOT/training_data_config.json" \
  --loss-weights "$ROUND02_ROOT/loss_weights_uniform32.json" \
  --output "$ROUND02_TRAINING_DIR" --hours TODO_training_hours \
  --max-steps TODO_max_steps --checkpoint-every 200 --seed 42
```

配方为 BF16 全参数、batch1、梯度累积8、lr1e-5、fresh AdamW/master FP32；不恢复第一轮 optimizer。原/fresh train与完整validation均各权重0.5；fresh抽帧80%在候选+25tick前缀均匀、20%全轨迹，32维loss均匀。所有周期checkpoint保留，step0先完整验证；CPU admission会核磁盘与数据身份。

### 已发生的训练失败与当前恢复入口

`training_run01` 已记录完成343个优化更新，随后 host MemAvailable 安全监护触发，账本终态 `host_memory_guard`、退出码137。最后完整持久化候选为step200；其后143个已执行完成更新未保存，另最多1个在途更新单列为不确定上界。该次失败实际占用 **3303.5951626489987秒**，已计入同一轮GPU账本，不能重置或忽略。旧run的日志、失败收据、完整step0/200候选及其验证结果继续保留；未写出 `training_stopped`，不得冒充正常训练完成。

当前恢复run为 `training_run02`：仅从旧run的step200权重恢复，**fresh AdamW，不恢复optimizer state**。本地新增上限1000步，每100步保存；内部2.65小时，保存预留450秒、验证预留180秒，外层budget上限10000秒。`cumulative_round_updates = 200 + local_step` 描述保留权重链的本轮累计更新；343个旧已执行更新及最多1个在途更新另计资源/步骤上界，不与继承200重复相加。恢复参数、旧失败证据和跨run完整验证候选见 `recovery_training_command.json`；全局选模仍在声明的候选roster间比较完整加权validation，本地step0同样有资格。**恢复是否完成、最终step和选中权重仍待实际终止及收据核验。**

恢复精度限制：`ModelRuntime` 在 `load_model` 后调用官方 `to_bfloat16_for_selected_params`，该函数先执行整体 `self.to(BF16)`，再将稳定层转回FP32，因此这些原FP32参数会经历 **FP32→BF16→FP32舍入**。checkpoint文件hardlink与SHA相同只证明文件字节一致，不能证明加载后的parameter数值完全不变；本次是weights-only加fresh Adam恢复，不能称无损续训。本轮所有π沿用同一冻结加载器，不据此事后更换协议。`runtime_postload_dtype_rounding_cpu_diagnostic.json` 以小型CPU toy调用官方函数证实这种变换；它不能将真实validation loss差值 `-2.573e-5` 全部归因于舍入，也不能证明或推断闭环物理表现受到何种影响。

已准备的原样执行入口如下。它核对receipt中的冻结文件SHA后 `exec` 保存的argv（含budget），不重新拼参数；输出固定为新run02，**现有run已启动时不得重复执行，也不得修改冻结receipt以覆写目录**。这是实际命令的复现记录，不是第二次运行的许可或新预算。

```bash
# 实际已准备入口；不要在运行中的同一输出目录重复启动。
python3 /home/ethan/Projects/Sai_Lab/.scratch/pi05_g1/round02/launch_recovery_training.py
# 它读取 /home/ethan/Projects/Sai_Lab/.scratch/pi05_g1/round02/recovery_training_command.json

# 独立只读前台进度（CPU，不启动训练；每60秒，账本终止且清理/记账关闭后final退出）
python3 -m experiments.pi05_g1_round02.recovery_training_watch \
  --progress "$ROUND02_ROOT/training_run02/progress.jsonl" \
  --ledger "$ROUND02_ROOT/gpu_budget.json"
```

恢复流程结束后的工厂和归档要显式选择真实终止run的 `--training-dir` / `--training-progress`，不能用旧失败run冒充结束。

### 3. 生成计划与阶段边界核验（CPU）

等待本次训练GPU阶段结束、清理和记账关闭，再由工厂验证候选roster内全部checkpoint metadata、递归祖先、best pointer和完整混合验证来源。只哈希选定/第一轮/官方base及GR00T payload，不运行模型。

```bash
"$ROUND02_PY" -m experiments.pi05_g1_round02.make_benchmark_plan \
  --training-dir "$ROUND02_TRAINING_DIR" \
  --protocol "$ROUND02_ROOT/final_protocol.json" \
  --data-config "$ROUND02_ROOT/training_data_config.json" \
  --loss-weights "$ROUND02_ROOT/loss_weights_uniform32.json" \
  --output "$ROUND02_EVALROOT"

"$ROUND02_PY" -m experiments.pi05_g1_round02.verify_plan_phase \
  --plan "$ROUND02_EVALROOT/primary_plan.json" --phase start \
  --receipt "$ROUND02_EVALROOT/TODO_primary_start_verification.json"
```

产物：`selection_receipt.json`、主60次 `primary_plan.json`、可选20次 `optional_base_plan.json`、全部config。主计划按case顺序逐个执行，同case的三政策顺序按case index轮转，以平衡机器热状态。大模型 hash 只在工厂和阶段起止核验，不放进每case重复核验的 frozen_files；服务自身原有加载检查仍保留。

### 4. 执行主60次（GPU，预算监督）

```bash
"$ROUND02_PY" -m experiments.pi05_g1_round02.budget \
  --ledger "$ROUND02_LEDGER" --stage evaluation --max-seconds TODO_evaluation_stage_seconds -- \
  "$ROUND02_PY" -m experiments.pi05_g1_round02.batch \
  --plan "$ROUND02_EVALROOT/primary_plan.json" \
  --receipt "$ROUND02_EVALROOT/TODO_primary_batch_receipt.json"

"$ROUND02_PY" -m experiments.pi05_g1_round02.verify_plan_phase \
  --plan "$ROUND02_EVALROOT/primary_plan.json" --phase end \
  --receipt "$ROUND02_EVALROOT/TODO_primary_end_verification.json"
```

失败退出也应保存阶段结束核验。optional阶段仅在剩余硬预算足以完整20次时运行：对 `optional_base_plan.json` 同样做 start核验、budget→batch、end核验，全部使用新收据。不得只挑部分case执行后宣称完整政策成绩。

### 5. CPU审计、外置录像与汇总

```bash
"$ROUND02_PY" -m experiments.pi05_g1_round02.audit_episodes \
  "$ROUND02_EVALROOT/gr00t_n17" "$ROUND02_EVALROOT/pi05_round01" "$ROUND02_EVALROOT/pi05_round02" \
  --output "$ROUND02_EVALROOT/TODO_final_physics_audit.json"

"$ROUND02_PY" -m experiments.pi05_g1_round02.video_export \
  --case-dir "$ROUND02_EVALROOT/TODO_policy/TODO_case" \
  --output-dir "$ROUND02_EVALROOT/video_exports/TODO_policy/TODO_case"
# 单case导出；代表case另加 --wall-clock。

# 全部GPU作业及账本清理结束后，一次性CPU导出所有已完成主计划/可选计划。
# 工具持有ledger锁，最多4 worker；全部case四路视频，固定p0_seed40另加真实wall-clock。
"$ROUND02_PY" -m experiments.pi05_g1_round02.batch_video_export \
  --evaluation-root "$ROUND02_EVALROOT" \
  --gpu-ledger "$ROUND02_ROOT/gpu_budget.json" --workers 4

"$ROUND02_PY" -m experiments.pi05_g1_round02.benchmark \
  --cases "$ROUND02_ROOT/final_cases.json" \
  --policy "gr00t_n17=$ROUND02_EVALROOT/gr00t_n17" \
  --policy "pi05_round01=$ROUND02_EVALROOT/pi05_round01" \
  --policy "pi05_round02=$ROUND02_EVALROOT/pi05_round02" \
  --replay-gate "$ROUND02_ROOT/final_interface_and_teacher_qualification/gate_bounded/result.json" \
  --output "$ROUND02_EVALROOT/TODO_benchmark_report.json"
```

若可选完整20次已结束并审计，可另存四政策报告，增加 `--policy "official_pi05_base=$ROUND02_EVALROOT/official_pi05_base"`，不覆盖三政策主报告。汇总检查实际初态配对、验收/接口hash、四类视频、原始launch wall和trace timing，缺失保留 incomplete。

完整且evidence-qualified的汇总生成后，CPU导出图表和录像CSV；不读取未完成案例，不按成功挑代表case：

```bash
# plot_venv已准备Matplotlib/NumPy；--output必须是新目录，产出PNG/PDF/CSV及hash manifest。
/home/ethan/Projects/Sai_Lab/.scratch/pi05_g1/round02/plot_venv/bin/python \
  -m experiments.pi05_g1_round02.plot_benchmark \
  --summary "$ROUND02_EVALROOT/TODO_benchmark_report.json" \
  --output "$ROUND02_EVALROOT/TODO_new_plots"

"$ROUND02_PY" -m experiments.pi05_g1_round02.video_index \
  --summary "$ROUND02_EVALROOT/TODO_benchmark_report.json" \
  --output "$ROUND02_EVALROOT/TODO_new_video_index.csv"

"$ROUND02_PY" -m experiments.pi05_g1_round02.video_compare \
  --case-dir "$ROUND02_EVALROOT/gr00t_n17/p0_seed40" \
  --case-dir "$ROUND02_EVALROOT/pi05_round01/p0_seed40" \
  --case-dir "$ROUND02_EVALROOT/pi05_round02/p0_seed40" \
  --output-dir "$ROUND02_EVALROOT/TODO_new_representative_comparison" --wall-clock
```

图表只接收完整3/4政策报告，保留所有配对和失败阶段；无共同成功case的速度为N/A。视频索引绑定summary、manifest和原result身份；原速、物理时间字幕与真正wall-clock明确分列。并排录像固定 `p0_seed40`，短片结束后标明末帧hold，最终outcome是回顾标签，不表示提前完成。

### 6. 新目录复现计划（CPU；执行仍需新的授权预算）

```bash
"$ROUND02_PY" -m experiments.pi05_g1_round02.clone_plan \
  --source "$ROUND02_EVALROOT" \
  --output "$ROUND02_ROOT/TODO_new_reproduction_root"
```

clone仅重定位输出、视频和config路径，保留模型、来源、规范、双gate和原selection字节。然后在新root执行相同 phase核验、budget→batch、审计/导出/汇总。原始目录或partial均不得覆盖；模型、训练结果、原始数据与录像长期保全至日期备份目录，不作cache删除。

## 留出动作误差辅助诊断（可选预算阶段）

必须先冻结主闭环的最终选模收据，再依据剩余预算决定是否完整执行。不能用此 test 结果回选 checkpoint、修改 adapter 或调最终场景。共用 norm v2 比较第一轮 step1200、本轮选定 checkpoint 与官方 base；不比较 GR00T 的离线动作误差。

规划只读取 selection、manifest 的 episode 长度和原 TRAIN raw extrema，不读取 test 观测。原21条 test 以固定不重叠 horizon50覆盖，chunk与有效帧数量从 manifest 计算并写入 plan，不能硬填预期数。activehands 由原 TRAIN raw `max-min>0` 固定，不从测试结果挑维。

```bash
"$ROUND02_PY" -m experiments.pi05_g1_round02.offline_diagnostic plan \
  --selection-receipt "$ROUND02_EVALROOT/selection_receipt.json" \
  --train-raw-bounds "$ROUND02_ROOT/physical_bounds_B_train_raw_extrema_candidate.json" \
  --hours-per-model TODO_offline_hours_per_model \
  --output "$ROUND02_ROOT/TODO_new_offline_diagnostics"

# run-job 才会调用未修改的原 evaluate_offline.py 并启动 GPU。
# 对 pi05_round01、pi05_round02、official_pi05_base 分别执行，全部计入同轮 ledger。
"$ROUND02_PY" -m experiments.pi05_g1_round02.budget \
  --ledger "$ROUND02_LEDGER" --stage evaluation --max-seconds TODO_offline_job_seconds -- \
  "$ROUND02_PY" -m experiments.pi05_g1_round02.offline_diagnostic run-job \
  --plan "$ROUND02_ROOT/TODO_new_offline_diagnostics/plan.json" \
  --policy TODO_policy_id

"$ROUND02_PY" -m experiments.pi05_g1_round02.offline_diagnostic summarize \
  --plan "$ROUND02_ROOT/TODO_new_offline_diagnostics/plan.json" \
  --output "$ROUND02_ROOT/TODO_new_offline_summary.json"
```

run-job 前后流式核对权重、norm、split manifest及原 evaluator/runtime 源码 SHA，写独立新收据；原输出本身没有这些 SHA，不能只凭其 checkpoint 路径宣称输入身份一致。汇总只在三项 status、全部 chunk/有效帧与收据身份绑定完整时标记 complete；超时、缺失、篡改均保留 incomplete，不冒充完整误差。

输出32维 MAE/RMSE，并按 arms14、TRAIN非恒定 activehands、allhands14、height 及独立 vx/vy/yaw 分组；MAE 为维均值，RMSE 为逐维MSE均值的平方根。arms14/activehands/allhands14单位rad，height单位m，vx/vy单位m/s，yaw单位rad/s；每组同时输出明确unit字段。兼容保留的nav3标为 `mixed-units numeric-only`，unit为null、无定义物理单位；它既不是无量纲评分，也不得用于跨组排名，必须以分别报告的vx/vy/yaw与逐维值解释。这里是原模型输出经公共 norm 还原、**未施加模拟器动作限幅**的辅助误差；原 evaluator 对数据 target 同样执行归一化再还原，分位数跨度坍缩的非恒定通道会丢目标信息，v2已修已知vx/vy问题。动作误差不等于抓取/稳定放置成功，也不能单凭其改善认定闭环收益。

## 最终一次性长期保全（CPU，全部GPU与报告完成后）

`archive_round02` 是前台one-shot，不删除原文件、不覆盖已有备份；新备份根必须尚不存在。它要求所选进度有真实 `training_stopped`、主60次完整且汇总证据合格、账本无running/cleanup/accounting_open。恢复run须显式 `--training-progress`；旧失败run仍在round根内一并保全。当前训练未完成时不得执行。

元数据/JSON/JSONL/URDF/文本与源码独立复制。模型、原图/视频/NPZ以及USD/mesh/纹理，只有明确root/user-authorized永久冻结receipt校验通过、当前用户所有且同文件系统时才能硬链接；root-owned或未确认不可变的载荷回退copy。硬链接共享inode，冻结后禁止原位写入。receipt schema为 `pi05_g1_round02_immutable_payload_receipt_v1`，包含 `permanently_frozen:true`、绝对 `round_root` 和 `files:[{path,bytes,sha256,reason}]`，可另绑定 `gpu_budget_sha256`；所有最终载荷streaming hash及冻结声明由GPU退出后统一生成，不在运行期预做。

最小物理复现资产共236文件、1,432,253,751字节（stat级盘点，0符号链接），使用下列三个root和四个独立file，保留绝对路径映射 `external/<original absolute path>` 及相对依赖树。source6.0根包含USD、对应URDF及下载收据；homie meshes为URDF实际引用的50项；冻结Isaac6.0树179项完整保留背景/对象/材质纹理依赖。不要归档整个homie、frozen或agile父目录中的无关env/6.1重复资产。

```bash
python3 -m experiments.pi05_g1_round02.archive_round02 \
  --round-root "$ROUND02_ROOT" \
  --backup-root /home/ethan/ProjectBackups/TODO_date/TODO_new_round02_backup \
  --benchmark-summary "$ROUND02_EVALROOT/TODO_benchmark_report.json" \
  --training-progress "$ROUND02_ROOT/training_run02/progress.jsonl" \
  --immutable-payload-receipt "$ROUND02_ROOT/TODO_final_immutable_payload_receipt.json" \
  --external-artifact-root /home/ethan/models/unitree_g1/agile_t1/source_6_0 \
  --external-artifact-root /home/ethan/models/unitree_g1/homie_v2/meshes \
  --external-artifact-root /home/ethan/models/unitree_g1/task_assets/20261001_frozen/Assets/Isaac/6.0 \
  --external-artifact-file /home/ethan/models/unitree_g1/agile_t1/unitree_g1_velocity_height_recurrent_student.onnx \
  --external-artifact-file /home/ethan/models/unitree_g1/homie_v2/g1_29dof_with_hand.urdf \
  --external-artifact-file /home/ethan/models/unitree_g1/homie_v2/g1_physics.json \
  --external-artifact-file /home/ethan/models/unitree_g1/task_assets/20261001_frozen/native_task_objects_t1_60_diagnostic_v2.json
```

归档另保全GR00T discovery14个外部payload、round02源码/测试/README、实际source_harness本地Python及必要小配置、第一轮继承模块与Dockerfile/复现文档、OpenPI固定commit/dirty状态/本地patch及必要src。第一轮备份以清单SHA引用并独立复制其元数据，不删除或迁移旧备份。`retention_manifest.json` 为每文件记录相对路径、bytes、SHA、源inode/owner和copy/hardlink方法；环境、cache、`.git/LFS`及`.incomplete`排除项明确记录。符号链接拒绝；失败留下 `ARCHIVE_INCOMPLETE.json`，不能将partial当完成。固定OCI digest/SDK引用继续保留，不导出整套镜像、SDK或已安装环境。

## 最终交付待填表

| 字段 | 最终实测/证据 |
| --- | --- |
| teacher route、实际合格 train/validation、拒收及去重 | 已审计12尝试：7/9 train成功、3/3 validation成功；04不稳、08漏抓拒收并保留；见teacher_all_twelve_audit_summary.json及转换manifest |
| 训练更新数、所有完整验证候选、选定step/weights SHA；是否step0 | run01完成343更新后hostguard137，持久化200；恢复run02继承200且freshAdam；最终更新/全局选择/weights SHA TODO，等待实际终止 |
| protocol/cases/norm/limits/source及双gate SHA、阶段起止核验 | TODO |
| 三政策成功/20、失误率、失败阶段、16/20门槛 | TODO |
| 配对改善、Wilson/conditional与5cluster不确定性；收益是否证实 | TODO |
| 成功匹配case的simtime与真实wall差、cold/warm RPC | TODO |
| 原速头部/全身录像完整索引、代表真实wall-clock视频 | TODO |
| optional官方base是否因预算执行完整20次 | TODO |
| 全部GPU占用（含失败）、基础设施缺失、模型/数据/接口/物理限制 | TODO |
| 实体机器人能力 | 本轮仅物理仿真；实体机器人仍需实机闭环验证 |
