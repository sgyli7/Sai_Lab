# LingBot-VLA 2.0：GB10 / Unitree G1 本地微调官方资料核查

核查日期：2026-10-17（Asia/Shanghai）。本记录只读取官方仓库、官方模型仓库和依赖项目源码；未启动 GPU、未安装依赖、未修改 π0.5 工程。目标机器的 GB10 / ARM64 / 128 GB 来自主任务上下文，本次未重新测量硬件。

任务上下文已更新：以 `experiments/pi05_g1_mobile` 的长距离搬箱为比较目标，50 Hz、action horizon 50；外部 action 为上肢/手 28 + height 1 + base velocity 3，state 为上肢/手 28 + waist 3 + padding 1。这些是主任务提供的合同摘要，不是上游 G1 定义。本记录末尾“移动搬箱补充核查”优先于早期仅以 round02 为参照的研究背景；没有在此自行确定 adapter。

## 结论

可以开始独立环境和数据适配工作，但官方没有现成 G1 配置，也没有 GB10 单卡训练验证。最接近上游的原生监督参照路线是：完整保留发布模型的 native-depth 查询/头部结构，下载发布 checkpoint（已包含 Depth 和 DINO 教师），另下 MoGe，使用 Qwen 的配置与处理器文件，先跑 micro batch 1、冻结 VLM 的训练冒烟。移动搬箱项目拟用的结构保持 action-only 路线另见末尾补充核查，其实现不需要构建教师。待取得 GPU 队列时隙后才做 CUDA 验证。**冻结 VLM 是控制训练内存的工程选择，不是对实验效果的承诺。** 依据：[模型构造与冻结](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py)、[训练入口](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/tasks/vla/train_lingbotvla.py)。

**不能把 `align_params: {}` 当作发布 checkpoint 的无损 action-only 开关。** 它确实会跳过教师加载，却同时移除 depth/video 查询与头部；严格 post-training 权重加载会因多余 checkpoint key 失败。若要 action-only，必须实现并审查“保留模型结构，跳过教师和辅助损失”的独立开关，或明确记录经过 key 过滤的结构变体；本记录不建议用 `post_training: false` 掩盖错配。依据：[结构分支](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L479)、[严格加载](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/module_utils.py#L215)。

## 官方来源与版本

- 源码：[Robbyant/lingbot-vla-v2](https://github.com/Robbyant/lingbot-vla-v2)，本次浅克隆固定为 `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`。
- 预训练权重：[robbyant/lingbot-vla-v2-6b](https://huggingface.co/robbyant/lingbot-vla-v2-6b)，API revision 为 `11c703bf6a5c1f45b3b69168482da11fdbba53d7`。模型卡明确回链上述仓库；发布 tensor 为 F32。[模型卡](https://huggingface.co/robbyant/lingbot-vla-v2-6b)、[官方 API](https://huggingface.co/api/models/robbyant/lingbot-vla-v2-6b?blobs=true)
- 本次源代码只读副本：`/tmp/lingbot-vla-v2-upstream-research`；它不是后续实验安装目录。

## 下载大小与真正需要的文件

以下是官方 HF API `siblings[].size` 的字节求和，GB 为十进制，**不是运行内存估计**。API 数据是核查时快照；下载时应固定 revision。来源：[VLA API](https://huggingface.co/api/models/robbyant/lingbot-vla-v2-6b?blobs=true)、[Qwen API](https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct?blobs=true)、[MoGe API](https://huggingface.co/api/models/Ruicheng/moge-2-vitb-normal?blobs=true)。

| 内容 | 字节 / 体积 | 使用判断 |
|---|---:|---|
| VLA 全仓 | 28,239,981,618 B，28.240 GB | 包括模型、tokenizer 资产、Depth 与 DINO 教师 |
| VLA 六个 safetensors | 25,503,889,124 B，25.504 GB / 23.752 GiB | 必需；index 中 tensor 净大小为 25,503,630,044 B |
| VLA `depth/model.pt` | 1,316,220,456 B | native-depth 训练的 LingBot-Depth 教师，已附在 VLA 仓库 |
| VLA `dino_video/teacher_step_10000.pth` | 1,401,509,792 B | native future-video 训练；同目录 `config.yaml` 为 1,388 B |
| MoGe `Ruicheng/moge-2-vitb-normal/model.pt` | 419,110,160 B | 标准 native-depth 训练另需此文件 |
| Qwen3-VL-4B 全仓 | 8,887,292,732 B，8.887 GB | **本微调入口不需要完整下载** |
| Qwen 两个 safetensors | 8,875,719,344 B | VLA 已包含对应 VLM 参数，不另外加载 |
| Qwen 配置、processor、tokenizer 等资产 | 11,499,994 B，约 11.50 MB | 包括 `config.json`、`tokenizer*`、`preprocessor_config.json`、`video_preprocessor_config.json`、`chat_template.json`、`vocab.json`、`merges.txt`、`generation_config.json` |

Qwen 资产 revision：`ebb281ec70b05090aa6165b016eac8ec08e71b17`；MoGe revision：`ca5f0e07ff01d3e5a364c1d954ed12ee1814b368`。Qwen 只需资产的判断来自源码：V2 先 `AutoConfig.from_pretrained(tokenizer_path)`，再 `_from_config` 构造 VLM；`AutoTokenizer` / `build_processor` 同样使用这个路径，真正模型参数由 VLA checkpoint loader 注入。**没有另一次 `Qwen...from_pretrained` 权重加载。** [构造器](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L126)、[处理器](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/auto.py)、[参数加载器](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/loader.py)。

发布仓库 `config.json` 只有 `{"vlm_family": "qwen3_vl"}`，不能单独当作完整训练/推理模型配置；配置需来自代码中的 `LingbotVLAV2Config`、训练 YAML、Qwen config。[发布 config](https://huggingface.co/robbyant/lingbot-vla-v2-6b/blob/11c703bf6a5c1f45b3b69168482da11fdbba53d7/config.json)、[checkpoint index](https://huggingface.co/robbyant/lingbot-vla-v2-6b/blob/11c703bf6a5c1f45b3b69168482da11fdbba53d7/model.safetensors.index.json)

## 依赖与 ARM64 / Blackwell 风险

官方训练环境固定 Python 3.12、Torch 2.8.0、TorchVision 0.23.0、TorchAudio 2.8.0、TorchCodec 0.6.0、Transformers 4.57.3、Triton 3.4.0；脚本额外装 FlashAttention 2.8.3、LeRobot v0.4.2（`--no-deps`）以及本地 MoGe/Depth。安装脚本会始终安装 depth 依赖，示例 flash wheel 是 `linux_x86_64`，不能直接在 aarch64 使用。应在独立容器/环境中重建依赖解析，不对 π 环境运行该安装脚本。[依赖清单](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/requirements.txt)、[安装脚本](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/tools/create_train_env.sh)

存在两个不同层次的 attention 设置：

1. VLA 联合 attention 的正式选择只有 `flex` / `flex_cached` / `eager`，**没有名为 `sdpa` 的配置值**。`eager` 是源码里的 einsum/softmax 实现。 [V2 分派](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L416)、[eager 实现](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/utils.py#L225)
2. Qwen VLM text config 和 action expert config 在构造器中硬编码为 `flash_attention_2`；vision 另受 `vit_attn_implementation` 控制（默认也为 flash）。仅设置 VLA `attention_implementation: eager` 并不能移除 FlashAttention 安装/初始化检查。vision 的源码分派可走 HF `sdpa` 接口，但 text/action 的硬编码仍需隔离副本的小补丁；完整正确性需实际 forward/backward 检验。 [构造器](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L126)、[vision attention](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py)

GB10 的官方 compute capability 是 12.1。FlashAttention 2.8.3 的 setup 会生成 `linux_<machine>` 平台名，并含 CUDA >= 12.8 时的 `sm_120` 编译分支；**这说明不能一概称为“不支持 ARM64 / Blackwell”，也不能据此确认 GB10 已兼容**。未完成 aarch64 wheel、GB10 12.1、Torch ABI 和 backward 内核的实机验证。可先做 SDPA/eager 回退，再评估源码编译；不应以当前 FlashAttention README 的其他新版能力冒充已固定的 2.8.3 兼容性。[NVIDIA 设备表](https://developer.nvidia.com/cuda/gpus)、[FlashAttention v2.8.3 setup.py](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/setup.py)

MoE 有 `eager` 与 `fused` 两种实现，fused 使用 grouped GEMM / Triton。更改实现会改变专家参数存储形式，需检查 checkpoint 加载转换，不能为绕过编译错误直接把 `fused` 改为 `eager` 后假定等价。[专家实现](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py)、[加载与分派](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/auto.py)

## native-depth、教师与 action-only

标准 native 训练由非空 `train.align_params` 启用，教师是 MoGe + LingBot-Depth；`use_future_video: true` 时再建 DINO 教师。它们被 `requires_grad=False`、`.cuda()`、`.eval()`，是在线生成目标的冻结教师，不是优化器训练对象。上游加载的是 RGB 图片，使用教师生成 depth feature targets，因此“native-depth”不等于机器人必须有深度传感器。[训练器](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/tasks/vla/train_lingbotvla.py#L382)、[教师构建](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/vision_models/module_utils.py#L70)

把 loss weight 设为 0 仍会加载/运行教师；训练末的日志还会除以 depth loss weight，因此不能把 0 权重当作可靠开关。即使跳过教师构建，保留配置的模型 forward 仍会调用 depth/video loss，DINO patch loss 对缺失 targets 会抛错。**action-only 在数学上不需要教师，但当前官方训练器没有完整、结构保持的 action-only 开关。** 这是源码事实与工程推断的界限。[教师及日志控制](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/tasks/vla/train_lingbotvla.py)、[V2 辅助损失](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L846)、[DINO targets 校验](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py#L1445)

推理入口只构建 VLA 并加载其 checkpoint，不调用教师构建函数，因此**推理无需额外 MoGe/Depth/DINO 权重**，但仍保留训练配置中定义的查询结构。[推理 loader](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/deploy/lingbot_vla_v2_policy.py#L272)

## 单卡训练、内存与冻结

- 训练主入口始终初始化 NCCL，最小调用仍应经 `torchrun --nproc-per-node=1`。上游 `train.sh` 会探测可见 GPU，并开启 HF offline；下载须提前完成。 [train.sh](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/train.sh)、[训练 main](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/tasks/vla/train_lingbotvla.py#L335)
- `ParallelState.fsdp_enabled` 实际条件为 `fsdp_size > 1`。单 rank 不会获得 FSDP 分片，且非 FSDP 分支要求 `init_device: cuda`，不允许 FSDP offload。因此写 `data_parallel_mode: fsdp2` 不会让单卡自动得到内存分片。 [parallel state](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/distributed/parallel_state.py#L270)、[parallelize 限制](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/distributed/torch_parallelize.py#L102)
- 首轮可用 AdamW；Distributed Muon 明确要求 FSDP2 且至少 2 个 data-parallel ranks，不适合此单卡路线。batch 关系为 `global_batch_size = micro_batch_size × data_parallel_size × gradient_accumulation_steps`。 [参数验证](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/utils/arguments.py#L550)、[优化器说明](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/configs/vla/Training_Config.md)
- 正式开关 `train_expert_only: true` 冻结整个 Qwen VLM；`freeze_vision_encoder: true` 只冻结视觉编码器。前者可减少梯度和 optimizer state。`freeze_vit` 旧字段直接访问 `model.visual`，不应优先用于 V2。 [冻结实现](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L197)、[train flags](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/tasks/vla/train_lingbotvla.py#L120)
- 有 `utils/lora_utils.py` 和 PEFT 依赖，但本次全仓检查未发现官方 VLA train/config 中调用 `add_lora_to_model` 的接线；因此 LoRA 不能作为现成 CLI 功能承诺。 [LoRA 工具](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/utils/lora_utils.py)、[训练入口](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/tasks/vla/train_lingbotvla.py)

内存推算（不是测试）：以全量 F32 参数大小 25.50 GB 计算，weights + gradients + 两份 Adam moments 已约 102 GB，尚未计入 activation、教师、CUDA workspace、checkpoint 导出副本和系统内存；128 GB 统一内存不能据此保证全量训练可用。上游 mixed-precision 开关默认还会先 `model.float()`，不能只按 BF16 权重文件体积估算。冻结 VLM、batch 1、gradient checkpointing、先关闭 torch.compile 是合理的首测手段，最终以排队后的实际峰值为准。[parameter net size](https://huggingface.co/robbyant/lingbot-vla-v2-6b/blob/11c703bf6a5c1f45b3b69168482da11fdbba53d7/model.safetensors.index.json)、[float upcast](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/distributed/torch_parallelize.py#L108)

## G1、55 维映射与数据契约

本次固定 revision 的 `configs/robot_configs` 只有 RoboTwin 和 AgileX Cobot Magic 配置，没有官方 Unitree G1 embodiment 定义；对 `configs`、`docs`、VLA data/model 的 `unitree` / 独立单词 `g1` 检索也未找到 G1 映射。**因此应复用已确认的 pi05_g1_mobile 机器人和任务契约，由主任务定义 G1 适配，不能拿“支持 humanoid”直接推导某个 G1 关节顺序。** [官方配置目录](https://github.com/Robbyant/lingbot-vla-v2/tree/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/configs/robot_configs)

55 维的具体拼接由 `data.joints` 顺序决定，不能照 README 分类列表重新排列。按官方 `real_robot.yaml` 的**原顺序**，加上末尾 padding 后的半开区间为： [训练模板](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/configs/vla/real_robot/real_robot.yaml)、[拼接实现](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/utils.py#L546)

| 区间 | 含义 | 配额 |
|---|---|---:|
| `[0,14)` | arm.position | 14 |
| `[14,28)` | end.position | 14 |
| `[28,30)` | effector.position | 2 |
| `[30,34)` | waist.position | 4 |
| `[34,36)` | head.position | 2 |
| `[36,39)` | base.position | 3 |
| `[39,51)` | hand.position | 12 |
| `[51,55)` | reserved / padding | 4 |

缺失 feature 会填 0，对应 `state_joint_mask` / `action_joint_mask` 为 false；不足配额的关节也会补 0。联合 55 维不是 G1 的 55 个实际 actuator，不能直接向机器人发送。映射文件可以从原 `observation.state` / `action` 的多个切片拼接，并在 `unapply` 恢复原 feature 形状。 [FeatureTransform](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/utils.py)

数据正式支持本地 LeRobot v2.1 与 v3.0；单数据集用 `data.data_name=<robot_config_name>`，多数据集文本每行是 `<robot_config_name> <local_path>`。`subtract_state: true` 在归一化之前把 action 变成相对当前 state 的差量，推理 unapply 会加回；官方建议实机 arm relative、effector absolute，但这是推荐，**不能覆盖 pi05_g1_mobile 已确认的动作语义**。effector/velocity 映射不允许 subtract_state；EEF relative rotation 的指南仍建议不启用。 [自定义数据指南](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/README.md)、[变换顺序与恢复](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/utils.py#L375)

归一化需要用最终 G1 mapping、最终 subtract_state 与最终数据划分重新生成统计，并在机器人 YAML 的 `norm_stats` 和训练 `data.norm_stats_file` 对齐。支持 meanstd、bounds_99_woclip、identity 等；官方实机模板 meanstd、RoboTwin bounds_99_woclip 只是模板差异，不证明 G1 应选其中某个。不要复用 π 模型格式的统计 JSON 直接交给 LingBot。[统计流程](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/README.md#4-compute-normalization-statistics)、[Normalizer 构建](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/utils.py#L135)

## 可实施的最小路线与入口

1. 固定上述源码和 checkpoint revision，在独立 LingBot 环境准备依赖；CPU 检查 Python imports、LeRobot metadata/video decode、G1 mapping 往返和 mask；保持现有 π 目录及数据只读。
2. 下载 28.240 GB VLA 仓库 + 419 MB MoGe + 11.50 MB Qwen 资产；保留完整 native 配置首次加载，核对 1,708 个 checkpoint keys 全量匹配。发布 index 提供 key 列表。 [checkpoint index](https://huggingface.co/robbyant/lingbot-vla-v2-6b/blob/11c703bf6a5c1f45b3b69168482da11fdbba53d7/model.safetensors.index.json)
3. 基于已确认 pi05_g1_mobile contract 写独立 G1 robot YAML 和 split manifest、生成 LingBot 统计；保持训练和离线评估数据划分与 π 对齐，不把 train loss 当机器人成功率。
4. 获得单 GPU 队列时隙后，先检查 CUDA matmul、SDPA/attention 和 MoE forward/backward，再做 1–3 step 训练和 checkpoint reload；建议首测 `micro_batch_size=1`、`global_batch_size=1`、`gradient_accumulation_steps=1`、`train_expert_only=true`、`enable_gradient_checkpointing=true`、`optimizer=adamw`、`use_compile=false`、`init_device=cuda`。这些为待验证的低内存工程起点，不是官方 GB10 配方。
5. 通过冒烟后，明确选择 native 完整目标，或单独开发并验证保持结构的 action-only 变体，才增加训练步数并做相同任务的 held-out open-loop / simulator 评估。

官方训练形式（以下 `<...>` 是待填路径，不是可直接执行命令）：

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh tasks/vla/train_lingbotvla.py <g1_training_yaml> \
  --data.data_name <g1_config_name> \
  --data.train_path <existing_mobile_lerobot_train_directory> \
  --train.output_dir <lingbot_only_output_directory>
```

统计入口是 `scripts/compute_norm_stats.py`，open-loop 是 `scripts/open_loop_eval.py --robo_name <g1_config_name>`；实时服务入口是 `python -m deploy.lingbot_vla_v2_policy`。推理 loader 会向 `hf_ckpt` 的上三级查找 `lingbotvla_cli.yaml`，因此导出的 checkpoint 目录层级和训练配置需要一起保存，不能只拷贝 safetensors。 [统计脚本](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/scripts/compute_norm_stats.py)、[离线评估](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/scripts/open_loop_eval.py)、[推理服务](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/deploy/lingbot_vla_v2_policy.py)

## 确定阻塞与待实测项

| 项目 | 确定性 | 解除条件 |
|---|---|---|
| 没有现成官方 G1 mapping | 已由固定 revision 源码检查确认 | 对照 pi05_g1_mobile contract 实现并往返验证，不能自行猜关节顺序 |
| 直接 `align_params={}` + strict native checkpoint post-training | 明确会结构错配 | 保留 native，或显式结构保持 action-only 开关 |
| 零辅助 loss 权重仍加载教师且日志除零 | 源码控制流确定 | 不用零权重冒充关闭开关 |
| V2 text/action 硬编码 flash | 源码确定 | 可用 aarch64 flash 构建，或小范围 attention 回退补丁并验证 |
| 单 rank 得不到 FSDP 分片 | 源码确定 | 控制训练参数/activation 内存，不能依赖单卡 FSDP offload |
| GB10 torch/flash/triton/torchcodec 组合 | 未实测 | 在隔离环境完成 import + forward/backward + decode 验证 |
| 128 GB 内完整 native 训练峰值 | 未实测 | 等待 GPU 资源时隙，记录峰值、速度和失败阶段 |

这份核查没有更改 G1 observation/action contract，也没有把“代码存在路径”写成“GB10 已跑通”。

## 移动搬箱补充核查（同日，更新目标）

### 55 维并非硬编码的 G1 关节字典

源码没有一个独立的全局 `G1_ACTION_55` 或 29-DoF G1 字典。`FeatureInfo.update_info` 按 `data.joints` 的 YAML list 顺序生成 `joints` / `joints_max_dim`，`FeatureTransform.pad_and_concat` 再按这个顺序分别拼 state 与 action；两者共享分组区间，但拥有各自独立 mask。官方完整实机模板的精确顺序是 **arm14 → end14 → effector2 → waist4 → head2 → base3 → hand12**，共 51 维，最后 `prepare_state` / `prepare_action` 右补到 55。RoboTwin 仅配置前三组，剩余 25 维补零。这说明前面的 55 维表是完整实机模板的展开结果，不是库强制的 universal slot 枚举。[分组解析与拼接](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/utils.py#L49)、[完整实机模板](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/configs/vla/real_robot/real_robot.yaml)、[RoboTwin 模板](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/configs/vla/robotwin/robotwin.yaml)

| 完整模板区间 | state feature | action feature | 宽度 | 对 mobile 的确定信息与未决语义 |
|---|---|---|---:|---|
| `[0,14)` | `observation.state.arm.position` | `action.arm.position` | 14 | 有 14 个 arm 数值位置；没有 G1 左右臂关节名/顺序定义 |
| `[14,28)` | `observation.state.end.position` | `action.end.position` | 14 | EEF pose，不是另外 14 个 arm/hand actuator；若外部合同不含 EEF，应 padding/mask |
| `[28,30)` | `observation.state.effector.position` | `action.effector.position` | 2 | 两个 effector 值；没有“G1 每手第 7 个手指关节”之官方解释 |
| `[30,34)` | `observation.state.waist.position` | `action.waist.position` | 4 | 数值容量足以容纳 state waist3 与 action height1，但源码没有指定哪一位为 height；是否采用互不重叠的 `[height, waist3]` 是本地 adapter 决策 |
| `[34,36)` | `observation.state.head.position` | `action.head.position` | 2 | 合同无 head 时 padding/mask |
| `[36,39)` | `observation.state.base.position` | `action.base.position` | 3 | 模板名称为 **position**；源码/配置没有解释成 `(vx, vy, yaw_rate)`，不能把导航速度默认为此官方语义 |
| `[39,51)` | `observation.state.hand.position` | `action.hand.position` | 12 | 仅 12 个 hand 值；原 upper28 的实际 arm/hand 拆分仍需按本地合同核对 |
| `[51,55)` | 无 feature，尾部 padding | 无 feature，尾部 padding | 4 | 默认 mask=false；若改作额外 actuator，属于新的本地 slot 语义 |

**维度容量可以无损，预训练语义不能自动认定。** `arm14 + effector2 + hand12 = 28` 是可用数值容量，不证明任意“上肢/手28”都能按该分解保留身体语义。若外部合同实际是 arm14 + hand14，则必须由主任务明确两维如何表达；不能静默丢两维，也不能仅把原32整体右补23称为已对齐官方语义。waist4 可以承载“state waist3、action height1”的不同有效 mask，甚至在这四位内给 height 与 waist 分配互不重叠的位，但官方没有其具体组成定义。[mapping 与 reverse mapping](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/utils.py#L203)

base velocity3 必须明确保留速度语义。配置解析器允许在 `data.joints` 声明自定义同宽组，例如命名为 `base.velocity`；它只验证 robot feature 是否出现在声明中，名字含 `velocity` 时还禁止 `subtract_state=true`。因此源码支持一个显式 velocity schema，但它是本地 adapter 扩展，不是发布 checkpoint 对 `base.position` 的官方重释。若 state 没有 base velocity，则该 state 组三维 mask=false；action 的三维仍可有效。对 action 有效32/state有效31，mask 总数应分别校核为32/31，外部 state 的 padding1不能被误当成真实传感器。[feature 校验](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/utils.py#L168)、[velocity 限制](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/utils.py#L225)

时间合同方面，loader 用 LeRobot metadata.fps 生成 action offsets `t/fps, t=0..chunk_size-1`；在 50 Hz、chunk_size=50 时对应 0 到 0.98 秒的 50 个动作样本。未来图像若启用则取 `[0, (chunk_size-1)/fps]`；action-only 可不加载它。[时间采样](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/base_dataset.py#L231)

另一个确定的源码陷阱：`relative_type` 对所有 action 默认是 `quaternion_local`。只写 arm `subtract_state:true` 会在 apply/unapply 命中四元数分支并断言 feature 必须包含 `end.position`。若最终 contract 决定 arm 需要普通关节差分，需显式给非四元数 relative_type（例如 YAML `null`，helper 对 None 返回 false），而非照抄只写 subtract_state 的旧示例。本记录不替主任务决定是否使用差分。[默认值](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/utils.py#L230)、[helper](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/ee_pose_transform.py#L62)

### 保持 native 查询结构、关闭教师监督的最小改动边界

当前源码没有已接线的独立开关。下面只是拟议的开关位置与依赖清单，**没有实现这些修改**：引入一个与结构配置分离的 `supervision_enabled` 布尔值，保留 `align_params`、`self.use_depth_align`、`self.use_future_depth`、`self.use_future_video` 及 shared-query/projector 参数原样。

| 源码位置 | 现有行为 | 结构保持变体必须分别控制的行为 |
|---|---|---|
| `train_lingbotvla.py:382–417` | 非空 align_params 就构建教师，并验证 future-image | 教师构建与教师所需 future-image 校验加独立 supervision gate；不要清空模型 align_params |
| `train_lingbotvla.py:442`、dataset transform | `use_depth_align` 控制教师用 PIL 图；`use_future_image` 控制未来 RGB 解码 | 当前 RGB 始终保留；教师关闭时可不产生 PIL targets / 不加载 future image；不要把该数据标志反向用于删除模型查询 |
| `train_lingbotvla.py:763–791` | 在线运行 MoGe/Depth/DINO 生成 targets | supervision 关闭时跳过整个教师前向，不建立教师 CUDA 对象 |
| `modeling_lingbot_vla_v2.py:846–897` | 非空 align_params 就计算 depth/video prediction 与 loss | 只对辅助-loss分支增加 gate，返回 0 loss 和 None predictions；保持 forward 的返回项数量不变 |
| `modeling_lingbot_vla_v2.py:580–720` | learned/current/future query token 与 RGB/text 拼接 | 必须保留；共享投影参与 action 上下文，不应置 use_future_video=false 来省教师 |
| `train_lingbotvla.py:900–902,1069+` | 辅助损失日志除权重；周期可视化引用 targets | 跳过关闭监督后的该类日志/可视化，不用 loss weight=0 作为开关 |

依据：[训练器](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/tasks/vla/train_lingbotvla.py)、[模型 query/aux 分支](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py)、[数据变换](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/utils.py)

保留结构后，depth/video output heads 中只被辅助损失使用的参数可能得不到 gradient；query seeds 和参与 prefix 的 shared projection 仍参与 action 路径。未来转为多卡 DDP 时，unused parameters 行为需要额外检查。MoE sequence auxiliary/z-loss 是另一个独立目标，关闭教师不等于自动关闭 MoE routing 正则。[query 构造](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L582)、[MoE loss](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L1074)

**纯 action 路径只需要当前 RGB、image masks/grid、语言 tokens/masks 与 state；不需要输入真实 depth，也不需要运行深度估计器。** inference 的 `sample_actions` / `predict_velocity` 保留 native 查询，但没有 targets 或教师调用。其 query 是可学习向量与当前 image embeddings/投影构成，不依赖未来实际图像；因此不加载 future RGB 不要求改变 query 结构。[prefix](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L500)、[sample_actions](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L920)

### 冻结 VLM 的实际边界

固定 revision 没有名为 `freeze_vlm` 的正式训练字段，对应的是 `train_expert_only:true`。它将 `self.qwenvl` 全部参数设为 requires_grad=false，故包含 vision encoder、language model、token embeddings；并保持它 eval。`qwen_expert` 不在这个对象内，仍可训练；state/action/time projections、native query seeds 与外置 shared/head projections 也不会因此自动冻结，`state_proj` 另由 `train_state_proj` 控制。[VLM freeze](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L197)、[state proj freeze](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py#L1042)

结构保留的 action-only 若仍训练 query seeds，这些 query 的梯度要穿过冻结的语言网络。`requires_grad=false` 冻结参数不等于允许把整个联合 VLM forward 包成 `no_grad`；后者会切断 query/shared projection 的学习，这是需避免的额外语义变化。依据是 prefix 外置 query 被传进逐层联合 forward。[query 到逐层模型路径](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L320)

### BF16 初始化与训练 dtype

官方 loader 在有 weights_path 时已提供节省构造内存的路径：`init_empty_weights()` 将参数注册为 meta，在该 context 内直接构造 VLA，再对 meta 模型 `.to(init_kwargs['torch_dtype'])`；`load_model_weights` 随后才 `to_empty(device=init_device)`，逐个 tensor 读取 safetensors，并按目标参数 dtype 拷贝。**按这条路径把目标 dtype 设 BF16，不会先建立一份完整的真实 FP32 参数模型再转换。** F32 checkpoint 读取仍有单 tensor 暂存/转换开销，buffers 另行恢复，不能称为零额外内存。[loader](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/loader.py#L129)、[meta 注册](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/module_utils.py#L56)、[逐 tensor dispatch](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/module_utils.py#L145)

训练器通过 `enable_mixed_precision=false` 选择 BF16 目标 dtype；该 flag 为 true 时选 FP32，并在 parallelize 再做 `model.float()`。在单 rank 没有 FSDP 自动 casting 的情况下，BF16 参数还需匹配输入：FeatureTransform 固定输出 F32 state/action，而主模型调用没有 autocast，`embed_suffix` 直接调用 Linear。因此隔离 runner 需要明确采用主 forward 的 BF16 autocast或适当输入转换；仅改上述 flag 会存在 Linear dtype 错配。`action_fp32` 是另一条运行期局部 Linear cast 路径，不等于参数必须完整 FP32。[dtype 选择](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/tasks/vla/train_lingbotvla.py#L371)、[upcast](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/distributed/torch_parallelize.py#L108)、[embed_suffix](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla.py#L1126)

自写 loader 若改走 Transformers `_from_config`，应显式传 `dtype=torch.bfloat16` 或 torch_dtype alias；Transformers 4.57.3 的此函数才会临时设置 default dtype。直接 `Model(config).to(bfloat16)` 不带 meta/no-init，不能保证没有完整 FP32 构造峰值。[HF v4.57.3 _from_config](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/modeling_utils.py#L2271)

### Torch 2.11 + cu130 的 SDPA 路线边界

这是本地 runtime 选择，超出上游固定 Torch 2.8.0/Triton 3.4.0 的已声明环境。保留全部参数 key/shape 的 flash-free 改动在源码层面可行：将 V2 构造器的 VLM text 与 action expert `_attn_implementation` 改为 eager/SDPA，vision 使用 `vit_attn_implementation=sdpa`；再用已存在的 VLA eager 路径，或新增保持同一接口的 VLA SDPA 分派。三处 attention 配置均不创建/删除 learnable weights，因此不要求修改 checkpoint 结构。但完整 forward/backward 和数值等价尚未验证。[V2 构造器](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L126)、[vision SDPA 分派](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py#L77)

VLA eager 的接口收 `[B,L,H,D]` 的 Q/K/V 和 true=允许注意的布尔 `[B,Lq,Lk]` mask，显式将 KV 头 repeat 到 Q 头数量，scale 为 `head_dim**-0.5`，输出 `[B,Lq,Hq*D]`。Torch SDPA 可表达同样运算，但需要转置为 `[B,H,L,D]`、mask 增加 head broadcast 维，并保持相同 GQA repeat/enable_gqa 语义、dropout=0、is_causal=false（实际非标准因果结构来自原 mask）。不能仅以 `is_causal=true` 代替原 mask；masked 全行与数值误差也应对照 eager 检查。[eager 原实现](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/utils.py#L225)、[Torch 2.11 SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)

V2 联合 forward 会无条件把各层 q/k/v 转成 F32。若把这些原样交给 SDPA，会影响可选 fused kernel 与显存/速度；若在 SDPA 边界改为 BF16，又是数值策略变更，应明确记录并对照测试。SDPA 回退不解决 fused MoE/Triton 的其他兼容性，不能把“取消 flash 包”当作 Torch 2.11 全栈验证完成。[qkv.float](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L336)、[MoE 实现](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py)

### processor 的最小资产清单

对于已固定的 Transformers 4.57.3、默认 fast tokenizer 和官方 `build_processor`，源码推导的最小本地集合是以下五个 Qwen 文件，合计 **7,045,551 B**；尚未在全新环境做“删到最少”的实际加载测试：

1. `config.json`（1,505 B）：AutoConfig 与 processor class fallback。
2. `tokenizer_config.json`（10,868 B）：特殊 token、tokenizer class，以及其内嵌 chat template。
3. `tokenizer.json`（7,032,403 B）：fast tokenizer 的完整后端资产。
4. `preprocessor_config.json`（390 B）：image processor class、patch/merge、normalization 参数。
5. `video_preprocessor_config.json`（385 B）：即便只用 RGB，Qwen3VLProcessor 的 attributes 也会创建 video_processor，不能仅因“不输入视频”省略此配置并依赖默认值。

保留 `chat_template.json`（5,502 B）则六文件共 **7,051,053 B**，有利于维持完整 processor 的 chat template；LingBot 当前语言路径调用 `processor.tokenizer.apply_chat_template`，tokenizer_config 已内嵌它。`vocab.json` / `merges.txt` 是 slow tokenizer fallback所需，fast tokenizer.json 直接加载路径不依赖它们；若希望不依赖 fast-only 假设，继续使用前文九文件 11.50 MB 集合。`generation_config.json` 对 action flow 模型不必需；Qwen safetensors 仍不必需。[Qwen 固定资产](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17)、[processor attributes](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L78)、[组件加载循环](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/processing_utils.py#L1421)、[fast tokenizer 分支](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/tokenization_utils_fast.py#L102)、[LingBot prompt 模板调用](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/transform.py#L454)

### 参数与专家规模：官方 checkpoint header 的只读统计

官方 HF API 给出 `safetensors.parameters.F32 = 6,375,907,511`，即约 **6.376 B**。本次仅使用 HTTP Range 读取六个 safetensors 的 JSON header（均小于 64 KB），按 tensor 名称前缀对 `shape` 乘积求和，未下载 tensor 数据、未实例化 GPU 模型。结果如下。这是发布 checkpoint 的 tensor 元素统计，不替代最终模型的 `named_parameters()` / requires_grad 实测。[官方 API](https://huggingface.co/api/models/robbyant/lingbot-vla-v2-6b?blobs=true)、[固定 checkpoint index](https://huggingface.co/robbyant/lingbot-vla-v2-6b/blob/11c703bf6a5c1f45b3b69168482da11fdbba53d7/model.safetensors.index.json)

| checkpoint 分类规则 | 元素数 | 约 B |
|---|---:|---:|
| 名字含 `.qwenvl.`（vision + language） | 4,437,815,808 | 4.438 |
| `.qwen_expert.` 的 routed `.mlp.experts.` | 1,358,954,496 | 1.359 |
| `.qwen_expert.` 的 `.mlp.shared_expert.` | 58,392,576 | 0.058 |
| `.qwen_expert.` 其余（attention/router/norm 等） | 369,323,904 | 0.369 |
| 全部 `.qwen_expert.` 合计 | 1,786,670,976 | 1.787 |
| 外置 query/head/state/action/time projections 等 | 151,420,727 | 0.151 |
| 总计 | 6,375,907,511 | 6.376 |

源码/实机配置定义 action expert 36 层、hidden 768；每层 routed experts 32、每 token top-k 4、routed intermediate 512、shared intermediate 704。每层 routed bank 的三组投影为 `32 × 3 × 768 × 512 = 37,748,736`，36 层正好得到 header 中的 1,358,954,496；shared 三组投影为 `36 × 3 × 768 × 704 = 58,392,576`。top-k 只说明 token 的路由稀疏度，不代表训练时只加载四个专家权重，也不能直接推导整体 active parameter 数。[官方 MoE 配置](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/configs/vla/real_robot/real_robot.yaml#L46)、[action expert 构造](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py#L81)、[fused expert shape](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py#L114)

若仅冻结 `.qwenvl.`，理论上未被冻结的 checkpoint 元素上限为 **1,938,091,703（约 1.938 B）**；action-only 时某些仅辅助输出 heads 没有梯度、state_proj 也可能另外冻结，所以实际 optimizer state 的规模仍需运行时统计。这解释了冻结 VLM 为什么能显著减少梯度/Adam state，但不能把它说成只训练一个很小的线性 action head。
