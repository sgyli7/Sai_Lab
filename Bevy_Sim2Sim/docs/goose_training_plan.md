# Goose V0.1｜Bevy Sim2Sim 长期训练计划

版本：2026-10-11（北京时间）。实施分支：`codex/goose50_training`。

**执行状态：本聊天只推进Standup，物理200Hz、策略／驱动50Hz；完整鲁棒目标active。** 原生MuJoCo每Tick一次推理、4次5ms真实积分，prepare/commit、目标限速、延迟、热、功率和公开历史每20ms更新。原50Hz及Bevy资格独立。每两小时唤醒仍暂停，不联系硬件/G1/Move、不新建聊天或PM。用户指出的折颈终姿已改为实际抬头整理，接续始终保留同一世界和驱动状态。

**唯一现行检查点045：局部差分跟踪未解锁起身，纠正旧HoST导出缩放也没有新增能力。** 165在同一真正倒地F32出生、原200/50合同的12包物理预测中，旧相对差分与绝对差分的18列响应全部非零，166独立复核；不支持“舍入抹掉全部反馈”。167两臂相同出生／H12／maxnfev4／代价／界，各5万积分预算，分别推进154包／3.08s与127包／2.54s后在预测中耗尽预算；全部保存实际内步合法，0站立／0保持，不能把预算停止当物理违规或算法收敛，169独立计数与轨迹复核通过，不加码此路线。164核清060课程已接、065实际未撤力／β仍1、143未接完整课程。168复用065真实20更新权重，仅改ONNX一个β Constant .25→实际1；其余节点／权重不变、650公开样本均值对照通过，但零辅助真冷倒地三例均在.065s膝越限、0/3。因此导出纠正单独不足，不证明完整HoST不适用。[045实测与边界](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_045/report.md)。

**045现行下一项：完整成熟探索课程与最终零辅助接线。** 收口局部Jacobian／轨迹拟合；在当前原生GPU200链复用060/061及锁定作者四Critic／PPO／storage，明确latent增量→原18request、实际训练β→导出、settle／upright辅助／成功撤除及最终零helper边界。先短实测控制／施力／奖励／终止／GAE及真正倒地ONNX，再有界学习，不重复旧单Critic模仿配方或参数网格，不把未完成的课程算成功复现。155完整走路／停止后缀unit0/1、一次50Hz gamma=.998及去固定13秒／额外Idle保留，仍未在线学习。本轮101815CPU积分／101839位置查询／12闭环ORT，另离线1300Torch和1300ORT均值对照；新增GPU／PPO／Adam0。原047四个已知点／046保留，robustActor null、完整M0-S／M0-T／Bevy false、Goal active、heartbeat暂停，无外部消息。以下044及更早是历史，不是当前派工。

**历史检查点044：原成功路径不能直接作GPU示范，极小出生表示误差已单独使固定Teacher失败。** 160当前完整phase0/N32真实GPU冷对照：原047在1.450s自碰、原046固定请求在3.290s髋yaw越限，两臂0保持。161原CPU double完整13s通过、保持8.42s、六共享字段与046逐位一致；162复用该保存baseline，仅将出生官方INTEGRATION一次F64→F32→F64，之后仍CPU double／原请求／原驱动，3.565s踝越限，179包target仍逐位相同。该单例证明固定路径对出生表示误差极脆弱，不证明全部GPU/PPO失败或硬件不可行。两个记录器错误完整保留、未重复已保存baseline。157/159已完成只读历史与原碰撞过滤核查。[044实测与边界](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_044/report.md)。

**044早交接与当时下一项（已由045取代）：** 保留155完整走路／停止后缀unit0/1及一次50Hz gamma折扣，删除固定13s／额外Idle；原7.58s ready后7.64s嘴越限，仍0奖励，尚未在线学习。下一项建立当前训练后端可执行、带初态扰动余量的反馈Teacher或完整成熟探索候选；原046只作参考，不继续把它当GPU已成功示范扩大模仿PPO。先有界真倒地验证，再学习并支付实际完整Move成功。原047四个已知点／Teacher全部保留，robustActor null、完整资格仍false、Goal active、heartbeat暂停，无外部消息。以下043及更早是历史，不是当前派工。

**历史检查点043：保留原4/7成功基线，新学习候选未晋升；早交接只奖励完整稳定后缀。** 143一个公开神经图保留原嘴部输出、内部探索17轴，外部仍原18。144接口测试通过，145真实冷测仍4/7，149复核35初态／25轨迹字段及接触与078逐位相同。148实际4096决策／16384GPU积分／1原生TRPO／160CUDA Adam后，root过严的未使用参数字节保护触发；152重用其保存rollout、独立CPU随机流做1原生TRPO／160CPU Adam，确认参数微漂不改变嘴输出，联合KL.007386通过。但153独立冷测0/7，全在1.44–1.885秒躯干／胫部自碰撞失败，不扩大此模仿候选、不覆盖047。没有新增鲁棒起身。[043实测与边界](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_043/report.md)。

**历史043早交接与当时下一项（已由044取代）：** 147确认原连续站立4.58秒起、原3秒保持在7.58秒完成；080已删固定13秒及额外Idle，但实际Move在7.64秒嘴越限、奖励0。155新入口强制完整移动／停止后缀才给unit1，部分走1m但停车失败也为0；原gamma=.998仅按50Hz源转移折扣一次，相同完整成功越早越优。真实旧失败及部分成功的保存数据回归通过，尚未接入在线学习，不把接口称训练成果。[155指标与入口](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_full_program_early_handoff_entry_155/protocol.json)。下一项只做156当前GPU本体上原047和046路径的真正倒地可执行性接收，先确认成功源路径在训练后端确实可执行，再谈扩训；代码冻结、实际资源预检因另一应用延期，新增运行0。root不联系其他聊天，不启动重复作业／奖励网格；Goal active、robustActor null、完整M0-S／M0-T／Move14／Bevy仍false，heartbeat暂停。

以下042及更早为历史记录，不作当前派工。

**历史检查点042：真实GPU短学习完成，更新与终止接线通过，鲁棒起身仍未完成。** 130原生SB3-Contrib TRPO八更新，32768控制决策／131072 GPU世界积分／1280 CUDA Adam，183.33秒；原200/50Hz、模型、控制与正式门不改，一个世界始终真正完全倒地，31个原uniform RSI，没有半秒截断。8次GAE／KL与ONNX对照通过；新1/4/8版独立冷测均0/7。第8版6/7全13秒合法，但0站立瞬间／0保持，不把生存当技能。原047四个已知点通过及046 Teacher完整保留。[042实际学习与边界](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_042/report.md)。

**已定位的接线问题与因果边界：** 122查出旧半秒任务有非法非done正奖励、已失败prefix尾值及没有phase0实际训练动作，126/130另具名正式ET／cold1+RSI31已修正。SB3原KL按18轴平均，root漏做joint→per-axis单位换算；127同保存rollout/RNG控制完全逐位重现126，修正后联合KL.007281通过，129独立复核成立。但128两版仍0/7，因此修正只证明更新幅度，不证明起身。118/126实际失败与旧结果不改，不称完整DeepMimic平台复现。

**早交接目标保留：** 080已移除固定13秒和额外Idle，原站立保持完成立即交给冻结Move；只有同世界真实Move后缀成功才支付1，失败0，gamma=.998按50Hz源决策折扣一次，同样成功越早回报越高。实际047在7.58秒接管、7.64秒嘴轴越限，仍奖励0；第6秒外观不能代替实际接管。080尚未接入130这批模仿学习，不宣称已按早交接目标训练。[080实现／实测](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_source_switch_reward_env_080/README.md)。

**模型诊断已收口，现行下一项为真实首失败内步重放。** 131同CAD/BREP确认一处原下肢出口空域被凸包填实；135具名11→15叶局部候选已静态构建，138独立接收通过。root137真实连续冷对照：原Teacher和047仍13秒通过、站立8.42秒，新局部候选两臂分别3.08／2.08秒失败；首公开观测／请求／target相同，首5ms状态已因接触分岔。135不晋升，不继续逐点补孔。[137连续对照](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_torso_exit_local5_cold_dynamic_receive_137/README.md)。136证实嘴部范围来源为未认证机械止挡的操作候选；合法目标仍可伴随被动越限，但保存inner3位姿不足，不能判定全部根因或删限位凑分。[136来源／时间对齐](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_beak_passive_limit_source_audit_136/report.md)。139已冻结实际phase0八包失败请求；root140真实1024 GPU积分（0NN／优化）在.155秒复现jaw越限，目标+.02953rad、实际力矩+.39838Nm已刹车而jaw仍负速，Teacher控制32内步全合法。141独立比较40共有字段全体NOT_MATCHED（17项非逐位、10项超预定包络）；仅首失败事件、原请求／target及absolute F32时钟匹配，缺历史现场fullData，不认证整段严格复现或完整根因。当前只据此选择一个具名驱动／完整成熟课程候选，不追加原模仿配方长训或reward／LR／std网格，不删限位凑分。[140实测](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_phase0_beak_actual_inner_gpu_replay_140/README.md)。新8版[真实13秒失败录像](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_joint_trpo_learning_continuous_media_134/update0008_nominal_actual13s_standup_failed.mp4)已查看。Goal active、robustActor null、M0-S完整物理／M0-T／Move14／Bevy false；heartbeat仍暂停，不联系其他聊天。

以下041及更早为历史记录，不作当前派工。

**历史检查点041：原生GPU查询和N32紧凑采样接线通过短实测，尚无新增鲁棒起身。** 103在同完整输入/请求的原CPU查询与新GPU查询两臂实际运行：目标、ctrl、驱动力界及held驱动逐位一致，观察最大差1.19e−7；独立104复核实际支撑力误差0N、16组接触多重集合一致。未来状态非逐位，最大qpos差5.96e−8m/rad，不把静态隔离或短结果当作无因果扰动证明。105另具名泛化N32及audit/rollout存档开关；107两模式238项完整预步输入逐位匹配，原判据/更新计数一致、逐packet CPU FK为0。两臂各世界只累计0.08秒，不授技能、吞吐或完整GPU训练资格。[041报告](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_041/report.md)。

**041当时下一项（已由042取代）：108完整参考状态的GPU RSI/reset及私有双时钟，随后有界成熟课程pilot。** 106核清096缺RSI/episode/reset和作者时长课程；072/073曾有CPU RSI但不能自动授给新GPU链。保留原65＋5×65/原18、cadence phase及既有5项reward，reference clock不暗入Actor，不称完整作者phase/optimizer复现。108仅在备份目录准备接线；先冻结不同phase和selected-mask的真实首次step/隔离准入，再接RSL/明确terminal bootstrap；不扩大旧096失败配置。原047、080早交接奖励保留，080尚未正式PPO。本轮1088 GPU世界积分/64native calls、218CPU FK、0CPU积分/NN/PPO/Adam；104/106只读另为0。首次103独占资源preflight延期0执行；两个另具名短诊断记录同机原有进程，只运行64/1024积分且0优化，未停止作业、不声称GPU空闲，源端长训仍排队。Goal active、鲁棒Actor null、M0-S完整GPU训练/M0-T/Move14/Bevy false；heartbeat仍暂停，不联系其他聊天。

以下040及更早为历史记录，不作当前派工。

**历史检查点040：同状态GPU几何判据通过，修复足底支撑力分量接线错误。** 098在51独立实际保存状态（52行含1重复）上，原越限/穿透/up/COM/离地分类零分歧，包含实际失败与站立状态。官方source315返回force3＋torque3；旧027把index5的z扭矩当支撑力，新接口用index2。100由实际096完整保存solver Data提取，与独立旋转contact力的1N分类16/16一致，旧口径漏报4例；101独立复核通过。没有新step或用query力冒充live支撑，不据此归因全部学习失败或追授旧技能。[040报告](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_040/report.md)、[独立复核](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_gpu_query_independent_review_101/report.md)。

**040当时下一项（已由041取代）：102把已验收GPU postquery与正确live足底力接入当前逐5ms packet。** 有界实现准备中、未运行；根节点将冻结同初态/请求对照public obs、驱动目标、实际力矩、接触和4×5ms计数，原50Hz状态更新及门槛不改。此周期0积分/NN/PPO/Adam，56 CPU位置查询含初始化＋52 Jacobian、52 GPU query world、8保存solver力读取。原039一次PPO的KL失败不重写、不扩同配置；较优047、早交接080保留。Goal active、鲁棒Actor null、M0-S完整GPU训练/M0-T/Move14/Bevy false；heartbeat暂停、不联系其他聊天。

以下039及更早为历史记录，不作当前派工。

**历史检查点039：原生200Hz GPU接线和一次真实PPO更新已运行，尚无新增鲁棒起身。** 原模型/原50Hz驱动保留；089故障停批、093二十次输入恢复与mask隔离通过。091未来逐位回放未通过（qpos最大差3.73e−9），不授予确定性Teacher搜索。094启动错误已保留，096另具名修正后实际128 GPU世界积分／32决策／1 RSL PPO／20 CUDA Adam；ONNX误差1.49e−8，但原KL门0.070506>0.02失败，不扩大该学习配置，不覆盖047。[039报告](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_039/report.md)、[独立保存数据复核](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_native200_gpu_chain_review_095/report.md)。

**039当时下一项（已由040取代）：当前模型GPU独立post-state查询与原终止/奖励几何的同状态对照。** 复用027官方kinematics/com/collision/velocity查询，098有界实现准备中；根节点冻结实际保存状态后query-only验证，不改物理/驱动/门槛，不重复失败学习配置。合计本周期355 GPU世界积分／182native calls、531 CPU FK、0 CPU积分、1 PPO／20 CUDA Adam。096仅0.32秒链路，无13秒冷验收或技能资格；原协议陈旧字段另具名posthoc澄清而不改历史。原早交接奖励080保留，未混入本次接线验证。Goal active、鲁棒Actor null、M0-S完整GPU训练/M0-T/Move14/Bevy false；heartbeat暂停、不联系其他聊天。

以下038及更早为历史记录，不作当前派工。

**历史检查点038：共同成功路径的邻域脆弱性已实测，反馈候选仍未新增能力。** 078同真实bank275倒地快照/原驱动、只改颈q或qd，Teacher与Actor两臂35初态字段逐字节相同；当前046固定请求名义通过、六微扰均失败，047四个已有训练点通过/三个邻域失败。083保存证据显示名义踝最小余量0.8056mrad；q+1e−6在首5ms改变接触多重集，随后分岔，3.04秒髋yaw越限。这排除“共同路径邻域仍能执行，仅学生拟合导致失败”的单一解释，不证明硬件不可起身或接触算法缺陷。[038报告](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_038/report.md)、[同请求连续失败对照](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_nominal_neighborhood_continuous_media_087/identical046_Teacher_nominal_vs_neck_p1e6_to_failure3040ms.mp4)。

**真实成熟学习准入：** 084调用作者锁定sampler→GAE→DAPG/Fisher/CG→baseline完整RL顺序，1更新KL .00500144合格但冷测0/7，停止；不冒充作者从零BC或平台整体复现。086另具名公开状态距离门在既有3900示范中心强制保留原动作；2更新KL均合格、仍4/7，三个邻域失败，没有新起身。四点是结构保留，不是学习泛化；示范均值反馈梯度为0。085独立八条GAE/returns逐字节复算、四个成功冷轨迹24字段和35初态与078逐字节相同。新候选不覆盖较优047、不扩大此配置预算。[084](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_dapg_seeded_workflow_admission_084/README.md)、[086](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_dapg_anchored_feedback_admission_086/README.md)、[独立复核](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_dapg_independent_review_085/actual086_review.md)。

**038当时下一项（已由039取代）：当前原生200Hz模型与原50Hz驱动的GPU接线。** 先核已安装官方实现和真实旧AUTORESET/积分支持边界，形成具名最小短准入候选；仅XML可加载/旧GPU计数不继承当前模型资格，计算提速不自动证明学习收益。禁止删flag凑准入、重复奖励/LR/std网格或同数据RBF拟合。当前周期72,318 CPU积分/78,698运行FK/24,281 base或ONNX调用，3 DAPG/92 baseline CPU Adam；PPO/CUDA Adam/GPU物理0，087另960图形FK且新积分/训练0。原早交接奖励037保留，未混入本DAPG。Goal active、鲁棒Actor null、M0-S GPU/M0-T/Move14/Bevy false；heartbeat仍暂停，不联系其他聊天。

以下037及更早为历史记录，不作当前派工。

**历史检查点037：早交接奖励已完成具名单世界候选接线与真实负例验证，尚未用于正式PPO。** 080在原完全倒地、全prefix合法、原同Actor连续3秒保持且开始≤10秒后立即交给冻结Move，没有固定13秒或额外Idle。保留同一世界/驱动，实际后缀决定0/1并返回源switch转移，done=true/timeouts=0；后缀动作不混源采样，原γ仅按50Hz源决策折扣一次。原模仿正奖励仅作诊断，不让等待持续赚task分。5纯测试通过；最终047真正bank275在7.58秒接管，7.64秒beak_hinge越限1.4668mrad（原门.1mrad），奖励0，身体up仍.9997。这不是提前交接阳性，也不等于身体仍未站起。[080代码/实际收据](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_source_switch_reward_env_080/README.md)、[037报告](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_037/report.md)。

**037当时下一项（已由038取代）：最新名义Teacher的共同邻域可执行性。** 076只读证据确认旧boundary是各自优化的±1e−4中心，不是当前046固定请求的原点六邻域。先冻结一个有界真实对照，区分共同轨迹物理可执行性与047映射偏离，再选择局部反馈专家或完整示范约束学习；077已锁定DAPG作者完整工作流，不把PPO加BC或单组件导入称复现。停止本配置长训与奖励/LR/std网格，不重复旧中心矩阵、同数据RBF拟合或无证据扩训。080当前单世界/冻结Actor，在线版本与批量需另接；最终已显式记录源Actor版本，不能把在线动作统一归047。原13秒独立恢复验收、邻域与共同模型完整Move资格不变。三次080含首保存失败共4584CPU积分/5766FK/1146神经调用，PPO/Adam/GPU物理0；两次有效源14字段同字节，较优047/Teacher保留。Goal active、鲁棒Actor null、M0-S GPU/M0-T/Move14/Bevy false；heartbeat仍暂停，不联系其他聊天。

以下036及更早为历史记录，不作当前派工。

**历史检查点036：确定性动作表达缺口已修正，新反馈pilot仍未改善能力。** 068查得065 currentq/Tanh/beta=.25均值无法逐点表达2157/3900既有示范；训练Gaussian样本仍无界，不据此归因全部历史失败。070同初态原Teacher保持8.42秒，支持投影从.14秒偏离并在3.51秒颈越限；071独立43项复核通过。072原18绝对Gaussian均值／047公开基础＋无界390反馈、无beta/settle，零反馈新ONNX真正倒地13秒保持8.42秒且共享实际字段逐位复现。原生PPO首轮冷测0/5；073相同首轮逐位重现后按原生调度有界16轮，51200CPU积分/320CUDA Adam，7完整真倒地episode0成功、0共同up/COM达标内步，最终冷测仍0/5。原.02 KL准入失败未改成通过，不覆盖047、不授鲁棒或Bevy资格。[036报告](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_036/report.md)、[真实连续失败录像](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_failure_media_075/actual_ONNX073_fullfall_to_first_failure.mp4)。

**036当时下一项（已由037取代）：能纠正偏离状态的Teacher／成熟示范约束反馈学习。** 停止本配置长训与奖励/LR/std网格；开环轨迹不是任意状态专家，不能仅堆同数据拟合或喊DAgger。新候选须先保留名义成功并改善固定邻域，才扩方向。066早交接指标与一次终奖接口保留，原控制共同模型完整交接正例仍无，未接入本PPO，不造成功标签。主线仍200Hz物理／50Hz策略与驱动、原18语义、公开输入及原正式门。Goal active、鲁棒Actor null、M0-S GPU/M0-T/Move14/Bevy false；heartbeat暂停，不联系其他聊天。临时产物均ProjectBackups，生产只此计划。

以下035及更早为历史记录，不作当前派工。

**历史检查点035：早交接指标/终奖接口已实现，新的探索pilot仍未取得合格策略。** 066隔离适配器17测试与父进程重测通过：真实058四条早接失败均0，051走1m开发事件1、完整停止事件0；接管/非零指令/真实验证时间分开，奖励不重复折扣，尚未接入PPO，现无共同模型原控制完整交接正例。[奖励接口](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_handoff_terminal_bonus_066/README.md)。

**本轮真实学习与边界：** 作者HoST PPO/storage和四组奖励函数保持，声明Goose模型、原轴缩放、公开390输入及beta未入Actor等适配。065完成20 PPO/400 CUDA Adam/128000决策/512000 CPU积分，GPU物理0；175 episode中75真倒地、合格0，force/beta课程递减未触发，不能称完整成熟课程已跑通或被否定。8/20版冷测各0/5，保持0，未覆盖047。原KL准入失败保留，继续只属锁定作者调度诊断。064第4轮保存因磁盘满失败，账本PPO4/CPU102400、完整轨迹只前3轮；有效和partial均无损压缩保留。065以磁盘余量保护结束。[035报告](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_035/report.md)。

**035当时下一项（已由036取代）：真正倒地的启动控制接入及训练/部署动作路径。** 训练首31控制/.62秒屏蔽增量，导出从Tick1直接输出，课程beta仍1而最终导出.25；不能忽略这些未完成迁移条件。067只对照首31请求public-F32 currentq hold，原臂.095秒右膝越限，hold臂.220秒双胫自穿透.825mm，均未起身；屏蔽也不是已证明有效的启动方案，不把其当前适配继续扩大长训。先冻结一个实际可执行的启动/反馈候选，再推进早交接时间奖励和邻域能力。保留较优047/Teacher，正式鲁棒Actor null、目标active、M0-S GPU/M0-T/Move14/Bevy false；heartbeat仍暂停，临时产物遵守备份路径，不联系其他聊天。

以下034及更早为冻结历史，不作当前派工。

**历史检查点034：原成功终姿保留，反馈学习仍未合格；新增真实早交接计时。** 原047同Actor原倒地13秒开发点、自然抬头终姿和旧Teacher保护未变，正式鲁棒Actor仍null。新原生CPU MuJoCo／CUDA RSL PPO链路的零残差650控制与047逐位一致；054／056各8更新、057128更新确实完成优化，合计144原生PPO／2880CUDA Adam／147422 CPU训练积分，GPU物理0。05732／64／128三个导出候选严格冷测各0/5、保持0秒，关闭当前小残差配置继续长训，不覆盖成功基线。[034报告](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_034/report.md)、[057独立复核](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_native057_independent_audit_001/README.md)。

**用户最新要求：越早稳定交接到其他ONNX，表现越好。** 旧051固定Standup13秒后已调用Move，再额外Idle3秒，16秒才发非零移动指令，不能当最早可用时间。058实际从原bank275完全倒地分别在6.00／7.58／10.12秒直接接原Move385，无额外Idle；三条guard候选及6秒原控制均失败，嘴部关节在接管后.060–.120秒越过原.1mrad门。7.58秒此前已真实连续站立3秒，10.12秒身体速度仅.00037m/s，仍无法据此授予交接成功；也不证明更早交接不可行。[真实连续倒地→10.12秒接管→首失败录像](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_early_move_handoff_058/continuous_evidence/cold_fall_10120ms_rawMove_first_failure.mp4)、[四例收据](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_early_move_handoff_058/physical_receipt.json)。

**时间目标与本轮接线结论：** 已实现实际收据评价，分开记录首次站立、Move Actor接管、非零指令和后续合法1m／完整停止成功；早切失败为0，成功越早时间分越高，复用原50Hz gamma=.998，尚未注入PPO。原051只保留16秒指令后的走1m开发阶段，完整停止仍失败；没有成功就不填造handoff奖励标签，不重复折扣。[计时指标](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_034/handoff_latency_metrics.json)、[目标合同](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_034/handoff_objective_contract.json)。作者奖励根角速度已正确从MuJoCo body-local转world；训练探索可继续正式微量违规但前缀永远false、冷验收门不变。057所有32768决策/60个完成episode审计未误计成功，末198合法状态奖励复算误差0；语义修正未取得能力，不把它认定为历史失败的全部原因。

**034当时下一项（由035取代）：取得实际可交接终态并缩短到达时间，同时回完整成熟探索／轨迹学习建立反馈能力。** 保留047与最佳Teacher，停止同数据堆拟合、当前残差预算扩张和奖励/LR/std网格。恢复终态必须包含头嘴与原驱动目标／速度相容，接续正例由同世界真实Move执行见证；不能仅凭upright、静止或等待3秒标成功。正式同Actor3秒保持与未见倒地资格独立保留，运行时取消人为固定13+3秒等待要求。058只作为失败负例；下一新奖励／课程候选在实际正例接线核验后冻结，不以import作者PPO或精确点RBF冒充完整成熟流程。共同模型、GPU物理、原控制、四向鲁棒、Move14／Bevy仍false。原每两小时唤醒继续暂停，不联系其他聊天、不新建PM；生产仅此计划，临时产物全部在ProjectBackups。

以下033及更早记录为已冻结历史，旧“下一项”不作当前派工。

**历史检查点033：自然终姿已形成，原世界接续能走1m，完整接续与鲁棒性仍失败。** 新047 ONNX（SHA256 `bace11bc951e86c50b68de9303121725bada57afa08465b17b81813c8ab44859`）只因新增一条合格末段示范而一次拟合：3900实际公开样本，650右名义行替换、其余3250行逐位保留；65当前＋前5帧公开历史→18原轴请求，运行时无Teacher/时钟/出生ID/参考phase。原bank275完整倒地→同Actor站起保持→抬头13秒实际成功；未guard独立13出生仍6个训练点通过、7个邻近测试失败。正式鲁棒Actor仍null，未覆盖原035/022及最佳Teacher。[新ONNX](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_natural_jaw_neural_actor_047/multiteacher_history_actor.onnx)、[实际自然终姿](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_natural_jaw_neural_handoff_048/continuous_evidence/actual_neural_final13s_pose.png)、[033报告](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_033/report.md)。

**实际Move接续与明确变化：** 使用Move当前保留385（SHA256 `a67046a90bf797a6c016c4e40dc327feab353de1e1e1b24b46857b5ab5c67b70`），原18输出不改写。原控制下起身13秒＋待机3秒通过，再走.5712m后嘴耦合轴越限。具名050/051只把全程嘴驱动目标下限改为.001rad，原机械门槛和其余17轴/PD/碰撞不变；这是新控制候选，不继承原合同。051同世界真实起身→抬头→待机3秒→走1.0059m，停止0.19秒时beak_coupler_link越限.3862mrad，限.1mrad，完整停止窗口未完成。新目录重跑的27内步/25控制字段逐位相同；失败不放宽。Standup11叶/implicitfast与Move14叶/discrete身份仍不同，不授Move14或Bevy资格。[连续真实起身→抬头→Move走1m→停止失败录像](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_guarded_locomotion_phase_eval_051/continuous_evidence/neural_natural_pose_rawMove385_walk1m_stop_failure_continuous.mp4)、[实际收据](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_guarded_locomotion_phase_eval_051/receipt.json)、[新目录复现](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_033/reproduction_launcher_receipt.json)。

**本轮诊断收口与勘误：** 固定动作的几何／积分器配对、作者预测采样、共同邻域搜索、慢放、早切Idle及目标margin都未取得Move14完整Teacher，不继续同一局部网格、不归咎DOF不可行。050误把站立脚接地见证用于每个walking内步；051独立新评估允许walking短暂flight，原物理/躯干/起身与Idle3秒、停止1秒后3秒支持/速度/漂移门槛保留，仍越限失败。旧034末“.176m/s”实际是root速度；wholeCOM末值.02594，但停止1秒后仍65/401内步>.04且缺完整后续3秒窗口，资格仍失败。039/040缓存自比较不作确定性证据，042真实回滚独立匹配；旧渲染收据写错路径已按渲染前SHA恢复，051分离输出保护物理收据。细节和实际资源分项在033，CPU搜索/回放不算PPO样本。

**新GPU冷测与下一项：统一学习后端，建立真实反馈鲁棒性。** 053冻结047在实际Warp物理冷启动，289GPU积分/73ORT CPU推理，1.445秒躯干—右胫自穿透.1292mm超过原.1mm门，未起身；50静态数组相同、首观测差7.08e−8，不把后端差异全部归为浮点因果。052首积分后的接触收集器错误保留，053只修零力占位收集并独立冷跑，未PPO。下一小批采用同原生CPU MuJoCo生成Teacher和学习rollout，CUDA用于神经/PPO，先受保护零残差均值13秒对照与成熟RSL/RSI短pilot；不把CUDA Adam称GPU物理，也不扩大失败Warp配置。保留035/047自然终姿和原控制，停止RBF同数据堆轮数及嘴guard/margin网格。真实GPU物理迁移、完整共同模型接续与后续资格独立；资源使用结束已释放，不接管他人作业。正式冷验收保持完全倒地、同Actor、零辅助、10秒内开始连续3秒合格站立；邻域与方向资格未过不得宣称鲁棒。M0-S完整GPU/M0-T/Move14/Bevy仍false。生产仅更新此计划，全部临时产物在ProjectBackups。

以下032及更早记录为已冻结历史，旧“下一项”不作当前派工。

**历史检查点032：自然终姿的多Teacher神经原型，衔接仍未合格。** 新具名035 ONNX（SHA256 `403cf72e370e8db4c0229e76be102c2dfa85ae847ead7408012e12d69a847a45`）接受6条真正倒地／微扰Teacher、3900实际公开65维状态及18维F32请求，单次监督求解，部署只用当前65＋前5帧公开历史。同一个Actor从原右275、左250、真实背倒111及右颈−1e−6／±1e−4rad连续13秒起身、保持、抬头到自然终姿；这6个训练出生通过，另外7个邻近测试失败。13条独立ORT冷测试的保存状态／实际力矩／接触／时钟由17418次离线FK重新核对一致，6条成功路径与对应实际Teacher逐位一致。正式鲁棒Actor仍null，不将开发点统计当未见分布成功率。[新ONNX](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_natural_terminal_six_teacher_actor_035/multiteacher_history_actor.onnx)、[实际连续抬头录像](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_natural_terminal_six_teacher_actor_035/continuous_evidence/right275_natural_terminal_neural_continuous.mp4)、[独立13例](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_natural_terminal_six_teacher_actor_035/independent_natural_terminal_001/receipt.json)。

**Move接续的实际边界：** 已只读核对用户引用的Move聊天与冻结278 Actor，没有跨聊天发消息。质量／惯量／18驱动和轴序一致，但Move采用14碰撞叶／discrete200，Standup采用11叶／implicitfast200，不能继承资格。自然终姿从原完整倒地连续运行后直接调用原Move，在13.105秒嘴部被动轴越限。034具名对照仅将Move嘴请求固定F32 .04、其余17轴保持原输出，保留同一世界走出1.0029m／2.6秒；随后停止3秒漂移.0438m，但末速度.176m/s高于.04门槛，完整程序仍失败。嘴保持是明确动作适配，原Move资格未继承。[真实起身→接续→走1m录像](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_beak_hold_handoff_diagnostic_034/continuous_evidence/natural_recovery_to_adapted_move_1m_continuous.mp4)、[原请求／嘴保持配对](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_beak_hold_handoff_diagnostic_034/pair_receipt.json)。

**本轮学习与Teacher范围：** 027实际原生RSL PPO64更新、32768控制决策、131072GPU训练积分、1280CUDA Adam；8和64更新各0/5冷启动，停止扩大该配置并保留022。三分支交付左侧、右侧具名微扰及真实背倒Teacher，新的固定反馈邻域仍0/4，未取得一般反馈专家资格。028→030→033→035只因接收不同合格实际数据而各做一次拟合，没有继续同一数据堆轮数。Move原物理模型的独立Teacher合法前缀只到2.76秒；036修正已定位的TRF极小初始半径后，76种不同F32候选仍未通过完整240ms预测，保留失败证据，不宣称硬件不可行或MPC路线被否决。[本轮报告及保护](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_032/report.md)。

**032当时下一项（由033取代）：统一全程物理合同与接续资格。** 保留035自然终姿原型、较优旧Actor和Teacher；先在Move冻结物理模型中取得从真倒地到自然站姿并接上原Move的完整合法Teacher，再训练相同公共Actor。不得途中切换积分器／碰撞、重置世界／驱动或用嘴保持诊断授原Move资格。邻域扩展须有已验证的反馈能力；已有精确点成功不支持继续同轨迹无限拟合。M0-S-GPU完整物理／M0-T／四方向／Bevy资格仍false，原越限、穿透、3秒保持与停止门槛保持。

以下031及更早记录为已冻结历史，旧“下一项”不作当前派工。

**历史检查点031：真正右侧bank275的同Actor神经单例已通过，邻域鲁棒性仍未通过。** 冻结022 ONNX（SHA256 `64e43c1a1ae0a01af52f8f067fa7af91c5c886b3961275b0790ad4e0bd3a36ac`）从原完全倒地连续13秒运行、保持合格站立8.42秒，第379Tick／7.58秒完成连续3秒保持。原始冷启动、独立进程重测、原生ORT1.30.0 CPU及新目录复现均成功，19字段逐位一致；所有内步无越限／自碰撞，最大穿地2.518mm，驱动力矩和时钟遵守合同。无Teacher／MPC在神经执行中、无辅助力／位置写回／重置／站立Actor接管。[连续神经成功录像](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_031/continuous_evidence/neural200_full_fall_success_continuous.mp4)、[031报告](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_031/report.md)、[冻结候选](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_031/candidate_manifest.json)。

**031资格边界：这是RBF单轨迹监督回归开发原型，不是鲁棒PPO恢复策略。** 650个公开状态径向基单元和线性动作头，65输入／18输出float32，使用当前公开前63维，内部double计算；无时钟／phase／私有Actor量／运行时轨迹索引。8个颈pitch小角度／速度扰动均失败，最小±0.000001rad；额外200Hz forward的零扰动对照通过，不能把刷新当这组失败原因。[近邻失败连续录像](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_031/continuous_evidence/neural200_neck_p1e6_boundary_failure_continuous.mp4)。只授予原始右侧开发单例，四方向／任意初态资格仍false，正式合格恢复Actor仍null，M0-S/GPU／M0-T／Bevy未授予。

**031已定位的执行差异：** 018／020的微小离线拟合误差仍被物理／反馈轨迹放大，分别4.72／4.90秒失败。019专家能救回实际学生174控制步的偏离，但未通过附近初态专家准入，不能称任意状态反馈专家或通用DAgger。021先把预测和真实执行动作统一float32并重新生成合格200Hz Teacher，022再用同精度监督并通过独立闭环。024只删除同一022的最终float32 Cast、保持全部权重／世界／控制不变，首Tick差7.52e−15，4.99秒越限；保留Cast的原Actor成功。证明该精确轨迹存在动作精度执行问题，不证明所有PPO失败的根因都是精度，也不证明这种单轨迹拟合具有反馈鲁棒性。

**031本体与资源：** 使用独立 `implicitfast200` MJCF，SHA256 `70981622b88daa756e0b376bfb033612de546e76aa167c69f5a60274bb87de15`；从父模型只改源dt／原生积分器及独立名称／用途，原质量／惯量／11碰撞叶／20轴限制／18驱动未改，未保留加大yaw或软化接触候选。[本体入口](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_bank275_implicitfast200_workflow_014/candidate200_implicitfast/robot.xml)。003–025及复现启动器实际629910CPU积分／8398神经调用，含回滚预测与独立评估，不是PPO样本；本阶段新PPO／GPU物理0，GPU仅离线EGL渲染。18轨迹的独立物理／接触／力矩／时钟／连续保持重算全部一致；119当前保护项与8项旧50Hz成功基线全部匹配。原030课程训练账保持原身份，不与本监督原型混计。[复核](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_031/independent_verification.json)、[资源](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_031/cycle_resource_ledger.json)、[复现启动器](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_checkpoint_031/reproduce_neural_point.py)。

**031下一项只解一个问题：原倒地右侧邻域的真实反馈恢复。** 保留022严格单例作为受保护基线；回完整成熟探索／轨迹学习流程，以新200Hz实际成功路径和学生失败作为课程／可学习性依据，不再重复同一离线数据拟合或旧discrete局部优化网格。若采用HoST，四Critic／公开历史／被动settle／upright条件辅助／成功驱动的辅助撤除和动作幅度课程完整接线，Goose适配具名，最终原倒地零辅助验收；旧002头高度进阶与不足4%倒地帧不算完整起身。新源模型GPU小批量未准入不扩大长训；单例附近尚无反馈能力，不扩左／后／前或授予四方向。右侧路径可执行不证明所有方向可行，只有具体支撑不可达／壳体阻挡／力矩不足证据才改硬件。历史下一批全部保持历史身份。

**保留检查点030：右侧bank275闭环神经单例仍未打通，本轮接收／课程测试已封存。** 同一真实完全倒地快照下，旧Teacher50的13秒成功逐位重现，但Teacher200在1.155秒右膝越限、017学生在0.305秒失败。017首步观测完全相同，颈pitch请求误差0.007506经映射成为0.019516rad目标误差及约1.95Nm实际力矩差；只纠正该首步把失败延至0.500秒，尚未起身。局部原生跟踪器3.10秒失败，未取得反馈专家资格、未启动DAgger。旧Teacher仍是搜索轨迹，不能直接作为200Hz任意学生状态的正确反馈标签。[接收、控制边界与连续录像](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_bank275_200hz_feedback_001/report.md)、[200Hz独立MJCF入口](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_bank275_200hz_feedback_001/candidate200/robot.xml)。

**030课程候选结果：** 原生HoST四Critic／PPO/storage与条件辅助、辅助撤除、动作rescale课程，带声明的Goose动作／公开历史／RSI／后端适配，非完整G1成功流程复现或对029单变量。实际512PPO、10240CUDA Adam、262144决策、1048576训练CPU积分，GPU物理0，约20.94分钟；最终辅助力0、rescale=.25。18个具名中间候选均未完成原倒地神经单例；较长前缀288更新1.32秒，身体仍倒地，最终512更新0.075秒越限。独立进程原倒地冷启动与保存轨迹逐位一致，ONNX对训练网络最大差5.96e-7，失败最终候选没有覆盖较长前缀或旧较优Actor。所有物理／起身门槛保持；采用作者adaptive desired_KL=.01，实际最高.1394，不声称通过历史外置.02 KL诊断门槛。[完整课程／训练／独立ONNX／录像](/home/ethan/ProjectBackups/2026-10-11/Sai_Lab/goose_standup_bank275_host200_course_002/report.md)。

**030当时结论（已执行并由031更新）：200Hz原倒地上的可执行局部反馈能力。** 10%重置并不等于10%真实倒地帧，本批倒地训练帧仅3.73%–3.90%；608次高度进阶全部来自轨迹280／300／320，出生前沿仍300，没有神经保持3秒。不能以辅助递减、头高度进阶或合法前缀冒充完整起身，也不能据此归咎硬件。先重建能在200Hz闭环执行、通过原倒地独立重放的专家／单例反馈，再供神经学习；不继续同一离线数据堆轮数、扩大本失败配置、奖励/LR/std网格或重复四频率矩阵。训练可用轨迹中间状态，开发验收始终原始完全倒地、同一个Actor、零辅助、全程合法、10秒内开始同一连续3秒保持。单例通过后再解锁小扰动与其他方向。合格恢复Actor仍null，M0-S/M0-T/Bevy资格仍未授予。

**030当时顺序（历史，不另行派工）：** 单例闭环能力→附近小扰动→左／后／前倒。若采用HoST探索路线，完整采用被动settle、条件辅助力、成功驱动的辅助递减与动作幅度课程，最后全程零辅助验收；模型尺度、质量和驱动限制按Goose合同，不照抄G1数值。硬件微调只依据具体支撑不可达、壳体阻挡或力矩不足证据。历史“下一项”不再派工，不扩大029失败配置、重复四频率矩阵或联系其他聊天。

**唯一现行入口：本聊天独立负责 Standup 完全倒地起身。** Move已由用户拆分，本聊天不实施移动/转向、不覆盖原策略、不联系其负责人。电脑训练不考虑软底。当前21机器人刚体（另有world）、18主动轴＋2被动轴、11实际凸/box叶、10.430690821kg。独立源候选已改用原生implicitfast200；硬件、本体、碰撞、关节限制和驱动未改，旧discrete物理身份保留；不主动联系硬件/G1/其他聊天，不新建聊天/PM。旧冻结ONNX数值、旧十二臂PPO和本轮去共同学习混杂的新具名四频率对照分别封存；同一原integration_frequency_audit子Agent完成后闲置，其他既有Agent保持闲置。原移动、起身基线和最佳Teacher均受哈希保护。

**保留检查点029：原生HoST分组学习与公共连续动作前端的配对训练已完成，完整起身仍未交付、未晋升Actor。** 六臂共768原生HoST PPO/393216决策/15360CPU Adam；GPU被移动分支使用，实训前冻结统一CPU后端，实际CUDA Adam/新GPU物理0。单组与四组各0/24完全倒地恢复、0合法完整直立瞬间；三配对seed四组均缩短前缀，总均值3.8417→2.4633秒，准备站姿18/27→0/27仅诊断。三对初始Actor/观察/出生/首行为物理批逐位同一身份，原.02 KL门槛全部通过，原生PPO/storage未改。与E66存在声明的共同适配差异，不称单变量比较或完整作者课程复现。已看连续真实前倒1.80秒失败录像，躯干持续前倾、最终关节越限，没有接近站起。该配置不扩大长训，不覆盖较优行为与最佳Teacher。[029训练、独立验收与录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_standup_native_host_checkpoint_029/report.md)。

**029资源与下一项：** 本轮408851CPU积分（训练393216、评估15635），正常训练/评估FK415828、接触额外15635、视频52；CPU Actor70710/Critic140160及影子/导出另记。393216训练行无非法非done，24原生GAE快照与15635保存状态/接触的越限、力矩、自碰撞、时钟和同段成功复核一致；13输入、3实现、028的12文件、E66的1334文件与24保护文件匹配。累计主线CPU14361194、历史GPU物理11166356、CUDA Adam920700、TRPO7274、HoST PPO4992、RSL诊断PPO649；本轮CPU Adam15360单列，独立频率4115881CPU单列。下一项核对完整成熟课程与既有4224次适配，再冻结一个最终零辅助验收的有界候选；不照抄G1身高常数，不继续奖励/LR/std网格、旧静态拟合或局部规划。原频率Agent继续闲置；目标未完成，M0-S/M0-T/Bevy资格不变。

**保留检查点028：任务组时间系数对照已完成，完整起身仍未交付、未晋升Actor。** E66只恢复作者task汇总系数，三共同seed的真倒地合法前缀平均2.0583→3.1167秒，但完整恢复0/24、合法完整站起瞬间0；准备站姿4/27另算。新控制臂128行为/物理批、最终Actor/Critic及17条独立ONNX轨迹逐位复现原F3/73101/50Hz，支持本次单变量比较。4臂共512原生RSL PPO/262144决策/10240CUDA Adam，物理CPU、GPU物理0；原模型/驱动/频率/其余三奖励组/课程/出生/学习配置/门槛保持，任务奖励数值变大不算能力收益。保存真正左侧倒地到5.16秒自碰撞失败的连续录像并看过：躯干持续贴地，COM下降，没有接近起身。局部合法时长收益不足以解锁能力，不扩大同配置长训。下一项回完整锁定HoST流程核对四Critic/分组GAE归一化和历史接线，先在实际失败批做作者调用链准入，不继续扫奖励权重/LR/std或局部求解器，不把静态差异宣布为全部失败根因。[028报告与连续录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_standup_task_scale_checkpoint_028/report.md)。

**028资源与保护：** 本轮272491CPU积分（1024准入、262144训练、9323评估），包含65536控制复现训练、新修正学习196608；正常FK277353、接触额外9323、视频136及神经调用另列。9323保存状态/接触记录的20轴越限、力矩、自碰撞、计时和同段3秒成功重算一致；19输入、027的11文件、065的18诊断与24基线匹配，1334独立E66产物封存。主线累计CPU13952343、历史GPU物理11166356、CUDA Adam920700、TRPO7274、原HoST PPO4224、RSL诊断PPO649；独立频率4115881CPU单列。原频率Agent保持闲置，未重复四频率矩阵/新建Agent/联系其他聊天。合格恢复Actor null、目标仍未完成，M0-S/M0-T/Bevy资格未变。

**保留检查点027：学习退化已实际复现，未修复、未晋升Actor。** 冻结73101/50Hz训练后ONNX从原本稳定的同一完整快照，两次均123Tick/2.46秒因右膝越限失败，19字段逐位复现；未训练保持请求Actor实际150Tick/3秒合法。准备站姿只作诊断，完全倒地仍未交付。直接调用native训练Actor仍同Tick失败，动作与ONNX最大差2.98e-8、q轨迹差5.08e-8；65536保存决策无非法非done或重置错序，首真实批12物理终止的原生GAE未跨重置/timeout接续（误差1.49e-8）。26奖励函数体与HoST源AST一致，但汇总存在明确适配差异：原任务项乘1/约束乘dt，Goose任务组额外乘.02、相对其余组缩小50倍；本频率pilot还有单Critic提前合计与作者四组Critic/优势归一化差异。仅证明调用链差异，不宣布它们是学习失败根因，不归因全部历史HoST/Teacher。当时下一项只对作者任务组时间系数做具名、等预算、其余保持的独立对照，不同时改Critic/奖励权重/课程/频率，不覆盖主线。[027真实回放、补充覆盖与三个结论](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_standup_native_failure_checkpoint_027/report.md)。

**027资源与补充接收：** 原频率Agent已接收用户最新公平训练补充，对应026的12臂/双层实际评估已有完整证据，继续闲置，不重复矩阵。主线本轮519CPU积分/531FK/519CPU ONNX、123native Actor/33Critic，0GPU/新PPO/Adam；累计CPU13679852，其余与026相同，专项4115881 CPU另列。519保存状态/20轴越限与力矩重算零误差；5148频率产物、026的17文件及24保护基线匹配。红反馈仍失败；合格Actor null、目标未完成，M0-S/M0-T/Bevy资格不变。

**保留检查点026：公平独立PPO对照已完成，四档完整恢复仍0/24，未晋升任何Actor。** 50/100/200/500Hz×3共同训练种子，全部从头Actor/Critic/空Adam、相同模型/驱动/奖励/课程/出生分布，策略和外层驱动均50Hz。每臂128原生PPO/65536决策/1310.72世界秒/2560真实CUDA Adam；共同normalizer漂移和10s/13s时钟混杂已消除，原0.02最大minibatch平均KL门槛全部通过，实测最大0.00386553。三个种子自身平均合法前缀分别2.0583/4.3442/6.9708/5.0083秒；各自策略迁源50Hz为2.0583/2.4858/6.1300/2.3342秒。200Hz三种子配对均保留部分前缀收益，但没有完整起身或合法完整直立瞬间；准备站姿另算，不授予Bevy资格。只有3独立训练种子，24是8出生×3策略实例，不能宣称完整能力的统计显著差距。所有训练后策略的合法前缀低于未训练保持请求基线，不能只以高频横向优势称作学习成功。[026三个结论、接触量化、计算成本与复核](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_standup_frequency_learning_checkpoint_026/report.md)。

**026独立资源与保护：** 全批训练3342336次CPU积分、786432控制决策/1536PPO/30720CUDA Adam，GPU物理0；完整专项4115881CPU积分，含33份独立评估772050、日志失败512及登记适配前重复评估983，额外两项不计入等预算比较。父任务接收561轨迹和772050保存内步/求解接触，20轴越限重算误差0、实际力矩越界0、时钟及同段成功判据一致；181输入、30基线与5148冻结产物匹配。接收新增积分/FK/推理/优化器0，主线累计CPU仍13679333，其余与025相同，专项单列。当前没有频率训练在运行，原子Agent闲置；不重复矩阵或自动升主线频率。200Hz仅保留为后续具名源候选，未授予源/目标/恢复资格；主线与Bevy50Hz合同保持。

**保留检查点025：当前q基准对照失败，拒绝该候选训练；完全倒地起身仍未交付。** 原063前倒68Tick失败实际回放两次、19字段逐位匹配。064仅改变公共请求基准，8真倒地（含前倒）＋9准备站姿×3共同种子×2基准，初态/噪声/模型/驱动/奖励/门槛保持。当前q虽缩小目标差，却让三种子真倒地平均合法前缀0.335/1.9025/0.7325秒降至0.2625/0.265/0.275秒，原预冻结准入未过，不进入训练。102条轨迹、4023保存内步、44输入和24保护文件一致；189条驱动记录无制动边界清零证据，不以此猜测DOF不足。主线本批4159CPU积分/4471FK、0GPU物理/网络/优化器；累计CPU13679333，其余与024相同。原062/063、移动/起身、最佳Teacher均保留。合格Actor仍null。[025证据](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_standup_control_anchor_checkpoint_025/report.md)。

**保留窗口后检查点024：128轮能力pilot实际完成但未晋升，完全倒地起身仍0/70，合格Actor仍null。** 063恢复062的Actor/Critic/Adam，原生RSL PPO/GAE/Adam不改；公共65×6/原18目标请求、模型/驱动/物理/奖励/课程/出生分布保持。独立具名任务时钟13秒、≤10秒起身起点与同段3秒保持，旧四频率协议/成绩不改。真实65536CPU积分、128PPO/2560CUDA Adam，152.69秒；最大minibatch平均KL0.003912984过原0.02门槛、网络实际更新、ONNX对照误差7.45e-9。训练223个真倒地结束episode无站起；独立7+旧63真倒地均无完整瞬间，原8短流仍全部合法、9准备站姿最长hold1.58秒。与062相同7案例合法13秒相比，063平均合法前缀降到2.79秒，因此保留062、不覆盖原移动/起身/最佳Teacher、不扩大同配置长训。70个已开封右26/左29/前12/后3回归不授正式资格；不能以失败证明DOF不足。137产物、18保护文件和10675保存内步越限/力矩/时钟核验匹配。本轮CPU76211、GPU物理0；024主线累计CPU13675174、历史GPU物理11166356、CUDA Adam910460、TRPO7274、HoST PPO4224、RSL诊断PPO137，独立频率专项另算。[024完整失败、基线保护与连续视频](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_standup_capability_checkpoint_024/report.md)。

**保留023首次失败与动作连续性证据：首次失败已证明动作接管问题，原生PPO配对入口通过固定学习条件检查，完整倒地起身仍0/7，合格Actor仍null。** 同一完整世界/驱动快照，059真实请求8条短流0/8合法，只保持上次公共请求8/8合法，两次逐位重现；初态均合法，四个踝roll距限位0.22–0.28rad且旧速度20ms外推不越限。062用相同零输出头、Actor/Critic学习参数/种子/16世界初始观测，配对绝对请求与公共上次请求＋学习增量；各1024真实CPU积分、2原生RSL PPO/40 CUDA Adam，最大minibatch平均KL0.00141364/0.00128791，均过预冻结0.02。相同采样预算内物理终止60→20；所有结束episode仍物理失败，不能声称探索全稳定。独立ONNX增量组短流8/8通过、真倒地合法13秒由初始6/7到训练后7/7，但两臂完整起身均0/7、无起身瞬间。七例右3/左3/后1是已开封开发回归，无前倒正式覆盖，不授予四向/Bevy资格。新skip适配不增加Actor私有信息，不改奖励、驱动、模型、物理或门槛；旧残差ONNX0/63失败仍保留，不能归因全部历史失败。158产物及18保护文件匹配、8601保存内步越限重算零误差。[023因果回放、配对PPO与连续失败视频](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_standup_action_continuity_checkpoint_023/report.md)。

**保留022归一化因果与两个未通过pilot：** 同一16世界×32Tick真实采样，原生PPO禁止权重更新后20个minibatch的LR轨迹仍与真实更新完全相同，最终1e-5；逐Tick恢复采样normalizer后分布/概率逐位一致、KL全零。结束统计下零梯度联合KL均值32.44074、最大746.55225，真实Adam又放大KL。这是独立新鲜RSL配置的共同限制，不归因所有历史训练器。原生关闭Actor/Critic额外normalization并降低初始LR的059及沿用同输入、fixed3e-6的060均完成2更新/40Adam，最大minibatch平均KL分别0.05168957、0.02304235，均未过预冻结0.02诊断门槛；不扩大门槛、不继续扫LR、不晋升ONNX。122原输入哈希、18保护文件及532保存qpos越限重算匹配；059/060首8轨迹逐位相同。所有本体、奖励、驱动、物理与技能门槛保持。[022因果证据、失败与资源](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_standup_learning_cause_checkpoint_022/report.md)。

**保留021独立物理频率与新鲜PPO审查：** 四档50/100/200/500Hz×3共同种子，每臂65536控制决策、1310.72世界秒、128原生PPO/2560CUDA Adam；分别同频及源50Hz代理测试，真倒地均0/21（同7个已开封出生×3策略，无前倒正式覆盖）。500Hz最终自身平均合法前缀0.12571秒，迁50Hz为0.08667秒，收益主要来自一个种子。局部接触/探索截断因果收益成立，尚未证明最终PPO学习成绩的显著差距。父任务重新核对924条保存轨迹、147517内步qpos/力矩、3416交付文件与3437原输入身份检查；全部匹配。主线和目标50Hz不自动升频，未晋升任何Actor，M0-S/M0-T及恢复资格仍false。[021完整接收、三个结论与资源](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_standup_frequency_reception_checkpoint_021/report.md)。

**保留020局部规划结果：** 成熟预测采样实际失败状态修复及六个浮点时钟边界因果检查，正确动作分支合法、错取上一请求越限；只修舍入尺度查表，合法实际前缀74→124Tick，11段预演/实际fingerprint逐位一致，仍未起身，随后没有合法完整1秒展望。不继续同一局部规划预算/权重网格。[020证据](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_standup_planning_clock_checkpoint_020/report.md)。022新增诊断CPU2560、真实原生RSL PPO5/100CUDA Adam，另列影子PPO1调用/0Adam。023新增CPU10689（061回放40、062训练2048/独立评估8601）、原生RSL PPO4/80CUDA Adam；GPU物理均0，保存接收0新积分/推理/优化器。023历史主线累计CPU13598963、历史GPU物理11166356、CUDA Adam907900、TRPO7274、原HoST PPO4224及RSL诊断PPO9分别计账；CPU含规划/回撤/回放，不全算训练样本。ONNX/导出对照/FK及渲染单独计数。其他integration/Move作业不属于本Standup审查，不重复、接管、停止或混算资源；GPU前检查实际占用，不声称整机空闲。

**保留019行为与失败结论：** 原完全右侧倒地私有纠偏Teacher仍3.92秒完整起身、500Tick合法保持5.16秒，392／500Tick回放逐位复现；不是自主ONNX成功。046／049冷启动均0/7，旧／新TVLQR相同两噪声seed均0/2；完整一致示范仍失败，标签冲突不是全部根因。原较优行为和原最佳Teacher均未覆盖。参见[019报告](/home/ethan/ProjectBackups/2026-10-08/Sai_Lab/goose_standup_teacher_transfer_checkpoint_019/report.md)。

**用户独立物理频率／PPO因果审查已完成：** 同一 integration_frequency_audit 子Agent保留冻结ONNX数值实验，并完成四档各自新训练策略的同频及源50Hz测试。策略、prepare/commit、目标限速、驱动历史、热状态、phase、奖励、gamma/GAE和超时仍按20ms外层语义；每Tick1/2/4/10次真实原生积分，全部内步检查原门槛。高频实际力矩按原17隐式位置执行器和真实嘴转子随状态变化，不把外层更新重复执行。实验全部独立备份，不改主线参数、模型或已有基线；源50Hz迁移只算代理，不等于Rapier/Bevy资格。[专项最终入口](/home/ethan/ProjectBackups/2026-10-08/Sai_Lab/goose_integrator_frequency_audit_001/final_report.md)。

**用户补充已逐项核查，未重复训练：** 四档各自从头PPO、共同种子/课程/奖励、等控制决策/模拟时间/更新预算和自身→源50Hz两层评估均有实际收据；只读核查新增积分/推理/优化器0。边界包括GPU物理0、每臂仅65536决策、无前倒正式覆盖、训练10秒/评估13秒（晚于7秒起身在训练中无法保持3秒），以及评估全程合法更严格；首起身与最长保持尚未绑定同一连续段，但保存数据没有真正倒地假成功。保留旧协议及成绩，未来具名验证先统一计时与连续成功段，不能改旧判据凑收益。[补充覆盖表与三个结论](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_frequency_requirement_coverage_001/report.md)。

**独立数值、学习与迁移结论分别保留：** 相同状态/动作14分支，合法50Hz0/14、100/200/500Hz各8/14；高频减少穿地，但1000Hz仍6例越限。discrete RefSafe随dt裁剪有效接触刚度，不是只增加同一数值映射的调用。独立新PPO四档均无完整技能；12臂首轮后LR全部降到1e-5且停满128轮。021单批normalizer反事实联合KL均值57.9–70.8（阈值0.02）；022已补实际32步调度、零权重更新因果回放，但仍不证明全部失败根因。不能以全失败断言高频无价值或DOF不足。[新鲜PPO及迁移报告](/home/ethan/ProjectBackups/2026-10-08/Sai_Lab/goose_integrator_frequency_audit_001/ppo_four_arm_001/report.md)。本integrator专项Warp加载拒绝冻结AUTORESET禁用flag，实际GPU物理0；统一CPU原生3.15＋CUDA原生RSL5.4.2 PPO，未删flag凑准入。专项CPU真实积分3563368、比较786432控制决策/1536PPO/30720Adam，加pilot1544PPO/30880Adam；约58分钟比较训练，限于本专项的作业已退出，资源不重复加进主线。

**十小时窗口结论：未达到自主鲁棒起身目标。** 窗口为2026-10-08 09:21:03至19:21:03北京时间。最终512更新神经ONNX预先冻结后，63个未参与训练/调参的真实完全倒地新出生独立验收 **0/63成功、0/63完整站起瞬间**：右23、左26均合法运行13秒但停在地面；前12和后2触发穿地/关节越限。4个开发倒地也是0/4。9个名义站立初态保持3秒仅算站立。最终候选标记FAILED_UNQUALIFIED，合格恢复Actor仍为null、长期目标未完成。该63集四向覆盖不平衡，尤其后倒仅2例，不能宣称正式四向资格。

**十小时窗口冻结检查点016（历史结案）：** 最终一批512原生TRPO／2097152实际GPU积分／81920 CUDA Adam，耗584.44秒，其中真正倒地训练帧530912，网络参数确实更新；ONNX对网络最大误差5.96e-8。窗口累计11166057实际GPU积分、12868123 CPU积分、4224原生HoST PPO、7274原生TRPO、891720 CUDA Adam；CPU含规划分支/诊断/回放，不能全部称作训练样本；FK/优化器/渲染与物理计数分开。12份保护文件全部哈希匹配。[完整结案与资源账本](/home/ethan/ProjectBackups/2026-10-08/Sai_Lab/goose_standup_gpu_chain_checkpoint_016/report.md)。现行主线累计以027为准，独立频率专项另算。

**016历史公共ONNX具名候选：** `obs[1,65]`＋前五帧公共`history[1,5,65]`→原18轴目标请求`actions[1,18]`与`next_history[1,5,65]`，冷历史为零。每Tick一个完整神经图、一次20ms真实积分/驱动，decimation=1；没有出生身份、参考/Teacher目录或站立Actor接管。最终网络使用对公共上次请求的残差，属于明确Goose适配，不声称与作者原Actor完全相同。[最终ONNX（未合格）](/home/ethan/ProjectBackups/2026-10-08/Sai_Lab/goose_standup_sealed_ONNX_evaluation_001/actor.onnx)，SHA256 `b2044557c2ced00f239016e5dea2c764ee33f9ad993c6ac9475acdbd033a52ab`。[冻结候选清单](/home/ethan/ProjectBackups/2026-10-08/Sai_Lab/goose_standup_gpu_chain_checkpoint_016/final_candidate_manifest.json)。未接入Bevy、未晋升、没有latest自动替换。

**已核验的成熟方案与适配缺口：** 作者HoST `70bb580949a336a920833700e4b5dc3bf7fe87ce`、HumanUP `7516e0f27e6f4d1e7365cf64ea577a78247bd8cb`、DeepMimic `1f915c52fcd4b95b5f5f15b759ae91bd81e9a801`；源码与适配范围保存在[成熟流程记录](/home/ethan/ProjectBackups/2026-10-08/Sai_Lab/goose_standup_mature_workflow_review_001/maturity_readme.md)。原十小时窗口的PPO更新过冲配置已关闭，其训练器为未重写优化器的原生SB3-Contrib TRPO、平均联合KL≤0.01；窗口后具名学习诊断062/063使用原生RSL5.4.2 PPO/GAE/Adam，另设预冻结0.02最大minibatch平均KL准入，不能混称同一训练器。完整身体模仿奖励独立标量CPU误差≤1.67e-7；实际轨迹760完整快照用于训练Reference State Initialization，不用于伪造测试出生。

作者DeepMimic失败回报为0，本适配曾额外叠加HumanUP−50，已修正。相同初始Actor/Critic逐位一致、同种子的两批32更新对照均零倒地恢复，因此不能认定此项不一致是全部根因。上游部分reset清空未选世界ctrl的适配问题已用完整积分/驱动状态逐位回归修复。原反向课程v18在右294/后161/左250停滞且真正倒地训练帧0，已关闭；均匀RSI残差020/021完成后仍没有真倒地成功，不继续相同配置长训。所有中间ONNX各自保留，最终候选参考294Tick的保持1.68秒低于中间384更新的3秒，没有覆盖旧较优行为。

**主线H16准入053已完成：** 固定官方MJPC Python predictive-sampling Policy及源码提交，同原生20ms完整驱动／世界快照两次500Tick回放，q/qv/目标／热状态／实际力矩／ctrl／边界／物理指标逐位匹配，回撤及零噪声副本逐位复现。只接入作者动作样条与完整过渡适配，未运行规划搜索、未进行神经训练，不声称已复现完整MJPC约束规划。复制状态及矩阵限幅的小范围适配与首次失败证据单独记录。本项1000CPU真实积分、1006FK、0GPU／优化器／ONNX；主线累计CPU13533449，其余累计与019相同。参见[053准入报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_mjpc_transition_admission_053/report.md)。

**029结束时的历史瓶颈与下一项（已被030取代）：** 完全倒地自主ONNX未交付，合法完整右侧Teacher和较优Actor仍保留。026将数值、学习、迁源50Hz分开，主线频率不自动改变；027/028/029实际诊断分别限定各自范围。029作者四组与公共连续前端组合已真实否决，不能仅导入原生PPO/storage就称完整成熟方案复现。下一项核对锁定作者的settle、upright条件外力、成功驱动的force/action-rescale课程及既有4224次历史适配，再选择能取得最终零辅助能力的一个受控候选。没有断言私有Critic漏接，不照抄G1常数、不继续奖励/LR/std网格、旧静态拟合或局部规划，不扩大失败同配置或降低资格门槛。原频率Agent已接收补充并完成，继续闲置；不重复矩阵/联系硬件、G1、Move或新建Agent/聊天/PM。正式未见集另封，旧63只作回归，M0-S/M0-T未过不扩大源/目标长训或授予Bevy。

**新局部诊断连续录像：** [同世界完全倒地→14冻结神经请求→私有滚动规划停止](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_mjpc_receding_clock_fix_057/continuous_evidence/right_student_prefix_receding_planner_stopped_continuous.mp4)。实际2.48秒／124Tick，未起身；双视角同时、无重置／接管，只以CPU查询保存姿态渲染，不新增积分，不替代原完整Teacher录像。

**实际连续录像：** [最终ONNX左侧完全倒地13秒](/home/ethan/ProjectBackups/2026-10-08/Sai_Lab/goose_standup_final_continuous_delivery_001/left_fallen_final_neural_13s_continuous.mp4)、[63例收据及全部轨迹](/home/ethan/ProjectBackups/2026-10-08/Sai_Lab/goose_standup_sealed_ONNX_evaluation_001/independent_001/receipt.json)、[方向统计](/home/ethan/ProjectBackups/2026-10-08/Sai_Lab/goose_standup_sealed_ONNX_evaluation_001/direction_summary.json)。物理50Hz/录像25fps、双视角同一世界、无剪切/重置/接管。最佳Teacher与原学习基线均保留，Teacher仅作动力可行证据。

| 保护的真实完全倒地Teacher | 首次完整站起 | 实际范围 |
| --- | --- | --- |
| 右侧原bank275 | 3.92秒 | 合法保持24.42秒；29.62秒开始穿地，连续录像含失败尾段 |
| 左侧原bank250 | 4.78秒 | 达到完整起身瞬间，未保持3秒 |
| 后倒原bank225 | 3.22秒 | 达到完整起身瞬间，未保持3秒 |

[最佳右侧Teacher连续录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_standup_continuous_delivery_001/right_full_getup_and_long_hold_with_failure_continuous.mp4)与[12份哈希保护](/home/ethan/ProjectBackups/2026-10-08/Sai_Lab/goose_standup_gpu_chain_checkpoint_016/preservation_receipt.json)。原计划与各轮历史检查点完整保存在[窗口结案前计划快照](/home/ethan/ProjectBackups/2026-10-08/Sai_Lab/goose_standup_gpu_chain_checkpoint_016/training_plan_before_final_window_close.md)；其历史“下一批”不作当前派工。所有临时工具、模型、媒体、轨迹、依赖、构建与收据在ProjectBackups，生产仓库本轮仅更新此计划入口。

下面三条主线保留整体产品需求，当前本聊天只推进M3恢复。

## 1. 目标与当前基线

以下为整体产品的三条主线；用户已拆分负责人，本聊天只实施M3跌倒恢复，其他主线保留需求与接口。

| 主线 | 最终行为 |
| --- | --- |
| 移动 | 站稳；前进走路0.4m/s、跑步0.7m/s与Shift切换；后退、横移、转向、启停；通过室内接缝、低障碍、缓坡和小台阶 |
| 跌倒恢复 | 前倒、后倒、左右侧倒、真实扰动跌倒中自主起身，站稳后继续移动 |
| 拾物 | 用户在 Bevy 选定物体，接近、弯腰用嘴叼起、站直、携带、放到指定位置 |

- 当前源端物理 **200Hz**、策略与驱动状态 **50Hz**：每20ms一次推理／prepare／commit和4次5ms原生积分，明确计数，无隐藏子步。Bevy目标端物理50Hz独立验收；原始及历史50Hz合同不覆盖。
- 首版载荷 **100、200、300 g**，目标位姿由 Bevy 提供，不纳入自主视觉识别。
- 不训练独立坐站、踢球、轮滑、特殊翻滚；姿态变化可作为三条主线的内部动作。
- 共享 DGX Spark，先小实验筛选，每批长训最多两小时。

原始溯源输入是 `/home/ethan/Projects/Sai_Rotbots/artifacts/Goose_V0.1/goose_460_training_checkpoint_20261002.zip`。16,314 个清单文件哈希全部匹配；源装配 460 件、33 个运行刚体、18 主动轴、14 被动坐标、10.430762603 kg、65 观测／18 动作。原证据只有 0.2 秒冒烟和 RSL 初始化，没有优化器更新或合格策略。

权威为包内 `training_checkpoint_contract.json` 和 `stage_one.py`，不使用历史 53／10 的 `control.py`、`runtime.py`、`rsl.py`。零碰撞 training_reference 只用于参数/FK核对，不能作为任务训练本体。

MD 经验落实为：先核本体、实际驱动和观测时效，再调奖励；冷启动与热切换分别测；恢复必须包含继续移动；拾物按实体结果验收；训练完成和候选晋升分开。参见 [MD 冻结交付状态](/home/ethan/ProjectBackups/2026-10-01/Sai_Lab/microduck_player_package_015/DELIVERY_STATUS.md)。

## 2. 技术路线与接口

### 源端200Hz物理与50Hz策略／驱动

原包物理 0.1 ms、力矩 5 ms、策略 20 ms，必须建立新合同，不覆写原模型身份。

1. 当前Standup源端为独立 `implicitfast200` 具名候选（本体入口见031），从已核验004及后续 `goose_task_proxy_11_discrete_mjlab160_v1` 父模型派生，21体／18主动轴／11叶（9原凸网格＋2内接原生足box）；总质量、COM、完整惯量、轴位和自碰撞过滤保持。足box的遗漏上部形状与有界制动修正保存在历史版本与合同，不继承原完整形状或控制资格。
2. 足底使用普通原生刚体接触，不使用软底弹簧、压缩曲线或四点替换。软底静载与行程资格不再阻挡电脑能力训练；本期不要求仿真实物软底。
3. 当前上游 MuJoCo 3.15使用原生 `implicitfast` 5ms积分，旧 `discrete` 保持父模型与历史身份；位置驱动采用独立合同：17轴原生隐式位置驱动，嘴部仍驱动真实输入转子；原动作目标、限速、力矩限额、名义颈部前馈、延迟、热代理与350W功率预算显式保留。它不继承旧显式PD的实际力矩身份。
4. 原33体微脚垫、凝聚软底、自定义BE／discrete和材料诊断路线只保留历史失败，不作为当前训练派工；源与目标资格分别验收。

Rapier 每 Tick 只进行一次 20 ms 积分，CCD 不引入额外时间步。内层约束迭代与物理积分分别计数；历史 PGS 参数诊断不作为新网格派工。弹簧采用 ForceBased SI 单位，碰撞体不重复增加质量。

嘴部采用实际四杆闭合约束，电机只驱动 `beak_input_rotor`，与 head_roll 产生相反反力，约束将力传给 jaw/coupler。禁止逐 Tick 写 qpos、FK 搬动物件或焊接附着辅助抓取。

任务用碰撞代理保留脚、嘴、壳及运动干涉关键表面、来源和过滤映射；不填实空心结构，不扩大自碰撞排除范围凑成功。原邻接过滤、显式嘴销配合排除与碰撞 masks 均可追溯。

### 训练与目标接入

**MuJoCo 200Hz物理／50Hz策略 → mjlab／MuJoCo Warp + RSL-RL PPO → Rapier 零样本评估 → 必要的有界目标微调 → Bevy CPU ONNX 独立验收。**

- 当前执行顺序先完成源端模型和成熟链路。复用 MicroDuck 的 `MjSpec`／`EntityCfg`、任务配置／注册、原生 VecEnv、上游 PPO、复载与导出组织方式；Goose 只增加必要的模型、65／18 控制适配及任务项，不另写训练器、通用环境框架或求解器。MicroDuck 的 BAM、轴序、身体尺度和奖励数值不直接移植。
- 统一 MJCF 从已核验交付派生：显式质量与完整惯量、关节坐标、18 轴驱动、嘴闭环及来源映射只有一份权威；视觉细节与任务碰撞代理分开。原约 15,715 凸块不能直接作为成熟批量训练模型。代理保留空腔及脚／嘴／外壳关键接触面，经几何与载荷短检查再采用。
- 源端优先原生 `mj_step` 和上游已支持的积分器／约束。v2–v6 的自定义 CPU 修正仅用于解释历史失败；未经独立兼容检查不进入 mjlab／Warp，不为保留某个局部方案继续改内核。
- 当前新源候选明确设定 `timestep=0.005`、`decimation=4`，策略50Hz。驱动prepare／commit及目标限速等仍按20ms更新，原生隐式驱动力矩在每5ms求解；奖励、折扣和终止保持原物理时间语义并检查每个内步。历史50Hz配置保持原身份。当前源配置为 mjlab 1.6.0、上游 MuJoCo／MuJoCo Warp 3.15、warp-lang 1.15；原十小时窗口实际 RSL-RL 安装为5.0.1，原生更新AST及GRU源码与作者5.4.2另行对照；窗口后062/063另具名使用原生RSL5.4.2，实际导入路径及身份以各自收据为准。实际加载路径及代码哈希另存收据；历史1.3／3.10环境不与当前身份混用。
- 物理准入分为 **M0-S 源端** 和 **M0-T 目标端**。源端物理及 GPU 小批量链路通过后，可先做有界源端站立／移动基础训练，保留 `target_qualified=false`。源策略形成后执行零样本目标评估；M0-T 未过不做目标训练或授予 Bevy 能力资格。两侧和最终行为验收门槛保持不变，不用源成绩代替迁移成绩。
- GPU 小批量验证完整惯量、约束、驱动、接触容量和步进，才扩大并行数；CUDA 网络不等于 GPU 物理。
- 新增 Rapier 持久无头 batch worker 与 RSL VecEnv，训练和游戏复用物理步、驱动、观测与重置。
- 分别报告源端、零样本迁移、目标适配后、Bevy 游戏四层成绩。
- 复用现有单世界、事件队列、物理线程和位姿发布，Goose 独立 profile；默认 MicroDuck 60 Hz 和 61／14 合同保持兼容，不继承其 BAM、尺寸或轴序。
- 正式游戏不依赖 Python。

### 三个 Actor 和任务接口

- 移动 **65→18**，零速度承担站立；恢复 **65→18**。
- 拾物 **82→18**：基础 65＋物体相对机身位置 3＋相对姿态四元数 4＋放置点相对机身位置 3＋六阶段 one-hot 6＋目标有效性 1。
- 阶段为接近、弯腰对准、合嘴、抬起、携带、放置；TaskGoal 提供物体身份、位姿、放置目标、有效性，训练/游戏坐标、布局、归一化一致。
- 18 轴顺序和目标动作语义以原合同为准。额外真实质量、接触力、地形可用于 Critic 和评价，不暗中混入 Actor。
- 显式状态机每 Tick 只选择一个 Actor；目标失效中止拾物，跌倒恢复优先；物体持有状态依据真实接触判断，无瞬移补救。

## 3. 里程碑与训练课程

| 里程碑 | 工作 | 退出依据 |
| --- | --- | --- |
| M0 物理接入 | M0-S：统一 MJCF、原生源端逐轴／嘴／足底／接触短检查；M0-T：目标端同版本复核 | 分别报告源／目标资格；整体 M0 仍要求同版本两侧通过 |
| M1 链路 | 优先锁源端依赖、小 batch GPU rollout、一次真实 PPO 更新、复载与 ONNX 对照；目标 batch 后续按需要接入 | 源端物理已通过，真实积分和优化器更新、有限数值与完整收据；目标资格单列 |
| M2 平地 | 站立→低速前后→横移→转向→启停反向→走路0.4/跑步0.7与切换→混合指令 | 两档持续速度、切换和独立移动验收；首个可操作 Bevy 候选 |
| M3 恢复 | 四类倒地→随机初态→真实扰动→继续移动 | 完整链通过，自然跌倒与直接倒地分别报告 |
| M4 地形 | 5/10/20 mm、正反缓坡、小台阶、方向变化与停止 | 各档达标，终点稳定 |
| M5 拾物 | 空载探地→近夹取→三重量→随机位置→运输转向→放置 | 各重量完整物体任务通过 |
| M6 集成 | 热切换、持物异常、用户输入、科学站实体、性能、连续运行 | 冻结游戏与三主线报告 |

M2/M3 共享站立基础并行；M4 需要稳定移动与恢复；拾物可达性从 M0 排查，完整学习在移动基础合格后展开。

移动奖励覆盖速度/yaw 跟踪、路线进度、终点稳定，并约束滑移、摆动、限位、力矩与功率。恢复覆盖姿态改善、稳定支撑、保持和继续移动，允许声明的身体/腿部接地，取消旧 nonfoot、低高度、低 upright 的立即终止。拾物覆盖接近夹持点、真实接触、离地、保持、运输、放置；低头本身不是成功。

逐体读取随机化范围，名义条件开始逐步加难；热切换保留前驱策略姿态、历史动作、驱动状态和相位。

## 4. 固定验收标准

结果出现前冻结协议。失败报告未通过，不移动出生点、改阈值或拼接重置后的轨迹。

| 项目 | 标准 |
| --- | --- |
| 物理 | 20 冷重置、逐轴/脚底/嘴部，无非有限状态、警告、异常能量增长或穿透逃逸；源端实际200Hz积分／50Hz策略与次数一致，目标端50Hz另行准入 |
| 电脑刚性足底 | 原生接触、真实支撑面和滑移／穿透结果；取消软底压缩、曲线及行程验收前置条件 |
| 站立 | 60 秒无跌倒，漂移≤5 cm |
| 移动 | 前进走路0.4、跑步0.7、后退0.15、横移0.1 m/s、转向0.6 rad/s与左右±π掉头；200独立案例整体≥95%，两档及切换分别报告；稳定前进段≥20秒，实际COM前进均速分别≥0.4/0.7m/s，同时报告速度分布、位移、路径/航向和步态，不用指令/峰值代替 |
| 输入停止 | 短长按、Shift先/W先、单独Shift、走↔跑重复切换、两档带转向、转向中松Shift、松W、反向、失焦、暂停恢复；停止1秒内≤0.04 m/s，随后3秒漂移≤5 cm；切换保全实际姿态/驱动/动作历史，不靠重置过关 |
| 地形 | 5/10/20 mm、±5°坡、20 mm台阶，每类≥50例、≥90%；不授予连续楼梯资格 |
| 恢复 | 四方向各50例，10秒起身、稳定3秒、再移动1 m；整体≥95%、每类≥90%；真实扰动另测 |
| 拾物 | 三重量、圆柱夹持段/带把手物体；底部离地≥80 mm、保持5秒、携带2 m并转向、放置误差≤100 mm；每重量≥60例、≥90% |
| 集成 | 同世界移动→跌倒→恢复→移动→拾物→运输→放置；20固定程序全过，30分钟实际运行 |

恢复还要求直立度≥0.95、COM高度≥名义站立85%，不能瞬时竖起就交接。开发、训练、验收集独立；正式评估独立进程、确定性 ONNX、auto_reset=false。先核同状态观测/动作目标/实际力矩，再比较接触任务，不要求跨引擎长轨迹逐位一致。

性能要求持续50 Hz、无积累欠账，物理＋推理P95≤16 ms；渲染独立并插值，目标机实测画质、帧率与交互延迟。

## 5. 执行、排期与交付

每轮：**单一可检验问题→冻结版本与指标→短对照→≤两小时训练→独立评估→晋升或保留失败。**

- 首轮5–15分钟 pilot 测吞吐、内存和接触容量，再定 batch/更新；记录真实物理样本、优化器更新和资源消耗。
- 当前用户已明确G1暂停且禁止主动联系其他聊天：GPU前仅检查实际占用，不发G1/硬件协调消息、不停止他人进程。root只负责Standup、独立评估与恢复交付；既有IKTeacher子Agent闲置，不按历史派工自动唤醒。GPU有界使用，不干扰其他进程。
- 两次受控实验无改善就回查物理、初态、观测、终止和奖励分项，避免局部死循环。
- 保存模型/合同/代码/依赖/配置/seed/策略哈希；候选显式指定，不用 latest。M0-S 和源 GPU 链路未过禁止源端长训，M0-T 未过禁止目标训练；数值/约束/容量异常停止该批。
- 每关交付通过/失败/适用范围、录像、算力和下一实验，基于真实吞吐滚动估时。
- 当前实施聊天的 `Goose ProjectManager` heartbeat 已按用户要求暂停，不自行恢复，不新建聊天。原用户授权的 `bevy_performance` 子 agent 当前闲置，不按历史周期自动启动。Move由独立负责人推进，本聊天持续完成Standup；既有IKTeacher Agent已完成且闲置，不按历史派工自动唤醒其他Agent。性能工作不得改碰撞形状／过滤、求解配置、物理参数或观测／动作合同来凑速度。

历史移动线曾运行源端有界PPO；本Standup窗口已实际执行GPU蒸馏、HoST PPO及原生TRPO，物理在CPU，当前资格与实际累计以页面顶部最新冻结检查点为准。最早M0产物归档 `/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/`；本页为唯一计划和状态入口。最终交付冻结策略、本体/合同、Bevy包、复现命令、三主线报告及未通过清单。制造/电气/实物载荷仍由机器人工程侧维护；仿真资格不自动等于实物资格。

完成当前可行性诊断后，长期路线仍为**统一MJCF／冻结碰撞代理 → 原生源端准入 → 成熟小批量PPO与导出 → 站立／移动／恢复源策略 → 目标迁移**。当前不唤醒Bevy性能agent；本聊天只做恢复准备、训练与评估，Move分支独立推进，GPU实验有界。每条能力线每周期只选能解锁下一阶段的一个问题；局部实验达到诊断目的即收口，连续两次无改善就回到模型或成熟流程，不追加求解器分支。历史检查点中的“下一批”只记录当时决定，以本页顶部现行任务为准。

现行首要工作以页面顶部为准；不重新启动已收口的300Actor/投影参考配置或扩大局部搜索。源/目标/正式三主线资格仍分别报告，完整目标未达。

参考状态初始化课程依据：[DeepMimic §Training / Reference State Initialization](https://arxiv.org/html/1804.02717v3) 与[作者reset实现](https://github.com/xbpeng/DeepMimic/blob/master/DeepMimicCore/scenes/SceneImitate.cpp)。仅采纳reset阶段的参考状态课程思想，不采纳运行时写真实姿态、不同物理步长或作者控制器合同。
