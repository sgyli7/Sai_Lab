# Goose V0.2 · 硬件验证数据入口

本目录专属于 **Goose V0.2 硬件验证数据**。已完成输入与历史资料归属交接，并归档[购物车静力筛查](analysis/cart_static_screen_001/README.md)的已执行代码、输入和结果。静力归档没有重新求解或积分；当前没有启动训练、GPU或定时任务。

后续实验由 **Sai_Rotbots 的 Goose V0.2 硬件优化聊天**负责组织和授权。Picking 只负责 V0.1，不是 V0.2 的执行者；Move、暂停的 Standup 和 G1 工作树均不参与本数据工作区。历史文件中保留的 Picking 路径只是来源记录，不代表当前归属。主实验计划与硬件CAD仍由 Rotbots 本聊天维护，本目录不另立主线。

本机独立源工作树从已提交的 `05b3a38a8d8ec1988cfbd21850f373ae98052a24` 建立，首源提交为 `201b90f81d911ae27d478ba5840e334197ab5f1c`。远端 `codex/goose-v02-hardware-validation` 为避免带入其他未发布训练历史，从已发布main `2128db711351b8c33fe2505e487473f35a3d2966` 仅新增本专属目录，首发布提交为 `558e356b639cacd4232cc040955f966586bf96dd`；两者专属目录子树完全相同，全仓树不相同。后续发布沿该远端分支仅更新本目录，不改V0.1或其他引擎/训练资产。见[发布来源收据](analysis/low_handle_data_integration_001/publication_lineage.json)。

## 固定输入

固定鹅输入是 [physics_v0_2_20261018_04](inputs/physics_v0_2_20261018_04/README.md)，包含归档中的全部 **53个文件**；[原ZIP](inputs/physics_v0_2_20261018_04_native_reference.zip)也逐字保留。

- ZIP SHA256：`1dd11b212697f2b0005c045e4874c9ae1657e0f97863ae0cfde57b09c2d2a9ba`。
- 整机质量 `16.970203928833527 kg`，23刚体、18控制轴、4内部机械坐标、14凸碰撞块。
- 200Hz驱动/物理、50Hz策略接口；连续扭矩、速度/电压和350W预算均保持原包定义。
- 内嵌物理和执行器合同的02版身份照原样保留。没有根据碰撞体积重新生成质量、重心或惯量，没有修改任何轴或能力参数。

[input_versions.json](input_versions.json)登记输入身份，[copy_manifest.json](copy_manifest.json)提供全部复制文件的源、目标、字节数与SHA256；运行输入内部的manifest和文件均未改写。只读检查入口不会导入包内运行适配器或MuJoCo。

## 已关闭历史

| 数据集 | 真实范围与结论 | 资料入口 |
| --- | --- | --- |
| `grip_100g_pilot_001` | 100g物体、固定躯干/头部的局部夹持探索；未通过自由整机抓取、搬放或长期保持。 | [精选结果](history/grip_100g_pilot_001/selected/summary.json) · [原始索引](history/grip_100g_pilot_001/raw_data_index.json) |
| `ground_drag_load_002` | 0.5–4kg前置探索。0.5kg自由整机倾斜失败；局部固定躯干牵拉不能当作整机拖拽成功。未通过整机0.5m、60s或20kg。 | [正式收口分类](history/ground_drag_load_002/selected/final_classification.json) · [原始索引](history/ground_drag_load_002/raw_data_index.json) |

001原manifest记录178项，002正式manifest记录204项；两份目录及382项所记录文件均已只读哈希核验。002正式manifest SHA256为 `13b53f113b1a7186c532924396dd1dd642508686da85f9e2555744d7484bbf5c`。

大型原始日志、逐步轨迹、MJB及连续视频留在原合规目录：

```text
/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/grip_100g_pilot_001
/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/ground_drag_load_002
```

没有移动或修改原数据，没有将这些大型原始文件加入本工作区。这里保存原始manifest、复现源、原订单/配置、场景声明、真实冷启动状态及精选结果。它们都保持原字节，旧绝对路径和截止时间没有被偷偷改成新任务。汇总入口是 [raw_data_index.json](raw_data_index.json)。

历史源码与 `REPRODUCE.md` 是归档资料，**不能直接执行 `pilot.py serve`、`drag_pilot.py`、旧渲染程序或原复核脚本**：部分程序会写原结果或使用已结束的订单。完整历史复现仍需原始大数据及其声明环境；未来运行须由Rotbots本聊天另发有界新单、新输出目录和数据映射，不能覆写旧目录。本次未运行这些归档程序。

## 当前低把手候选：03/R4与碰撞004

[当前车体](assets/low_profile_stock_cart_03/README.md)使实际咬杆、pitch与yaw枢轴同高420mm；前端10°形状已在CAD中固定，初始车pitch坐标为0，不能再转10°。[独立审核](analysis/cart_low_handle_independent_002/independent_release_gate.json)的接受范围以原报告为准，仅限声明的静态候选。

空车设计估计7.815342442756589kg，含0.35kg未定硬件预留；134件真实CAD的MCI计入10车刚体、9被动轴，预留独立I仍为实验分配。车20块＝8box＋6解析cylinder＋6凸mesh；mesh共212支持点/400凸面、单块最多64。新增两片真实偏置承力板各一块，保留板间空气，不将它们合成填满空隙的凸包；螺钉不逐件设碰撞。鹅14＋车20＝34，地面另1，货物实例0。解析primitive不报网格面数。使用输入声明中的选定keyframe，不能把默认中性q当已审核姿态。

旧02/碰撞003保留为**被拒绝的负载优化候选**，并未覆盖成新版本。其杆心420mm而自由pitch枢轴375.7197mm，20N会额外产生约0.8856Nm，需要约3.5265N向下嘴载；补齐后的两种声明假设下均略退步。见[原静力归档](analysis/low_handle_study_001/REPORT.md)、[三次补充LP原结果](analysis/low_handle_pitch_load_supplement_001/README.md)和[G拒绝意见](analysis/cart_low_handle_independent_001/independent_release_gate.json)。原Fz0数值是负担重分配，head_pitch约.152→.471而neck根约.511→.378，不能称所有轴减负或节电10%。

新R4水平20N相对共高pitch不产生该额外力矩，仍须承担真实下游自重舌载：旧2.171112N、新2.205510N。原已执行的有限理想静力比较约.68562→.62001，仅支持这一姿态负担重分配；不是最大拖力、功耗、夹持或20kg能力升级。20N叠加3N侧力的持续工况仍FAIL，20kg动态、整机0.5m、60s均NO_RESULT。滚阻、关节摩擦、弹簧刚度/预紧未标定，平地无近障碍以外仍未资格。

原53输入/ZIP、旧R2/002、原001/003数值/源码及原20项manifest均保留。后续CAD构建在13:02/13:05以重定向覆盖了旧构建日志，当前日志26,798B与原manifest52,186B不同；[收据](analysis/low_handle_data_integration_001/appended_build_log_receipt.json)明确19吻合＋1覆盖日志，不伪称20项一直同hash。[新增复制映射](analysis/low_handle_data_integration_001/source_copy_manifest.json)保留原绝对路径和新归档位置，归档代码不能原地执行。

## 历史R2理货车静态输入与来源纠正

[车体输入](assets/low_profile_stock_cart_01/README.md)采用 `low_profile_stock_cart_01_collision_002`；[C已锁22项清单](analysis/cart_collision_001/artifact_manifest.json)为当前身份，旧collision_001保存在资产history ZIP，不能沿用旧类型摘要。当前车18块＝8 box＋6解析cylinder＋4凸mesh；4mesh合计188支持点/360凸面，单块最多64。鹅14＋车18＝32块，地面另1，没有货物实例。车体10刚体、9被动轴；原鹅04的全部53文件及MCI/轴/驱动未改。

修正靠嘴轴后静态净空3.7701mm，对照真实CAD约3.8545mm；这只覆盖声明的单姿态夹持区。叉架、轴、制动等36项未单独做接触叶，但其1.722290933kg全部保留在质量/惯量中。首版仅用于平整、无近障碍的静态输入，不能证明门槛、窄障碍、整段连续扫掠或夹持拖运。被动轴摩擦、滚阻、弹簧当前为0占位；0.35kg预留的COM/完整I也是实验分配，不代表真实硬件已确定。

[独立G准入](analysis/cart_independent_review_001/independent_release_gate.json)已FINAL，接受范围仅为有上述限制的静态碰撞首版；未授20kg、整机0.5m、60s或训练资格。车榜和20kg无轮滑块榜仍均为 `NO_RESULT`。本次集成人只做哈希、JSON/XML和资产路径检查，没有重新compile、forward、LP或积分。C/G先前已完成的静态加载/有限forward由各自原报告说明，不冒充本次新验收。

[来源元数据纠正链](assets/low_profile_stock_cart_01/source_inputs/r2_metadata_correction_01/source_metadata_changed_physics_unchanged.json)明确区分：R2原身份中的mass metadata `eb430f…` 与当前 `be6945…`。只把support内矩形x下界30改为80mm；反向单字段替换按原格式序列化，精确恢复旧SHA，质量、COM、完整I和其他字节不变。原参数、STEP、GLB身份未变；C冻结source_identity登记当前hash，其原报告保持原样，不伪称各时刻一直同hash。小参数/质量/身份/纠正收据已复制到source_inputs，大CAD源仍由Rotbots维护。最终support审阅另说明实际rail壁厚1.5mm；旧2mm标签不作当前依据，300mm高假设货物COM与150mm干涉筐也不混用。

[旧D静力归档](analysis/cart_static_screen_001/README.md)保留9.2119kg旧车情景。其实际TCP约(300.0000624,0,485.5002402)mm/root z约0.005m，与当前杆中心(300,0,480)mm/context root z0.0055m不同；约5.5mm外力点差在20N下对应约0.110Nm。旧u约0.6405只保留为近似历史筛查，不能登记为当前车杆静力已通过。本次未重算LP，也不改原D结果。

## 新目标与缺口

[benchmark_register.json](benchmark_register.json)把两类条件分榜，均为 `NO_RESULT`：

1. **20kg有效货物＋低矮理货车**：20kg指货物，车重另计。当前为03/R4车体与collision_004静态候选，空车设计估计7.815342kg（含0.35kg未定硬件预留）；没有20kg货物实例或动态试验，榜单仍NO_RESULT。
2. **20kg无轮滑块**：独立条件，不与有轮车结果合并。物体几何、接触材料和运行订单尚未确定。

下一阶段还需要自由整机负载姿态、后退/停止控制、真实支撑位移与速度、起动/持续拉力、夹持保持、轮地行为及声明的摩擦/柔度。现有探索不能证明机械最大载荷，不能声称通过20kg、整机0.5m或60s。鹅04保持原冻结输入；理货车另列为静态首版候选，不修改鹅的质量、轴或驱动。

## 仅静态核查

使用Python标准库，不需要仿真环境，也不调用任何模型、接触或训练函数：

```sh
cd experiments/goose_v0_2/hardware_validation
python tools/verify_static.py
# 本机原始归档仍可访问时，额外核对复制源和原始manifest全部文件：
python tools/verify_static.py --check-originals
```

该入口只核对哈希、ZIP、JSON/XML结构、相对资产路径及源码语法；不import运行适配器或历史脚本，不调用forward/step，不重放接触。输出到stdout；[static_validation.json](static_validation.json)是交接时已保存的静态收据，不是新物理验收。本次核查不执行任何历史脚本。资料的发布身份以Git提交和发布收据为准，目录存在或静态校验通过不代表动态任务通过。
