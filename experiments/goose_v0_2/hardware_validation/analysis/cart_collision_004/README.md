# collision004 · R4/420mm同高pitch · 静态首版

实际20车叶，鹅14，合计34，地面另计，未加载货物。6mesh共212支持顶点/400凸包面，单块64；另8解析box/6解析cylinder无运行面数。两片offset承力板使用CAD六角外凸轮廓各一叶，保留22mm中隙；18个有限探点无误填。wheel仍真实Ø125×25；原36接触省略范围保留，新10小五金无独立叶但全部质量计入。

新资产`../../assets/low_profile_stock_cart_03/`，显式keyframe `collinear_420_front_fixed10`。**pitch初角0**，固定前端10°不能double-rotate。实际咬点(300,0,420)mm、pitch枢轴同高；roll真实固定轴方向[cos10,0,−sin10]。图是实际编译代理CPU白底渲染，作者已亲自查看。

`coupled_static_load.json`采用R2/R4真实CAD全pitch下游同集合规则：旧0.326599kg/2.171112N舌载，新0.385699kg/2.205510N；旧2.16N为部分集合近似，当前含轴向挡圈等完整下游件，未修改旧冻结资料。20N无平衡：68.562%→62.001%；自重完全平衡：63.182%→56.535%。同高力线使额外竖向项0。侧向3N仍**FAIL107.698%**，没有删工况或新DOF。

只新增3LP/3forward，复用原真实04静力库定义，不运行144矩阵或改求解器。旧2个完全平衡Fz0结果按相同机器人/姿态/外载复用；源代码/哈希和每个tau均保留。此静力求解使用未连接的机器人真实J^TF和给定车端嘴力，不是滚动车或嘴摩擦仿真。

证据：loaded_collision_stats、initial_state、mci_and_plate_gap_checks、metadata_correction_chain、source_input_immutability、earlier_collision_immutability、resource_receipt。新134全实体MCI对已加载COM/完整I差2.22e−16m/4.71e−11kg·m²，明确保留预留惯量实验假设。旧002的22项和003的30项逐项字节不变。

实际构建源`build_collision.py`。MuJoCo3.10 CPU OSMesa softpipe；入口仍读取明确锁定的Rotbots R4与Lab04，仅输出新A03/D004；不是任意复制就能运行的portable CLI。复算需新输出/输入映射，禁止覆写历史包。0积分/训练/GPU；加载2forward＋少数耦合静力3forward，不含任何动态拖拽、20kg、0.5m、60s或转弯资格。
