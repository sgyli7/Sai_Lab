# Low stock cart02 · collision003 · actual bite height420mm

新独立候选`low_profile_stock_cart_02_collision_003`。**车18叶＝8解析box＋6解析cylinder＋4凸mesh**，实际MuJoCo3.10读回；4mesh共188支持顶点、360凸包三角面，单块最多64点。解析体没有运行三角面数。鹅14＋车18＝32接触叶，地面另计；没有货物实例，20kg仍为目标，不是验收成绩。

`cart.xml`为单车，`goose_cart_scene.xml`为鹅＋车；完整相对引用本数据树`inputs/physics_v0_2_20261018_04`。**加载后使用显式keyframe `low_handle_420_pitch10`（索引0）**，不要把默认零位当设计姿态：

```python
m=mujoco.MjModel.from_xml_path('goose_cart_scene.xml')
d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0)
mujoco.mj_forward(m,d) # 非积分；本次没有调用mj_step
```

`initial_state.json`记录独立/合场景qpos、29维机器人qpos、实际杆中心world(0.300,0,0.420)m、cart原点world(0.646125977018,0,0)m、pitch初角+10°、真实支座与近嘴距离。用关键帧摆姿态，没有改04源轴位、限位或robot.xml。车轮/前转向/拖头共9个被动铰链保持，不通过焊接鹅、吸附或锁车摆图。

唯一硬件输入是Rotbots已锁02、122实体完整MCI，身份见`source_identity.json`。空车7.730322989642kg，含0.35kg明确未建紧固/锁止预留；满20kg货物目标总27.730323kg。质量由CAD实体和预留分配而来，**不从碰撞体积重算**。真实CAD中pitch下游件已转+10°，构建时将其COM/完整惯量绕真实pitch逆转至各刚体零位轴，再聚合；模型关键帧+10°重新得到原始CAD姿态。`body_mass_map.json`保留完整原始122件、变换矩阵/体帧和源归属。加载MCI对原生CAD＋同一预留模型的总COM误差2.22e−16m、完整惯量最大差3.33e−15kg·m²。

预留0.35kg的COM=(260,0,180)mm来自源，**内禀惯量仍是650×450×30mm均匀框架分布的实验假设**；不是CAD实算或称量。轴承内外环运动质量划分亦为待采购核对的假设。

`cart_collision.glb`和对应PNG是实际关键帧加载代理导出/CPU渲染，不是新CAD图或AI图。保留空篮腔、真实Ø125×25轮、Ø12咬杆与两支独立叉臂；部分装饰孔/排水槽和壳体台阶有明确代理近似。**本版仅平整地面/该保存FK**，36项叉板/座/轴/刹车接触省略合计1.722290933kg，质量全部保留；安装板侧向最大超轮胎27.5mm。没有门槛/地缝/叉架近脚、近障碍或扫掠资格。

当前轮阻、转向/轴承摩擦、pitch平衡弹簧刚度/预载仍为0占位/未冻结。不能据此声称真实车低滚阻或可拖20kg。**无动力学、训练、GPU，未授动态拖运或转弯资格**。硬件静力另报告：20N直拉改善，但20N＋3N侧向FAIL，不能由本包加载通过覆盖。

旧车01/collision002全部文件保持原字节。源02/CAD及原04不因代理生成而修改。
