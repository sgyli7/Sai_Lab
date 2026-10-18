# 008 有载全身拖车诊断：未完成连续半米

已结束 6/6 次、6357/48000 个真实物理步；0 GPU、0 PPO。鹅、004真实自由车、自由20kg货物始终同场。未认证任何真实交替摆脚周期，未完成末端3秒保持，不能声称会连续后退或嘴硬件通过。

|试验|真实步数|终止|身体最大后退 mm|车最后记录PRE-state后退 mm|货最后记录PRE-state后退 mm|
|---|---:|---|---:|---:|---:|
|1|2200|20mm setup+return|20.83|-15.92|-15.84|
|2|668|diagnostic_gross_material_slip_over20mm|14.12|9.85|9.91|
|3|2188|tilt_over25deg|5.66|39.32|39.38|
|4|931|diagnostic_gross_material_slip_over20mm|6.77|9.00|9.07|
|5|164|tilt_over25deg|7.37|9.15|9.17|
|6|206|tilt_over25deg|5.23|12.18|12.19|

负后退距离表示最终向前。1仅setup，包含返回段；2–6主尝试无返回。3的车约39mm移动发生在身体未后退/颈牵引、最终脱嘴的失败过程中，不能算全身拖运。

2/3旧runner曾按计划相位清空滑移，已披露并离线复核；旧失败后段失格。4起采用真实air/plant epoch与逐5ms积分前停止。终止文件的保守5ms预测与已执行滑动分列，不能合并称实测。

5加入真实足载荷/COM/身体姿态反馈与腿世界目标IK，嘴跟身体实际后退加≤5mm预瞄；6仅改pitch反馈符号。两次均在settle倾斜过25°，未到卸载转移，不证明硬件不可能；现Teacher尚无稳定受载全身支撑控制。窄single原caps数学缺口另见raw/hardware_torque_boundary/REPORT.md，不能静默增大电机。

嘴杆代理接触及>0.5mm穿透保留，嘴未获公平硬件合格判定。下一最短缺口是先让同场受载Teacher在原驱动下稳定身体并真实卸载一只脚，再讨论连续交替；禁止把相位请求或颈缩当步行。

原速明亮物理代理视频：
- [试验1](/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_loaded_diagnostic_008/attempt_001/neutral_proxy_original_speed.mp4)
- [试验2](/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_loaded_diagnostic_008/attempt_002/neutral_proxy_original_speed.mp4)
- [试验3](/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_loaded_diagnostic_008/attempt_003/neutral_proxy_original_speed.mp4)
- [试验4](/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_loaded_diagnostic_008/attempt_004/neutral_proxy_original_speed.mp4)
- [试验5](/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_loaded_diagnostic_008/attempt_005/neutral_proxy_original_speed.mp4)
- [试验6](/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_loaded_diagnostic_008/attempt_006/neutral_proxy_original_speed.mp4)

原始证据：/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_loaded_diagnostic_008
源验证：机器人53项True，cart004 manifest 37项True；无全局索引改动。

边界和执行缺陷更正详 results.json；本报告不授载重、可靠性或0.5m性能资格。

## 终点语义勘误

上表车货距离、原summary全部final字段及results原root_final/cart_final/cargo_final字段均指**最后已记录PRE-state**，不是真实终止POST状态；原attempt摘要与轨迹未改。1–3真实终点取最后完成积分POST-state；4–6取与此前POST完全连续的terminal prestate，触发状态未再积分。

|试验|真实终止身体后退mm|真实终止车后退mm|真实终止货后退mm|终点类型|
|---|---:|---:|---:|---|
|1|-0.748|-15.929|-15.843|last_integrated_poststate|
|2|14.640|9.861|9.923|last_integrated_poststate|
|3|-12.235|39.325|39.383|last_integrated_poststate|
|4|-2.445|9.018|9.081|terminal_unintegrated_prestate|
|5|-2.457|9.750|9.771|terminal_unintegrated_prestate|
|6|2.959|12.844|12.847|terminal_unintegrated_prestate|

试验5最后已记录row倾角24.8661°，真实terminal25.0675°；试验6分别24.8985°/25.1441°。terminal的真实位置与倾角可用于终点，但其滑移累计另含未执行5ms保守预测，不能当作已发生位移。实际认证后退脚步仍为0。来源：独立final_results_review.json，SHA8a66deddb4543d21ec460bf1f5487053e171cb138c15647d801215779e7c811c。
