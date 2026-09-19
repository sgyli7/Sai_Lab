extends RefCounted
## The active actor's capabilities are the source of the visible instructions.
static func describe(kind: String, task: String, finished: bool, available: Dictionary, grabbing: bool = false) -> String:
	var lines: Array[String] = []
	if kind == "sai":
		lines.append("当前操控：Sai 001 · " + ("自动运输" if task.begins_with("cargo") else "轮腿 / 机械臂"))
		if finished:
			lines.append("任务已结束 · R / 0 重新开始 · 左上菜单切换任务")
		elif grabbing:
			lines.append("正在自动靠近 / 抓取入仓 · X 取消 / 松开 · R / 0 归位")
		elif task.begins_with("cargo"):
			lines.append("自动抓取 → 入仓 → 夹紧运输 · R / 0 重新开始")
		else:
			lines.append("W / S 前进 / 后退 · A / D 转向 · 按住 Shift 下蹲，松开恢复")
			if task in ["drive", "sort"]:
				lines.append("B 选择物件 · G 抓取入仓 · X 取消 / 松开 · R / 0 归位")
			else:
				lines.append("W 驶过台阶 · R / 0 归位")
	elif kind == "roller":
		lines.append("当前操控：MicroDuck · 轮滑模式")
		lines.append("W 滑行 · S 制动 · A / D 转向 · 空格 松开推进")
		lines.append("Y / 2 蹲起 · 0 归位")
	else:
		lines.append("当前操控：MicroDuck · 步行模式")
		lines.append("W / S 前进 / 后退 · A / D 转向 · Q / E 平移 · 空格 停止")
		var skills: Array[String] = []
		for entry in [["sitstand","Y / 2 坐起"],["ground_pick","G / 1 捡地"],
			["kick_left","K / 3 左踢"],["kick_right","L / 4 右踢"],["roulade","R / 5 前滚"],
			["stand_hold","7 站定"],["sprint","左 Shift + W 加速（A / D 转向，松开恢复步行）"]]:
			if available.get(entry[0], false): skills.append(entry[1])
		lines.append(" · ".join(skills))
		lines.append("B 选择踢击目标 · 0 归位")
	lines.append("右键 环视 · 滚轮 缩放 · Tab 观景 · F5 / F6 / F7 切换机器人 · Esc 退出")
	return "\n".join(lines)
