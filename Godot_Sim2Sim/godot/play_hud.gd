extends CanvasLayer
## On-screen D-pad + skill buttons. Hold bits live in extra_held; taps via signal.

signal tap(action: String)

var extra_held: Array = []
var time_scale: float = 1.0
var _status: Label
var _scale_label: Label
var _slider: HSlider
var _help: Label
var _held_bits: Dictionary = {}
var _tap_btns: Dictionary = {}
var _strafe_btns: Array = []


func _ready() -> void:
	layer = 100
	_build()


func set_status(text: String) -> void:
	if _status != null:
		_status.text = text


func configure_standalone(font: Font = null, sprint_available: bool = false) -> void:
	if font != null:
		var theme := Theme.new()
		theme.default_font = font
		(_status.get_parent() as Control).theme = theme
		for label in [_status,_help]:
			label.add_theme_color_override("font_shadow_color",Color(0.0,0.0,0.0,0.8))
			label.add_theme_constant_override("shadow_offset_x",1)
			label.add_theme_constant_override("shadow_offset_y",1)
	# The player uses a fixed simulation timestep. Replace the research pacing
	# slider with explicit standing access and pause, without changing physics.
	_slider.hide()
	_scale_label.hide()
	var box := _slider.get_parent()
	_tap_btn(box,"站立 7","stand")
	_tap_btn(box,"暂停 F8","pause")
	_tap_btns.stand.disabled = not _strafe_btns[0].visible
	_help.text += " · 7站立 F8暂停"
	if sprint_available: _help.text = _help.text.replace("W/↑ 前进","W/↑ 前进 · 左Shift+W 加速（可同时 A/D 转向）")
	# Two rows fit next to the direction controls without hiding the exit button.
	var old_skills: Control = _tap_btns.idle.get_parent()
	var skills := GridContainer.new()
	skills.columns = 5
	skills.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	skills.offset_left = -600
	skills.offset_right = -24
	skills.offset_top = -124
	skills.offset_bottom = -24
	skills.add_theme_constant_override("h_separation",8)
	skills.add_theme_constant_override("v_separation",8)
	old_skills.get_parent().add_child(skills)
	for button in old_skills.get_children(): button.reparent(skills)
	old_skills.queue_free()
	_tap_btns.idle.text = "松开 · 空格"
	_tap_btns.quit.text = "退出 Esc"


func _hold(bit: String, on: bool) -> void:
	if on:
		_held_bits[bit] = true
	else:
		_held_bits.erase(bit)
	extra_held = _held_bits.keys()


func _build() -> void:
	var root := Control.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(root)

	_status = Label.new()
	_status.position = Vector2(16, 12)
	_status.size = Vector2(1240, 48)
	_status.add_theme_font_size_override("font_size", 18)
	_status.text = "Godot play  ·  点窗口后按住 WASD / 方向键"
	root.add_child(_status)

	var help := Label.new()
	help.position = Vector2(16, 44)
	help.size = Vector2(1240, 40)
	help.add_theme_font_size_override("font_size", 14)
	help.modulate = Color(0.85, 0.9, 0.85)
	help.text = "W/↑ 前进  S/↓ 后退  A/← 左转  D/→ 右转  Q/E 平移  空格 Idle  ·  1捡地 2坐下 3/4踢球 5前滚 6轮滑 0重置 Esc退出"
	_help = help
	root.add_child(help)

	var scale_box := HBoxContainer.new()
	scale_box.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	scale_box.offset_left = -420
	scale_box.offset_top = 12
	scale_box.offset_right = -16
	scale_box.offset_bottom = 52
	scale_box.add_theme_constant_override("separation", 8)
	root.add_child(scale_box)
	_scale_label = Label.new()
	_scale_label.custom_minimum_size = Vector2(150, 36)
	_scale_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	_scale_label.add_theme_font_size_override("font_size", 16)
	_scale_label.text = "时间流速 1.00×"
	scale_box.add_child(_scale_label)
	_slider = HSlider.new()
	_slider.min_value = 0.25
	_slider.max_value = 3.0
	_slider.step = 0.05
	_slider.value = 1.0
	_slider.custom_minimum_size = Vector2(220, 36)
	_slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_slider.value_changed.connect(_on_time_scale)
	scale_box.add_child(_slider)

	var dpad := HBoxContainer.new()
	dpad.set_anchors_preset(Control.PRESET_BOTTOM_LEFT)
	dpad.offset_left = 24
	dpad.offset_bottom = -24
	dpad.offset_top = -120
	dpad.offset_right = 500
	dpad.add_theme_constant_override("separation", 8)
	root.add_child(dpad)
	_hold_btn(dpad, "←", "left")
	var mid := VBoxContainer.new()
	mid.add_theme_constant_override("separation", 8)
	dpad.add_child(mid)
	_hold_btn(mid, "↑", "fwd")
	_hold_btn(mid, "↓", "back")
	_hold_btn(dpad, "→", "right")
	_strafe_btns.append(_hold_btn(dpad, "Q 左移", "strafe_l"))
	_strafe_btns.append(_hold_btn(dpad, "E 右移", "strafe_r"))

	var skills := HBoxContainer.new()
	skills.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	skills.offset_right = -24
	skills.offset_bottom = -24
	skills.offset_top = -76
	skills.offset_left = -900
	skills.add_theme_constant_override("separation", 8)
	root.add_child(skills)
	_tap_btn(skills, "Idle", "idle")
	_tap_btn(skills, "重置 0", "reset")
	_tap_btn(skills, "捡地 1", "pick")
	_tap_btn(skills, "坐下 2", "sit")
	_tap_btn(skills, "左踢 3", "kick_left")
	_tap_btn(skills, "右踢 4", "kick_right")
	_tap_btn(skills, "前滚 5", "roulade")
	_tap_btn(skills, "轮滑 6", "switch_robot")
	_tap_btn(skills, "推一把 P", "push")
	_tap_btn(skills, "退出", "quit")


func set_mode(mode: String) -> void:
	var roller := mode == "roller"
	if _help != null:
		if roller:
			_help.text = "W/↑ 滑行  S/↓ 刹车  A/← 左转  D/→ 右转  空格 Idle  ·  2下蹲滑行  6走路  0重置 Esc退出"
		else:
			_help.text = "W/↑ 前进  S/↓ 后退  A/← 左转  D/→ 右转  Q/E 平移  空格 Idle  ·  1捡地 2坐下 3/4踢球 5前滚 6轮滑 0重置 Esc退出"
	for action in ["pick", "kick_left", "kick_right", "roulade"]:
		if _tap_btns.has(action):
			(_tap_btns[action] as CanvasItem).visible = not roller
	if _tap_btns.has("sit"):
		(_tap_btns["sit"] as Button).text = "下蹲 2" if roller else "坐下 2"
	for b in _strafe_btns:
		(b as CanvasItem).visible = not roller
	if _tap_btns.has("switch_robot"):
		(_tap_btns["switch_robot"] as Button).text = "走路 6" if roller else "轮滑 6"


func _on_time_scale(v: float) -> void:
	time_scale = clampf(v, 0.25, 3.0)
	if _scale_label != null:
		_scale_label.text = "时间流速 %.2f×" % time_scale


func _hold_btn(parent: Control, label: String, bit: String) -> Button:
	var b := Button.new()
	b.text = label
	b.custom_minimum_size = Vector2(72, 44)
	b.button_down.connect(func() -> void: _hold(bit, true))
	b.button_up.connect(func() -> void: _hold(bit, false))
	parent.add_child(b)
	return b


func _tap_btn(parent: Control, label: String, action: String) -> Button:
	var b := Button.new()
	b.text = label
	b.custom_minimum_size = Vector2(84, 44)
	b.pressed.connect(func() -> void: tap.emit(action))
	parent.add_child(b)
	_tap_btns[action] = b
	return b
