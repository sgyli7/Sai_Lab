extends CanvasLayer
const VisualProfile = preload("res://atelier/visual_profile.gd")
var workshop: Node3D
var help_panel: PanelContainer
var state_label: Label
var debug_panel: PanelContainer
var debug_label: Label
var roller := false
var prop_label: Label

func _ready() -> void:
	layer=110
	roller=workshop!=null and "roller" in workshop.server._robot_scene
	var root:=Control.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter=Control.MOUSE_FILTER_IGNORE
	if VisualProfile.value("MD_UI_FONT")=="1":
		root.theme=Theme.new()
		root.theme.default_font=load("res://atelier/ui_font.tres")
	add_child(root)
	var masthead:=PanelContainer.new()
	masthead.position=Vector2(24,20);masthead.size=Vector2(320,73)
	var compact:=VisualProfile.value("MD_COMPACT_HUD")=="1"
	if compact:masthead.size=Vector2(214,61)
	var mast_style:=StyleBoxFlat.new()
	mast_style.bg_color=Color(.86,.85,.79,.96)
	mast_style.border_color=Color("434047");mast_style.set_border_width_all(1)
	masthead.add_theme_stylebox_override("panel",mast_style);root.add_child(masthead)
	if compact:
		var accent:=ColorRect.new();accent.color=Color("e0bd38")
		accent.position=Vector2(24,20);accent.size=Vector2(3,61);root.add_child(accent)
	var title:=Label.new()
	title.position=Vector2(39,29);title.text="MICRODUCK"
	title.add_theme_color_override("font_color",Color("292830"))
	title.add_theme_font_size_override("font_size",21);root.add_child(title)
	if compact:title.position=Vector2(39,25);title.add_theme_font_size_override("font_size",20)
	var subtitle:=Label.new()
	subtitle.position=Vector2(40,61);subtitle.text="小小维修站    /    SERVICE ATELIER 01"
	subtitle.add_theme_color_override("font_color",Color("55545a"))
	subtitle.add_theme_font_size_override("font_size",12);root.add_child(subtitle)
	if compact:subtitle.position=Vector2(40,56);subtitle.text="小小维修站    /    01"
	var panel:=PanelContainer.new()
	panel.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_LEFT)
	panel.offset_left=32;panel.offset_top=-70;panel.offset_bottom=-26;panel.offset_right=590
	var style:=StyleBoxFlat.new()
	style.bg_color=Color(.86,.85,.79,.93);style.border_color=Color("434047")
	style.set_border_width_all(1);style.content_margin_left=16;style.content_margin_right=16
	style.content_margin_top=10;style.content_margin_bottom=10
	panel.add_theme_stylebox_override("panel",style);root.add_child(panel)
	var hint:=Label.new()
	hint.text="WASD 移动    右键 环视    滚轮 缩放    Tab 观景    F1 帮助"
	if roller:hint.text="WASD 滑行 / 转向    右键 环视    滚轮 缩放    Tab 观景    F1 帮助"
	hint.add_theme_font_size_override("font_size",14)
	hint.add_theme_color_override("font_color",Color("36343a"));panel.add_child(hint)
	help_panel=PanelContainer.new();help_panel.position=Vector2(32,100)
	help_panel.add_theme_stylebox_override("panel",style);root.add_child(help_panel)
	var help:=Label.new()
	help.text="操作指南\n\nW / S  前进 / 后退    A / D  转向\nQ / E  平移    空格  停止\n\n1  捡地    2  坐下 / 站起\n3 / 4  左踢 / 右踢    5  前滚\n6  切换轮滑    0  重置\n\nTab  观景 / 游玩    Esc  退出\nF2  运行信息"
	if roller:help.text="轮滑操作\n\nW  滑行    S  刹车\nA / D  转向    空格  松开推进\n\n2 / Y  下蹲滑行后起身\n6  切回步行    0  重置\n\n右键  环视    滚轮  缩放\nTab  观景 / 游玩    Esc  退出\nF2  运行信息"
	if workshop.loose_props!=null:
		help.text+="\n\n动态小物件：维修坪南侧\n"
		if not roller:help.text+="B  切换踢击目标，再按 3 / 4 踢击\n"
		help.text+="0  同时归位机器人与小物件"
	help.add_theme_color_override("font_color",Color("36343a"))
	help.add_theme_font_size_override("font_size",16);help_panel.add_child(help)
	help_panel.visible=false
	var state_panel:=PanelContainer.new()
	state_panel.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_RIGHT)
	state_panel.offset_left=-230;state_panel.offset_right=-32
	state_panel.offset_top=-70;state_panel.offset_bottom=-26
	state_panel.add_theme_stylebox_override("panel",style);root.add_child(state_panel)
	state_label=Label.new();state_label.text="●  待机"
	state_label.add_theme_color_override("font_color",Color("36343a"))
	state_label.add_theme_font_size_override("font_size",14);state_panel.add_child(state_label)
	debug_panel=PanelContainer.new();debug_panel.position=Vector2(900,100)
	debug_panel.add_theme_stylebox_override("panel",style);root.add_child(debug_panel)
	debug_label=Label.new();debug_label.add_theme_font_size_override("font_size",14)
	debug_label.custom_minimum_size.x=290;debug_label.autowrap_mode=TextServer.AUTOWRAP_WORD_SMART
	debug_label.add_theme_color_override("font_color",Color("36343a"));debug_panel.add_child(debug_label)
	debug_panel.visible=false
	if workshop.loose_props!=null:
		var prop_panel:=PanelContainer.new()
		prop_panel.set_anchors_and_offsets_preset(Control.PRESET_TOP_RIGHT)
		prop_panel.offset_left=-335;prop_panel.offset_right=-24;prop_panel.offset_top=20;prop_panel.offset_bottom=73
		prop_panel.add_theme_stylebox_override("panel",style);root.add_child(prop_panel)
		prop_label=Label.new();prop_label.add_theme_font_size_override("font_size",14)
		prop_label.add_theme_color_override("font_color",Color("36343a"));prop_panel.add_child(prop_label)
	for control in root.find_children("*","Control",true,false):
		control.mouse_filter=Control.MOUSE_FILTER_IGNORE

func _process(_delta: float) -> void:
	if workshop==null or state_label==null:return
	if prop_label!=null:
		prop_label.text="小物件可推动 · 0 归位" if roller else "踢击目标："+workshop.loose_props.target_label()+"\nB 切换目标    3 / 4 踢击    0 归位"
	var server: Node=workshop.server
	if server._peer==null:
		state_label.text="○  等待控制器"
		return
	var original: String=server._hud._status.text if server._hud!=null else "standing"
	original=original.to_lower()
	var translated:="就绪"
	for key in {"standing":"待机","walk":"行走","sit":"坐下 / 站起","pick":"捡地","kick":"踢击","roulade":"前滚","roller":"轮滑"}:
		if key in original:
			translated={"standing":"待机","walk":"行走","sit":"坐下 / 站起","pick":"捡地","kick":"踢击","roulade":"前滚","roller":"轮滑"}[key]
			break
	state_label.text="●  "+translated
	if roller and translated=="行走":state_label.text="●  滑行"
	# A stopped command can coexist with a toppled physical robot. Keep that
	# distinction visible without interrupting deliberate acrobatic skills.
	if translated in ["待机","行走"] and server._base!=null and server._base.global_basis.y.dot(Vector3.UP)<.35:
		state_label.text="○  姿态失稳 · 0 重置"
	if debug_panel.visible:
		var recent: Array=server.draw_times.slice(maxi(0,server.draw_times.size()-30))
		var total:=0.0
		for value in recent:total+=value
		var fps:=1000.0*recent.size()/maxf(.01,total)
		debug_label.text="运行信息\n\n实际渲染  %.1f FPS\n渲染上限  %.0f FPS\n物理步长  5 ms\n策略  %s" % [fps,1000000.0/server.draw_period_usec,original]

func toggle_help() -> void:
	help_panel.visible=not help_panel.visible

func toggle_debug() -> void:
	debug_panel.visible=not debug_panel.visible
