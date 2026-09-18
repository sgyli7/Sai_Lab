extends Node3D
## A persistent world and window. Only the robot and its native physics space change.
const StationLayout = preload("res://science_station/layout.gd")
const TERRAIN_LAYER := 8
signal scene_chosen
var checkpoint := "service"
var checkpoint_visits: Array = []
var route_result: Dictionary = {}
var probe_spawn: Variant = null

const TASKS := {
	"drive":{"label":"自由驾驶","task":"drive","origin":[0.,0.,0.]},
	"sort":{"label":"场景物件 · 辅助抓取入仓","task":"drive","origin":[-.48,0.,1.28]},
	"cargo18":{"label":"取件入仓 · 18 mm 运送","task":"cargo","obstacle":.018,"origin":[-.45,0.,.85]},
	"cargo25":{"label":"重载工位 · 25 mm 运送","task":"cargo","obstacle":.025,"origin":[-.60,0.,2.10]},
	"up20":{"label":"20 mm 检修台 · 上阶","task":"drive","riser":.02,"origin":[-1.4,0.,3.15]},
	"down20":{"label":"20 mm 检修台 · 下阶","task":"drive","riser":.02,"descending":true,"origin":[-1.4,0.,3.15]},
	"up40":{"label":"40 mm 检修台 · 上阶","task":"drive","riser":.04,"origin":[-1.4,0.,4.50]},
	"down40":{"label":"40 mm 检修台 · 下阶","task":"drive","riser":.04,"descending":true,"origin":[-1.4,0.,4.50]},
	"up60":{"label":"60 mm 实验台 · 上阶","task":"drive","riser":.06,"skill":"ascent60","origin":[2.7,0.,2.8]},
	"down60":{"label":"60 mm 实验台 · 下阶","task":"drive","riser":.06,"skill":"descent60","descending":true,"origin":[2.7,0.,2.8]}
}
var options: Dictionary
var profiles: Dictionary
var atelier: Node3D
var actor: Node3D
var active_robot := ""
var active_task := "drive"
var switching := false
var _headless := false
var _robot_scene := ""
var _bodies := {}
var _base: RigidBody3D
var _hud = null
var _peer = null
var _t := 0.0
var _cam_yaw := .65
var _cam_pitch := .32
var _cam_dist := 1.10
var _cam_snap := false
var _cam_look := Vector3.ZERO
var drag := false
var label: Label
var task_menu: OptionButton
var controls_hint: Label
var grab_row: HBoxContainer
var robot_buttons: Dictionary = {}
var course_root: Node3D
var events: Array = []
var plan_index := 0
var elapsed := 0.0
var task_result := {}
var capture_start := 0
var next_capture := 0
var frame_index := 0
var capture_primed := false
var frames: Array = []
var capture_job := -1
var dropped := 0
var recording_shot := "interactive"
var stopping := false
var world_id: int
var samples: Array = []
var grab_sessions: Array = []
var next_sample := 0.0

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	_headless = DisplayServer.get_name() == "headless"
	options = {"scene":"workshop","drive_speed":.5,"fast_check":false,"choose_scene":false,
		"robot":"microduck","task":"drive","output":"user://run","record":false,
		"sai_controller":"native","sai_physics_hz":1000,"plan":{}}
	if FileAccess.file_exists("res://hub/options.json"):
		var saved=JSON.parse_string(FileAccess.get_file_as_string("res://hub/options.json"))
		if saved is Dictionary:options.merge(saved,true)
	for arg in OS.get_cmdline_user_args():
		if arg=="--choose-scene":options.choose_scene=true
		elif arg=="--fast-check":options.fast_check=true
		elif arg=="--record":options.record=true
		elif arg.begins_with("--scene="):options.scene=arg.trim_prefix("--scene=")
		elif arg.begins_with("--robot="):options.robot=arg.trim_prefix("--robot=")
		elif arg.begins_with("--task="):options.task=arg.trim_prefix("--task=")
		elif arg.begins_with("--output="):options.output=arg.trim_prefix("--output=")
		elif arg.begins_with("--sai-controller="):options.sai_controller=arg.trim_prefix("--sai-controller=")
		elif arg.begins_with("--sai-physics-hz="):options.sai_physics_hz=int(arg.trim_prefix("--sai-physics-hz="))
		elif arg.begins_with("--drive-speed="):options.drive_speed=float(arg.trim_prefix("--drive-speed="))
		elif arg.begins_with("--plan="):
			var plan_path:=arg.trim_prefix("--plan=")
			var loaded_plan=JSON.parse_string(FileAccess.get_file_as_string(plan_path)) if FileAccess.file_exists(plan_path) else null
			if loaded_plan is Dictionary:
				options.plan=loaded_plan
			else:
				push_error("Cannot load native test plan: "+plan_path)
				get_tree().quit(2)
				return
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(str(options.output)))
	profiles = JSON.parse_string(FileAccess.get_file_as_string("res://hub/physics_profiles.json"))
	get_tree().auto_accept_quit = false
	get_window().close_requested.connect(_finish)
	get_window().title = "Robot Sim2Sim · 小小维修站"
	get_window().size = Vector2i(1920,1080)
	get_window().content_scale_size = Vector2i(1280,720)
	get_window().content_scale_mode = Window.CONTENT_SCALE_MODE_CANVAS_ITEMS
	get_viewport().msaa_3d = Viewport.MSAA_4X
	if options.get("choose_scene",false):
		_make_scene_picker()
		await scene_chosen
	get_window().title = "Robot Sim2Sim · "+scene_title()
	atelier = load("res://science_station/station.gd" if is_science_station() else "res://atelier/workshop.gd").new()
	add_child(atelier)
	atelier.build(self)
	if atelier.hud != null: atelier.hud.queue_free(); atelier.hud = null
	# One common set of scenery contacts supports both published collision masks.
	_set_scenery_masks(self)
	get_node("World/Floor").collision_layer |= TERRAIN_LAYER
	if not _headless: _make_hud()
	world_id = atelier.get_instance_id()
	active_task = options.task
	if is_science_station():
		checkpoint=options.plan.get("initial_checkpoint","service")
		checkpoint_visits.append({"id":checkpoint,"time":0.})
		if options.plan.has("initial_position"):
			var p:Array=options.plan.initial_position;probe_spawn=Vector3(p[0],atelier.ground_height(p[0],p[1]),p[1])
	if options.plan.has("route"):
		var probe:Node=load("res://hub/route_probe.gd").new();probe.hub=self;probe.points=options.plan.route;add_child(probe)
	select_robot(options.robot)

func _set_scenery_masks(node: Node) -> void:
	if node is StaticBody3D:
		node.collision_layer = 3
		if node.is_in_group("sai_driving_surface"): node.collision_layer |= TERRAIN_LAYER
		node.collision_mask = 5
	elif node is RigidBody3D:
		node.collision_layer = 5
		node.collision_mask = 3
	for child in node.get_children(): _set_scenery_masks(child)

func is_science_station() -> bool:
	return options.get("scene","workshop") == "science_station"

func scene_title() -> String:
	return StationLayout.TITLE if is_science_station() else "小小维修站"

func spawn_point() -> Vector3:
	if probe_spawn != null:return probe_spawn
	return StationLayout.CHECKPOINTS[checkpoint].position if is_science_station() else Vector3.ZERO

func task_settings() -> Dictionary:
	var spec: Dictionary = TASKS[active_task].duplicate(true)
	var stair_profile:=str(options.get("sai_stair_profile",""))
	if active_robot=="sai" and active_task.begins_with("up") and not stair_profile.is_empty():
		spec["skill"]=stair_profile
		spec["payload"]=true
	if is_science_station():
		var p := spawn_point()
		spec = {"label":"自由探索", "task":"drive", "origin":[p.x,p.y,p.z]}
	return spec

func _make_scene_picker() -> void:
	var layer := CanvasLayer.new();add_child(layer)
	var background := ColorRect.new();background.color=Color("e9e5cf")
	background.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT);layer.add_child(background)
	var center := CenterContainer.new();center.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT);layer.add_child(center)
	var column := VBoxContainer.new();column.add_theme_constant_override("separation",20);center.add_child(column)
	var emblem:=TextureRect.new();emblem.texture=load("res://science_station/icon.svg")
	emblem.custom_minimum_size=Vector2(80,80);emblem.expand_mode=TextureRect.EXPAND_IGNORE_SIZE
	emblem.stretch_mode=TextureRect.STRETCH_KEEP_ASPECT_CENTERED;column.add_child(emblem)
	var heading := Label.new();heading.text="选择探索地点";heading.horizontal_alignment=HORIZONTAL_ALIGNMENT_CENTER
	heading.add_theme_font_override("font",load("res://atelier/ui_font.tres"));heading.add_theme_font_size_override("font_size",34)
	heading.add_theme_color_override("font_color",Color("343944"));column.add_child(heading)
	for entry in [["science_station","风口科学站","观测塔 · 样本处理 · 岩丘步道"],["workshop","小小维修站","机械小院 · 检修台 · 运送练习"]]:
		var button := Button.new();button.text=entry[1]+"\n"+entry[2];button.custom_minimum_size=Vector2(550,110)
		button.add_theme_font_override("font",load("res://atelier/ui_font.tres"));button.add_theme_font_size_override("font_size",23)
		for state in ["normal","hover","pressed","focus"]:
			var style:=StyleBoxFlat.new();style.bg_color=Color("f6f1dc") if state=="normal" else Color("c2d4d6")
			style.border_color=Color("4d5663");style.set_border_width_all(1 if state=="normal" else 2)
			style.set_corner_radius_all(8);button.add_theme_stylebox_override(state,style)
		for state in ["font_color","font_hover_color","font_pressed_color","font_focus_color"]:
			button.add_theme_color_override(state,Color("343944"))
		button.pressed.connect(func(): options.scene=entry[0];active_task="drive";options.task="drive";layer.queue_free();scene_chosen.emit())
		column.add_child(button)
		if entry[0]=="science_station":button.grab_focus()
	var note:=Label.new();note.text="MicroDuck  ·  Roller  ·  Sai 001";note.horizontal_alignment=HORIZONTAL_ALIGNMENT_CENTER
	note.add_theme_color_override("font_color",Color("73777d"));column.add_child(note)


func select_robot(kind: String) -> void:
	if switching or kind not in ["microduck","roller","sai"]: return
	if is_science_station() and _base!=null and probe_spawn==null:
		var nearest:float=INF
		for visit in checkpoint_visits:
			var distance:float=_base.global_position.distance_squared_to(StationLayout.CHECKPOINTS[visit.id].position)
			if distance<nearest:nearest=distance;checkpoint=visit.id
	switching = true
	_change_robot.call_deferred(kind)

func _change_robot(kind: String) -> void:
	var before := {"pid":OS.get_process_id(),"hub":get_instance_id(),"atelier":atelier.get_instance_id(),
		"space":str(get_world_3d().space),"from":active_robot,"to":kind,"time":elapsed,
		"props_before":_prop_snapshot()}
	atelier.loose_props.set_frozen(true)
	get_tree().paused = true
	_base = null
	if actor != null:
		if active_robot == "sai" and actor.grab != null: actor.grab.perform("cancel")
		_save_native_trace()
		actor.set_process(false)
		actor.set_physics_process(false)
		if active_robot == "sai":
			actor.finished = true
			if not actor.native_mode and actor.peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
				actor.peer.put_data((JSON.stringify({"finish":true})+"\n").to_utf8_buffer())
		actor.queue_free()
		await get_tree().process_frame
		actor = null
	if get_tree().has_meta("microduck_session"): get_tree().remove_meta("microduck_session")
	var profile: Dictionary = profiles.sai if kind == "sai" else profiles.microduck
	for key in profile:
		if not str(key).begins_with("physics/3d/") and not str(key).begins_with("physics/jolt_physics_3d/"): continue
		var value = profile[key]
		if key == "physics/3d/default_gravity_vector": value = Vector3(value[0],value[1],value[2])
		ProjectSettings.set_setting(key,value)
	# Jolt reads settings_changed, then copies solver settings into each new space.
	ProjectSettings.settings_changed.emit()
	var previous_world: World3D = get_world_3d()
	get_viewport().world_3d = World3D.new()
	var physics_hz := int(options.get("sai_physics_hz",1000)) if kind == "sai" else 200
	if kind == "sai" and (physics_hz < 50 or physics_hz % 50 != 0):
		push_error("Sai physics rate must be an integer multiple of the 50 Hz controller")
		get_tree().quit(2)
		return
	Engine.physics_ticks_per_second = physics_hz
	Engine.max_physics_steps_per_frame = maxi(8,int(ceil(float(physics_hz)/20.0))) if kind == "sai" else 32
	get_node("World/Floor").physics_material_override.friction = .8 if kind == "sai" else 1.
	active_robot = kind
	task_result.clear()
	_build_task_course()
	if kind == "sai":
		actor = load("res://hub/sai.gd").new()
		actor.hub = self
		var origin: Array = task_settings().origin
		actor.position = Vector3(origin[0],origin[1],origin[2])
		add_child(actor)
		_base = actor.robot.bodies.chassis
	else:
		get_tree().set_meta("microduck_session",{"mode":"roller" if kind == "roller" else "walk", "steps":0,
			"rows":[],"replay":{},"trace_path":options.output+"/native-%d.json" % events.size() if not options.plan.is_empty() else "","seconds":0.,"segment":-1,"resets":0,"switches":0,
			"first_fall":null,"started_usec":Time.get_ticks_usec(),"seed":915000,"error":"","events":[],"profiles":{}})
		actor = load("res://standalone/main.tscn").instantiate()
		actor.set_script(load("res://hub/microduck.gd"))
		actor.hub = self
		# The native controller retains its expected node paths, but the persistent
		# hub supplies the one visible environment and one colliding floor.
		actor.get_node("World/Floor").collision_layer = 0
		actor.get_node("World/Floor").collision_mask = 0
		actor.get_node("World/Floor/FloorMesh").hide()
		actor.get_node("WorldEnvironment").environment = null
		actor.get_node("World/Sun").hide()
		actor.get_node("World/FillLight").hide()
		add_child(actor)
		_base = actor._base
		_robot_scene = actor._robot_scene
		_bodies = actor._bodies
		if not _headless:
			var name_key := "microduck_roller" if kind == "roller" else "microduck"
			atelier.mesh_roles = JSON.parse_string(FileAccess.get_file_as_string("res://atelier/visual_mesh_roles.json"))[name_key]
			atelier.normal_meshes = JSON.parse_string(FileAccess.get_file_as_string("res://atelier/robot_normal_map.json"))[name_key]
			atelier._paint_robot(actor.get_node("RobotHost"))
	get_node("World/Camera3D").make_current()
	_cam_dist = 1.65 if kind == "sai" else 1.10
	if is_science_station(): _cam_yaw=-.30;_cam_pitch=.10
	atelier.follow_initialized = false
	Engine.max_fps = 0 if options.get("fast_check",false) else 30
	Engine.max_physics_steps_per_frame = maxi(8,int(ceil(float(physics_hz)/20.0))) if kind == "sai" else 32
	get_window().title = "Robot Sim2Sim · "+scene_title()+" / "+kind
	get_tree().paused = false
	atelier.loose_props.set_frozen(false)
	switching = false
	before["new_space"] = str(get_world_3d().space)
	before["actor"] = actor.get_instance_id()
	before["bodies"] = actor.robot.bodies.size() if kind == "sai" else actor._bodies.size()
	before["physics_hz"] = Engine.physics_ticks_per_second
	before["settings"] = {"speculative":ProjectSettings.get_setting("physics/jolt_physics_3d/simulation/speculative_contact_distance"),
		"velocity_steps":ProjectSettings.get_setting("physics/jolt_physics_3d/simulation/velocity_steps")}
	before["props_after"] = _prop_snapshot()
	before["spawn"] = [_base.global_position.x,_base.global_position.y,_base.global_position.z]
	before["checkpoint"] = checkpoint
	before["world_preserved"] = world_id == atelier.get_instance_id()
	events.append(before)
	print("HUB_ROBOT_READY ",JSON.stringify(before))
	# Keep the previous scenario alive until viewport and Node3D migration finish.
	await get_tree().process_frame
	previous_world = null
	if capture_start == 0:
		capture_start = Time.get_ticks_usec()
		next_capture = capture_start

func _paint_role_for_node(mesh: MeshInstance3D) -> String:
	return actor._paint_role_for_node(mesh) if actor != null and active_robot != "sai" else "shell"

func _mesh_id(mesh: MeshInstance3D) -> int:
	return actor._mesh_id(mesh) if actor != null and active_robot != "sai" else 0

func _orbit_offset(yaw: float,pitch: float,distance: float) -> Vector3:
	return Vector3(sin(yaw)*cos(pitch),sin(pitch),cos(yaw)*cos(pitch))*distance

func _task_box(parent: Node3D,name_text: String,p: Vector3,size: Vector3,color: String) -> void:
	var body := StaticBody3D.new()
	body.name = name_text
	body.position = p
	body.collision_layer = 3 | TERRAIN_LAYER
	body.collision_mask = 5
	body.physics_material_override = PhysicsMaterial.new()
	body.physics_material_override.friction = .8
	var collision := CollisionShape3D.new()
	var shape := BoxShape3D.new()
	shape.size = size
	shape.margin = .0002
	collision.shape = shape
	body.add_child(collision)
	parent.add_child(body)
	if not _headless:
		# The hub floor already renders y=0. Keep the full contact box, but
		# draw only its exposed part so the entry decks cannot fight it.
		var bottom := maxf(0.0, p.y-size.y*.5)
		var top := p.y+size.y*.5
		if top <= bottom+.000001: return
		var mesh := MeshInstance3D.new()
		var box := BoxMesh.new()
		box.size = Vector3(size.x,top-bottom,size.z)
		mesh.position.y = (top+bottom)*.5-p.y
		var surface := SurfaceTool.new()
		surface.begin(Mesh.PRIMITIVE_TRIANGLES)
		var faces := box.get_faces()
		for i in range(0,faces.size(),3):
			var a:=faces[i];var b:=faces[i+1];var c:=faces[i+2]
			# Open underside: its inverted-hull ink pass must not draw on
			# the floor either. Elevated boxes retain their visible bottom.
			if bottom==0.0 and maxf(a.y,maxf(b.y,c.y)) < -box.size.y*.5+.000001: continue
			var normal:=-(b-a).cross(c-a).normalized()
			for v in [a,b,c]:
				surface.set_normal(normal);surface.add_vertex(v)
		surface.index()
		mesh.mesh = surface.commit()
		mesh.material_override = atelier.materials[color]
		body.add_child(mesh)

func _build_task_course() -> void:
	if course_root != null:
		remove_child(course_root)
		course_root.free()
	course_root = Node3D.new()
	course_root.name = "WorkshopTasks"
	add_child(course_root)
	if is_science_station(): return
	for key in ["cargo18","cargo25","up20","up40","up60"]:
		var spec: Dictionary = TASKS[key]
		if active_task == key.replace("up","down"): spec = TASKS[active_task]
		var zone := Node3D.new()
		zone.name = key
		zone.position = Vector3(spec.origin[0],spec.origin[1],spec.origin[2])
		course_root.add_child(zone)
		if spec.task == "cargo":
			for i in range(3):
				_task_box(zone,"course_"+str(i),Vector3(.55+.35*i,spec.obstacle-.04,0),Vector3(.09,.08,.9),"purple")
		else:
			var bounds := [-1.,.45,.63,.81,.99,3.]
			for i in range(5):
				var height: float = spec.riser * (4-i if spec.get("descending",false) else i)
				_task_box(zone,"stair_"+str(i),Vector3((bounds[i]+bounds[i+1])*.5,height-.5,0),Vector3(bounds[i+1]-bounds[i],1.,1.),"paper" if i%2==0 else "floor")
		if not _headless:
			var sign_label := Label3D.new()
			sign_label.text = spec.label
			sign_label.font = load("res://atelier/ui_font.tres")
			sign_label.font_size = 108
			sign_label.pixel_size = .0008/3.
			sign_label.modulate = Color("36343a")
			sign_label.outline_size = 0
			sign_label.position = Vector3(.50,.003,.58)
			sign_label.rotation_degrees.x = -90
			zone.add_child(sign_label)

func _make_hud() -> void:
	var canvas := CanvasLayer.new()
	add_child(canvas)
	canvas.visible = not options.plan.get("cinematic",false)
	var root := Control.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	root.theme = Theme.new()
	root.theme.set_color("font_color","Button",Color("36343a"))
	root.theme.set_color("font_color","OptionButton",Color("36343a"))
	for cls in ["Button","OptionButton"]:
		for state in ["normal","hover","pressed","focus"]:
			var style := StyleBoxFlat.new()
			style.bg_color = Color("e7d65f") if state in ["hover","pressed"] else Color("ece8df")
			style.border_color = Color("56535b")
			style.set_border_width_all(1)
			style.set_corner_radius_all(4)
			style.content_margin_left=10;style.content_margin_right=10
			style.content_margin_top=6;style.content_margin_bottom=6
			root.theme.set_stylebox(state,cls,style)
	root.theme.default_font = load("res://atelier/ui_font.tres")
	canvas.add_child(root)
	var panel := VBoxContainer.new()
	panel.position = Vector2(24,20)
	root.add_child(panel)
	label = Label.new()
	label.add_theme_color_override("font_color",Color("292830"))
	label.add_theme_font_size_override("font_size",18)
	label.add_theme_color_override("font_outline_color",Color("ece8df"))
	label.add_theme_constant_override("outline_size",5)
	panel.add_child(label)
	var row := HBoxContainer.new()
	panel.add_child(row)
	for kind in ["microduck","roller","sai"]:
		var button := Button.new()
		button.text = {"microduck":"F5 · MicroDuck","roller":"F6 · MD 轮滑","sai":"F7 · Sai 001"}[kind]
		button.focus_mode = Control.FOCUS_NONE
		button.toggle_mode = true
		robot_buttons[kind] = button
		button.pressed.connect(select_robot.bind(kind))
		row.add_child(button)
	task_menu = OptionButton.new()
	task_menu.focus_mode = Control.FOCUS_NONE
	for key in TASKS: task_menu.add_item(TASKS[key].label)
	task_menu.item_selected.connect(func(index):
		active_task = TASKS.keys()[index]
		select_robot("sai"))
	panel.add_child(task_menu)
	grab_row = HBoxContainer.new()
	panel.add_child(grab_row)
	for entry in [["B · 选择物件", "cycle"], ["G · 抓取入仓", "pick"], ["X · 取消 / 松开", "cancel"]]:
		var button := Button.new()
		button.text = entry[0]
		button.focus_mode = Control.FOCUS_NONE
		button.pressed.connect(_grab_action.bind(entry[1]))
		grab_row.add_child(button)
	var hint_panel := PanelContainer.new()
	hint_panel.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_LEFT)
	hint_panel.offset_left = 24
	hint_panel.offset_top = -176
	hint_panel.offset_right = 850
	hint_panel.offset_bottom = -20
	var hint_style := StyleBoxFlat.new()
	hint_style.bg_color = Color(.925,.910,.870,.95)
	hint_style.set_content_margin_all(12)
	hint_style.set_corner_radius_all(5)
	hint_panel.add_theme_stylebox_override("panel",hint_style)
	root.add_child(hint_panel)
	controls_hint = Label.new()
	var hint := controls_hint
	hint.add_theme_color_override("font_color",Color("36343a"))
	hint.add_theme_font_size_override("font_size",13)
	hint_panel.add_child(hint)
	_refresh_controls()

func _refresh_controls() -> void:
	if controls_hint == null: return
	var available: Dictionary = actor.brain.available if actor != null and active_robot != "sai" else {}
	var finished: bool = actor != null and active_robot == "sai" and actor.finished
	var grabbing: bool = actor != null and active_robot == "sai" and actor.grab != null and actor.grab.busy
	controls_hint.text = load("res://hub/controls.gd").describe(active_robot,active_task,finished,available,grabbing)
	if actor != null and active_robot == "sai" and actor.grab != null:
		controls_hint.text += "\n" + actor.grab.status_text()
	elif actor != null and active_robot == "microduck":
		controls_hint.text += "\n踢击目标：" + atelier.loose_props.target_label()
	grab_row.visible = active_robot == "sai" and active_task in ["drive", "sort"] and not finished
	for i in range(grab_row.get_child_count()):
		grab_row.get_child(i).disabled = grabbing if i < 2 else not grabbing
	for kind in robot_buttons:
		robot_buttons[kind].set_pressed_no_signal(kind == active_robot)

func _grab_action(action: String) -> void:
	if switching or stopping or active_robot != "sai" or actor == null or actor.grab == null or actor.finished: return
	actor.grab.perform(action)

func _input(event: InputEvent) -> void:
	if atelier==null:
		if event is InputEventKey and event.pressed and event.physical_keycode==KEY_ESCAPE:get_tree().quit()
		return
	if event is InputEventKey and event.pressed and not event.echo:
		if event.physical_keycode==KEY_TAB:
			if is_science_station():atelier.cycle_view()
			else:atelier.set_view("tour" if atelier.view=="follow" else "follow")
			get_viewport().set_input_as_handled()
		elif event.physical_keycode in [KEY_F5,KEY_F6,KEY_F7]:
			select_robot({KEY_F5:"microduck",KEY_F6:"roller",KEY_F7:"sai"}[event.physical_keycode])
			get_viewport().set_input_as_handled()
		elif event.physical_keycode == KEY_0 or (event.physical_keycode == KEY_R and active_robot == "sai"):
			atelier.loose_props.reset()
			select_robot(active_robot)
			get_viewport().set_input_as_handled()
		elif event.physical_keycode == KEY_ESCAPE:
			_finish()
			get_viewport().set_input_as_handled()
		elif active_robot == "sai" and event.physical_keycode in [KEY_B,KEY_G,KEY_X]:
			_grab_action({KEY_B:"cycle",KEY_G:"pick",KEY_X:"cancel"}[event.physical_keycode])
			get_viewport().set_input_as_handled()
		elif active_robot == "roller" and event.physical_keycode == KEY_B:
			get_viewport().set_input_as_handled()
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_RIGHT: drag = event.pressed
		if event.pressed and event.button_index in [MOUSE_BUTTON_WHEEL_UP,MOUSE_BUTTON_WHEEL_DOWN]:
			_cam_dist = clampf(_cam_dist*(.90 if event.button_index==MOUSE_BUTTON_WHEEL_UP else 1.10),.35,4.)
	if event is InputEventMouseMotion and drag:
		_cam_yaw -= event.relative.x*.006
		_cam_pitch = clampf(_cam_pitch+event.relative.y*.004,.05,1.2)

func _physics_process(delta: float) -> void:
	if switching or actor == null or stopping: return
	if is_science_station():
		for key in StationLayout.CHECKPOINTS:
			if key!=checkpoint and _base.global_position.distance_to(StationLayout.CHECKPOINTS[key].position)<.85:
				checkpoint=key;probe_spawn=null;checkpoint_visits.append({"id":key,"time":elapsed})
	elapsed += delta
	_t = actor.robot.sim_time_seconds() if active_robot == "sai" else actor._t
	if elapsed >= next_sample:
		next_sample = elapsed+.1
		var p := _base.global_position
		var v := _base.linear_velocity
		samples.append({"time":elapsed,"robot_time":_t,"robot":active_robot,"task":active_task,
			"position":[p.x,p.y,p.z],"velocity":[v.x,v.y,v.z],"upright":_base.global_basis.y.y,
			"wall_usec":Time.get_ticks_usec(),"fps":Engine.get_frames_per_second(),
			"draw_calls":Performance.get_monitor(Performance.RENDER_TOTAL_DRAW_CALLS_IN_FRAME),
			"controller":actor.command.get("stage","") if active_robot=="sai" else actor.brain.policy,
			"held":actor.movement_command() if active_robot=="sai" else actor._held_now.duplicate()})
		if is_science_station() and not options.plan.is_empty():
			var motion:Array=[]
			for item in atelier.loose_props.items:
				var b:RigidBody3D=item.body;var q:Vector3=b.global_position;var u:Vector3=b.linear_velocity
				motion.append({"name":str(b.name),"position":[q.x,q.y,q.z],"velocity":[u.x,u.y,u.z]})
			samples[-1]["prop_motion"]=motion
	var plan: Dictionary = options.plan
	if plan.get("finish_on_grab",false) and active_robot == "sai" and actor.grab != null and not actor.grab.deliveries.is_empty():
		if _t-actor.grab.deliveries[-1].time > 3.: _finish.call_deferred()
	var actions: Array = plan.get("events",[])
	while plan_index < actions.size() and elapsed >= float(actions[plan_index].at):
		var action: Dictionary = actions[plan_index]
		plan_index += 1
		if action.has("robot"): select_robot(action.robot)
		if action.has("task"): active_task=action.task; select_robot("sai")
		if action.has("key"):
			var event := InputEventKey.new()
			event.physical_keycode = OS.find_keycode_from_string(action.key)
			event.keycode = event.physical_keycode
			event.location = int(action.get("location",0))
			event.pressed = action.get("pressed",true)
			Input.parse_input_event(event)
		if action.has("mouse_button"):
			var event:=InputEventMouseButton.new()
			event.button_index={"right":MOUSE_BUTTON_RIGHT,"up":MOUSE_BUTTON_WHEEL_UP,"down":MOUSE_BUTTON_WHEEL_DOWN}[action.mouse_button]
			event.pressed=action.get("pressed",true);Input.parse_input_event(event)
		if action.has("motion"):
			var event:=InputEventMouseMotion.new();event.relative=Vector2(action.motion[0],action.motion[1]);Input.parse_input_event(event)
	if plan.get("seconds",0.) > 0 and elapsed >= float(plan.seconds): _finish.call_deferred()

func _process(delta: float) -> void:
	if switching or stopping or actor == null: return
	if not _headless:
		atelier.update_camera(delta)
		_update_recording_camera()
		atelier.update_printed_labels()
		label.text = "%s / %s\n%s" % [scene_title(),active_robot.to_upper(),_stage_label(str(actor.command.get("stage","就绪"))) if active_robot=="sai" else actor.SKILL_LABELS.get("sprint" if actor.brain.sprinting else actor.brain.policy,actor.brain.policy)]
		if active_robot == "microduck": label.text += " · 实测 %.2f m/s" % actor.measured_speed_mps
		if is_science_station():label.text+=" · "+StationLayout.CHECKPOINTS[checkpoint].title
		_refresh_controls()
		task_menu.visible = active_robot == "sai" and not is_science_station()
		task_menu.select(TASKS.keys().find(active_task))
		if options.record: _capture()

func _update_recording_camera() -> bool:
	# Filming uses a tripod: no follow, orbit input, or obstacle-driven zoom.
	# Only the camera changes; physics and controller time remain untouched.
	var shots: Array = options.plan.get("fixed_shots",[])
	if shots.is_empty(): return false
	var shot: Dictionary = shots[0]
	for candidate in shots:
		if _t >= float(candidate.at): shot = candidate
	var camera: Camera3D = get_node("World/Camera3D")
	var position := Vector3(shot.position[0],shot.position[1],shot.position[2])
	var target := Vector3(shot.target[0],shot.target[1],shot.target[2])
	camera.global_transform = Transform3D(Basis.looking_at(target-position,Vector3.UP),position)
	camera.fov = float(shot.fov)
	recording_shot = str(shot.id)
	return true

func _capture() -> void:
	if not capture_primed:
		capture_primed=true;return
	var now := Time.get_ticks_usec()
	if now < next_capture: return
	var interval:=int(1000000.*float(options.plan.get("capture_interval",.033333)))
	# Keep an absolute cadence: frame jitter must not turn 30 Hz into 15 Hz.
	next_capture=maxi(next_capture+interval,now)
	if capture_job >= 0 and not WorkerThreadPool.is_task_completed(capture_job):
		dropped += 1
		return
	if capture_job >= 0: WorkerThreadPool.wait_for_task_completion(capture_job)
	var folder: String = options.output+"/frames"
	DirAccess.make_dir_recursive_absolute(folder)
	var path := folder.path_join("%05d.jpg" % frame_index)
	frame_index += 1
	var texture := get_viewport().get_texture()
	var image := texture.get_image()
	var camera: Camera3D = get_node("World/Camera3D")
	var basis := camera.global_basis
	frames.append({"file":path,"milliseconds":float(now-capture_start)/1000.,"sim_seconds":_t,"hub_seconds":elapsed,
		"robot":active_robot,"task":active_task,"shot":recording_shot,
		"controls":controls_hint.text,
		"camera_position":[camera.global_position.x,camera.global_position.y,camera.global_position.z],
		"camera_basis":[[basis.x.x,basis.x.y,basis.x.z],[basis.y.x,basis.y.y,basis.y.z],[basis.z.x,basis.z.y,basis.z.z]],"fov":camera.fov})
	capture_job = WorkerThreadPool.add_task(func(): image.save_jpg(path,.97))

func on_task_finished(result: Dictionary) -> void:
	task_result = result
	var file := FileAccess.open(options.output+"/task.json",FileAccess.WRITE)
	file.store_string(JSON.stringify(result))
	print("HUB_TASK_FINISHED ",result.get("success","pending independent evaluation")," seconds=",_t)
	if options.plan.get("finish_on_task",false):
		_finish.call_deferred()
	else:
		for body in actor.robot.bodies.values(): body.freeze = true
		if actor.robot.item != null: actor.robot.item.freeze = true

func _finish() -> void:
	if stopping: return
	stopping = true
	if atelier == null:
		get_tree().quit();return
	_save_native_trace()
	if capture_job >= 0: WorkerThreadPool.wait_for_task_completion(capture_job)
	var result := {"pid":OS.get_process_id(),"hub":get_instance_id(),"atelier":atelier.get_instance_id(),
		"events":events,"samples":samples,"seconds":elapsed,"frames":frames,"dropped":dropped,"active_robot":active_robot,
		"task":active_task,"task_success":task_result.get("success",null)}
	result["route"] = route_result
	result["scene"] = options.get("scene","workshop")
	result["checkpoint"] = checkpoint
	result["checkpoint_visits"] = checkpoint_visits
	result["grab_sessions"] = grab_sessions
	result["props"] = atelier.loose_props.telemetry()
	if _base != null: result["position"] = [_base.global_position.x,_base.global_position.y,_base.global_position.z]
	var file := FileAccess.open(options.output+"/hub.json",FileAccess.WRITE)
	file.store_string(JSON.stringify(result,"  "))
	get_tree().quit()

func _prop_snapshot() -> Array:
	var result: Array = []
	for item in atelier.loose_props.items:
		var p: Vector3 = item.body.global_position
		result.append({"id":item.body.get_instance_id(),"position":[p.x,p.y,p.z]})
	return result

func _stage_label(stage: String) -> String:
	return {"rolling":"平地驾驶","crouched":"下蹲驾驶","crouch_blocked":"前方台阶 · 松开 Shift 爬阶","stairs":"台阶驾驶","ready":"就绪",
		"lower_body":"降低车身","approach":"靠近零件","pregrasp":"对准夹爪","grasp":"抓取零件",
		"close":"夹爪闭合","lift":"抬起零件","raise_body":"升起车身","front_clearance":"避让前沿",
		"transfer_1":"移向货仓","transfer_2":"移向货仓","transfer_3":"移向货仓","transfer_4":"移向货仓",
		"place":"放入货仓","release":"松开夹爪","clear_fixed_finger":"退出夹爪","retreat":"收回机械臂",
		"secure_cargo":"夹紧货物","loaded_settle":"稳定车身","loaded_crawl":"夹紧运输","drive":"驾驶"}.get(stage,stage)

func _save_native_trace() -> void:
	if actor == null: return
	if active_robot == "sai":
		if actor.grab != null:
			grab_sessions.append({"history":actor.grab.history.duplicate(true),"deliveries":actor.grab.deliveries.duplicate(true),"retained":actor.grab.retention()})
		if not options.plan.is_empty() and not actor.records.is_empty():
			var file := FileAccess.open(options.output+"/sai-native-trace.json",FileAccess.WRITE)
			file.store_string(JSON.stringify({"rows":actor.records,"grab_sessions":grab_sessions}))
		return
	if actor.session.trace_path == "": return
	var file := FileAccess.open(actor.session.trace_path,FileAccess.WRITE)
	file.store_string(JSON.stringify({"robot":active_robot,"pid":OS.get_process_id(),"rows":actor.session.rows,
		"first_fall":actor.session.first_fall,"deployment":actor.deployment}))
