extends "res://physics_server.gd"
const VisualProfile = preload("res://atelier/visual_profile.gd")
## Workshop terrain is local to this scene; the shared robot controller stays unchanged.

var atelier: Node3D
var draw_period_usec: int = 33334
var draw_last_usec: int = 0
var draw_next_usec: int = 0
var draw_times: Array[float] = []
var draw_costs: Array[float] = []
var viewport_gpu_times: Array[float] = []
var viewport_cpu_times: Array[float] = []
var measure_viewport := false
var atelier_baseline := false
var atelier_mode := "play"
var baseline_view := "follow"
var recording_dir := ""
var recording_start: int = 0
var recording_last: int = 0
var recording_frames: Array = []
var recording_next: int = 0
var recording_period: int = 66667
var recording_costs: Array[float] = []
var recording_readback_costs: Array[float] = []
var recording_task: int = -1
var recording_index := 0
var recording_dropped := 0
var recording_format := "jpg"
var recording_mutex:=Mutex.new()
var presentation_ready := false
var drawing := false
var shadow_size := 4096

func _ready() -> void:
	atelier_baseline = OS.get_environment("MD_MODE") == "baseline"
	atelier_mode = OS.get_environment("MD_MODE")
	super._ready()
	if _headless:
		atelier=load("res://atelier/workshop.gd").new()
		add_child(atelier);atelier.build(self)
		return
	draw_period_usec = int(1000000.0 / maxf(1.0, float(OS.get_environment("MD_RENDER_FPS") if OS.has_environment("MD_RENDER_FPS") else "30")))
	RenderingServer.render_loop_enabled = false
	measure_viewport=OS.get_environment("MD_BENCHMARK")=="1"
	if measure_viewport:RenderingServer.viewport_set_measure_render_time(get_viewport().get_viewport_rid(),true)
	get_window().title = "MicroDuck · 小小维修站"
	if OS.has_environment("MD_WIDTH") and OS.has_environment("MD_HEIGHT"):
		get_window().size=Vector2i(int(OS.get_environment("MD_WIDTH")),int(OS.get_environment("MD_HEIGHT")))
		get_window().content_scale_size=Vector2i(1280,720)
		get_window().content_scale_mode=Window.CONTENT_SCALE_MODE_CANVAS_ITEMS
	get_viewport().msaa_3d = Viewport.MSAA_4X
	if VisualProfile.value("MD_MSAA")=="8":get_viewport().msaa_3d=Viewport.MSAA_8X
	if VisualProfile.value("MD_SHADOW_FILTER")=="medium":RenderingServer.directional_soft_shadow_filter_set_quality(RenderingServer.SHADOW_QUALITY_SOFT_MEDIUM)
	if VisualProfile.value("MD_SHADOW_FILTER")=="high":RenderingServer.directional_soft_shadow_filter_set_quality(RenderingServer.SHADOW_QUALITY_SOFT_HIGH)
	shadow_size=int(ProjectSettings.get_setting("rendering/lights_and_shadows/directional_shadow/size",4096))
	if VisualProfile.value("MD_SHADOW_SIZE")=="8192":
		shadow_size=8192
		RenderingServer.directional_shadow_atlas_set_size(shadow_size,true)
	atelier = load("res://atelier/workshop.gd").new()
	add_child(atelier)
	if not atelier_baseline:
		atelier.build(self)
	else:
		atelier.server = self
		atelier.camera = get_node("World/Camera3D")
		atelier.camera.near = 0.015
		atelier.camera.far = 100.0
	print("ATELIER isolated presentation; render cap=", 1000000.0 / draw_period_usec)
	presentation_ready=true

func _process(delta: float) -> void:
	super._process(delta)
	if _headless:
		return
	_draw_if_due()
	# The baseline lockstep launcher uses --fixed-fps; keep idle loops polite.
	if _peer == null:
		OS.delay_usec(2000)

func _draw_if_due() -> void:
	if not presentation_ready or drawing:return
	var now := Time.get_ticks_usec()
	if now >= draw_next_usec:
		drawing=true
		if draw_last_usec > 0:
			draw_times.append(float(now - draw_last_usec) / 1000.0)
		draw_last_usec = now
		if draw_next_usec == 0: draw_next_usec = now
		draw_next_usec += (int(maxi(0, now-draw_next_usec)/draw_period_usec)+1)*draw_period_usec
		var render_start := Time.get_ticks_usec()
		atelier.update_printed_labels()
		RenderingServer.force_draw(true)
		draw_costs.append(float(Time.get_ticks_usec()-render_start)/1000.0)
		if measure_viewport:
			viewport_gpu_times.append(RenderingServer.viewport_get_measured_render_time_gpu(get_viewport().get_viewport_rid()))
			viewport_cpu_times.append(RenderingServer.viewport_get_measured_render_time_cpu(get_viewport().get_viewport_rid()))
		# Interactive sessions keep only recent diagnostics; bounded captures retain every sample.
		if atelier_mode in ["play","tour"] and draw_times.size()>1800:
			draw_times=draw_times.slice(900)
			draw_costs=draw_costs.slice(900)
		if recording_dir != "" and now >= recording_next:
			recording_last = now
			recording_next+=(int(maxi(0,now-recording_next)/recording_period)+1)*recording_period
			if recording_task>=0 and not WorkerThreadPool.is_task_completed(recording_task):
				recording_dropped+=1
			else:
				_finish_recording_task()
				var path := recording_dir.path_join("%05d.%s" % [recording_index,recording_format])
				recording_index+=1
				var capture_start:=Time.get_ticks_usec()
				var img := get_viewport().get_texture().get_image()
				recording_readback_costs.append(float(Time.get_ticks_usec()-capture_start)/1000.0)
				var cam:=get_node("World/Camera3D") as Camera3D
				var p:=cam.global_position
				var b:=cam.global_basis
				var sample: Dictionary={"milliseconds":float(now-recording_start)/1000.0,"sim_seconds":_t,"view":atelier.view,"camera_position":[p.x,p.y,p.z],"camera_basis":[[b.x.x,b.x.y,b.x.z],[b.y.x,b.y.y,b.y.z],[b.z.x,b.z.y,b.z.z]],"fov":cam.fov,"orbit_yaw":_cam_yaw,"orbit_pitch":_cam_pitch,"orbit_distance":_cam_dist}
				recording_task=WorkerThreadPool.add_task(_save_recorded_frame.bind(img,path,sample,recording_format),false,"Atelier frame save")
		drawing=false

func _save_recorded_frame(img: Image,path: String,sample: Dictionary,format: String) -> void:
	# One bounded worker owns the immutable Image; it never accesses scene nodes.
	var start:=Time.get_ticks_usec()
	var err:=img.save_jpg(path,.97) if format=="jpg" else img.save_png(path)
	recording_mutex.lock()
	recording_costs.append(float(Time.get_ticks_usec()-start)/1000.0)
	if err==OK:
		sample["file"]=path
		recording_frames.append(sample)
	recording_mutex.unlock()

func _finish_recording_task() -> void:
	if recording_task>=0:
		WorkerThreadPool.wait_for_task_completion(recording_task)
		recording_task=-1

func _exit_tree() -> void:
	_finish_recording_task()

func _follow_camera(dt: float) -> void:
	if atelier != null and atelier.has_method("update_camera") and atelier.update_camera(dt):
		# The original lockstep wait keeps calling this visual hook. Drawing here
		# prevents Python's 50 Hz command cadence from quantizing render intervals.
		_draw_if_due()
		return
	super._follow_camera(dt)

func _freeze(value: bool) -> void:
	super._freeze(value)
	if atelier!=null and atelier.loose_props!=null:atelier.loose_props.set_frozen(value)

func _do_reset(cmd: Dictionary) -> void:
	if atelier!=null and atelier.loose_props!=null:atelier.loose_props.reset()
	super._do_reset(cmd)

func _send_dict(data: Dictionary) -> void:
	# Same counted physics reply, no diagnostic RPC that would insert extra ticks.
	if OS.get_environment("MD_PROP_TELEMETRY")=="1" and data.get("cmd","") in ["step","reset"] and atelier!=null and atelier.loose_props!=null:
		data["workshop_props"]=atelier.loose_props.telemetry()
	super._send_dict(data)

func _handle(cmd: Variant) -> void:
	# Presentation events attached to a normal control command must not create
	# extra returns from the inherited physics command pump.
	if str(cmd.get("cmd",""))=="step":
		if atelier!=null and atelier.loose_props!=null and atelier.loose_props.selected>0 and cmd.has("place_ball"):
			var bp:Array=cmd.place_ball
			var target:=_m2g(Vector3(float(bp[0]),float(bp[1]),float(bp[2])))
			# Ignore the controller's far-away "hide ball" setup command.
			if _base!=null and Vector2(target.x-_base.global_position.x,target.z-_base.global_position.z).length()<.4:
				atelier.loose_props.place_target(target)
				cmd=cmd.duplicate();cmd.place_ball=[5.0,5.0,.035]
		for action in cmd.get("atelier_actions",[]):_apply_presentation_action(action)
	if str(cmd.get("cmd", "")) == "atelier_prop_target":
		if atelier==null or atelier.loose_props==null:
			_send_dict({"ok":false,"error":"loose props disabled"})
		else:
			atelier.loose_props.selected=clampi(int(cmd.get("index",0)),0,atelier.loose_props.items.size())
			_send_dict({"ok":true,"target":atelier.loose_props.target_label()})
		return
	if str(cmd.get("cmd", "")) == "atelier_camera_lock":
		# Replay a measured camera for exact before/after framing; no body is moved.
		atelier.capture_camera=cmd.camera
		atelier.update_camera(0.0)
		_send_dict({"ok":true,"cmd":"atelier_camera_lock"})
		return
	if str(cmd.get("cmd", "")) == "atelier_mouse":
		if cmd.has("button"):
			var ev:=InputEventMouseButton.new()
			ev.button_index=int(cmd.button);ev.pressed=bool(cmd.get("pressed",false))
			ev.position=get_viewport().get_visible_rect().size*.5
			Input.parse_input_event(ev)
		else:
			var ev:=InputEventMouseMotion.new()
			ev.relative=Vector2(float(cmd.get("x",0)),float(cmd.get("y",0)))
			ev.position=get_viewport().get_visible_rect().size*.5
			Input.parse_input_event(ev)
		_send_dict({"ok":true,"cmd":"atelier_mouse"})
		return
	if str(cmd.get("cmd", "")) == "atelier_reset_metrics":
		draw_times.clear();draw_costs.clear();viewport_gpu_times.clear();viewport_cpu_times.clear();draw_last_usec=0
		_send_dict({"ok":true,"cmd":"atelier_reset_metrics"})
		return
	if str(cmd.get("cmd", "")) == "atelier_camera":
		var cam:=get_node("World/Camera3D") as Camera3D
		var p:=cam.global_position
		var b:=cam.global_basis
		_send_dict({"ok":true,"cmd":"atelier_camera","position":[p.x,p.y,p.z],"basis":[[b.x.x,b.x.y,b.x.z],[b.y.x,b.y.y,b.y.z],[b.z.x,b.z.y,b.z.z]],"fov":cam.fov,"near":cam.near,"far":cam.far,"view":atelier.view,"viewport":[get_viewport().size.x,get_viewport().size.y]})
		return
	if str(cmd.get("cmd", "")) == "atelier_camera_probe":
		# Presentation-only test input. Does not move a body or modify the physics world.
		var p: Array=cmd.get("target",[0,.16,0])
		var q: Array=cmd.get("wanted",[0,.50,2])
		var target:=Vector3(p[0],p[1],p[2])
		var wanted:=Vector3(q[0],q[1],q[2])
		var safe: Vector3=atelier._unobstructed_position(target,wanted)
		_send_dict({"ok":true,"cmd":"atelier_camera_probe","position":[safe.x,safe.y,safe.z],"obstacles":atelier.camera_obstacles.size()})
		return
	if str(cmd.get("cmd", "")) == "atelier_contract":
		var physics: Array = []
		_collect_contract(self, physics)
		_send_dict({"ok":true,"cmd":"atelier_contract","physics":physics,"prop_target":atelier.loose_props.selected if atelier!=null and atelier.loose_props!=null else 0,"ticks":Engine.physics_ticks_per_second,"view":atelier.view if atelier!=null else "follow","help_visible":atelier.hud.help_panel.visible if atelier!=null and atelier.hud!=null else false,"debug_visible":atelier.hud.debug_panel.visible if atelier!=null and atelier.hud!=null else false})
		return
	if str(cmd.get("cmd", "")) == "close":
		_finish_recording_task()
		RenderingServer.render_loop_enabled = true
	if str(cmd.get("cmd", "")) == "atelier_record":
		_finish_recording_task()
		recording_dir = str(cmd.get("directory", ""))
		if recording_dir != "":
			DirAccess.make_dir_recursive_absolute(recording_dir)
			recording_start = Time.get_ticks_usec()
			recording_last = 0
			recording_next=recording_start
			recording_period=int(1000000.0/clampf(float(cmd.get("fps",15)),1,30))
			recording_format="png" if str(cmd.get("format","jpg"))=="png" else "jpg"
			recording_index=0;recording_dropped=0
			recording_frames.clear()
			recording_costs.clear()
			recording_readback_costs.clear()
		_send_dict({"ok":true,"cmd":"atelier_record","frames":recording_frames})
		return
	if str(cmd.get("cmd", "")) == "atelier_view":
		baseline_view = str(cmd.get("view", "follow"))
		if atelier != null:
			atelier.set_view(str(cmd.get("view", "follow")))
		elif cmd.has("position") and cmd.has("target"):
			var cam := get_node("World/Camera3D") as Camera3D
			var p: Array = cmd.position
			var t: Array = cmd.target
			cam.position = Vector3(p[0], p[1], p[2])
			cam.look_at(Vector3(t[0], t[1], t[2]))
		_send_dict({"ok": true, "cmd": "atelier_view"})
		return
	if str(cmd.get("cmd", "")) == "screenshot":
		_follow_camera(0.0)
		var due := draw_period_usec - (Time.get_ticks_usec() - draw_last_usec)
		if due > 0: OS.delay_usec(due)
		atelier.update_printed_labels()
		RenderingServer.force_draw(true)
		draw_last_usec = Time.get_ticks_usec()
		draw_next_usec = draw_last_usec + draw_period_usec
	if str(cmd.get("cmd", "")) == "atelier_metrics":
		_finish_recording_task()
		_send_dict({"ok": true, "cmd": "atelier_metrics", "frame_ms": draw_times,"viewport_gpu_ms":viewport_gpu_times,"viewport_cpu_ms":viewport_cpu_times, "render_cost_ms": draw_costs, "render_cap": 1000000.0 / draw_period_usec, "renderer": RenderingServer.get_video_adapter_name(),"rendering_method":RenderingServer.get_current_rendering_method(),"msaa_3d":get_viewport().msaa_3d,"shadow_atlas_size":shadow_size,"visual_profile":VisualProfile.resolved(),"visible_draw_calls":RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_DRAW_CALLS_IN_FRAME),"video_memory_bytes":RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_VIDEO_MEM_USED), "recording":recording_frames,"recording_cost_ms":recording_costs,"recording_readback_ms":recording_readback_costs,"recording_dropped":recording_dropped,"recording_format":recording_format})
		return
	super._handle(cmd)
	if str(cmd.get("cmd", "")) == "reset" and atelier != null:
		atelier.reset_camera()

func _apply_presentation_action(action: Dictionary) -> void:
	match str(action.get("type","")):
		"view":atelier.set_view(str(action.get("view","follow")))
		"zoom":_cam_dist=clampf(_cam_dist*(1.0+CAM_DIST_DRAG*40.0*float(action.get("d",0))),CAM_DIST_MIN,CAM_DIST_MAX)
		"mouse":
			if action.has("button"):
				var event:=InputEventMouseButton.new()
				event.button_index=int(action.button);event.pressed=bool(action.get("pressed",false))
				event.position=get_viewport().get_visible_rect().size*.5
				Input.parse_input_event(event)
			else:
				var event:=InputEventMouseMotion.new()
				event.relative=Vector2(float(action.get("x",0)),float(action.get("y",0)))
				event.position=get_viewport().get_visible_rect().size*.5
				Input.parse_input_event(event)
			Input.flush_buffered_events()

func _collect_contract(node: Node, output: Array) -> void:
	if node is PhysicsBody3D or node is CollisionShape3D or node is Joint3D:
		var item := {"path":str(get_path_to(node)),"type":node.get_class()}
		if node is RigidBody3D:
			item["mass"]=node.mass
			item["inertia"]=[node.inertia.x,node.inertia.y,node.inertia.z]
		if node is PhysicsBody3D:
			item["layer"]=node.collision_layer
			item["mask"]=node.collision_mask
		if node is CollisionShape3D:
			item["disabled"]=node.disabled
			item["shape"]=node.shape.get_class()
		output.append(item)
	for child in node.get_children():_collect_contract(child,output)
