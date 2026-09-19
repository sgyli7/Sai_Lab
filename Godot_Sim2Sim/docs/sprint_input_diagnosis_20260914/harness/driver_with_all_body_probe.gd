extends "res://physics_server.gd"
## In-process controller. Physics, observations, reset and actuation stay in the
## same superclass used by the Python research driver.

const Brain = preload("res://standalone/play_brain.gd")
const Contract = preload("res://standalone/policy_contract.gd")
const Motion = preload("res://standalone/motion_control.gd")
const BrakeTask = preload("res://standalone/brake_task_state.gd")
const CONTROL_DT := 0.02
const MOTOR_KT := 0.36601349688984386
const SKILL_LABELS := {"standing":"站立","walking":"行走","sprint":"加速行走","sitstand":"坐下 / 起身",
	"ground_pick":"捡地","kick_left":"左脚踢球","kick_right":"右脚踢球",
	"roulade":"前滚翻","roller":"轮滑","roller_crouch":"轮滑下蹲 / 起身"}
var deployment: Dictionary
var robot_config: Dictionary
var bank: Dictionary = {}
var brain = Brain.new()
var motion = Motion.new()
var brake_task = BrakeTask.new()
var local_reply: Dictionary = {}
var home := PackedFloat32Array()
var last_action := PackedFloat32Array()
var heading: Array = [1.0,0.0]
var session: Dictionary
var ready_to_run := false
var fall_time := 0.0
var start_usec := 0
var forced_skill := ""
var render_fps := 0
var control_config: Dictionary = {}
var ui_font: Font

func _start_controller() -> void:
	# Deliberately no TCPServer, process launch, socket, or external controller.
	pass

func _send_dict(value: Dictionary) -> void:
	local_reply = value

func _ready() -> void:
	start_usec = Time.get_ticks_usec()
	deployment = _read_json("res://runtime_assets/deployment.json")
	if deployment.is_empty(): return
	if get_tree().has_meta("microduck_session"):
		session = get_tree().get_meta("microduck_session")
	else:
		session = {"mode":"walk","steps":0,"rows":[],"replay":{},"trace_path":"",
			"seconds":0.0,"segment":-1,"resets":0,"switches":0,"first_fall":null,
			"started_usec":start_usec,"seed":915000,"error":"","events":[],"profiles":{}}
		var replay_requested := false
		for arg in OS.get_cmdline_user_args():
			if arg == "--roller": session.mode = "roller"
			elif arg.begins_with("--replay="):
				replay_requested = true
				session.replay = _read_json(arg.trim_prefix("--replay="))
			elif arg.begins_with("--trace="): session.trace_path = arg.trim_prefix("--trace=")
			elif arg.begins_with("--seconds="):
				var text := arg.trim_prefix("--seconds=")
				if not text.is_valid_float():
					_fatal("Run duration must be a number")
					return
				session.seconds = float(text)
			elif arg.begins_with("--seed="): session.seed = int(arg.trim_prefix("--seed="))
		if session.error != "": return
		if not is_finite(float(session.seconds)) or float(session.seconds) < 0.0:
			_fatal("Run duration must be nonnegative and finite")
			return
		if replay_requested:
			var replay_error := _replay_error(session.replay)
			if replay_error != "":
				_fatal(replay_error)
				return
			session.mode = session.replay.get("mode",session.mode)
			session.seconds = session.replay.get("seconds",session.seconds)
		get_tree().set_meta("microduck_session",session)
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--skill="): forced_skill=arg.trim_prefix("--skill=")
		elif arg.begins_with("--render-fps="): render_fps=int(arg.trim_prefix("--render-fps="))
	seed(int(session.seed))
	control_config = session.replay.get("control_config",deployment.get("control_config",{}))
	motion.settings = control_config.get(session.mode,{})
	robot_config = deployment.robots[session.mode]
	_robot_scene = robot_config.scene
	_spec_path = robot_config.spec
	_base_name = robot_config.base_body
	if FileAccess.get_sha256(_spec_path) != robot_config.spec_sha256:
		_fatal("Robot specification checksum mismatch")
		return
	if int(deployment.physics_hz) != 200 or int(deployment.decimation) != 4:
		_fatal("Unsupported physics/control contract")
		return
	if not _load_models(): return
	if not _load_ui_font(): return
	if OS.get_cmdline_user_args().has("--self-test"):
		_self_test()
		return
	home = PackedFloat32Array(robot_config.home)
	last_action.resize(14)
	var flags := {"walking":true,"standing":false,"sitstand":false,"ground_pick":false,
		"kick_left":false,"kick_right":false,"roulade":false,"roller_crouch":false}
	var limits: Dictionary
	if session.mode == "roller":
		flags.roller_crouch = true
		limits = deployment.roller_limits
	else:
		flags.sprint = bank.has("sprint")
		var sim: Dictionary = deployment.policies.walking.manifest.get("sim2sim",{})
		limits = sim.get("twist_limits",{})
		flags.standing = bool(sim.get("use_stand_policy",true))
		flags.stand_hold = true
		for key in ["sitstand","ground_pick","kick_left","kick_right","roulade"]: flags[key]=true
	limits = limits.duplicate()
	var overrides: Dictionary = control_config.get(session.mode,{}).get("twist_limits",{})
	for key in overrides:
		if not key in ["vmax_x","vmin_x","vmax_y","vmin_y","vmax_ang","accel","decel","switch_on","switch_off","switch_threshold","sprint_vmax_x","sprint_vmax_ang","sprint_yaw_reversal_s"] or not is_finite(float(overrides[key])):
			_fatal("Invalid control limit: "+str(key))
			return
		limits[key]=overrides[key]
	if float(limits.get("sprint_yaw_reversal_s",0.0)) < 0.0:
		_fatal("Sprint yaw reversal duration must be nonnegative")
		return
	brain.configure(flags,limits)
	super._ready()
	Engine.max_fps = render_fps
	process_mode = Node.PROCESS_MODE_ALWAYS
	if _hud != null: _hud.configure_standalone(ui_font,session.mode == "walk" and bank.has("sprint"))
	if Engine.physics_ticks_per_second != 200:
		_fatal("Physics must run at 200 Hz")
		return
	_handle({"cmd":"set_tau_limit","limit":MOTOR_KT*float(robot_config.current_limit_a)})
	_reset_controller()
	ready_to_run = true
	print("STANDALONE_READY "+JSON.stringify({"mode":session.mode,"models":bank.size(),
		"runtime":"1.29.0","physics_hz":200,"policy_hz":50,"tcp":false}))

func _read_json(path: String) -> Dictionary:
	if not FileAccess.file_exists(path):
		_fatal("Missing resource: "+path)
		return {}
	var value: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	if typeof(value) != TYPE_DICTIONARY:
		_fatal("Invalid JSON object: "+path)
		return {}
	return value

func _load_ui_font() -> bool:
	var item: Dictionary = deployment.get("ui_font",{})
	if item.is_empty() or FileAccess.get_sha256(item.get("path","")) != item.get("sha256",""):
		_fatal("Embedded interface font is missing or has changed")
		return false
	var font_file := FontFile.new()
	font_file.data = FileAccess.get_file_as_bytes(item.path)
	if font_file.get_face_count() <= int(item.face_index):
		_fatal("Embedded interface font has an invalid face")
		return false
	var font := FontVariation.new()
	font.base_font = font_file
	font.variation_face_index = int(item.face_index)
	var required := "站立行走坐起捡地左踢右踢前滚轮滑刹车暂停复位"
	for i in range(required.length()):
		if not font.has_char(required.unicode_at(i)):
			_fatal("Embedded interface font lacks a required glyph")
			return false
	ui_font = font
	return true

func _load_models() -> bool:
	if not ClassDB.class_exists("MicroDuckPolicy"):
		_fatal("Native inference extension is unavailable")
		return false
	for skill in deployment.policies:
		var item: Dictionary = deployment.policies[skill]
		if FileAccess.get_sha256(item.path) != item.sha256:
			_fatal("Model checksum mismatch: "+skill)
			return false
		var state_mode: String = item.get("state_input", "")
		if state_mode != "" and (state_mode != "planar_com_velocity_height_v1" or skill not in ["walking", "sprint", "roller"] or item.get("time_input_s", 0.0) or item.get("heading_input", false)):
			_fatal("Unknown or incompatible policy state input: "+state_mode)
			return false
		var task_mode: String = item.get("task_input", "")
		if task_mode != "" and (task_mode != "brake_markov_68_v1" or skill != "roller" or state_mode == ""):
			_fatal("Unknown or incompatible task observation: "+task_mode)
			return false
		var policy = ClassDB.instantiate("MicroDuckPolicy")
		if not policy.load_model(FileAccess.get_file_as_bytes(item.path)):
			_fatal("Cannot load "+skill+": "+policy.get_last_error())
			return false
		if policy.get_runtime_version() != "1.29.0":
			_fatal("ONNX Runtime version mismatch")
			return false
		if policy.get_metadata().get("sim2sim_brake_state_input", "") != state_mode:
			_fatal("Policy state contract does not match deployment: "+skill)
			return false
		if policy.get_metadata().get("sim2sim_roller_task_input", "") != task_mode:
			_fatal("Policy task contract does not match deployment: "+skill)
			return false
		bank[skill] = policy
	return true

func _reset_controller() -> void:
	brain.reset_motion()
	motion.reset()
	brake_task.reset()
	last_action.fill(0.0)
	heading = [1.0,0.0]
	fall_time = 0.0
	var poses: Array = session.replay.get("initial_poses",{}).get(session.mode,robot_config.poses)
	_handle({"cmd":"reset","ctrl":Array(home),"bodies":poses,
		"report_bodies":_telemetry_bodies("standing")})
	if not local_reply.get("missing",[]).is_empty():
		_fatal("Reset references missing rigid bodies")
	_report_mode = "research" if session.trace_path != "" or _needs_task_telemetry() else "lite"

func _physics_process(delta: float) -> void:
	if not ready_to_run or get_tree().paused: return
	_refresh_after_physics(delta)
	if _remaining <= 0:
		if not _headless: _sample_held()
		var held := _held_now.duplicate()
		var taps := _taps.duplicate()
		var order := _held_press_order.duplicate()
		_send_state("step")
		_pending_send = false
		var elapsed := int(session.steps)*CONTROL_DT
		if session.seconds > 0.0 and elapsed >= session.seconds-1e-9:
			_finish()
			return
		if not session.replay.is_empty():
			var event := _replay_input(elapsed)
			held = event.held
			taps = event.taps
			order = event.order
		if not _decide(held,taps,order,elapsed): return
	_advance_physics_tick(delta)

func _replay_input(elapsed: float) -> Dictionary:
	var result := {"held":[],"taps":[],"order":[]}
	var selected := -1
	var segments: Array = session.replay.get("segments",[])
	for i in range(segments.size()):
		if float(segments[i].get("at",0.0)) <= elapsed+1e-9: selected=i
	if selected >= 0:
		var segment: Dictionary = segments[selected]
		result.held = segment.get("held",[])
		result.order = segment.get("order",result.held)
		if selected != int(session.segment): result.taps=segment.get("taps",[])
		session.segment = selected
	return result

func _replay_error(replay: Dictionary) -> String:
	if replay.get("mode",session.mode) not in ["walk","roller"]:
		return "Invalid replay robot mode"
	if typeof(replay.get("seconds",null)) not in [TYPE_INT,TYPE_FLOAT]:
		return "Replay duration must be a number"
	var seconds := float(replay.get("seconds",0.0))
	if not is_finite(seconds) or seconds <= 0.0:
		return "Replay duration must be positive and finite"
	var segments: Variant = replay.get("segments",[])
	if typeof(segments) != TYPE_ARRAY or segments.is_empty():
		return "Replay requires a nonempty segment array"
	var previous := -1.0
	for segment in segments:
		if typeof(segment) != TYPE_DICTIONARY:
			return "Replay segment must be an object"
		if typeof(segment.get("at",null)) not in [TYPE_INT,TYPE_FLOAT]:
			return "Replay time must be a number"
		var at := float(segment.get("at",0.0))
		if not is_finite(at) or at < 0.0 or at >= seconds or at <= previous:
			return "Replay times must increase within the duration"
		previous=at
		for key in ["held","order","taps"]:
			var values: Variant = segment.get(key,[])
			if typeof(values) != TYPE_ARRAY:
				return "Replay "+key+" must be an array"
			var allowed := ["pick","sit","kick_left","kick_right","roulade","stand","switch_robot","reset","push","quit"] if key == "taps" else ["fwd","back","left","right","strafe_l","strafe_r","idle","sprint"]
			for value in values:
				if value not in allowed:
					return "Unknown replay "+key+" input: "+str(value)
	return ""

func _decide(held: Array, taps: Array, order: Array, elapsed: float) -> bool:
	var raw := local_reply.duplicate(true)
	var body := Contract.body_state(raw,robot_config)
	var out: Dictionary = brain.tick(held,taps,CONTROL_DT,order)
	if out.quit:
		_finish()
		return false
	if out.switch_robot:
		session.events.append({"kind":"switch_robot","time":elapsed,"from":session.mode})
		session.mode = "walk" if session.mode == "roller" else "roller"
		session.switches += 1
		ready_to_run = false
		# Reloading frees the old robot and ORT sessions; only replay/trace survives.
		get_tree().call_deferred("reload_current_scene")
		return false
	if out.reset:
		session.events.append({"kind":"reset","time":elapsed,"mode":session.mode})
		session.resets += 1
		_reset_controller()
		return false
	if out.push:
		var angle := randf()*TAU
		_handle({"cmd":"nudge","linvel":[cos(angle),sin(angle),0.0]})
	var skill: String = "roller" if session.mode == "roller" and out.policy == "walking" else out.policy
	if session.mode == "walk" and out.sprint: skill = "sprint"
	if forced_skill != "": skill=forced_skill
	if not bank.has(skill):
		_fatal("Unknown skill: "+skill)
		return false
	var item: Dictionary = deployment.policies[skill]
	var command: PackedFloat32Array = out.command
	if float(item.time_input_s) > 0.0:
		if out.started_skill == skill:
			var r := Contract.quat_matrix(body.base_quat)
			var yaw := atan2(r[1][0],r[0][0])
			heading = [cos(yaw),sin(yaw)]
		command = Contract.time_command(float(item.time_input_s)-brain.behavior_t,
			item.time_input_s,body,item.heading_input,heading)
	var requested_command := command.duplicate()
	command = motion.command(command,body,skill,CONTROL_DT)
	var obs := Contract.observation(raw,body,last_action,command,home)
	if item.get("state_input", "") == "planar_com_velocity_height_v1":
		obs = Contract.brake_state_observation(obs,body)
	if item.get("task_input", "") == "brake_markov_68_v1":
		obs = brake_task.observe(obs,raw,robot_config.get("support_groups",[]),_t)
		if obs.size()!=68:
			_fatal("Task observation is missing wheel telemetry or has invalid time")
			return false
	else:
		brake_task.reset()
	var action: PackedFloat32Array = bank[skill].infer(obs)
	if action.size() != 14:
		_fatal("Inference failed for "+skill+": "+bank[skill].get_last_error())
		return false
	var infer_usec: int = bank[skill].get_last_infer_usec()
	_record_latency(skill,infer_usec)
	var ctrl := Contract.control(action,home,float(robot_config.action_scale))
	var tilt := rad_to_deg(acos(clampf(-obs[5],-1.0,1.0)))
	var fell: bool = tilt > 70.0 or body.base_pos[2] < 0.055
	if fell and session.first_fall == null: session.first_fall=elapsed
	if session.trace_path != "":
		session.rows.append({"t":elapsed,"episode_t":_t,"mode":session.mode,"skill":skill,
			"wall_usec":Time.get_ticks_usec()-session.started_usec,
			"held":held,"taps":taps,"order":order,"command":Array(command),
			"requested_command":Array(requested_command),
			"obs":Array(obs),"action":Array(action),"last_action":Array(last_action),
			"ctrl":Array(ctrl),"raw":raw,"body":body,"tilt":tilt,"fell":fell,
			"infer_usec":infer_usec})
	last_action = action.duplicate()
	var hud_status: String = SKILL_LABELS.get(skill,skill)
	if fell and skill != "roulade": hud_status += " · 已失去平衡，按 0 复位"
	var step := {"cmd":"step","ctrl":Array(ctrl),"n_substeps":4,"hud":hud_status,
		"report":"research" if session.trace_path != "" or _needs_task_telemetry() else "lite"}
	_research_bodies = _telemetry_bodies(skill)
	if out.started_skill in ["kick_left","kick_right"]:
		step.place_ball=Contract.ball_position(body,out.started_skill)
	_handle(step)
	session.steps += 1
	return true

func _needs_task_telemetry() -> bool:
	return session.mode == "roller" and deployment.policies.roller.get("task_input","") != ""

func _telemetry_bodies(_skill: String) -> Array:
	return _bodies.keys()

func _finish() -> void:
	ready_to_run = false
	var result := {"schema_version":1,"mode":session.mode,"steps":session.steps,
		"sim_seconds":session.steps*CONTROL_DT,"wall_seconds":(Time.get_ticks_usec()-session.started_usec)/1e6,
		"resets":session.resets,"switches":session.switches,"first_fall":session.first_fall,
		"error":session.error,"physics_hz":200,"policy_hz":50,"tcp":false,
		"runtime_version":"1.29.0","models":{},"final_raw":local_reply}
	result.events = session.events
	result.control_config = control_config
	result.inference = _latency_summary()
	for skill in deployment.get("policies",{}): result.models[skill]=deployment.policies[skill].sha256
	if session.trace_path != "":
		var file := FileAccess.open(session.trace_path,FileAccess.WRITE)
		if file == null:
			push_error("Cannot write trace: "+session.trace_path)
			get_tree().quit(2)
			return
		file.store_string(JSON.stringify({"summary":result,"rows":session.rows}))
		file.close()
	var log_summary := result.duplicate()
	log_summary.erase("final_raw")
	print("STANDALONE_RESULT "+JSON.stringify(log_summary))
	get_tree().quit(0 if session.error == "" else 2)

func _fatal(message: String) -> void:
	push_error("Standalone acceptance stopped: "+message)
	ready_to_run = false
	if session != null and not session.is_empty():
		session.error = message
		_finish()
	else:
		get_tree().quit(2)

func _exit_tree() -> void:
	for policy in bank.values(): policy.unload()
	bank.clear()

func _record_latency(skill: String, usec: int) -> void:
	var profile: Dictionary = session.profiles.get(skill,{"count":0,"total_usec":0,"max_usec":0,"histogram":{}})
	profile.count += 1
	profile.total_usec += usec
	profile.max_usec = maxi(profile.max_usec,usec)
	# Bounded storage during indefinite interactive operation. Overflow is
	# recorded in the final bucket and the exact maximum remains available.
	var bucket := mini(usec,20000)
	profile.histogram[bucket] = int(profile.histogram.get(bucket,0))+1
	session.profiles[skill] = profile

func _latency_summary() -> Dictionary:
	var result := {}
	for skill in session.profiles:
		var profile: Dictionary = session.profiles[skill]
		var metrics := {"count":profile.count,"mean_usec":float(profile.total_usec)/profile.count,
			"max_usec":profile.max_usec,"histogram_cap_usec":20000}
		var buckets: Array = profile.histogram.keys()
		buckets.sort()
		for percent in [50,95,99]:
			var cumulative := 0
			for bucket in buckets:
				cumulative += int(profile.histogram[bucket])
				if cumulative >= ceil(float(profile.count)*percent/100.0):
					metrics["p%d_usec"%percent]=bucket
					break
		result[skill]=metrics
	return result

func _self_test() -> void:
	var fixture := _read_json("res://runtime_assets/self_test.json")
	if fixture.is_empty(): return
	var results: Array = []
	var passed := true
	for item in fixture.cases:
		var maximum := 0.0
		if item.sha256 != deployment.policies[item.skill].sha256:
			_fatal("Self-test model checksum mismatch")
			return
		for i in range(item.observations.size()):
			var action: PackedFloat32Array = bank[item.skill].infer(PackedFloat32Array(item.observations[i]))
			if action.size() != 14:
				_fatal("Self-test inference failed: "+item.skill)
				return
			for j in range(14): maximum=maxf(maximum,absf(action[j]-float(item.actions[i][j])))
		passed = passed and maximum < 1e-5
		results.append({"skill":item.skill,"max_abs":maximum,"cases":item.observations.size(),
			"real_count":item.get("real_count",0),"boundary_count":item.get("boundary_count",0)})
	print("STANDALONE_SELF_TEST "+JSON.stringify({"passed":passed,"models":results,
		"runtime":"1.29.0","tolerance":1e-5,"ui_font":deployment.ui_font}))
	get_tree().quit(0 if passed else 2)

func _on_hud_tap(action: String) -> void:
	if action == "pause":
		_toggle_pause()
	else:
		super._on_hud_tap(action)

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if OS.get_environment("SIM2SIM_TRACE_INPUT") == "1":
			print("STANDALONE_KEY "+JSON.stringify({"physical":event.physical_keycode,
				"logical":event.keycode,"step":session.steps}))
		if event.physical_keycode == KEY_F8:
			_toggle_pause()
			get_viewport().set_input_as_handled()
			return
	super._unhandled_input(event)

func _input(event: InputEvent) -> void:
	super._input(event)
	if OS.get_environment("SIM2SIM_TRACE_INPUT") == "1" and event is InputEventKey:
		print("STANDALONE_INPUT "+JSON.stringify({"physical":event.physical_keycode,
			"logical":event.keycode,"pressed":event.pressed,"echo":event.echo}))

func _toggle_pause() -> void:
	get_tree().paused = not get_tree().paused
	var event := {"kind":"pause","paused":get_tree().paused,"step":session.steps,
		"wall_usec":Time.get_ticks_usec()-session.started_usec,"episode_t":_t}
	session.events.append(event)
	print("STANDALONE_EVENT "+JSON.stringify(event))
	if _hud != null: _hud.set_status("已暂停 · F8 继续" if get_tree().paused else "继续运行")
