extends "res://standalone/driver.gd"
var hub: Node3D
var beak: Node3D

func _ready() -> void:
	super._ready()
	if hub != null and ready_to_run and bool(deployment.get("pickable_enabled",false)):
		beak = load("res://hub/microduck_beak.gd").new()
		beak.actor = self
		beak.hub = hub
		add_child(beak)

func _setup_play_ui() -> void:
	pass

func _follow_camera(_delta: float) -> void:
	pass

func _decide(held: Array, taps: Array, order: Array, elapsed: float) -> bool:
	if taps.has("switch_robot"):
		hub.select_robot("roller" if session.mode == "walk" else "sai")
		return false
	return super._decide(held,taps,order,elapsed)

func _physics_process(delta: float) -> void:
	if _headless: _sample_held()
	super._physics_process(delta)

func _handle(command: Variant) -> void:
	if beak != null and session.mode == "roller" and command is Dictionary and command.get("cmd","") == "step" and (beak.pending or beak.grip != null):
		command = command.duplicate(true)
		var ctrl: Array = command.ctrl
		# The roller actors have no floor-pick skill. Lower the existing head
		# servos toward a raised object, then return to home to hoist it.
		ctrl[5] = -0.8 if beak.pending else float(home[5])
		ctrl[6] = 1.0 if beak.pending else float(home[6])
		command.ctrl = ctrl
	if command is Dictionary and command.get("cmd","") == "step" and command.has("place_ball"):
		if hub.atelier.loose_props.selected > 0:
			var point: Array = command.place_ball
			hub.atelier.loose_props.place_target(_m2g(Vector3(point[0],point[1],point[2])))
			command = command.duplicate()
			command.erase("place_ball")
	if hub.is_science_station() and command is Dictionary:
		command=command.duplicate(true)
		if command.get("cmd","")=="reset":
			var offset:=_g2m(hub.spawn_point())
			for pose in command.get("bodies",[]):
				for i in range(3):pose.pos[i]+=offset[i]
		if command.has("place_ball"):
			var point: Array=command.place_ball
			point[2]+=hub.atelier.ground_height(float(point[0]),-float(point[1]))
	super._handle(command)

func _ground_height_at(body_pos: Array) -> float:
	if hub.has_method("is_polar_range") and hub.is_polar_range() and hub.polar_spawn!="under":return hub.polar_support_height()
	return hub.atelier.ground_height(float(body_pos[0]),-float(body_pos[1])) if hub.is_science_station() else 0.
