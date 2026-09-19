extends SceneTree
## Research baseline: G alone must collect a floor object without selecting it.

func _init() -> void:
	call_deferred("_run")

func _run() -> void:
	var hub = load("res://hub/main.tscn").instantiate()
	root.add_child(hub)
	for i in range(20): await physics_frame
	var duck = hub.actor
	var beak = duck.beak
	var props = hub.atelier.loose_props
	var target: RigidBody3D = props.items[2].body
	# A floor ball at the duck's feet, beneath the natural reach of the head.
	# No target selection or midair repositioning is used during the trial.
	var mouth: Vector3 = beak.head.to_global(beak.TIP_LOCAL)
	var floor_y: float = hub.atelier.ground_height(mouth.x, mouth.z)
	var radius: float = float(target.get_meta("grasp_size",Vector3.ONE*.07).y)*.5
	target.global_position = Vector3(mouth.x, floor_y+radius, mouth.z)
	target.linear_velocity = Vector3.ZERO
	target.angular_velocity = Vector3.ZERO
	target.freeze = false
	props.selected = 0
	var initial_y := target.global_position.y
	var start_position := target.global_position
	var key := InputEventKey.new()
	key.physical_keycode = KEY_G
	key.keycode = KEY_G
	key.pressed = true
	Input.parse_input_event(key)
	var closest := INF
	var highest := initial_y
	for tick in range(1000):
		await physics_frame
		if is_instance_valid(target):
			closest = minf(closest,beak.head.to_global(beak.TIP_LOCAL).distance_to(beak._center(target)))
			highest = maxf(highest,beak._center(target).y)
		if tick%200==0:
			print("G_PICK_FRAME ",JSON.stringify({"tick":tick,"phase":duck.brain.pick_phase,
				"policy":duck.brain.policy,"mouth_y":beak.head.to_global(beak.TIP_LOCAL).y,
				"object_y":beak._center(target).y,"pending":beak.pending,"grip":beak.grip!=null}))
	var result := {"input":"G only","selection":props.selected,"object":str(target.name),
		"object_start":start_position,"closest_m":closest,"lift_m":highest-initial_y,
		"attached":beak.grip!=null,"events":beak.history,"model":duck.deployment.policies.ground_pick.path}
	print("G_PICK_RESULT ",JSON.stringify(result))
	quit(0 if result.attached and result.lift_m>=.05 and result.selection==0 else 1)
