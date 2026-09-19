extends SceneTree
## Headless integration probe for the local Pickable model and beak grip.
var hub

func _init() -> void:
	call_deferred("_run")

func _run() -> void:
	hub = load("res://hub/main.tscn").instantiate()
	root.add_child(hub)
	for i in range(12): await physics_frame
	var duck = hub.actor
	assert(duck != null and duck.beak != null)
	var beak = duck.beak
	var pick_index := 2
	var override_mass := -1.0
	var carry_walk := false
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--pick-index="): pick_index = int(arg.trim_prefix("--pick-index="))
		if arg.begins_with("--mass-kg="): override_mass = float(arg.trim_prefix("--mass-kg="))
		if arg == "--carry-walk": carry_walk = true
	if pick_index < 0 or pick_index >= hub.atelier.loose_props.items.size():
		push_error("Invalid loose prop index"); quit(2); return
	var item: Dictionary = hub.atelier.loose_props.items[pick_index]
	var target: RigidBody3D = item.body
	if override_mass > 0.0: target.mass = override_mass
	var tip: Vector3 = beak.head.to_global(beak.TIP_LOCAL)
	var target_y: float = item.spawn.origin.y
	if duck.session.mode == "roller":
		# Roller head geometry cannot reach the floor; place the same prop on
		# a raised support. The robot ignores this test platform.
		var support := StaticBody3D.new()
		support.collision_layer = 4
		support.collision_mask = 1
		support.position = Vector3(tip.x,0.05,tip.z)
		var shape := CollisionShape3D.new()
		var box := BoxShape3D.new()
		box.size = Vector3(0.22,0.10,0.22)
		shape.shape = box
		support.add_child(shape)
		hub.add_child(support)
		target.collision_mask |= 4
		target_y = 0.10 + (0.035 if str(target.name).begins_with("Ball") else 0.002)
	target.global_position = Vector3(tip.x, target_y, tip.z)
	target.linear_velocity = Vector3.ZERO
	target.angular_velocity = Vector3.ZERO
	target.force_update_transform()
	hub.atelier.loose_props.selected = pick_index+1
	var initial_height: float = beak._center(target).y
	_send_key(KEY_H)
	var attached := false
	var max_open_radians := 0.0
	var moving_collision_active := false
	var carry_start := Vector2.ZERO
	for tick in range(1100):
		if carry_walk and tick in [820,1050]:
			var key := InputEventKey.new()
			key.physical_keycode = KEY_W
			key.keycode = KEY_W
			key.pressed = tick == 820
			Input.parse_input_event(key)
			if tick == 820: carry_start = Vector2(duck._base.global_position.x,duck._base.global_position.z)
		await physics_frame
		if beak.grip != null: attached = true
		max_open_radians = maxf(max_open_radians, absf(beak.pivot.rotation.z))
		if not beak.moving_shapes.is_empty() and not beak.moving_shapes[0].disabled: moving_collision_active = true
		if tick % 200 == 0:
			print("PICKABLE_PROBE ", JSON.stringify({"tick":tick,"tip_y":beak.head.to_global(beak.TIP_LOCAL).y,
				"object_y":beak._center(target).y,"gap":beak.head.to_global(beak.TIP_LOCAL).distance_to(beak._center(target)),
				"attached":attached,"phase":duck.brain.pick_phase,"message":beak.message}))
	var lift: float = beak.max_lift
	_send_key(KEY_X)
	for i in range(60): await physics_frame
	var result := {"attached":attached,"lift_m":lift,"released":beak.grip == null,
		"carry_distance_m":carry_start.distance_to(Vector2(duck._base.global_position.x,duck._base.global_position.z)) if carry_walk else 0.0,
		"max_open_radians":max_open_radians,"moving_collision_active":moving_collision_active,
		"closed_after_release":absf(beak.pivot.rotation.z) < 0.001 and not beak.lower_shapes[0].disabled,
		"initial_height_m":initial_height,"final_height_m":target.global_position.y,
		"policy_count":duck.bank.size(),"history":beak.history}
	print("PICKABLE_RESULT ", JSON.stringify(result))
	quit(0 if attached and lift > 0.03 and result.released and result.policy_count >= 9
		and max_open_radians > 0.1 and moving_collision_active and result.closed_after_release
		and (not carry_walk or result.carry_distance_m > 0.10) else 1)

func _send_key(code: Key) -> void:
	var key := InputEventKey.new()
	key.physical_keycode = code
	key.keycode = code
	key.pressed = true
	Input.parse_input_event(key)
