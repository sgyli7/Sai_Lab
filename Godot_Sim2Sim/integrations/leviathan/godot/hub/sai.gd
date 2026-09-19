extends "res://sai_release/main.gd"
var hub: Node3D
var settings: Dictionary
var grab: Node3D
var support_observer:RefCounted

func _ready() -> void:
	settings = hub.task_settings()
	task = settings.task
	riser = settings.get("riser",0.)
	descending = settings.get("descending",false)
	cargo_obstacle_height = settings.get("obstacle",.018)
	duration = 45.
	visuals = not hub._headless
	if hub.options.plan.get("auto_drive",false): test_case = "W"
	for action in ["forward","reverse","left","right","crouch"]:
		if InputMap.has_action(action): InputMap.erase_action(action)
	super._ready()
	if hub.active_task in ["drive", "sort"] and not hub.atelier.loose_props.items.is_empty():
		grab = load("res://hub/scene_grab.gd").new()
		grab.scene = self
		add_child(grab)
	if visuals:
		var paint = load("res://hub/sai_materials.gd").new()
		paint.scene = self
		paint._make_materials()
		paint._paint_robot()
		paint.free()
	if test_case != "": add_child(load("res://hub/replay_input.gd").new())

func build_ground() -> void:
	# The persistent workshop owns all floor and task colliders.
	pass

func build_view() -> void:
	# The hub owns the camera, interface and movie capture throughout switches.
	pass

func _process(_delta: float) -> void:
	pass

func _unhandled_input(_event: InputEvent) -> void:
	pass

func movement_command() -> Array:
	if hub.driving_leviathan():return [0.,0.,0.]
	var request: Array = super.movement_command()
	var speed: float = float(hub.options.get("drive_speed", .5)) if task == "drive" and riser <= 0.0 else .16
	return preload("res://sai/driving_input.gd").vehicle_command(request, speed)

func height_scan() -> Array:
	var base: RigidBody3D = robot.bodies.chassis
	var direction: Vector3 = base.global_basis * Vector3.RIGHT
	var yaw := atan2(-direction.z,direction.x)
	var heights: Array = []
	for x in [-.36,-.18,0.,.18,.36,.54,.72,.9]:
		for y in [-.24,0.,.24]:
			var sx: float = base.global_position.x + cos(yaw)*x - sin(yaw)*y
			var sz: float = base.global_position.z - sin(yaw)*x - cos(yaw)*y
			var query := PhysicsRayQueryParameters3D.create(Vector3(sx,base.global_position.y+1.,sz),Vector3(sx,base.global_position.y-1.,sz),hub.TERRAIN_LAYER)
			var hit := get_world_3d().direct_space_state.intersect_ray(query)
			heights.append(float(hit.position.y)-global_position.y if not hit.is_empty() else -.002)
	return heights

func exchange(state: Dictionary) -> Dictionary:
	state["hub_config"] = settings
	state["stair_course"] = riser > 0.0
	state["terrain_path_heights"] = preload("res://sai/terrain_scan.gd").wheel_path(self, hub.TERRAIN_LAYER)
	state["world_origin"] = robot.source(global_position)
	# Record actual wheel state in the same 50 Hz packet as the policy input.
	# These diagnostics do not participate in observation construction/control.
	var wheel_positions: Array = []
	var supported := 0
	for leg in specification.leg_order:
		var wheel: RigidBody3D = robot.bodies[str(leg)+"_wheel"]
		wheel_positions.append(robot.source(wheel.position))
		if not wheel.get_colliding_bodies().is_empty(): supported += 1
	state["wheel_positions"] = wheel_positions
	state["wheels_supported"] = supported
	# The release FK/task operates in its original local metre frame.
	var offset: Array = robot.source(global_position)
	for i in range(3): state.tool_m[i] -= offset[i]
	if grab != null: state["workshop_grab"] = grab.observation()
	if hub.leviathan!=null:
		if support_observer==null:support_observer=load("res://hub/leviathan_support.gd").new()
		var frames:Array=[]
		if command.has("task_support_body_id"):frames.append(command.task_support_body_id)
		if state.has("workshop_grab"):frames.append(state.workshop_grab.get("target_support_body_id","world"))
		support_observer.observe(hub.leviathan,robot,specification,state,frames)
	var result := super.exchange(state)
	if grab != null: grab.accept(result)
	return result

func finish_run() -> void:
	finished = true
	peer.put_data((JSON.stringify({"finish":true})+"\n").to_utf8_buffer())
	var result := {"task":hub.active_task,"physics":"Godot/Jolt","engine":Engine.get_version_info().string,
		"physics_hz":2000,"controller_hz":50,"body_count":robot.bodies.size(),"hinges":23,"sliders":2,
		"riser":riser,"descending":descending,"cleared_at":cleared_at,"tread":stair_tread,
		"input_events":input_events,"samples":records,"duration_s":robot.tick*.0005}
	hub.on_task_finished(result)

# Release alpha.3 cargo checks; only world-origin correction and completion routing differ.
func finish_cargo() -> void:
	finished=true
	peer.put_data((JSON.stringify({"finish":true})+"\n").to_utf8_buffer())
	var last: Dictionary=records[-1]
	var p: Array=last.object_chassis_m
	var placed: bool=last.cargo_supported
	var bilateral := 0
	for r in records:
		if "arm_gripper" in r.contacts and "arm_moving_jaw" in r.contacts:bilateral+=1
	var result := {"engine":Engine.get_version_info().string,"physics":ProjectSettings.get_setting("physics/3d/physics_engine"),"independent_physics":true,
		"jolt_penetration_slop_m":ProjectSettings.get_setting("physics/jolt_physics_3d/simulation/penetration_slop"),
		"jolt_speculative_contact_distance_m":ProjectSettings.get_setting("physics/jolt_physics_3d/simulation/speculative_contact_distance"),
		"physics_hz":2000,"controller_hz":50,"body_count":robot.bodies.size(),"joint_count":robot.drives.size(),"hinge_count":23,"slider_count":2,
		"max_object_height_m":max_height,"two_finger_contact_samples":bilateral,"placed_in_cargo":placed,
		"success":placed and max_height>.20 and bilateral>10 and robot.tick*0.0005>=float(command.end),
		"final_object_chassis_m":p,"duration_s":robot.tick*0.0005,"samples":records}
	if command.get("transport_required",false):
		result["transport_distance_m"]=command.transport_distance_m
		result["transport_policy_sha256"]=command.policy_sha256
		result.success=result.success and command.transport_distance_m>.7
	var wheel_edge := INF
	for key in specification.leg_order:
		var b: RigidBody3D=robot.bodies[str(key)+"_wheel"]
		var axis_x: float=(b.global_basis*robot.gv([0,1,0])).x
		var extent: float=0.048*sqrt(maxf(0.0,1.0-axis_x*axis_x))+0.016*absf(axis_x)
		var center: Vector3=b.global_transform*robot.gv(specification.bodies[str(key)+"_wheel"].collision[0].pos)
		wheel_edge=minf(wheel_edge,center.x-global_position.x-extent)
	result["cargo_obstacle_height_m"]=cargo_obstacle_height
	result["actual_wheel_course_contacts"]=course_contacts.keys()
	result["rearmost_wheel_edge_m"]=wheel_edge
	result["physics_cargo_checks"]=cargo_checks
	result["outside_cargo_steps"]=cargo_outside_steps
	result["unclamped_steps"]=cargo_unclamped_steps
	result["max_transport_lateral_m"]=max_transport_lateral
	result["cargo_coupling"]="force-level elastic belt, 20000 N/m and 4 Ns/m per branch; uncalibrated approximation"
	result["success"]=result.success and cargo_checks>0 and cargo_outside_steps==0 and cargo_unclamped_steps==0 and wheel_edge>1.295 and course_contacts.size()==3 and max_transport_lateral<0.30 and command.mode!="abort"
	if output!="":
		var file := FileAccess.open(output,FileAccess.WRITE)
		file.store_string(JSON.stringify(result))
		file.close()
	print("GODOT_TASK_COMPLETE success=",result.success," item=",p," bilateral=",bilateral)
	hub.on_task_finished(result)
