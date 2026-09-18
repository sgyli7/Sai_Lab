extends "res://sai_release/main.gd"
var hub: Node3D
var settings: Dictionary
var grab: Node3D
var native_controller
var native_mode := true
var payload_start_reset := false

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
	native_mode = hub.options.get("sai_controller","native") == "native"
	if native_mode:
		_ready_native()
	else:
		super._ready()
	if hub.active_task in ["drive", "sort"]:
		grab = load("res://hub/scene_grab.gd").new()
		grab.scene = self
		add_child(grab)
		if native_mode:
			native_controller=preload("res://sai/native_grab_controller.gd").new(native_controller,specification)
	if visuals:
		var paint = load("res://hub/sai_materials.gd").new()
		paint.scene = self
		paint._make_materials()
		paint._paint_robot()
		paint.free()
	if test_case != "": add_child(load("res://hub/replay_input.gd").new())

func _ready_native() -> void:
	for pair in [["forward",KEY_W],["reverse",KEY_S],["left",KEY_A],["right",KEY_D],["crouch",KEY_SHIFT]]:
		InputMap.add_action(pair[0])
		var event := InputEventKey.new()
		event.physical_keycode=pair[1]
		InputMap.action_add_event(pair[0],event)
	specification=JSON.parse_string(FileAccess.get_file_as_string("res://sai_agent/robot.json"))
	robot=load("res://robot.gd").new()
	add_child(robot)
	build_ground()
	robot.setup(specification,visuals,4*riser if descending else 0.0)
	if task=="cargo" or bool(settings.get("payload",false)):
		robot.build_item()
		if task!="cargo":
			# Free 100 g validation payload, initialized on the open rear tray.
			robot.item.position=robot.gv([-.09,0.,.284+4*riser if descending else .284])
	if initial_yaw!=0.0:
		var initial_basis := Basis(Vector3.UP,initial_yaw)
		for body in robot.bodies.values():
			body.position=initial_basis*body.position
			body.basis=initial_basis*body.basis
	var profile := ""
	if not str(settings.get("skill","")).is_empty():
		profile="res://sai_policy/experimental/"+str(settings.skill)+".json"
	native_controller=preload("res://sai/native_controller.gd").new(profile)
	if not native_controller.last_error.is_empty():
		push_error("Sai native controller startup failed: "+native_controller.last_error)
		get_tree().quit(2)
		return
	if task=="cargo":native_controller=preload("res://sai/native_cargo_controller.gd").new(native_controller,specification)
	print("SAI_NATIVE_READY backend=onnxruntime flat=",native_controller.flat_policy_sha256,
		" stairs=",native_controller.stair_profile_id," bodies=",robot.bodies.size()," joints=",robot.drives.size())

func build_ground() -> void:
	# The persistent workshop owns the terrain; the adapter changes only leg torque.
	robot.set_script(preload("res://sai/compliant_robot.gd"))

func build_view() -> void:
	# The hub owns the camera, interface and movie capture throughout switches.
	pass

func _process(_delta: float) -> void:
	pass

func _unhandled_input(_event: InputEvent) -> void:
	pass

func movement_command() -> Array:
	var request: Array = super.movement_command()
	if bool(settings.get("payload",false)) and robot.sim_time_seconds()<float(settings.get("payload_settle_seconds",8.0)):
		return [0.0,0.0,0.0]
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
	state["terrain_edge_heights"] = preload("res://sai/terrain_scan.gd").edge_profile(self, hub.TERRAIN_LAYER)
	state["wheel_ground_heights"] = preload("res://sai/terrain_scan.gd").wheel_ground(self, hub.TERRAIN_LAYER)
	state["world_origin"] = robot.source(global_position)
	state["experimental_profile"] = not str(settings.get("skill","")).is_empty()
	# The accepted flat-motion actor and suspension-v2 are deployed together;
	# descents use their contact-following branch instead of the stair actor.
	state["contact_following_descent"] = true
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
	state["arm_gravity_bias"] = _arm_gravity_bias()
	state.merge(_leg_support_state(),true)
	var result: Dictionary = native_controller.command(state) if native_mode else super.exchange(state)
	if grab != null: grab.accept(result)
	return result

func _arm_gravity_bias() -> Array:
	# Jolt-native inverse statics: for each arm hinge, cancel the gravity
	# moment of its actual descendant rigid bodies around the actual joint axis.
	# This consumes engine transforms/COMs and therefore cannot drift from the
	# collision articulation through a parallel MuJoCo FK model.
	var result: Array=[]
	for joint_index in range(16,22):
		var joint: Dictionary=robot.drives[joint_index]
		var origin: Vector3=joint.child.global_position
		var axis: Vector3=(joint.parent.global_basis*joint.axis).normalized()
		var gravity_moment:=Vector3.ZERO
		for body_index in range(joint_index,22):
			var body: RigidBody3D=robot.drives[body_index].child
			var com:=body.global_transform*body.center_of_mass
			gravity_moment+=(com-origin).cross(Vector3.DOWN*body.mass*9.81)
		result.append(-axis.dot(gravity_moment))
	return result

func _leg_support_state() -> Dictionary:
	var total_mass:=0.0
	var weighted_com:=Vector3.ZERO
	for body in robot.bodies.values():
		# robot.state() and wheel_positions use the actor-local release frame.
		# Keep the aggregate CoM in that same frame; global coordinates include
		# the science-station checkpoint offset and corrupt support allocation.
		var com:Vector3=body.transform*body.center_of_mass
		total_mass+=body.mass
		weighted_com+=com*body.mass
	weighted_com/=total_mass
	var jacobians:Array=[]
	var gravity_bias:Array=[]
	gravity_bias.resize(16);gravity_bias.fill(0.0)
	for leg in range(4):
		var jac:Array=[];jac.resize(16);jac.fill(0.0)
		var wheel:RigidBody3D=robot.bodies[str(specification.leg_order[leg])+"_wheel"]
		for offset in range(3):
			var joint_index:=leg*4+offset
			var joint:Dictionary=robot.drives[joint_index]
			var origin:Vector3=joint.child.global_position
			var axis:Vector3=(joint.parent.global_basis*joint.axis).normalized()
			jac[joint_index]=axis.cross(wheel.global_position-origin).y
			var gravity_moment:=Vector3.ZERO
			for body_index in range(joint_index,leg*4+4):
				var body:RigidBody3D=robot.drives[body_index].child
				var com:Vector3=body.global_transform*body.center_of_mass
				gravity_moment+=(com-origin).cross(Vector3.DOWN*body.mass*9.81)
			gravity_bias[joint_index]=-axis.dot(gravity_moment)
		jacobians.append(jac)
	return {"robot_mass":total_mass,"robot_com_position":robot.source(weighted_com),
		"leg_support_jacobian":jacobians,"leg_gravity_bias":gravity_bias}

func _physics_process(_delta: float) -> void:
	if not native_mode:
		super._physics_process(_delta)
		return
	if finished or robot==null or native_controller==null:return
	if bool(settings.get("payload",false)) and not payload_start_reset and robot.sim_time_seconds()>=float(settings.get("payload_settle_seconds",8.0)):
		_reset_loaded_start()
	var state: Dictionary=robot.state()
	if task=="cargo":
		max_height=maxf(max_height,robot.item.position.y)
		if command.get("mode","")=="transport":
			cargo_checks+=1
			if not state.cargo_inside:cargo_outside_steps+=1
			if not state.cargo_bilateral:cargo_unclamped_steps+=1
			max_transport_lateral=maxf(max_transport_lateral,absf(float(state.base_position[1])))
			for key in specification.leg_order:
				for other in robot.bodies[str(key)+"_wheel"].get_colliding_bodies():
					if str(other.name).begins_with("course_"):course_contacts[str(other.name)]=true
	if robot.is_control_tick():
		if test_case!="":inject_test_keys(float(state.time))
		state["robot_id"]="Sai_Agent_001"
		state["command"]=movement_command()
		state["terrain_heights"]=height_scan()
		state["physics_owner"]="Godot/Jolt"
		command=exchange(state)
		if command.is_empty():return
		state["policy_action"]=command.get("policy_action",[])
		state["policy_observation"]=command.get("policy_observation",[])
		state["upright"]=robot.bodies.chassis.global_basis.y.y
		var wheel_positions: Array=[]
		var supported:=0
		for name in ["front_left_wheel","front_right_wheel","rear_left_wheel","rear_right_wheel"]:
			var wheel: RigidBody3D=robot.bodies[name]
			wheel_positions.append(robot.source(wheel.position))
			if not wheel.get_colliding_bodies().is_empty():supported+=1
		state["wheel_positions"]=wheel_positions
		state["wheels_supported"]=supported
		state["controller_stage"]=command.stage
		state["effective_crouch"]=command.get("effective_crouch",0.0)
		state["stair_profile"]=command.get("stair_profile","")
		state["contract_id"]=command.get("contract_id","")
		state["controller_backend"]=command.get("controller_backend","")
		if robot.item!=null:
			var base:RigidBody3D=robot.bodies.chassis
			var local:Vector3=base.global_transform.affine_inverse()*robot.item.global_position+robot.gv(specification.bodies.chassis.origin_m)
			state["object_world_m"]=robot.source(robot.item.position)
			state["object_chassis_m"]=robot.source(local)
			state["object_linear_world"]=robot.source(robot.item.linear_velocity)
		if task=="cargo":
			var contacts:Array=[]
			for body in robot.item.get_colliding_bodies():contacts.append(str(body.name))
			var base:RigidBody3D=robot.bodies.chassis
			var local:Vector3=base.global_transform.affine_inverse()*robot.item.global_position+robot.gv(specification.bodies.chassis.origin_m)
			state["object_world_m"]=robot.source(robot.item.position);state["object_chassis_m"]=robot.source(local)
			state["contacts"]=contacts;state["stage"]=command.stage;state["mode"]=command.mode
			state["belt_error_m"]=[state.q[22]-specification.cargo.drive_metres_per_radian*state.q[24],state.q[23]-specification.cargo.drive_metres_per_radian*state.q[24]]
			records.append(state)
			if float(state.time)>=float(command.end) or float(state.upright)<.6:
				finish_cargo();return
		# Timed plans are the integration-test seam for the interactive controller.
		# Keep the exact 50 Hz state and command together so steering drift and
		# articulation jitter cannot be hidden by the coarser hub telemetry.
		if task!="cargo" and (test_case!="" or not hub.options.plan.is_empty()):
			state["controller_command"] = command.duplicate(true)
			records.append(state)
		if riser>0.0 and test_case=="W" and cleared_at<0.0:
			var cleared:=true
			for position in wheel_positions:
				if position[0]<=.45+3*stair_tread+.08:cleared=false
			if cleared:cleared_at=float(state.time)
		if (test_case!="" and float(state.time)>=duration) or float(state.upright)<.6 or (cleared_at>=0.0 and float(state.time)-cleared_at>=3.0):
			finish_run()
			return
	robot.apply_command(state,command)

func _reset_loaded_start()->void:
	var base:RigidBody3D=robot.bodies.chassis
	var desired:=Transform3D(Basis.IDENTITY,Vector3(0.0,base.position.y,0.0))
	var delta:=desired*base.transform.affine_inverse()
	for body in robot.bodies.values():
		body.transform=delta*body.transform
		body.linear_velocity=Vector3.ZERO
		body.angular_velocity=Vector3.ZERO
	if robot.item!=null:
		robot.item.transform=delta*robot.item.transform
		robot.item.linear_velocity=Vector3.ZERO
		robot.item.angular_velocity=Vector3.ZERO
	payload_start_reset=true
	native_controller.reset()

func finish_run() -> void:
	finished = true
	if not native_mode: peer.put_data((JSON.stringify({"finish":true})+"\n").to_utf8_buffer())
	var result := {"task":hub.active_task,"physics":"Godot/Jolt","engine":Engine.get_version_info().string,
		"physics_hz":Engine.physics_ticks_per_second,"controller_hz":50,"control_decimation":robot.control_decimation(),"body_count":robot.bodies.size(),"hinges":23,"sliders":2,
		"riser":riser,"descending":descending,"cleared_at":cleared_at,"tread":stair_tread,
		"input_events":input_events,"samples":records,"duration_s":robot.sim_time_seconds(),
		"controller_backend":"godot-native-onnxruntime" if native_mode else "python-tcp-oracle"}
	hub.on_task_finished(result)

# Release alpha.3 cargo checks; only world-origin correction and completion routing differ.
func finish_cargo() -> void:
	finished=true
	if not native_mode:peer.put_data((JSON.stringify({"finish":true})+"\n").to_utf8_buffer())
	var last: Dictionary=records[-1]
	var p: Array=last.object_chassis_m
	var placed: bool=last.cargo_supported
	var bilateral := 0
	for r in records:
		if "arm_gripper" in r.contacts and "arm_moving_jaw" in r.contacts:bilateral+=1
	var result := {"engine":Engine.get_version_info().string,"physics":ProjectSettings.get_setting("physics/3d/physics_engine"),"independent_physics":true,
		"controller_backend":"godot-native-onnxruntime" if native_mode else "python-tcp-oracle",
		"jolt_penetration_slop_m":ProjectSettings.get_setting("physics/jolt_physics_3d/simulation/penetration_slop"),
		"jolt_speculative_contact_distance_m":ProjectSettings.get_setting("physics/jolt_physics_3d/simulation/speculative_contact_distance"),
		"physics_hz":Engine.physics_ticks_per_second,"controller_hz":50,"control_decimation":robot.control_decimation(),"body_count":robot.bodies.size(),"joint_count":robot.drives.size(),"hinge_count":23,"slider_count":2,
		"max_object_height_m":max_height,"two_finger_contact_samples":bilateral,"placed_in_cargo":placed,
		"success":placed and max_height>.20 and bilateral>10 and robot.sim_time_seconds()>=float(command.end),
		"final_object_chassis_m":p,"duration_s":robot.sim_time_seconds(),"samples":records}
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
