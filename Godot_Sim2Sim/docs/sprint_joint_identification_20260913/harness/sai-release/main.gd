extends Node3D
# The policy supplies joint/speed targets. Godot/Jolt owns every body state.
var robot
var peer := StreamPeerTCP.new()
var buffer := ""
var port := 19341
var test_case := ""
var output := ""
var screenshot := ""
var screenshot_at := .5
var captured := false
var cleared_at := -1.0
var task := "drive"
var cargo_obstacle_height := .018
var cargo_checks := 0
var cargo_outside_steps := 0
var cargo_unclamped_steps := 0
var max_transport_lateral := 0.0
var max_height := 0.0
var course_contacts: Dictionary={}
var duration := 12.0
var visuals := true
var riser := 0.0
var descending := false
var stair_tread := .18
var initial_yaw := 0.0
var records: Array = []
var input_events: Array = []
var injected: Dictionary = {}
var command: Dictionary = {}
var finished := false
var camera: Camera3D
var hud: Label
var sensor_views: Array = []
var specification: Dictionary

func _ready() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--port="):port=int(arg.split("=")[1])
		elif arg.begins_with("--case="):test_case=arg.split("=")[1]
		elif arg.begins_with("--output="):output=arg.substr(9)
		elif arg.begins_with("--screenshot="):screenshot=arg.substr(13)
		elif arg.begins_with("--screenshot-at="):screenshot_at=float(arg.split("=")[1])
		elif arg.begins_with("--duration="):duration=float(arg.split("=")[1])
		elif arg.begins_with("--stairs="):riser=float(arg.split("=")[1])
		elif arg.begins_with("--stair-tread="):stair_tread=float(arg.split("=")[1])
		elif arg.begins_with("--initial-yaw="):initial_yaw=float(arg.split("=")[1])
		elif arg.begins_with("--task="):task=arg.split("=")[1]
		elif arg.begins_with("--cargo-obstacle-height="):cargo_obstacle_height=float(arg.split("=")[1])
		elif arg=="--descending":descending=true
		elif arg=="--no-visuals":visuals=false
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
	# Rotate the complete articulated initial state before the first physics tick.
	# All joint anchors retain their original local frames; no runtime pose drive.
	if initial_yaw!=0.:
		var initial_basis := Basis(Vector3.UP,initial_yaw)
		for body in robot.bodies.values():
			body.position=initial_basis*body.position
			body.basis=initial_basis*body.basis
	if task=="cargo":robot.build_item()
	if visuals:build_view()
	peer.connect_to_host("127.0.0.1",port)
	var deadline := Time.get_ticks_msec()+10000
	while peer.get_status()!=StreamPeerTCP.STATUS_CONNECTED:
		peer.poll()
		if Time.get_ticks_msec()>deadline:
			push_error("Sai controller connection timeout")
			get_tree().quit(2)
			return
		OS.delay_usec(1000)
	peer.set_no_delay(true)
	print("SAI_GODOT_READY bodies=",robot.bodies.size()," joints=",robot.drives.size())

func box_surface(name_text: String,left: float,right: float,width: float,top: float,depth: float) -> void:
	var body := StaticBody3D.new()
	body.name=name_text
	body.collision_layer=2
	body.collision_mask=5
	body.position=Vector3((left+right)/2,top-depth/2,0)
	var pm := PhysicsMaterial.new()
	pm.friction=.8
	body.physics_material_override=pm
	add_child(body)
	var shape := BoxShape3D.new()
	shape.size=Vector3(right-left,depth,width)
	shape.margin=.0002
	var cs := CollisionShape3D.new()
	cs.shape=shape
	body.add_child(cs)
	if visuals:
		var mesh := BoxMesh.new()
		mesh.size=shape.size
		var mi := MeshInstance3D.new()
		mi.mesh=mesh
		mi.material_override=robot.material(Color(.57,.61,.64))
		body.add_child(mi)

func build_ground() -> void:
	box_surface("ground",-10,10,20,-.002 if riser>0 else 0.,.1)
	if task=="cargo":
		for i in range(3):
			var center: float=.55+.35*i
			box_surface("course_"+str(i),center-.045,center+.045,.9,cargo_obstacle_height,.08)
	if riser<=0:return
	var boundaries := [-1.,.45,.45+stair_tread,.45+2*stair_tread,.45+3*stair_tread,3.]
	for i in range(5):
		box_surface("stair_"+str(i),boundaries[i],boundaries[i+1],1.,riser*(4-i if descending else i),1.)

func inject_test_keys(time_s: float) -> void:
	var desired: Dictionary = {}
	if time_s>=1:
		if test_case in ["W","WA","W_shift"]:desired[KEY_W]=true
		if test_case=="S":desired[KEY_S]=true
		if test_case in ["A","WA"]:desired[KEY_A]=true
		if test_case=="D":desired[KEY_D]=true
		if test_case in ["shift","W_shift"] and time_s>=3 and time_s<8:desired[KEY_SHIFT]=true
	for key in [KEY_W,KEY_S,KEY_A,KEY_D,KEY_SHIFT]:
		var pressed: bool=desired.has(key)
		if pressed!=injected.get(key,false):
			var event := InputEventKey.new()
			event.keycode=key
			event.physical_keycode=key
			event.pressed=pressed
			Input.parse_input_event(event)
			injected[key]=pressed
			input_events.append({"time":time_s,"key":key,"pressed":pressed})

func movement_command() -> Array:
	var speed := .14 if test_case in ["WA","W_shift"] else .16
	var yaw_speed := .3 if test_case=="WA" else .45
	if cleared_at>=0:return [0.,0.,0.]
	return [speed*Input.get_axis("reverse","forward"),yaw_speed*Input.get_axis("right","left"),1. if Input.is_action_pressed("crouch") else 0.]

func height_scan() -> Array:
	var base: RigidBody3D=robot.bodies.chassis
	var direction: Vector3=base.global_basis*Vector3.RIGHT
	var yaw := atan2(-direction.z,direction.x)
	var heights: Array = []
	for x in [-.36,-.18,0.,.18,.36,.54,.72,.9]:
		for y in [-.24,0.,.24]:
			var sx: float=base.position.x+cos(yaw)*x-sin(yaw)*y
			var sy: float=-base.position.z+sin(yaw)*x+cos(yaw)*y
			var query := PhysicsRayQueryParameters3D.create(Vector3(sx,base.position.y+1.,-sy),Vector3(sx,base.position.y-1.,-sy),2)
			var hit := get_world_3d().direct_space_state.intersect_ray(query)
			heights.append(float(hit.position.y) if not hit.is_empty() else -.002)
	return heights

func exchange(state: Dictionary) -> Dictionary:
	peer.put_data((JSON.stringify(state)+"\n").to_utf8_buffer())
	var deadline := Time.get_ticks_msec()+5000
	while buffer.find("\n")<0:
		peer.poll()
		var count := peer.get_available_bytes()
		if count>0:
			var packet := peer.get_data(count)
			if packet[0]==OK:buffer+=packet[1].get_string_from_utf8()
		elif Time.get_ticks_msec()>deadline:
			push_error("Sai controller response timeout")
			get_tree().quit(2)
			return {}
		else:OS.delay_usec(50)
	var newline := buffer.find("\n")
	var line := buffer.substr(0,newline)
	buffer=buffer.substr(newline+1)
	return JSON.parse_string(line)

func _physics_process(_delta: float) -> void:
	if finished or robot==null or peer.get_status()!=StreamPeerTCP.STATUS_CONNECTED:return
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
	if robot.tick%40==0:
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
		var supported := 0
		for name in ["front_left_wheel","front_right_wheel","rear_left_wheel","rear_right_wheel"]:
			var wheel: RigidBody3D=robot.bodies[name]
			wheel_positions.append(robot.source(wheel.position))
			if not wheel.get_colliding_bodies().is_empty():supported+=1
		state["wheel_positions"]=wheel_positions
		state["wheels_supported"]=supported
		state["controller_stage"]=command.stage
		state["effective_crouch"]=command.get("effective_crouch",0.)
		state["stair_profile"]=command.get("stair_profile","")
		state["contract_id"]=command.get("contract_id","")
		if task=="cargo":
			var contacts: Array=[]
			for body in robot.item.get_colliding_bodies():contacts.append(str(body.name))
			var base: RigidBody3D=robot.bodies.chassis
			var local: Vector3=base.global_transform.affine_inverse()*robot.item.position+robot.gv(specification.bodies.chassis.origin_m)
			state["object_world_m"]=robot.source(robot.item.position)
			state["object_chassis_m"]=robot.source(local)
			state["contacts"]=contacts
			state["stage"]=command.stage
			state["FK_tool_error_m"]=command.FK_tool_error_m
			state["mode"]=command.mode
			state["belt_error_m"]=[state.q[22]-specification.cargo.drive_metres_per_radian*state.q[24],state.q[23]-specification.cargo.drive_metres_per_radian*state.q[24]]
			records.append(state)
			if float(state.time)>=float(command.end) or float(state.upright)<.6:
				finish_cargo()
				return
		if riser>0 and test_case=="W" and cleared_at<0:
			var cleared := true
			for position in wheel_positions:
				if position[0]<=.45+3*stair_tread+.08:cleared=false
			if cleared:cleared_at=float(state.time)
		if test_case!="":records.append(state)
		if (test_case!="" and float(state.time)>=duration) or float(state.upright)<.6 or (cleared_at>=0 and float(state.time)-cleared_at>=3):
			finish_run()
			return
	robot.apply_command(state,command)

func finish_run() -> void:
	if task=="cargo" and not records.is_empty():
		finish_cargo()
		return
	finished=true
	peer.put_data((JSON.stringify({"finish":true})+"\n").to_utf8_buffer())
	if output!="":
		var file := FileAccess.open(output,FileAccess.WRITE)
		file.store_string(JSON.stringify({"robot_id":"Sai_Agent_001","case":test_case,
			"physics":"Godot/Jolt","engine":Engine.get_version_info().string,
			"physics_hz":2000,"controller_hz":50,"body_count":robot.bodies.size(),
			"riser":riser,"descending":descending,"cleared_at":cleared_at,
			"tread":stair_tread,"initial_yaw":initial_yaw,
			"hinges":23,"sliders":2,"input_events":input_events,"samples":records}))
		file.close()
	print("SAI_GODOT_FINISHED seconds=",robot.tick*.0005)
	get_tree().quit()

func build_view() -> void:
	var light := DirectionalLight3D.new()
	light.rotation_degrees=Vector3(-55,-35,0)
	light.shadow_enabled=true
	light.light_energy=.7
	add_child(light)
	var environment := WorldEnvironment.new()
	environment.environment=Environment.new()
	environment.environment.background_mode=Environment.BG_COLOR
	environment.environment.background_color=Color(.72,.77,.82)
	environment.environment.ambient_light_source=Environment.AMBIENT_SOURCE_COLOR
	environment.environment.ambient_light_color=Color.WHITE
	environment.environment.ambient_light_energy=.22
	add_child(environment)
	camera=Camera3D.new()
	camera.fov=45
	add_child(camera)
	camera.current=true
	var canvas := CanvasLayer.new()
	add_child(canvas)
	hud=Label.new()
	hud.position=Vector2(20,16)
	hud.add_theme_color_override("font_color",Color(.05,.07,.1))
	hud.add_theme_font_size_override("font_size",20)
	canvas.add_child(hud)
	for i in range(specification.cameras.size()):
		var spec: Dictionary=specification.cameras[i]
		var viewport := SubViewport.new()
		viewport.size=Vector2i(320,180)
		viewport.world_3d=get_world_3d()
		viewport.render_target_update_mode=SubViewport.UPDATE_ALWAYS
		add_child(viewport)
		var sensor := Camera3D.new()
		sensor.fov=spec.pinhole_fovy_deg
		sensor.near=.005
		viewport.add_child(sensor)
		sensor.current=true
		var rect := TextureRect.new()
		rect.texture=viewport.get_texture()
		rect.position=Vector2(940,20+i*200)
		rect.size=Vector2(320,180)
		canvas.add_child(rect)
		sensor_views.append({"camera":sensor,"spec":spec})

func _process(_delta: float) -> void:
	if camera==null or robot==null:return
	var base: RigidBody3D=robot.bodies.chassis
	camera.position=base.position+Vector3(.62,.36,.60)
	camera.look_at(base.position+Vector3(0,.05,0))
	hud.text="Sai_Agent_001\nW/S drive · A/D turn · Shift crouch · R restart · Esc quit\n"+str(command.get("stage","connecting"))
	if task=="cargo":hud.text="Sai_Agent_001\nPickup · secure cargo · transport · R restart · Esc quit\n"+str(command.get("stage","connecting"))
	for view in sensor_views:
		var spec: Dictionary=view.spec
		var body: RigidBody3D=robot.bodies[spec.body]
		var position: Vector3=body.global_transform*(robot.gv(spec.optical_center_m)-robot.gv(specification.bodies[spec.body].origin_m))
		view.camera.global_position=position
		view.camera.look_at(position+body.global_basis*robot.gv(spec.forward),body.global_basis*robot.gv(spec.up))
	if Input.is_key_pressed(KEY_ESCAPE):finish_run()
	if screenshot!="" and not captured and robot.tick*.0005>screenshot_at:
		captured=true
		await RenderingServer.frame_post_draw
		get_viewport().get_texture().get_image().save_png(screenshot)

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
		wheel_edge=minf(wheel_edge,center.x-extent)
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
	get_tree().quit(0 if result.success else 3)

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.physical_keycode==KEY_R and event.pressed and not event.echo:
		finished=true
		if peer.get_status()==StreamPeerTCP.STATUS_CONNECTED:
			peer.put_data((JSON.stringify({"finish":true})+"\n").to_utf8_buffer())
		# The launcher recreates both processes' state, including recurrent
		# action history, crouch slew, course, cargo and native Jolt joints.
		get_tree().quit(75)
