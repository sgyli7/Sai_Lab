extends Node3D
## Gameplay assistance: IK moves the arm; a nearby object may be constrained to
## the gripper. Jolt moves the object, and release restores free rigid-body motion.
## This is separate from the release's contact-only cargo demonstration.
var scene: Node3D
var selected := 0
var serial := 0
var request := "idle"
var busy := false
var target: RigidBody3D
var grip: PinJoint3D
var marker: Label3D
var message := "B 选择物件，G 靠近并抓取 → 蓝色货仓"
var deliveries: Array[Dictionary] = []
var history: Array[Dictionary] = []
var released_at := -1.0
var cleanup_body: RigidBody3D
var grasp_start_height := 0.0
var max_lift := 0.0
var slot := 0
var captured := false
var released := false
var target_can_sleep := true

func _ready() -> void:
	_choose_nearest()
	if not scene.hub._headless:
		marker = Label3D.new()
		marker.font = load("res://atelier/ui_font.tres")
		marker.font_size = 48
		marker.pixel_size = .0006
		marker.modulate = Color("e5be32")
		marker.outline_modulate = Color("37333c")
		marker.outline_size = 6
		marker.billboard = BaseMaterial3D.BILLBOARD_ENABLED
		add_child(marker)

func _items() -> Array:
	return scene.hub.atelier.loose_props.items

func _body() -> RigidBody3D:
	return _items()[selected].body

func _center(body: RigidBody3D) -> Vector3:
	return body.global_transform * body.get_meta("grasp_center",Vector3.ZERO)

func _choose_nearest() -> void:
	var nearest := INF
	for i in range(_items().size()):
		var distance: float = _center(_items()[i].body).distance_to(scene.robot.bodies.chassis.global_position)
		if distance < nearest: nearest = distance; selected = i

func _name(body: RigidBody3D) -> String:
	var kind := "收纳盒" if str(body.name).begins_with("Bin") else "小瓶" if str(body.name).begins_with("Bottle") else "小球"
	return "%s · %.0f g" % [kind,body.mass*1000.]

func _cargo_bounds(body: RigidBody3D) -> Array:
	var base: RigidBody3D = scene.robot.bodies.chassis
	var inverse := base.global_transform.affine_inverse()
	var origin: Vector3 = scene.robot.gv(scene.specification.bodies.chassis.origin_m)
	var low := Vector3(INF,INF,INF)
	var high := Vector3(-INF,-INF,-INF)
	for child in body.get_children():
		if not child is CollisionShape3D: continue
		var points: PackedVector3Array = child.shape.points if child.shape is ConvexPolygonShape3D else child.shape.get_debug_mesh().get_faces()
		for point in points:
			var local: Vector3 = inverse*(child.global_transform*point)+origin
			low = low.min(local)
			high = high.max(local)
	return [low,high]

func _in_cargo(body: RigidBody3D) -> bool:
	var bounds := _cargo_bounds(body)
	var low: Vector3 = bounds[0]
	var high: Vector3 = bounds[1]
	return low.x >= -.149 and high.x <= -.034 and low.z >= -.115 and high.z <= .115 and low.y > .253 and high.y < .45

func _supported(body: RigidBody3D) -> bool:
	for other in body.get_colliding_bodies():
		if other == scene.robot.bodies.chassis or str(other.name).begins_with("cargo_slide_"): return true
	return false

func perform(action: String) -> void:
	if action == "cycle":
		if busy: return
		if scene.hub.is_science_station():
			for offset in range(1,_items().size()+1):
				var index:int=(selected+offset)%_items().size()
				if _center(_items()[index].body).distance_to(scene.robot.bodies.chassis.global_position)<=1.6:
					selected=index;break
		else:
			selected = (selected+1)%_items().size()
		message = "G 靠近并抓取 → 蓝色货仓"
	elif action == "cancel":
		if not busy: return
		_detach()
		serial += 1
		request = "cancel"
		busy = false
		_restore_sleep()
		message = "已取消 · 物件已松开"
		history.append({"event":"cancel","time":scene.robot.sim_time_seconds()})
	elif action == "pick":
		if busy: return
		if cleanup_body != null:
			message = "机械臂正在退出货仓，请稍候"; return
		target = _body()
		if _in_cargo(target): message = "所选物件已在货仓 · B 选择其他物件"; return
		if _center(target).distance_to(scene.robot.bodies.chassis.global_position) > 1.6:
			message = "物件太远 · 请先驾驶靠近（1.6 m 内）"; return
		# Two bays clear the side clamp pads even for the 70 mm balls.
		# Recheck physical occupancy after movement/switches.
		var occupied := [false,false]
		for item in _items():
			if _in_cargo(item.body):
				var local: Vector3 = scene.robot.bodies.chassis.to_local(_center(item.body))
				occupied[0 if local.z >= 0. else 1] = true
		slot = occupied.find(false)
		if slot < 0: message = "货仓已满 · 0 归位后可继续练习"; return
		_cleanup_collisions()
		target_can_sleep = target.can_sleep
		target.can_sleep = false
		target.sleeping = false
		serial += 1
		request = "pick"
		busy = true
		captured = false
		released = false
		grasp_start_height = _center(target).y
		max_lift = 0.
		message = "正在靠近物件 · X 取消"
		history.append({"event":"start","object":str(target.name),"id":target.get_instance_id(),"time":scene.robot.sim_time_seconds()})

func observation() -> Dictionary:
	var body: RigidBody3D = target if is_instance_valid(target) else _body()
	var center := _center(body)-scene.global_position
	var tool: Vector3 = scene.robot.bodies.arm_gripper.global_transform*scene.robot.gv(scene.specification.tool_local_m)
	var rest_height: float = body.get_meta("grasp_size").y*.5 if str(body.name).begins_with("Ball") else body.get_meta("grasp_center").y
	return {"serial":serial,"request":request,"target_m":scene.robot.source(center),
		"rest_height_m":rest_height,
		"hand_contacts_blocked":body.get_collision_exceptions().has(scene.robot.bodies.arm_gripper),
		"held_offset_m":scene.robot.source(_center(body)-tool),
		"object":str(body.name),"held":grip != null,"slot":slot,"busy":busy}

func accept(command: Dictionary) -> void:
	if not busy: return
	max_lift = maxf(max_lift,_center(target).y-grasp_start_height)
	var stage: String = command.get("grab_stage","")
	if command.get("assist_grip",false) and grip == null and not captured:
		var tool: Vector3 = scene.robot.bodies.arm_gripper.global_transform*scene.robot.gv(scene.specification.tool_local_m)
		# Never snap a remote body into the hand. The IK must first reach it.
		var distance := tool.distance_to(_center(target))
		var touching := false
		for other in target.get_colliding_bodies():
			if other in [scene.robot.bodies.arm_gripper,scene.robot.bodies.arm_moving_jaw]: touching = true
		if distance < .026 or (touching and distance < .06):
			_attach()
	if command.get("assist_release",false) and grip != null:
		_detach()
		released = true
	if stage == "complete" or stage == "failed":
		_detach()
		var placed := captured and released and max_lift > .08 and _in_cargo(target) and _supported(target)
		busy = false
		request = "idle"
		message = "已放入蓝色货仓 · B 选下一件" if placed else command.get("grab_message","未放稳 · 请靠近物件后重试")
		var result := {"event":"complete","object":str(target.name),"id":target.get_instance_id(),
			"success":placed,"captured":captured,"released":released,"lift_m":max_lift,"time":scene.robot.sim_time_seconds()}
		var bounds := _cargo_bounds(target)
		result["cargo_bounds"] = [[bounds[0].x,bounds[0].y,bounds[0].z],[bounds[1].x,bounds[1].y,bounds[1].z]]
		result["supported"] = _supported(target)
		deliveries.append(result)
		history.append(result)
		print("WORKSHOP_GRAB ",JSON.stringify(result))
		_restore_sleep()
	else:
		message = command.get("grab_message",scene.hub._stage_label(command.get("stage",""))) + " · X 取消"

func _attach() -> void:
	# A point grip leaves object rotation free, avoiding forced wrist rotation
	# of bottles/boxes into the cargo rim. The object still swings under gravity.
	grip = PinJoint3D.new()
	grip.name = "AssistedSceneGrip"
	# Own the exceptions explicitly, including after the joint is removed.
	grip.exclude_nodes_from_collision = false
	add_child(grip)
	grip.global_position = _center(target)
	grip.node_a = grip.get_path_to(scene.robot.bodies.arm_gripper)
	grip.node_b = grip.get_path_to(target)
	for key in ["arm_gripper","arm_moving_jaw"]:
		target.add_collision_exception_with(scene.robot.bodies[key])
	captured = true
	history.append({"event":"attach","object":str(target.name),"time":scene.robot.sim_time_seconds()})

func _detach() -> void:
	if grip == null: return
	grip.free()
	grip = null
	# Joint3D removes the body's pair exceptions when leaving the tree, even
	# when we added them ourselves. Reapply them until the hand clears the prop.
	if is_instance_valid(scene.robot):
		for key in ["arm_gripper","arm_moving_jaw"]:
			if is_instance_valid(scene.robot.bodies[key]): target.add_collision_exception_with(scene.robot.bodies[key])
	cleanup_body = target
	released_at = scene.hub._t
	history.append({"event":"release","object":str(target.name),"time":released_at})

func _cleanup_collisions() -> void:
	if not is_instance_valid(cleanup_body): return
	if is_instance_valid(scene.robot):
		for key in ["arm_gripper","arm_moving_jaw"]:
			if is_instance_valid(scene.robot.bodies[key]):
				cleanup_body.remove_collision_exception_with(scene.robot.bodies[key])
	cleanup_body = null

func _restore_sleep() -> void:
	if is_instance_valid(target): target.can_sleep = target_can_sleep

func _process(_delta: float) -> void:
	if cleanup_body != null and scene.robot.sim_time_seconds()-released_at > 1.0:
		var tool: Vector3 = scene.robot.bodies.arm_gripper.global_transform*scene.robot.gv(scene.specification.tool_local_m)
		# Restore hand contacts after the retreat clears the larger scene props.
		if tool.distance_to(_center(cleanup_body)) > .15: _cleanup_collisions()
	if marker != null:
		marker.global_position = _center(_body())+Vector3(0,.14,0)
		marker.text = "▼ " + _name(_body())
		marker.visible = not scene.hub.options.plan.get("cinematic",false) and (not scene.hub.is_science_station() or _center(_body()).distance_to(scene.robot.bodies.chassis.global_position)<=1.6)

func _physics_process(_delta: float) -> void:
	if grip == null or str(target.name).begins_with("Ball"): return
	# A bounded game-assist torque keeps tall/open props upright while held.
	# No pose assignment; Jolt integrates this torque and all contact forces.
	var base: RigidBody3D = scene.robot.bodies.chassis
	var forward := base.global_basis.x
	var yaw := atan2(-forward.z,forward.x)
	var desired := Basis(Vector3.UP,yaw)
	var error := (desired*target.global_basis.transposed()).get_rotation_quaternion()
	var torque := error.get_axis()*wrapf(error.get_angle(),-PI,PI)*target.mass*.12-target.angular_velocity*target.mass*.006
	target.apply_torque(torque.limit_length(target.mass*.18))

func status_text() -> String:
	return "目标：%s → 蓝色货仓\n%s" % [_name(_body()),message]

func retention() -> Dictionary:
	var result := {}
	for item in _items(): result[str(item.body.name)] = _in_cargo(item.body)
	return result

func _exit_tree() -> void:
	_detach()
	_cleanup_collisions()
	_restore_sleep()
