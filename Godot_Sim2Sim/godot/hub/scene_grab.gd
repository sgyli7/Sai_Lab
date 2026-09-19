extends Node3D
## Contact-only scene pickup. Jolt integrates the free item and both gripper jaws.
## No proximity constraint or pose assignment is allowed during the attempt.
var scene: Node3D
var selected := 0
var serial := 0
var request := "idle"
var busy := false
var target: RigidBody3D
var held := false
var bilateral_frames := 0
var max_bilateral_run := 0
var bilateral_run := 0
var debug_contact_tick := 0
var marker: Label3D
var message := "B 选择物件，G 靠近并抓取 → 蓝色货仓"
var deliveries: Array[Dictionary] = []
var history: Array[Dictionary] = []
var grasp_start_height := 0.0
var max_lift := 0.0
var slot := 0
var captured := false
var released := false

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
		serial += 1
		request = "cancel"
		busy = false
		held = false
		message = "已取消 · 物件已松开"
		history.append({"event":"cancel","time":scene.robot.sim_time_seconds()})
	elif action == "pick":
		if busy: return
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
		target.sleeping = false
		serial += 1
		request = "pick"
		busy = true
		captured = false
		released = false
		held = false
		bilateral_frames = 0
		bilateral_run = 0
		max_bilateral_run = 0
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
		"object":str(body.name),"held":held,"bilateral_frames":bilateral_frames,"slot":slot,"busy":busy}

func accept(command: Dictionary) -> void:
	if not busy: return
	max_lift = maxf(max_lift,_center(target).y-grasp_start_height)
	var stage: String = command.get("grab_stage","")
	if command.get("assist_release",false) and captured and not held:
		released = true
	if stage == "complete" or stage == "failed":
		var placed := captured and released and max_bilateral_run >= 20 and max_lift > .08 and _in_cargo(target) and _supported(target)
		busy = false
		request = "idle"
		held = false
		message = "已放入蓝色货仓 · B 选下一件" if placed else "未完成物理夹持与入仓 · 请调整位置后重试"
		var result := {"event":"complete","object":str(target.name),"id":target.get_instance_id(),
			"success":placed,"captured":captured,"released":released,"lift_m":max_lift,
			"bilateral_frames":bilateral_frames,"max_bilateral_run":max_bilateral_run,"time":scene.robot.sim_time_seconds()}
		var bounds := _cargo_bounds(target)
		result["cargo_bounds"] = [[bounds[0].x,bounds[0].y,bounds[0].z],[bounds[1].x,bounds[1].y,bounds[1].z]]
		result["supported"] = _supported(target)
		deliveries.append(result)
		history.append(result)
		print("WORKSHOP_GRAB ",JSON.stringify(result))
	else:
		message = command.get("grab_message",scene.hub._stage_label(command.get("stage",""))) + " · X 取消"

func _process(_delta: float) -> void:
	if marker != null:
		marker.global_position = _center(_body())+Vector3(0,.14,0)
		marker.text = "▼ " + _name(_body())
		marker.visible = not scene.hub.options.plan.get("cinematic",false) and (not scene.hub.is_science_station() or _center(_body()).distance_to(scene.robot.bodies.chassis.global_position)<=1.6)

func _physics_process(_delta: float) -> void:
	if not busy or not is_instance_valid(target): return
	var contacts := target.get_colliding_bodies()
	if OS.has_environment("SAI_GRAB_DEBUG"):
		debug_contact_tick += 1
		if debug_contact_tick % 100 == 0:
			var direct := PhysicsServer3D.body_get_direct_state(target.get_rid())
			var rows: Array = []
			if direct != null:
				for i in range(direct.get_contact_count()):
					var other = direct.get_contact_collider_object(i)
					if other in [scene.robot.bodies.arm_gripper,scene.robot.bodies.arm_moving_jaw]:
						var p: Vector3 = direct.get_contact_local_position(i)
						var n: Vector3 = direct.get_contact_local_normal(i)
						rows.append({"body":str(other.name),"point":scene.robot.source(target.global_transform*p),
							"normal":scene.robot.source(target.global_basis*n),"impulse":direct.get_contact_impulse(i).length()})
			print("[DEBUG-SAI-GRIP] ",JSON.stringify({"t":scene.robot.sim_time_seconds(),
				"object":scene.robot.source(_center(target)),"contacts":rows}))
	held = contacts.has(scene.robot.bodies.arm_gripper) and contacts.has(scene.robot.bodies.arm_moving_jaw)
	if held:
		bilateral_frames += 1
		bilateral_run += 1
		max_bilateral_run = maxi(max_bilateral_run,bilateral_run)
	else:
		bilateral_run = 0
	max_lift = maxf(max_lift,_center(target).y-grasp_start_height)
	if not captured and max_bilateral_run >= 20 and max_lift > .015:
		captured = true
		history.append({"event":"physical_lift","object":str(target.name),"time":scene.robot.sim_time_seconds()})

func status_text() -> String:
	return "目标：%s → 蓝色货仓\n%s" % [_name(_body()),message]

func retention() -> Dictionary:
	var result := {}
	for item in _items(): result[str(item.body.name)] = _in_cargo(item.body)
	return result

func _exit_tree() -> void:
	pass
