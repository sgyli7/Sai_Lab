extends Node3D
## Godot side of MicroDuck's lower mouth and Sai-style assisted point grip.
const MAX_MASS_KG := 0.020
const REACH_M := 0.055
## jaw_soft's inertial local Y is the lateral hinge axis, and -Z points
## forward. The site coordinate below is mouth_tip expressed in that inertial
## frame at STAND (the generated Godot RigidBody3D uses the inertial frame).
## The previous Z rotation twisted the mouth around its nose and the old tip
## coordinate pointed at the head's side, invalidating prior pickup probes.
const PIVOT_LOCAL := Vector3(-0.004, 0.0, 0.045)
const TIP_LOCAL := Vector3(-0.01440896, -0.00309805, -0.04465586)
var actor
var hub
var head: RigidBody3D
var pivot: Node3D
var moving_body: AnimatableBody3D
var lower_shapes: Array[CollisionShape3D] = []
var moving_shapes: Array[CollisionShape3D] = []
var excluded_bodies: Array[PhysicsBody3D] = []
var target: RigidBody3D
var grip: PinJoint3D
var pending := false
var open := false
var requested_at := 0.0
var start_height := 0.0
var max_lift := 0.0
var message := "靠近地面轻物件后按 G 自动捡起；X 松开"
var history: Array[Dictionary] = []

func _ready() -> void:
	head = actor._bodies.get("jaw_soft")
	if head == null: push_error("MicroDuck lower beak needs jaw_soft"); return
	pivot = Node3D.new()
	pivot.name = "MovableLowerBeak"
	head.add_child(pivot)
	pivot.position = PIVOT_LOCAL
	moving_body = AnimatableBody3D.new()
	moving_body.name = "LowerBeakCollider"
	moving_body.collision_layer = 2
	moving_body.collision_mask = 1
	pivot.add_child(moving_body)
	for body in actor._bodies.values():
		if body is PhysicsBody3D: moving_body.add_collision_exception_with(body)
	# The roller MJCF has four earlier wheel geoms, shifting these generated
	# ordinal names by four. Require both lower-mouth parts in either scene.
	var visual_indices := [59, 61] if actor.session.mode == "roller" else [55, 57]
	var collision_indices := [60, 62] if actor.session.mode == "roller" else [56, 58]
	for index in visual_indices:
		var mesh: Node3D = head.get_node_or_null("vis_unnamed_%d_%d" % [index,index])
		if mesh == null: push_error("Missing lower beak visual %d" % index); continue
		mesh.reparent(pivot, true)
	for index in collision_indices:
		var shape: CollisionShape3D = head.get_node_or_null("col_unnamed_%d_%d" % [index,index])
		if shape == null: push_error("Missing lower beak collision %d" % index); continue
		lower_shapes.append(shape)
		var moved := CollisionShape3D.new()
		moved.name = "Moving%d" % index
		moved.shape = shape.shape
		moved.transform = Transform3D(Basis.IDENTITY,-PIVOT_LOCAL)*shape.transform
		moved.disabled = true
		moving_body.add_child(moved)
		moving_shapes.append(moved)

func _nearest_floor_target() -> RigidBody3D:
	var best: RigidBody3D = null
	var best_distance := INF
	var mouth := head.to_global(TIP_LOCAL)
	var candidates: Array = [actor._bodies.get("ball")]
	for item in hub.atelier.loose_props.items: candidates.append(item.body)
	for candidate in candidates:
		if not (candidate is RigidBody3D) or candidate.mass > MAX_MASS_KG: continue
		var center := _center(candidate)
		var relative := actor._base.global_basis.inverse() * (center-actor._base.global_position)
		# The GroundPick skill bends at the feet and has no object navigation.
		if relative.x < -0.03 or relative.x > 0.27 or absf(relative.z) > 0.14: continue
		if center.y > actor._base.global_position.y + 0.04: continue
		var distance := Vector2(center.x-mouth.x,center.z-mouth.z).length()
		if distance < best_distance and distance < 0.18:
			best = candidate
			best_distance = distance
	return best

func _center(body: RigidBody3D) -> Vector3:
	return body.global_transform * body.get_meta("grasp_center", Vector3.ZERO)

func perform(action: String) -> void:
	if head == null: return
	if action == "open":
		if grip == null and not pending:
			open = not open
			message = "嘴已张开" if open else "嘴已合上"
	elif action == "cancel":
		if grip != null:
			_detach()
			message = "已松开 · 最大吊升 %.0f mm" % (max_lift*1000.0)
		elif pending:
			pending = false
			open = false
			message = "已取消抓取"
	elif action == "pick":
		if grip != null or pending: return
		target = _nearest_floor_target()
		if target == null: message = "脚边没有可达的 20 g 内地面物件"; return
		if actor.brain.busy(): message = "请等当前动作结束"; return
		pending = true
		open = true
		requested_at = actor._t
		if actor.session.mode == "walk": actor.brain.tap("pick")
		message = "正在低头靠近 %s · X 取消" % target.name
		history.append({"event":"selected","object":str(target.name),"mass_kg":target.mass,"time":actor._t,
			"start_position":target.global_position})

func _physics_process(delta: float) -> void:
	if head == null: return
	var angle := 0.48 if open else 0.0
	pivot.rotation.y = move_toward(pivot.rotation.y, angle, delta*2.4)
	var displaced := absf(pivot.rotation.y) > 0.015
	for shape in lower_shapes: shape.disabled = displaced
	for shape in moving_shapes: shape.disabled = not displaced
	if pending:
		if not is_instance_valid(target): pending = false; open = false; return
		var mouth := head.to_global(TIP_LOCAL)
		var distance := mouth.distance_to(_center(target))
		# A wide box presents its near wall before its grasp centre reaches the
		# mouth. Keep the 75 mm centre limit for bottles and balls.
		var size: Vector3 = target.get_meta("grasp_size", Vector3.ZERO)
		var reach := REACH_M + maxf(0.0, size.x-0.070)*0.5
		if distance <= reach and actor.brain.pick_phase > 0.12:
			open = false
		var bodies := target.get_colliding_bodies()
		var mouth_contact := bodies.has(head) or bodies.has(moving_body)
		if mouth_contact and distance <= reach and not open and absf(pivot.rotation.y) < 0.18:
			_attach()
		elif actor._t-requested_at > 4.2:
			pending = false
			open = false
			message = "嘴未够到物件 · 调整位置后重试"
			history.append({"event":"miss","object":str(target.name),"last_gap_m":distance,
				"mouth_contact":mouth_contact,"time":actor._t})
	if grip != null and is_instance_valid(target):
		max_lift = maxf(max_lift, _center(target).y-start_height)

func _attach() -> void:
	# No teleport: the lower mouth must reach the object's grasp point first.
	pending = false
	open = false
	start_height = _center(target).y
	max_lift = 0.0
	grip = PinJoint3D.new()
	grip.name = "MicroDuckAssistedBeakGrip"
	grip.exclude_nodes_from_collision = false
	add_child(grip)
	grip.global_position = _center(target)
	grip.node_a = grip.get_path_to(head)
	grip.node_b = grip.get_path_to(target)
	# Large boxes otherwise wedge against the legs during the upward arc.
	# The point joint still transmits their full weight to the head.
	for body in actor._bodies.values():
		if body is PhysicsBody3D and body != target:
			target.add_collision_exception_with(body)
			excluded_bodies.append(body)
	target.add_collision_exception_with(moving_body)
	excluded_bodies.append(moving_body)
	message = "已夹住 · 正在抬头吊起；X 松开" if actor.session.mode == "roller" else "已夹住 · 等待捡地动作抬头；X 松开"
	history.append({"event":"contact_grip","object":str(target.name),"time":actor._t,
		"object_position":target.global_position})

func _detach() -> void:
	if grip == null: return
	grip.free()
	grip = null
	if is_instance_valid(target):
		for body in excluded_bodies:
			if is_instance_valid(body): target.remove_collision_exception_with(body)
	excluded_bodies.clear()
	history.append({"event":"release","object":str(target.name),"lift_m":max_lift,"time":actor._t})

func status_text() -> String:
	return message

func _exit_tree() -> void:
	_detach()
