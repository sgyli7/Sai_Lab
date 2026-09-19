extends Node3D
## Native Jolt reduced-order carrier. All motion follows real rigid-body forces.
## Bodies use COM/principal-inertia frames; assets retain canonical design frames.

const CONVERSION := Basis(Vector3(1,0,0),Vector3(0,0,-1),Vector3(0,1,0))
class Driving:
	static func clamp_request(spec:Dictionary,request:Vector2,measured_speed:float=0.)->Vector2:
		if not request.is_finite():return Vector2.ZERO
		var cfg:Dictionary=spec.get("driving",{})
		if cfg.is_empty():return Vector2(clampf(request.x,-.4,.4),clampf(request.y,-.003,.003))
		var speed:=clampf(request.x,-float(cfg.reverse_max_m_s),float(cfg.forward_max_m_s))
		var limit:=minf(float(cfg.yaw_rate_max_rad_s),minf(absf(speed)/float(cfg.minimum_turn_radius_m),
			float(cfg.lateral_acceleration_m_s2)/maxf(maxf(absf(speed),absf(measured_speed)),.01)))
		return Vector2(speed,clampf(request.y,-limit,limit))

	static func advance_request(spec:Dictionary,previous:Vector2,requested:Vector2,dt:float)->Vector2:
		var cfg:Dictionary=spec.get("driving",{})
		if cfg.is_empty():return Vector2(move_toward(previous.x,requested.x,.025*dt),move_toward(previous.y,requested.y,.0003*dt))
		var crossing:=previous.x*requested.x<0.
		var target:=requested
		if crossing:target.x=0.
		var slowing:=crossing or absf(target.x)<absf(previous.x)
		var rate:float=cfg.braking_m_s2 if slowing else cfg.acceleration_m_s2
		return Vector2(move_toward(previous.x,target.x,rate*dt),move_toward(previous.y,target.y,float(cfg.yaw_acceleration_rad_s2)*dt))

var specification: Dictionary
var bodies: Dictionary = {}
var drives: Dictionary = {}
var joint_rids: Array[RID] = []
var rest_inertial: Dictionary = {}
var design_offsets: Dictionary = {}
var descendants: Dictionary = {}
var requested := Vector2.ZERO
var filtered_request := Vector2.ZERO
var axis_targets: Dictionary = {}
var axis_positions: Dictionary = {}
var targets: Dictionary = {}
var last_angles: Dictionary = {}
var cables: Array[Dictionary] = []
var last_tensions: Array[float] = []
var time_s := 0.0
var control_phase := 0.0
var control_updates := 0
var active := false
var debug_shapes := false
var normal_loads: Dictionary = {}
var collision_impulse_estimates: Dictionary = {}
var include_body_poses := false
var restraints: Dictionary = {}
var locked: Dictionary = {}
var last_restraint_tensions: Dictionary = {}
var attachments: Dictionary = {}
var connector_failures: Array = []
var cable_reactions: Array[Dictionary] = []
var wheel_integral: Dictionary = {}
var lift_integral: Dictionary = {}
var implementation_sha256 := ""
var profile_timing := false
var timing_usec: Dictionary = {}
var timing_steps := 0
var batch_cargo_connectors := true
var use_native_cargo := false
var native_cargo: RefCounted
var cargo_backend_started := false
var cargo_backend_was_native := false
var use_native_wheels := false
var native_wheels: RefCounted
var wheel_backend_started := false
var wheel_backend_was_native := false
var _jel_state: Dictionary = {}
var _jel_details_pending := false
# Keep detailed observations current without allocating every Variant field on
# unobserved ticks. Force calculation and hydraulic history still run each tick.
var jel_state: Dictionary:
	get:
		if _jel_details_pending:
			var snapshot:Dictionary=native_jel.snapshot(false)
			if snapshot.size()!=specification.get("jel_groups",{}).size():
				push_error("Compact JEL snapshot absent for current physical step");get_tree().quit(3);return {}
			for stem in snapshot:_jel_state[stem]=snapshot[stem].telemetry
			_jel_details_pending=false
		return _jel_state
var jel_pressure_state: Dictionary = {}
var use_native_jel := false
var use_native_jel_bodies := false
var use_native_jel_compact := false
var jel_compact_started := false
var jel_compact_selected := false
var jel_body_boundary_started := false
var jel_body_boundary_selected := false
var native_jel: RefCounted
var jel_backend_started := false
var jel_backend_was_native := false
var jel_failures: Array = []
const JEL = preload("res://jel_forces.gd")

func _record_timing(label: String, started: int) -> int:
	if not profile_timing: return 0
	var now := Time.get_ticks_usec()
	timing_usec[label] = int(timing_usec.get(label,0))+now-started
	return now

static func gv(value) -> Vector3:
	return Vector3(float(value[0]),float(value[2]),-float(value[1]))

static func source(value: Vector3) -> Array:
	return [value.x,-value.z,value.y]

static func canonical_basis(rows: Array) -> Basis:
	var value := Basis(Vector3(rows[0][0],rows[1][0],rows[2][0]),
		Vector3(rows[0][1],rows[1][1],rows[2][1]),Vector3(rows[0][2],rows[1][2],rows[2][2]))
	return CONVERSION*value*CONVERSION.transposed()

static func configure_native_solver() -> void:
	# Shared standalone baseline; call before creating a vehicle/terrain world.
	ProjectSettings.set_setting("physics/3d/default_gravity",9.81)
	ProjectSettings.set_setting("physics/3d/default_linear_damp",0.)
	ProjectSettings.set_setting("physics/3d/default_angular_damp",0.)
	ProjectSettings.set_setting("physics/jolt_physics_3d/simulation/velocity_steps",32)
	ProjectSettings.set_setting("physics/jolt_physics_3d/simulation/position_steps",4)
	ProjectSettings.set_setting("physics/jolt_physics_3d/simulation/penetration_slop",.0002)
	ProjectSettings.set_setting("physics/jolt_physics_3d/simulation/speculative_contact_distance",.001)
	ProjectSettings.settings_changed.emit()

func setup(manifest: Dictionary, show_debug_shapes: bool = false) -> void:
	implementation_sha256 = FileAccess.get_sha256("res://physics_runtime.gd")
	specification = manifest
	debug_shapes = show_debug_shapes
	for key in specification.bodies:
		var data: Dictionary = specification.bodies[key]
		var body := RigidBody3D.new()
		body.name = str(key)
		body.mass = float(data.mass_kg)
		body.center_of_mass_mode = RigidBody3D.CENTER_OF_MASS_MODE_CUSTOM
		body.center_of_mass = Vector3.ZERO
		body.inertia = gv(data.principal_inertia_kg_m2).abs()
		body.linear_damp_mode = RigidBody3D.DAMP_MODE_REPLACE
		body.angular_damp_mode = RigidBody3D.DAMP_MODE_REPLACE
		body.linear_damp = 0.0
		body.angular_damp = 0.0
		body.can_sleep = false
		body.collision_layer = 2
		body.collision_mask = 25
		if "_wheel_" in str(key):
			body.collision_layer = 4
			body.collision_mask = 1
		if str(key).ends_with("_hook") or data.has("cargo"):
			body.collision_layer = 8
			body.collision_mask = 27
		for collision in data.collision:
			if collision.group == "mechanism":
				body.collision_layer = 32
				body.collision_mask = 27
		body.contact_monitor = true
		body.max_contacts_reported = 16
		var material := PhysicsMaterial.new()
		material.friction = .8
		material.bounce = 0.0
		body.physics_material_override = material
		add_child(body)
		body.global_transform = Transform3D(canonical_basis(data.inertial_axes_world),gv(data.com_world_m))
		bodies[key] = body
		rest_inertial[key] = body.global_basis
		design_offsets[key] = body.global_transform.affine_inverse()*Transform3D(Basis.IDENTITY,gv(data.position))
		for shape_data in data.collision:
			_add_shape(body,shape_data)
	for key in specification.bodies:
		var data: Dictionary = specification.bodies[key]
		if data.parent == null: continue
		_make_joint(str(key),data)
	for key in specification.joints:
		var definition: Dictionary = specification.joints[key]
		if definition.type == "orientation_channel":
			var anchor := gv(specification.bodies[definition.body].position)
			var parent: RigidBody3D = bodies[definition.parent]
			var child: RigidBody3D = bodies[definition.body]
			drives[key] = {"parent":parent,"child":child,"parent_id":definition.parent,"body_id":definition.body,
				"parent_anchor":parent.global_transform.affine_inverse()*anchor,"child_anchor":child.global_transform.affine_inverse()*anchor,"definition":definition}
		axis_targets[key] = 0.0
		axis_positions[key] = 0.0
		targets[key] = 0.0
		last_angles[key] = 0.0
		if specification.joints[key].mode == "wheel":
			normal_loads[key] = 0.0
			wheel_integral[key] = 0.0
		if specification.joints[key].mode == "lift": lift_integral[key] = 0.0
		descendants[key] = []
		for body_key in specification.bodies:
			var ancestor = body_key
			while ancestor != null:
				if ancestor == specification.joints[key].body:
					descendants[key].append(body_key)
					break
				ancestor = specification.bodies[ancestor].parent
	for key in specification.bodies:
		var data: Dictionary = specification.bodies[key]
		if not data.has("rope"): continue
		var rope: Dictionary = data.rope.duplicate(true)
		rope.body = key
		rope.local_anchor = bodies[key].global_transform.affine_inverse()*gv(rope.anchor_world)
		rope.payout_speed = 0.0
		rope.target_length = float(rope.length)
		cables.append(rope)
	for key in specification.get("locks",{}):
		var lock: Dictionary = specification.locks[key]
		if lock.initial_active: _lock_axis(lock.joint,float(lock.value))
	for key in specification.get("cargo_restraints",{}):
		var record: Dictionary = specification.cargo_restraints[key].duplicate(true)
		for strap in record.straps:
			strap.cargo_local = bodies[key].global_transform.affine_inverse()*gv(strap.cargo_anchor_world)
			strap.support_local = bodies[record.support].global_transform.affine_inverse()*gv(strap.support_anchor_world)
		for connector in record.get("connectors",[]):
			connector.cargo_local = bodies[key].global_transform.affine_inverse()*gv(connector.cargo_anchor_world)
			connector.support_local = bodies[record.support].global_transform.affine_inverse()*gv(connector.support_anchor_world)
		restraints[key] = record
	active = true

func _add_shape(body: RigidBody3D, data: Dictionary) -> void:
	var node := CollisionShape3D.new()
	var world := Transform3D(Basis.IDENTITY,gv(data.center))
	if data.has("quaternion"):
		var q: Array = data.quaternion
		var rotation := Basis(Quaternion(float(q[1]),float(q[2]),float(q[3]),float(q[0])))
		world.basis = CONVERSION*rotation*CONVERSION.transposed()
	if data.type == "box":
		var shape := BoxShape3D.new()
		shape.size = gv(data.size).abs()
		node.shape = shape
	elif data.type == "sphere":
		var shape := SphereShape3D.new()
		shape.radius = float(data.radius)
		node.shape = shape
	elif data.type == "cylinder":
		var shape := CylinderShape3D.new()
		shape.radius = float(data.radius)
		shape.height = float(data.length)
		node.shape = shape
		world.basis = Basis(Quaternion(Vector3.UP,gv(data.axis).normalized()))
	elif data.type == "convex":
		var shape := ConvexPolygonShape3D.new()
		var points := PackedVector3Array()
		for vertex in data.vertices_local_m: points.append(gv(vertex))
		shape.points = points
		node.shape = shape
	else:
		push_error("Unsupported carrier collision: "+str(data.type))
		return
	node.shape.margin = .001
	body.add_child(node)
	node.global_transform = world
	if debug_shapes:
		# Debug only: production visuals are imported authored neutral assets.
		var display := MeshInstance3D.new()
		if data.type == "box":
			var mesh := BoxMesh.new()
			mesh.size = gv(data.size).abs()
			display.mesh = mesh
		elif data.type == "sphere":
			var mesh := SphereMesh.new()
			mesh.radius = float(data.radius)
			mesh.height = 2*float(data.radius)
			display.mesh = mesh
		elif data.type == "convex":
			# Native shape outline is diagnostic only, never a model-authoring path.
			display.mesh = node.shape.get_debug_mesh()
		else:
			var mesh := CylinderMesh.new()
			mesh.top_radius = float(data.radius)
			mesh.bottom_radius = float(data.radius)
			mesh.height = float(data.length)
			display.mesh = mesh
		var paint := StandardMaterial3D.new()
		paint.albedo_color = Color(.12,.16,.17) if data.group == "wheel" else Color(.73,.77,.74)
		display.material_override = paint
		node.add_child(display)

func _make_joint(key: String, data: Dictionary) -> void:
	var parent: RigidBody3D = bodies[data.parent]
	var child: RigidBody3D = bodies[key]
	var joint = data.joint
	var rid := PhysicsServer3D.joint_create()
	joint_rids.append(rid)
	var anchor := gv(joint.get("anchor_world_m",data.position) if joint != null else data.position)
	if joint != null and joint.type == "ball":
		var frame := Transform3D(Basis.IDENTITY,anchor)
		PhysicsServer3D.joint_make_generic_6dof(rid,parent.get_rid(),parent.global_transform.affine_inverse()*frame,
			child.get_rid(),child.global_transform.affine_inverse()*frame)
		for axis_index in range(3):
			PhysicsServer3D.generic_6dof_joint_set_flag(rid,axis_index,PhysicsServer3D.G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT,true)
			PhysicsServer3D.generic_6dof_joint_set_param(rid,axis_index,PhysicsServer3D.G6DOF_JOINT_LINEAR_LOWER_LIMIT,0.)
			PhysicsServer3D.generic_6dof_joint_set_param(rid,axis_index,PhysicsServer3D.G6DOF_JOINT_LINEAR_UPPER_LIMIT,0.)
			PhysicsServer3D.generic_6dof_joint_set_flag(rid,axis_index,PhysicsServer3D.G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT,false)
		PhysicsServer3D.joint_disable_collisions_between_bodies(rid,true)
		return
	if joint == null:
		var world := Transform3D(Basis.IDENTITY,anchor)
		PhysicsServer3D.joint_make_generic_6dof(rid,parent.get_rid(),parent.global_transform.affine_inverse()*world,
			child.get_rid(),child.global_transform.affine_inverse()*world)
		for axis in range(3):
			PhysicsServer3D.generic_6dof_joint_set_flag(rid,axis,PhysicsServer3D.G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT,true)
			PhysicsServer3D.generic_6dof_joint_set_flag(rid,axis,PhysicsServer3D.G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT,true)
			PhysicsServer3D.generic_6dof_joint_set_param(rid,axis,PhysicsServer3D.G6DOF_JOINT_LINEAR_LOWER_LIMIT,0.)
			PhysicsServer3D.generic_6dof_joint_set_param(rid,axis,PhysicsServer3D.G6DOF_JOINT_LINEAR_UPPER_LIMIT,0.)
			PhysicsServer3D.generic_6dof_joint_set_param(rid,axis,PhysicsServer3D.G6DOF_JOINT_ANGULAR_LOWER_LIMIT,0.)
			PhysicsServer3D.generic_6dof_joint_set_param(rid,axis,PhysicsServer3D.G6DOF_JOINT_ANGULAR_UPPER_LIMIT,0.)
		return
	var axis := gv(joint.axis).normalized()
	var alignment := Basis(Quaternion(Vector3.RIGHT if joint.type == "slide" else Vector3.BACK,axis))
	var world := Transform3D(alignment,anchor)
	var frame_a := parent.global_transform.affine_inverse()*world
	var frame_b := child.global_transform.affine_inverse()*world
	if joint.type == "slide":
		PhysicsServer3D.joint_make_slider(rid,parent.get_rid(),frame_a,child.get_rid(),frame_b)
		PhysicsServer3D.slider_joint_set_param(rid,PhysicsServer3D.SLIDER_JOINT_LINEAR_LIMIT_LOWER,float(joint.limits[0]) if joint.limits != null else 1.)
		PhysicsServer3D.slider_joint_set_param(rid,PhysicsServer3D.SLIDER_JOINT_LINEAR_LIMIT_UPPER,float(joint.limits[1]) if joint.limits != null else -1.)
	else:
		PhysicsServer3D.joint_make_hinge(rid,parent.get_rid(),frame_a,child.get_rid(),frame_b)
		PhysicsServer3D.hinge_joint_set_flag(rid,PhysicsServer3D.HINGE_JOINT_FLAG_USE_LIMIT,joint.limits != null)
		if joint.limits != null:
			# Godot hinge's limit angle has opposite sign to positive physical twist.
			PhysicsServer3D.hinge_joint_set_param(rid,PhysicsServer3D.HINGE_JOINT_LIMIT_LOWER,-float(joint.limits[1]))
			PhysicsServer3D.hinge_joint_set_param(rid,PhysicsServer3D.HINGE_JOINT_LIMIT_UPPER,-float(joint.limits[0]))
	PhysicsServer3D.joint_disable_collisions_between_bodies(rid,true)
	drives[joint.name] = {"rid":rid,"parent":parent,"child":child,"parent_id":data.parent,"body_id":key,
		"parent_anchor":parent.global_transform.affine_inverse()*anchor,
		"child_anchor":child.global_transform.affine_inverse()*anchor,"definition":joint}

func get_design_transform(body_id: String) -> Transform3D:
	if not bodies.has(body_id):
		var alias:Dictionary=specification.get("body_aliases",{}).get(body_id,{})
		if not alias.is_empty():
			var q:Array=alias.quaternion_local_wxyz
			var local_basis:=Basis(Quaternion(float(q[1]),float(q[2]),float(q[3]),float(q[0])))
			return get_design_transform(str(alias.body))*Transform3D(CONVERSION*local_basis*CONVERSION.transposed(),gv(alias.position_local_m))
	return bodies[body_id].global_transform*design_offsets[body_id]

func set_vehicle_command(speed: float, yaw_rate: float) -> void:
	var measured_speed:float=bodies.front.linear_velocity.length() if bodies.has("front") else 0.
	requested = Driving.clamp_request(specification,Vector2(speed,yaw_rate),measured_speed)

func set_axis_target(joint_id: String, value: float) -> bool:
	if not specification.joints.has(joint_id):
		push_error("Unknown carrier axis: "+joint_id)
		return false
	var definition: Dictionary = specification.joints[joint_id]
	if definition.has("command_limits"):value=clampf(value,float(definition.command_limits[0]),float(definition.command_limits[1]))
	if definition.limits != null: value = clampf(value,float(definition.limits[0]),float(definition.limits[1]))
	if joint_id.begins_with("crane_") and (joint_id.ends_with("_luff_joint") or joint_id.ends_with("_extend_joint")):
		var stem := "_".join(joint_id.split("_").slice(0,2))
		if attachments.has(stem+"_hook"):
			var future := axis_targets.duplicate()
			future[joint_id] = value
			var radius := (14.+float(future[stem+"_extend_joint"]))*cos(deg_to_rad(20.)-float(future[stem+"_luff_joint"]))
			if not _load_allowed(stem+"_hook",attachments[stem+"_hook"].cargo,radius): return false
	axis_targets[joint_id] = value
	if locked.has(joint_id) and absf(float(locked[joint_id])-value) > .00001:
		var rid: RID = drives[joint_id].rid
		PhysicsServer3D.slider_joint_set_param(rid,PhysicsServer3D.SLIDER_JOINT_LINEAR_LIMIT_LOWER,float(definition.limits[0]))
		PhysicsServer3D.slider_joint_set_param(rid,PhysicsServer3D.SLIDER_JOINT_LINEAR_LIMIT_UPPER,float(definition.limits[1]))
		locked.erase(joint_id)
	return true

func _load_allowed(hook: String,cargo: String,radius: float) -> bool:
	var chart: Array = specification.crane_load_chart
	var capacity := 0.0
	if radius <= float(chart[0][0]):
		capacity = float(chart[0][1])
	else:
		for i in range(1,chart.size()):
			if radius <= float(chart[i][0]):
				capacity = lerpf(float(chart[i-1][1]),float(chart[i][1]),(radius-float(chart[i-1][0]))/(float(chart[i][0])-float(chart[i-1][0])))
				break
	var connected: Dictionary = {cargo:true}
	for iteration in range(restraints.size()):
		var added := false
		for key in restraints:
			if restraints[key].secured and connected.has(restraints[key].support) and not connected.has(key):
				connected[key] = true
				added = true
		if not added: break
	var gross: float = specification.bodies[hook].mass_kg
	for key in connected: gross += float(specification.bodies[key].mass_kg)
	return gross <= capacity

func _lock_axis(joint_id: String, value: float) -> void:
	var rid: RID = drives[joint_id].rid
	PhysicsServer3D.slider_joint_set_param(rid,PhysicsServer3D.SLIDER_JOINT_LINEAR_LIMIT_LOWER,value)
	PhysicsServer3D.slider_joint_set_param(rid,PhysicsServer3D.SLIDER_JOINT_LINEAR_LIMIT_UPPER,value)
	locked[joint_id] = value

func set_secured(cargo_id: String, secured: bool) -> void:
	if restraints.has(cargo_id): restraints[cargo_id].secured = secured

func set_winch_target(crane: String, length: float) -> void:
	var body_id := crane if crane.ends_with("_extend") else crane+"_extend"
	for cable in cables:
		if cable.body == body_id: cable.target_length = clampf(length,1.5,25.)

func attach_cargo(crane: String, cargo_id: String) -> Dictionary:
	var hook_id := crane if crane.ends_with("_hook") else crane+"_hook"
	if not restraints.has(cargo_id) or not bodies.has(hook_id):
		return {"attached":false,"reason":"unknown hook or cargo"}
	if attachments.has(hook_id): return {"attached":false,"reason":"hook occupied"}
	for value in attachments.values():
		if value.cargo == cargo_id: return {"attached":false,"reason":"cargo already attached"}
	var base := get_design_transform(hook_id.trim_suffix("_hook")+"_yaw").origin
	var displacement := get_design_transform(cargo_id).origin-base
	if not _load_allowed(hook_id,cargo_id,Vector2(displacement.x,displacement.z).length()):
		return {"attached":false,"reason":"gross load exceeds radius load chart"}
	var cargo: Dictionary = specification.bodies[cargo_id].cargo
	var center := gv(cargo.center)
	var size := gv(cargo.size).abs()
	var pitch: Array = cargo.get("corner_pitch_m",[5.853,2.259])
	var available: Array[Vector3] = []
	for sx in [-1,1]:
		for sy in [-1,1]: available.append(center+Vector3(sx*float(pitch[0])/2,size.y/2,sy*float(pitch[1])/2))
	var pairs: Array[Dictionary] = []
	for anchor in specification.bodies[hook_id].spreader_anchors_world:
		var hook_local: Vector3 = bodies[hook_id].global_transform.affine_inverse()*get_design_transform(hook_id)*gv(Vector3(anchor[0],anchor[1],anchor[2])-Vector3(specification.bodies[hook_id].position[0],specification.bodies[hook_id].position[1],specification.bodies[hook_id].position[2]))
		var p: Vector3 = bodies[hook_id].global_transform*hook_local
		var nearest := -1
		var best := INF
		for i in range(available.size()):
			var q: Vector3 = get_design_transform(cargo_id)*(available[i]-gv(specification.bodies[cargo_id].position))
			if p.distance_to(q) < best:
				best = p.distance_to(q)
				nearest = i
		if best > .15: return {"attached":false,"reason":"spreader corner farther than 0.15 m","distance_m":best}
		var target: Vector3 = get_design_transform(cargo_id)*(available[nearest]-gv(specification.bodies[cargo_id].position))
		pairs.append({"support_local":hook_local,"cargo_local":bodies[cargo_id].global_transform.affine_inverse()*target,
			"rest_offset_local":[0.,0.,0.],"stiffness":1e7,"damping":5e4,"max_force_n":500000.,"break_displacement_m":.25,"bilateral":true})
		available.remove_at(nearest)
	attachments[hook_id] = {"cargo":cargo_id,"connectors":pairs}
	return {"attached":true,"reason":"four physical corner connectors closed"}

func release_cargo(crane: String) -> bool:
	var hook_id := crane if crane.ends_with("_hook") else crane+"_hook"
	return attachments.erase(hook_id)

func _drive_targets() -> void:
	var front := get_design_transform("front")
	var rolling_steer := absf(filtered_request.x) >= .03
	var yaw_rate := filtered_request.y if rolling_steer else 0.
	for key in drives:
		var drive: Dictionary = drives[key]
		var definition: Dictionary = drive.definition
		if definition.mode != "steer": continue
		var transform := get_design_transform(drive.body_id)
		var parent_transform := get_design_transform(drive.parent_id)
		var local: Vector3 = front.affine_inverse()*transform.origin
		# Canonical Y-left is Godot -Z.
		var desired := front.basis*Vector3(filtered_request.x+yaw_rate*local.z,0.,-yaw_rate*local.x)
		var parent_velocity := parent_transform.basis.inverse()*desired
		var angle := atan2(-parent_velocity.z,parent_velocity.x) if parent_velocity.length() > .000001 else 0.0
		var direction := 1.0
		if absf(angle) > PI/2:
			angle = atan2(parent_velocity.z,-parent_velocity.x)
			direction = -1.0
		targets[key] = clampf(angle,float(definition.limits[0]),float(definition.limits[1]))
		var stem: String = definition.get("group",str(drive.body_id).trim_suffix("_steer"))
		var relative := parent_transform.basis.inverse()*transform.basis
		var measured: float = joint_state(key).q if definition.type == "orientation_channel" else atan2(-relative.x.z,relative.x.x)
		var maximum_rate: float = definition.get("speed_limit",.005)
		var steer_rate := clampf(.6*(float(targets[key])-measured),-maximum_rate,maximum_rate) if rolling_steer else 0.
		if rolling_steer: axis_positions[key] = move_toward(float(axis_positions[key]),float(targets[key]),maximum_rate*.02)
		targets[key] = axis_positions[key]
		var local_yaw := front.basis.y*yaw_rate+parent_transform.basis.y*steer_rate
		var forward := get_design_transform(stem+"_pitch").basis.x
		for side in ["l","r"]:
			for number in range(3):
				var wheel: String = stem+"_wheel_"+side+"_"+str(number)
				var velocity := desired+local_yaw.cross(get_design_transform(wheel).origin-transform.origin)
				targets[wheel+"_joint"] = forward.dot(velocity)/1.5

func joint_state(key: String) -> Dictionary:
	var drive: Dictionary = drives[key]
	var definition: Dictionary = drive.definition
	var parent: RigidBody3D = drive.parent
	var child: RigidBody3D = drive.child
	var parent_design := get_design_transform(drive.parent_id)
	var child_design := get_design_transform(drive.body_id)
	var axis := parent_design.basis*gv(definition.axis)
	var pa: Vector3 = parent.global_transform*drive.parent_anchor
	var pb: Vector3 = child.global_transform*drive.child_anchor
	var q := 0.0
	var velocity := 0.0
	if definition.type == "orientation_channel":
		var relative := parent_design.basis.transposed()*child_design.basis
		var canonical := CONVERSION.transposed()*relative*CONVERSION
		var omega := CONVERSION.transposed()*(parent_design.basis.transposed()*(child.angular_velocity-parent.angular_velocity))
		var state: Dictionary = JEL.orientation_state(canonical,omega)
		q = state.angles[int(definition.component)]
		velocity = state.rates[int(definition.component)]
		axis = parent_design.basis*CONVERSION*state.axes[int(definition.component)]
	elif definition.type == "slide":
		q = (pb-pa).dot(axis)
		var va := parent.linear_velocity+parent.angular_velocity.cross(pa-parent.global_position)
		var vb := child.linear_velocity+child.angular_velocity.cross(pb-child.global_position)
		velocity = (vb-va).dot(axis)
	else:
		var relative := parent_design.basis.inverse()*child_design.basis
		var quaternion := relative.orthonormalized().get_rotation_quaternion()
		var raw := wrapf(2*atan2(Vector3(quaternion.x,quaternion.y,quaternion.z).dot(gv(definition.axis)),quaternion.w),-PI,PI)
		q = float(last_angles[key])+wrapf(raw-float(last_angles[key]),-PI,PI)
		last_angles[key] = q
		velocity = (child.angular_velocity-parent.angular_velocity).dot(axis)
	return {"q":q,"velocity":velocity,"axis":axis,"parent_anchor":pa,"child_anchor":pb}

func _gravity_bias(key: String, joint: Dictionary) -> float:
	var force_sum := Vector3.ZERO
	var torque_sum := Vector3.ZERO
	for body_key in descendants[key]:
		var body: RigidBody3D = bodies[body_key]
		var force := Vector3(0.,-body.mass*float(specification.gravity),0.)
		force_sum += force
		torque_sum += (body.global_position-joint.parent_anchor).cross(force)
	for reaction in cable_reactions:
		if reaction.body in descendants[key]:
			force_sum += reaction.force
			torque_sum += (reaction.point-joint.parent_anchor).cross(reaction.force)
	return -joint.axis.dot(force_sum if specification.joints[key].type == "slide" else torque_sum)

func _physics_process(delta: float) -> void:
	if not active: return
	if wheel_backend_started and use_native_wheels!=wheel_backend_was_native:
		push_error("Wheel backend cannot change during an episode; reset the world explicitly");get_tree().quit(3);return
	wheel_backend_started=true;wheel_backend_was_native=use_native_wheels
	var timing_start := Time.get_ticks_usec() if profile_timing else 0
	if control_phase <= .0000001:
		control_updates += 1
		filtered_request = Driving.advance_request(specification,filtered_request,requested,.02)
		for key in axis_targets:
			if specification.joints[key].mode == "steer": continue
			var rate: float = specification.joints[key].get("speed_limit",1e9)
			axis_positions[key] = move_toward(float(axis_positions[key]),float(axis_targets[key]),rate*.02)
		targets = axis_positions.duplicate()
		_drive_targets()
		for key in lift_integral:
			var s := joint_state(key)
			var error: float = float(targets[key])-s.q
			var maximum: float = specification.joints[key].max_force
			var last: float = drives[key].get("last_force",0.)
			if absf(last) < maximum*.999 or error*last < 0.:
				lift_integral[key] = clampf(float(lift_integral[key])+float(specification.joints[key].stiffness)*1.5*error*.02,-maximum,maximum)
		for key in wheel_integral:
			var s := joint_state(key)
			var error: float = float(targets[key])-s.velocity
			var maximum: float = specification.joints[key].max_force
			var last: float = drives[key].get("last_force",0.)
			if normal_loads[key] < 1.:
				wheel_integral[key] = float(wheel_integral[key])*exp(-.02/.2)
			elif absf(last) < maximum*.999 or error*last < 0.:
				wheel_integral[key] = clampf(float(wheel_integral[key])+4e5*error*.02,-maximum,maximum)
		for lock in specification.get("locks",{}).values():
			if locked.has(lock.joint) or absf(float(axis_targets[lock.joint])-float(lock.value)) > .00001: continue
			var value := joint_state(lock.joint)
			if absf(value.q-float(lock.value)) < .004 and absf(value.velocity) < .01:
				_lock_axis(lock.joint,float(lock.value))
		control_phase += .02
	timing_start = _record_timing("control",timing_start)
	_apply_jel()
	timing_start = _record_timing("jel_forces",timing_start)
	_estimate_rolling_loads(delta)
	timing_start = _record_timing("rolling_loads",timing_start)
	_apply_cables(delta)
	timing_start = _record_timing("cables",timing_start)
	_apply_restraints()
	timing_start = _record_timing("cargo_connectors",timing_start)
	var native_wheels_applied := false
	for key in drives:
		var drive: Dictionary = drives[key]
		var definition: Dictionary = drive.definition
		if definition.type == "orientation_channel" or definition.mode == "jel_guide": continue
		if use_native_wheels and definition.mode == "wheel":
			if not native_wheels_applied:
				if not _apply_native_wheels():return
				native_wheels_applied=true
			continue
		var s := joint_state(key)
		var target: float = targets.get(key,0.)
		var force: float
		var resistance := 0.0
		if definition.mode == "suspension":
			force = -float(definition.preload_n)-float(definition.stiffness)*(s.q-target)-float(definition.damping)*s.velocity
		elif definition.mode == "wheel":
			var load_n: float = normal_loads.get(key,0.)
			var rolling: float = float(specification.get("rolling_resistance_coefficient",0.))*load_n*float(definition.radius)
			resistance = rolling*tanh(s.velocity/.005)
			force = 8e5*(target-s.velocity)+rolling*tanh(target/.005)+float(wheel_integral[key])
		else:
			force = float(definition.stiffness)*(target-s.q)-float(definition.damping)*s.velocity
			if definition.mode in ["axis","lift"]: force += _gravity_bias(key,s)
			if definition.mode == "lift": force += float(lift_integral[key])
		var maximum: float = definition.max_force
		if definition.has("power_w"): maximum = minf(maximum,float(definition.power_w)/maxf(absf(s.velocity),1e-9))
		force = clampf(force,-maximum,maximum)
		drive.last_force = force
		var parent: RigidBody3D = drive.parent
		var child: RigidBody3D = drive.child
		if definition.type == "slide":
			child.apply_force(s.axis*force,s.child_anchor-child.global_position)
			parent.apply_force(-s.axis*force,s.parent_anchor-parent.global_position)
		else:
			child.apply_torque(s.axis*(force-resistance))
			parent.apply_torque(-s.axis*(force-resistance))
	_record_timing("axis_forces",timing_start)
	if profile_timing: timing_steps += 1
	control_phase -= delta
	time_s += delta

func _apply_native_wheels() -> bool:
	if native_wheels == null:
		var status := GDExtensionManager.load_extension("res://native/leviathan_jel.gdextension")
		if status != GDExtensionManager.LOAD_STATUS_OK and status != GDExtensionManager.LOAD_STATUS_ALREADY_LOADED:
			push_error("Native wheel extension unavailable");get_tree().quit(3);return false
		native_wheels=ClassDB.instantiate("LeviathanWheels")
		if native_wheels == null:
			push_error("Native wheel class unavailable");get_tree().quit(3);return false
		var abi:Dictionary=native_wheels.abi_info()
		if abi.precision!=("double" if OS.has_feature("double") else "single") or not native_wheels.configure(drives,design_offsets,float(specification.get("rolling_resistance_coefficient",0.))):
			push_error("Native wheel ABI or physical drive configuration failed");get_tree().quit(3);return false
	# The native configure call verifies that all wheel-force calls form one
	# contiguous subsequence. Hitch forces stay before it; service axes after it.
	var result:Dictionary=native_wheels.step(targets,normal_loads,wheel_integral,last_angles,true)
	if result.get("count",0)!=wheel_integral.size() or not result.get("applied",false):
		push_error("Native wheel step failed; no reference fallback");get_tree().quit(3);return false
	return true

func _canonical_kinematics(key: String) -> Dictionary:
	var body: RigidBody3D = bodies[key]
	var design := get_design_transform(key)
	var linear := body.linear_velocity+body.angular_velocity.cross(design.origin-body.global_position)
	return {"pose":Transform3D(CONVERSION.transposed()*design.basis*CONVERSION,CONVERSION.transposed()*design.origin),
		"velocity":CONVERSION.transposed()*linear,"omega":CONVERSION.transposed()*body.angular_velocity}

func _apply_jel() -> void:
	if use_native_jel_bodies and not OS.has_feature("double"):
		push_error("Native JEL body boundary currently qualified only for double ABI");get_tree().quit(3);return
	if use_native_jel_compact and not (use_native_jel and use_native_jel_bodies):
		push_error("Compact JEL requires native core and body boundary");get_tree().quit(3);return
	if jel_compact_started and use_native_jel_compact!=jel_compact_selected:
		push_error("Compact JEL selector cannot change during an episode");get_tree().quit(3);return
	jel_compact_started=true;jel_compact_selected=use_native_jel_compact
	if use_native_jel_bodies and not use_native_jel:
		push_error("Native JEL body boundary requires native JEL core");get_tree().quit(3);return
	if jel_body_boundary_started and use_native_jel_bodies!=jel_body_boundary_selected:
		push_error("JEL body boundary cannot change during an episode");get_tree().quit(3);return
	jel_body_boundary_started=true;jel_body_boundary_selected=use_native_jel_bodies
	# Selection is an episode option. Hot switching would discard accumulated
	# pressure/energy in one backend and incorrectly precharge the other.
	if jel_backend_started and use_native_jel!=jel_backend_was_native:
		push_error("JEL backend cannot change during an episode; reset the world explicitly");get_tree().quit(3);return
	jel_backend_started=true;jel_backend_was_native=use_native_jel
	var batch: Dictionary = {}
	if use_native_jel:
		if native_jel == null:
			var status := GDExtensionManager.load_extension("res://native/leviathan_jel.gdextension")
			if status != GDExtensionManager.LOAD_STATUS_OK and status != GDExtensionManager.LOAD_STATUS_ALREADY_LOADED:
				push_error("Native JEL requested but extension could not load");get_tree().quit(3);return
			native_jel = ClassDB.instantiate("LeviathanJEL")
			if native_jel != null:
				var abi:Dictionary=native_jel.abi_info()
				if abi.precision!=("double" if OS.has_feature("double") else "single"):
					push_error("JEL extension and engine ABI precision differ");get_tree().quit(3);return
			if native_jel == null or not native_jel.configure(specification.get("jel_groups",{})):
				push_error("Native JEL configuration failed");get_tree().quit(3);return
		if use_native_jel_bodies:
			if not native_jel.has_method("step_bodies"):
				push_error("Native JEL body boundary is absent from selected extension");get_tree().quit(3);return
			if use_native_jel_compact:
				if not native_jel.has_method("step_bodies_compact") or not native_jel.has_method("snapshot"):
					push_error("Compact JEL methods absent");get_tree().quit(3);return
				batch=native_jel.step_bodies_compact(bodies,design_offsets,targets,1./Engine.physics_ticks_per_second,true)
			else:
				batch=native_jel.step_bodies(bodies,design_offsets,targets,1./Engine.physics_ticks_per_second,true)
		else:
			var measured: Dictionary = {}
			for group in specification.get("jel_groups",{}).values():
				for key in [group.hull,group.truck]:
					if not measured.has(key):measured[key]=_canonical_kinematics(key)
			batch=native_jel.step(measured,targets,1./Engine.physics_ticks_per_second)
		if batch.size()!=specification.get("jel_groups",{}).size():
			push_error("Native JEL step failed; no reference fallback");get_tree().quit(3);return
	_jel_details_pending=use_native_jel_compact
	for stem in specification.get("jel_groups",{}):
		var group: Dictionary = specification.jel_groups[stem]
		var evaluated: Dictionary
		if use_native_jel:evaluated=batch[stem]
		else:
			var hull := _canonical_kinematics(group.hull)
			var truck := _canonical_kinematics(group.truck)
			if not jel_pressure_state.has(stem):jel_pressure_state[stem]={}
			evaluated=JEL.evaluate(group,hull.pose,truck.pose,hull.velocity,hull.omega,truck.velocity,truck.omega,
				float(targets.get(stem+"_susp_joint",0.)),float(targets.get(stem+"_steer_joint",0.)),jel_pressure_state[stem],1./Engine.physics_ticks_per_second)
		_jel_state[stem] = evaluated.telemetry
		if not evaluated.telemetry.envelope_failures.is_empty():
			var known := false
			for failure in jel_failures:
				if failure.truck == stem: known = true
			if not known:jel_failures.append({"time_s":time_s,"truck":stem,"reasons":evaluated.telemetry.envelope_failures.duplicate()})
		if use_native_jel_bodies:continue # Same forces already applied inside the optional native boundary.
		var a: RigidBody3D = bodies[group.hull]
		var b: RigidBody3D = bodies[group.truck]
		for item in evaluated.forces:
			var force: Vector3 = CONVERSION*item.force_truck
			var p: Vector3 = CONVERSION*item.hull_point
			var q: Vector3 = CONVERSION*item.truck_point
			b.apply_force(force,q-b.global_position)
			a.apply_force(-force,p-a.global_position)
		var torque: Vector3 = CONVERSION*evaluated.buffer_torque_world
		b.apply_torque(torque);a.apply_torque(-torque)

func _estimate_rolling_loads(delta: float) -> void:
	# Jolt's exposed contact impulse is EstimateCollisionResponse, not the
	# solved reaction including joint-transmitted weight. Never label it wheel load.
	var touching: Dictionary = {}
	for key in normal_loads:
		normal_loads[key] = 0.
		var direct := PhysicsServer3D.body_get_direct_state(drives[key].child.get_rid())
		var estimate := 0.0
		var count := 0
		if direct != null:
			count = direct.get_contact_count()
			for contact in range(count):
				estimate += absf(direct.get_contact_impulse(contact).dot(direct.get_contact_local_normal(contact)))/delta
		collision_impulse_estimates[key] = estimate
		touching[key] = count > 0
	for group in specification.get("truck_load_groups",{}).values():
		var key: String = group.suspension_joint
		var definition: Dictionary = specification.joints[key]
		var s := joint_state(key)
		var stem: String = str(definition.body).trim_suffix("_susp")
		var force: float = -float(_jel_state[stem].lift_vertical_n) if _jel_state.has(stem) else -float(definition.preload_n)-float(definition.stiffness)*(s.q-float(targets.get(key,0.)))-float(definition.damping)*s.velocity
		if not _jel_state.has(stem):force = clampf(force,-float(definition.max_force),float(definition.max_force))
		var load_n := maxf(0.,float(group.mass_below_suspension_kg)*float(specification.gravity)-force*(1. if _jel_state.has(stem) else s.axis.y))
		var count := 0
		for wheel in group.wheels:
			if touching[wheel]: count += 1
		if count > 0:
			for wheel in group.wheels:
				if touching[wheel]: normal_loads[wheel] = load_n/count

func _apply_cables(delta: float) -> void:
	var previous := last_tensions.duplicate()
	last_tensions.clear()
	cable_reactions.clear()
	for index in range(cables.size()):
		var cable: Dictionary = cables[index]
		var tension_before: float = previous[index] if index < previous.size() else 0.
		var maximum_speed: float = cable.get("loaded_max_speed",cable.max_speed) if attachments.has(cable.hook) else cable.max_speed
		var speed := minf(maximum_speed,float(cable.power_w)/maxf(tension_before,1.))
		var payout := clampf((float(cable.target_length)-float(cable.length))*2.,-speed,speed)
		if tension_before > float(cable.working_load_n):
			payout = maxf(payout,minf(float(cable.max_speed),.3*(tension_before/float(cable.working_load_n)-1.)))
		cable.length = clampf(float(cable.length)+payout*delta,1.5,25.)
		cable.payout_speed = payout
		var anchor_body: RigidBody3D = bodies[cable.body]
		var hook: RigidBody3D = bodies[cable.hook]
		var p: Vector3 = anchor_body.global_transform*cable.local_anchor
		var q: Vector3 = get_design_transform(cable.hook).origin
		var tension := _apply_tether(anchor_body,hook,p,q,float(cable.length),float(cable.stiffness),float(cable.damping),float(cable.payout_speed))
		last_tensions.append(tension)
		cable_reactions.append({"body":cable.body,"point":p,"force":(q-p).normalized()*tension})

func _apply_tether(a: RigidBody3D,b: RigidBody3D,p: Vector3,q: Vector3,length: float,stiffness: float,damping: float,payout: float = 0.) -> float:
	var difference := q-p
	var distance := difference.length()
	if distance <= length or distance < .0000001: return 0.0
	var axis := difference/distance
	var va := a.linear_velocity+a.angular_velocity.cross(p-a.global_position)
	var vb := b.linear_velocity+b.angular_velocity.cross(q-b.global_position)
	var separation := (vb-va).dot(axis)
	var tension := maxf(0.,stiffness*(distance-length)+damping*(separation-payout))
	a.apply_force(axis*tension,p-a.global_position)
	b.apply_force(-axis*tension,q-b.global_position)
	return tension

func _apply_restraints() -> void:
	if cargo_backend_started and use_native_cargo != cargo_backend_was_native:
		push_error("Cargo backend cannot change during an episode");set_physics_process(false);return
	cargo_backend_started = true;cargo_backend_was_native = use_native_cargo
	if use_native_cargo:
		if native_cargo == null:
			var status := GDExtensionManager.load_extension("res://native/leviathan_jel.gdextension")
			if status != GDExtensionManager.LOAD_STATUS_OK and status != GDExtensionManager.LOAD_STATUS_ALREADY_LOADED:
				push_error("Native cargo extension load failed");set_physics_process(false);return
			native_cargo = ClassDB.instantiate("LeviathanCargo")
			if native_cargo == null or native_cargo.abi_info().precision != "double" or not OS.has_feature("double"):
				push_error("Native cargo candidate requires matching double engine/ABI");set_physics_process(false);return
			if not native_cargo.configure(restraints,bodies,design_offsets):
				push_error("Native cargo configuration failed");set_physics_process(false);return
		var batch:Dictionary = native_cargo.step(time_s)
		if not batch.has("tensions") or batch.tensions.size() != restraints.size():
			push_error("Native cargo step failed");set_physics_process(false);return
		last_restraint_tensions = batch.tensions
		connector_failures.append_array(batch.failures)
	else:
		last_restraint_tensions.clear()
		for key in restraints:
			var record: Dictionary = restraints[key]
			var forces: Array[float] = []
			if record.secured:
				var cargo: RigidBody3D = bodies[key]
				var support: RigidBody3D = bodies[record.support]
				for strap in record.straps:
					var p: Vector3 = support.global_transform*strap.support_local
					var q: Vector3 = cargo.global_transform*strap.cargo_local
					forces.append(_apply_tether(support,cargo,p,q,float(strap.length),float(strap.stiffness),float(strap.damping)))
				if batch_cargo_connectors:
					forces.append_array(_apply_corner_group(str(record.support),str(key),record))
				else:
					for connector in record.get("connectors",[]):
						var force := _apply_connector(str(record.support),str(key),connector)
						if force < 0.:
							record.secured = false
							connector_failures.append({"time_s":time_s,"cargo":key,"type":"twistlock_displacement"})
							break
						forces.append(force)
			last_restraint_tensions[key] = forces
	for key in attachments.keys():
		var attachment: Dictionary = attachments[key]
		for connector in attachment.connectors:
			if _apply_connector(key,attachment.cargo,connector) < 0.:
				attachments.erase(key)
				connector_failures.append({"time_s":time_s,"cargo":attachment.cargo,"type":"spreader_displacement"})
				break

func _apply_corner_group(support_id: String,cargo_id: String,record: Dictionary) -> Array[float]:
	# Four corners share the same two body states. Accumulate exact equivalent
	# COM wrenches, avoiding repeated native-property queries and force calls.
	var a: RigidBody3D = bodies[support_id]
	var b: RigidBody3D = bodies[cargo_id]
	var at := a.global_transform
	var bt := b.global_transform
	var rotation: Basis = (at*design_offsets[support_id]).basis
	var inverse := rotation.transposed()
	var av := a.linear_velocity
	var bv := b.linear_velocity
	var aw := a.angular_velocity
	var bw := b.angular_velocity
	var sum_force := Vector3.ZERO
	var sum_a_torque := Vector3.ZERO
	var sum_b_torque := Vector3.ZERO
	var forces: Array[float] = []
	for connector in record.get("connectors",[]):
		var p: Vector3 = at*connector.support_local
		var q: Vector3 = bt*connector.cargo_local
		var error: Vector3 = inverse*(q-p)-gv(connector.rest_offset_local)
		if error.length() > float(connector.break_displacement_m):
			record.secured = false
			connector_failures.append({"time_s":time_s,"cargo":cargo_id,"type":"twistlock_displacement"})
			break
		var velocity := inverse*(bv+bw.cross(q-bt.origin)-av-aw.cross(p-at.origin)-aw.cross(q-p))
		var local: Vector3 = float(connector.stiffness)*error+float(connector.damping)*velocity
		if not connector.get("bilateral",false): local.y = maxf(0.,local.y) if error.y > 0. else 0.
		var force: Vector3 = (rotation*local).limit_length(float(connector.max_force_n))
		sum_force += force
		# (p-COM)xF + (q-p)xF = (q-COM)xF; support-frame
		# couple is retained, so total angular momentum is conserved.
		sum_a_torque += (q-at.origin).cross(force)
		sum_b_torque -= (q-bt.origin).cross(force)
		forces.append(force.length())
	a.apply_central_force(sum_force)
	a.apply_torque(sum_a_torque)
	b.apply_central_force(-sum_force)
	b.apply_torque(sum_b_torque)
	return forces

func _apply_connector(support_id: String,cargo_id: String,connector: Dictionary) -> float:
	var a: RigidBody3D = bodies[support_id]
	var b: RigidBody3D = bodies[cargo_id]
	var p: Vector3 = a.global_transform*connector.support_local
	var q: Vector3 = b.global_transform*connector.cargo_local
	var rotation := get_design_transform(support_id).basis
	var error: Vector3 = rotation.transposed()*(q-p)-gv(connector.rest_offset_local)
	if error.length() > float(connector.break_displacement_m): return -1.
	var va := a.linear_velocity+a.angular_velocity.cross(p-a.global_position)
	var vb := b.linear_velocity+b.angular_velocity.cross(q-b.global_position)
	var velocity := rotation.transposed()*(vb-va-a.angular_velocity.cross(q-p))
	var local: Vector3 = float(connector.stiffness)*error+float(connector.damping)*velocity
	if not connector.get("bilateral",false): local.y = maxf(0.,local.y) if error.y > 0. else 0.
	var force: Vector3 = (rotation*local).limit_length(float(connector.max_force_n))
	a.apply_force(force,p-a.global_position)
	a.apply_torque((q-p).cross(force))
	b.apply_force(-force,q-b.global_position)
	return force.length()

func state() -> Dictionary:
	var result := {"time_s":time_s,"physics_owner":"Godot/Jolt","physics_hz":Engine.physics_ticks_per_second,
		"front_position_m":source(get_design_transform("front").origin),
		"rear_position_m":source(get_design_transform("rear").origin) if bodies.has("rear") else null,"rope_tensions_n":last_tensions.duplicate(),"finite":true,
		"contact_count":0,"joints":{},"rolling_load_estimate_n":normal_loads.duplicate(),
		"collision_impulse_estimates_n":collision_impulse_estimates.duplicate(),
		"normal_load_source":"Native Jolt reaction unavailable; rolling loads estimated from suspension. Contact impulses are collision estimates only.",
		"cargo_restraint_tensions_n":last_restraint_tensions.duplicate(true),"active_locks":locked.duplicate(),"cargo_secured":{}}
	for key in restraints: result.cargo_secured[key] = restraints[key].secured
	for body in bodies.values():
		result.finite = result.finite and body.global_position.is_finite() and body.linear_velocity.is_finite()
		result.contact_count += body.get_contact_count()
	for key in drives:
		var s := joint_state(key)
		result.joints[key] = {"q":s.q,"velocity":s.velocity,"force":drives[key].get("last_force",0.)}
	for hull in ["front","rear"]:
		if not bodies.has(hull):continue
		var converted := CONVERSION.transposed()*get_design_transform(hull).basis*CONVERSION
		result[hull+"_rpy_deg"] = [rad_to_deg(atan2(converted.y.z,converted.z.z)),
			rad_to_deg(asin(clampf(-converted.x.z,-1.,1.))),rad_to_deg(atan2(converted.x.y,converted.x.x))]
	result.cargo_poses = {}
	result.rope_lengths_m = {}
	result.crane_attachments = {}
	result.actuator_integral_forces = wheel_integral.duplicate()
	result.actuator_integral_forces.merge(lift_integral)
	result.connector_failures = connector_failures.duplicate(true)
	result.jel = jel_state.duplicate(true)
	result.jel_failures = jel_failures.duplicate(true)
	if profile_timing and timing_steps > 0:
		result.mean_callback_usec = {}
		for key in timing_usec: result.mean_callback_usec[key] = float(timing_usec[key])/timing_steps
	for cable in cables: result.rope_lengths_m[cable.body] = cable.length
	for key in attachments: result.crane_attachments[key] = attachments[key].cargo
	if include_body_poses: result.bodies = {}
	for key in bodies:
		if not include_body_poses and not specification.bodies[key].has("cargo"): continue
		var design := get_design_transform(key)
		var converted := CONVERSION.transposed()*design.basis*CONVERSION
		var q := converted.orthonormalized().get_rotation_quaternion()
		var pose := {"position":source(design.origin),"quaternion":[q.w,q.x,q.y,q.z]}
		if specification.bodies[key].has("cargo"): result.cargo_poses[key] = pose
		if include_body_poses: result.bodies[key] = pose
	return result

func _exit_tree() -> void:
	active = false
	for rid in joint_rids:
		if rid.is_valid(): PhysicsServer3D.free_rid(rid)
	joint_rids.clear()
