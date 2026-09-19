extends Node3D
## Measured-state animation of imported neutral meshes. Never commands physics.
## Canonical group-local vertices remain canonical in the GLB; only their world
## frame receives the canonical-to-Godot rotation (not a second local conversion).
const C := Basis(Vector3(1,0,0),Vector3(0,0,-1),Vector3(0,1,0))
var runtime: Node3D
var motion: Dictionary
var body_nodes: Dictionary = {}
var groups: Dictionary = {}
var belts: Dictionary = {}
var buckets: Dictionary = {}
var body_frames: Dictionary = {}
var joint_values: Dictionary = {}
var rope_lengths: Dictionary = {}
var distances: Dictionary = {}
var valid := false
var failure := ""
var report: Dictionary = {}
var last_update_ms := 0.0

static func vec(a) -> Vector3:
	return Vector3(float(a[0]),float(a[1]),float(a[2]))

static func matrix(a: Array) -> Transform3D:
	return Transform3D(Basis(Vector3(a[0][0],a[1][0],a[2][0]),
		Vector3(a[0][1],a[1][1],a[2][1]),Vector3(a[0][2],a[1][2],a[2][2])),Vector3(a[0][3],a[1][3],a[2][3]))

static func rows_basis(a: Array) -> Basis:
	return Basis(Vector3(a[0][0],a[1][0],a[2][0]),Vector3(a[0][1],a[1][1],a[2][1]),Vector3(a[0][2],a[1][2],a[2][2]))

static func world_frame(canonical: Transform3D) -> Transform3D:
	return Transform3D(C*canonical.basis,C*canonical.origin)

func _index(node: Node,index: Dictionary) -> void:
	index[str(node.name)] = node
	for child in node.get_children(): _index(child,index)

func _meshes(node: Node,result: Array[MeshInstance3D]) -> void:
	if node is MeshInstance3D: result.append(node)
	for child in node.get_children(): _meshes(child,result)

func setup(physics: Node3D,asset: Node3D,mapping: Dictionary) -> bool:
	runtime = physics
	motion = mapping
	var index: Dictionary = {}
	_index(asset,index)
	var missing: Array[String] = []
	for id in mapping.required_bodies:
		if not runtime.bodies.has(id) and not runtime.specification.get("body_aliases",{}).has(id): missing.append("physical body "+str(id))
	for id in mapping.required_joints:
		if not runtime.drives.has(id): missing.append("measured joint "+str(id))
	for id in mapping.body_rest_frames:
		if not runtime.specification.bodies.has(id): continue
		var rest := matrix(mapping.body_rest_frames[id])
		var physical_rest: Dictionary = runtime.specification.bodies[id]
		var q: Array = physical_rest.quaternion
		var axes := Basis(Quaternion(float(q[1]),float(q[2]),float(q[3]),float(q[0])))
		var angle := rest.basis.orthonormalized().get_rotation_quaternion().angle_to(axes.get_rotation_quaternion())
		if rest.origin.distance_to(vec(physical_rest.position)) > .0001 or angle > .0001:
			missing.append("rest-frame version mismatch "+str(id))
	for group in mapping.preserve_groups:
		if not index.has(group.node_name): missing.append("imported group "+str(group.node_name))
	if not missing.is_empty():
		failure = "Visual binding rejected: "+", ".join(missing.slice(0,12))
		push_error(failure)
		return false
	# Fixed cargo and merged hitch frames still own imported geometry. Bind the
	# complete original name set, including aliases, not only mechanism anchors.
	var visual_body_ids: Dictionary = runtime.bodies.duplicate()
	visual_body_ids.merge(runtime.specification.get("body_aliases",{}))
	for id in visual_body_ids:
		if index.has(id): body_nodes[id] = index[id]
	for belt in mapping.belts:
		belts[belt.id] = belt.duplicate(true)
		belts[belt.id]._axes = rows_basis(belt.path_axes_local)
		belts[belt.id]._center = vec(belt.center_local)
	var instances := 0
	for group in mapping.preserve_groups:
		var node: Node3D = index[group.node_name]
		if group.get("instance_family","") == "track_link":
			var children: Array[MeshInstance3D] = []
			_meshes(node,children)
			var bindings: Array[Dictionary] = []
			for child in children:
				var key := str(child.mesh.get_instance_id())+"_"+str(child.material_override.get_instance_id() if child.material_override else 0)
				if not buckets.has(key):
					buckets[key] = {"mesh":child.mesh,"material":child.material_override,"count":0}
				var bucket: Dictionary = buckets[key]
				var local := child.transform
				var ancestor := child.get_parent()
				while ancestor != node:
					local = ancestor.transform*local
					ancestor = ancestor.get_parent()
				bindings.append({"bucket":key,"index":int(bucket.count),"local":local,"identity":local==Transform3D.IDENTITY})
				bucket.count += 1
				instances += 1
			groups[group.id] = {"instances":bindings}
			# Mesh resources are retained by MultiMesh, not regenerated in Godot.
			node.queue_free()
		else:
			groups[group.id] = {"node":node}
	for key in buckets:
		var bucket: Dictionary = buckets[key]
		var multimesh := MultiMesh.new()
		multimesh.transform_format = MultiMesh.TRANSFORM_3D
		multimesh.mesh = bucket.mesh
		multimesh.instance_count = int(bucket.count)
		var node := MultiMeshInstance3D.new()
		node.name = "ImportedBeltInstances_"+str(key)
		node.multimesh = multimesh
		node.material_override = bucket.material
		add_child(node)
		bucket.node = node
		bucket.multimesh = multimesh
	var belt_link_count := 0
	for group in mapping.preserve_groups:
		if group.get("instance_family","") == "track_link": belt_link_count += 1
	report = {"body_bindings":body_nodes.size(),"motion_groups":groups.size(),"belt_links":belt_link_count,
		"multimesh_batches":buckets.size(),"instanced_material_parts":instances,"source":"Imported GLB mesh resources; actual native encoder/body/rope state"}
	valid = true
	return update_measured()

func _point(anchor: Dictionary) -> Vector3:
	var frame: Transform3D = body_frames[anchor.body]
	return frame*vec(anchor.point)

func _stadium(phase: float,half: float,radius: float) -> Vector4:
	var straight := 2.*half
	var arc := PI*radius
	var p := fposmod(phase,1.)*(2.*straight+2.*arc)
	if p < straight: return Vector4(-half+p,radius,1.,0.)
	p -= straight
	if p < arc:
		var angle := PI/2.-p/radius
		return Vector4(half+radius*cos(angle),radius*sin(angle),sin(angle),-cos(angle))
	p -= arc
	if p < straight: return Vector4(half-p,-radius,-1.,0.)
	var angle := -PI/2.-(p-straight)/radius
	return Vector4(-half+radius*cos(angle),radius*sin(angle),sin(angle),-cos(angle))

func update_measured() -> bool:
	if not valid: return false
	var start := Time.get_ticks_usec()
	for id in body_nodes:
		var transform: Transform3D = runtime.get_design_transform(id)
		body_nodes[id].global_transform = transform*Transform3D(C,Vector3.ZERO)
	for id in motion.required_bodies:
		var native: Transform3D = runtime.get_design_transform(id)
		body_frames[id] = Transform3D(C.transposed()*native.basis*C,C.transposed()*native.origin)
	for id in motion.required_joints: joint_values[id] = float(runtime.joint_state(id).q)
	rope_lengths.clear()
	for cable in runtime.cables: rope_lengths[cable.body] = float(cable.length)
	for id in belts:
		var belt: Dictionary = belts[id]
		var total := 0.0
		for joint in belt.joint_ids: total += float(joint_values[joint])
		distances[id] = total/belt.joint_ids.size()*float(belt.drive_radius_m)
		var parent: Transform3D = body_frames[belt.body]
		belt._world_axes = parent.basis*belt._axes
		belt._world_center = parent*belt._center
	for operation in motion.operations:
		var op: Dictionary = operation
		var frame := Transform3D.IDENTITY
		match str(op.kind):
			"belt_link":
				var belt: Dictionary = belts[op.belt]
				var phase: float = float(op.initial_phase)+float(distances[op.belt])/float(belt.circumference_m)
				var path := _stadium(phase,float(belt.half_straight_m),float(belt.radius_m))
				# The stadium already supplies its unit tangent. Construct its
				# basis directly, avoiding atan2 -> sin/cos and two matrix products
				# for each of the 1024 links. This is the same measured encoder pose.
				var axes: Basis = belt._world_axes
				frame = Transform3D(Basis(axes.x*path.z+axes.z*path.w,axes.y,axes.z*path.z-axes.x*path.w),
					belt._world_center+axes.x*path.x+axes.z*path.y)
			"wheel":
				var belt: Dictionary = belts[op.belt]
				frame = body_frames[belt.body]*matrix(op.frame_local)*Transform3D(Basis(vec(op.axis_group),float(distances[op.belt])/float(op.radius_m)),Vector3.ZERO)
			"joint_translation":
				var local := matrix(op.frame_local)
				local.origin += vec(op.axis_local)*float(joint_values[op.joint])*float(op.ratio)
				frame = body_frames[op.reference_body]*local
			"span":
				var a := _point(op.a)
				var b := _point(op.b)
				var direction := b-a
				var length := direction.length()
				var finish: float = length if op.end_m == null else float(op.end_m)
				if length < .000001 or finish <= float(op.start_m)+.000001:
					failure = "Collapsed measured span: "+str(op.group)
					valid = false
					push_error(failure)
					return false
				direction /= length
				var reference_frame: Transform3D = body_frames[op.reference_body]
				var x: Vector3 = reference_frame.basis*vec(op.reference_local)
				x -= direction*x.dot(direction)
				if x.length() < .000001: x = direction.cross(Vector3.UP if absf(direction.y)<.9 else Vector3.RIGHT)
				x = x.normalized()
				var y := direction.cross(x)
				var span_length := finish-float(op.start_m)
				frame = Transform3D(Basis(x,y,direction*span_length/float(op.rest_length_m)),a+direction*(finish+float(op.start_m))/2.)
			"cable_drum":
				if not rope_lengths.has(op.rope_state_id):
					failure = "Missing measured cable payout: "+str(op.rope_state_id)
					valid = false
					return false
				var angle: float = (float(rope_lengths[op.rope_state_id])-float(op.rest_length_m))/float(op.radius_m)
				frame = body_frames[op.reference_body]*matrix(op.frame_local)*Transform3D(Basis(vec(op.axis_group),angle),Vector3.ZERO)
			_:
				failure = "Unknown neutral motion operation: "+str(op.kind)
				valid = false
				return false
		if not frame.origin.is_finite():
			failure = "Nonfinite measured visual frame"
			valid = false
			return false
		var converted := world_frame(frame)
		var binding: Dictionary = groups[op.group]
		if binding.has("node"):
			binding.node.global_transform = converted
		else:
			for instance in binding.instances:
				var multimesh: MultiMesh = buckets[instance.bucket].multimesh
				multimesh.set_instance_transform(int(instance.index),converted if instance.identity else converted*instance.local)
	last_update_ms = float(Time.get_ticks_usec()-start)/1000.
	return true

func measured_binding_frames() -> Dictionary:
	# Read the rendered nodes/instances back for comparison against the independent
	# Python evaluator. This checks the actual assignments, including mesh-local
	# transforms and the glTF axis boundary, not only our computed operation data.
	var result: Dictionary = {}
	for id in groups:
		var binding: Dictionary = groups[id]
		var rendered: Transform3D
		if binding.has("node"):
			rendered = binding.node.global_transform
		else:
			var instance: Dictionary = binding.instances[0]
			var bucket: Dictionary = buckets[instance.bucket]
			rendered = bucket.node.global_transform*bucket.multimesh.get_instance_transform(int(instance.index))*instance.local.affine_inverse()
		var frame := Transform3D(C.transposed()*rendered.basis,C.transposed()*rendered.origin)
		result[id] = [[frame.basis.x.x,frame.basis.y.x,frame.basis.z.x,frame.origin.x],
			[frame.basis.x.y,frame.basis.y.y,frame.basis.z.y,frame.origin.y],
			[frame.basis.x.z,frame.basis.y.z,frame.basis.z.z,frame.origin.z],[0.,0.,0.,1.]]
	return result
