extends RefCounted
## Read-only support observations. All body motion remains native Jolt physics.
var support_id:="world"
var unsupported_s:=0.0

func frame(adapter:Node3D,id:String)->Dictionary:
	if id=="world" or not adapter.vehicle.runtime.bodies.has(id):
		return {"pose":Transform3D.IDENTITY,"velocity":Vector3.ZERO,"omega":Vector3.ZERO}
	var body:RigidBody3D=adapter.vehicle.runtime.bodies[id]
	var pose:Transform3D=adapter.vehicle.runtime.get_design_transform(id)
	var com:Vector3=body.global_transform*body.center_of_mass
	return {"pose":pose,"velocity":body.linear_velocity+body.angular_velocity.cross(pose.origin-com),"omega":body.angular_velocity}

func observe(adapter:Node3D,robot:Node3D,specification:Dictionary,state:Dictionary,requested_frames:Array=[])->void:
	state.physics_owner="Godot/Jolt";state.robot_id=specification.robot_id
	var measured:String=adapter.support_body_from_contacts(robot,specification.leg_order)
	var ground_contact:=false
	for leg in specification.leg_order:
		for other in robot.bodies[str(leg)+"_wheel"].get_colliding_bodies():
			if not adapter.vehicle.runtime.bodies.values().has(other) and not robot.bodies.values().has(other):ground_contact=true
	if measured!="world" or ground_contact:
		support_id=measured;unsupported_s=0.
	else:
		unsupported_s+=.02
		if unsupported_s>.2:support_id="world"
	var support:=frame(adapter,support_id)
	var pose:Transform3D=support.pose
	var base:RigidBody3D=robot.bodies.chassis
	var com:Vector3=base.global_transform*base.center_of_mass
	var velocity:Vector3=base.linear_velocity+base.angular_velocity.cross(base.global_position-com)
	# Retain the host's untranslated-world-axis task frame for FK and grabs.
	state.base_position=robot.source(base.global_position-robot.global_position)
	state.base_linear_world=robot.source(velocity)
	var relative:Transform3D=pose.affine_inverse()*base.global_transform
	var rotation:Array=[]
	for axis in [Vector3.RIGHT,Vector3.FORWARD,Vector3.UP]:rotation.append(robot.source(relative.basis*axis))
	var point_velocity:Vector3=support.velocity+support.omega.cross(base.global_position-pose.origin)
	var observation:Dictionary={"body_id":support_id,"base_position":robot.source(relative.origin),
		"base_rotation_columns":rotation,
		"base_linear_world":robot.source(pose.basis.transposed()*(velocity-point_velocity)),
		"base_angular_world":robot.source(pose.basis.transposed()*(base.angular_velocity-support.omega)),
		"projected_up_body":robot.source(base.global_basis.transposed()*Vector3.UP)}
	var direction:Vector3=relative.basis*Vector3.RIGHT
	var yaw:=atan2(-direction.z,direction.x)
	observation.terrain_heights=_scan(adapter,robot,pose,yaw,[-.36,-.18,0.,.18,.36,.54,.72,.9],[-.24,0.,.24])
	observation.terrain_path_heights=_scan(adapter,robot,pose,yaw,[-.18,0.,.18,.36,.54],[-.16,0.,.16])
	state.support_frame=observation
	state.support_frames={}
	var ids:Array=["world",support_id,"front","rear"]
	for id in requested_frames:
		if not ids.has(id):ids.append(id)
	for id in ids:
		var current:Dictionary=frame(adapter,str(id))
		var current_pose:Transform3D=current.pose
		var columns:Array=[]
		for axis in [Vector3.RIGHT,Vector3.FORWARD,Vector3.UP]:columns.append(robot.source(current_pose.basis*axis))
		state.support_frames[id]={"origin_m":robot.source(current_pose.origin-robot.global_position),"rotation_columns":columns}

func _scan(adapter:Node3D,robot:Node3D,pose:Transform3D,yaw:float,xs:Array,ys:Array)->Array:
	var heights:Array=[]
	var base:RigidBody3D=robot.bodies.chassis
	for x in xs:
		for y in ys:
			var origin:Vector3=base.global_position+pose.basis*Vector3(cos(yaw)*x-sin(yaw)*y,1.,-sin(yaw)*x-cos(yaw)*y)
			var query:=PhysicsRayQueryParameters3D.create(origin,origin-2.*pose.basis.y,adapter.DRIVING_LAYER)
			var hit:Dictionary=robot.get_world_3d().direct_space_state.intersect_ray(query)
			heights.append((pose.affine_inverse()*hit.position).y if not hit.is_empty() else -1.)
	return heights
