extends RefCounted
## Extra collision rays for deciding whether a step lies in the wheel corridor.
## The actor's published 24 height observations remain unchanged.
static func wheel_path(scene: Node3D, collision_mask: int) -> Array:
	var base: RigidBody3D = scene.robot.bodies.chassis
	var direction: Vector3 = base.global_basis * Vector3.RIGHT
	var yaw := atan2(-direction.z, direction.x)
	var heights: Array = []
	for x in [-.18, 0., .18, .36, .54]:
		for y in [-.16, 0., .16]:
			var sx: float = base.global_position.x + cos(yaw)*x - sin(yaw)*y
			var sz: float = base.global_position.z - sin(yaw)*x - cos(yaw)*y
			var query := PhysicsRayQueryParameters3D.create(Vector3(sx,base.global_position.y+1.,sz),Vector3(sx,base.global_position.y-1.,sz),collision_mask)
			var hit := scene.get_world_3d().direct_space_state.intersect_ray(query)
			heights.append(float(hit.position.y)-scene.global_position.y if not hit.is_empty() else -.002)
	return heights

static func edge_profile(scene: Node3D, collision_mask: int) -> Array:
	var base: RigidBody3D = scene.robot.bodies.chassis
	var direction: Vector3 = base.global_basis * Vector3.RIGHT
	var yaw := atan2(-direction.z, direction.x)
	var heights: Array = []
	for i in range(46):
		var x: float = -.30 + .02*i
		for y in [-.146, 0., .146]:
			var sx: float = base.global_position.x + cos(yaw)*x - sin(yaw)*y
			var sz: float = base.global_position.z - sin(yaw)*x - cos(yaw)*y
			heights.append(_height(scene, Vector3(sx,base.global_position.y,sz), collision_mask))
	return heights

static func wheel_ground(scene: Node3D, collision_mask: int) -> Array:
	var heights: Array = []
	for leg in scene.specification.leg_order:
		var wheel: RigidBody3D = scene.robot.bodies[str(leg)+"_wheel"]
		heights.append(_height(scene,wheel.global_position,collision_mask))
	return heights

static func _height(scene: Node3D, point: Vector3, collision_mask: int) -> float:
	var query := PhysicsRayQueryParameters3D.create(point+Vector3.UP, point-Vector3.UP, collision_mask)
	var hit := scene.get_world_3d().direct_space_state.intersect_ray(query)
	return float(hit.position.y)-scene.global_position.y if not hit.is_empty() else -.002
