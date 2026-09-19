extends RigidBody3D
# Observation only: never changes force, velocity, transform or sleeping state.
# Jolt contact impulses are estimates for multi-body contacts; retain raw values.
var contact_observations: Array = []
var observed_step := 0.0
func _integrate_forces(state: PhysicsDirectBodyState3D) -> void:
	contact_observations=[]
	observed_step=state.step
	for i in range(state.get_contact_count()):
		var other=state.get_contact_collider_object(i)
		var impulse := state.get_contact_impulse(i)
		var normal := state.get_contact_local_normal(i)
		var point := state.get_contact_local_position(i)
		contact_observations.append({"body":str(other.name),"impulse_godot": [impulse.x,impulse.y,impulse.z],
			"normal_godot":[normal.x,normal.y,normal.z],"point_godot":[point.x,point.y,point.z]})
