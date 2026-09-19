extends "res://physics_server.gd"

# Upstream infer_policy.py applies 0.003 N m bearing friction to passive wheels.
# A zero-speed bounded hinge motor supplies dissipative constraint friction.
# Jolt converts Godot's max impulse to torque by dividing by the physics step.
const BEARING_FRICTION := 0.003

func _apply_pd() -> void:
	super._apply_pd()
	var impulse := BEARING_FRICTION * Engine.time_scale / float(Engine.physics_ticks_per_second)
	for joint in _joints:
		if not str(joint["name"]).begins_with("passive_"):
			continue
		var hinge: HingeJoint3D = joint["node"]
		hinge.set_flag(HingeJoint3D.FLAG_ENABLE_MOTOR, true)
		hinge.set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, 0.0)
		hinge.set_param(HingeJoint3D.PARAM_MOTOR_MAX_IMPULSE, impulse)
