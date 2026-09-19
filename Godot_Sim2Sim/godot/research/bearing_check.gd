extends Node3D

var wheel: RigidBody3D
var elapsed := 0.0
var samples: Array = []

func _ready() -> void:
	var anchor := StaticBody3D.new()
	anchor.name = "Anchor"
	add_child(anchor)
	wheel = RigidBody3D.new()
	wheel.name = "Wheel"
	wheel.mass = 0.1
	wheel.inertia = Vector3(0.0001, 0.0001, 0.0001)
	wheel.gravity_scale = 0.0
	wheel.angular_damp_mode = RigidBody3D.DAMP_MODE_REPLACE
	wheel.angular_damp = 0.0
	wheel.can_sleep = false
	wheel.collision_layer = 0
	wheel.collision_mask = 0
	var shape := CollisionShape3D.new()
	shape.shape = SphereShape3D.new()
	(shape.shape as SphereShape3D).radius = 0.05
	wheel.add_child(shape)
	add_child(wheel)
	var hinge := HingeJoint3D.new()
	hinge.node_a = NodePath("../Anchor")
	hinge.node_b = NodePath("../Wheel")
	add_child(hinge)
	hinge.set_flag(HingeJoint3D.FLAG_USE_LIMIT, false)
	hinge.set_flag(HingeJoint3D.FLAG_ENABLE_MOTOR, true)
	hinge.set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, 0.0)
	hinge.set_param(HingeJoint3D.PARAM_MOTOR_MAX_IMPULSE, 0.003 / float(Engine.physics_ticks_per_second))
	wheel.angular_velocity = Vector3(0, 0, 40)

func _physics_process(delta: float) -> void:
	samples.append([elapsed, wheel.angular_velocity.z])
	elapsed += delta
	if elapsed >= 1.01:
		print("BEARING_CHECK " + JSON.stringify(samples))
		get_tree().quit()
