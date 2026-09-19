extends Node3D
## 6DOF Y spring at a point OFFSET from the dynamic COM (static node_a).
## If this sags mg/k, sole-corner springs on a foot are valid.

var _box: RigidBody3D
var _t: float = 0.0
var _printed: Dictionary = {}


func _ready() -> void:
	Engine.max_physics_steps_per_frame = 1
	var floor := StaticBody3D.new()
	floor.name = "Floor"
	floor.position = Vector3.ZERO
	floor.collision_layer = 0
	floor.collision_mask = 0
	add_child(floor)

	_box = RigidBody3D.new()
	_box.name = "Mass"
	_box.mass = 1.0
	_box.can_sleep = false
	_box.gravity_scale = 1.0
	_box.position = Vector3(0, 0.05, 0)
	_box.center_of_mass_mode = RigidBody3D.CENTER_OF_MASS_MODE_CUSTOM
	_box.center_of_mass = Vector3.ZERO
	var csh := CollisionShape3D.new()
	var cbox := BoxShape3D.new()
	cbox.size = Vector3(0.1, 0.1, 0.1)
	csh.shape = cbox
	csh.disabled = true
	_box.add_child(csh)
	add_child(_box)

	var j := Generic6DOFJoint3D.new()
	j.name = "Spring"
	# Joint at the bottom of the box (COM is 5 cm above), child of static floor
	# so the frame stays world-aligned.
	j.position = Vector3(0, 0.0, 0)
	floor.add_child(j)
	j.node_a = j.get_path_to(floor)
	j.node_b = j.get_path_to(_box)
	for axis in ["x", "y", "z"]:
		j.set("linear_limit_%s/enabled" % axis, false)
		j.set("angular_limit_%s/enabled" % axis, false)
		j.set("linear_spring_%s/enabled" % axis, true)
		j.set("angular_spring_%s/enabled" % axis, false)
	j.set("linear_spring_x/stiffness", 1.0e5)
	j.set("linear_spring_x/damping", 200.0)
	j.set("linear_spring_z/stiffness", 1.0e5)
	j.set("linear_spring_z/damping", 200.0)
	j.set("linear_spring_y/stiffness", 2000.0)
	j.set("linear_spring_y/damping", 89.0)
	j.set("linear_spring_y/equilibrium_point", 0.0)
	print("spring_y_offset start box_y=%.4f joint_y=0 expect_sag=0.0049" % _box.position.y)


func _physics_process(delta: float) -> void:
	_t += delta
	var y := _box.global_position.y
	for tmark in [0.05, 0.20, 0.50, 1.00, 1.50, 2.00]:
		var key := str(tmark)
		if _t + 1e-6 >= tmark and not _printed.has(key):
			_printed[key] = true
			print("t=%.2f box_y=%.4f vy=%.3f" % [_t, y, _box.linear_velocity.y])
	if _t >= 2.05:
		print("spring_y_offset done")
		get_tree().quit(0)
