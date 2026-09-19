extends Node3D
## Does Godot 4.7 built-in Jolt actually apply Generic6DOF linear springs?

var _box: RigidBody3D
var _t: float = 0.0
var _printed: Dictionary = {}


func _ready() -> void:
	Engine.max_physics_steps_per_frame = 1
	var anchor := StaticBody3D.new()
	anchor.name = "Anchor"
	anchor.position = Vector3(0, 0, 0)
	var ash := CollisionShape3D.new()
	var abox := BoxShape3D.new()
	abox.size = Vector3(0.2, 0.2, 0.2)
	ash.shape = abox
	ash.disabled = true
	anchor.add_child(ash)
	add_child(anchor)

	_box = RigidBody3D.new()
	_box.name = "Mass"
	_box.mass = 1.0
	_box.can_sleep = false
	_box.gravity_scale = 1.0
	_box.position = Vector3(0, 0.0, 0)
	var csh := CollisionShape3D.new()
	var cbox := BoxShape3D.new()
	cbox.size = Vector3(0.1, 0.1, 0.1)
	csh.shape = cbox
	_box.add_child(csh)
	add_child(_box)

	var j := Generic6DOFJoint3D.new()
	j.name = "Spring"
	anchor.add_child(j)
	j.node_a = j.get_path_to(anchor)
	j.node_b = j.get_path_to(_box)
	for axis in ["x", "y", "z"]:
		j.set("linear_limit_%s/enabled" % axis, false)
		j.set("angular_limit_%s/enabled" % axis, false)
		j.set("linear_spring_%s/enabled" % axis, true)
		j.set("angular_spring_%s/enabled" % axis, true)
		j.set("angular_spring_%s/stiffness" % axis, 1.0e5)
		j.set("angular_spring_%s/damping" % axis, 100.0)
	j.set("linear_spring_x/stiffness", 1.0e5)
	j.set("linear_spring_x/damping", 200.0)
	j.set("linear_spring_z/stiffness", 1.0e5)
	j.set("linear_spring_z/damping", 200.0)
	# k=2000 N/m → 1 kg sag mg/k = 4.9 mm if the spring is real
	j.set("linear_spring_y/stiffness", 2000.0)
	j.set("linear_spring_y/damping", 89.0)
	j.set("linear_spring_y/equilibrium_point", 0.0)
	print("spring_y_test start y=%.4f k=2000 expect_sag=0.0049" % _box.position.y)


func _physics_process(delta: float) -> void:
	_t += delta
	var y := _box.global_position.y
	for tmark in [0.05, 0.20, 0.50, 1.00, 1.50, 2.00]:
		var key := str(tmark)
		if _t + 1e-6 >= tmark and not _printed.has(key):
			_printed[key] = true
			print("t=%.2f y=%.4f vy=%.3f" % [_t, y, _box.linear_velocity.y])
	if _t >= 2.05:
		print("spring_y_test done")
		get_tree().quit(0)
