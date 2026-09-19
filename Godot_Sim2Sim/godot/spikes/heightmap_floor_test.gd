extends Node3D
## HeightMap egg-carton vs a 0.37 kg foot-sized box (Jolt).
##
## Spike: A=1.2 mm, λ=30 mm, 1 cm cells → box sits (y≈0.0107).
## C3 on the robot: stance ~0.10 m puts one foot on peaks and the other
## in valleys; the valley foot tunnels (ankle z≈−0.016). t=0.80 tilt 86°
## with hip contact, jaw x≈+0.03. Not MJ C3. Do not put this in main.tscn.

const A := 0.0012  # 1.2 mm amplitude (6x slop)
const LX := 0.03
const LZ := 0.03
const CELL := 0.01
const N := 81  # (N-1)*CELL = 0.8 m


func _ready() -> void:
	Engine.max_physics_steps_per_frame = 1
	var floor_b := StaticBody3D.new()
	floor_b.name = "Floor"
	floor_b.collision_layer = 1
	floor_b.collision_mask = 1
	var mat := PhysicsMaterial.new()
	mat.friction = 1.0
	mat.bounce = 0.0
	floor_b.physics_material_override = mat
	var csh := CollisionShape3D.new()
	csh.scale = Vector3(CELL, 1.0, CELL)
	var hm := HeightMapShape3D.new()
	hm.map_width = N
	hm.map_depth = N
	var data := PackedFloat32Array()
	data.resize(N * N)
	var half := float(N - 1) * 0.5
	for j in range(N):
		for i in range(N):
			var x := (float(i) - half) * CELL
			var z := (float(j) - half) * CELL
			data[j * N + i] = A * sin(TAU * x / LX) * sin(TAU * z / LZ)
	hm.map_data = data
	csh.shape = hm
	floor_b.add_child(csh)
	add_child(floor_b)
	print("heightmap A=%.4f lx=%.3f lz=%.3f cell=%.3f n=%s h0=%.5f hmax=%.5f" % [
		A, LX, LZ, CELL, N, data[(N * N) / 2], hm.get_max_height(),
	])

	var box := RigidBody3D.new()
	box.name = "Mass"
	box.mass = 0.37
	box.can_sleep = false
	box.position = Vector3(0, 0.04, 0)
	var bsh := CollisionShape3D.new()
	var bbox := BoxShape3D.new()
	bbox.size = Vector3(0.045, 0.02, 0.020)
	bsh.shape = bbox
	box.add_child(bsh)
	add_child(box)


var _t: float = 0.0
var _printed: Dictionary = {}
var _ymin: float = 1.0e9


func _physics_process(delta: float) -> void:
	_t += delta
	var box := get_node_or_null("Mass") as RigidBody3D
	if box == null:
		return
	var y := box.global_position.y
	_ymin = minf(_ymin, y)
	for tmark in [0.12, 0.50, 1.00, 2.00]:
		var key := str(tmark)
		if _t + 1e-6 >= tmark and not _printed.has(key):
			_printed[key] = true
			print("t=%.2f y=%.4f vy=%.3f ymin=%.4f xz=(%.4f,%.4f)" % [
				_t, y, box.linear_velocity.y, _ymin, box.global_position.x, box.global_position.z,
			])
	if _t >= 2.05:
		print("heightmap done ymin=%.4f" % _ymin)
		get_tree().quit(0)
