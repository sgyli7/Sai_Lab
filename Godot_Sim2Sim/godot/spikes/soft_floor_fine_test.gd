extends Node3D
## Large SoftBody sheet + ~5 mm verts vs a 45×20 mm foot plate.
##
## 12 cm plate sits on ~4 cm verts (soft_floor_test.tscn). This file is the
## missing combo: enough verts under one real foot. It does not sit.
##
## Tried here (all punch or trampoline, ymin metres below the sheet):
## - 8 cm pin lattice, total_mass=8
## - 20 mm pin lattice, total_mass=8 (brief hit then fall)
## - 40 mm pin *lines*, total_mass=8 (launch then fall)
## - 8 cm lattice, total_mass=400 (per-vertex mass ≈ the 12 cm plate)
## All-pin is still forbidden (zero Rigid contacts). Do not wire into main.tscn.

const PIN_SPAN := 0.08


func _ready() -> void:
	Engine.max_physics_steps_per_frame = 1
	var sb := $SoftFloor as SoftBody3D
	var mesh: Mesh = sb.mesh
	var verts: PackedVector3Array = mesh.surface_get_arrays(0)[Mesh.ARRAY_VERTEX]
	var xmin := 1.0e9
	var xmax := -1.0e9
	var zmin := 1.0e9
	var zmax := -1.0e9
	for v in verts:
		xmin = minf(xmin, v.x)
		xmax = maxf(xmax, v.x)
		zmin = minf(zmin, v.z)
		zmax = maxf(zmax, v.z)
	var n_pin := 0
	for i in range(verts.size()):
		var v: Vector3 = verts[i]
		var on_edge := (
			v.x <= xmin + 1e-4 or v.x >= xmax - 1e-4
			or v.z <= zmin + 1e-4 or v.z >= zmax - 1e-4
		)
		var on_grid := (
			absf(fmod(v.x - xmin, PIN_SPAN)) < 0.003
			and absf(fmod(v.z - zmin, PIN_SPAN)) < 0.003
		)
		if on_edge or on_grid:
			sb.set_point_pinned(i, true)
			n_pin += 1
	var dx := (xmax - xmin) / maxf(1.0, sqrt(float(verts.size())) - 1.0)
	print("soft_floor_fine nvert=%s npin=%s spacing~%.4f rid=%s" % [verts.size(), n_pin, dx, sb.get_physics_rid()])

	var box := RigidBody3D.new()
	box.name = "Mass"
	box.mass = 0.37
	box.can_sleep = false
	box.position = Vector3(0, 0.04, 0)
	var csh := CollisionShape3D.new()
	var cbox := BoxShape3D.new()
	cbox.size = Vector3(0.045, 0.02, 0.020)
	csh.shape = cbox
	box.add_child(csh)
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
			print("t=%.2f y=%.4f vy=%.3f ymin=%.4f" % [_t, y, box.linear_velocity.y, _ymin])
	if _t >= 2.05:
		print("soft_floor_fine done ymin=%.4f (foot plate punches; 12cm plate sits)" % _ymin)
		get_tree().quit(0)
