extends Node3D
## Scene-independent wrapper for a CAD/Blender bundle. No authored mesh creation.
var runtime: Node3D
var binding: Node3D
var asset: Node3D
var valid := false
var failure := ""

func load_bundle(directory: String, placement: Transform3D = Transform3D.IDENTITY, visuals: bool = true) -> bool:
	if runtime != null:
		failure = "A vehicle bundle can only be initialized once."
		return false
	# Gravity remains vertical. Placement supports translation/yaw, with no scale.
	if not placement.basis.is_equal_approx(placement.basis.orthonormalized()) or not placement.basis.y.is_equal_approx(Vector3.UP) or placement.basis.determinant() < .999:
		failure = "Vehicle placement must be a rigid translation/yaw transform."
		return false
	var manifest_path := directory.path_join("physics.json")
	if not FileAccess.file_exists(manifest_path):
		failure = "Vehicle physics manifest is missing."
		return false
	var manifest = JSON.parse_string(FileAccess.get_file_as_string(manifest_path))
	if not manifest is Dictionary:
		failure = "Vehicle physics manifest is invalid."
		return false
	runtime = load(directory.path_join("physics_runtime.gd")).new()
	runtime.name = "NativeCarrier"
	add_child(runtime)
	runtime.setup(manifest)
	# Initial placement only, before any simulation tick. Native joint/force
	# attachment frames are local, so they remain on the same physical parts.
	for body in runtime.bodies.values():
		body.global_transform = placement*body.global_transform
	if visuals:
		var path := directory.path_join("leviathan.glb")
		if not ResourceLoader.exists(path):
			failure = "Import the authored GLB with the Godot editor before loading."
			runtime.active = false
			return false
		asset = load(path).instantiate()
		add_child(asset)
		binding = load(directory.path_join("visual_binding.gd")).new()
		add_child(binding)
		if not binding.setup(runtime,asset,JSON.parse_string(FileAccess.get_file_as_string(directory.path_join("visual_motion.json")))):
			failure = binding.failure
			runtime.active = false
			return false
	valid = true
	return true

func _process(_delta: float) -> void:
	if valid and binding != null and not binding.update_measured():
		valid = false
		failure = binding.failure
		runtime.set_vehicle_command(0.,0.)
