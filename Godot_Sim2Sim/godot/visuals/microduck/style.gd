extends RefCounted
## Startup-only presentation. No geometry, physics, camera or process callbacks.
const PALETTE := {
	"graphite": Color("6c6a68"), "yellow": Color("e0bd38"),
	"purple": Color("89729e"), "rubber": Color("39383d"),
	"metal": Color("969b95"), "floor": Color("b2b1a2"),
	"paper": Color("c6c4b7")
}
var materials: Dictionary = {}
var parts: Dictionary = {}

func apply(server: Node) -> void:
	if DisplayServer.get_name() == "headless":
		return
	var enamel: Shader = load("res://visuals/microduck/enamel.gdshader")
	var ink: Shader = load("res://visuals/microduck/ink.gdshader")
	for key in PALETTE:
		var mat := ShaderMaterial.new()
		mat.shader = enamel
		mat.set_shader_parameter("pigment", PALETTE[key])
		mat.set_shader_parameter("hatch_strength", 0.03)
		if OS.get_environment("SIM2SIM_VISUAL_INK") != "0":
			var outline := ShaderMaterial.new()
			outline.shader = ink
			outline.set_shader_parameter("line_pixels", 0.60)
			mat.next_pass = outline
		materials[key] = mat
	# Exact compiled-scene identities avoid applying old mesh IDs to a new robot.
	var catalog: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://visuals/microduck/robot_parts.json"))
	parts = catalog.get(FileAccess.get_sha256(server._robot_scene), {})
	if server._robot != null:
		_paint(server._robot, server)
	_tone_environment(server)
	_tone_floor(server)
	server.set_meta("microduck_visual_style", true)

func _paint(node: Node, server: Node) -> void:
	if node is MeshInstance3D and str(node.name).begins_with("vis_"):
		var mesh_instance := node as MeshInstance3D
		var role: String = server._paint_role_for_node(mesh_instance)
		var key: String = {"shell":"graphite", "trim":"yellow", "accent":"purple", "mech":"rubber"}.get(role, "graphite")
		var part: String = parts.get(str(server._mesh_id(mesh_instance)), "")
		if not part.is_empty():
			key = "graphite"
			if part in ["jaw", "soft_mouth_top"]:
				key = "yellow"
			elif part in ["foot_left", "foot_right", "noenoeil"]:
				key = "purple"
			elif part.begins_with("seeed_bearing") or part == "upper_leg_rigidity_plate":
				key = "metal"
			elif part in ["xl330", "np_f970", "lens", "m12_lens_holder", "speaker", "sole_left", "sole_right", "tire"] or "pcb" in part:
				key = "rubber"
		var mat: Material = materials[key]
		if part == "lens":
			var lens := ShaderMaterial.new()
			lens.shader = load("res://visuals/microduck/lens.gdshader")
			lens.set_shader_parameter("aperture_radius", 0.0045)
			lens.set_shader_parameter("reflection_strength", 0.65)
			mat = lens
		if mesh_instance.mesh != null:
			for surface in range(mesh_instance.mesh.get_surface_count()):
				mesh_instance.set_surface_override_material(surface, mat)
	for child in node.get_children():
		_paint(child, server)

func _tone_environment(server: Node) -> void:
	var world_environment := server.get_node_or_null("WorldEnvironment") as WorldEnvironment
	if world_environment != null and world_environment.environment != null:
		var environment: Environment = world_environment.environment.duplicate()
		environment.background_mode = Environment.BG_COLOR
		environment.background_color = Color("babfb3")
		environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
		environment.ambient_light_color = Color("c9c4ce")
		environment.ambient_light_energy = 0.45
		environment.tonemap_mode = Environment.TONE_MAPPER_LINEAR
		environment.tonemap_exposure = 1.0
		environment.fog_light_color = Color("babfb3")
		world_environment.environment = environment
	var sun := server.get_node_or_null("World/Sun") as DirectionalLight3D
	if sun != null:
		sun.light_color = Color("fff9ed")
		sun.light_energy = 0.52
		# Metre-scale default bias detaches shadows from this 25 cm robot,
		# especially in Forward+. Keep normal bias for self-shadow acne.
		sun.shadow_bias = 0.005
	var fill := server.get_node_or_null("World/FillLight") as DirectionalLight3D
	if fill != null:
		fill.light_color = Color("c9c4ce")
		fill.light_energy = 0.08

func _tone_floor(server: Node) -> void:
	var floor_mesh := server.get_node_or_null("World/Floor/FloorMesh") as MeshInstance3D
	if floor_mesh == null or floor_mesh.mesh == null:
		return
	var original := floor_mesh.get_active_material(0) as StandardMaterial3D
	if original == null:
		return
	# Keep the original checker size and UVs: only replace its two colors.
	var mat: StandardMaterial3D = original.duplicate()
	var checker := Image.create(2, 2, false, Image.FORMAT_RGBA8)
	checker.set_pixel(0, 0, PALETTE.floor)
	checker.set_pixel(1, 1, PALETTE.floor)
	checker.set_pixel(0, 1, PALETTE.paper)
	checker.set_pixel(1, 0, PALETTE.paper)
	mat.albedo_texture = ImageTexture.create_from_image(checker)
	floor_mesh.set_surface_override_material(0, mat)
