extends SceneTree
## Bounded integration probe. Run headless for the training-resource exclusion check.
var scene: Node
var before: Dictionary

func _initialize() -> void:
	call_deferred("_run")

func _contract(node: Node, output: Dictionary) -> void:
	if node is PhysicsBody3D or node is CollisionShape3D or node is Joint3D or node is Camera3D:
		var values: Dictionary = {}
		for property in node.get_property_list():
			if int(property.usage) & PROPERTY_USAGE_STORAGE:
				values[property.name] = var_to_str(node.get(property.name))
		output[str(scene.get_path_to(node))] = values
	for child in node.get_children():
		_contract(child, output)

func _run() -> void:
	var headless := DisplayServer.get_name() == "headless"
	var default_entry := OS.get_environment("STYLE_PROBE_DEFAULT") == "1"
	# Headless uses the default entry, visible compares original and new materials.
	if not headless and not default_entry:
		OS.set_environment("SIM2SIM_VISUAL_STYLE", "legacy")
	scene = load("res://main.tscn").instantiate()
	root.add_child(scene)
	root.size = Vector2i(1920, 1080)
	Engine.max_fps = 30
	_contract(scene, before)
	var shader_paths := ["style.gd", "enamel.gdshader", "ink.gdshader", "lens.gdshader"]
	var result := {"headless":headless, "renderer":RenderingServer.get_current_rendering_method()}
	if headless:
		assert(not scene.has_meta("microduck_visual_style"))
		for path in shader_paths:
			assert(not ResourceLoader.has_cached("res://visuals/microduck/" + path), path)
		result["new_visual_resources_loaded"] = false
	else:
		if default_entry:
			assert(scene.has_meta("microduck_visual_style"), "Default visible entry did not apply style")
			result["default_entry_applied"] = true
		scene.set_process(false)
		await process_frame
		await process_frame
		RenderingServer.force_draw()
		root.get_texture().get_image().save_png(OS.get_environment("STYLE_PROBE_DIR") + "/before.png")
		# Compare immediately around apply(), excluding normal camera settling frames.
		before.clear()
		_contract(scene, before)
		load("res://visuals/microduck/style.gd").new().apply(scene)
		assert(scene.has_meta("microduck_visual_style"))
		var after: Dictionary = {}
		_contract(scene, after)
		assert(before == after, "Style changed a physics or camera property")
		result["physics_camera_properties_unchanged"] = true
		await process_frame
		await process_frame
		RenderingServer.force_draw()
		root.get_texture().get_image().save_png(OS.get_environment("STYLE_PROBE_DIR") + "/after.png")
	result["contract_nodes"] = before.size()
	result["ok"] = true
	var file := FileAccess.open(OS.get_environment("STYLE_PROBE_DIR") + "/probe.json", FileAccess.WRITE)
	file.store_string(JSON.stringify(result, "  "))
	file.close()
	print("STYLE_PROBE ", JSON.stringify(result))
	scene.queue_free()
	await process_frame
	quit(0)
