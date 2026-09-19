extends SceneTree
## Render evidence for the lower beak from fixed front, side, and oblique views.
var hub

func _init() -> void:
	call_deferred("_run")

func _run() -> void:
	hub = load("res://hub/main.tscn").instantiate()
	root.add_child(hub)
	for i in range(20): await process_frame
	var duck = hub.actor
	var head: RigidBody3D = duck._bodies.get("jaw_soft")
	print("BEAK_TRANSFORMS ",JSON.stringify({"head_basis":head.global_basis,"head_position":head.global_position,
		"pivot_position":duck.beak.pivot.position if duck.beak!=null else Vector3.ZERO,
		"meshes":duck.beak.pivot.get_children().map(func(node): return {"name":node.name,"position":node.position,"basis":node.basis}) if duck.beak!=null else []}))
	var output := "/tmp/microduck-beak-visual"
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--capture-dir="): output = arg.trim_prefix("--capture-dir=")
	DirAccess.make_dir_recursive_absolute(output)
	var camera: Camera3D = hub.get_node("World/Camera3D")
	var center := head.global_position
	var views := {"front":Vector3(0.5,0.06,0),"side":Vector3(0.0,0.06,0.5),
		"oblique":Vector3(0.38,0.10,0.38)}
	for view in views:
		camera.global_position = center + views[view]
		camera.look_at(center)
		camera.fov = 24
		for i in range(4): await process_frame
		var path: String = output.path_join("%s-neutral.png" % view)
		root.get_texture().get_image().save_png(path)
		print("BEAK_FRAME ",path)
	if duck.beak != null:
		duck.beak.perform("open")
		for i in range(130): await physics_frame
		for view in views:
			camera.global_position = center + views[view]
			camera.look_at(center)
			camera.fov = 24
			for i in range(4): await process_frame
			var path: String = output.path_join("%s-open.png" % view)
			root.get_texture().get_image().save_png(path)
			print("BEAK_FRAME ",path)
	quit()
