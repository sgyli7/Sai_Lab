extends SceneTree
## Native desktop entry smoke: activate the default scene with keyboard focus.
func _initialize() -> void:
	call_deferred("run")

func run() -> void:
	var path:="res://hub/options.json"
	var previous:=FileAccess.get_file_as_string(path)
	var options:Dictionary=JSON.parse_string(previous)
	options.choose_scene=true;options.robot="microduck";options.record=false
	options.plan={"seconds":2.};options.output=OS.get_environment("PICKER_OUTPUT")
	FileAccess.open(path,FileAccess.WRITE).store_string(JSON.stringify(options))
	var hub:Node3D=load("res://hub/main.tscn").instantiate();root.add_child(hub)
	await process_frame
	FileAccess.open(path,FileAccess.WRITE).store_string(previous)
	for i in range(6):await process_frame
	RenderingServer.force_draw(false)
	root.get_texture().get_image().save_png(options.output+"/picker.png")
	var focused:Control=root.gui_get_focus_owner()
	if not focused is Button or not focused.text.begins_with("风口科学站"):
		push_error("Science station must be the keyboard default");quit(1);return
	var event:=InputEventKey.new();event.keycode=KEY_ENTER;event.physical_keycode=KEY_ENTER;event.pressed=true
	Input.parse_input_event(event)
	await process_frame
	event=InputEventKey.new();event.keycode=KEY_ENTER;event.physical_keycode=KEY_ENTER;event.pressed=false
	Input.parse_input_event(event)
	await create_timer(1.).timeout
	if hub.atelier==null or not hub.is_science_station():
		push_error("Scene picker did not start the science station");quit(1);return
	print("SCENE_PICKER accepted Enter; science station started")
