extends SceneTree
## Check the unified world picker and its Sainiverse handoff without a desktop.

func _initialize() -> void:
	call_deferred("_probe")

func _probe() -> void:
	var hub:Node=load("res://hub/main.tscn").instantiate()
	root.add_child(hub)
	await process_frame
	var scene_button:Button=_find_button(hub,"03 · 极地雪原")
	if scene_button==null:
		push_error("Polar world is missing from the unified picker");quit(2);return
	scene_button.pressed.emit()
	await process_frame
	var vehicle_picker:AcceptDialog=_find_vehicle_picker(hub)
	if vehicle_picker==null:
		push_error("Polar vehicle choice did not open");quit(2);return
	var vehicle_button:Button=_find_button(vehicle_picker,"驾驶 Sainiverse v0.1")
	if vehicle_button==null:
		push_error("Sainiverse choice is missing");quit(2);return
	print("POLAR_PICKER_HANDOFF_OK world=03 vehicle=Sainiverse_v0.1")
	vehicle_picker.custom_action.emit("003")

func _find_button(node:Node,label:String)->Button:
	if node is Button and (node as Button).text.begins_with(label):return node
	for child in node.get_children(true):
		var found:Button=_find_button(child,label)
		if found!=null:return found
	return null

func _find_vehicle_picker(node:Node)->AcceptDialog:
	if node is AcceptDialog and (node as AcceptDialog).title=="03 · 极地雪原":return node
	for child in node.get_children(true):
		var found:AcceptDialog=_find_vehicle_picker(child)
		if found!=null:return found
	return null
