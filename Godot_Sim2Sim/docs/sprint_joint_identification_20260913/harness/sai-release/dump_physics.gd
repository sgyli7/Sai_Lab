extends SceneTree
func _initialize() -> void:
	var values := {}
	for entry in ProjectSettings.get_property_list():
		if str(entry.name).begins_with("physics/"):
			var value = ProjectSettings.get_setting(entry.name)
			if value is Vector3: value = [value.x,value.y,value.z]
			values[entry.name] = value
	var file := FileAccess.open(OS.get_environment("HUB_PHYSICS_DUMP"),FileAccess.WRITE)
	file.store_string(JSON.stringify(values,"  "))
	quit()
