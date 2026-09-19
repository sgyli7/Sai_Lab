extends SceneTree
## Deterministic controller contract check; no robot or physics simulation.
const Brain = preload("res://standalone/play_brain.gd")

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	var fixture: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(args[0]))
	var errors: Array = []
	var maximum := 0.0
	var server = load("res://physics_server.gd").new()
	for location in [KEY_LOCATION_RIGHT,KEY_LOCATION_LEFT]:
		var event := InputEventKey.new()
		event.physical_keycode=KEY_SHIFT;event.location=location;event.pressed=true
		server._input(event);server._sample_held()
		if server._held_now.has("sprint") != (location==KEY_LOCATION_LEFT):errors.append("Shift location")
		event.pressed=false;server._input(event)
	server._left_shift_down=true
	server._notification(Node.NOTIFICATION_APPLICATION_FOCUS_OUT)
	server._sample_held()
	if server._held_now.has("sprint"):errors.append("Shift stuck after focus loss")
	server.free()
	for case in fixture.cases:
		var brain := Brain.new()
		brain.configure(case.flags,case.limits)
		for step in case.steps:
			var actual: Dictionary = brain.tick(step.held,step.taps,0.02,step.order)
			if actual.policy != step.policy or actual.sprint != step.sprint:
				errors.append({"case":case.name,"step":step,"actual":actual})
			for index in range(13):
				maximum = maxf(maximum,absf(actual.command[index]-float(step.command[index])))
	print(JSON.stringify({"check":"sprint_brain_contract","cases":fixture.cases.size(),
		"max_abs_command":maximum,"mismatches":errors.size(),"passed":errors.is_empty() and maximum<1e-6}))
	quit(0 if errors.is_empty() and maximum<1e-6 else 1)
