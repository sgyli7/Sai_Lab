extends Node
## Only explicit --case runs: native focus changes can release synthetic keys.
## Restore the release test's intended held events before its next control tick.
## Interactive driving never installs this node; normal key release stays normal.
var scene: Node3D

func _ready() -> void:
	scene = get_parent()
	process_physics_priority = -10

func _physics_process(_delta: float) -> void:
	if scene.finished or not scene.robot.is_control_tick(): return
	for key in scene.injected:
		if scene.injected[key] and not Input.is_physical_key_pressed(key):
			var event := InputEventKey.new()
			event.physical_keycode = key
			event.keycode = key
			event.pressed = true
			Input.parse_input_event(event)
			# Keep logical press/release history intact; report restorations separately.
			print("SAI_REPLAY_RESTORE key=", key, " time=", scene.robot.sim_time_seconds())
