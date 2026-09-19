extends SceneTree

const Brain = preload("res://standalone/play_brain.gd")
const Contract = preload("res://standalone/policy_contract.gd")
var failures: Array = []
var maximum := 0.0

func compare(actual: Variant, expected: Variant, label: String, tolerance: float = 1e-6) -> void:
	if typeof(expected) in [TYPE_ARRAY, TYPE_PACKED_FLOAT32_ARRAY]:
		if actual.size() != expected.size():
			failures.append(label+" size")
			return
		for i in range(expected.size()): compare(actual[i],expected[i],label+"[%d]"%i,tolerance)
	elif typeof(expected) in [TYPE_INT, TYPE_FLOAT]:
		var error := absf(float(actual)-float(expected))
		maximum = maxf(maximum,error)
		if error > tolerance and failures.size() < 20: failures.append([label,actual,expected,error])
	elif actual != expected and failures.size() < 20:
		failures.append([label,actual,expected])

func _initialize() -> void:
	var path := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--fixtures="): path=arg.trim_prefix("--fixtures=")
	var fixture: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(path))
	var deployment: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://runtime_assets/deployment.json"))
	var ticks := 0
	for sequence in fixture.brains:
		var brain = Brain.new()
		brain.configure(sequence.flags,sequence.limits)
		for item in sequence.cases:
			var actual: Dictionary = brain.tick(item.held,item.taps,item.dt,item.order)
			for key in item.expected:
				# Text formatting of signed zero differs between Godot and Python.
				if key != "status": compare(actual[key],item.expected[key],sequence.mode+"/%d/"%ticks+key)
			for key in ["vel","behavior_t","pick_phase","rise_t"]:
				compare(brain.get(key),item[key],sequence.mode+"/%d/"%ticks+key)
			ticks += 1
	for item in fixture.observations:
		var robot: Dictionary = deployment.robots[item.mode]
		var body := Contract.body_state(item.raw,robot)
		compare(body.base_pos,item.body.base_pos,"position",1e-12)
		compare(body.base_quat,item.body.base_quat,"quaternion",1e-12)
		compare(Contract.observation(item.raw,body,PackedFloat32Array(item.last),PackedFloat32Array(item.command),PackedFloat32Array(robot.home)),item.obs,"observation",1e-7)
		compare(Contract.control(PackedFloat32Array(item.last),PackedFloat32Array(robot.home),robot.action_scale),item.control,"control",1e-7)
		compare(Contract.ball_position(body,"kick_left"),item.ball_left,"ball left",1e-12)
		compare(Contract.ball_position(body,"kick_right"),item.ball_right,"ball right",1e-12)
	print("NATIVE_CONTRACT_RESULT "+JSON.stringify({"passed":failures.is_empty(),"brain_ticks":ticks,
		"observations":fixture.observations.size(),"max_error":maximum,"failures":failures}))
	quit(0 if failures.is_empty() else 1)
