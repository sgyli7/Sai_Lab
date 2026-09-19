extends SceneTree
## Compares the in-process Sai controller with fixtures emitted by the Python oracle.

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	var fixture_path := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--fixtures="):fixture_path=arg.trim_prefix("--fixtures=")
	if fixture_path.is_empty():
		push_error("--fixtures is required")
		quit(2)
		return
	var fixture: Dictionary=JSON.parse_string(FileAccess.get_file_as_string(fixture_path))
	var maximum: Dictionary={"observation":0.0,"action":0.0,"target_leg":0.0,"wheel_speed":0.0,"effective_crouch":0.0}
	var result_keys: Dictionary={"observation":"policy_observation","action":"policy_action","target_leg":"target_leg","wheel_speed":"wheel_speed","effective_crouch":"effective_crouch"}
	var branches: Array=[]
	var passed:=true
	for group in fixture.groups:
		var controller=preload("res://sai/native_controller.gd").new(str(group.profile))
		if not controller.last_error.is_empty():
			push_error(controller.last_error)
			quit(2)
			return
		for item in group.cases:
			var actual: Dictionary=controller.command(item.state)
			var expected: Dictionary=item.expected
			for key in maximum:
				var result_key: String=result_keys[key]
				var a: Array=_numbers(actual.get(result_key))
				var b: Array=_numbers(expected.get(result_key))
				if a.size()!=b.size():passed=false;continue
				for i in range(a.size()):maximum[key]=maxf(maximum[key],absf(float(a[i])-float(b[i])))
			for key in ["stage","stair_profile","contract_id"]:
				if actual[key]!=expected[key]:passed=false
			branches.append({"case":item.id,"stage":actual.stage,"profile":actual.stair_profile})
	passed=passed and maximum["observation"]<2e-6 and maximum["action"]<1e-5 and maximum["target_leg"]<2e-6 and maximum["wheel_speed"]<2e-6 and maximum["effective_crouch"]<1e-12
	print("SAI_NATIVE_CONTRACT_RESULT "+JSON.stringify({"passed":passed,"max_abs_error":maximum,"branches":branches}))
	quit(0 if passed else 1)

func _numbers(value: Variant) -> Array:
	if typeof(value)==TYPE_FLOAT or typeof(value)==TYPE_INT:
		return [value]
	return value if value is Array else []
