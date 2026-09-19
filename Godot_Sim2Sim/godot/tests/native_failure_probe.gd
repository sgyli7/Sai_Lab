extends SceneTree

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	var fixture_path := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--fixtures="): fixture_path=arg.trim_prefix("--fixtures=")
	if fixture_path.is_empty() or not ClassDB.class_exists("MicroDuckPolicy"):
		push_error("Missing native failure probe fixtures or extension")
		quit(1)
		return
	var fixtures: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(fixture_path))
	var policy = ClassDB.instantiate("MicroDuckPolicy")
	var obs := PackedFloat32Array()
	obs.resize(61)
	var good := FileAccess.get_file_as_bytes(fixtures.good)
	var reports: Array = []
	var passed := true
	for item in fixtures.invalid_loads:
		var started_valid: bool = policy.load_model(good) and policy.infer(obs).size()==14
		var loaded: bool = policy.load_model(FileAccess.get_file_as_bytes(item.path))
		var load_error: String = policy.get_last_error()
		var stale_rejected: bool = policy.infer(obs).is_empty()
		var ok: bool = started_valid and not loaded and not load_error.is_empty() and stale_rejected
		passed=passed and ok
		reports.append({"case":item.name,"passed":ok,"error":load_error,"stale_model_rejected":stale_rejected})
	var nonfinite_loaded: bool = policy.load_model(FileAccess.get_file_as_bytes(fixtures.nonfinite))
	var nonfinite_rejected: bool = policy.infer(obs).is_empty() and not policy.get_last_error().is_empty()
	passed=passed and nonfinite_loaded and nonfinite_rejected
	reports.append({"case":"nonfinite_action","passed":nonfinite_loaded and nonfinite_rejected,"error":policy.get_last_error()})
	var good_loaded: bool = policy.load_model(good)
	for value in [NAN,INF,-INF]:
		var invalid := obs.duplicate()
		invalid[0]=value
		var rejected: bool = policy.infer(invalid).is_empty() and not policy.get_last_error().is_empty()
		passed=passed and good_loaded and rejected
		reports.append({"case":"nonfinite_observation","passed":rejected,"error":policy.get_last_error()})
	policy.unload()
	var unloaded_rejected: bool = policy.infer(obs).is_empty() and not policy.get_last_error().is_empty()
	var recovered: bool = policy.load_model(good) and policy.infer(obs).size()==14
	passed=passed and unloaded_rejected and recovered
	policy.unload()
	reports.append({"case":"explicit_release_and_recovery","passed":unloaded_rejected and recovered})
	print("NATIVE_FAILURE_RESULT "+JSON.stringify({"passed":passed,"cases":reports}))
	quit(0 if passed else 1)
