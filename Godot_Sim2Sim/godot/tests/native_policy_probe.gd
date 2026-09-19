extends SceneTree

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	var fixture_path := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--fixtures="):
			fixture_path = arg.trim_prefix("--fixtures=")
	if fixture_path.is_empty() or not ClassDB.class_exists("MicroDuckPolicy"):
		push_error("Native policy class or fixtures unavailable")
		quit(1)
		return
	var fixtures: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(fixture_path))
	var reports: Array = []
	var passed := true
	for item in fixtures["cases"]:
		var policy = ClassDB.instantiate("MicroDuckPolicy")
		var raw := FileAccess.get_file_as_bytes(item["path"])
		var hash := HashingContext.new()
		hash.start(HashingContext.HASH_SHA256)
		hash.update(raw)
		if hash.finish().hex_encode() != item["sha256"] or not policy.load_model(raw):
			push_error("Load failed: " + str(item["skill"]) + ": " + str(policy.get_last_error()))
			quit(1)
			return
		var max_error := 0.0
		var times: Array = []
		for i in range(item["observations"].size()):
			var obs := PackedFloat32Array(item["observations"][i])
			var action: PackedFloat32Array = policy.infer(obs)
			if action.size() != 14:
				push_error(str(policy.get_last_error()))
				quit(1)
				return
			times.append(policy.get_last_infer_usec())
			for j in range(14):
				max_error = maxf(max_error, absf(action[j] - float(item["actions"][i][j])))
		var bad_obs := PackedFloat32Array(item["observations"][0])
		bad_obs[0] = NAN
		var invalid_rejected: bool = policy.infer(bad_obs).is_empty() and not policy.get_last_error().is_empty()
		var shape_rejected: bool = policy.infer(PackedFloat32Array([0.0])).is_empty()
		passed = passed and max_error < 1e-5 and invalid_rejected and shape_rejected
		var report := {"skill":item["skill"], "max_abs_error":max_error,
			"samples":item["observations"].size(), "infer_usec":times,
			"invalid_rejected":invalid_rejected, "shape_rejected":shape_rejected,
			"runtime":policy.get_runtime_version()}
		for cycle in range(3):
			policy.unload()
			if not policy.load_model(raw) or policy.infer(PackedFloat32Array(item["observations"][0])).size() != 14:
				passed = false
		policy.unload()
		reports.append(report)
	print("NATIVE_POLICY_RESULT " + JSON.stringify({"passed":passed,"models":reports}))
	quit(0 if passed else 1)
