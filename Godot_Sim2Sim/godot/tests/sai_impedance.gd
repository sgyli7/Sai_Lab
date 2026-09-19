extends SceneTree
func _initialize() -> void:
	var path:=OS.get_environment("SAI_IMPEDANCE_FIXTURE")
	var fixtures:Array=JSON.parse_string(FileAccess.get_file_as_string(path))
	var got:Array=[]
	for f in fixtures:
		got.append(preload("res://sai/impedance.gd").torque(f.command,float(f.q),float(f.v),int(f.joint)))
	print("IMPEDANCE_PARITY ",JSON.stringify(got))
	quit()
