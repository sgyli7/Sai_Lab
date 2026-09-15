extends SceneTree
## Probe the current native stair reference with zero neural residual.
func _initialize() -> void:
	var controller=preload("res://sai/native_controller.gd")
	var action:=PackedFloat32Array();action.resize(16);action.fill(0.)
	var near:Array=[];var far:Array=[]
	for i in range(8):
		for _j in range(3):
			var x:float=-.36+i*.18
			near.append(.02 if x>.1 else 0.)
			far.append(.02 if x>.5 else 0.)
	var same:=true;var maximum:=0.
	var standing:Array=controller.targets(action,[.12,0.,0.],0.)
	for i in range(160):
		var a:Array=controller.targets_stairs(action,[.12,0.,0.],0.,float(i)/160.,near)
		var b:Array=controller.targets_stairs(action,[.12,0.,0.],0.,float(i)/160.,far)
		same=same and a==b
		for j in [1,2,5,6,9,10,13,14]:maximum=maxf(maximum,absf(float(b[j])-float(standing[j])))
	var report:={"native_zero_actor_near_far_identical":same,"max_hip_knee_reference_rad":maximum,"phases_checked":160}
	var file:=FileAccess.open(OS.get_environment("SAI_TEMPLATE_AUDIT_OUT"),FileAccess.WRITE)
	file.store_string(JSON.stringify(report,"  "));file.close()
	print(JSON.stringify(report));quit(0 if same and maximum>.18 else 1)
