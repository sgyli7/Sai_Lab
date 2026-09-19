extends SceneTree

func _initialize() -> void:
	run.call_deferred()

func run() -> void:
	Engine.physics_ticks_per_second=2000
	var host:=Node3D.new();root.add_child(host)
	var adapter:Node3D=load("res://hub/leviathan.gd").new();host.add_child(adapter)
	var specification:Dictionary=JSON.parse_string(FileAccess.get_file_as_string("res://station_interface.json"))
	if not adapter.load_vehicle(host,specification,"res://leviathan",true):
		push_error(adapter.failure);quit(2);return
	var vehicle:Node3D=adapter.vehicle
	var index:Dictionary={};vehicle.binding._index(vehicle.asset,index)
	var checked:=0;var failures:Array=[];var maximum:=0.
	var c:=Basis(Vector3(1,0,0),Vector3(0,0,-1),Vector3(0,1,0))
	for id in vehicle.runtime.bodies:
		if not index.has(id):continue
		checked+=1
		var expected:Transform3D=vehicle.runtime.get_design_transform(id)*Transform3D(c,Vector3.ZERO)
		var actual:Transform3D=index[id].global_transform
		var error:float=actual.origin.distance_to(expected.origin)
		maximum=maxf(maximum,error)
		if error>.0001 or not actual.basis.is_equal_approx(expected.basis):failures.append(id)
	var result:Dictionary={"passed":checked>90 and failures.is_empty(),"checked":checked,"maximum_position_error_m":maximum,"unbound_bodies":failures}
	FileAccess.open("res://visual-alignment-result.json",FileAccess.WRITE).store_string(JSON.stringify(result,"  "))
	print("VISUAL_ALIGNMENT ",JSON.stringify(result));quit(0 if result.passed else 1)
