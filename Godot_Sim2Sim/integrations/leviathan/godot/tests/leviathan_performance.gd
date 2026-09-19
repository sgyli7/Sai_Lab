extends SceneTree
var vehicle:Node3D
var start:=0
var frames:=0
func _initialize() -> void:
	run.call_deferred()
func run() -> void:
	Engine.physics_ticks_per_second=2000;Engine.max_physics_steps_per_frame=100;Engine.max_fps=0
	var floor:=StaticBody3D.new();var collision:=CollisionShape3D.new();var box:=BoxShape3D.new()
	box.size=Vector3(400,1,400);collision.shape=box;collision.position.y=-.5;floor.add_child(collision);root.add_child(floor)
	vehicle=load("res://leviathan/imported_vehicle.gd").new();root.add_child(vehicle)
	if not vehicle.load_bundle("res://leviathan",Transform3D.IDENTITY,false):push_error(vehicle.failure);quit(2);return
	vehicle.runtime.use_native_jel=true;vehicle.runtime.profile_timing=true
	vehicle.runtime.set_vehicle_command(.2,0.)
	start=Time.get_ticks_usec()
func _process(_delta:float) -> bool:
	if start==0:return false
	frames+=1
	if vehicle.runtime.time_s>=2.:
		var wall:=float(Time.get_ticks_usec()-start)/1e6
		var result:Dictionary={"simulation_s":vehicle.runtime.time_s,"wall_s":wall,"ratio":vehicle.runtime.time_s/wall,"timing":vehicle.runtime.timing_usec,"frames":frames,"bodies":vehicle.runtime.bodies.size()}
		FileAccess.open("res://performance-result.json",FileAccess.WRITE).store_string(JSON.stringify(result,"  "))
		print("CARRIER_PERFORMANCE ",JSON.stringify(result));quit()
	return false
