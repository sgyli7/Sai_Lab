extends SceneTree
func _initialize() -> void:
	run.call_deferred()
func run() -> void:
	var scene:Node3D=load("res://hub/main.tscn").instantiate();scene.set_script(null)
	var host:=Host.new();host.options.leviathan=JSON.parse_string(FileAccess.get_file_as_string("res://station_interface.json"))
	for child in scene.get_children():child.owner=null;scene.remove_child(child);host.add_child(child)
	scene.free();root.add_child(host)
	var station:Node3D=load("res://polar_range/scene.gd").new();host.add_child(station);station.build(host)
	var hub:Node3D=load("res://hub/hub.gd").new();hub._set_scenery_masks(host);hub.free()
	await physics_frame
	var space:=host.get_world_3d().direct_space_state
	var missing:=0;var error:=0.;var grade:=0.;var low:=INF;var high:=-INF;var apron_error:=0.
	for x in range(90,801,5):
		for z in [-24.,-12.,0.,12.,24.]:
			var expected:float=station.ground_height(x,z)
			var hit:=space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(x,expected+2.,z),Vector3(x,expected-2.,z),8))
			if hit.is_empty():missing+=1;continue
			error=maxf(error,absf(hit.position.y-expected))
			if x<190:apron_error=maxf(apron_error,absf(expected-3.))
			if x>=200:
				low=minf(low,expected);high=maxf(high,expected)
				grade=maxf(grade,absf(station.ground_height(x+1.,z)-expected))
	var result:Dictionary={"missing":missing,"surface_error_m":error,"apron_error_m":apron_error,"test_lane_max_grade":grade,"relief_m":high-low,"triangles":station.landscape_builder.triangle_count}
	result.passed=missing==0 and error<.002 and apron_error<.001 and grade<.08 and high-low>.5
	FileAccess.open("res://polar-result.json",FileAccess.WRITE).store_string(JSON.stringify(result,"  "))
	print("POLAR_TERRAIN ",JSON.stringify(result));quit(0 if result.passed else 1)
class Host extends Node3D:
	var _headless:=true
	var _bodies:Dictionary={}
	var options:Dictionary={"scene":"polar_range","polar":true}
