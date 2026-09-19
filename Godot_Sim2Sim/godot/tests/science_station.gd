extends SceneTree
const Layout=preload("res://science_station/layout.gd")
func _initialize() -> void:
	call_deferred("run")
func run() -> void:
	var scene:Node3D=load("res://hub/main.tscn").instantiate()
	scene.set_script(null)
	# A small host supplies the same fields the production builder reads.
	var host:=Host.new()
	for child in scene.get_children():child.owner=null;scene.remove_child(child);host.add_child(child)
	scene.free();root.add_child(host)
	var station:Node3D=load("res://science_station/station.gd").new();host.add_child(station);station.build(host)
	var hub:Node3D=load("res://hub/hub.gd").new()
	hub._set_scenery_masks(host);hub.free()
	await physics_frame
	var checks:Dictionary={"terrain_patches":0,"tower_passages":true,"berth_clear":true,"ground":true,"six_props":station.loose_props.items.size()==6}
	for child in station.get_children():
		if str(child.name).begins_with("Terrain_"):checks.terrain_patches+=1
	var space:=host.get_world_3d().direct_space_state
	checks["sai_terrain_layer"]=true
	for p in [Vector3(-18.,.3,-3.),Vector3(10.,.3,2.),Vector3(-7.5,8.,-.5)]:
		var terrain:=space.intersect_ray(PhysicsRayQueryParameters3D.create(p,Vector3(p.x,-.2,p.z),8))
		if terrain.is_empty() or absf(terrain.position.y-Layout.height_at(p.x,p.z))>.00005:checks.sai_terrain_layer=false
	# Buildings still collide physically, while their roofs are excluded from driving-height rays.
	var roof:=space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(-7.5,8.,-.5),Vector3(-7.5,-.2,-.5),3))
	checks["sai_building_stays_solid"]=not roof.is_empty() and roof.position.y > 1.
	var landscape=load("res://science_station/landscape.gd").new()
	checks["closed_horizon"]=true
	for band in range(9):
		var r:float=[1.,1.20,1.52,1.95,2.55,3.5,5.5,9.,18.][band]
		if landscape.ridge(0.,r,band).distance_to(landscape.ridge(TAU,r,band))>.0001:checks.closed_horizon=false
	for x in [-22.,-18.,-15.,0.,5.,13.,22.]:
		for z in [-24.,-17.,-3.,1.,7.,12.]:
			var expected:float=Layout.height_at(x,z)
			var hit:=space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(x,expected+.012,z),Vector3(x,expected-.012,z),8))
			if hit.is_empty() or absf(hit.position.y-Layout.height_at(x,z))>.002:checks.ground=false
	for x in [-21.83,-19.14,-17.27,-14.77]:
		var hit:=space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(x,.3,-3.17),Vector3(x,-.2,-3.17),3))
		# Jolt compresses triangle vertices; allow 50 micrometres at this scale.
		if hit.is_empty() or absf(hit.position.y-Layout.height_at(x,-3.17))>.00005:
			checks.ground=false;print("GROUND_ERROR ",x," ",hit," expected=",Layout.height_at(x,-3.17))
	checks["near_relief"]=Layout.height_at(-19.,7.)>.32 and Layout.height_at(-19.,7.)<.9
	checks["prepared_berth_level"]=true
	for x in range(3,24,2):
		for z in range(-8,5,2):
			if absf(Layout.height_at(x,z))>.00001:checks.prepared_berth_level=false
	checks["patch_seams"]=true
	for x in [-16.001,-16.,-15.999,-8.001,-8.,-7.999]:
		for z in [-10.17,6.37,10.11]:
			var expected:float=Layout.height_at(x,z)
			var hit:=space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(x,expected+.012,z),Vector3(x,expected-.012,z),8))
			if hit.is_empty() or absf(hit.position.y-expected)>.00005:checks.patch_seams=false
	checks["observatory_prepared_ground"]=true
	for x in [-37.5,-35.,-32.5]:
		for z in [-22.,-20.4,-18.8]:
			if absf(station.ground_height(x,z)-1.1)>.0001:checks.observatory_prepared_ground=false
	var foundation:=space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(-32.,1.6,-18.6),Vector3(-32.,1.,-18.6),3))
	checks["observatory_foundation_solid"]=not foundation.is_empty() and absf(foundation.position.y-1.34)<.03
	var rock_hit:=space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(-36.,1.3,4.),Vector3(-31.,1.3,4.),3))
	checks["outer_outcrop_solid"]=not rock_hit.is_empty()
	checks["outer_ground_join"]=true
	for x in [-24.03,24.03]:
		for z in [-22.,0.,12.]:
			var expected:float=station.ground_height(x,z)
			var hit:=space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(x,expected+.03,z),Vector3(x,expected-.03,z),8))
			if hit.is_empty() or absf(hit.position.y-expected)>.00005:checks.outer_ground_join=false
	var corrected:Vector3=station._unobstructed_position(Vector3(-22,.12,-3),Vector3(-16,.12,-3))
	checks["camera_crest"]=corrected.x < -18.
	checks["camera_near_relief"]=station._unobstructed_position(Vector3(-13,.10,7),Vector3(-21,.10,7)).x > -19.
	for p in [Vector3(7,.20,-17),Vector3(16,.20,-21)]:
		var query:=PhysicsRayQueryParameters3D.create(p+Vector3(-3,0,0),p+Vector3(3,0,0),3)
		if not space.intersect_ray(query).is_empty():checks.tower_passages=false
	for x in [4.,8.,13.,18.,22.]:
		var hit:=space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(x,.22,-7),Vector3(x,.22,3),3))
		if not hit.is_empty():checks.berth_clear=false
	checks["lab_door"]=space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(-7.5,.30,2.6),Vector3(-7.5,.30,-.7),3)).is_empty()
	checks["service_bay"]=space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(-3,.3,7.1),Vector3(-3,.3,10.4),3)).is_empty()
	# Outbound left-wheel envelope measured from the native roller reversal.
	checks["lab_exit_margin"]=space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(-8.02,.10,.70),Vector3(-8.02,.10,1.30),3)).is_empty()
	checks["lab_wall_solid"]=not space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(-4.9,.8,-.5),Vector3(-6.5,.8,-.5),3)).is_empty()
	checks["command_hull_solid"]=not space.intersect_ray(PhysicsRayQueryParameters3D.create(Vector3(-8.,1.5,-17.),Vector3(-8.,1.5,-21.),3)).is_empty()
	checks["camera_lab"]=station._unobstructed_position(Vector3(-4.5,2.4,-.5),Vector3(-8.5,2.4,-.5)).x > -5.7
	var passed:bool=checks.terrain_patches==30 and checks.ground and checks.tower_passages and checks.berth_clear and checks.six_props and checks.camera_crest and checks.camera_near_relief and checks.closed_horizon and checks.lab_door and checks.service_bay and checks.lab_wall_solid and checks.command_hull_solid and checks.camera_lab and checks.lab_exit_margin
	passed=passed and checks.sai_terrain_layer and checks.sai_building_stays_solid and checks.near_relief and checks.prepared_berth_level and checks.patch_seams and checks.observatory_prepared_ground and checks.observatory_foundation_solid and checks.outer_outcrop_solid and checks.outer_ground_join
	print("SCIENCE_GEOMETRY ",JSON.stringify({"passed":passed,"checks":checks}))
	host.queue_free();await process_frame
	quit(0 if passed else 1)
class Host extends Node3D:
	var _headless:=true
	var _bodies:Dictionary={}
	var options:Dictionary={"scene":"science_station"}
