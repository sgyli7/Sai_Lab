extends SceneTree
## Geometric openings and real Jolt motion on the new neighbourhood boundaries.
class Host extends Node3D:
	var _headless:=true
var errors: Array[String]=[]
var host: Node3D
var w: Node3D

func _initialize() -> void:call_deferred("run")
func check(ok: bool,why: String) -> void:
	if not ok:errors.append(why)
func ray(a: Vector3,b: Vector3) -> Dictionary:
	return host.get_world_3d().direct_space_state.intersect_ray(PhysicsRayQueryParameters3D.create(a,b,1))
func proxy(p: Vector3,v: Vector3) -> RigidBody3D:
	var b:=RigidBody3D.new();b.position=p;b.mass=.2;b.gravity_scale=0
	b.collision_layer=2;b.collision_mask=1;b.continuous_cd=true
	b.contact_monitor=true;b.max_contacts_reported=8;b.linear_damp=0;b.angular_damp=0
	var s:=BoxShape3D.new();s.size=Vector3(.08,.18,.08)
	var c:=CollisionShape3D.new();c.shape=s;b.add_child(c)
	host.add_child(b);b.linear_velocity=v;return b

func run() -> void:
	host=Host.new();root.add_child(host)
	var scene=load("res://atelier/atelier.tscn").instantiate()
	var world:=Node3D.new();world.name="World";host.add_child(world)
	var floor_node=scene.get_node("World/Floor")
	floor_node.get_parent().remove_child(floor_node);floor_node.owner=null;world.add_child(floor_node);scene.free()
	w=load("res://atelier/workshop.gd").new();host.add_child(w);w.build(host)
	await physics_frame;await physics_frame
	check(w.collision_body.get_child_count()>260,"Missing new structural collisions")
	var solids=[
		["annex",Vector3(-.7,.5,-1.80),Vector3(-.7,.5,-2.4)],
		["shed_side",Vector3(2.99,.2,-2.25),Vector3(2.3,.2,-2.25)],
		["shed_back",Vector3(1.8,.3,-2.5),Vector3(1.8,.3,-3.0)],
		["tank",Vector3(-2.38,.55,-.90),Vector3(-2.38,.55,-1.5)],
		["fence",Vector3(-2.8,.2,-1.0),Vector3(-3.2,.2,-1.0)],
		["wheel",Vector3(2.93,.54,-.4),Vector3(2.93,.54,-.9)],
		["spool",Vector3(3.0,.16,.7),Vector3(3.0,.16,.4)],
		["barrel",Vector3(-2.61,.15,-.3),Vector3(-2.61,.15,-.7)],
		["motor",Vector3(3.2,.18,.4),Vector3(3.2,.18,.1)]
	]
	var hits:Dictionary={}
	for item in solids:
		var hit:=ray(item[1],item[2]);check(not hit.is_empty(),"Missing boundary: "+item[0])
		if not hit.is_empty():hits[item[0]]=[hit.position.x,hit.position.y,hit.position.z]
	for item in [
		["doorway",Vector3(2.18,.15,-1.42),Vector3(2.18,.15,-2.36)],
		["left_exit",Vector3(-1.2,.15,.05),Vector3(-3.4,.15,.05)],
		["right_apron",Vector3(1.9,.15,.1),Vector3(2.8,.15,.1)],
		["under_shelf",Vector3(3.10,.31,-.4),Vector3(3.10,.31,-.98)]]:
		check(ray(item[1],item[2]).is_empty(),"Blocked opening: "+item[0])
	var wall:=proxy(Vector3(3.02,.2,-2.22),Vector3(-.8,0,0))
	var doorway:=proxy(Vector3(2.18,.2,-1.43),Vector3(0,0,-.6))
	var wall_contact:=false;var door_contact:=false
	for k in range(200):
		await physics_frame
		wall_contact=wall_contact or not wall.get_colliding_bodies().is_empty()
		door_contact=door_contact or not doorway.get_colliding_bodies().is_empty()
	check(wall_contact and wall.position.x>=2.684,"Duck-sized rigid probe passed shed wall")
	check(doorway.position.z< -2.0 and not door_contact,"Duck-sized rigid probe cannot pass doorway")
	var report={"engine":ProjectSettings.get_setting("physics/3d/physics_engine"),"shapes":w.collision_body.get_child_count(),"boundary_hits":hits,
		"wall_proxy_x":wall.position.x,"wall_contact":wall_contact,"door_proxy_z":doorway.position.z,"door_contact":door_contact,"errors":errors}
	FileAccess.open(OS.get_environment("MD_YARD_REPORT"),FileAccess.WRITE).store_string(JSON.stringify(report,"  "))
	print(JSON.stringify(report));quit(0 if errors.is_empty() else 1)
