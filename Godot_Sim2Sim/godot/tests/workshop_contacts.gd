extends SceneTree
## Run against the real procedural solids and Jolt, independently of neural control.
class Host extends Node3D:
	var _headless:=true

var failures: Array[String]=[]
var reports: Array[Dictionary]=[]
var host: Node3D
var workshop: Node3D

func _initialize() -> void:
	call_deferred("run")

func check(ok: bool,message: String) -> void:
	if not ok:failures.append(message)

func ray(a: Vector3,b: Vector3) -> Dictionary:
	var query:=PhysicsRayQueryParameters3D.create(a,b,1)
	return host.get_world_3d().direct_space_state.intersect_ray(query)

func run() -> void:
	host=Host.new();root.add_child(host)
	var scene=load("res://atelier/atelier.tscn").instantiate()
	var world=Node3D.new();world.name="World";host.add_child(world)
	var floor_node=scene.get_node("World/Floor")
	floor_node.get_parent().remove_child(floor_node);floor_node.owner=null;world.add_child(floor_node)
	scene.free()
	workshop=load("res://atelier/workshop.gd").new();host.add_child(workshop);workshop.build(host)
	await physics_frame
	await physics_frame
	check(ProjectSettings.get_setting("physics/3d/physics_engine")=="Jolt Physics","Wrong physics engine")
	check(workshop.collision_body.get_child_count()>80,"Missing workshop structural colliders")
	# Known real structures and open air below the workbench.
	for p in [Vector3(0,.4,-1.63),Vector3(-1.55,.3,-.70),Vector3(2.70,.15,.90)]:
		var hit=ray(p+Vector3(0,1,0),p-Vector3(0,1,0))
		check(not hit.is_empty() and hit.collider==workshop.collision_body,"Missing prop at "+str(p))
	var open=ray(Vector3(-.5,.2,-.7),Vector3(-.5,.2,-1.40))
	check(open.is_empty(),"Workbench leg space blocked by an oversized collision box")
	var floor_hit=ray(Vector3(0,.5,0),Vector3(0,-.1,0))
	check(not floor_hit.is_empty() and absf(floor_hit.position.y)<.00001,"Spawn floor is no longer flush")
	var bodies: Array[RigidBody3D]=[]
	for item in workshop.contact_course.specimens:
		var p:Vector3=item.body.position
		var hit=ray(p+Vector3(0,.5,0),p-Vector3(0,.1,0))
		var expected:float=(.015 if item.name=="Ramp_30mm" else item.mesh.get_aabb().end.y)+p.y
		check(not hit.is_empty() and hit.collider==item.body,"Missed specimen "+item.name)
		if not hit.is_empty():
			check(absf(hit.position.y-expected)<.0002,"Surface/collision mismatch "+item.name)
			reports.append({"specimen":item.name,"height":hit.position.y,"expected":expected,"normal":[hit.normal.x,hit.normal.y,hit.normal.z]})
		# Free dynamic shoe-sized probe, with the duck's environment layer/mask.
		# Drop on the level specimens; the ramp is verified by slope-normal rays.
		if item.name=="Ramp_30mm":
			check(not hit.is_empty() and hit.normal.y>.99 and hit.normal.z<-.06,"Ramp slope/winding incorrect")
			continue
		var body:=RigidBody3D.new();body.mass=.1;body.collision_layer=2;body.collision_mask=1
		body.max_contacts_reported=8;body.contact_monitor=true
		body.position=Vector3(p.x,expected+.05,p.z)
		var shape:=BoxShape3D.new();shape.size=Vector3(.024,.006,.024)
		var collision:=CollisionShape3D.new();collision.shape=shape;body.add_child(collision)
		host.add_child(body);bodies.append(body)
		body.set_meta("expected",expected+.003)
		body.set_meta("specimen",item.name)
	for i in range(200):await physics_frame
	for body in bodies:
		check(absf(body.position.y-float(body.get_meta("expected")))<.001,"Probe penetrated/floated: "+str(body.get_meta("specimen")))
		check(body.get_colliding_bodies().size()>0,"No real contact for "+str(body.get_meta("specimen")))
	var report={"engine":ProjectSettings.get_setting("physics/3d/physics_engine"),"workshop_shapes":workshop.collision_body.get_child_count(),"specimens":reports,"dynamic_contacts":bodies.size(),"failures":failures}
	print(JSON.stringify(report))
	if OS.has_environment("MD_CONTACT_REPORT"):FileAccess.open(OS.get_environment("MD_CONTACT_REPORT"),FileAccess.WRITE).store_string(JSON.stringify(report,"  "))
	quit(0 if failures.is_empty() else 1)
