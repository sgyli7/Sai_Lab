extends SceneTree
## Tests real Jolt free-body contacts, angular response, reset and lockstep pause.
class Host extends Node3D:
	var _headless:=true
	var _bodies: Dictionary={}
var errors: Array[String]=[]
var metrics: Dictionary={}
var host:Node3D

func _initialize() -> void:
	OS.set_environment("MD_STATIC_COURSE","0")
	call_deferred("run")
func check(ok:bool,message:String) -> void:
	if not ok:errors.append(message)
func run() -> void:
	host=Host.new();root.add_child(host)
	var scene=load("res://atelier/atelier.tscn").instantiate()
	var world=Node3D.new();world.name="World";host.add_child(world)
	var floor_node=scene.get_node("World/Floor")
	floor_node.get_parent().remove_child(floor_node);floor_node.owner=null;world.add_child(floor_node);scene.free()
	var w=load("res://atelier/workshop.gd").new();host.add_child(w);w.build(host)
	var props=w.loose_props
	check(props!=null and props.items.size()==6,"Expected six loose bodies")
	check(w.contact_course==null,"Static course should not be the default")
	for item in props.items:
		var body:RigidBody3D=item.body
		check(body.mass>=.006-.000001 and body.mass<=.015+.000001,"Mass outside reference ball range")
		check(body.continuous_cd and not body.lock_rotation,"Body must rotate with CCD")
		check(body.inertia==Vector3.ZERO and body.center_of_mass_mode==RigidBody3D.CENTER_OF_MASS_MODE_AUTO,"Engine should compute inertia and center of mass")
	props.set_frozen(false)
	for i in range(400):await physics_frame
	for item in props.items:
		var body:RigidBody3D=item.body
		check(body.get_colliding_bodies().size()>0,"Body did not rest on floor: "+str(body.name))
		check(body.linear_velocity.length()<.01,"Body unstable at rest: "+str(body.name)+" velocity="+str(body.linear_velocity)+" position="+str(body.position))
		metrics[str(body.name)]={"rest_speed":body.linear_velocity.length(),"rest_displacement":body.position.distance_to(item.spawn.origin)}
		check(body.position.distance_to(item.spawn.origin)<.005,"Object drifted more than 5mm while settling: "+str(body.name))
		var pos:Vector3=body.global_position
		if str(body.name).begins_with("Bin"):
			var hit=host.get_world_3d().direct_space_state.intersect_ray(PhysicsRayQueryParameters3D.create(pos+Vector3(0,.09,0),pos-Vector3(0,.02,0),1))
			check(not hit.is_empty() and hit.collider==body and absf(hit.position.y-pos.y-.003)<.0005,"Bin opening is filled by a convex hull")
	# Isolate mass response in free flight. This is an explicit impulse unit
	# experiment, not footage or a claim about policy-generated kicks.
	for i in range(props.items.size()):
		var body:RigidBody3D=props.items[i].body
		body.global_position=Vector3(-2.0+i*.65,.6,2.4)
		body.linear_velocity=Vector3.ZERO;body.angular_velocity=Vector3.ZERO
		body.apply_impulse(Vector3(.0015,0,0),Vector3(0,.025,0))
	await physics_frame
	await physics_frame
	var velocities:Dictionary={}
	for item in props.items:
		var body:RigidBody3D=item.body
		velocities[str(body.name)]=body.linear_velocity.x
		check(body.linear_velocity.x>.05,"No impulse translation: "+str(body.name))
		check(body.angular_velocity.length()>.1,"No off-center rotation: "+str(body.name))
		metrics[str(body.name)].merge({"mass":body.mass,"impulse_velocity":body.linear_velocity.x,"angular_speed":body.angular_velocity.length()})
	check(absf(velocities.Ball10g/velocities.Ball15g-1.5)<.01,"Equal impulse must accelerate 10g ball 1.5x the 15g ball")
	props.set_frozen(true)
	var frozen_poses:Array=[]
	for item in props.items:frozen_poses.append(item.body.transform)
	for i in range(12):await physics_frame
	for i in range(props.items.size()):check(props.items[i].body.transform.is_equal_approx(frozen_poses[i]),"Prop advanced while lockstep was frozen")
	props.set_frozen(false)
	for item in props.items:
		check(item.body.linear_velocity.is_equal_approx(item.linear),"Resume lost linear momentum")
		check(item.body.angular_velocity.is_equal_approx(item.angular),"Resume lost angular momentum")
	props.reset()
	for item in props.items:
		check(item.body.transform.is_equal_approx(item.spawn),"Reset failed to restore prop position")
		check(item.body.linear_velocity==Vector3.ZERO and item.body.angular_velocity==Vector3.ZERO,"Reset retained velocity")
	var report={"engine":ProjectSettings.get_setting("physics/3d/physics_engine"),"bodies":metrics,"errors":errors}
	print(JSON.stringify(report))
	if OS.has_environment("MD_LOOSE_REPORT"):FileAccess.open(OS.get_environment("MD_LOOSE_REPORT"),FileAccess.WRITE).store_string(JSON.stringify(report,"  "))
	quit(0 if errors.is_empty() else 1)
