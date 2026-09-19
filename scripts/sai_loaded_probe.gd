extends "res://main.gd"
var experiment:Dictionary
var physics_rows:Array=[]
var previous_velocity:=Vector3.ZERO
var previous_deck_velocity:=Vector3.ZERO
var filtered:=Vector3.ZERO
var previous_filtered:=Vector3.ZERO
var cargo_initial:=Vector3.ZERO
var initial_recorded:=false
var max_slip:=0.0
var lost:=false
var contact_impulse_peak:=0.0
var latest_ground:Array=[0.,0.,0.,0.]

func _ready() -> void:
	experiment=JSON.parse_string(FileAccess.get_file_as_string("res://experiment.json"))
	super._ready()
	robot.build_item()
	robot.item.mass=float(experiment.mass)
	# Initial placement only: a free physical body resting on the existing tray.
	robot.item.position=Vector3(-.09,.284+float(experiment.initial_ground),0.)
	robot.item.basis=Basis(Vector3.UP,PI/2) if experiment.get("clamped",false) else Basis.IDENTITY

func build_ground() -> void:
	robot.set_script(preload("res://sai/compliant_robot.gd"))
	var terrain:Dictionary=experiment.terrain
	var faces:=PackedVector3Array()
	for j in range(terrain.y.size()-1):
		for i in range(terrain.x.size()-1):
			var a:=Vector3(terrain.x[i],terrain.z[j][i],-terrain.y[j])
			var b:=Vector3(terrain.x[i+1],terrain.z[j][i+1],-terrain.y[j])
			var c:=Vector3(terrain.x[i],terrain.z[j+1][i],-terrain.y[j+1])
			var d:=Vector3(terrain.x[i+1],terrain.z[j+1][i+1],-terrain.y[j+1])
			faces.append_array(PackedVector3Array([a,b,c,b,d,c]))
	var ground:=StaticBody3D.new();ground.collision_layer=2;ground.collision_mask=5
	var pm:=PhysicsMaterial.new();pm.friction=.8;ground.physics_material_override=pm
	add_child(ground)
	var shape:=ConcavePolygonShape3D.new();shape.set_faces(faces);shape.backface_collision=true
	var collision:=CollisionShape3D.new();collision.shape=shape;ground.add_child(collision)
	if riser>0.:
		var start:float=float(experiment.stair_start)
		var bounds:Array=[-4.,start,start+.18,start+.36,start+.54,4.]
		for i in range(5):box_surface("stair_"+str(i),bounds[i],bounds[i+1],2.,riser*(4-i if descending else i),.5)

func movement_command() -> Array:
	var t:float=robot.sim_time_seconds()
	if t<1. or cleared_at>=0. or (riser<=0. and t>=7.):return [0.,0.,0.]
	return [.16 if riser>0. else float(experiment.get("drive_speed",.5)),float(experiment.get("turn_rate",0.)),0.]

func exchange(state:Dictionary) -> Dictionary:
	state["terrain_path_heights"]=preload("res://sai/terrain_scan.gd").wheel_path(self,2)
	state["terrain_edge_heights"]=preload("res://sai/terrain_scan.gd").edge_profile(self,2)
	latest_ground=preload("res://sai/terrain_scan.gd").wheel_ground(self,2)
	state["wheel_ground_heights"]=latest_ground
	var result:Dictionary=super.exchange(state)
	if experiment.get("clamped",false):result["cargo_target_rad"]=.067/.01909859317102744
	return result

func _physics_process(delta:float) -> void:
	if robot!=null and robot.item!=null and not finished:
		var base:RigidBody3D=robot.bodies.chassis
		var item:RigidBody3D=robot.item
		var t:float=robot.sim_time_seconds()
		if riser>0. and cleared_at<0.:
			var cleared:=true
			for leg in specification.leg_order:
				if robot.bodies[str(leg)+"_wheel"].position.x<=float(experiment.stair_start)+.54+.20:cleared=false
			if cleared:cleared_at=t
		var local:Vector3=base.global_transform.affine_inverse()*item.global_position
		var cargo:Vector3=Vector3(local.x,-local.z,local.y)
		if t>=1. and not initial_recorded:cargo_initial=cargo;initial_recorded=true
		if initial_recorded:
			max_slip=maxf(max_slip,Vector2(cargo.x-cargo_initial.x,cargo.y-cargo_initial.y).length())
			lost=lost or cargo.z<.02 or absf(cargo.y)>.12 or cargo.x<-.17 or cargo.x>-.025
		var deck_offset:Vector3=base.global_basis*Vector3(-.09,.06,0.)
		var deck_velocity:Vector3=base.linear_velocity+base.angular_velocity.cross(deck_offset)
		var acc:Vector3=(item.linear_velocity-previous_velocity)/delta
		var deck_acc:Vector3=(deck_velocity-previous_deck_velocity)/delta
		filtered+=(acc-filtered)*(delta/(.015+delta))
		var jerk:Vector3=(filtered-previous_filtered)/delta
		var supported:=false
		var left_pad:=false
		var right_pad:=false
		for body in item.get_colliding_bodies():
			if body==base:supported=true
			if body.name=="cargo_slide_-1":left_pad=true
			if body.name=="cargo_slide_1":right_pad=true
		var ds:PhysicsDirectBodyState3D=PhysicsServer3D.body_get_direct_state(item.get_rid())
		if ds!=null and t>=1.5:
			for i in range(ds.get_contact_count()):contact_impulse_peak=maxf(contact_impulse_peak,ds.get_contact_impulse(i).length())
		if t>=1.5:
			physics_rows.append([t,acc.x,-acc.z,acc.y,filtered.x,-filtered.z,filtered.y,jerk.x,-jerk.z,jerk.y,deck_acc.x,-deck_acc.z,deck_acc.y,
				base.global_basis.y.y,cargo.z,base.position.x,base.linear_velocity.x,
				base.position.y-(float(latest_ground[0])+float(latest_ground[1])+float(latest_ground[2])+float(latest_ground[3]))/4.-.2192,1. if supported else 0.,1. if left_pad and right_pad else 0.])
		previous_velocity=item.linear_velocity;previous_deck_velocity=deck_velocity;previous_filtered=filtered
	super._physics_process(delta)

func finish_run() -> void:
	var file:=FileAccess.open(output.get_base_dir().path_join("cargo-physics.json"),FileAccess.WRITE)
	file.store_string(JSON.stringify({"samples":physics_rows,"max_slip":max_slip,"cargo_lost":lost,"contact_impulse_peak_Ns":contact_impulse_peak}))
	file.close()
	super.finish_run()
