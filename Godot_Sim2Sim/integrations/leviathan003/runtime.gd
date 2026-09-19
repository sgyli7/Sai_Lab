extends Node3D
## Two native Jolt hulls. Same reduced patch dynamics as source/physics.py.
## This is transport physics: no link-by-link belt solver and no weapons logic.
var bodies:Dictionary={}
var cfg:Dictionary
var policy:PackedFloat64Array=PackedFloat64Array([.8,2.,2.5])
var requested:=Vector2.ZERO
var filtered:=Vector2.ZERO
var patches:Array[Vector3]=[]
var normal_loads:Array=[]
var suspension:Array=[]
var visual_nodes:Dictionary={}
var visual_rest:Dictionary={}
var asset:Node3D
var belt_materials:Dictionary={}
var track_distance:Dictionary={"front":0.,"rear":0.}
var ground_mask:=2
var visual_basis:=Basis(Vector3.RIGHT,-PI/2)
var elapsed:=0.0
var max_speed:=0.0
var min_upright:=1.0
var max_hitch_error:=0.0
var cpu_usec:=0.0
var step_count:=0

func load_bundle(directory:String,placement:Transform3D=Transform3D.IDENTITY,visuals:bool=true)->bool:
	cfg=JSON.parse_string(FileAccess.get_file_as_string(directory.path_join("physics.json")))
	var p:Dictionary=JSON.parse_string(FileAccess.get_file_as_string(directory.path_join("policy.json")))
	policy=PackedFloat64Array(p.weights)
	for q in cfg.patch_offsets:patches.append(Vector3(q[0],q[2],-q[1]))
	for i in 2:
		var id:String="front" if i==0 else "rear"
		var body:=RigidBody3D.new();body.name=id;body.mass=cfg.mass_kg[i]
		body.inertia=Vector3(cfg.inertia_diagonal[0],cfg.inertia_diagonal[2],cfg.inertia_diagonal[1])
		body.center_of_mass_mode=RigidBody3D.CENTER_OF_MASS_MODE_CUSTOM;body.center_of_mass=Vector3.ZERO
		body.linear_damp=0.;body.angular_damp=0.;body.can_sleep=false;body.collision_layer=16|8;body.collision_mask=1|2|4
		body.continuous_cd=true
		body.transform=placement*Transform3D(Basis.IDENTITY,Vector3(cfg.hull_centers_x[i],cfg.com_z,0));add_child(body);bodies[id]=body
		var shape:=BoxShape3D.new();shape.size=Vector3(34,.6,27)
		var col:=CollisionShape3D.new();col.shape=shape;col.position=Vector3(0,-2.8,0);body.add_child(col)
		if id=="front":
			# Coarse exterior contact proxies; no walkable interior or Sai route is implemented.
			_add_box(body,Vector3(25,0,0),Vector3(18,.3,7))
			_add_box(body,Vector3(25,4.9,0),Vector3(18,.2,7))
			for side in [-1,1]:_add_box(body,Vector3(25,2.5,side*3.5),Vector3(18,5,.15))
			_add_box(body,Vector3(34,2.5,0),Vector3(.2,5,7))
		var a:Array=[];a.resize(16);a.fill(0.);normal_loads.append(a)
		var s:Array=[];s.resize(16);s.fill(0.);suspension.append(s)
	var hitch:=PinJoint3D.new();hitch.name="PermanentArticulation";hitch.position=placement*Vector3(cfg.hitch_x,cfg.hitch_z,0);add_child(hitch)
	hitch.node_a=bodies.front.get_path();hitch.node_b=bodies.rear.get_path();hitch.exclude_nodes_from_collision=true
	if visuals:
		var document:=GLTFDocument.new();var state:=GLTFState.new()
		if document.append_from_file(directory.path_join("leviathan003.glb"),state)!=OK:return false
		asset=document.generate_scene(state);asset.name="AuthoredModel";add_child(asset)
		_index(asset)
		for id in visual_nodes:
			var node:Node3D=visual_nodes[id]
			var raw:Transform3D=node.global_transform
			var hull:String="front" if str(id).begins_with("front") else "rear"
			var world:Transform3D=placement*Transform3D(visual_basis,Vector3.ZERO)*raw
			node.reparent(bodies[hull]);node.global_transform=world
			visual_rest[id]=node.transform
			if "_bogie_" in str(id):_attach_belt_shader(node,hull)
	return true

func _attach_belt_shader(node:Node,hull:String)->void:
	if node is MeshInstance3D:
		for i in node.mesh.get_surface_count():
			var source:Material=node.get_active_material(i)
			if source is StandardMaterial3D:
				var material:=ShaderMaterial.new();material.shader=load("res://leviathan003/belt.gdshader")
				material.set_shader_parameter("paint",source.albedo_color);material.set_shader_parameter("metal",source.metallic)
				node.set_surface_override_material(i,material)
				if not belt_materials.has(hull):belt_materials[hull]=[]
				belt_materials[hull].append(material)
	for child in node.get_children():_attach_belt_shader(child,hull)

func _add_box(body:RigidBody3D,pos:Vector3,size:Vector3)->void:
	var s:=BoxShape3D.new();s.size=size;var c:=CollisionShape3D.new();c.position=pos;c.shape=s;body.add_child(c)

func _index(node:Node)->void:
	var name_:String=str(node.name)
	if name_ in ["front","rear"] or ("_bogie_" in name_ and not "__" in name_):visual_nodes[name_]=node
	for child in node.get_children():_index(child)

func set_vehicle_command(speed:float,yaw:float)->void:
	if not is_finite(speed) or not is_finite(yaw):requested=Vector2.ZERO;return
	requested=Vector2(clampf(speed,-5.,cfg.max_speed),yaw)

func get_design_transform(id:String)->Transform3D:
	return bodies[id].global_transform*Transform3D(Basis.IDENTITY,Vector3(0,-2.8,0))

func _physics_process(dt:float)->void:
	if bodies.is_empty():return
	var start:int=Time.get_ticks_usec()
	var front:RigidBody3D=bodies.front
	var speed:float=front.linear_velocity.dot(front.global_basis.x)
	var rate:float=cfg.max_brake if absf(requested.x)<absf(filtered.x) or requested.x*filtered.x<0 else cfg.max_accel
	filtered.x=move_toward(filtered.x,requested.x,rate*dt)
	var yawmax:float=minf(cfg.max_yaw,minf(cfg.lateral_accel/maxf(absf(speed),1.),absf(filtered.x)/50.))
	filtered.y=move_toward(filtered.y,clampf(requested.y,-yawmax,yawmax),.02*dt)
	var accel:float=clampf((filtered.x-speed)*policy[0],-cfg.max_brake,cfg.max_accel)
	accel=clampf(accel,-cfg.max_brake,maxf(0,cfg.power_w/1e7/maxf(absf(speed),2)-cfg.rolling*9.81))
	var space:PhysicsDirectSpaceState3D=get_world_3d().direct_space_state
	var h:=0
	for id in ["front","rear"]:
		var body:RigidBody3D=bodies[id];var rb:Basis=body.global_basis;var pos:Vector3=body.global_position
		var heading:float=atan2(-rb.x.z,rb.x.x);var fh:float=atan2(-front.global_basis.x.z,front.global_basis.x.x)
		var err:float=wrapf(fh-heading,-PI,PI)
		var target_yaw:float=filtered.y+(policy[1]*err if h else 0.)
		var total_force:=Vector3.ZERO;var total_torque:=Vector3.ZERO
		for j in 16:
			var offset:Vector3=rb*patches[j];var p:Vector3=pos+offset
			var query:=PhysicsRayQueryParameters3D.create(p+Vector3.UP*4.,p-Vector3.UP*4.,ground_mask)
			query.exclude=[bodies.front.get_rid(),bodies.rear.get_rid()]
			var hit:Dictionary=space.intersect_ray(query)
			if hit.is_empty():normal_loads[h][j]=0.;continue
			var contact:Vector3=hit.position;var normal:Vector3=hit.normal
			var vv:Vector3=body.linear_velocity+body.angular_velocity.cross(offset)
			var k:float=5e6*9.81/16/.65;var c:float=2*.8*sqrt(k*5e6/16)
			var load_:float=clampf(k*(.65-p.y+contact.y)-c*vv.dot(normal),0,5e6*9.81/16*3)
			normal_loads[h][j]=load_;suspension[h][j]=contact.y-p.y
			var local_y:float=-patches[j].z
			var steer:float=atan2(target_yaw*patches[j].x,maxf(absf(speed),2)-target_yaw*local_y)*signf(filtered.x+1e-9)
			steer=clampf(steer,-.45,.45)
			var forward:Vector3=rb*Vector3(cos(steer),0,-sin(steer));forward=(forward-normal*forward.dot(normal)).normalized()
			var lateral:Vector3=normal.cross(forward) # +Y source is -Z Godot
			var roll:float=-cfg.rolling*load_*tanh(vv.dot(forward)*2)
			var fx:float=5e6/16*accel+roll+5e6/16*cfg.rolling*9.81*tanh(speed*2)
			fx-=5e6/16*clampf((target_yaw-body.angular_velocity.y)*policy[1],-.15,.15)*local_y
			var fy:float=-5e6/16*policy[2]*vv.dot(lateral)
			var tangent:Vector3=forward*fx+lateral*fy
			tangent*=minf(1,cfg.friction*load_/maxf(tangent.length(),1))
			var force:Vector3=normal*load_+tangent
			total_force+=force;total_torque+=(contact-pos).cross(force)
		body.apply_central_force(total_force);body.apply_torque(total_torque)
		track_distance[id]+=body.linear_velocity.dot(rb.x)*dt
		h+=1
	# Same 20 MN m s/rad relative ball damping as MuJoCo.
	var relative:Vector3=bodies.rear.angular_velocity-front.angular_velocity
	bodies.rear.apply_torque(-relative*2e7);front.apply_torque(relative*2e7)
	var a:Vector3=front.global_transform*Vector3(cfg.hitch_x-cfg.hull_centers_x[0],cfg.hitch_z-cfg.com_z,0)
	var b:Vector3=bodies.rear.global_transform*Vector3(cfg.hitch_x-cfg.hull_centers_x[1],cfg.hitch_z-cfg.com_z,0)
	max_hitch_error=maxf(max_hitch_error,a.distance_to(b))
	elapsed+=dt;max_speed=maxf(max_speed,speed);min_upright=minf(min_upright,front.global_basis.y.y)
	cpu_usec+=Time.get_ticks_usec()-start;step_count+=1

func _process(_dt:float)->void:
	for hull in belt_materials:
		for material in belt_materials[hull]:material.set_shader_parameter("travel_m",track_distance[hull])
	for id in visual_nodes:
		if not "_bogie_" in str(id):continue
		var h:int=0 if str(id).begins_with("front") else 1
		var offset_:int=(8 if "_fore_" in str(id) else 0)+(4 if str(id).ends_with("left") else 0)
		var sag:=0.
		for k in 4:sag+=float(suspension[h][offset_+k])*.25
		var node:Node3D=visual_nodes[id];var tr:Transform3D=visual_rest[id];tr.origin.y+=clampf(sag,-1.5,1.5)
		var turn:float=clampf(atan2(filtered.y*tr.origin.x,maxf(absf(filtered.x),2.)),-.45,.45)
		tr.basis=Basis(Vector3.UP,turn)*tr.basis;node.transform=tr

func state()->Dictionary:
	var f:RigidBody3D=bodies.front
	return {"time":elapsed,"speed_m_s":f.linear_velocity.dot(f.global_basis.x),"yaw":f.angular_velocity.y,"upright":f.global_basis.y.y,
		"position":[f.position.x,f.position.y,f.position.z],"max_speed_kmh":max_speed*3.6,"min_upright":min_upright,"hitch_error_m":max_hitch_error,
		"physics_cpu_ms":cpu_usec/maxf(1,step_count)/1000.,"bodies":2,"patches":32}
