extends Node3D
# Ported from the existing r24 independent Godot physics implementation.
# All robot state is advanced by Godot/Jolt; Python only returns motor targets.
var specification: Dictionary
var bodies: Dictionary = {}
var drives: Array = []
var joint_rids: Array[RID] = []
var last_q: Array = []
var item: RigidBody3D
var tick := 0
var collision_margin := 0.0002
var show_visuals := true
var command: Dictionary = {}
var initial_ground_height := 0.0

func setup(spec: Dictionary, visuals: bool, ground_height: float = 0.0) -> void:
	specification = spec
	show_visuals = visuals
	initial_ground_height = ground_height
	build_robot()

func gv(a) -> Vector3:
	return Vector3(float(a[0]),float(a[2]),-float(a[1]))


func source(v: Vector3) -> Array:
	return [v.x,-v.z,v.y]


func basis_from_rows(rows: Array) -> Basis:
	var source_basis := Basis(Vector3(rows[0][0],rows[1][0],rows[2][0]),Vector3(rows[0][1],rows[1][1],rows[2][1]),Vector3(rows[0][2],rows[1][2],rows[2][2]))
	var f := Basis(Vector3(1,0,0),Vector3(0,0,-1),Vector3(0,1,0))
	return f*source_basis*f.transposed()


func material(c: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color=c
	m.roughness=0.65
	return m


func build_robot() -> void:
	for name in specification.bodies:
		var d: Dictionary=specification.bodies[name]
		var body := RigidBody3D.new()
		body.name=name
		body.position=gv(d.origin_m)+Vector3(0,0.224-float(specification.bodies.chassis.origin_m[2])+initial_ground_height,0)
		body.mass=d.mass_kg
		body.center_of_mass_mode=RigidBody3D.CENTER_OF_MASS_MODE_CUSTOM
		body.center_of_mass=gv(d.com_local_m)
		var ii: Array=d.diagonal_inertia_kgm2
		body.inertia=Vector3(ii[0],ii[2],ii[1])
		# Godot's diagonal inertia and child-axis rotor surrogate are an
		# explicit approximation, not MuJoCo's full tensor/relative armature.
		if d.has("joint"):
			var rotor: float=0.028 if d.joint.kind=="arm" else 0.0003 if d.joint.kind=="wheel" else 0.002
			if d.joint.kind=="cargo_slide":rotor=0.0
			elif d.joint.kind=="cargo_drive":rotor=0.00001
			var axis := gv(d.joint.axis)
			body.inertia+=Vector3(axis.x*axis.x,axis.y*axis.y,axis.z*axis.z)*rotor
		body.linear_damp_mode=RigidBody3D.DAMP_MODE_REPLACE
		body.angular_damp_mode=RigidBody3D.DAMP_MODE_REPLACE
		body.linear_damp=0
		body.angular_damp=0
		body.can_sleep=false
		body.collision_layer=1
		body.collision_mask=6
		if name.ends_with("_wheel"):
			body.contact_monitor=true
			body.max_contacts_reported=16
		var pm := PhysicsMaterial.new()
		pm.friction=0.8
		body.physics_material_override=pm
		add_child(body)
		bodies[name]=body
		for c in d.collision:
			var cs := CollisionShape3D.new()
			if c.type=="box":
				var shape := BoxShape3D.new()
				shape.size=Vector3(c.size[0]*2,c.size[2]*2,c.size[1]*2)
				cs.shape=shape
				cs.position=gv(c.pos)
			elif c.type=="cylinder":
				var shape := CylinderShape3D.new()
				shape.radius=c.size[0]
				shape.height=c.size[1]*2
				cs.shape=shape
				cs.position=gv(c.pos)
				cs.rotation.x=PI/2
			elif c.type=="mesh":
				var shape := ConvexPolygonShape3D.new()
				var points := PackedVector3Array()
				for v in c.convex_points_m:points.append(gv(v))
				shape.points=points
				cs.shape=shape
			else:
				var a := gv(c.fromto.slice(0,3))
				var b := gv(c.fromto.slice(3,6))
				var shape := CapsuleShape3D.new()
				shape.radius=c.size[0]
				shape.height=a.distance_to(b)+2*c.size[0]
				cs.shape=shape
				cs.position=(a+b)/2
				cs.quaternion=Quaternion(Vector3.UP,(b-a).normalized())
			cs.shape.margin=collision_margin
			body.add_child(cs)
		if show_visuals:
			var scene=load(d.godot_glb)
			if scene is PackedScene:
				var root=scene.instantiate()
				body.add_child(root)
				for mi in root.find_children("*","MeshInstance3D",true,false):
					var index := int(str(mi.name).get_slice("_",str(mi.name).get_slice_count("_")-1))
					var c: Array=d.visuals[index].rgba
					mi.material_override=material(Color(c[0],c[1],c[2],c[3]))
	for name in specification.bodies:
		var d: Dictionary=specification.bodies[name]
		if d.parent==null:continue
		var parent: RigidBody3D=bodies[d.parent]
		var child: RigidBody3D=bodies[name]
		var axis := gv(d.joint.axis).normalized()
		if d.joint.kind=="cargo_slide":
			var slider_align := Basis(Quaternion(Vector3.RIGHT,axis))
			var slider := PhysicsServer3D.joint_create()
			PhysicsServer3D.joint_make_slider(slider,parent.get_rid(),Transform3D(slider_align,child.position-parent.position),child.get_rid(),Transform3D(slider_align,Vector3.ZERO))
			PhysicsServer3D.slider_joint_set_param(slider,PhysicsServer3D.SLIDER_JOINT_LINEAR_LIMIT_LOWER,0.0)
			PhysicsServer3D.slider_joint_set_param(slider,PhysicsServer3D.SLIDER_JOINT_LINEAR_LIMIT_UPPER,0.067)
			PhysicsServer3D.slider_joint_set_param(slider,PhysicsServer3D.SLIDER_JOINT_ANGULAR_LIMIT_LOWER,0.0)
			PhysicsServer3D.slider_joint_set_param(slider,PhysicsServer3D.SLIDER_JOINT_ANGULAR_LIMIT_UPPER,0.0)
			joint_rids.append(slider)
			drives.append({"parent":parent,"child":child,"axis":axis,"kind":d.joint.kind,"rest":child.position-parent.position})
			last_q.append(0.0)
			continue
		var align := Basis(Quaternion(Vector3(0,0,1),axis))
		var center: float=0 if d.joint.kind=="wheel" else (d.joint.range_rad[0]+d.joint.range_rad[1])/2
		# Center the hinge's reference interval. Godot measures the opposite
		# sign, angle=center-q; this preserves SO101 travel across q=-PI.
		var frame_a := Transform3D(align,child.position-parent.position)
		var frame_b := Transform3D(Basis(axis,-center)*align,Vector3.ZERO)
		var rid := PhysicsServer3D.joint_create()
		PhysicsServer3D.joint_make_hinge(rid,parent.get_rid(),frame_a,child.get_rid(),frame_b)
		PhysicsServer3D.hinge_joint_set_flag(rid,PhysicsServer3D.HINGE_JOINT_FLAG_USE_LIMIT,d.joint.kind!="wheel")
		if d.joint.kind!="wheel":
			PhysicsServer3D.hinge_joint_set_param(rid,PhysicsServer3D.HINGE_JOINT_LIMIT_LOWER,center-d.joint.range_rad[1])
			PhysicsServer3D.hinge_joint_set_param(rid,PhysicsServer3D.HINGE_JOINT_LIMIT_UPPER,center-d.joint.range_rad[0])
		joint_rids.append(rid)
		drives.append({"parent":parent,"child":child,"axis":axis,"kind":d.joint.kind})
		last_q.append(0.0)


func build_item() -> void:
	item=RigidBody3D.new()
	item.set_script(load("res://item_observation.gd"))
	item.name="item"
	item.mass=0.1
	item.can_sleep=false
	item.linear_damp_mode=RigidBody3D.DAMP_MODE_REPLACE
	item.angular_damp_mode=RigidBody3D.DAMP_MODE_REPLACE
	item.linear_damp=0
	item.angular_damp=0
	item.collision_layer=4
	item.collision_mask=3
	item.contact_monitor=true
	item.max_contacts_reported=64
	var h: Array=specification.object.initial_pose_m
	item.position=gv([h[0][3],h[1][3],h[2][3]])
	item.basis=basis_from_rows(h)
	var p := PhysicsMaterial.new()
	p.friction=0.8
	item.physics_material_override=p
	add_child(item)
	var c := CollisionShape3D.new()
	var shape := BoxShape3D.new()
	var size: Array=specification.object.size_m
	shape.size=Vector3(size[0],size[2],size[1])
	shape.margin=collision_margin
	c.shape=shape
	item.add_child(c)
	var mi := MeshInstance3D.new()
	var mesh := BoxMesh.new()
	mesh.size=shape.size
	mi.mesh=mesh
	mi.material_override=material(Color(0.95,0.60,0.12))
	item.add_child(mi)


func state() -> Dictionary:
	var base: RigidBody3D=bodies.chassis
	var q: Array=[]
	var v: Array=[]
	for i in range(drives.size()):
		var d: Dictionary=drives[i]
		if d.kind=="cargo_slide":
			var relative_position: Vector3=d.parent.global_basis.inverse()*(d.child.position-d.parent.position)-d.rest
			q.append(relative_position.dot(d.axis))
			var velocity: Vector3=d.child.linear_velocity-d.parent.linear_velocity-d.parent.angular_velocity.cross(d.child.position-d.parent.position)
			v.append(velocity.dot(d.parent.global_basis*d.axis))
			continue
		var relative: Basis=d.parent.global_basis.inverse()*d.child.global_basis
		var quat := relative.orthonormalized().get_rotation_quaternion()
		var raw := wrapf(2*atan2(Vector3(quat.x,quat.y,quat.z).dot(d.axis),quat.w),-PI,PI)
		var angle: float=last_q[i]+wrapf(raw-last_q[i],-PI,PI)
		q.append(angle)
		last_q[i]=angle
		var axis: Vector3=d.parent.global_basis*d.axis
		v.append((d.child.angular_velocity-d.parent.angular_velocity).dot(axis))
	var gripper: RigidBody3D=bodies.arm_gripper
	var cargo := cargo_state()
	return {"time":tick*0.0005,"q":q,"v":v,"base_position":source(base.position),
		"cargo_supported":cargo.supported,"cargo_bounds_m":cargo.bounds,
		"cargo_bilateral":cargo.bilateral,"cargo_inside":cargo.inside,
		"tool_m":source(gripper.global_transform*gv(specification.tool_local_m)),
		"base_rotation_columns":[source(base.global_basis*gv([1,0,0])),source(base.global_basis*gv([0,1,0])),source(base.global_basis*gv([0,0,1]))],
		"base_linear_world":source(base.linear_velocity),"base_angular_world":source(base.angular_velocity)}


func cargo_state() -> Dictionary:
	if item == null:return {"supported":false,"inside":false,"bilateral":false,"bounds":[]}
	var base: RigidBody3D=bodies.chassis
	var h := base.global_transform.affine_inverse()*item.global_transform
	var dims: Array=specification.object.size_m
	var low := Vector3(INF,INF,INF)
	var high := Vector3(-INF,-INF,-INF)
	for x in [-1,1]:
		for y in [-1,1]:
			for z in [-1,1]:
				var p := h*gv([x*dims[0]/2,y*dims[1]/2,z*dims[2]/2])+gv(specification.bodies.chassis.origin_m)
				var a: Array=source(p)
				low=low.min(Vector3(a[0],a[1],a[2]))
				high=high.max(Vector3(a[0],a[1],a[2]))
	var supported := false
	var left := false
	var right := false
	for b in item.get_colliding_bodies():
		if str(b.name)=="chassis":supported=true
		if str(b.name)=="cargo_slide_-1":left=true
		if str(b.name)=="cargo_slide_1":right=true
		if str(b.name).begins_with("arm_"):supported=false;break
	var inside := low.x>=-0.146 and high.x<=-0.037 and low.y>=-0.112 and high.y<=0.112 and low.z>.254 and high.z<.315
	supported=supported and inside
	return {"supported":supported,"inside":inside,"bilateral":left and right,"bounds":[[low.x,low.y,low.z],[high.x,high.y,high.z]]}


func apply_cargo(s: Dictionary) -> void:
	var ratio: float=specification.cargo.drive_metres_per_radian
	var rotor: Dictionary=drives[24]
	var motor: float=clampf(4.0*(command.get("cargo_target_rad",0.0)-s.q[24])-0.06*s.v[24],-0.12,0.12)-0.002*s.v[24]
	# Force-level elastic belt approximation: one motor, two passive sliders.
	# Spring forces act on the robot only. Never reposition or attach the item.
	for i in [22,23]:
		var d: Dictionary=drives[i]
		var tension: float=20000.0*(ratio*s.q[24]-s.q[i])+4.0*(ratio*s.v[24]-s.v[i])
		var f: Vector3=(d.parent.global_basis*d.axis)*(tension-1.0*s.v[i])
		d.child.apply_force(f,Vector3.ZERO)
		d.parent.apply_force(-f,d.child.position-d.parent.position)
		motor-=ratio*tension
	var torque: Vector3=(rotor.parent.global_basis*rotor.axis)*motor
	rotor.child.apply_torque(torque)
	rotor.parent.apply_torque(-torque)


func _exit_tree() -> void:
	for rid in joint_rids:PhysicsServer3D.free_rid(rid)

func apply_command(s: Dictionary, next_command: Dictionary) -> void:
	command = next_command
	for i in range(22):
		var d: Dictionary=drives[i]
		var u: float
		if i<16:
			if d.kind=="wheel":
				if command.mode=="transport":u=clampf(0.4*(command.wheel_speed[i/4]-s.v[i]),-1.3,1.3)
				else:u=clampf(2*(command.target_leg[i]-s.q[i])-0.4*s.v[i],-1.3,1.3)
			else:u=clampf(80*(command.target_leg[i]-s.q[i])-2*s.v[i],-8,8)
			u-=0.03*s.v[i]
		else:
			var cap: float=command.grip_cap if i==21 else 2.94
			u=clampf(998.22*(command.target_arm[i-16]-s.q[i])-2.731*s.v[i]+command.arm_bias[i-16],-cap,cap)
			u-=0.60*s.v[i]+0.052*tanh(s.v[i]/0.01)
		var torque: Vector3=(d.parent.global_basis*d.axis)*u
		d.child.apply_torque(torque)
		d.parent.apply_torque(-torque)
	apply_cargo(s)
	tick+=1
