extends Node3D
## Light free bodies, isolated from robot joints/observations. SI units.
## Body transforms are written only for initial placement or explicit reset/kick setup.
const REFERENCE_BALL_MASS := .015
var items: Array[Dictionary] = []
var w: Node3D
var visuals := true
var frozen := true
var selected := 0 # 0 keeps the original policy football.
var reference_mass := REFERENCE_BALL_MASS

func build(workshop: Node3D, visible_geometry: bool, ball_mass: float=REFERENCE_BALL_MASS, layout: Array=[]) -> void:
	w=workshop;visuals=visible_geometry;reference_mass=ball_mass
	_bin("Bin12g",Vector3(-.24,.002,1.28),.8,"paper")
	_bottle("Bottle6g",Vector3(.00,.002,1.26),.4,"porcelain","purple")
	_ball("Ball10g",Vector3(.21,.002,1.29),2.0/3.0,.035,"yellow")
	_bottle("Bottle8g",Vector3(.44,.002,1.41),8.0/15.0,"paper","yellow")
	_ball("Ball15g",Vector3(.65,.002,1.31),1.0,.035,"purple")
	_bin("Bin10g",Vector3(.88,.002,1.42),2.0/3.0,"purple")
	if not layout.is_empty():
		for i in range(items.size()):
			var body: RigidBody3D = items[i].body
			body.position += Vector3(layout[i].x,0,layout[i].y)-Vector3(body.position.x,0,body.position.z)
			body.position.y += w.ground_height(body.position.x,body.position.z)
			items[i].spawn=body.transform
	if visuals and layout.is_empty():
		w._label("05 / LOOSE PARTS",Vector3(.32,.002,1.07),34,.0005,"graphite",Vector3(-90,0,0))
		w._label("12g     6g     10g     8g     15g     10g",Vector3(.32,.002,1.64),24,.00045,"shadow",Vector3(-90,0,0))

func _body(id: String,p: Vector3,ratio: float,friction: float,bounce: float) -> RigidBody3D:
	var body:=RigidBody3D.new();body.name=id;body.position=p
	body.mass=reference_mass*ratio;body.collision_layer=1;body.collision_mask=3
	body.continuous_cd=true;body.freeze=true
	body.max_contacts_reported=16;body.contact_monitor=true
	body.linear_damp=.04;body.angular_damp=.03
	var material:=PhysicsMaterial.new();material.friction=friction;material.bounce=bounce
	body.physics_material_override=material
	add_child(body)
	items.append({"body":body,"spawn":body.transform,"linear":Vector3.ZERO,"angular":Vector3.ZERO})
	return body

func _part(body: RigidBody3D,mesh: Mesh,p: Vector3,color: String,shape: Shape3D=null) -> void:
	if shape==null:shape=mesh.create_convex_shape(true,false)
	shape.margin=.0002
	var collision:=CollisionShape3D.new();collision.shape=shape;collision.position=p
	body.add_child(collision)
	if visuals:
		var visual:=MeshInstance3D.new();visual.mesh=mesh;visual.position=p
		visual.material_override=w.materials[color];body.add_child(visual)

func _trim(body: RigidBody3D,p: Vector3,size: Vector3,color: String) -> void:
	if not visuals:return
	var visual:=MeshInstance3D.new();visual.mesh=w._rounded_box(size,.0005)
	visual.material_override=w.materials[color];visual.position=p;body.add_child(visual)

func _bin(id: String,p: Vector3,ratio: float,color: String) -> void:
	var body:=_body(id,p,ratio,.48,.04)
	body.set_meta("grasp_center",Vector3(0,.045,0))
	body.set_meta("grasp_size",Vector3(.090,.050,.064))
	# A real open tub: floor and four walls, never a solid convex hull across the opening.
	_part(body,w._rounded_box(Vector3(.090,.003,.064),.0006),Vector3(0,.0015,0),color)
	for side in [-1,1]:
		_part(body,w._rounded_box(Vector3(.003,.047,.064),.0006),Vector3(side*.0435,.0265,0),color)
		_part(body,w._rounded_box(Vector3(.084,.047,.003),.0006),Vector3(0,.0265,side*.0305),color)
		_trim(body,Vector3(0,.029,side*.0326),Vector3(.031,.008,.0008),"graphite")
		_trim(body,Vector3(side*.0456,.044,0),Vector3(.0008,.008,.047),"yellow")

func _cylinder(top: float,bottom: float,height: float) -> CylinderMesh:
	var mesh:=CylinderMesh.new();mesh.top_radius=top;mesh.bottom_radius=bottom
	mesh.height=height;mesh.radial_segments=24;mesh.rings=1
	return mesh

func _bottle(id: String,p: Vector3,ratio: float,color: String,cap: String) -> void:
	var body:=_body(id,p,ratio,.35,.06)
	body.set_meta("grasp_center",Vector3(0,.032,0))
	body.set_meta("grasp_size",Vector3(.036,.078,.036))
	_part(body,_cylinder(.017,.018,.045),Vector3(0,.0225,0),color)
	_part(body,_cylinder(.008,.017,.013),Vector3(0,.0515,0),color)
	_part(body,_cylinder(.008,.008,.012),Vector3(0,.064,0),color)
	_part(body,_cylinder(.010,.010,.008),Vector3(0,.074,0),cap)
	_trim(body,Vector3(0,.026,.0182),Vector3(.018,.018,.0008),"purple" if cap=="yellow" else "yellow")
	_trim(body,Vector3(0,.026,.0188),Vector3(.009,.002,.0004),"ink")

func _ball(id: String,p: Vector3,ratio: float,radius: float,color: String) -> void:
	var body:=_body(id,Vector3(p.x,radius,p.z),ratio,.50,.12)
	body.set_meta("grasp_center",Vector3.ZERO)
	body.set_meta("grasp_size",Vector3.ONE*radius*2)
	var mesh:=SphereMesh.new();mesh.radius=radius;mesh.height=radius*2
	mesh.radial_segments=32;mesh.rings=16
	# A small convex shell avoids spontaneous rolling of analytic spheres on
	# the retained large floor/Jolt settings. Submillimetre faceting supplies stable support.
	var hull:=SphereMesh.new();hull.radius=radius;hull.height=radius*2
	hull.radial_segments=20;hull.rings=10
	var shape:=hull.create_convex_shape(true,false)
	_part(body,mesh,Vector3.ZERO,color,shape)
	# Start on a supporting facet, rather than balancing the hull on a pole.
	# This is initial placement only; rotation remains free during simulation.
	var faces:=hull.get_faces()
	for i in range(0,faces.size(),3):
		var a:=faces[i];var b:=faces[i+1];var c:=faces[i+2]
		var normal:Vector3=(b-a).cross(c-a).normalized()
		if normal.dot(a)<0:normal=-normal
		if normal.y<-.98:
			body.basis=Basis(Quaternion(normal,Vector3.DOWN))
			body.position.y=absf(normal.dot(a))+.0002
			break
	items[-1].spawn=body.transform
	if visuals:
		# A separated inset mark follows the body, making spin visible.
		var dot:=SphereMesh.new();dot.radius=.006;dot.height=.012;dot.radial_segments=16;dot.rings=8
		var visual:=MeshInstance3D.new();visual.mesh=dot;visual.position=Vector3(0,0,radius-.005)
		visual.material_override=w.materials.graphite;body.add_child(visual)

func set_frozen(value: bool) -> void:
	if frozen==value:return
	for item in items:
		var body:RigidBody3D=item.body
		if value:
			item.linear=body.linear_velocity;item.angular=body.angular_velocity
		body.freeze=value
		if not value:
			body.linear_velocity=item.linear;body.angular_velocity=item.angular
			body.sleeping=false
	frozen=value

func reset() -> void:
	frozen=true
	for item in items:
		var body:RigidBody3D=item.body
		body.freeze=true;body.transform=item.spawn;body.force_update_transform()
		body.linear_velocity=Vector3.ZERO;body.angular_velocity=Vector3.ZERO
		item.linear=Vector3.ZERO;item.angular=Vector3.ZERO

func cycle_target() -> void:
	if w.server.get("options") is Dictionary and w.server.options.get("scene","workshop") == "science_station":
		for offset in range(1,items.size()+2):
			var index: int = (selected+offset)%(items.size()+1)
			if _nearby(index):
				selected=index;return
		return
	selected=(selected+1)%(items.size()+1)

func _nearby(index:int) -> bool:
	if not (w.server.get("options") is Dictionary) or w.server.options.get("scene","workshop")!="science_station":return true
	var body:RigidBody3D=w.server._bodies.get("ball") if index==0 else items[index-1].body
	return body!=null and body.global_position.distance_to(w.server._base.global_position)<=1.6

func target_label() -> String:
	if not _nearby(selected):return "附近没有目标 · 靠近物件后按 B"
	if selected==0:return "原球 · %.0f g" % (reference_mass*1000.0)
	var body:RigidBody3D=items[selected-1].body
	var kind:="收纳箱" if str(body.name).begins_with("Bin") else "小瓶" if str(body.name).begins_with("Bottle") else "小球"
	return "%s · %.0f g" % [kind,body.mass*1000.0]

func place_target(position: Vector3) -> void:
	if selected==0 or not _nearby(selected):return
	var item:Dictionary=items[selected-1];var body:RigidBody3D=item.body
	var spawn:Transform3D=item.spawn
	var height: float = spawn.origin.y-w.ground_height(spawn.origin.x,spawn.origin.z)+w.ground_height(position.x,position.z)
	body.global_transform=Transform3D(spawn.basis,Vector3(position.x,height,position.z))
	body.force_update_transform();body.linear_velocity=Vector3.ZERO;body.angular_velocity=Vector3.ZERO
	item.linear=Vector3.ZERO;item.angular=Vector3.ZERO;body.sleeping=false

func telemetry() -> Array:
	var result: Array=[]
	for item in items:
		var body:RigidBody3D=item.body;var p:=body.global_position;var q:=body.global_basis.get_rotation_quaternion()
		var v:=body.linear_velocity;var av:=body.angular_velocity
		var contacts: Array=[]
		var state:=PhysicsServer3D.body_get_direct_state(body.get_rid())
		if state!=null:
			for i in range(state.get_contact_count()):
				var other=state.get_contact_collider_object(i)
				contacts.append({"body":str(other.name) if other is Node else "unknown","impulse":state.get_contact_impulse(i).length()})
		result.append({"name":str(body.name),"mass":body.mass,"position":[p.x,p.y,p.z],"quaternion_xyzw":[q.x,q.y,q.z,q.w],"linear_velocity":[v.x,v.y,v.z],"angular_velocity":[av.x,av.y,av.z],"contacts":contacts})
	return result
