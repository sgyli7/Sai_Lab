extends Node3D
const VisualProfile = preload("res://atelier/visual_profile.gd")
## Original procedural artwork in metres; major solids share their mesh with static collision.
const PALETTE := {
	"paper": Color("c6c4b7"), "porcelain": Color("d9d7c8"),
	"graphite": Color("6c6a68"), "ink": Color("24232b"),
	"yellow": Color("e0bd38"), "purple": Color("89729e"),
	"metal": Color("969b95"), "floor": Color("b2b1a2"),
	"shadow": Color("727c79"), "screen": Color("454e4d"),
	"light": Color("efe5b8"), "rubber": Color("39383d")}
const VIEWS := {
	"overview": [Vector3(3.0,2.2,3.6),Vector3(0,.18,-.12),44.0],
	"robot": [Vector3(.39,.29,.42),Vector3(0,.16,0),40.0],
	"backlight": [Vector3(-.9,.47,-.9),Vector3(0,.16,0),46.0],
	"corridor": [Vector3(-1.1,.52,1.25),Vector3(.55,.30,-1.0),51.0],
	"detail": [Vector3(.9,.67,-.25),Vector3(.40,.38,-1.25),42.0]}
var server: Node3D
var camera: Camera3D
var view := "follow"
var materials: Dictionary = {}
var builders: Dictionary = {}
var primitive_cache: Dictionary = {}
var world_objects := 0
var tour_time := 0.0
var hud: CanvasLayer
var mesh_roles: Dictionary = {}
var normal_meshes: Dictionary = {}
var follow_initialized := false
var follow_look := Vector3.ZERO
var previous_view := "follow"
var camera_obstacles: Array[AABB] = []
var rotors: Array[MeshInstance3D] = []
var capture_camera: Dictionary = {}
var open_route := VisualProfile.value("MD_OPEN_ROUTE")=="1"
var printed_labels: Array[Dictionary] = []
var visuals_enabled := true
var collisions_enabled := OS.get_environment("MD_WORKSHOP_COLLISIONS") != "0"
var collision_body: StaticBody3D
var collision_cache: Dictionary = {}
var contact_course: Node3D
var loose_props: Node3D
var solid_scope := false
var palette: Dictionary = PALETTE.duplicate()
var label_font: Font

func build(host: Node3D) -> void:
	server=host; name="AtelierVisuals"
	visuals_enabled=not server._headless
	if collisions_enabled:
		collision_body=StaticBody3D.new();collision_body.name="WorkshopCollisions"
		collision_body.collision_layer=1;collision_body.collision_mask=1
		collision_body.physics_material_override=server.get_node("World/Floor").physics_material_override
		add_child(collision_body)
	if not visuals_enabled:
		_build_solids()
		return
	camera=server.get_node("World/Camera3D")
	camera.fov=48.0; camera.near=.015; camera.far=100.0
	for key in palette:
		var mat:=ShaderMaterial.new()
		mat.shader=load("res://atelier/enamel.gdshader")
		mat.set_shader_parameter("pigment",palette[key])
		if OS.get_environment("MD_HATCH")=="0":mat.set_shader_parameter("hatch_strength",0.0)
		var ink:=ShaderMaterial.new()
		ink.shader=load("res://atelier/ink.gdshader")
		ink.set_shader_parameter("line_pixels",.72)
		if OS.get_environment("MD_INK") != "0": mat.next_pass=ink
		materials[key]=mat
		var builder:=SurfaceTool.new()
		builder.begin(Mesh.PRIMITIVE_TRIANGLES); builders[key]=builder
	_environment(); _floor(); _build_solids()
	_build_details()
	_commit_geometry()
	var all_roles: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://atelier/visual_mesh_roles.json"))
	var model_name: String = "microduck_roller" if "roller" in server._robot_scene else "microduck"
	mesh_roles = all_roles[model_name]
	if VisualProfile.value("MD_ROBOT_NORMALS")=="1":
		normal_meshes=JSON.parse_string(FileAccess.get_file_as_string("res://atelier/robot_normal_map.json"))[model_name]
	elif VisualProfile.value("MD_ROBOT_NORMALS")=="soft":
		normal_meshes=JSON.parse_string(FileAccess.get_file_as_string("res://atelier/robot_normal_map_soft.json"))[model_name]
	_paint_robot(server.get_node("RobotHost")); _install_hud()
	if OS.get_environment("MD_MODE")=="tour":set_view("tour")
	if OS.has_environment("MD_SHOWCASE_SHOT"):set_view("showcase")
	print("ATELIER built ",world_objects," original parts into ",materials.size()," material batches")

func _build_details() -> void:
	load("res://atelier/workshop_details.gd").new().build(self)

func ground_height(_x: float, _z: float) -> float:
	return 0.0

func _build_solids() -> void:
	solid_scope=true
	_architecture()
	_workbench(Vector3(-.52,0,-1.22))
	_power_station(Vector3(-1.55,0,-.70))
	_storage(Vector3(1.42,0,-.78))
	_service_post(Vector3(.74,0,-1.34))
	_cargo_group(); _small_details()
	load("res://atelier/workshop_yard.gd").new().build(self)
	load("res://atelier/neighbourhood.gd").new().build(self)
	solid_scope=false
	if collisions_enabled and OS.get_environment("MD_STATIC_COURSE")=="1":
		contact_course=load("res://atelier/contact_course.gd").new()
		contact_course.name="ContactCourse";add_child(contact_course)
		contact_course.build(self,visuals_enabled)
	if collisions_enabled and OS.get_environment("MD_STATIC_COURSE")!="1":
		loose_props=load("res://atelier/loose_props.gd").new()
		loose_props.name="LooseProps";add_child(loose_props)
		var reference_mass:=.015
		var bodies:Variant=server.get("_bodies")
		if bodies is Dictionary and bodies.has("ball"):reference_mass=bodies.ball.mass
		loose_props.build(self,visuals_enabled,reference_mass)

func _solid(mesh: Mesh, p: Vector3, basis: Basis=Basis.IDENTITY) -> void:
	if not collisions_enabled or not solid_scope or collision_body==null:return
	var id:=mesh.get_instance_id()
	if not collision_cache.has(id):
		var shape:=mesh.create_convex_shape(true,false)
		shape.margin=.0002
		collision_cache[id]=shape
	var collider:=CollisionShape3D.new()
	collider.name="Solid_%03d" % collision_body.get_child_count()
	collider.shape=collision_cache[id]
	collider.transform=Transform3D(basis,p)
	collision_body.add_child(collider)

func _environment() -> void:
	var env:=Environment.new()
	env.background_mode=Environment.BG_COLOR; env.background_color=Color("babfb3")
	if OS.get_environment("MD_NEIGHBOURHOOD")!="0":
		var sky:=Sky.new();var sky_material:=ShaderMaterial.new()
		sky_material.shader=load("res://atelier/district_sky.gdshader");sky.sky_material=sky_material
		env.sky=sky;env.background_mode=Environment.BG_SKY
	env.ambient_light_source=Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color=Color("c9c4ce"); env.ambient_light_energy=.45
	env.tonemap_mode=Environment.TONE_MAPPER_LINEAR
	env.fog_enabled=true; env.fog_light_color=Color("babfb3"); env.fog_density=.012
	# Keep atmospheric depth on geometry without washing the whole sky to one color.
	if OS.get_environment("MD_NEIGHBOURHOOD")!="0":env.fog_sky_affect=.18
	server.get_node("WorldEnvironment").environment=env
	var sun:=server.get_node("World/Sun") as DirectionalLight3D
	sun.rotation_degrees=Vector3(-48,-32,0); sun.light_color=Color("fff9ed"); sun.light_energy=.52
	if OS.has_environment("MD_SUN_ENERGY"):sun.light_energy=float(OS.get_environment("MD_SUN_ENERGY"))
	# Match the small robot/props: .1 detaches Forward+ contact shadows.
	sun.shadow_enabled=OS.get_environment("MD_SHADOWS") != "0"; sun.shadow_bias=.005; sun.shadow_normal_bias=2.0
	sun.directional_shadow_max_distance=6.0
	sun.directional_shadow_mode=DirectionalLight3D.SHADOW_ORTHOGONAL
	server.get_node("World/FillLight").light_energy=.08

func _add(mesh: Mesh,p: Vector3,color: String,basis: Basis=Basis.IDENTITY) -> void:
	if not visuals_enabled:return
	(builders[color] as SurfaceTool).append_from(mesh,0,Transform3D(basis,p)); world_objects+=1

func _box(p: Vector3,size: Vector3,color: String,bevel: float=.007,rot: float=0.0) -> void:
	if (size.y>.035 and maxf(size.x,size.z)>.10) or (size.y>=.02 and maxf(size.x,size.z)>.40):
		var bounds:=AABB(-size*.5,size)
		camera_obstacles.append(Transform3D(Basis(Vector3.UP,rot),p)*bounds)
	var key:=str(size)+":"+str(bevel)
	if not primitive_cache.has(key): primitive_cache[key]=_rounded_box(size,bevel)
	_add(primitive_cache[key],p,color,Basis(Vector3.UP,rot))
	# Paint, seams and tiny fittings have no independent collision. Structural
	# parts keep their rounded mesh hull, including open spaces under furniture.
	if size[size.min_axis_index()]>=.018 and p.y+size.y*.5>.015:
		_solid(primitive_cache[key],p,Basis(Vector3.UP,rot))

func _rounded_box(size: Vector3,radius: float) -> ArrayMesh:
	var half:=size*.5
	var r:=minf(radius,minf(half.x,minf(half.y,half.z))*.90)
	var inner:=half-Vector3.ONE*r
	var st:=SurfaceTool.new(); st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for axis in range(3):
		for s in [-1.0,1.0]:
			var normal:=Vector3.ZERO; normal[axis]=s
			var u:=Vector3.ZERO; u[(axis+1)%3]=1.0
			var v:=normal.cross(u)
			var hu:=half[(axis+1)%3]; var hv:=absf(v.dot(half))
			var xs: Array[float]=[-hu,-hu+r,hu-r,hu]; var ys: Array[float]=[-hv,-hv+r,hv-r,hv]
			for i in range(3):
				for j in range(3):
					for ij in [Vector2i(i,j),Vector2i(i,j+1),Vector2i(i+1,j),Vector2i(i+1,j),Vector2i(i,j+1),Vector2i(i+1,j+1)]:
						var q: Vector3=normal*half[axis]+u*xs[ij.x]+v*ys[ij.y]
						var core:=q.clamp(-inner,inner); var n:=(q-core).normalized()
						st.set_normal(n); st.set_uv(Vector2(float(ij.x)/3.0,float(ij.y)/3.0))
						st.add_vertex(core+n*r)
	st.index()
	return st.commit()

func _cylinder(p: Vector3,radius: float,height: float,color: String,axis: Vector3=Vector3.UP) -> void:
	var mesh:=CylinderMesh.new()
	mesh.top_radius=radius;mesh.bottom_radius=radius;mesh.height=height
	mesh.radial_segments=16;mesh.rings=1
	var basis:=Basis.IDENTITY
	if axis.normalized().dot(Vector3.UP)<.999: basis=Basis(Quaternion(Vector3.UP,axis.normalized()))
	_add(mesh,p,color,basis)
	if radius>=.011 and height>=.015:_solid(mesh,p,basis)

func _line(a: Vector3,b: Vector3,radius: float=.0013,color: String="ink") -> void:
	var d:=b-a
	if d.length()<.0001:return
	_cylinder((a+b)*.5,radius,d.length(),color,d.normalized())

func _label(text: String,p: Vector3,size: int=42,pixel: float=.0005,color: String="ink",rot: Vector3=Vector3.ZERO) -> void:
	if not visuals_enabled:return
	var label:=Label3D.new()
	if label_font!=null:label.font=label_font
	label.text=text;label.position=p;label.font_size=size*3;label.pixel_size=pixel/3.0
	label.modulate=palette[color];label.outline_size=0
	label.no_depth_test=false;label.shaded=false;label.rotation_degrees=rot
	label.texture_filter=BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS
	add_child(label)
	if VisualProfile.value("MD_LABEL_LOD")=="1":printed_labels.append({"node":label,"color":palette[color],"em":size*pixel})

func update_printed_labels() -> void:
	if printed_labels.is_empty() or camera==null:return
	var projection_scale:=float(get_viewport().size.y)/(2.0*tan(deg_to_rad(camera.fov)*.5))
	var world_to_camera:=camera.global_transform.affine_inverse()
	for item in printed_labels:
		var label: Label3D=item.node
		var depth: float=-(world_to_camera*label.global_position).z
		var pixels: float=float(item.em)*.75*projection_scale/maxf(.01,depth)
		var opacity:=smoothstep(3.0,7.0,pixels)
		if absf(label.modulate.a-opacity)>.005:
			var color: Color=item.color;color.a=opacity;label.modulate=color

func _bolts(p: Vector3,size: Vector2) -> void:
	for x in [-1,1]:
		for y in [-1,1]:
			var point:=p+Vector3(x*size.x*.5,y*size.y*.5,0)
			_cylinder(point,.005,.002,"metal",Vector3.FORWARD)
			_line(point+Vector3(-.002,0,.0015),point+Vector3(.002,0,.0015),.0006)

func _panel(p: Vector3,size: Vector2,color: String) -> void:
	_box(p,Vector3(size.x,size.y,.014),"ink",.008)
	_box(p+Vector3(0,0,.009),Vector3(size.x-.008,size.y-.008,.009),color,.007)
	_bolts(p+Vector3(0,0,.016),size-Vector2.ONE*.027)

func _floor() -> void:
	var floor_node:=server.get_node("World/Floor/FloorMesh") as MeshInstance3D
	var ground:=ShaderMaterial.new()
	ground.shader=load("res://atelier/floor.gdshader")
	ground.set_shader_parameter("pigment",PALETTE.floor)
	ground.set_shader_parameter("seam_color",PALETTE.shadow)
	ground.set_shader_parameter("backdrop",Color("babfb3"))
	ground.set_shader_parameter("open_route",open_route)
	ground.set_shader_parameter("yard_dressing",OS.get_environment("MD_YARD_DRESSING")!="0")
	ground.set_shader_parameter("neighbourhood",OS.get_environment("MD_NEIGHBOURHOOD")!="0")
	ground.set_shader_parameter("bay_paint",PALETTE.yellow)
	floor_node.set_surface_override_material(0,ground)
	# Flush parking paint is evaluated once in the floor shader, including corners.
	_label("MD / 01",Vector3(0,.002,.32),46,.0007,"ink",Vector3(-90,0,0))
	for i in range(8):_box(Vector3(.75+i*.11,.001,.67),Vector3(.055,.001,.018),"porcelain",.0002)
	for x in [-1.25,1.25]:
		_box(Vector3(x,.0006,0),Vector3(.10,.001,1.8),"shadow",.0003)
		for i in range(34):_box(Vector3(x,.0015,-.86+i*.052),Vector3(.085,.001,.008),"metal",.0002)

func _architecture() -> void:
	_box(Vector3(0,.42,-1.63),Vector3(3.55,.84,.15),"paper",.025)
	_box(Vector3(0,.055,-1.53),Vector3(3.6,.10,.20),"graphite",.014)
	_box(Vector3(0,.83,-1.575),Vector3(3.65,.065,.25),"porcelain",.014)
	for x in [-1.72,-.82,.24,1.72]:_box(Vector3(x,.43,-1.532),Vector3(.036,.73,.04),"metal",.005)
	for x in [-1.20,-.32,.93]:
		_panel(Vector3(x,.51,-1.535),Vector2(.65,.45),"porcelain")
		for j in range(7):_line(Vector3(x-.29,.32+j*.052,-1.516),Vector3(x+.29,.32+j*.052,-1.516),.001)
	for x in [-1.80,1.80]:
		if open_route and x>0:
			_box(Vector3(x,.20,-1.40),Vector3(.10,.40,.30),"paper",.016)
			_box(Vector3(x,.41,-1.40),Vector3(.13,.035,.32),"porcelain",.006)
			for z in [-1.4]:
				_box(Vector3(x,.27,z),Vector3(.145,.54,.05),"graphite",.007)
				_box(Vector3(x,.40,z+.031),Vector3(.105,.035,.012),"yellow",.003)
			continue
		if open_route and x<0:
			for ends in [Vector2(-1.55,-.37),Vector2(.44,.96)]:
				var center: float=(ends.x+ends.y)*.5
				var length: float=ends.y-ends.x
				_box(Vector3(x,.20,center),Vector3(.10,.40,length),"paper",.016)
				_box(Vector3(x,.41,center),Vector3(.13,.035,length+.02),"porcelain",.006)
			for z in [-1.4,-.38,.45,.9]:
				_box(Vector3(x,.27,z),Vector3(.145,.54,.05),"graphite",.007)
			continue
		_box(Vector3(x,.20,-.30),Vector3(.10,.40,2.5),"paper",.016)
		_box(Vector3(x,.41,-.30),Vector3(.13,.035,2.52),"porcelain",.006)
		for z in [-1.4,-.55,.3,.9]:_box(Vector3(x,.27,z),Vector3(.145,.54,.05),"graphite",.007)
	_line(Vector3(-1.66,.74,-1.44),Vector3(1.60,.74,-1.44),.018,"graphite")
	_line(Vector3(-1.66,.72,-1.44),Vector3(-1.66,.16,-1.44),.018,"graphite")
	for x in [-1.45,-.9,-.2,.45,1.1,1.55]:_box(Vector3(x,.74,-1.465),Vector3(.035,.075,.042),"metal",.004)
	_label("MICRODUCK / SERVICE & SUPPLY",Vector3(-.37,.815,-1.442),42,.00046)
	_label("小 小 维 修 站",Vector3(-.40,.69,-1.42),36,.00048)
	_box(Vector3(-.65,.82,-1.13),Vector3(1.75,.035,.65),"paper",.012)
	for x in [-1.44,.13]:_line(Vector3(x,.64,-1.51),Vector3(x,.79,-.84),.014,"graphite")
	_box(Vector3(-.65,.788,-1.00),Vector3(.58,.014,.035),"light",.005)

func _workbench(p: Vector3) -> void:
	for x in [-.53,.53]:
		for z in [-.16,.16]:_box(p+Vector3(x,.19,z),Vector3(.036,.38,.036),"graphite",.005)
	_box(p+Vector3(0,.39,0),Vector3(1.20,.046,.448),"graphite",.012)
	_box(p+Vector3(0,.418,0),Vector3(1.17,.012,.43),"metal",.004)
	_box(p+Vector3(-.33,.21,0),Vector3(.40,.31,.37),"paper",.010)
	for y in [.13,.23,.33]:
		_panel(p+Vector3(-.33,y,.192),Vector2(.36,.08),"yellow" if y>.3 else "porcelain")
		_box(p+Vector3(-.33,y,.215),Vector3(.12,.012,.018),"graphite",.004)
	_box(p+Vector3(.30,.115,0),Vector3(.45,.027,.36),"graphite",.004)
	_crate(p+Vector3(.29,.15,.015),Vector3(.24,.16,.25),"purple","PARTS")
	_box(p+Vector3(.32,.44,-.09),Vector3(.20,.025,.14),"graphite",.006)
	_panel(p+Vector3(.32,.54,-.085),Vector2(.22,.17),"paper")
	_panel(p+Vector3(.32,.55,-.064),Vector2(.16,.105),"screen")
	_label("SYSTEM OK\n14 / 14",p+Vector3(.32,.55,-.042),27,.00035,"light")
	for x in [.265,.32,.375]:_cylinder(p+Vector3(x,.478,-.049),.008,.008,"yellow",Vector3.BACK)
	_box(p+Vector3(-.23,.435,.03),Vector3(.28,.008,.19),"purple",.003)
	_cylinder(p+Vector3(-.24,.459,.015),.052,.038,"metal")
	_cylinder(p+Vector3(-.24,.483,.015),.021,.014,"ink")
	for i in range(6):
		var a:=i*TAU/6
		_cylinder(p+Vector3(-.24+cos(a)*.039,.482,.015+sin(a)*.039),.004,.005,"graphite")
	_line(p+Vector3(-.44,.44,.10),p+Vector3(-.34,.44,.10),.005,"metal")
	_box(p+Vector3(-.455,.44,.10),Vector3(.045,.013,.022),"yellow",.004)
	_panel(p+Vector3(-.16,.615,-.29),Vector2(.64,.24),"graphite")
	for i in range(12):
		for j in range(4):_cylinder(p+Vector3(-.435+i*.05,.535+j*.048,-.27),.0024,.002,"metal",Vector3.BACK)
	for i in range(5):
		var x:=-.38+i*.105
		_line(p+Vector3(x,.54,-.252),p+Vector3(x,.66,-.252),.004,"metal")
		_box(p+Vector3(x,.555,-.247),Vector3(.016,.055,.012),"purple" if i%2 else "yellow",.004)
		_cylinder(p+Vector3(x,.666,-.25),.012,.012,"metal",Vector3.BACK)

func _crate(p: Vector3,size: Vector3,color: String,title: String) -> void:
	_box(p+Vector3(0,size.y*.5,0),size,color,.011)
	_box(p+Vector3(0,size.y-.022,0),Vector3(size.x+.006,.027,size.z+.006),color,.007)
	for x in [-1,1]:
		for z in [-1,1]:_box(p+Vector3(x*(size.x*.5-.012),size.y*.5,z*(size.z*.5-.012)),Vector3(.028,size.y+.006,.028),"graphite",.005)
	for x in [-.25,.25]:_box(p+Vector3(x*size.x,size.y-.026,size.z*.5+.005),Vector3(.024,.037,.012),"metal",.003)
	_box(p+Vector3(0,size.y*.55,size.z*.5+.005),Vector3(size.x*.35,.023,.012),"ink",.004)
	_label(title,p+Vector3(0,size.y*.25,size.z*.5+.007),27,.00035)
	# Rear hinges, recessed side grips and a serial plate remain readable on a walk-around.
	for x in [-.27,.27]:
		_cylinder(p+Vector3(x*size.x,size.y-.024,-size.z*.5-.004),.0045,size.x*.19,"metal",Vector3.RIGHT)
	_box(p+Vector3(0,size.y*.53,-size.z*.5-.003),Vector3(size.x*.33,.023,.008),"ink",.004)
	_label(title+" / MD",p+Vector3(0,size.y*.25,-size.z*.5-.006),25,.00034,"graphite",Vector3(0,180,0))
	for side in [-1,1]:
		_box(p+Vector3(side*(size.x*.5+.003),size.y*.53,0),Vector3(.007,.026,size.z*.35),"graphite",.004)
		_line(p+Vector3(side*(size.x*.5+.004),size.y-.038,-size.z*.37),p+Vector3(side*(size.x*.5+.004),size.y-.038,size.z*.37),.0007,"ink")
		for z in [-.27,.27]:
			_cylinder(p+Vector3(side*size.x*.34,size.y+.001,z*size.z),.0038,.002,"metal")

func _power_station(p: Vector3) -> void:
	_box(p+Vector3(0,.26,0),Vector3(.30,.51,.52),"graphite",.021)
	_panel(p+Vector3(0,.36,.274),Vector2(.265,.21),"purple")
	_panel(p+Vector3(0,.15,.274),Vector2(.265,.16),"paper")
	for i in range(7):_box(p+Vector3(0,.105+i*.014,.287),Vector3(.18,.004,.004),"ink",.0005)
	_label("POWER\n02",p+Vector3(-.06,.37,.297),27,.00044,"porcelain")
	_cylinder(p+Vector3(.07,.37,.30),.027,.013,"yellow",Vector3.BACK)
	_line(p+Vector3(.07,.37,.309),p+Vector3(.084,.383,.309),.0014)
	for i in range(2):
		var z:=-.10+i*.22
		_cylinder(p+Vector3(.08,.64,z),.066,.22,"paper")
		_cylinder(p+Vector3(.08,.53,z),.072,.025,"metal")
		_cylinder(p+Vector3(.08,.75,z),.04,.027,"graphite")
		_cylinder(p+Vector3(.08,.773,z),.019,.02,"yellow")
		_line(p+Vector3(.08,.782,z),p+Vector3(.22,.782,z),.013,"graphite")
	var previous:=p+Vector3(.14,.35,.2)
	for i in range(1,17):
		var t:=i/16.0
		var q:=p+Vector3(.14+.30*sin(t*PI),.35-.31*sin(t*PI*.5),.2+.30*t)
		_line(previous,q,.007,"rubber");previous=q
	_box(previous,Vector3(.030,.020,.065),"yellow",.005)

func _storage(p: Vector3) -> void:
	for x in [-.25,.25]:
		for z in [-.22,.22]:_box(p+Vector3(x,.35,z),Vector3(.027,.70,.027),"graphite",.004)
	for y in [.04,.29,.55,.72]:_box(p+Vector3(0,y,0),Vector3(.57,.025,.50),"metal",.005)
	_crate(p+Vector3(-.13,.06,.01),Vector3(.22,.19,.34),"paper","M-01")
	_crate(p+Vector3(.13,.06,.01),Vector3(.22,.19,.34),"yellow","M-02")
	_crate(p+Vector3(0,.31,.04),Vector3(.42,.17,.33),"purple","SPARES")
	for x in [-.16,-.04,.08,.20]:
		_cylinder(p+Vector3(x,.61,0),.034,.12,"porcelain")
		_cylinder(p+Vector3(x,.68,0),.023,.016,"graphite")
	_panel(p+Vector3(0,.76,.26),Vector2(.45,.08),"paper")
	_label("03 / STOCK",p+Vector3(0,.76,.28),30,.00045)

func _service_post(p: Vector3) -> void:
	_box(p+Vector3(0,.24,0),Vector3(.37,.46,.31),"graphite",.021)
	_panel(p+Vector3(0,.25,.167),Vector2(.32,.35),"porcelain")
	_panel(p+Vector3(0,.34,.185),Vector2(.22,.10),"screen")
	_label("CHARGE 98%",p+Vector3(0,.34,.209),25,.00037,"light")
	for x in [-.075,0,.075]:_cylinder(p+Vector3(x,.20,.198),.022,.013,"purple" if x==0 else "graphite",Vector3.BACK)
	_box(p+Vector3(0,.11,.20),Vector3(.18,.013,.008),"yellow",.002)
	_box(p+Vector3(0,.49,0),Vector3(.42,.045,.35),"purple",.012)
	_cylinder(p+Vector3(.13,.59,-.06),.017,.17,"graphite")
	_cylinder(p+Vector3(.13,.68,-.06),.026,.04,"yellow")
	_label("04",p+Vector3(-.07,.43,.20),36,.00055)

func _cargo_group() -> void:
	_crate(Vector3(-1.44,0,.64),Vector3(.40,.25,.34),"paper","MD / 071")
	_crate(Vector3(-1.40,.255,.64),Vector3(.33,.20,.29),"yellow","TOOLS")
	_crate(Vector3(2.70 if open_route else 1.48,0,.90 if open_route else .46),Vector3(.34,.30,.40),"purple","RETURN")
	# Leave the fixed corridor sightline clear of the foreground handle post.
	var cart_x: float=-1.00 if open_route else 1.46
	if VisualProfile.value("MD_CART_FRAME")=="1":
		_box(Vector3(cart_x,.08,.99),Vector3(.43,.030,.39),"graphite",.008)
	else:
		_box(Vector3(cart_x,.04,.99),Vector3(.43,.08,.39),"graphite",.010)
	for x in [cart_x-.17,cart_x+.17]:
		for z in [.84,1.13]:_cylinder(Vector3(x,.045,z),.045,.023,"rubber",Vector3.RIGHT)
	_box(Vector3(cart_x,.105,.99),Vector3(.41,.04,.37),"metal",.010)
	_crate(Vector3(cart_x,.13,.99),Vector3(.34,.16,.29),"paper","07")
	for x in [cart_x-.16,cart_x+.16]:_line(Vector3(x,.10,.83),Vector3(x,.42,.79),.013,"graphite")
	_line(Vector3(cart_x-.16,.42,.79),Vector3(cart_x+.16,.42,.79),.016,"purple")

func _small_details() -> void:
	for x in [-1.60,1.60]:
		for z in [-1.3,-.1,.9]:
			_box(Vector3(x,.002,z),Vector3(.10,.004,.11),"graphite",.004)
			_cylinder(Vector3(x,.005,z),.011,.002,"metal")
	for i in range(4):_box(Vector3(-.8+i*.10,.001,-.74),Vector3(.045,.001,.055),"yellow",.0002,.35)
	_line(Vector3(-1.04,.04,-.64),Vector3(-1.04,.57,-.64),.012,"graphite")
	_line(Vector3(-1.04,.57,-.64),Vector3(-.91,.69,-.64),.014,"metal")
	_box(Vector3(-.88,.68,-.64),Vector3(.13,.035,.09),"yellow",.008)
	_box(Vector3(-.88,.659,-.64),Vector3(.10,.008,.068),"light",.003)
	_cylinder(Vector3(-1.04,.014,-.64),.075,.025,"graphite")
	_label("KEEP CLEAR",Vector3(.84,.002,.86),35,.0006,"shadow",Vector3(-90,0,0))

func _commit_geometry() -> void:
	for key in builders:
		var mesh: ArrayMesh=builders[key].commit()
		if mesh==null or mesh.get_surface_count()==0:continue
		mesh.surface_set_material(0,materials[key])
		var instance:=MeshInstance3D.new();instance.name="Workshop_"+key;instance.mesh=mesh
		add_child(instance)
	builders.clear()

func _paint_robot(node: Node) -> void:
	if node is MeshInstance3D:
		var mi:=node as MeshInstance3D
		var role: String=server._paint_role_for_node(mi)
		var key: String={"shell":"graphite","trim":"yellow","accent":"purple","mech":"rubber"}.get(role,"graphite")
		var geom_id: String = str(server._mesh_id(mi))
		var part: String = mesh_roles.get(geom_id, "")
		if normal_meshes.has(geom_id) and ResourceLoader.exists(normal_meshes[geom_id]):mi.mesh=load(normal_meshes[geom_id])
		if part in ["jaw", "soft_mouth_top"]: key="yellow"
		elif part in ["foot_left", "foot_right", "noenoeil"]: key="purple"
		elif part.begins_with("seeed_bearing") or part=="upper_leg_rigidity_plate": key="metal"
		elif part in ["xl330", "np_f970", "lens", "m12_lens_holder", "speaker", "sole_left", "sole_right", "tire"] or "pcb" in part: key="rubber"
		elif str(mi.name).begins_with("vis_ball_"): key="yellow"
		else: key="graphite"
		var mat: ShaderMaterial=materials[key].duplicate()
		if part!="lens" and not VisualProfile.value("MD_ROBOT_HATCH").is_empty():
			mat.set_shader_parameter("hatch_strength",float(VisualProfile.value("MD_ROBOT_HATCH")))
		if part=="lens":
			mat.shader=load("res://atelier/lens.gdshader")
			mat.set_shader_parameter("diagnostic_surface",OS.get_environment("MD_LENS_DEBUG")=="1")
			if VisualProfile.value("MD_LENS_COATING")=="1":
				mat.set_shader_parameter("aperture_radius",.0045)
				mat.set_shader_parameter("reflection_strength",.65)
		if part=="lens" and VisualProfile.value("MD_LENS_INK")=="0":mat.next_pass=null
		if mat.next_pass != null:
			var ink: ShaderMaterial=mat.next_pass.duplicate()
			ink.set_shader_parameter("line_pixels",.60);mat.next_pass=ink
		for i in range(mi.mesh.get_surface_count()):mi.set_surface_override_material(i,mat)
	for child in node.get_children():_paint_robot(child)

func _install_hud() -> void:
	if server._hud!=null:server._hud.visible=false
	hud=load("res://atelier/atelier_hud.gd").new();hud.workshop=self;add_child(hud)

func set_view(which: String) -> void:
	capture_camera.clear()
	previous_view=view
	view=which
	if hud!=null:hud.visible=which=="follow"
	if which=="follow":follow_initialized=false
	update_camera(0.0)

func reset_camera() -> void:
	capture_camera.clear()
	follow_initialized=false
	server._cam_yaw=.65
	server._cam_pitch=.32
	server._cam_dist=1.10
	server._cam_snap=false

func _follow(dt: float) -> void:
	var base: RigidBody3D=server._base
	if base==null:return
	var target:=base.global_position+Vector3(0,.09,0)
	var d:=clampf(dt,0.0,.08)
	if not follow_initialized:
		follow_look=target
		camera.position=target+server._orbit_offset(server._cam_yaw,server._cam_pitch,server._cam_dist)
		follow_initialized=true
	follow_look.x=lerpf(follow_look.x,target.x,1-exp(-5*d))
	follow_look.z=lerpf(follow_look.z,target.z,1-exp(-5*d))
	follow_look.y=lerpf(follow_look.y,target.y,1-exp(-2*d))
	server._cam_look=follow_look
	var desired: Vector3=follow_look+server._orbit_offset(server._cam_yaw,server._cam_pitch,server._cam_dist)
	desired=_unobstructed_position(follow_look,desired)
	var smoothed:=camera.position.lerp(desired,1-exp(-8*d))
	if VisualProfile.value("MD_CAMERA_RELEASE")=="slow":
		var current_radius:=camera.position.distance_to(follow_look)
		var proposed_radius:=smoothed.distance_to(follow_look)
		if proposed_radius>current_radius:
			var released_radius:=lerpf(current_radius,maxf(current_radius,desired.distance_to(follow_look)),1-exp(-3.0*d))
			smoothed=follow_look+(smoothed-follow_look).normalized()*minf(proposed_radius,released_radius)
	# Pull inward immediately when occluded; ease outward once the route clears.
	camera.position=_unobstructed_position(follow_look,smoothed)
	camera.look_at(follow_look)
	if VisualProfile.value("MD_OCCLUSION_FOV")=="1":
		var framing_fov:=48.0
		if desired.distance_to(follow_look)<server._cam_dist*.98:
			var framing_ratio: float=server._cam_dist/maxf(.1,camera.position.distance_to(follow_look))
			framing_fov=clampf(rad_to_deg(2.0*atan(tan(deg_to_rad(24.0))*framing_ratio)),48.0,78.0)
		# Widen with the inward collision correction, then return gently.
		camera.fov=framing_fov if framing_fov>=camera.fov else lerpf(camera.fov,framing_fov,1-exp(-3*d))
	else:camera.fov=48.0

func _unobstructed_position(target: Vector3,wanted: Vector3) -> Vector3:
	var nearest:=wanted.distance_to(target)
	for raw in camera_obstacles:
		var obstacle:=raw.grow(.025)
		if obstacle.has_point(target):continue
		var hit: Variant=obstacle.intersects_segment(target,wanted)
		if hit is Vector3:nearest=minf(nearest,maxf(.025,target.distance_to(hit)-.025))
	return target+(wanted-target).normalized()*nearest

func update_camera(dt: float) -> bool:
	if camera==null:return false
	for rotor in rotors:rotor.rotation.x=server._t*.8
	if view=="showcase":
		_showcase_camera(dt)
		return true
	if not capture_camera.is_empty():
		var p: Array=capture_camera.position
		var b: Array=capture_camera.basis
		camera.global_transform=Transform3D(Basis(Vector3(b[0][0],b[0][1],b[0][2]),Vector3(b[1][0],b[1][1],b[1][2]),Vector3(b[2][0],b[2][1],b[2][2])),Vector3(p[0],p[1],p[2]))
		camera.fov=float(capture_camera.fov)
		camera.near=float(capture_camera.near);camera.far=float(capture_camera.far)
		return true
	if view=="follow":
		_follow(dt)
		return true
	if view=="tour":
		tour_time+=dt
		var a:=.6+sin(tour_time*.11)*.5
		camera.position=Vector3(sin(a)*3.2,1.35+sin(tour_time*.07)*.22,cos(a)*3.2)
		camera.look_at(Vector3(0,.24,-.30));camera.fov=47;return true
	if VIEWS.has(view):
		var config: Array=VIEWS[view]
		camera.position=config[0];camera.look_at(config[1]);camera.fov=config[2];return true
	return false

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.physical_keycode==KEY_B and loose_props!=null:
			loose_props.cycle_target();get_viewport().set_input_as_handled()
		if event.physical_keycode==KEY_TAB:
			set_view("tour" if view=="follow" else "follow");get_viewport().set_input_as_handled()
		if event.physical_keycode==KEY_F1 and hud!=null:
			hud.toggle_help();get_viewport().set_input_as_handled()
		if event.physical_keycode==KEY_F2 and hud!=null:
			hud.toggle_debug();get_viewport().set_input_as_handled()

func _showcase_camera(dt: float) -> void:
	if server._base==null:return
	var shot:=OS.get_environment("MD_SHOWCASE_SHOT")
	var target: Vector3=server._base.global_position
	target.y=.15
	var elapsed: float=maxf(0.0,server._t)
	var yaw:=.70
	var pitch:=.24
	var distance:=.95
	if shot=="wide":
		yaw=.48+minf(elapsed,14.0)*.018;pitch=.46;distance=1.55
	elif shot=="detail":
		yaw=.85-minf(elapsed,14.0)*.020;pitch=.15;distance=.72
	elif shot=="action":
		yaw=.40;pitch=.25;distance=1.25
	else:
		yaw=.68+minf(elapsed,14.0)*.012
	if not follow_initialized:
		follow_look=target;follow_initialized=true
	follow_look=follow_look.lerp(target,1.0-exp(-6.0*maxf(dt,0.0)))
	var position_wanted: Vector3=follow_look+server._orbit_offset(yaw,pitch,distance)
	camera.position=_unobstructed_position(follow_look,position_wanted)
	camera.look_at(follow_look)
	camera.fov=46.0
