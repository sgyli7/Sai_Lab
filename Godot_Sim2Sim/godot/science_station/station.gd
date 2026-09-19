extends "res://atelier/workshop.gd"
const Layout=preload("res://science_station/layout.gd")
var towers: Array[Dictionary]=[]
var landscape_builder:RefCounted

func _init() -> void:
	label_font=load("res://atelier/ui_font.tres")
	palette.merge({"blue":Color("6796ae"),"sand":Color("ffb46b"),"rock":Color("a6b1c2"),"distant":Color("8faec6"),
		"paper":Color("d7d8ce"),"porcelain":Color("e9e8da"),"graphite":Color("464e58"),
		"metal":Color("889999"),"glass":Color("263f51"),"signal":Color("c57d52"),
		"silt":Color("7f899f"),"chalk":Color("c8cad4"),"ochre":Color("ed9c62"),"far_rock":Color("a0a9c2"),"strata":Color("576477")})

func ground_height(x:float,z:float) -> float:
	if absf(x)<=24. and z>=-26. and z<=14.:return Layout.height_at(x,z)
	return landscape_builder.ground(x,z) if landscape_builder!=null else 0.

func _build_details() -> void:
	for key in materials:
		if materials[key].next_pass != null:
			materials[key].next_pass.set_shader_parameter("fade_start",35.)
			materials[key].next_pass.set_shader_parameter("fade_end",150.)

func _environment() -> void:
	super._environment()
	RenderingServer.directional_shadow_atlas_set_size(8192,true)
	RenderingServer.directional_soft_shadow_filter_set_quality(RenderingServer.SHADOW_QUALITY_SOFT_HIGH)
	var env:Environment=server.get_node("WorldEnvironment").environment
	var sky:=Sky.new();var sky_mat:=ShaderMaterial.new()
	sky_mat.shader=load("res://science_station/sky.gdshader");sky.sky_material=sky_mat
	env.sky=sky;env.background_mode=Environment.BG_SKY
	env.ambient_light_color=Color("c6c3d6");env.ambient_light_energy=.48
	env.fog_light_color=Color("bdcbd0");env.fog_density=.0025;env.fog_sky_affect=0.
	var sun:DirectionalLight3D=server.get_node("World/Sun")
	sun.rotation_degrees=Vector3(-48,-36,0);sun.light_energy=.96
	sun.shadow_bias=.02;sun.shadow_normal_bias=1.
	sun.directional_shadow_mode=DirectionalLight3D.SHADOW_PARALLEL_4_SPLITS
	sun.directional_shadow_max_distance=55.
	sun.directional_shadow_split_1=.035;sun.directional_shadow_split_2=.12;sun.directional_shadow_split_3=.4
	camera.far=500.;camera.near=.015

func _floor() -> void:
	server.get_node("World/Floor/FloorMesh").hide()
	server.get_node("World/Floor/CollisionShape3D").disabled=true

func _build_solids() -> void:
	# Also configure the floor in headless mode, where the material hook is skipped.
	server.get_node("World/Floor/FloorMesh").hide()
	server.get_node("World/Floor/CollisionShape3D").disabled=true
	_terrain()
	solid_scope=true
	_service();_samples();_tower(Vector3(7,0,-17),7.,1.32,"01");_tower(Vector3(16,0,-21),10.,1.7,"02")
	load("res://science_station/command_station.gd").new().build(self)
	_field_details()
	solid_scope=false
	_landscape()
	loose_props=load("res://atelier/loose_props.gd").new();loose_props.name="LooseProps";add_child(loose_props)
	loose_props.build(self,visuals_enabled,.015,Layout.PROPS)

func _terrain() -> void:
	var material:ShaderMaterial
	if visuals_enabled:
		material=ShaderMaterial.new();material.shader=load("res://science_station/ground.gdshader")
	for ix in range(6):
		for iz in range(5):
			var st:=SurfaceTool.new();st.begin(Mesh.PRIMITIVE_TRIANGLES)
			for i in range(32):
				for j in range(32):
					var x:float=-24.+ix*8.+i*.25;var z:float=-26.+iz*8.+j*.25
					var a:=Vector3(x,ground_height(x,z),z);var b:=Vector3(x+.25,ground_height(x+.25,z),z)
					var c:=Vector3(x,ground_height(x,z+.25),z+.25);var d:=Vector3(x+.25,ground_height(x+.25,z+.25),z+.25)
					for point in [a,b,c,b,d,c]:
						if visuals_enabled:st.set_normal(Layout.surface_normal(point.x,point.z))
						st.add_vertex(point)
			st.index();var mesh:=st.commit()
			var patch:=StaticBody3D.new();patch.name="Terrain_%d_%d"%[ix,iz]
			patch.add_to_group("sai_driving_surface")
			patch.collision_layer=3;patch.collision_mask=5
			patch.physics_material_override=server.get_node("World/Floor").physics_material_override
			var shape:=ConcavePolygonShape3D.new();shape.set_faces(mesh.get_faces());shape.backface_collision=true
			var collider:=CollisionShape3D.new();collider.shape=shape;patch.add_child(collider);add_child(patch)
			if visuals_enabled:
				var visual:=MeshInstance3D.new();visual.mesh=mesh;visual.material_override=material;patch.add_child(visual)

func _service() -> void:
	load("res://science_station/service_module.gd").new().build(self)

func _samples() -> void:
	load("res://science_station/laboratory.gd").new().build(self,Vector3(-7.5,0,-.5))

func _tower(p:Vector3,h:float,r:float,id:String) -> void:
	towers.append({"position":p,"height":h,"radius":r})
	# Four independent splayed supports leave the space underneath traversable.
	for i in range(4):
		var a:float=PI*.25+i*PI*.5
		var foot:=p+Vector3(cos(a)*(r+.65),.10,sin(a)*(r+.65))
		var neck:=p+Vector3(cos(a)*r*.72,1.65,sin(a)*r*.72)
		_box(foot,Vector3(.9,.20,.72),"graphite",.025,a)
		_beam(foot+Vector3(0,.1,0),neck,.22,.32,"metal")
		_beam(foot+Vector3(0,.2,0),neck+Vector3(.05,0,.05),.085,.37,"blue")
		_cylinder(neck,.12,.18,"yellow",Vector3.RIGHT)
	_cylinder(p+Vector3(0,1.57,0),r*.72,.32,"graphite")
	_ring(p+Vector3(0,1.72,0),r*.88,.055,"metal")
	# Tapered enclosure assembled from deliberate bands and narrow recessed joints.
	var bottom:=1.8;var top:float=h-.65
	for j in range(4):
		var y0:float=lerpf(bottom,top,j/4.);var y1:float=lerpf(bottom,top,(j+1)/4.)
		var r0:float=lerpf(r,r*.87,j/4.);var r1:float=lerpf(r,r*.87,(j+1)/4.)
		_frustum(p+Vector3(0,(y0+y1)*.5,0),r0,r1,y1-y0-.025,"paper")
		_ring(p+Vector3(0,y0+.012,0),r0+.008,.013,"graphite")
		if j==1:_ring(p+Vector3(0,y0+.06,0),r0+.012,.032,"blue")
	_dome(p+Vector3(0,top,0),r*.87,.40,"paper")
	for i in range(6):
		var a:float=i*TAU/6.
		var start:=p+Vector3(sin(a)*(r+.012),bottom+.15,cos(a)*(r+.012))
		var end:=p+Vector3(sin(a)*(r*.88+.01),top-.1,cos(a)*(r*.88+.01))
		_line(start,end,.0035,"strata")
		# Sparse fasteners, paired service panels and repairs, not evenly dense greebles.
		if i%3==0:
			for j in range(3):
				var t:float=.23+j*.18;var q:Vector3=start.lerp(end,t)
				_box(q,Vector3(.20,.11,.025),"blue" if j==1 else "metal",.006,-a)
	for i in range(8):
		var a:float=i*TAU/8.
		var q:=p+Vector3(sin(a)*r*.70,1.45,cos(a)*r*.70)
		_box(q,Vector3(.18,.30,.22),"graphite",.015,a)
		_line(q,q+Vector3(sin(a)*.22,-.31,cos(a)*.22),.035,"metal")
		_cylinder(q+Vector3(0,-.13,0),.08,.12,"yellow")
	for i in range(3):
		var q:=p+Vector3((i-1)*r*.43,top+.35,0)
		_cylinder(q+Vector3(0,.15,0),.032,.30,"graphite")
		_line(q,q+Vector3(0,1.1+float(i%2)*1.45,0),.012,"graphite")
	var panel_y:float=bottom+1.0
	var panel_r:float=lerpf(r,r*.87,(panel_y-bottom)/(top-bottom))
	_box(p+Vector3(0,panel_y,panel_r+.02),Vector3(.90,.38,.18),"paper",.022)
	_label("W / "+id,p+Vector3(0,panel_y,panel_r+.115),64,.0032,"blue")
	_cabinet(p+Vector3(-r-.55,0,1.6),"blue","OBS / "+id)
	# A supported service platform above the robot route.
	_box(p+Vector3(0,1.72,r+.2),Vector3(r*1.6,.09,.75),"metal",.015)
	for x in [-r*.7,r*.7]:
		_line(p+Vector3(x,1.76,r+.52),p+Vector3(x,2.28,r+.52),.016,"graphite")
	_line(p+Vector3(-r*.7,2.28,r+.52),p+Vector3(r*.7,2.28,r+.52),.018,"graphite")
	_box(p+Vector3(.24,1.72,-r-.25),Vector3(.85,.07,.65),"metal",.014)
	for j in range(9):_line(p+Vector3(.05,.16+j*.18,-r-.30),p+Vector3(.43,.16+j*.18,-r-.30),.014,"metal")
	for x in [.05,.43]:_line(p+Vector3(x,.04,-r-.30),p+Vector3(x,1.9,-r-.30),.021,"graphite")
	load("res://science_station/tower_details.gd").new().build(self,p,h,r,id)

func _field_details() -> void:
	_sign(Vector3(7.2,0,5.8),"设备泊位 / 20 × 12 m\nEQUIPMENT / 02",.85)
	_cabinet(Vector3(2.5,0,-.8),"blue","POWER / 02")
	_cabinet(Vector3(-1.6,0,-4.),"yellow","ATMOSPHERE")
	_cabinet(Vector3(-3.4,0,-10.),"blue","METEO / 08")
	for p in [Vector3(-10.9,0,3.5),Vector3(1.8,0,-8.5),Vector3(-10.5,0,-7.5),Vector3(4,0,-13.5)]:_lamp(p,2.2)
	_sign(Vector3(-11.5,0,-2),"岩丘步道\nFIELD LOOP",.6)
	for p in [Vector3(-2.8,0,-1),Vector3(-4,0,-6),Vector3(1.9,0,-12)]:
		_cylinder(p+Vector3(0,.19,0),.065,.38,"metal")
		_cylinder(p+Vector3(0,.40,0),.085,.065,"yellow")
	# Low field relay with a visible dish and a small shaded rest spot.
	_box(Vector3(-13,.65,-12),Vector3(1.45,1.3,1.2),"paper",.08)
	_box(Vector3(-13,1.34,-12),Vector3(1.6,.06,1.35),"purple",.018)
	_dish(Vector3(-13,1.9,-12),.48)
	load("res://science_station/station_details.gd").new().relay(self,Vector3(-13,0,-12))
	_label("02",Vector3(17,.015,-1.8),120,.023,"blue",Vector3(-90,0,0))
	_cabinet(Vector3(23.55,0,-6),"blue","SUPPLY")
	for p in [Vector3(-12.5,0,5),Vector3(-16,0,-16),Vector3(2,0,-22)]:
		p.y=ground_height(p.x,p.z)
		_box(p+Vector3(0,.26,0),Vector3(.9,.06,.35),"paper",.012)
		for x in [-.35,.35]:_box(p+Vector3(x,.12,0),Vector3(.065,.24,.28),"graphite",.008)

func _landscape() -> void:
	landscape_builder=load("res://science_station/landscape.gd").new()
	landscape_builder.build(self)

func _cabinet(p:Vector3,color:String,label_text:String) -> void:
	_box(p+Vector3(0,.34,0),Vector3(.45,.68,.32),"graphite",.02)
	_panel(p+Vector3(0,.37,.173),Vector2(.39,.53),color)
	for i in range(4):_box(p+Vector3(0,.20+i*.045,.187),Vector3(.25,.012,.01),"ink",.002)
	_label(label_text,p+Vector3(0,.52,.189),28,.0007,"ink")
	_box(p+Vector3(0,.70,0),Vector3(.49,.04,.36),"paper",.012)

func _lamp(p:Vector3,h:float) -> void:
	_box(p+Vector3(0,.10,0),Vector3(.18,.20,.18),"graphite",.02)
	_box(p+Vector3(0,h*.5,0),Vector3(.065,h,.08),"metal",.015)
	_box(p+Vector3(0,h-.10,0),Vector3(.14,.42,.15),"paper",.035)
	_box(p+Vector3(0,h-.09,.09),Vector3(.085,.23,.025),"blue",.014)
	_box(p+Vector3(.15,h+.12,0),Vector3(.53,.10,.24),"porcelain",.04)
	_box(p+Vector3(.19,h+.061,0),Vector3(.22,.015,.14),"light",.006)

func _sign(p:Vector3,text:String,h:float) -> void:
	for x in [-.27,.27]:_cylinder(p+Vector3(x,h*.45,0),.015,h*.9,"graphite")
	_box(p+Vector3(0,h,0),Vector3(.78,.31,.045),"paper",.014)
	_label(text,p+Vector3(0,h,.024),32,.0012,"ink")

func _beam(a:Vector3,b:Vector3,width:float,depth:float,color:String) -> void:
	var mesh:Mesh=_rounded_box(Vector3(width,a.distance_to(b),depth),.015)
	var basis:=Basis(Quaternion(Vector3.UP,(b-a).normalized()))
	_add(mesh,(a+b)*.5,color,basis);_solid(mesh,(a+b)*.5,basis)
	camera_obstacles.append(Transform3D(basis,(a+b)*.5)*mesh.get_aabb())

func _frustum(p:Vector3,r0:float,r1:float,h:float,color:String) -> void:
	var mesh:=CylinderMesh.new();mesh.bottom_radius=r0;mesh.top_radius=r1;mesh.height=h;mesh.radial_segments=32;mesh.rings=1
	_add(mesh,p,color);_solid(mesh,p);camera_obstacles.append(AABB(p-Vector3(maxf(r0,r1),h*.5,maxf(r0,r1)),Vector3(maxf(r0,r1)*2,h,maxf(r0,r1)*2)))

func _dome(p:Vector3,r:float,h:float,color:String) -> void:
	# A Godot hemisphere's height is its full cap height, not its diameter.
	var mesh:=SphereMesh.new();mesh.radius=r;mesh.height=r;mesh.is_hemisphere=true;mesh.radial_segments=32;mesh.rings=12
	var basis:=Basis.from_scale(Vector3(1,h/r,1))
	_add(mesh,p,color,basis);_solid(mesh,p,basis)
	camera_obstacles.append(Transform3D(basis,p)*mesh.get_aabb())

func _ring(p:Vector3,r:float,thickness:float,color:String) -> void:
	for i in range(48):
		var a:float=i*TAU/48.;var b:float=(i+1)*TAU/48.
		_line(p+Vector3(sin(a)*r,0,cos(a)*r),p+Vector3(sin(b)*r,0,cos(b)*r),thickness,color)

func _dish(p:Vector3,r:float) -> void:
	_cylinder(p-Vector3(0,.25,0),.045,.5,"graphite")
	var front:=SurfaceTool.new();front.begin(Mesh.PRIMITIVE_TRIANGLES)
	var back:=SurfaceTool.new();back.begin(Mesh.PRIMITIVE_TRIANGLES)
	var tilt:=Basis(Vector3.RIGHT,.70)
	for j in range(5):
		for i in range(32):
			var a:float=i*TAU/32.;var b:float=(i+1)*TAU/32.
			var u:float=j*r/5.;var v:float=(j+1)*r/5.
			var q:=Vector3(sin(a)*u,.27*u*u/r,cos(a)*u)
			var s:=Vector3(sin(b)*u,.27*u*u/r,cos(b)*u)
			var t:=Vector3(sin(a)*v,.27*v*v/r,cos(a)*v)
			var e:=Vector3(sin(b)*v,.27*v*v/r,cos(b)*v)
			for point in [q,s,t,s,e,t]:front.add_vertex(point)
			for point in [q,t,s,s,t,e]:back.add_vertex(point-Vector3(0,.018,0))
	front.generate_normals();front.index();back.generate_normals();back.index()
	_add(front.commit(),p,"blue",tilt);_add(back.commit(),p,"paper",tilt)
	for i in range(32):
		var a:float=i*TAU/32.;var b:float=(i+1)*TAU/32.
		_line(p+tilt*Vector3(sin(a)*r,.27*r,cos(a)*r),p+tilt*Vector3(sin(b)*r,.27*r,cos(b)*r),.012,"paper")
	var focus:=p+tilt*Vector3(0,r*.95,0)
	for i in range(3):
		var a:float=i*TAU/3.
		_line(p+tilt*Vector3(sin(a)*r*.9,.25*r,cos(a)*r*.9),focus,.008,"graphite")
	_cylinder(focus,.045,.08,"paper",tilt*Vector3.UP)

func _unobstructed_position(target:Vector3,wanted:Vector3) -> Vector3:
	var clear:Vector3=super._unobstructed_position(target,wanted)
	# The same triangulated ground height used by physics also protects the
	# entire camera segment, including a crest between the target and camera.
	var steps:int=maxi(1,ceili(target.distance_to(clear)/.04))
	for i in range(1,steps+1):
		var p:Vector3=target.lerp(clear,float(i)/steps)
		if p.y<ground_height(p.x,p.z)+.025:
			return target.lerp(clear,float(maxi(0,i-1))/steps)
	return clear

func update_camera(dt:float) -> bool:
	if camera==null:return false
	if Layout.VIEWS.has(view) or view=="tour":
		var shot:Array=Layout.VIEWS["overview" if view=="tour" else view]
		camera.position=shot[0];camera.look_at(shot[1]);camera.fov=shot[2];return true
	var result:bool=super.update_camera(dt)
	if view=="follow":
		var floor_y:float=ground_height(camera.position.x,camera.position.z)+.06
		if camera.position.y<floor_y:camera.position.y=floor_y;camera.look_at(follow_look)
	return result

func cycle_view() -> void:
	var views:Array=["follow"]+Layout.VIEWS.keys()
	set_view(views[(views.find(view)+1)%views.size()])
