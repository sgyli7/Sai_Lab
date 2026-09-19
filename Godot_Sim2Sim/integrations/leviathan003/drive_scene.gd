extends Node3D
const REVIEW_VIEWS=["whole","rear","track","bridge","crane","panel_back","house_port","stairs","house_starboard","panel_front","reservoir_clamps","reservoir_cradle","frame_marks","bridge_port"]
var runtime:Node3D
var camera:Camera3D
var label:Label
var samples:Array=[]
var frames:Array[float]=[]
var started:int
var sim_start:=0.
var next_sample:=0.
var stop_time:=0.
var mode:="manual"
var output:=""
var yaw:=.92
var pitch:=.28
var distance:=100.
var dragging:=false
var screenshot_done:=false
var previous_frame_usec:=0
var view_mode:=0
var detail_captures:Dictionary={}
var auto_speed:=27.7777777778
var ground:Node3D
var server:Node3D
var visuals_enabled:=true
var terrain_name:="flat"

func _ready()->void:
	var opt:Dictionary=JSON.parse_string(FileAccess.get_file_as_string("res://leviathan003/options.json"))
	terrain_name=opt.get("terrain","flat");server=self;visuals_enabled=DisplayServer.get_name()!="headless"
	mode=opt.get("mode","manual");stop_time=opt.get("seconds",0.);output=opt.get("output","")
	get_window().title="Robot_Godot_Sim2Sim · 利维坦 003"
	get_window().size=Vector2i(1920,1080)
	get_window().focus_exited.connect(func():if runtime!=null:runtime.set_vehicle_command(0,0))
	var env:=Environment.new();env.background_mode=Environment.BG_COLOR;env.background_color=Color("98adbd")
	env.ambient_light_source=Environment.AMBIENT_SOURCE_COLOR;env.ambient_light_color=Color("c1d0d8");env.ambient_light_energy=.32
	env.tonemap_mode=Environment.TONE_MAPPER_FILMIC
	env.ssao_enabled=true;env.ssao_radius=1.5;env.ssao_intensity=1.3
	get_viewport().msaa_3d=Viewport.MSAA_4X
	var worldenv:=WorldEnvironment.new();worldenv.environment=env;add_child(worldenv)
	var sun:=DirectionalLight3D.new();sun.rotation_degrees=Vector3(-36,-32,0);sun.light_energy=.7;sun.shadow_enabled=true;sun.directional_shadow_max_distance=240;add_child(sun)
	ground=Node3D.new();ground.name="World";add_child(ground)
	var floor:=StaticBody3D.new();floor.name="Floor";floor.physics_material_override=PhysicsMaterial.new();floor.collision_layer=2;floor.collision_mask=1|4|16;ground.add_child(floor)
	var plane:=WorldBoundaryShape3D.new();var collision:=CollisionShape3D.new();collision.shape=plane;floor.add_child(collision)
	var mesh:=PlaneMesh.new();mesh.size=Vector2(30000,30000)
	var mat:=StandardMaterial3D.new();mat.albedo_color=Color("b6c3c8");mat.roughness=.95
	var vis:=MeshInstance3D.new();vis.mesh=mesh;vis.material_override=mat;ground.add_child(vis)
	# Sparse metre-scale markers make measured motion visible without a heavy terrain mesh.
	var stripe:=BoxMesh.new();stripe.size=Vector3(.35,.015,2)
	var stripe_mat:=StandardMaterial3D.new();stripe_mat.albedo_color=Color("768e95");stripe.material=stripe_mat
	var mm:=MultiMesh.new();mm.transform_format=MultiMesh.TRANSFORM_3D;mm.mesh=stripe;mm.instance_count=600
	for i in 600:mm.set_instance_transform(i,Transform3D(Basis.IDENTITY,Vector3(i*10-500,.015,22 if i%2 else -22)))
	var mmi:=MultiMeshInstance3D.new();mmi.multimesh=mm;ground.add_child(mmi)
	var placement:=Transform3D.IDENTITY
	if terrain_name=="polar":
		collision.disabled=true;vis.hide();mmi.hide()
		var landscape:RefCounted=load("res://leviathan003/terrain.gd").new();landscape.build(self)
		placement.origin=Vector3(134,3,0)
		if visuals_enabled:
			var sky:=Sky.new();var sm:=ShaderMaterial.new();sm.shader=load("res://polar_range/sky.gdshader");sky.sky_material=sm;env.sky=sky;env.background_mode=Environment.BG_SKY
	runtime=load("res://leviathan003/runtime.gd").new();add_child(runtime)
	if not runtime.load_bundle("res://leviathan003",placement,DisplayServer.get_name()!="headless"):
		push_error("003 asset load failed");get_tree().quit(2);return
	camera=Camera3D.new();camera.near=.25;camera.far=20000;camera.fov=52;camera.current=true;add_child(camera)
	var ui:=CanvasLayer.new();add_child(ui);label=Label.new();label.position=Vector2(24,22);label.add_theme_font_size_override("font_size",22);label.add_theme_color_override("font_color",Color("203940"));ui.add_child(label)
	var fontpath:="res://atelier/ui_font.tres"
	var font:=FontFile.new()
	if font.load_dynamic_font(ProjectSettings.globalize_path("res://atelier/fonts/NotoSansCJK-Regular.ttc"))==OK:
		var variation:=FontVariation.new();variation.base_font=font;variation.variation_face_index=2;label.add_theme_font_override("font",variation)
	started=Time.get_ticks_usec();sim_start=runtime.elapsed

func _unhandled_input(e:InputEvent)->void:
	if e is InputEventMouseButton:
		if e.button_index==MOUSE_BUTTON_RIGHT:dragging=e.pressed
		if e.button_index==MOUSE_BUTTON_WHEEL_UP:distance=maxf(8,distance*.9)
		if e.button_index==MOUSE_BUTTON_WHEEL_DOWN:distance=minf(300,distance/ .9)
	if e is InputEventMouseMotion and dragging:yaw-=e.relative.x*.004;pitch=clampf(pitch+e.relative.y*.004,.04,1.3)
	if e is InputEventKey and e.pressed and not e.echo:
		if e.physical_keycode==KEY_ESCAPE:runtime.set_vehicle_command(0,0);_finish()
		if e.physical_keycode==KEY_TAB:view_mode=(view_mode+1)%REVIEW_VIEWS.size()
		if e.physical_keycode==KEY_F12:_capture("manual")

func _physics_process(_dt:float)->void:
	if runtime==null:return
	var t:float=runtime.elapsed
	var speed:=0.;var turn:=0.
	if mode=="straight":speed=auto_speed
	elif mode=="turn":
		speed=10. if t<60 else 0.;turn=.018 if t>25 and t<60 else 0.
	elif mode=="manual" and get_window().has_focus():
		var forward:float=float(Input.is_physical_key_pressed(KEY_W) or Input.is_physical_key_pressed(KEY_UP))-float(Input.is_physical_key_pressed(KEY_S) or Input.is_physical_key_pressed(KEY_DOWN))
		var left:float=float(Input.is_physical_key_pressed(KEY_A) or Input.is_physical_key_pressed(KEY_LEFT))-float(Input.is_physical_key_pressed(KEY_D) or Input.is_physical_key_pressed(KEY_RIGHT))
		speed=forward*(auto_speed if Input.is_physical_key_pressed(KEY_SHIFT) else 10.);turn=left*.035 if forward!=0 else 0.
		if Input.is_physical_key_pressed(KEY_SPACE):speed=0.;turn=0.
	runtime.set_vehicle_command(speed,turn)
	if t>=next_sample:
		next_sample=t+.5;var s:Dictionary=runtime.state();s.wall_seconds=float(Time.get_ticks_usec()-started)/1e6;s.fps=Engine.get_frames_per_second();samples.append(s)
		if samples.size()%20==0:print(JSON.stringify(s))
	if stop_time>0 and t>=stop_time:_finish()

func _process(dt:float)->void:
	if runtime==null:return
	var wall_now:=Time.get_ticks_usec()
	if runtime.elapsed>2 and previous_frame_usec>0:frames.append(float(wall_now-previous_frame_usec)/1e6)
	previous_frame_usec=wall_now
	var target:Vector3=(runtime.bodies.front.position+runtime.bodies.rear.position)*.5+Vector3(0,2,0)
	var eye_offset:=Vector3(cos(yaw)*cos(pitch),sin(pitch),sin(yaw)*cos(pitch))*distance
	if view_mode==1:eye_offset=Vector3(-75,42,-90)
	elif view_mode==2:
		target=runtime.bodies.front.global_transform*Vector3(10.5,-7,11.8);eye_offset=Vector3(9,6,12)
	elif view_mode==3:
		target=runtime.bodies.front.global_transform*Vector3(24,2,0);eye_offset=Vector3(13,6,14)
	if view_mode==4:
		target=runtime.bodies.rear.global_transform*Vector3(-14.7,6,5.5);eye_offset=Vector3(-13,7,15)
	elif view_mode==5:
		target=runtime.bodies.front.global_transform*Vector3(1,14,0);eye_offset=Vector3(-15,9,-18)
	elif view_mode==6:
		target=runtime.bodies.front.global_transform*Vector3(-1,1,-7);eye_offset=Vector3(24,12,-30)
	elif view_mode==7:
		target=runtime.bodies.front.global_transform*Vector3(15.5,-.5,-6.5);eye_offset=Vector3(12,7,-14)
	elif view_mode==8:
		target=runtime.bodies.front.global_transform*Vector3(-2,2,7);eye_offset=Vector3(-21,13,29)
	elif view_mode==9:
		target=runtime.bodies.front.global_transform*Vector3(1,13,0);eye_offset=Vector3(19,7,23)
	elif view_mode==10:
		target=runtime.bodies.rear.global_transform*Vector3(6.5,9,4);eye_offset=Vector3(16,9,12)
	elif view_mode==11:
		target=runtime.bodies.rear.global_transform*Vector3(6.5,3,11.15);eye_offset=Vector3(-2.5,5,20)
	elif view_mode==12:
		target=runtime.bodies.front.global_transform*Vector3(8,-2.9,13.5);eye_offset=Vector3(4,2,12)
	elif view_mode==13:
		target=runtime.bodies.front.global_transform*Vector3(29,2,-3.5);eye_offset=Vector3(7,3,-13)
	camera.position=target+eye_offset;camera.look_at(target)
	var s:Dictionary=runtime.state();var ratio:float=runtime.elapsed/maxf(.001,float(Time.get_ticks_usec()-started)/1e6)
	label.text="利维坦 003  /  行驶验证版\n实测 %.1f km/h   ·   %.0f FPS   ·   仿真 %.2f×\nW/S 行驶  A/D 转向  Shift 100 km/h  空格制动\n右键环视  滚轮缩放  Tab 切换细节视角  F12 截图\n03 极地雪原 · 003 行驶版"%[s.speed_m_s*3.6,Engine.get_frames_per_second(),ratio]
	if not screenshot_done and runtime.elapsed>10 and DisplayServer.get_name()!="headless":screenshot_done=true;_capture("godot")
	if mode=="review":
		var selected:int=clampi(int((runtime.elapsed-2)/4),0,REVIEW_VIEWS.size()-1);view_mode=selected
		if runtime.elapsed>3+selected*4 and not detail_captures.has(selected):detail_captures[selected]=true;_capture(REVIEW_VIEWS[selected])

func _capture(name_:String)->void:
	if output.is_empty():return
	await RenderingServer.frame_post_draw
	get_viewport().get_texture().get_image().save_png(output.path_join(name_+".png"))

func _finish()->void:
	set_physics_process(false)
	var wall:float=float(Time.get_ticks_usec()-started)/1e6
	frames.sort();var median_ms:float=frames[frames.size()/2]*1000 if frames.size()>0 else 0.
	var p95_ms:float=frames[int(frames.size()*.95)]*1000 if frames.size()>0 else 0.
	var report:Dictionary={"engine":Engine.get_version_info(),"renderer":RenderingServer.get_current_rendering_method(),"headless":DisplayServer.get_name()=="headless","terrain":terrain_name,"mode":mode,"wall_seconds":wall,"sim_seconds":runtime.elapsed,"sim_wall_ratio":runtime.elapsed/wall,"median_frame_ms":median_ms,"p95_frame_ms":p95_ms,"resolution":[get_viewport().get_visible_rect().size.x,get_viewport().get_visible_rect().size.y],"state":runtime.state(),"samples":samples}
	if not output.is_empty():var f:=FileAccess.open(output.path_join("result.json"),FileAccess.WRITE);f.store_string(JSON.stringify(report,"  "))
	get_tree().quit()
