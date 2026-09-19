extends "res://atelier/workshop.gd"
## Independent map 03. Maps 01 and 02 retain their existing authored scenery.
var landscape_builder:=preload("res://polar_range/terrain.gd").new()

func _init() -> void:
	label_font=load("res://atelier/ui_font.tres")

func _floor() -> void:
	server.get_node("World/Floor/FloorMesh").hide()
	server.get_node("World/Floor/CollisionShape3D").disabled=true

func _build_solids() -> void:
	_floor()
	landscape_builder.build(self)
	loose_props=load("res://atelier/loose_props.gd").new();loose_props.name="LooseProps";add_child(loose_props)
	loose_props.w=self;loose_props.visuals=visuals_enabled

func _build_details() -> void:
	pass

func ground_height(x:float,z:float) -> float:
	return landscape_builder.ground(x,z)

func _environment() -> void:
	super._environment()
	var env:Environment=server.get_node("WorldEnvironment").environment
	var sky:=Sky.new();var material:=ShaderMaterial.new();material.shader=load("res://polar_range/sky.gdshader")
	sky.sky_material=material;env.sky=sky;env.background_mode=Environment.BG_SKY
	env.ambient_light_color=Color("b1c4d8");env.ambient_light_energy=.35
	env.tonemap_mode=Environment.TONE_MAPPER_FILMIC;env.tonemap_exposure=.85
	env.fog_enabled=true;env.fog_light_color=Color("acbecd");env.fog_density=.000045;env.fog_sky_affect=0.
	var sun:DirectionalLight3D=server.get_node("World/Sun")
	sun.rotation_degrees=Vector3(-24,-38,0);sun.light_energy=.72
	sun.shadow_normal_bias=.35;sun.shadow_bias=.01
	sun.directional_shadow_mode=DirectionalLight3D.SHADOW_PARALLEL_4_SPLITS
	sun.directional_shadow_max_distance=230.
	RenderingServer.directional_shadow_atlas_set_size(4096,true)
	camera.far=30000.;camera.near=.015

func cycle_view() -> void:
	set_view("follow")
