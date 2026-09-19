extends "res://main.gd"
## Isolated full articulated Jolt replay; no body-state driving after initial placement.
var experiment: Dictionary

func _ready() -> void:
	experiment=JSON.parse_string(FileAccess.get_file_as_string("res://experiment.json"))
	super._ready()

func build_ground() -> void:
	if riser>0.:
		super.build_ground()
		return
	var terrain: Dictionary=experiment.terrain
	var faces:=PackedVector3Array()
	for j in range(terrain.y.size()-1):
		for i in range(terrain.x.size()-1):
			var a:=Vector3(terrain.x[i],terrain.z[j][i],-terrain.y[j])
			var b:=Vector3(terrain.x[i+1],terrain.z[j][i+1],-terrain.y[j])
			var c:=Vector3(terrain.x[i],terrain.z[j+1][i],-terrain.y[j+1])
			var d:=Vector3(terrain.x[i+1],terrain.z[j+1][i+1],-terrain.y[j+1])
			faces.append_array(PackedVector3Array([a,b,c,b,d,c]))
	var ground:=StaticBody3D.new()
	ground.collision_layer=2;ground.collision_mask=5
	var material:=PhysicsMaterial.new();material.friction=.8
	ground.physics_material_override=material
	add_child(ground)
	var shape:=ConcavePolygonShape3D.new();shape.set_faces(faces);shape.backface_collision=true
	var collision:=CollisionShape3D.new();collision.shape=shape;ground.add_child(collision)

func movement_command() -> Array:
	if riser>0.:return super.movement_command()
	var t:float=robot.sim_time_seconds()
	var driving:bool=t>=1. and t<6.
	var turn:float=.3 if experiment.get("maneuver","")=="turn" and driving else 0.
	var low:float=float(experiment.get("crouch",0.))
	if experiment.get("maneuver","")=="crouch_cycle":low=1. if t>=2. and t<4. else 0.
	return [float(experiment.get("speed",.5)) if driving else 0.,turn,low]

func exchange(state: Dictionary) -> Dictionary:
	state["terrain_path_heights"]=preload("res://terrain_scan.gd").wheel_path(self,2)
	state["terrain_edge_heights"]=preload("res://terrain_scan.gd").edge_profile(self,2)
	state["wheel_ground_heights"]=preload("res://terrain_scan.gd").wheel_ground(self,2)
	# Do not force stair_course: real edge detection must work in free driving.
	return super.exchange(state)
