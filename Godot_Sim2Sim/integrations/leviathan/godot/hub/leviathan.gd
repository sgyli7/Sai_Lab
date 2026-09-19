extends Node3D
## Host adapter for the external Blender/CAD vehicle bundle.
## Layer 8 remains the host's driving-height query layer.
const VEHICLE_LAYERS := 16|32|64|128
const DRIVING_LAYER := 8
var vehicle: Node3D
var failure := ""
var manifest: Dictionary
var parked_state: Dictionary = {}

func load_vehicle(host:Node3D,specification:Dictionary,directory:String="res://leviathan",visuals:bool=true)->bool:
	# The UI host processes while paused. Its physical child must not integrate
	# hydraulic state or accumulate forces while the native world is stopped.
	process_mode=Node.PROCESS_MODE_PAUSABLE
	if not OS.has_feature("double"):
		failure="Leviathan's low-speed native configuration requires the validated double-precision engine."
		return false
	if Engine.physics_ticks_per_second!=2000 or int(ProjectSettings.get_setting("physics/jolt_physics_3d/simulation/velocity_steps"))!=32 or int(ProjectSettings.get_setting("physics/jolt_physics_3d/simulation/position_steps"))!=4:
		failure="Configure the validated 2000 Hz, 32/4 native space before creating the carrier."
		return false
	manifest=specification
	var source:Array=manifest.placement.canonical_translation_m
	var placement:=Transform3D(Basis(Vector3.UP,deg_to_rad(float(manifest.placement.yaw_deg))),Vector3(source[0],source[2],-source[1]))
	vehicle=load(directory.path_join("imported_vehicle.gd")).new()
	add_child(vehicle)
	if not vehicle.load_bundle(directory,placement,visuals):
		failure=vehicle.failure
		return false
	if visuals:
		# The motion manifest names articulated animation dependencies only.
		# Static cargo, radar and suspension roots still follow their own bodies.
		var index:Dictionary={}
		vehicle.binding._index(vehicle.asset,index)
		for id in vehicle.runtime.bodies:
			if index.has(id):vehicle.binding.body_nodes[id]=index[id]
		vehicle.binding.report.body_bindings=vehicle.binding.body_nodes.size()
		vehicle.binding.update_measured()
	# Preserve the reference vehicle's collision graph in unused host layers.
	# World=2, robot=1 and loose props=4 are the existing workshop convention.
	for body in vehicle.runtime.bodies.values():
		var reference_layer:int=body.collision_layer
		body.collision_layer=_mapped_bits(reference_layer)
		body.collision_mask=_mapped_bits(body.collision_mask)|4
		if reference_layer==2:body.collision_layer|=DRIVING_LAYER
	_adapt_world(host)
	return true

func _mapped_bits(value:int)->int:
	var result:=0
	for pair in [[1,2],[2,16],[4,32],[8,64],[16,1],[32,128]]:
		if value&int(pair[0]):result|=int(pair[1])
	return result

func _adapt_world(node:Node)->void:
	if node==self:return
	if node is StaticBody3D or node is RigidBody3D:
		node.collision_mask|=VEHICLE_LAYERS
	for child in node.get_children():_adapt_world(child)

func adapt_robot(actor:Node)->void:
	# Call after every robot replacement; do not rewrite its published layers.
	if actor is RigidBody3D:actor.collision_mask|=VEHICLE_LAYERS
	for child in actor.get_children():adapt_robot(child)

func set_parked(value:bool)->void:
	if vehicle==null:return
	vehicle.runtime.set_vehicle_command(0.,0.)
	if value and parked_state.is_empty():
		for id in vehicle.runtime.bodies:
			var body:RigidBody3D=vehicle.runtime.bodies[id]
			parked_state[id]=body.freeze
			body.linear_velocity=Vector3.ZERO
			body.angular_velocity=Vector3.ZERO
			body.freeze=true
		vehicle.runtime.set_physics_process(false)
	elif not value and not parked_state.is_empty():
		for id in vehicle.runtime.bodies:
			vehicle.runtime.bodies[id].freeze=bool(parked_state.get(id,false))
		parked_state.clear()
		vehicle.runtime.set_physics_process(true)

func support_body_from_contacts(robot:Node3D,leg_order:Array)->String:
	var counts:Dictionary={}
	for leg in leg_order:
		for body in robot.bodies[str(leg)+"_wheel"].get_colliding_bodies():
			if vehicle.runtime.bodies.get(str(body.name))==body:
				var key:=str(body.name);counts[key]=int(counts.get(key,0))+1
	var best:="world";var count:=0
	for key in counts:
		if int(counts[key])>count:count=int(counts[key]);best=str(key)
	return best
