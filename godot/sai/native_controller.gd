extends RefCounted
## Sai locomotion controller executed entirely inside Godot.
##
## The numerical contract is a direct port of sai_agent.control and
## sai_agent.godot_controller.  Inputs stay in Sai's public right-handed
## X-forward/Y-left/Z-up frame; state is sampled from Jolt by robot.gd.

const CONTROL_DT := 0.02
const STAND_HEIGHT := 0.2192
const CROUCH_DROP := 0.035
const SIDES := [1.0,-1.0,1.0,-1.0]
const FRONTS := [1.0,1.0,-1.0,-1.0]
const LEG_INDICES := [0,1,2,4,5,6,8,9,10,12,13,14]
const WHEEL_INDICES := [3,7,11,15]
const DEFAULT_STAIRS := {
	"speed":0.12,"lift_height":0.055,"leg_scale":0.18,"min_crouch":0.0,
	"route_center_y":null,"motion_phase_start_seconds":null,
	"yaw_correction_limit":0.4,"wheel_residual_scale":6.0,
}

var flat_policy
var stair_policy
var flat_policy_id := "sai-flat-motion-v1"
var flat_policy_sha256 := "094adb4484b3d812dfb8a056491beda34a23fd0fa6f463b7784854ebc354d53d"
var stair_profile_id := "stairs-dev40"
var stair_settings := DEFAULT_STAIRS.duplicate(true)
var experimental_profile := false
var phase_free_stairs := false
var phase_free_dense_edges := false
var phase_free_event_memory := false
var high_obstacle_memory := false
var high_obstacle_threshold_m := .05
var medium_obstacle_threshold_m := .03
var recovery_active := false
var recovery_stall_s := 0.0
var recovery_steps := 0
var recovery_best_x = null
var previous := PackedFloat32Array()
var crouch := 0.0
var desired_heading: Variant = null
var motion_origin: Variant = null
var last_error := ""
var stance_impedance
var terrain_suspension
var stair_suspension
var payload_clamp_target_rad:=0.0
var safe_residual_stairs:=false
var task_space_skills:=false
var joint_action_scales:=[.30,.60,1.02]
var target_slew_rad_s:=[3.0,4.0,6.0]
var previous_target:Array=[]
var previous_target_initialized:=false
var skill_previous:=PackedFloat32Array()

func _init(profile_path := "") -> void:
	previous.resize(16)
	skill_previous.resize(16)
	previous_target.resize(16);previous_target.fill(0.0)
	flat_policy = _load_policy("res://sai_policy/flat-motion-v1.onnx",flat_policy_sha256)
	if flat_policy==null:return
	var suspension_profile:Dictionary=JSON.parse_string(FileAccess.get_file_as_string("res://sai_policy/suspension-v2.json"))
	if suspension_profile.get("flat_policy_sha256","")!=flat_policy_sha256:
		_fail("Sai suspension profile does not match flat policy")
		return
	stance_impedance=preload("res://sai/stance_impedance.gd").new(suspension_profile)
	terrain_suspension=preload("res://sai/terrain_suspension.gd").new(suspension_profile)
	var stairs_path := "res://sai_policy/stairs-dev40.onnx"
	var stairs_hash := "0eca6930e7ccd3201023ce9dd0ce4b9cb0476a9490020da89802cac598b2e38d"
	if not str(profile_path).is_empty():
		var profile: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(profile_path))
		var schema:=int(profile.get("schema_version",0))
		if schema not in [1,2]:
			_fail("Unsupported experimental stair profile schema")
		if schema==2:
			var contract:=str(profile.get("contract",""))
			var expected_observation:=258 if contract=="sai-task-space-skills-v6" else 243 if contract=="sai-phase-free-stairs-v4" else 242 if contract in ["sai-phase-free-stairs-v3","sai-safe-residual-stairs-v5"] else 104
			if contract not in ["sai-phase-free-stairs-v2","sai-phase-free-stairs-v3","sai-phase-free-stairs-v4","sai-safe-residual-stairs-v5","sai-task-space-skills-v6"] or int(profile.get("observation_size",0))!=expected_observation or int(profile.get("action_size",0))!=16:
				_fail("Invalid phase-free stair policy contract")
			phase_free_stairs=true
			phase_free_dense_edges=contract in ["sai-phase-free-stairs-v3","sai-phase-free-stairs-v4","sai-safe-residual-stairs-v5"]
			phase_free_event_memory=contract=="sai-phase-free-stairs-v4"
			safe_residual_stairs=contract=="sai-safe-residual-stairs-v5"
			task_space_skills=contract=="sai-task-space-skills-v6"
			if safe_residual_stairs:
				joint_action_scales=profile.get("joint_action_scales",[])
				target_slew_rad_s=profile.get("target_slew_rad_s",[])
				if joint_action_scales.size()!=3 or target_slew_rad_s.size()!=3:_fail("Invalid safe-residual stair envelope")
			if contract=="sai-phase-free-stairs-v4":
				high_obstacle_threshold_m=float(profile.get("high_obstacle_threshold_m",.05))
				medium_obstacle_threshold_m=float(profile.get("medium_obstacle_threshold_m",.03))
				stair_suspension=preload("res://sai/terrain_suspension.gd").new({"geometry_parameters":profile.stair_suspension_geometry},true)
			payload_clamp_target_rad=float(profile.get("payload_clamp_target_rad",0.0))
		for key in profile.get("control",{}):
			if not stair_settings.has(key): _fail("Unknown experimental stair control setting: "+str(key))
		stair_settings.merge(profile.get("control",{}),true)
		stairs_path = str(profile_path).get_base_dir().path_join(str(profile.actor))
		stairs_hash = str(profile.onnx_sha256)
		stair_profile_id = str(profile.id)
		experimental_profile = true
	_validate_settings()
	if not last_error.is_empty():return
	stair_policy = _load_policy(stairs_path,stairs_hash)

func _fail(message: String) -> void:
	last_error = message
	push_error(message)

func _load_policy(path: String, expected_hash: String):
	if not ClassDB.class_exists("MicroDuckPolicy"):
		_fail("Native ONNX policy extension is unavailable")
		return null
	if not FileAccess.file_exists(path):
		_fail("Sai policy is missing: "+path)
		return null
	var raw := FileAccess.get_file_as_bytes(path)
	var hash := HashingContext.new()
	hash.start(HashingContext.HASH_SHA256)
	hash.update(raw)
	if hash.finish().hex_encode() != expected_hash:
		_fail("Sai policy hash mismatch: "+path)
		return null
	var policy = ClassDB.instantiate("MicroDuckPolicy")
	if not policy.load_model(raw):
		_fail("Sai policy load failed: "+str(policy.get_last_error()))
		return null
	return policy

func _validate_settings() -> void:
	for item in [["speed",0.02,0.16],["lift_height",0.0,0.1],["leg_scale",0.0,0.6],["min_crouch",0.0,1.0]]:
		var value := float(stair_settings[item[0]])
		if value < item[1] or value > item[2]: _fail("Experimental stair setting out of range: "+item[0])
	var route = stair_settings.route_center_y
	if route != null and not is_finite(float(route)): _fail("Non-finite route center")
	var start = stair_settings.motion_phase_start_seconds
	if start != null and (float(start)<0.0 or float(start)>=3.2): _fail("Invalid motion phase start")
	var correction := float(stair_settings.yaw_correction_limit)
	if correction <= 0.0 or correction > 1.2: _fail("Invalid heading correction limit")
	var wheel_residual:=float(stair_settings.wheel_residual_scale)
	if wheel_residual<0.0 or wheel_residual>12.0:_fail("Invalid stair wheel residual scale")

func reset() -> void:
	previous.fill(0.0)
	skill_previous.fill(0.0)
	previous_target.fill(0.0)
	previous_target_initialized=false
	crouch = 0.0
	desired_heading = null
	motion_origin = null
	high_obstacle_memory=false
	recovery_active=false
	recovery_stall_s=0.0
	recovery_steps=0
	recovery_best_x=null
	if stance_impedance!=null:stance_impedance.reset()
	if terrain_suspension!=null:terrain_suspension.reset()

func command(state: Dictionary) -> Dictionary:
	if state.get("robot_id") != "Sai_Agent_001" or state.get("physics_owner") != "Godot/Jolt":
		_fail("Unexpected robot or physics owner")
		return {}
	if state.q.size()<16 or state.v.size()<16 or state.command.size()!=3 or state.terrain_heights.size()!=24:
		_fail("Invalid articulated Sai state")
		return {}
	var requested: Array = [float(state.command[0]),float(state.command[1]),float(state.command[2])]
	if phase_free_dense_edges and stair_suspension!=null:
		if requested[0]<=.015:
			high_obstacle_memory=false;recovery_active=false;recovery_stall_s=0.0;recovery_steps=0;recovery_best_x=null
		var dense:Array=state.get("terrain_edge_heights",[])
		var maximum_rise:=0.0
		var medium_edge_near:=false
		if dense.size()==138:
			for i in range(135):
				maximum_rise=maxf(maximum_rise,float(dense[i+3])-float(dense[i]))
			for row in range(45):
				var edge_x:=-.30+.02*row
				if edge_x<-.22 or edge_x>.20:continue
				for column in range(3):
					if float(dense[(row+1)*3+column])-float(dense[row*3+column])>medium_obstacle_threshold_m:
						medium_edge_near=true
		if maximum_rise>high_obstacle_threshold_m:
			high_obstacle_memory=true;recovery_active=false;recovery_stall_s=0.0;recovery_steps=0;recovery_best_x=null
		elif maximum_rise>medium_obstacle_threshold_m and medium_edge_near and requested[0]>.015 and not high_obstacle_memory:
			var forward_x:=float(state.base_position[0])
			if recovery_active:
				recovery_steps+=1
				if recovery_steps>=25:
					recovery_active=false;recovery_steps=0;recovery_stall_s=0.0;recovery_best_x=forward_x
			elif recovery_best_x==null or forward_x>float(recovery_best_x)+.01:
				recovery_best_x=forward_x;recovery_stall_s=0.0
			else:
				recovery_stall_s+=CONTROL_DT
				if recovery_stall_s>=.6:recovery_active=true;recovery_steps=0
		else:
			recovery_active=false;recovery_stall_s=0.0;recovery_steps=0;recovery_best_x=null
	var relevant := _step_in_wheel_path(state,requested[2]>.5)
	var descending: bool = bool(state.get("contact_following_descent",false)) and _descending_in_path(state) and requested[0]>.015
	if descending: requested[0]=minf(.16,requested[0])
	var crouch_blocked: bool = requested[2]>.5 and requested[0]>.015 and _span(state.terrain_heights)>.008 and _step_in_wheel_path(state,true,false)
	if crouch_blocked: requested[0]=0.0
	var stairs: bool = not descending and relevant and stair_policy!=null and _span(state.terrain_heights)>.004 and requested[0]>.015
	if stairs: requested[0]=minf(requested[0],float(stair_settings.speed))
	if requested[0]<=.015: motion_origin=null
	elif motion_origin==null: motion_origin=float(state.time)
	var phase_time := float(state.time)
	if stairs and stair_settings.motion_phase_start_seconds!=null:
		phase_time=phase_time-float(motion_origin)+float(stair_settings.motion_phase_start_seconds)
	var requested_crouch := maxf(requested[2],float(stair_settings.min_crouch))
	crouch += clampf(requested_crouch-crouch,-.04,.04)
	if task_space_skills:
		return _task_space_skill_command(state,requested,phase_time)
	var expert_high:=high_obstacle_memory or recovery_active
	var observation := observation_direct(state,requested,previous,phase_free_dense_edges,phase_free_event_memory,expert_high) if stairs and phase_free_stairs else observation(state,requested,crouch,previous,phase_time*TAU/(3.2 if stairs else 2.4))
	var selected = stair_policy if stairs else flat_policy
	var inferred: PackedFloat32Array=selected.infer(observation)
	if inferred.size()!=16:
		_fail("Native Sai inference failed: "+str(selected.get_last_error()))
		return {}
	var action: PackedFloat32Array = filter_action(inferred,requested)
	var target := targets_direct_safe(action,requested,float(stair_settings.wheel_residual_scale)) if stairs and safe_residual_stairs else targets_direct(action,requested,float(stair_settings.wheel_residual_scale)) if stairs and phase_free_stairs else targets_stairs(action,requested,crouch,phase_time/3.2,state.terrain_heights,
		float(stair_settings.lift_height),float(stair_settings.leg_scale)) if stairs else targets(action,requested,crouch)
	var rotation: Array = state.base_rotation_columns
	var yaw := atan2(float(rotation[0][1]),float(rotation[0][0]))
	var yaw_rate := _dot3(rotation[2],state.base_angular_world)
	if stairs and stair_settings.route_center_y!=null and absf(requested[1])<1e-5:
		desired_heading=clampf(atan2(float(stair_settings.route_center_y)-float(state.base_position[1]),.5),-.4,.4)
	target=_heading(target,requested,yaw,yaw_rate)
	previous_target=target.duplicate()
	previous=action.duplicate()
	var stage := "descending" if descending else "crouch_blocked" if crouch_blocked else "stairs" if stairs else "crouched" if crouch>.5 else "rolling"
	var result:={"mode":"transport","stage":stage,"target_leg":target,"wheel_speed":[target[3],target[7],target[11],target[15]],
		"target_arm":[0.0,0.0,0.0,0.0,0.0,0.0],"arm_bias":state.get("arm_gravity_bias",[0.0,0.0,0.0,0.0,0.0,0.0]),
		"grip_cap":1.4,"cargo_target_rad":payload_clamp_target_rad,"policy_action":Array(action),"policy_observation":Array(observation),
		"physics_advanced_by_controller":false,"effective_crouch":crouch,"stair_profile":stair_profile_id,
		"flat_policy_id":flat_policy_id,"flat_policy_sha256":flat_policy_sha256,
		"terrain_step_relevant":relevant,"controller_backend":"godot-native-onnxruntime",
		"obstacle_expert":"high" if expert_high else "low",
		"contract_id":"sai-safe-residual-stairs-v5" if stairs and safe_residual_stairs else "sai-phase-free-stairs-v4" if stairs and stair_suspension!=null else "sai-phase-free-stairs-v3" if stairs and phase_free_dense_edges else "sai-phase-free-stairs-v2" if stairs and phase_free_stairs else "sai-experimental-stairs-v1" if experimental_profile else "sai-flat-v1+heading-v1"}
	var selected_suspension=stair_suspension if stairs and expert_high else terrain_suspension
	if selected_suspension!=null:result=selected_suspension.apply(result,state)
	return stance_impedance.apply(result,state) if stance_impedance!=null else result

func _task_space_skill_command(state:Dictionary,requested:Array,phase_time:float)->Dictionary:
	var flat_observation:=observation(state,requested,crouch,previous,phase_time*TAU/2.4)
	# v6 keeps the rolling prior terrain-neutral. Suspension rejects roughness
	# and the task-space policy handles edges; otherwise the flat actor creates
	# a second stair gait that changes with the number of visible steps.
	for i in range(58,82):flat_observation[i]=0.0
	var flat_inferred:PackedFloat32Array=flat_policy.infer(flat_observation)
	if flat_inferred.size()!=16:
		_fail("Native Sai flat-prior inference failed: "+str(flat_policy.get_last_error()))
		return {}
	var flat_action:=filter_action(flat_inferred,requested)
	var flat_target_action:=flat_action.duplicate()
	var front_wheel:=.5*(float(flat_action[3])+float(flat_action[7]))
	var rear_wheel:=.5*(float(flat_action[11])+float(flat_action[15]))
	flat_target_action[3]=front_wheel;flat_target_action[7]=front_wheel
	flat_target_action[11]=rear_wheel;flat_target_action[15]=rear_wheel
	var base_target:=targets(flat_target_action,requested,crouch)
	var rotation:Array=state.base_rotation_columns
	var yaw:=atan2(float(rotation[0][1]),float(rotation[0][0]))
	var yaw_rate:=_dot3(rotation[2],state.base_angular_world)
	base_target=_heading(base_target,requested,yaw,yaw_rate)
	var skill_observation:=observation_task_skill(state,requested,skill_previous,base_target)
	var inferred:PackedFloat32Array=stair_policy.infer(skill_observation)
	if inferred.size()!=16:
		_fail("Native Sai task-skill inference failed: "+str(stair_policy.get_last_error()))
		return {}
	var skill_action:=filter_action(inferred,requested)
	var projected_up:=Vector3(float(rotation[0][2]),float(rotation[1][2]),float(rotation[2][2]))
	var measured_pitch:=atan2(-projected_up.x,projected_up.z)
	var pitch_guard:=clampf((absf(measured_pitch)-deg_to_rad(7.0))/deg_to_rad(4.0),0.0,1.0)
	if pitch_guard>0.0 and measured_pitch>0.0:skill_action[14]=maxf(float(skill_action[14]),pitch_guard)
	elif pitch_guard>0.0 and measured_pitch<0.0:skill_action[14]=minf(float(skill_action[14]),-pitch_guard)
	var skill_relevant:=_step_in_wheel_path(state,requested[2]>.5,false)
	if not skill_relevant:skill_action[15]=0.0
	var target:=targets_task_skill(skill_action,base_target,previous_target)
	for i in range(16):previous_target[i]=float(target[i])-float(base_target[i])
	previous_target_initialized=true;previous=flat_action.duplicate();skill_previous=skill_action.duplicate()
	var intensity:=maxf(0.0,float(skill_action[15]))
	var stage:="task_skill" if intensity>.05 else "crouched" if crouch>.5 else "rolling"
	var result:={"mode":"transport","stage":stage,"target_leg":target,"wheel_speed":[target[3],target[7],target[11],target[15]],
		"target_arm":[0.0,0.0,0.0,0.0,0.0,0.0],"arm_bias":state.get("arm_gravity_bias",[0.0,0.0,0.0,0.0,0.0,0.0]),
		"grip_cap":1.4,"cargo_target_rad":payload_clamp_target_rad,"policy_action":Array(skill_action),"policy_observation":Array(skill_observation),
		"physics_advanced_by_controller":false,"effective_crouch":crouch,"stair_profile":stair_profile_id,
		"flat_policy_id":flat_policy_id,"flat_policy_sha256":flat_policy_sha256,"terrain_step_relevant":skill_relevant,
		"controller_backend":"godot-native-onnxruntime","contract_id":"sai-task-space-skills-v6",
		"skill_intensity":intensity,"terrain_observation_contract":"godot-oracle-terrain",
		"flat_prior_terrain_mode":"level"}
	if terrain_suspension!=null:result=terrain_suspension.apply(result,state)
	result["target_leg"]=_project_task_target_safety(result.target_leg)
	result["wheel_speed"]=[result.target_leg[3],result.target_leg[7],result.target_leg[11],result.target_leg[15]]
	return stance_impedance.apply(result,state) if stance_impedance!=null else result

static func _project_task_target_safety(target:Array)->Array:
	var result:=target.duplicate()
	for leg in range(4):
		result[leg*4]=clampf(float(result[leg*4]),-.30,.30)
		result[leg*4+1]=clampf(float(result[leg*4+1]),-.50,.50)
		result[leg*4+2]=clampf(float(result[leg*4+2]),-1.02,1.02)
	return result

static func observation(state: Dictionary, input_command: Array, effective_crouch: float,
		last_action: PackedFloat32Array, phase: float) -> PackedFloat32Array:
	var result := PackedFloat32Array()
	result.resize(82)
	var columns: Array = state.base_rotation_columns
	# Rotation row 2, then world linear velocity expressed in the body frame.
	for i in range(3): result[i]=float(columns[i][2])
	for i in range(3): result[3+i]=_dot3(columns[i],state.base_linear_world)
	for i in range(3): result[6+i]=_dot3(columns[i],state.base_angular_world)
	result[9]=input_command[0];result[10]=input_command[1];result[11]=effective_crouch
	for i in range(12):
		result[12+i]=float(state.q[LEG_INDICES[i]])
		result[24+i]=float(state.v[LEG_INDICES[i]])*.1
	for i in range(4): result[36+i]=float(state.v[WHEEL_INDICES[i]])*SIDES[i]*.1
	for i in range(16): result[40+i]=last_action[i]
	result[56]=sin(phase);result[57]=cos(phase)
	var ground := float(state.base_position[2])-STAND_HEIGHT
	for i in range(24): result[58+i]=clampf((float(state.terrain_heights[i])-ground)*5.0,-2.0,2.0)
	return result

static func observation_direct(state:Dictionary,input_command:Array,last_action:PackedFloat32Array,dense_edges:=false,event_memory:=false,high_obstacle_latched:=false)->PackedFloat32Array:
	var result:=PackedFloat32Array();result.resize(243 if event_memory else 242 if dense_edges else 104)
	var columns:Array=state.base_rotation_columns
	for i in range(3):result[i]=float(columns[i][2])
	for i in range(3):result[3+i]=_dot3(columns[i],state.base_linear_world)
	for i in range(3):result[6+i]=_dot3(columns[i],state.base_angular_world)
	for i in range(3):result[9+i]=float(input_command[i])
	for i in range(12):
		result[12+i]=float(state.q[LEG_INDICES[i]])
		result[24+i]=float(state.v[LEG_INDICES[i]])*.1
	for i in range(4):result[36+i]=float(state.v[WHEEL_INDICES[i]])*SIDES[i]*.1
	for i in range(16):result[40+i]=last_action[i]
	var ground:Array=state.get("wheel_ground_heights",[])
	if ground.size()!=4:return PackedFloat32Array()
	var local_ground:=_mean(ground)
	for i in range(24):result[56+i]=clampf((float(state.terrain_heights[i])-local_ground)*5.0,-2.0,2.0)
	var path:Array=state.get("terrain_path_heights",[])
	if path.size()!=15:return PackedFloat32Array()
	for i in range(15):result[80+i]=clampf((float(path[i])-local_ground)*5.0,-2.0,2.0)
	for i in range(4):result[95+i]=clampf((float(ground[i])-local_ground)*5.0,-1.0,1.0)
	var wheels:Array=state.get("wheel_positions",[])
	for i in range(4):
		var clearance:=float(wheels[i][2])-float(ground[i])-.048 if wheels.size()==4 else 0.0
		result[99+i]=clampf(clearance*20.0,-1.0,2.0)
	result[103]=(float(state.base_position[2])-local_ground-STAND_HEIGHT)*10.0
	if dense_edges:
		var dense:Array=state.get("terrain_edge_heights",[])
		if dense.size()!=138:return PackedFloat32Array()
		for i in range(138):result[104+i]=clampf((float(dense[i])-local_ground)*5.0,-2.0,2.0)
		if result.size()==243:result[242]=1.0 if high_obstacle_latched else 0.0
	return result

static func observation_task_skill(state:Dictionary,input_command:Array,last_action:PackedFloat32Array,base_target:Array)->PackedFloat32Array:
	var result:=observation_direct(state,input_command,last_action,true,false,false)
	if result.size()!=242 or base_target.size()!=16:return PackedFloat32Array()
	result.resize(258)
	for i in range(16):result[242+i]=float(base_target[i])
	var nearest_higher:=INF
	var nearest_lower:=-INF
	for i in range(56,95):
		var value:=float(result[i])
		if value>.06:nearest_higher=minf(nearest_higher,value)
		elif value<-.06:nearest_lower=maxf(nearest_lower,value)
	for i in range(104,242):
		var value:=float(result[i])
		if value>.06:nearest_higher=minf(nearest_higher,value)
		elif value<-.06:nearest_lower=maxf(nearest_lower,value)
	for i in range(56,95):
		if nearest_higher<INF:result[i]=minf(float(result[i]),nearest_higher)
		if nearest_lower>-INF:result[i]=maxf(float(result[i]),nearest_lower)
	for i in range(104,242):
		if nearest_higher<INF:result[i]=minf(float(result[i]),nearest_higher)
		if nearest_lower>-INF:result[i]=maxf(float(result[i]),nearest_lower)
	return result

static func targets_task_skill(action:PackedFloat32Array,base_target:Array,last_residual:Array)->Array:
	var result:=base_target.duplicate()
	var intensity:=clampf(float(action[15]),0.0,1.0)
	var theta0:=atan2(.05,.074833147)
	var beta0:=atan2(.05,.09797959)+theta0
	for leg in range(4):
		var side:float=SIDES[leg];var front:float=FRONTS[leg]
		var hip:=float(base_target[leg*4+1]);var knee:=float(base_target[leg*4+2])
		var theta:=front*theta0-hip/side
		var beta:=-front*beta0-knee/side
		var dx:=.09*sin(theta)+.11*sin(theta+beta)
		var down:=.09*cos(theta)+.11*cos(theta+beta)
		dx+=intensity*.060*float(action[leg])
		var vertical:=float(action[4+leg])
		var lift:=(.080 if vertical>=0.0 else .010)*vertical
		down-=intensity*lift
		down+=intensity*.025*float(action[12])
		down+=intensity*side*.146*tan(deg_to_rad(8.0)*float(action[13]))
		down+=intensity*front*.115*tan(deg_to_rad(12.0)*float(action[14]))
		var radius:=sqrt(dx*dx+down*down)
		var workspace_scale:=minf(.194/maxf(radius,1e-8),1.0)
		dx*=workspace_scale;down=clampf(down*workspace_scale,.105,.194)
		var cosine:=clampf((dx*dx+down*down-.09*.09-.11*.11)/(2.0*.09*.11),-1.0,1.0)
		beta=-front*acos(cosine)
		theta=atan2(dx,down)-atan2(.11*sin(beta),.09+.11*cos(beta))
		result[leg*4]=clampf(float(base_target[leg*4]),-.25,.25)
		result[leg*4+1]=clampf(side*(front*theta0-theta),-.50,.50)
		result[leg*4+2]=clampf(side*(-front*beta0-beta),-1.02,1.02)
		var axle_start:=8 if leg<2 else 10
		var axle_wheel:=.5*(float(action[axle_start])+float(action[axle_start+1]))
		result[leg*4+3]=float(base_target[leg*4+3])+intensity*6.0*axle_wheel*SIDES[leg]
	var slew:=[2.0,4.0,6.0,20.0]
	for i in range(16):
		var desired_residual:=float(result[i])-float(base_target[i])
		result[i]=float(base_target[i])+float(last_residual[i])+clampf(desired_residual-float(last_residual[i]),-float(slew[i%4])*.02,float(slew[i%4])*.02)
	return result

static func filter_action(action: PackedFloat32Array, input_command: Array) -> PackedFloat32Array:
	var result := PackedFloat32Array()
	result.resize(16)
	var moving: bool = sqrt(float(input_command[0])*float(input_command[0])+float(input_command[1])*float(input_command[1]))>1e-5
	for i in range(16): result[i]=clampf(action[i],-1.0,1.0) if moving else 0.0
	return result

static func targets(action: PackedFloat32Array, input_command: Array, effective_crouch: float) -> Array:
	var filtered := filter_action(action,input_command)
	var down := .172812737-CROUCH_DROP*effective_crouch
	var result: Array=[]
	result.resize(16)
	var theta0 := atan2(.05,.074833147)
	var beta0 := atan2(.05,.09797959)+theta0
	for leg in range(4):
		var beta: float = -FRONTS[leg]*acos(clampf((down*down-.09*.09-.11*.11)/(2.0*.09*.11),-1.0,1.0))
		var theta := -atan2(.11*sin(beta),.09+.11*cos(beta))
		var base := leg*4
		result[base]=.18*filtered[base]
		result[base+1]=.18*filtered[base+1]+SIDES[leg]*(FRONTS[leg]*theta0-theta)
		result[base+2]=.18*filtered[base+2]+SIDES[leg]*(-FRONTS[leg]*beta0-beta)
		result[base+3]=SIDES[leg]*((input_command[0]-input_command[1]*SIDES[leg]*.146)/.048+6.0*filtered[base+3])
	return result

static func targets_direct(action:PackedFloat32Array,input_command:Array,wheel_residual_scale:=6.0)->Array:
	var filtered:=filter_action(action,input_command)
	var result:Array=[];result.resize(16);result.fill(0.0)
	for leg in range(4):
		var base:=leg*4
		result[base]=clampf(.45*filtered[base],-.45,.45)
		result[base+1]=clampf(.70*filtered[base+1],-.70,.70)
		result[base+2]=clampf(1.20*filtered[base+2],-1.20,1.20)
		result[base+3]=SIDES[leg]*((float(input_command[0])-float(input_command[1])*SIDES[leg]*.146)/.048+wheel_residual_scale*filtered[base+3])
	return result

func targets_direct_safe(action:PackedFloat32Array,input_command:Array,wheel_residual_scale:=6.0)->Array:
	var filtered:=filter_action(action,input_command)
	var desired:Array=[];desired.resize(16);desired.fill(0.0)
	for leg in range(4):
		var base:=leg*4
		desired[base]=float(joint_action_scales[0])*filtered[base]
		desired[base+1]=float(joint_action_scales[1])*filtered[base+1]
		desired[base+2]=float(joint_action_scales[2])*filtered[base+2]
		desired[base+3]=SIDES[leg]*((float(input_command[0])-float(input_command[1])*SIDES[leg]*.146)/.048+wheel_residual_scale*filtered[base+3])
		for offset in range(4):
			var rate:=20.0 if offset==3 else float(target_slew_rad_s[offset])
			desired[base+offset]=float(previous_target[base+offset])+clampf(float(desired[base+offset])-float(previous_target[base+offset]),-rate*CONTROL_DT,rate*CONTROL_DT)
	return desired

static func targets_stairs(action: PackedFloat32Array,input_command: Array,effective_crouch: float,
		phase_cycles: float,scan_heights: Array,lift_height:=.055,leg_scale:=.18,stride:=.05) -> Array:
	var filtered := filter_action(action,input_command)
	var result := targets(filtered,input_command,effective_crouch)
	var active: bool = _span(scan_heights)>.004 and input_command[0]>.015
	var theta0 := atan2(.05,.074833147)
	var beta0 := atan2(.05,.09797959)+theta0
	var offsets := [0.0,.5,.75,.25]
	for leg in range(4):
		var phase := fposmod(phase_cycles-offsets[leg],1.0)
		var lift := lift_height*pow(sin(PI*phase/.25),2.0) if active and phase<.25 else 0.0
		var dx := (-stride*.5*cos(PI*phase/.25) if phase<.25 else stride*(.5-(phase-.25)/.75)) if active else 0.0
		var down := .172812737-CROUCH_DROP*effective_crouch-lift
		var beta: float = -FRONTS[leg]*acos(clampf((down*down+dx*dx-.09*.09-.11*.11)/(2.0*.09*.11),-1.0,1.0))
		var theta := atan2(dx,down)-atan2(.11*sin(beta),.09+.11*cos(beta))
		var base := leg*4
		result[base]=clampf(leg_scale*filtered[base],-.45,.45)
		result[base+1]=clampf(SIDES[leg]*(FRONTS[leg]*theta0-theta)+leg_scale*filtered[base+1],-.7,.7)
		result[base+2]=clampf(SIDES[leg]*(-FRONTS[leg]*beta0-beta)+leg_scale*filtered[base+2],-1.2,1.2)
	return result

func _heading(target: Array,input_command: Array,yaw: float,yaw_rate: float) -> Array:
	var parked: bool = sqrt(float(input_command[0])*float(input_command[0])+float(input_command[1])*float(input_command[1]))<1e-5
	if desired_heading==null or parked: desired_heading=yaw
	if parked:return target
	desired_heading=float(desired_heading)+float(input_command[1])*CONTROL_DT
	var error := atan2(sin(float(desired_heading)-yaw),cos(float(desired_heading)-yaw))
	var correction := clampf(1.5*error-.25*(yaw_rate-float(input_command[1])),
		-float(stair_settings.yaw_correction_limit),float(stair_settings.yaw_correction_limit))
	var result := target.duplicate()
	for leg in range(4): result[leg*4+3]-=correction*.146/.048
	return result

static func _step_in_wheel_path(state: Dictionary,crouched:=false,honor_course:=true) -> bool:
	if honor_course and (bool(state.get("stair_course",false)) or bool(state.get("experimental_profile",false))): return true
	var dense = state.get("terrain_edge_heights")
	if dense!=null and dense.size()==138:
		for value in _signed_edges(dense,crouched):
			if absf(value)>.008:return true
		return false
	var path = state.get("terrain_path_heights")
	if path==null:return true
	if path.size()!=15:return false
	var rows := 4 if crouched else 5
	for column in range(3):
		var delta: Array=[]
		for row in range(rows-1):delta.append(float(path[(row+1)*3+column])-float(path[row*3+column]))
		var sorted := delta.duplicate();sorted.sort()
		var median: float = float(sorted[sorted.size()/2]) if sorted.size()%2==1 else .5*(float(sorted[sorted.size()/2-1])+float(sorted[sorted.size()/2]))
		for value in delta:
			if absf(value-median)>.008:return true
	return false

static func _descending_in_path(state: Dictionary) -> bool:
	var dense=state.get("terrain_edge_heights")
	if dense==null or dense.size()!=138:return false
	var negative:=false
	for edge in _signed_edges(dense,false):
		if edge> .008:return false
		if edge<-.045:return false
		if edge<-.008:negative=true
	return negative

static func _signed_edges(dense: Array,crouched: bool) -> Array:
	var result: Array=[]
	for i in range(45):
		var x := -.30+.02*i
		if x<-.20 or x>=(.36 if crouched else .54):continue
		for column in range(3):
			var delta := float(dense[(i+1)*3+column])-float(dense[i*3+column])
			var neighbours: Array=[]
			for offset in [-2,-1,1,2]:
				var j:int=clampi(i+offset,0,44)
				neighbours.append(float(dense[(j+1)*3+column])-float(dense[j*3+column]))
			neighbours.sort()
			var background: float = .5*(float(neighbours[1])+float(neighbours[2]))
			result.append(delta-background)
	return result

static func _span(values: Array) -> float:
	var low:=INF;var high:=-INF
	for value in values:low=minf(low,float(value));high=maxf(high,float(value))
	return high-low

static func _mean(values:Array)->float:
	var total:=0.0
	for value in values:total+=float(value)
	return total/float(values.size())

static func _dot3(a,b) -> float:
	return float(a[0])*float(b[0])+float(a[1])*float(b[1])+float(a[2])*float(b[2])
