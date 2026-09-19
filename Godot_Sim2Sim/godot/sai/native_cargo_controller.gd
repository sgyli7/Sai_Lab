extends RefCounted
## In-process port of CargoGodotController and legacy_crawl.CrawlTransport.

const LEG_INDICES := [0,1,2,4,5,6,8,9,10,12,13,14]
const WHEEL_INDICES := [3,7,11,15]
const SIDES := [1.0,-1.0,1.0,-1.0]

var motion
var path
var specification:Dictionary
var grasp_end:=0.0
var clamp_end:=0.0
var end:=0.0
var target:=0.0
var target_leg:Array=[]
var target_arm:Array=[]
var crawl_start:Variant=null
var crawl_arm_hold:Array=[]
var crawl_start_base:Array=[]
var crawl_previous:Array=[]
var layers:Array
var policy_sha256:=""
var flat_policy_sha256:=""
var stair_profile_id:=""

func _init(base_controller,spec:Dictionary)->void:
	motion=base_controller;specification=spec
	flat_policy_sha256=motion.flat_policy_sha256;stair_profile_id=motion.stair_profile_id
	path=preload("res://sai/native_grab_controller.gd").new(motion,specification)
	grasp_end=path.end_time;clamp_end=grasp_end+8.0;end=clamp_end+22.0
	layers=JSON.parse_string(FileAccess.get_file_as_string("res://sai_policy/legacy-crawl57.json")).layers
	var hash:=HashingContext.new();hash.start(HashingContext.HASH_SHA256)
	hash.update(FileAccess.get_file_as_bytes("res://sai_policy/legacy-crawl57.json"));policy_sha256=hash.finish().hex_encode()
	crawl_previous.resize(16);crawl_previous.fill(0.0)

func reset()->void:
	motion.reset();crawl_start=null;target=0.0;crawl_previous.fill(0.0)

func command(state:Dictionary)->Dictionary:
	var now:=float(state.time);var result:Dictionary
	if now+1e-6<grasp_end:
		result=_grasp(state)
	elif now+1e-6<clamp_end:
		target=minf(.067,maxf(0.0,now-grasp_end)*.01)/float(specification.cargo.drive_metres_per_radian)
		result={"mode":"securing","stage":"secure_cargo","target_leg":target_leg,"target_arm":target_arm}
	elif crawl_start==null and not (bool(state.cargo_inside) and bool(state.cargo_bilateral)):
		end=now;result={"mode":"abort","stage":"cargo_not_secured","target_leg":target_leg,"target_arm":target_arm}
	else:
		result=_crawl(state)
	result.merge({"arm_bias":state.arm_gravity_bias,"grip_cap":1.4,"end":end,
		"cargo_target_rad":target,"physics_advanced_by_controller":false,
		"transport_required":crawl_start!=null},true)
	return result

func _grasp(state:Dictionary)->Dictionary:
	var p:Dictionary=path._path(float(state.time));var q:Array=p.q
	var point:Vector3=path._tool_point(q,float(p.drop))
	target_arm=path._ik(point,state,q)
	target_leg=path._leg_reference(float(p.drop))
	return {"mode":"manipulation","stage":p.label,"target_leg":target_leg,"target_arm":target_arm}

func _crawl(state:Dictionary)->Dictionary:
	if crawl_start==null:
		crawl_start=float(state.time);crawl_arm_hold=state.q.slice(16,22);crawl_start_base=state.base_position.duplicate()
	var t:float=float(state.time)-float(crawl_start)
	var speed:float=.12*minf(1.0,maxf(0.0,t))*minf(1.0,maxf(0.0,22.0-2.0-t))
	var columns:Array=state.base_rotation_columns
	var yaw:float=atan2(float(columns[0][1]),float(columns[0][0]))
	var yaw_command:float=clampf(-1.5*yaw-.5*(float(state.base_position[1])-float(crawl_start_base[1])),-.18,.18)
	var phase:float=TAU*maxf(0.0,t-.3)/2.4
	var obs:Array=[]
	for i in range(3):obs.append(-float(columns[i][2]))
	for vector in [state.base_linear_world,state.base_angular_world]:
		for i in range(3):
			obs.append(float(columns[i][0])*float(vector[0])+float(columns[i][1])*float(vector[1])+float(columns[i][2])*float(vector[2]))
	obs.append(speed);obs.append(yaw_command)
	for i in LEG_INDICES:obs.append(float(state.q[i]))
	for i in LEG_INDICES:obs.append(float(state.v[i])*.1)
	for j in range(4):obs.append(float(state.v[WHEEL_INDICES[j]])*SIDES[j]*.1)
	obs.append_array(crawl_previous)
	obs.append(sin(phase));obs.append(cos(phase))
	var action:Array=_infer(obs)
	for i in range(16):action[i]=clampf(float(action[i]),-1.0,1.0)
	crawl_previous=action.duplicate()
	var reference:Array=_crawl_reference(t,speed);var target_values:Array=[]
	for i in range(16):target_values.append(float(reference[i])+.18*float(action[i]))
	var wheels:Array=[]
	for leg in range(4):wheels.append(SIDES[leg]*((speed-yaw_command*SIDES[leg]*.146)/.048+6.0*float(action[WHEEL_INDICES[leg]])))
	return {"mode":"transport","stage":"loaded_crawl" if speed>.001 else "loaded_settle",
		"target_leg":target_values,"target_arm":crawl_arm_hold,"wheel_speed":wheels,
		"policy_action":action,"policy_observation":obs,"policy_sha256":policy_sha256,
		"transport_distance_m":float(state.base_position[0])-float(crawl_start_base[0]),
		"speed_command_mps":speed,"yaw_command_rads":yaw_command}

func _infer(input:Array)->Array:
	var values:Array=input.duplicate()
	for layer in layers:
		if layer.kind=="tanh":
			for i in range(values.size()):values[i]=tanh(float(values[i]))
		else:
			var next:Array=[]
			for row in range(layer.bias.size()):
				var value:=float(layer.bias[row])
				for column in range(values.size()):value+=float(layer.weight[row][column])*float(values[column])
				next.append(value)
			values=next
	return values

func _crawl_reference(t:float,speed:float)->Array:
	var result:Array=[];result.resize(16);result.fill(0.0)
	if t<.3-1e-8 or absf(speed)<.01:return result
	var sample:float=floor((t+1e-8)*50.0)/50.0;var cycle:float=(maxf(0.0,sample-.3)+.01)/2.4
	var offsets:=[0.0,.5,.75,.25]
	for leg in range(4):
		var front:float=1.0 if leg<2 else -1.0;var side:float=float(SIDES[leg]);var s:float=fposmod(cycle-float(offsets[leg]),1.0)
		var height:=.045*pow(sin(PI*s/.25),2.0) if s<.25 else 0.0
		var dx:=-.03*cos(PI*s/.25) if s<.25 else .06*(.5-(s-.25)/.75)
		var down:=.172812737-height
		var beta:float=-front*acos(clampf((down*down+dx*dx-.09*.09-.11*.11)/(2.0*.09*.11),-1.0,1.0))
		var theta:=atan2(dx,down)-atan2(.11*sin(beta),.09+.11*cos(beta))
		var theta0:=front*atan2(.05,.074833147);var beta0:=-front*(atan2(.05,.09797959)+atan2(.05,.074833147))
		result[leg*4+1]=side*(theta0-theta);result[leg*4+2]=side*(beta0-beta)
	return result
