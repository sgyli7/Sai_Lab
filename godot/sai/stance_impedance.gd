extends RefCounted
## Native form of sai_compliance.StanceImpedance.
## The gains and state machine are identical; kinematics and gravity moments
## come from the live Jolt articulation instead of a parallel MuJoCo model.

const LEG_AXES := [0,1,2,4,5,6,8,9,10,12,13,14]
const FLEX_AXES := [1,2,5,6,9,10,13,14]
const STAND_HEIGHT := 0.2192
const CROUCH_DROP := 0.035

var parameters: Array
var turn_blend := 0.25
var turn_support := 0.0
var height_reference: Variant = null
var stance: Array = []
var previous_feedforward: Array = []

func _init(profile: Dictionary) -> void:
	parameters = profile.parameters.duplicate()
	turn_blend = float(profile.get("turn_support_blend",.25))
	previous_feedforward.resize(16)
	previous_feedforward.fill(0.0)

func reset() -> void:
	turn_support = 0.0
	height_reference = null
	stance.clear()
	previous_feedforward.fill(0.0)

func apply(result: Dictionary,state: Dictionary) -> Dictionary:
	if state.get("wheel_ground_heights",[]).size()!=4:return result
	if state.get("wheel_positions",[]).size()!=4:return result
	if state.get("leg_support_jacobian",[]).size()!=4:return result
	if state.get("leg_gravity_bias",[]).size()!=16:return result
	var kp:=float(parameters[0]);var kd:=float(parameters[1]);var gravity:=float(parameters[2])
	var heave:=float(parameters[3]);var damper:=float(parameters[4]);var tau:=float(parameters[5])
	var ground:Array=state.wheel_ground_heights
	var wheels:Array=state.wheel_positions
	var contact:Array=[]
	for i in range(4):
		var gap:=float(wheels[i][2])-float(ground[i])-.048
		contact.append(clampf(1.0-(gap-.002)/.01,0.0,1.0))
	if stance.is_empty():stance=contact.duplicate()
	for i in range(4):stance[i]+=clampf(float(contact[i])-float(stance[i]),-.25,.25)
	contact=stance.duplicate()
	var gains:Array=[];var damping:Array=[];var feedforward:Array=[]
	gains.resize(16);damping.resize(16);feedforward.resize(16)
	gains.fill(80.0);damping.fill(2.0);feedforward.fill(0.0)
	for joint in FLEX_AXES:
		var leg:int=joint/4
		gains[joint]=80.0+(kp-80.0)*float(contact[leg])
		damping[joint]=2.0+(kd-2.0)*float(contact[leg])
	var desired_height:=_mean(ground)+STAND_HEIGHT-CROUCH_DROP*float(result.effective_crouch)
	if height_reference==null:height_reference=desired_height
	var delta:float=(desired_height-float(height_reference))*.02/(tau+.02)
	height_reference=float(height_reference)+delta
	var mass:=float(state.get("robot_mass",0.0))
	var force:=mass*9.81*gravity+clampf(heave*(float(height_reference)-float(state.base_position[2]))+
		damper*(delta/.02-float(state.base_linear_world[2])),-mass*9.81*.5,mass*9.81*.5)
	force=clampf(force,0.0,mass*9.81*1.5)
	var com:Array=state.get("robot_com_position",state.base_position)
	var xy:Array=[]
	for wheel in wheels:xy.append([float(wheel[0])-float(com[0]),float(wheel[1])-float(com[1])])
	var weights:=_weights(contact,xy)
	for leg in range(4):
		var jac:Array=state.leg_support_jacobian[leg]
		for offset in range(3):
			var joint:int=leg*4+offset
			feedforward[joint]-=float(jac[joint])*force*float(weights[leg])
	var bias:Array=state.leg_gravity_bias
	for joint in LEG_AXES:feedforward[joint]+=gravity*float(bias[joint])*float(contact[joint/4])
	for i in range(16):
		previous_feedforward[i]+=(float(feedforward[i])-float(previous_feedforward[i]))*(.02/.06)
		feedforward[i]=previous_feedforward[i]
	var goal:=turn_blend*minf(1.0,absf(float(state.command[1]))/.35)
	turn_support+=(goal-turn_support)*(.02/.12)
	if turn_support>0.0:
		for i in range(16):
			gains[i]=(1.0-turn_support)*float(gains[i])+80.0*turn_support
			damping[i]=(1.0-turn_support)*float(damping[i])+2.0*turn_support
			feedforward[i]=(1.0-turn_support)*float(feedforward[i])
	result["leg_kp"]=gains;result["leg_kd"]=damping;result["leg_feedforward"]=feedforward
	result["impedance_contract"]="sai-joint-impedance-v1"
	result["support_force_N"]=force;result["stance_weights"]=weights
	return result

func _weights(contact:Array,xy:Array)->Array:
	var active:=[false,false,false,false]
	for i in range(4):active[i]=float(contact[i])>1e-4
	var weights:=[0.0,0.0,0.0,0.0]
	for _attempt in range(4):
		var m00:=0.0;var m01:=0.0;var m02:=0.0;var m11:=0.0;var m12:=0.0;var m22:=0.0
		for i in range(4):
			if not active[i]:continue
			var c:=float(contact[i]);var x:=float(xy[i][0]);var y:=float(xy[i][1])
			m00+=c;m01+=c*x;m02+=c*y;m11+=c*x*x;m12+=c*x*y;m22+=c*y*y
		# Same damped least-squares system as Python training and CPU MuJoCo.
		# It remains solvable when only one axle supports the chassis at an edge.
		m00+=1e-8;m11+=1e-6;m22+=1e-6
		var matrix:=Basis(Vector3(m00,m01,m02),Vector3(m01,m11,m12),Vector3(m02,m12,m22))
		var solution:=matrix.inverse()*Vector3(1,0,0)
		var lowest:=0.0;var lowest_index:=-1
		for i in range(4):
			if not active[i]:continue
			var value:=float(contact[i])*(solution.x+float(xy[i][0])*solution.y+float(xy[i][1])*solution.z)
			weights[i]=value
			if value<lowest:lowest=value;lowest_index=i
		if lowest_index<0:
			for i in range(4):weights[i]=maxf(0.0,float(weights[i]))
			break
		active[lowest_index]=false;weights[lowest_index]=0.0
	var total:=_mean_sum(weights)
	var contact_sum:=_mean_sum(contact)
	if total>0.0:
		var scale:=minf(1.0,contact_sum)/total
		for i in range(4):weights[i]=float(weights[i])*scale
	return weights

func _mean(values:Array)->float:
	return _mean_sum(values)/float(values.size())

func _mean_sum(values:Array)->float:
	var result:=0.0
	for value in values:result+=float(value)
	return result
