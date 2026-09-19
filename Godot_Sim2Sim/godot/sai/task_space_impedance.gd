extends RefCounted
## Native low-level impedance for the v6 task-space skill contract.

const SIDES := [1.0,-1.0,1.0,-1.0]
const FRONTS := [1.0,1.0,-1.0,-1.0]
const LEG_AXES := [0,1,2,4,5,6,8,9,10,12,13,14]
const FLEX_AXES := [1,2,5,6,9,10,13,14]

var stance:Array=[]
var height_reference:Variant=null
var previous_feedforward:Array=[]

func _init()->void:
	previous_feedforward.resize(16);previous_feedforward.fill(0.0)

func reset()->void:
	stance.clear();height_reference=null;previous_feedforward.fill(0.0)

func apply(result:Dictionary,state:Dictionary)->Dictionary:
	var ground:Array=state.get("wheel_ground_heights",[])
	var wheels:Array=state.get("wheel_positions",[])
	if ground.size()!=4 or wheels.size()!=4:return result
	var contact:Array=[]
	for i in range(4):
		var gap:=float(wheels[i][2])-float(ground[i])-.048
		contact.append(clampf(1.0-(gap-.002)/.01,0.0,1.0))
	if stance.is_empty():stance=[1.0,1.0,1.0,1.0]
	for i in range(4):stance[i]+=clampf(float(contact[i])-float(stance[i]),-.25,.25)
	contact=stance.duplicate()
	var gains:Array=[];var damping:Array=[];var feedforward:Array=[]
	gains.resize(16);damping.resize(16);feedforward.resize(16)
	gains.fill(80.0);damping.fill(2.0);feedforward.fill(0.0)
	for joint in FLEX_AXES:
		var leg:int=joint/4
		gains[joint]=80.0+(45.05744684106064-80.0)*float(contact[leg])
		damping[joint]=2.0+(1.1532485502878331-2.0)*float(contact[leg])
	var desired_height:=_mean(ground)+.2192
	if height_reference==null:height_reference=desired_height
	var delta:float=(desired_height-float(height_reference))*(.02/(.15023543636516334+.02))
	height_reference=float(height_reference)+delta
	var mass:=float(state.get("robot_mass",0.0))
	var force:=mass*9.81*.9497251199685409+473.84859697921144*(float(height_reference)-float(state.base_position[2]))+49.991230469877635*(delta/.02-float(state.base_linear_world[2]))
	force=clampf(force,0.0,mass*9.81*1.5)
	var total:=maxf(_sum(contact),1e-5)
	var q:Array=state.q
	var theta0:=atan2(.05,.074833147)
	var beta0:=atan2(.05,.09797959)+theta0
	for leg in range(4):
		var side:float=SIDES[leg];var front:float=FRONTS[leg]
		var theta:=front*theta0-float(q[leg*4+1])/side
		var beta:=-front*beta0-float(q[leg*4+2])/side
		var dx:=.09*sin(theta)+.11*sin(theta+beta)
		var weight:=float(contact[leg])/total
		feedforward[leg*4+1]=-(-dx/side)*force*weight
		feedforward[leg*4+2]=-(-.11*sin(theta+beta)/side)*force*weight
	var bias:Array=state.get("leg_gravity_bias",[])
	if bias.size()==16:
		for joint in LEG_AXES:feedforward[joint]+=.9497251199685409*float(bias[joint])*float(contact[joint/4])
	for i in range(16):
		previous_feedforward[i]+=(float(feedforward[i])-float(previous_feedforward[i]))*(.02/.06)
		feedforward[i]=previous_feedforward[i]
	result["leg_kp"]=gains;result["leg_kd"]=damping;result["leg_feedforward"]=feedforward
	result["impedance_contract"]="sai-joint-impedance-v1"
	result["support_force_N"]=force
	var weights:Array=[]
	for value in contact:weights.append(float(value)/total)
	result["stance_weights"]=weights
	return result

func _sum(values:Array)->float:
	var result:=0.0
	for value in values:result+=float(value)
	return result

func _mean(values:Array)->float:
	return _sum(values)/float(values.size())
