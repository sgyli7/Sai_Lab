extends RefCounted
## Native port of sim2sim.sai_suspension.Suspension.

const SIDES := [1.0,-1.0,1.0,-1.0]
const FRONTS := [1.0,1.0,-1.0,-1.0]
const CROUCH_DROP := 0.035

var parameters:Array
var offset:=[0.0,0.0,0.0,0.0]
var apply_on_stairs:=false

func _init(profile:Dictionary,enable_on_stairs:=false)->void:
	parameters=profile.geometry_parameters.duplicate()
	apply_on_stairs=enable_on_stairs

func reset()->void:
	offset=[0.0,0.0,0.0,0.0]

func apply(result:Dictionary,state:Dictionary)->Dictionary:
	if state.get("wheel_ground_heights",[]).size()!=4 or (result.stage=="stairs" and not apply_on_stairs):
		reset();return result
	var heights:Array=state.wheel_ground_heights
	var mean:=0.0
	for value in heights:mean+=float(value)
	mean/=4.0
	var columns:Array=state.base_rotation_columns
	var xy:=[[.115,.146],[.115,-.146],[-.115,.146],[-.115,-.146]]
	var desired:Array=[]
	var low:=INF;var high:=-INF
	for value in heights:low=minf(low,float(value));high=maxf(high,float(value))
	var blend:=clampf((high-low)/.004,0.0,1.0)
	for i in range(4):
		var tilt:=float(xy[i][0])*float(columns[0][2])+float(xy[i][1])*float(columns[1][2])
		var value:=(-float(parameters[0])*(float(heights[i])-mean)+float(parameters[1])*tilt)*blend
		desired.append(clampf(value,-.025,.020))
		var change:float=(float(desired[i])-float(offset[i]))*(.02/(float(parameters[2])+.02))
		offset[i]+=clampf(change,-.003,.003)
	var nominal:=.172812737-CROUCH_DROP*float(result.effective_crouch)
	var target:Array=result.target_leg.duplicate()
	for leg in range(4):
		var corrected:=clampf(nominal+float(offset[leg]),.12,.195)
		var base_angles:=_angles(nominal,leg)
		var corrected_angles:=_angles(corrected,leg)
		target[leg*4+1]+=float(corrected_angles[0])-float(base_angles[0])
		target[leg*4+2]+=float(corrected_angles[1])-float(base_angles[1])
	result["target_leg"]=target
	result["suspension_offset_m"]=offset.duplicate()
	result["suspension_parameters"]=parameters.duplicate()
	return result

func _angles(down:float,leg:int)->Array:
	var beta:float=-FRONTS[leg]*acos(clampf((down*down-.09*.09-.11*.11)/(2.0*.09*.11),-1.0,1.0))
	var theta:float=-atan2(.11*sin(beta),.09+.11*cos(beta))
	return [-SIDES[leg]*theta,-SIDES[leg]*beta]
