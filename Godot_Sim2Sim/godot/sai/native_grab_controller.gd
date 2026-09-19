extends RefCounted
## In-process port of sim2sim.workshop_grab.WorkshopController.
## It retains the released path, docking, point-priority IK, grip timing,
## cancellation, and cargo-slot semantics without a Python/MuJoCo process.

var motion
var specification:Dictionary
var frames:Array
var axes:Array
var home:Array
var arm_ranges:Array=[]
var grab_serial:=-1
var phase:="idle"
var started:=0.0
var waited:=0.0
var last_time:=0.0
var anchor_rotation:Array
var anchor_translation:Vector3
var pick_delta:Vector3
var pick_height:=0.0
var wheel_origin:Array=[]
var arm_hold:Array=[0.0,0.0,0.0,0.0,0.0,0.0]
var cancel_start:Variant=null
var end_time:=0.0

func _init(base_controller,spec:Dictionary)->void:
	motion=base_controller
	specification=spec
	frames=JSON.parse_string(FileAccess.get_file_as_string("res://sai_policy/pick-place.json"))
	axes=JSON.parse_string(FileAccess.get_file_as_string("res://sai_policy/so101-axes.json"))
	home=specification.arm_home_source_deg.duplicate()
	for name in ["arm_pan_carrier","arm_upper","arm_lower","arm_wrist","arm_gripper","arm_moving_jaw"]:
		arm_ranges.append(specification.bodies[name].joint.range_rad.duplicate())
	end_time=1.5+float(frames.size()-1)*.22+2.0

func reset()->void:
	motion.reset();grab_serial=-1;phase="idle";cancel_start=null

func command(state:Dictionary)->Dictionary:
	var request:Dictionary=state.get("workshop_grab",{})
	var now:=float(state.time)
	if int(request.get("serial",-1))!=grab_serial:
		grab_serial=int(request.get("serial",-1))
		if request.get("request","")=="pick":
			phase="docking";started=now;cancel_start=null
		elif request.get("request","")=="cancel":
			phase="idle";cancel_start=now;arm_hold=state.q.slice(16,22)
	if phase=="docking":
		var target:=_v(request.target_m)
		var delta:=_transpose_mul(state.base_rotation_columns,target-_v(state.base_position))
		var distance:=Vector2(delta.x,delta.y).length()
		var angle:=atan2(delta.y,delta.x)
		if now-started>24.0:
			phase="idle"
			var failed:=_drive(state,[0.0,0.0,0.0])
			failed.merge({"grab_stage":"failed","grab_message":"靠近超时 · 请驾驶到物件前方再重试"},true)
			return failed
		if distance>=.20 and distance<=.27 and absf(angle)<.10:
			if Vector2(float(state.base_linear_world[0]),float(state.base_linear_world[1])).length()>.04:
				var stopping:=_drive(state,[0.0,0.0,0.0])
				stopping.merge({"grab_stage":"docking","grab_message":"正在停稳底盘"},true)
				return stopping
			phase="manipulation";started=now;waited=0.0;last_time=now
			anchor_rotation=state.base_rotation_columns.duplicate(true)
			anchor_translation=_v(state.base_position)-_mul(anchor_rotation,_v(specification.bodies.chassis.origin_m))
			pick_delta=_transpose_mul(anchor_rotation,target-anchor_translation)-Vector3(.24,0.0,.020373)
			pick_height=float(request.target_m[2]);wheel_origin=[]
			for i in [3,7,11,15]:wheel_origin.append(float(state.q[i]))
			arm_hold=state.q.slice(16,22)
		else:
			var speed:=clampf((distance-.24)*.7,-.06,.12) if absf(angle)<.3 else 0.0
			var docking:=_drive(state,[speed,clampf(angle*1.4,-.45,.45),0.0])
			docking.merge({"grab_stage":"docking","grab_message":"正在靠近所选物件"},true)
			return docking
	if phase=="manipulation":return _manipulate(state,request)
	var result:Dictionary=motion.command(state)
	if cancel_start!=null:
		var alpha:=clampf((now-float(cancel_start))/3.0,0.0,1.0)
		var target_arm:Array=[]
		for value in arm_hold:target_arm.append((1.0-alpha)*float(value))
		result.target_arm=target_arm
		if alpha>=1.0:cancel_start=null
	return result

func _drive(state:Dictionary,input:Array)->Dictionary:
	var copy:=state.duplicate(true);copy.command=input
	return motion.command(copy)

func _manipulate(state:Dictionary,request:Dictionary)->Dictionary:
	var now:=float(state.time)
	var elapsed:=now-started-waited
	if elapsed>=10.8 and elapsed<12.0 and not bool(request.get("held",false)):
		waited+=now-last_time;elapsed=minf(elapsed,10.8)
		if waited>4.0:
			phase="idle";cancel_start=now;arm_hold=state.q.slice(16,22)
			var failed:=_drive(state,[0.0,0.0,0.0])
			failed.merge({"grab_stage":"failed","grab_message":"夹爪未够到物件 · 请换个位置重试"},true)
			return failed
	last_time=now
	var path:=_path(elapsed)
	var q:Array=path.q
	var drop:=float(path.drop)
	if bool(request.get("held",false)) and elapsed<34.0:q[5]=74.0
	else:q[5]=60.0
	var point:=_tool_point(q,drop)
	var blend:=clampf((elapsed-15.8)/(29.0-15.8),0.0,1.0)
	blend=blend*blend*(3.0-2.0*blend)
	var release_height:=.259+float(request.rest_height_m)+.001
	var destination_delta:=Vector3(-.030,-.04 if int(request.slot)==0 else .04,release_height-.27923)
	var rotation:Array=state.base_rotation_columns
	var current_translation:=_v(state.base_position)-_mul(rotation,_v(specification.bodies.chassis.origin_m))
	var pickup_point:=_mul(anchor_rotation,point+pick_delta)+anchor_translation
	if bool(request.get("held",false)):
		pickup_point.z=maxf(pickup_point.z,pick_height-float(request.held_offset_m[2]))
	var cargo_point:=_mul(rotation,point+destination_delta)+current_translation
	point=pickup_point.lerp(cargo_point,blend)
	if bool(request.get("held",false)):point-=blend*_v(request.held_offset_m)
	var target_arm:=_ik(point,state,q)
	var release_point:=_mul(rotation,Vector3(-.095,-.04 if int(request.slot)==0 else .04,release_height))+current_translation
	var ready_to_release:=_v(request.target_m).distance_to(release_point)<.025
	var target_leg:=_leg_reference(drop)
	for leg in range(4):target_leg[leg*4+3]+=float(wheel_origin[leg])
	var result:={"mode":"manipulation","stage":path.label,"target_leg":target_leg,
		"target_arm":target_arm,"arm_bias":state.arm_gravity_bias,"grip_cap":1.4,"cargo_target_rad":0.0,
		"physics_advanced_by_controller":false,"grab_stage":"manipulation",
		"assist_grip":elapsed>=6.0 and elapsed<12.0,
		"assist_release":elapsed>=34.5 and ready_to_release,"grab_elapsed":elapsed,"tool_target_m":[point.x,point.y,point.z]}
	if elapsed>=end_time:
		result.grab_stage="complete";phase="idle";cancel_start=now;arm_hold=target_arm.duplicate()
	return result

func _path(time_s:float)->Dictionary:
	var u:=clampf((time_s-1.5)/.22,0.0,float(frames.size()-1))
	var index:=mini(int(u),frames.size()-2);var alpha:=u-float(index)
	var a:Dictionary=frames[index];var b:Dictionary=frames[index+1];var q:Array=[]
	for i in range(6):q.append(lerpf(float(a.q[i]),float(b.q[i]),alpha))
	var drop:=lerpf(float(a.drop_mm),float(b.drop_mm),alpha)
	if str(a.label) in ["close","lift","raise_body","front_clearance","transfer_1","transfer_2","transfer_3","transfer_4","over_cargo","place"]:
		q[5]+=clampf((float(q[5])-75.0)/3.6468,0.0,1.0)
	return {"q":q,"drop":drop,"label":str(b.label)}

func _leg_reference(drop_mm:float)->Array:
	var result:Array=[];result.resize(16);result.fill(0.0)
	for leg in range(4):
		var front:=1.0 if leg<2 else -1.0;var side:=1.0 if leg%2==0 else -1.0
		var down:=.172812737-drop_mm*.001
		var beta:float=-front*acos(clampf((down*down-.09*.09-.11*.11)/(2.0*.09*.11),-1.0,1.0))
		var theta:float=-atan2(.11*sin(beta),.09+.11*cos(beta))
		var theta0:=front*atan2(.05,.074833147)
		var beta0:=-front*(atan2(.05,.09797959)+atan2(.05,.074833147))
		result[leg*4+1]=side*(theta0-theta);result[leg*4+2]=side*(beta0-beta)
		result[leg*4+3]=-float(result[leg*4+1])-float(result[leg*4+2])
	return result

func _ik(world_target:Vector3,state:Dictionary,path_q:Array)->Array:
	var q:Array=[]
	for i in range(6):q.append(deg_to_rad(float(path_q[i])-float(home[i])))
	var rotation:Array=state.base_rotation_columns
	var translation:=_v(state.base_position)-_mul(rotation,_v(specification.bodies.chassis.origin_m))
	var local_target:=_transpose_mul(rotation,world_target-translation)
	for _iteration in range(40):
		var current:=_tool_from_joints(q);var error:=local_target-current
		if error.length()<.00005:break
		var jac:Array=[];var epsilon:=.0001
		for joint in range(5):
			var probe:=q.duplicate();probe[joint]=float(probe[joint])+epsilon
			jac.append((_tool_from_joints(probe)-current)/epsilon)
		var m:=Basis.IDENTITY.scaled(Vector3(1e-6,1e-6,1e-6))
		for column in jac:
			m.x+=Vector3(column.x*column.x,column.y*column.x,column.z*column.x)
			m.y+=Vector3(column.x*column.y,column.y*column.y,column.z*column.y)
			m.z+=Vector3(column.x*column.z,column.y*column.z,column.z*column.z)
		var solution:=m.inverse()*error
		for joint in range(5):
			var change:=clampf(jac[joint].dot(solution),-.10,.10)
			q[joint]=clampf(float(q[joint])+change,float(arm_ranges[joint][0])+.001,float(arm_ranges[joint][1])-.001)
	q[5]=clampf(q[5],float(arm_ranges[5][0])+.001,float(arm_ranges[5][1])-.001)
	return q

func _tool_from_joints(joints:Array)->Vector3:
	var degrees:Array=[]
	for i in range(6):degrees.append(float(home[i])+rad_to_deg(float(joints[i])))
	return _tool_point(degrees,0.0)

func _tool_point(degrees:Array,drop_mm:float)->Vector3:
	var root:=Transform3D(Basis.IDENTITY,Vector3(.083,0.0,.2566057044657792-drop_mm*.001))
	var chain:=Transform3D.IDENTITY
	for i in range(5):
		var axis:=_v(axes[i].direction).normalized();var pivot:=_v(axes[i].point_mm)*.001
		var basis:=Basis(axis,deg_to_rad(float(degrees[i])))
		chain=chain*Transform3D(basis,pivot-basis*pivot)
	var fixed:=Vector3(.006981563026764376,-.22018122292575873,.14328642212881962)
	var normal:=Vector3(-.01745207628636318,-.008801353052346078,.9998089623611818)
	return root*chain*(fixed+normal*.015)

func _mul(columns:Array,value:Vector3)->Vector3:
	return _v(columns[0])*value.x+_v(columns[1])*value.y+_v(columns[2])*value.z

func _transpose_mul(columns:Array,value:Vector3)->Vector3:
	return Vector3(_v(columns[0]).dot(value),_v(columns[1]).dot(value),_v(columns[2]).dot(value))

func _v(value)->Vector3:
	return Vector3(float(value[0]),float(value[1]),float(value[2]))
