extends Node
## Acceptance-only navigator: sends normal key events, never writes a body transform.
var hub:Node3D
var points:Array=[]
var index:=0
var held:Array=[]
var reached:Array=[]
var minimum_upright:=1.
var done:=false
var next_report:=0.
var turning:=true
var settle_until:=2.
var turn_key:=0
func _physics_process(_dt:float) -> void:
	if done or hub.switching or hub._base==null:return
	if hub.elapsed<2.:return
	var p:Vector3=hub._base.global_position
	if hub.elapsed>=next_report:
		next_report=hub.elapsed+10.;print("ROUTE_PROGRESS ",hub.elapsed," ",p," target=",points[index])
	minimum_upright=minf(minimum_upright,hub._base.global_basis.y.y) if hub.active_robot=="sai" else minimum_upright
	var target:=Vector3(points[index][0],0,points[index][1])
	var delta:=target-Vector3(p.x,0,p.z)
	if delta.length()<.24:
		reached.append({"target":points[index],"position":[p.x,p.y,p.z],"time":hub.elapsed})
		index+=1
		if index==points.size():
			_keys([]);done=true;hub.route_result={"passed":minimum_upright>.5,"reached":reached,"minimum_upright":minimum_upright}
			hub._finish.call_deferred();return
		target=Vector3(points[index][0],0,points[index][1]);delta=target-Vector3(p.x,0,p.z)
	var yaw:float
	if hub.active_robot=="sai":
		var direction:Vector3=hub._base.global_basis.x;yaw=atan2(-direction.z,direction.x)
	else:
		var body:Dictionary=hub.actor.Contract.body_state(hub.actor.local_reply,hub.actor.robot_config)
		var r:Array=hub.actor.Contract.quat_matrix(body.base_quat);yaw=atan2(r[1][0],r[0][0])
		minimum_upright=minf(minimum_upright,r[2][2])
	if minimum_upright<.5:
		_keys([]);done=true;hub.route_result={"passed":false,"reached":reached,"minimum_upright":minimum_upright,"reason":"robot fell"}
		hub._finish.call_deferred();return
	var error:float=wrapf(atan2(-delta.z,delta.x)-yaw,-PI,PI)
	var keys:Array=[]
	if hub.active_robot=="roller":
		# The wheel policy keeps coasting after key release. Brake the turn
		# before its heading crosses the target; keep driving through small
		# balance oscillations instead of repeatedly stopping on the incline.
		var rate:float=hub.actor.local_reply.get("base_angvel_local",[0.,0.,0.])[2]
		var predicted:float=wrapf(error-rate*.25,-PI,PI)
		if absf(error)<.70:keys.append(KEY_W)
		if absf(predicted)>.20:keys.append(KEY_A if predicted>0 else KEY_D)
		_keys(keys)
		hub.route_result={"passed":false,"reached":reached,"next":points[index],"position":[p.x,p.y,p.z]}
		return
	# Release movement before changing direction, just as a cautious player does.
	if hub.elapsed>=settle_until:
		if turning:
			if absf(error)<.10:turning=false;turn_key=0;settle_until=hub.elapsed+.45
			else:
				if turn_key==0:turn_key=KEY_A if error>0 else KEY_D
				keys.append(turn_key)
		elif absf(error)>.28:
			turning=true;turn_key=KEY_A if error>0 else KEY_D;settle_until=hub.elapsed+.45
		else:keys.append(KEY_W)
	_keys(keys)
	hub.route_result={"passed":false,"reached":reached,"next":points[index],"position":[p.x,p.y,p.z]}
func _keys(keys:Array) -> void:
	for key in [KEY_W,KEY_A,KEY_D]:
		if keys.has(key)!=held.has(key):
			var event:=InputEventKey.new();event.keycode=key;event.physical_keycode=key;event.pressed=keys.has(key);Input.parse_input_event(event)
	held=keys
