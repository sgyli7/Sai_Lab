extends RefCounted
## Float64 coordinate arithmetic, followed by the original float32 observation contract.
## Arrays deliberately avoid Godot's float32 Vector3/Basis for intermediate rotations.

static func quat_matrix(q: Array) -> Array:
	var w: float = q[0]
	var x: float = q[1]
	var y: float = q[2]
	var z: float = q[3]
	return [[1.0-2.0*(y*y+z*z),2.0*(x*y-z*w),2.0*(x*z+y*w)],
		[2.0*(x*y+z*w),1.0-2.0*(x*x+z*z),2.0*(y*z-x*w)],
		[2.0*(x*z-y*w),2.0*(y*z+x*w),1.0-2.0*(x*x+y*y)]]

static func matrix_quat(r: Array) -> Array:
	var t: float = r[0][0]+r[1][1]+r[2][2]
	var q: Array
	if t > 0.0:
		var s := 0.5/sqrt(t+1.0)
		q = [0.25/s,(r[2][1]-r[1][2])*s,(r[0][2]-r[2][0])*s,(r[1][0]-r[0][1])*s]
	elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
		var s := 2.0*sqrt(1.0+r[0][0]-r[1][1]-r[2][2])
		q = [(r[2][1]-r[1][2])/s,0.25*s,(r[0][1]+r[1][0])/s,(r[0][2]+r[2][0])/s]
	elif r[1][1] > r[2][2]:
		var s := 2.0*sqrt(1.0+r[1][1]-r[0][0]-r[2][2])
		q = [(r[0][2]-r[2][0])/s,(r[0][1]+r[1][0])/s,0.25*s,(r[1][2]+r[2][1])/s]
	else:
		var s := 2.0*sqrt(1.0+r[2][2]-r[0][0]-r[1][1])
		q = [(r[1][0]-r[0][1])/s,(r[0][2]+r[2][0])/s,(r[1][2]+r[2][1])/s,0.25*s]
	var norm := sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3])
	if norm < 1e-12:
		return [1.0,0.0,0.0,0.0]
	var factor := (-1.0 if q[0] < 0.0 else 1.0)/norm
	for i in range(4): q[i] *= factor
	return q

static func body_state(raw: Dictionary, robot: Dictionary) -> Dictionary:
	var ri := quat_matrix(raw.base_quat)
	var rq := quat_matrix(robot.iquat)
	var rb := [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
	var pos: Array = raw.base_pos.duplicate()
	for i in range(3):
		for j in range(3):
			for k in range(3): rb[i][j] += ri[i][k]*rq[j][k]
		for j in range(3): pos[i] -= rb[i][j]*robot.ipos[j]
	var result := {"base_pos":pos,"base_quat":matrix_quat(rb)}
	# Optional outer-loop feedback uses the same world COM velocity reported to
	# Python. A declared residual-state actor can also consume this telemetry.
	if raw.has("base_linvel"): result.base_linvel=raw.base_linvel
	return result

static func observation(raw: Dictionary, body: Dictionary, last: PackedFloat32Array,
		command: PackedFloat32Array, home: PackedFloat32Array) -> PackedFloat32Array:
	var obs := PackedFloat32Array()
	obs.resize(61)
	var q: Array = body.base_quat
	var w: float = q[0]
	var x: float = q[1]
	var y: float = q[2]
	var z: float = q[3]
	# v - w*(2*cross(xyz,v)) + cross(xyz,2*cross(xyz,v)), v=(0,0,-1).
	obs[3] = 2.0*w*y - 2.0*z*x
	obs[4] = -2.0*w*x - 2.0*z*y
	obs[5] = -1.0 + 2.0*(x*x+y*y)
	for i in range(3): obs[i] = raw.base_angvel_local[i]
	var joint_q := PackedFloat32Array(raw.q)
	for i in range(14):
		obs[6+i] = joint_q[i]-home[i]
		obs[20+i] = raw.qd[i]
		obs[34+i] = last[i]
	for i in range(13): obs[48+i] = command[i]
	return obs

static func time_command(elapsed: float, duration: float, body: Dictionary,
		heading_input: bool, heading: Array) -> PackedFloat32Array:
	var command := PackedFloat32Array()
	command.resize(13)
	command[0] = clampf(elapsed/duration,0.0,1.0)
	if heading_input:
		var r := quat_matrix(body.base_quat)
		var norm := maxf(sqrt(r[0][1]*r[0][1]+r[1][1]*r[1][1]),1e-6)
		var lx: float = r[0][1]/norm
		var ly: float = r[1][1]/norm
		command[1] = -heading[1]*ly-heading[0]*lx
		command[2] = -heading[1]*lx+heading[0]*ly
	return command

static func control(action: PackedFloat32Array, home: PackedFloat32Array, scale: float) -> PackedFloat32Array:
	# NumPy rounds the multiplication and addition separately to float32.
	var result := PackedFloat32Array()
	result.resize(14)
	var scale32 := PackedFloat32Array([scale])[0]
	for i in range(14): result[i] = action[i]*scale32
	for i in range(14): result[i] = home[i]+result[i]
	return result

static func ball_position(body: Dictionary, skill: String) -> Array:
	var q: Array = body.base_quat
	var yaw := atan2(2.0*(q[0]*q[3]+q[1]*q[2]),1.0-2.0*(q[2]*q[2]+q[3]*q[3]))
	var c := cos(yaw)
	var s := sin(yaw)
	var side := 0.042 if skill == "kick_left" else -0.042
	return [body.base_pos[0]+c*0.09-s*side,body.base_pos[1]+s*0.09+c*side,0.035]

static func brake_state_observation(obs: PackedFloat32Array, body: Dictionary) -> PackedFloat32Array:
	var result := obs.duplicate()
	var r := quat_matrix(body.base_quat)
	var yaw := atan2(r[1][0],r[0][0])
	var c := cos(yaw)
	var s := sin(yaw)
	var vx: float = body.base_linvel[0]
	var vy: float = body.base_linvel[1]
	result[58] = c*vx+s*vy
	result[59] = -s*vx+c*vy
	result[60] = body.base_pos[2]
	return result
