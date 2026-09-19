extends RefCounted
## Versioned causal 61 -> 68 observation extension; mirrors BrakeTaskState.
var last_stamp = null
var brake_start = null
var velocities: Array = []
var low_steps := 0
var values := PackedFloat32Array([0,0,0,0,0,0,0])

func reset() -> void:
	last_stamp=null
	brake_start=null
	velocities.clear()
	low_steps=0
	values.fill(0.0)

func observe(obs: PackedFloat32Array, raw: Dictionary, groups: Array, stamp: float) -> PackedFloat32Array:
	var bodies: Dictionary = {}
	for body in raw.get("body_states",raw.get("dump",[])): bodies[body.name]=body
	if groups.size()!=2: return PackedFloat32Array()
	var contacts := [0.0,0.0]
	for side in range(2):
		if groups[side].is_empty(): return PackedFloat32Array()
		for name in groups[side]:
			if not bodies.has(name): return PackedFloat32Array()
			if bodies[name].get("ground_contact",false): contacts[side]=1.0
	if last_stamp!=null and stamp<float(last_stamp)-1e-9: return PackedFloat32Array()
	if last_stamp==null or stamp>float(last_stamp)+1e-9:
		var braking := obs[48]<-0.01
		if not braking:
			brake_start=null
			velocities.clear()
			low_steps=0
		elif brake_start==null:
			brake_start=stamp
			velocities.clear()
			low_steps=0
		velocities.append([float(obs[58]),float(obs[59])])
		if velocities.size()>10: velocities.pop_front()
		var speed := 0.0
		var vx := 0.0
		var vy := 0.0
		for velocity in velocities:
			speed+=sqrt(velocity[0]*velocity[0]+velocity[1]*velocity[1])
			vx+=velocity[0]
			vy+=velocity[1]
		speed/=velocities.size()
		vx/=velocities.size()
		vy/=velocities.size()
		low_steps=low_steps+1 if braking and velocities.size()==10 and speed<0.05 else 0
		values=PackedFloat32Array([0.0 if brake_start==null else minf(6.0,stamp-float(brake_start)),
			minf(2.0,low_steps*0.02),speed,vx,vy,contacts[0],contacts[1]])
		last_stamp=stamp
	var result := obs.duplicate()
	result.append_array(values)
	return result
