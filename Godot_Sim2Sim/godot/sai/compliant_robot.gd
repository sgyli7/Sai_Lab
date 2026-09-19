extends "res://robot.gd"
## Extend the frozen released adapter without duplicating arm, cargo or wheel code.
func apply_command(s: Dictionary, next_command: Dictionary) -> void:
	super.apply_command(s,next_command)
	if next_command.get("impedance_contract","")!="sai-joint-impedance-v1":return
	# All torques accumulate before Jolt integrates the next physics step.
	# Replace only the saturated leg PD component; retain passive damping.
	for i in range(16):
		if i%4==3:continue
		var original:float=clampf(80.*(float(next_command.target_leg[i])-float(s.q[i]))-2.*float(s.v[i]),-8.,8.)
		var replacement:float=preload("res://sai/impedance.gd").torque(next_command,float(s.q[i]),float(s.v[i]),i)
		var d:Dictionary=drives[i]
		var correction:Vector3=(d.parent.global_basis*d.axis)*(replacement-original)
		d.child.apply_torque(correction)
		d.parent.apply_torque(-correction)
