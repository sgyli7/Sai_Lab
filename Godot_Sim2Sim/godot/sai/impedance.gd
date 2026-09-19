extends RefCounted
## Exact public motor contract: target PD plus support feedforward, then saturation.
static func torque(command: Dictionary, q: float, v: float, joint: int) -> float:
	return clampf(float(command.leg_kp[joint])*(float(command.target_leg[joint])-q)
		-float(command.leg_kd[joint])*v+float(command.leg_feedforward[joint]),-8.,8.)
