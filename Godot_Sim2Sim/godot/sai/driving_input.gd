extends RefCounted
## Vehicle steering: left/right steer the rear along that side while reversing.
## With no drive input, retain Sai's useful left/right pivot turn.
static func vehicle_command(request: Array, cruise_speed: float = .16) -> Array:
	var result := request.duplicate()
	result[0] = float(request[0]) * cruise_speed / .16
	if float(request[0]) < 0.0: result[1] = -float(request[1])
	return result
