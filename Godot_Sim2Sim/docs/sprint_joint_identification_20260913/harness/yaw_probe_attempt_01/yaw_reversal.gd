extends "res://standalone/motion_control_base.gd"
## Diagnostic control contract: ramp only an active sprint turn reversal.
var previous_request := 0.0
var filtered_turn := 0.0
var turn_step := 0.0
var transitioning := false

func reset() -> void:
	super.reset()
	previous_request=0.0
	filtered_turn=0.0
	turn_step=0.0
	transitioning=false

func command(input: PackedFloat32Array, body: Dictionary, skill: String, dt: float) -> PackedFloat32Array:
	var out := super.command(input,body,skill,dt)
	var duration := float(settings.get("sprint_yaw_reversal_s",0.0))
	var requested := float(input[2])
	if skill != "sprint" or duration<=0.0 or absf(requested)<=0.05:
		previous_request=requested if skill=="sprint" else 0.0
		filtered_turn=requested
		transitioning=false
		return out
	if previous_request*requested<0.0 and absf(previous_request)>0.05:
		transitioning=true
		turn_step=absf(requested-filtered_turn)*dt/duration
	if transitioning:
		filtered_turn=move_toward(filtered_turn,requested,turn_step)
		if absf(filtered_turn-requested)<1e-9: transitioning=false
		out[2]=filtered_turn
	else: filtered_turn=requested
	previous_request=requested
	return out
