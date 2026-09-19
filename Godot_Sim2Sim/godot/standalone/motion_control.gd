extends RefCounted
## Keep numerically paired with sim2sim.motion_control.MotionControl.
const Contract = preload("res://standalone/policy_contract.gd")
var settings: Dictionary = {}
var started := false
var target_yaw := 0.0
var brake_elapsed := 0.0
var was_braking := false
var brake_speed := 0.0
var brake_target_yaw := 0.0
var walk_has_moved := false
var walk_was_moving := false
var walk_idle_elapsed := 0.0
var walk_path_started := false
var walk_path_origin: Array = [0.0,0.0]
var walk_path_yaw := 0.0
var walk_path_speed := 0.0

func reset() -> void:
	started=false
	target_yaw=0.0
	brake_elapsed=0.0
	was_braking=false
	brake_speed=0.0
	brake_target_yaw=0.0
	walk_has_moved=false
	walk_was_moving=false
	walk_idle_elapsed=0.0
	walk_path_started=false
	walk_path_origin=[0.0,0.0]
	walk_path_yaw=0.0
	walk_path_speed=0.0

func command(input: PackedFloat32Array, body: Dictionary, skill: String, dt: float) -> PackedFloat32Array:
	var out := input.duplicate()
	var rotation := Contract.quat_matrix(body.base_quat)
	var yaw := atan2(rotation[1][0],rotation[0][0])
	var ordinary := skill=="walking"
	if skill=="sprint": skill="walking"
	if not started or skill not in ["walking","roller"]:
		target_yaw=yaw
		started=true
	if skill not in ["walking","roller"]:
		brake_elapsed=0.0
		was_braking=false
		walk_has_moved=false
		walk_was_moving=false
		walk_path_started=false
		return out
	if skill == "roller" and settings.get("heading_hold",false):
		target_yaw+=float(out[2])*dt
		var difference := target_yaw-yaw
		out[2]=clampf(atan2(sin(difference),cos(difference)),-1.0,1.0)
	elif skill == "walking" and float(settings.get("walk_heading_gain",0.0))>0.0:
		var idle_only: bool = settings.get("walk_heading_scope","all")=="idle_after_motion"
		var moving: bool = sqrt(float(out[0])*float(out[0])+float(out[1])*float(out[1]))>0.01 or absf(out[2])>0.05
		if moving and not walk_was_moving and settings.get("walk_reanchor_on_start",false):
			target_yaw=yaw
		walk_was_moving=moving
		if moving:
			walk_has_moved=true
			walk_idle_elapsed=0.0
		if absf(out[2])>0.05 or (idle_only and moving):
			target_yaw=yaw
		elif not idle_only or walk_has_moved:
			if idle_only and walk_idle_elapsed<float(settings.get("walk_idle_delay_s",0.0))-1e-9:
				target_yaw=yaw
			else:
				var difference := target_yaw-yaw
				var error := atan2(sin(difference),cos(difference))
				var limit := float(settings.get("walk_heading_limit",0.3))
				out[2]=clampf(float(settings.walk_heading_gain)*error,-limit,limit)
			walk_idle_elapsed+=dt
	if skill == "walking":
		var path_gain := float(settings.get("walk_ordinary_path_gain",settings.get("walk_path_gain",0.0)) if ordinary else settings.get("walk_path_gain",0.0))
		var straight: bool = input[0]>0.01 and absf(input[1])<0.01 and absf(input[2])<0.05
		if path_gain>0.0 and straight:
			if not walk_path_started:
				walk_path_origin=[body.base_pos[0],body.base_pos[1]]
				walk_path_yaw=yaw
				walk_path_started=true
				walk_path_speed=0.0
			var dx: float = float(body.base_pos[0])-float(walk_path_origin[0])
			var dy: float = float(body.base_pos[1])-float(walk_path_origin[1])
			var error := -sin(walk_path_yaw)*dx+cos(walk_path_yaw)*dy
			var damping := float(settings.get("walk_path_damping",0.0))
			if damping>0.0:
				var speed := -sin(walk_path_yaw)*float(body.base_linvel[0])+cos(walk_path_yaw)*float(body.base_linvel[1])
				walk_path_speed=0.8*walk_path_speed+0.2*speed
			var path_limit := float(settings.get("walk_path_limit",0.15))
			out[1]=clampf(-path_gain*error-damping*walk_path_speed,-path_limit,path_limit)
			var lookahead := float(settings.get("walk_path_lookahead",0.0))
			if lookahead>0.0 and float(settings.get("walk_heading_gain",0.0))>0.0:
				target_yaw=walk_path_yaw-atan2(error,lookahead)
				var difference := target_yaw-yaw
				var heading_error := atan2(sin(difference),cos(difference))
				var limit := float(settings.get("walk_heading_limit",0.3))
				out[2]=clampf(float(settings.walk_heading_gain)*heading_error,-limit,limit)
		else: walk_path_started=false
		var scale := float(settings.get("walk_translation_scale",1.0))
		out[0]=float(out[0])*scale
		out[1]=float(out[1])*scale
	var braking := skill == "roller" and out[0]<0.0
	if braking:
		if not was_braking: brake_elapsed=0.0
		var heading_gain := float(settings.get("brake_heading_gain",0.0))
		if heading_gain>0.0:
			if not was_braking or absf(out[2])>0.05: brake_target_yaw=yaw
			if absf(out[2])<=0.05:
				var difference := brake_target_yaw-yaw
				out[2]=clampf(heading_gain*atan2(sin(difference),cos(difference)),-1.0,1.0)
		var gain := float(settings.get("brake_velocity_gain",0.0))
		if gain>0.0:
			var speed: float = cos(yaw)*float(body.base_linvel[0])+sin(yaw)*float(body.base_linvel[1])
			brake_speed=speed if not was_braking else 0.8*brake_speed+0.2*speed
			out[0]=clampf(-gain*brake_speed,-0.5,0.2)
		if settings.has("brake_pulse_s") and brake_elapsed>=float(settings.brake_pulse_s)-1e-9:
			out[0]=0.0
		brake_elapsed+=dt
	else: brake_elapsed=0.0
	was_braking=braking
	return out
