extends RefCounted
## Native counterpart of PlayBrain's external-input path. Network-free state.

var yaw_reversing := false
const SKILL_TAPS := ["pick", "sit", "kick_left", "kick_right", "roulade", "stand"]
var available: Dictionary = {}
var limits: Dictionary = {}
var policy := "standing"
var vel := PackedFloat32Array([0.0, 0.0, 0.0])
var sit := false
var stand_hold := false
var sprinting := false
var pick_phase := 0.0
var behavior_t := 0.0
var rise_t := 0.0
var pick_period := 4.0
var crouch_period := 5.0
var kick_duration := 5.0
var roulade_duration := 5.0
var rise_duration := 3.0
var gait_walking := false
var press_order: Array = []
var previous_held: Array = []

func configure(flags: Dictionary, config: Dictionary) -> void:
	available = flags.duplicate()
	limits = {"vmax_x":0.3,"vmin_x":-0.3,"vmax_y":0.2,"vmin_y":-0.2,
		"vmax_ang":1.5,"switch_on":0.10,"switch_off":0.03,"accel":12.0,"decel":20.0,
		"sprint_vmax_x":0.5,"sprint_vmax_ang":0.8,"sprint_yaw_reversal_s":0.0}
	limits.merge(config, true)
	reset_motion()

func has_policy(name: String) -> bool:
	return bool(available.get(name, false))

func reset_motion() -> void:
	yaw_reversing=false
	vel.fill(0.0)
	sit = false
	stand_hold = false
	sprinting = false
	pick_phase = 0.0
	behavior_t = 0.0
	rise_t = 0.0
	gait_walking = false
	press_order.clear()
	previous_held.clear()
	if has_policy("standing"):
		policy = "standing"
	elif has_policy("sitstand"):
		policy = "sitstand"
	elif has_policy("walking"):
		policy = "walking"

func busy() -> bool:
	return rise_t > 0.0 or policy in ["ground_pick","roller_crouch","kick_left","kick_right","roulade"]

func advance(dt: float) -> void:
	if rise_t > 0.0:
		rise_t = maxf(0.0, rise_t - dt)
		if rise_t < 1e-9:
			rise_t = 0.0
		return
	if policy in ["ground_pick", "roller_crouch"]:
		pick_phase += dt / (crouch_period if policy == "roller_crouch" else pick_period)
		if pick_phase >= 1.0 - 1e-9:
			reset_motion()
	elif policy in ["kick_left", "kick_right", "roulade"]:
		behavior_t -= dt
		if behavior_t <= 1e-9:
			reset_motion()

func tap(action: String) -> void:
	if action == "stand":
		if not has_policy("stand_hold") or busy() or sit:
			return
		reset_motion()
		stand_hold = true
		policy = "standing"
		return
	if action in SKILL_TAPS and not busy(): stand_hold = false
	if action == "sit":
		if has_policy("roller_crouch") and not busy():
			policy = "roller_crouch"
			pick_phase = 0.0
			vel.fill(0.0)
			return
		if not has_policy("sitstand") or busy():
			return
		sit = not sit
		rise_t = 0.0 if sit else rise_duration
		vel.fill(0.0)
		gait_walking = false
		policy = "sitstand"
	elif action == "pick":
		if not has_policy("ground_pick") or busy() or sit:
			return
		policy = "ground_pick"
		pick_phase = 0.0
		vel.fill(0.0)
	elif action in ["kick_left", "kick_right", "roulade"]:
		if not has_policy(action) or busy() or sit:
			return
		policy = action
		behavior_t = roulade_duration if action == "roulade" else kick_duration
		vel.fill(0.0)

func last_pressed(held: Array, order: Array) -> String:
	for i in range(order.size() - 1, -1, -1):
		if held.has(order[i]):
			return str(order[i])
	return ""

func held_target(held: Array) -> PackedFloat32Array:
	if held.has("idle"):
		return PackedFloat32Array([0.0,0.0,0.0])
	var h := held.duplicate()
	for pair in [["fwd","back"],["strafe_l","strafe_r"],["left","right"]]:
		if h.has(pair[0]) and h.has(pair[1]):
			var keep: String = pair[0] if press_order.find(pair[0]) >= press_order.find(pair[1]) else pair[1]
			if press_order.is_empty():
				keep = pair[0] if pair[0] > pair[1] else pair[1]
			h.erase(pair[1] if keep == pair[0] else pair[0])
	sprinting = sprinting and h.has("fwd")
	var forward_limit: float = limits.sprint_vmax_x if sprinting else limits.vmax_x
	var turn_limit: float = limits.sprint_vmax_ang if sprinting else limits.vmax_ang
	var vx: float = (forward_limit if h.has("fwd") else 0.0) + (float(limits.vmin_x) if h.has("back") else 0.0)
	var vy: float = (float(limits.vmax_y) if h.has("strafe_l") else 0.0) + (float(limits.vmin_y) if h.has("strafe_r") else 0.0)
	var cap_x: float = forward_limit if vx >= 0.0 else -limits.vmin_x
	var cap_y: float = limits.vmax_y if vy >= 0.0 else -limits.vmin_y
	var length := sqrt(vx*vx + vy*vy)
	var cap := sqrt(cap_x*cap_x + cap_y*cap_y)
	if length > cap and cap > 0.0:
		vx *= cap/length
		vy *= cap/length
	var yaw: float = (turn_limit if h.has("left") else 0.0) - (turn_limit if h.has("right") else 0.0)
	return PackedFloat32Array([vx,vy,yaw])

func set_locomotion(held: Array, dt: float) -> void:
	var requested_sprint := held.has("sprint")
	held = held.filter(func(bit): return bit != "sprint")
	press_order = press_order.filter(func(bit): return bit != "sprint")
	sprinting = false
	if busy() or sit:
		return
	if stand_hold:
		if held.filter(func(bit): return bit != "idle").is_empty():
			policy = "standing"
			return
		stand_hold = false
	var yaw_new := (held.has("left") and not previous_held.has("left")) or (held.has("right") and not previous_held.has("right"))
	previous_held = held.duplicate()
	if held.has("idle") and last_pressed(held,press_order) == "idle":
		held = ["idle"]
	sprinting = has_policy("sprint") and requested_sprint and not held.has("idle")
	var target := held_target(held)
	for i in range(3):
		var current: float = vel[i]
		var next: float = target[i]
		var away := (current == 0.0 and next != 0.0) or (signf(next) == signf(current) and absf(next) > absf(current))
		var max_delta: float = (float(limits.accel) if away else float(limits.decel)) * dt
		if i==2:
			var reversal_seconds := float(limits.sprint_yaw_reversal_s)
			if not sprinting or reversal_seconds<=0.0 or absf(next)<=0.05:
				yaw_reversing=false
			elif current*next<0.0 and absf(current)>0.05:
				yaw_reversing=true
			# Keep the reversal active through zero; ordinary turns retain the existing ramp.
			if yaw_reversing: max_delta=2.0*float(limits.sprint_vmax_ang)*dt/reversal_seconds
		vel[i] = next if absf(next-current) <= max_delta else current + signf(next-current)*max_delta
		if i==2 and absf(float(vel[2])-next)<1e-7: yaw_reversing=false
	if has_policy("walking") and has_policy("standing"):
		var linear := sqrt(float(vel[0])*vel[0] + float(vel[1])*vel[1])
		var angular := absf(vel[2])
		if gait_walking:
			if linear <= limits.switch_off and angular <= limits.switch_off:
				gait_walking = false
		elif linear >= limits.switch_on or (yaw_new and angular >= limits.switch_off):
			gait_walking = true
		policy = "walking" if gait_walking else "standing"
	elif has_policy("walking"):
		policy = "walking"
	elif has_policy("standing"):
		policy = "standing"

func command_13() -> PackedFloat32Array:
	var result := PackedFloat32Array()
	result.resize(13)
	result.fill(0.0)
	if policy in ["ground_pick", "roller_crouch"]:
		result[0] = cos(TAU * pick_phase)
		result[1] = sin(TAU * pick_phase)
	elif policy == "sitstand":
		result[0] = 1.0 if sit else 0.0
	elif policy == "walking":
		for i in range(3):
			result[i] = vel[i]
	return result

func tick(held: Array, taps: Array, dt: float, order: Array) -> Dictionary:
	var held_now := held.duplicate()
	if taps.has("idle") and not held_now.has("idle"):
		held_now.append("idle")
	press_order = order.filter(func(bit): return held.has(bit))
	var started := ""
	if taps.has("reset"):
		reset_motion()
	else:
		advance(dt)
		var previous := policy
		for action in taps:
			if action in SKILL_TAPS:
				tap(action)
		if busy() and policy != previous:
			started = policy
		set_locomotion(held_now, dt)
	var status := policy
	if policy == "sitstand":
		status = "sit" if sit else ("rising" if rise_t > 0.0 else "sitstand-stand")
	elif policy == "walking":
		status = ("sprint" if sprinting else "walk") + " vx=%+.2f vy=%+.2f w=%+.2f" % [vel[0],vel[1],vel[2]]
	return {"policy":policy,"command":command_13(),"started_skill":started,
		"reset":taps.has("reset"),"quit":taps.has("quit"),"push":taps.has("push"),
		"switch_robot":taps.has("switch_robot"),"status":status,"sprint":sprinting}
