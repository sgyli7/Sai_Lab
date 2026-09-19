extends RefCounted
## Time demand changes the number of original 0.5 ms steps per wall second.
## Never enlarge the physical integration step to simulate a faster clock.
const PHYSICS_HZ:=2000
var rate:=1.0

func set_rate(value:float) -> void:
	rate=snappedf(clampf(value,.1,3.),.1)
	Engine.time_scale=rate
	Engine.physics_ticks_per_second=roundi(PHYSICS_HZ*rate)
	# Leave room for the requested rate, including catch-up after a slow frame.
	# The previous ten-step cap at 30 FPS imposed an artificial 0.15x ceiling.
	Engine.max_physics_steps_per_frame=maxi(100,ceili(Engine.physics_ticks_per_second/30.)*2)
