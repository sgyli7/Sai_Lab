extends Node
## Player ownership only: all filtering, forces and limits stay in the imported runtime.
var hub: Node3D
var selected := false
var held: Dictionary = {}
var orbit_yaw := .38
var orbit_pitch := .16
var distance := 145.0
var view_index := 0
var command := Vector2.ZERO
var focused := true
var wall_start:=0
var sim_start:=0.0

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	wall_start=Time.get_ticks_usec();sim_start=hub.elapsed
	get_window().focus_exited.connect(_focus_lost)
	get_window().focus_entered.connect(func(): focused=true)

func _focus_lost() -> void:
	focused=false
	stop()

func stop() -> void:
	held.clear()
	for action in ["forward","reverse","left","right","crouch"]:
		if InputMap.has_action(action):Input.action_release(action)
	command=Vector2.ZERO
	if hub.leviathan!=null:hub.leviathan.vehicle.runtime.set_vehicle_command(0.,0.)

func select(value:bool) -> void:
	stop()
	selected=value
	hub.drag=false
	if value and hub.actor!=null and hub.actor.grab!=null:hub.actor.grab.perform("cancel")
	hub.atelier.follow_initialized=false

func handle(event:InputEvent) -> void:
	if not selected:return
	if event is InputEventKey and not event.echo:
		var key:int=event.physical_keycode
		if key in [KEY_W,KEY_S,KEY_A,KEY_D,KEY_UP,KEY_DOWN,KEY_LEFT,KEY_RIGHT,KEY_SHIFT,KEY_SPACE]:
			if event.pressed and focused:held[key]=true
			else:held.erase(key)
			get_viewport().set_input_as_handled()

func _physics_process(_delta:float) -> void:
	if not selected:return
	if hub.switching or hub.stopping or get_tree().paused or not focused:
		stop();return
	var forward:float=float(held.has(KEY_W) or held.has(KEY_UP))-float(held.has(KEY_S) or held.has(KEY_DOWN))
	var left:float=float(held.has(KEY_A) or held.has(KEY_LEFT))-float(held.has(KEY_D) or held.has(KEY_RIGHT))
	var speed:float=.4 if held.has(KEY_SHIFT) else .2
	command=Vector2(forward*speed,left*.003 if forward!=0. else 0.)
	if held.has(KEY_SPACE):command=Vector2.ZERO
	hub.leviathan.vehicle.runtime.set_vehicle_command(command.x,command.y)

func update_camera(_delta:float) -> void:
	var runtime:Node3D=hub.leviathan.vehicle.runtime
	var front:Transform3D=runtime.get_design_transform("front")
	var rear:Transform3D=runtime.get_design_transform("rear")
	var target:Vector3=(front.origin+rear.origin)*.5+Vector3(0,8,0)
	var camera:Camera3D=hub.get_node("World/Camera3D")
	var offset:Vector3=hub._orbit_offset(orbit_yaw,orbit_pitch,distance)
	if view_index==1:offset=Vector3(-130,115,140)
	camera.global_position=target+offset
	camera.global_position.y=maxf(camera.global_position.y,hub.atelier.ground_height(camera.global_position.x,camera.global_position.z)+3.)
	camera.look_at(target)
	camera.fov=52.;camera.near=.25;camera.far=30000.

func telemetry() -> Dictionary:
	var runtime:Node3D=hub.leviathan.vehicle.runtime
	var body:RigidBody3D=runtime.bodies.front
	var front:Transform3D=runtime.get_design_transform("front")
	return {"selected":selected,"command":[command.x,command.y],"requested":[runtime.requested.x,runtime.requested.y],
		"position":[front.origin.x,front.origin.y,front.origin.z],"yaw":atan2(-front.basis.x.z,front.basis.x.x),
		"speed_m_s":body.linear_velocity.dot(front.basis.x),"yaw_rate_rad_s":body.angular_velocity.y,
		"upright":body.global_basis.y.y,"bodies":runtime.bodies.size(),
		"jel_failures":runtime.jel_failures.size(),"cargo_failures":runtime.connector_failures.size()}

func status_text() -> String:
	var t:=telemetry()
	var ratio:float=(hub.elapsed-sim_start)/maxf(.001,float(Time.get_ticks_usec()-wall_start)/1e6)
	return "03 / 极地雪原 · LEVIATHAN 001 · 10,966 吨\n指令 %+.2f m/s · 实测 %+.3f m/s · 转向 %+.3f °/s\n画面 %.0f FPS · 实际运行 %.2f×"%[command.x,t.speed_m_s,rad_to_deg(t.yaw_rate_rad_s),Engine.get_frames_per_second(),ratio]

func controls_text() -> String:
	return "F8 利维坦 / F7 Sai 001 · W/S 或 ↑/↓ 前进 / 倒车 · A/D 或 ←/→ 行驶转向\n按住行驶，松开减速 · Shift 0.4 m/s（巡航 0.2 m/s）· 空格停车 · 原地不转向\n右键拖动环视 · 滚轮缩放 · Tab 俯瞰 / 跟随 · 切换或窗口失焦自动发送停车指令\n母车前方为浅丘测试带 · 其余方向为开阔雪原"
