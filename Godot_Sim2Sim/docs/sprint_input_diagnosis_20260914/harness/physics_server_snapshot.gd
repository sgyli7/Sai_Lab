extends Node3D
## Lockstep physics server v2. Protocol is MuJoCo-native (Z-up, quat wxyz).
## Line-delimited JSON over TCP. Python is the only controller.

const ROBOT_SCENE := "res://generated/microduck/robot.tscn"
const DEFAULT_SPEC := "res://generated/microduck/robot_spec.json"
var _robot_scene: String = ROBOT_SCENE
var _spec_path: String = DEFAULT_SPEC
# Off: unilateral world-Y 6DOF at 4 sole corners of the SAME rigid foot
# (foot-floor collision excluded). Offset 6DOF is real (spike: 1 kg /
# k=2000 / 5 cm offset sags 4.9 mm). C3: 8→6→4 springs, first-hit sag
# 8 mm, t=0.92 tilt 28° (uniform sag, not MJ peel), jaw x only +0.10
# (no Coulomb), t=1.20 jaw impulse 0.17 unloads the rest, hips, z_min
# 0.0305. XZ damping with k=0 is a no-op. 4-corner springs are still a
# support polygon.
# Off: 2 outer-ridge points (MJ C3 first-hit sagittal pair), k=500×4.
# t=0.12 sag 7.7 mm / 4 on; t=0.92 still 4 on / 32° / ankle z 0.000
# (feet stay flat, torso folds, cx slides to −0.07); heel unloads at
# t=1.00; t=1.20 jaw+2 toe springs then launch, z_min 0.0309, jaw x
# +0.09. Sagittal Y springs resist pitch and have no Coulomb. Plant
# stays hard 16-gon vs static box.
const SOLE_SPRINGS := false
const SOLE_SPRING_K := 250.0
const SOLE_SPRING_C := 12.0
# Off: migrating unilateral Y 6DOF at jaw_soft's lowest hull vert
# (jaw-floor collision excluded) + heel camber. t=0.92 peel still
# 74.7° / feet on / jpad +24 mm; t=1.00 feet already gone as the
# spring first hits (8.8 mm, vn≈0.40); head punches to −47 mm, hips
# t=1.14, z_min 0.0274. No 81°+feet+head window to catch. Plant stays
# hard 16-gon vs static box.
const JAW_SPRINGS := false
const JAW_SPRING_K := 2000.0
const JAW_SPRING_C := 89.0
const SPRUNG_FLOOR := false
const SPRUNG_FLOOR_K := 2000.0
const SPRUNG_FLOOR_MASS := 0.15

# Cinemachine-style orbit-follow camera (docs/research_3c_camera.md §C).
const CAM_PITCH_MIN := 0.06
const CAM_PITCH_MAX := 1.25
const CAM_DIST_MIN := 0.35
const CAM_DIST_MAX := 2.5
const CAM_DEFAULT_YAW := 0.7853981634  # 45°, matches legacy (0.65, ·, 0.65) quadrant
const CAM_DEFAULT_PITCH := 0.5
const CAM_DEFAULT_DIST := 1.0
# Third-person follow: softer than locked tracking. Split H/V so biped
# footstep bob (Y) is filtered harder than planar chase — same idea as
# SpringArm + lagged look-target in TPS games (Godot spring_arm docs /
# Cinemachine body damping).
const CAM_POS_SMOOTH_H := 3.5  # planar look-target damping (was unified 10)
const CAM_POS_SMOOTH_V := 1.6  # vertical damping — kills walk bob / terrain shake
const CAM_RIG_SMOOTH := 5.0    # mild lag of camera rig toward desired orbit point
const CAM_YAW_SMOOTH := 3.0    # damping applied outside the deadzone (was 4)
const CAM_DIST_SMOOTH := 5.0
const CAM_YAW_DEADZONE := 0.6108652942  # 35°: small duck turns leave the shot alone
const CAM_LEAD_DIST := 0.35   # look-ahead (was 0.5; less whip on accel)
const CAM_LEAD_MAX := 0.25    # cap on the lead offset (m)
const CAM_YAW_DRAG := 0.005   # rad per px of right-drag
const CAM_PITCH_DRAG := 0.004
const CAM_DIST_DRAG := 0.0012 # wheel notch -> distance factor
const CAM_RECOVER_DELAY := 1.5  # s after mouse release before auto-follow
const CAM_SNAP_K := 25.0      # reset: fast exponential snap, not a teleport

var _server: TCPServer
var _peer: StreamPeerTCP
var _buf: PackedByteArray = PackedByteArray()
var _port: int = 9876
var _spec: Dictionary = {}
var _robot: Node = null
var _bodies: Dictionary = {}  # name -> RigidBody3D
var _base: RigidBody3D = null
var _base_name: String = "trunk_base"
var _joints: Array = []  # dicts
var _ctrl: PackedFloat32Array = PackedFloat32Array()
var _remaining: int = 0
var _pending_send: bool = false
var _report_mode: String = ""
var _t: float = 0.0
var _frozen: bool = true
var _pinned: Dictionary = {}
# Jolt zeros velocity on freeze-as-static. Lockstep freeze/unfreeze between
# Python commands was restarting every body from rest each control tick
# (drop vz stuck at g*dt=0.196, walk xy collapsed to ~0.14 m).
var _saved_lv: Dictionary = {}
var _saved_av: Dictionary = {}
var _reset_applied: Array = []
var _reset_missing: Array = []
var _reset_dump: Array = []
var _iquat_base: Quaternion = Quaternion.IDENTITY  # inertial-from-body, unused if aligned
var _dbg_ang_world: Array = [0.0, 0.0, 0.0]  # temp debug: Jolt world angular velocity
var _dbg_ang_jolt_local: Array = [0.0, 0.0, 0.0]  # Jolt ω in MuJoCo body frame
var _body_iquat: Dictionary = {}  # name -> Vector4(w,x,y,z)
var _body_ipos: Dictionary = {}  # name -> Vector3 in body frame
var _heel_hosts: Dictionary = {}  # heel_name -> host_name
var _heel_welds: Array = []  # HingeJoint3D
var _sole: Dictionary = {}  # body name -> {verts: PackedVector3Array, shapes: Array, radius: float}
var _floor_plate: RigidBody3D = null
var _floor_spring: Generic6DOFJoint3D = null
var _sole_springs: Array = []
var _jaw_spring: Generic6DOFJoint3D = null
var _jaw_body: RigidBody3D = null
var _jaw_pad_y: float = 0.0
var _held_now: Array = []
var _left_shift_down := false
var _held_press_order: Array = []  # held bits ordered oldest-press first
var _prev_held_set: Dictionary = {}  # bit -> true while physically held
var _taps: Array = []
var _hud: CanvasLayer = null
var _cam_offset: Vector3 = Vector3(0.65, 0.42, 0.65)
# Orbit-follow camera state (see CAM_* consts + _follow_camera()).
var _cam_yaw: float = CAM_DEFAULT_YAW
var _cam_pitch: float = CAM_DEFAULT_PITCH
var _cam_dist: float = CAM_DEFAULT_DIST
var _cam_look: Vector3 = Vector3.ZERO
var _cam_manual_until: float = 0.0  # seconds on the _cam_wall_seconds() clock
var _cam_dragging: bool = false
var _cam_inited: bool = false
var _cam_snap: bool = true  # fast exponential snap on (re)spawn/reset
var _cam_snap_started: float = 0.0  # wall-clock seconds when the last snap began
var _cam_last_sec: float = 0.0  # wall clock for camera damping (lockstep-aware)
var _window_title: String = "Microduck Sim2Sim"
var _headless: bool = false
var _foot_names: Array = []
var _research_bodies: Array = []
var _research_contact_events: Dictionary = {}
var _research_capture_path: String = ""
var _mj_basis: Dictionary = {}  # name -> Basis, refreshed each PD tick
# Kinematic velocities from pose finite difference. Jolt's reported
# angular_velocity includes Baumgarte/position-correction drift (~0.22 rad/s
# gyro-z and hip/head yaw qd while standing still). A raw 1-tick FD of that
# pose reconstructs the chatter; sampling it every 20 ms is phase-locked
# and reintroduces the bias. EMA (τ=1 tick, α=1-e^-1≈0.63) rejects the 200 Hz mode
# without the 20 ms delay of a decimation-length window (that delay
# destabilizes PD).
const KIN_VEL_TAU := 0.005
var _kin_basis: Dictionary = {}  # body name -> Basis (Godot world, last tick)
var _kin_omega: Dictionary = {}  # body name -> Vector3 Godot-world ω (EMA)
var _kin_after_tick: bool = false
var _timing_enabled: bool = false
var _timing_phys_t0: int = 0
var _timing_phys_usec: int = 0
var _timing_pd_usec: int = 0
var _timing_send_usec: int = 0
var _timing_wait_usec: int = 0
var _timing_wait_iters: int = 0
var _timing_json_bytes: int = 0
const SPIN_DELAY_USEC := 50

func _ready() -> void:
	_parse_args()
	_headless = DisplayServer.get_name() == "headless"
	# Headless lockstep uses --fixed-fps 200, so 1 physics step per "frame" is 200 Hz.
	# A vsync window is ~60 fps; 1 step/frame would make 4 substeps take ~66 ms for
	# 20 ms of sim (~0.3× realtime). Allow a burst of ticks per displayed frame.
	# --fixed-fps disables real-time sync (Engine.max_fps is not a wall-clock cap);
	# observed >>200 physics ticks/s confirms Jolt is not paced to 200 Hz wall.
	Engine.max_fps = 0
	OS.low_processor_usage_mode = false
	if _headless:
		Engine.max_physics_steps_per_frame = 1
	else:
		Engine.max_physics_steps_per_frame = 16
		DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	_load_spec()
	_instance_robot()
	_paint_robot_visuals()
	_collect_bodies()
	_setup_joints()
	_setup_heels()
	_setup_soles()
	_setup_sprung_floor()
	_setup_sole_springs()
	_setup_jaw_spring()
	_start_controller()
	print(
		"play_pacing display=%s max_phys=%s max_fps=%s vsync=%s"
		% [
			DisplayServer.get_name(),
			Engine.max_physics_steps_per_frame,
			Engine.max_fps,
			DisplayServer.window_get_vsync_mode(),
		]
	)
	_freeze(true)
	_setup_floor_checker()
	_setup_play_ui()
	_cam_last_sec = _cam_wall_seconds()
	if _hud != null and _hud.has_method("set_mode"):
		_hud.set_mode("roller" if "roller" in _robot_scene else "walk")
	_snapshot_kinematic_pose()
	# Load presentation resources only for visible play; headless workers skip them.
	if not _headless and OS.get_environment("SIM2SIM_VISUAL_STYLE") != "legacy":
		load("res://visuals/microduck/style.gd").new().apply(self)

	call_deferred("_maybe_dump_sim2sim_shot")


func _start_controller() -> void:
	_server = TCPServer.new()
	var err := _server.listen(_port, "127.0.0.1")
	if err != OK:
		push_error("listen failed on port %s err=%s" % [_port, err])
		get_tree().quit(1)
		return
	print("sim2sim_physics_server listening 127.0.0.1:%s" % _port)



func _maybe_dump_sim2sim_shot() -> void:
	## One-shot viewport PNG when SIM2SIM_SHOT=/abs/path.png is set.
	var path := OS.get_environment("SIM2SIM_SHOT")
	if path == "":
		return
	await get_tree().process_frame
	await get_tree().process_frame
	await get_tree().create_timer(0.8).timeout
	var tex := get_viewport().get_texture()
	if tex == null:
		push_warning("SIM2SIM_SHOT: no viewport texture")
		return
	var img := tex.get_image()
	if img == null:
		push_warning("SIM2SIM_SHOT: get_image failed")
		return
	var err := img.save_png(path)
	print("SIM2SIM_SHOT saved err=%s path=%s" % [err, path])
	if OS.get_environment("SIM2SIM_SHOT_QUIT") == "1":
		get_tree().quit(0)


func _parse_args() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--port="):
			_port = int(a.substr(7))
		elif a.begins_with("--spec="):
			_spec_path = a.substr(7)
		elif a.begins_with("--robot-scene="):
			_robot_scene = a.substr(14)
		elif a.begins_with("--base="):
			_base_name = a.substr(7)


func _load_spec() -> void:
	if not FileAccess.file_exists(_spec_path):
		push_warning("robot_spec.json missing — robot scene may still load")
		return
	var txt := FileAccess.get_file_as_string(_spec_path)
	var parsed = JSON.parse_string(txt)
	if typeof(parsed) != TYPE_DICTIONARY:
		push_error("robot_spec.json parse failed")
		return
	_spec = parsed
	for b in _spec.get("bodies", []):
		var q: Array = b.get("iquat_wxyz", [1, 0, 0, 0])
		_body_iquat[str(b["name"])] = Vector4(q[0], q[1], q[2], q[3])
		var ip: Array = b.get("ipos", [0, 0, 0])
		_body_ipos[str(b["name"])] = Vector3(float(ip[0]), float(ip[1]), float(ip[2]))


func _instance_robot() -> void:
	if not ResourceLoader.exists(_robot_scene):
		push_error("missing %s — run mjcf2godot first" % _robot_scene)
		return
	_robot = load(_robot_scene).instantiate()
	if "roller" in _robot_scene:
		_window_title = "Microduck Sim2Sim · rollers"
	else:
		_window_title = "Microduck Sim2Sim"
	DisplayServer.window_set_title(_window_title)
	$RobotHost.add_child(_robot)


func _collect_bodies() -> void:
	_bodies.clear()
	if _robot == null:
		return
	for child in _robot.get_children():
		if child is RigidBody3D:
			_bodies[str(child.name)] = child
			child.can_sleep = false
			child.freeze = true
			child.contact_monitor = true
			child.max_contacts_reported = 24
			# Re-assert inertial-frame COM after shapes load. Jolt can otherwise
			# keep a shape-derived COM even when the tscn says CUSTOM/zero.
			child.center_of_mass_mode = RigidBody3D.CENTER_OF_MASS_MODE_CUSTOM
			child.center_of_mass = Vector3.ZERO
	if _bodies.has(_base_name):
		_base = _bodies[_base_name]
	elif _bodies.size() > 0:
		_base = _bodies.values()[0]
		_base_name = _base.name


func _setup_joints() -> void:
	_joints.clear()
	if _robot == null or _spec.is_empty():
		return
	var act_by_joint: Dictionary = {}
	for a in _spec.get("actuators", []):
		act_by_joint[str(a["joint"])] = a
	for j in _spec.get("joints", []):
		if str(j.get("type", "")) != "hinge":
			continue
		var jname: String = str(j["name"])
		var node := _find_joint_node("joint_" + jname)
		if node == null:
			push_warning("HingeJoint3D not found: joint_%s" % jname)
			continue
		var parent_name: String = str(j.get("parent", ""))
		var child_name: String = str(j["body"])
		var parent: PhysicsBody3D = null
		if parent_name == "world" or int(j.get("parent_id", -1)) == 0:
			parent = _robot.get_node_or_null("WorldAnchor") as PhysicsBody3D
		elif _bodies.has(parent_name):
			parent = _bodies[parent_name]
		var child: RigidBody3D = _bodies.get(child_name)
		if child == null:
			push_warning("child body missing: %s" % child_name)
			continue
		var lo: float = -PI
		var hi: float = PI
		var limited := bool(j.get("limited", true))
		if limited:
			var rng: Array = j.get("range", [-PI, PI])
			lo = float(rng[0])
			hi = float(rng[1])
		node.set("angular_limit/enable", limited)
		# Godot hinge angles are clockwise; MuJoCo joint q is counterclockwise.
		node.set("angular_limit/lower", -hi)
		node.set("angular_limit/upper", -lo)
		node.set("motor/enable", false)
		var kp := 0.0
		var kv := 0.0
		var fmin := -INF
		var fmax := INF
		var act_index := -1
		if act_by_joint.has(jname):
			var act: Dictionary = act_by_joint[jname]
			kp = float(act.get("kp", 0.0))
			kv = float(act.get("kv", 0.0))
			var fr: Array = act.get("forcerange", [-0.96, 0.96])
			fmin = float(fr[0])
			fmax = float(fr[1])
			act_index = int(act.get("id", -1))
		var damping := float(j.get("damping", 0.0))
		var armature := float(j.get("armature", 0.0))
		var frictionloss := float(j.get("frictionloss", 0.0))
		var ab: Array = j.get("axis_parent_body", [0, 0, 1])
		var axis_pb := Vector3(float(ab[0]), float(ab[1]), float(ab[2]))
		if axis_pb.length() < 1e-9:
			axis_pb = Vector3(0, 0, 1)
		axis_pb = axis_pb.normalized()
		var ac: Array = j.get("axis_child_body", [0, 0, 1])
		var axis_cb := Vector3(float(ac[0]), float(ac[1]), float(ac[2]))
		if axis_cb.length() < 1e-9:
			axis_cb = Vector3(0, 0, 1)
		axis_cb = axis_cb.normalized()
		# Jolt has no joint-space armature. True map is I += A n n^T but Godot
		# only stores diagonal inertia. Add the diagonal of A nn^T and floor
		# the other principals so cond(I) stays bounded (full anisotropic
		# 1000:1 tensors made the first step |qd| explode).
		var n_i := Vector3(0, 0, 1)
		# Also apply to unactuated wheels: skipping A left I≈5e-7 on the two
		# locked hinge axes and Jolt's hinge solver welded the wheel (nudge
		# vx=0.5 died in 20 ms, ω stayed ~0). XML A=1e-4 is the bearing rotor.
		if armature > 0.0:
			var iq: Vector4 = _body_iquat.get(child_name, Vector4(1, 0, 0, 0))
			var r_iq := Basis(Quaternion(iq.y, iq.z, iq.w, iq.x))
			n_i = r_iq.transposed() * axis_cb
			if n_i.length_squared() > 1e-12:
				n_i = n_i.normalized()
			var I := child.inertia
			I.x += armature * n_i.x * n_i.x
			I.y += armature * n_i.y * n_i.y
			I.z += armature * n_i.z * n_i.z
			var imax: float = maxf(I.x, maxf(I.y, I.z))
			# cond=10: walk/cadence match. 15 and 30 both fell. Compensating
			# A(1-Σn^4) onto hip_roll made local_ppo fall in run.
			var ifloor: float = imax / 10.0
			I.x = maxf(I.x, ifloor)
			I.y = maxf(I.y, ifloor)
			I.z = maxf(I.z, ifloor)
			child.inertia = I
		var rq: Array = j.get("rest_rel_q0_wxyz", [1, 0, 0, 0])
		var rest_rel := Basis(Quaternion(float(rq[1]), float(rq[2]), float(rq[3]), float(rq[0])))
		_joints.append({
			"name": jname,
			"node": node,
			"parent": parent,
			"parent_name": parent_name,
			"child": child,
			"child_name": child_name,
			"kp": kp,
			"kv": kv,
			"fmin": fmin,
			"fmax": fmax,
			"damping": damping,
			"armature": armature,
			"frictionloss": frictionloss,
			"act_index": act_index,
			"axis_parent_body": axis_pb,
			"axis_child_body": axis_cb,
			"n_i": n_i,
			"rest_rel": rest_rel,
			"rest_rel_t": rest_rel.transposed(),
			"lo": lo,
			"hi": hi,
			"limited": limited,
			"q_rebake": 0.0,
		})
	var nu := int(_spec.get("nu", 0))
	_ctrl.resize(nu)
	_ctrl.fill(0.0)
	for j in _joints:
		var parent = j["parent"]
		if parent is RigidBody3D:
			_exclude_ancestors(j["child"], parent)
	_foot_names = _foot_body_names()


func _rebake_joints() -> void:
	# Jolt captures hinge frames when node_a/node_b are assigned. After a
	# kinematic teleport those frames still describe q=0, so the solver
	# yanks every limb back in one tick. Re-assign paths at the current pose
	# The q cache must describe the NEW teleported pose before rebaking.
	# Otherwise every reset inherits the previous episode's end-stop offset.
	# Hinge angle h = -(q - q_reset), so XML [lo, hi] maps to
	# Godot [q_reset-hi, q_reset-lo]. This is not a soft-limit tuning choice.
	_refresh_mj_basis()
	for j in _joints:
		var node: HingeJoint3D = j["node"]
		var q := _joint_q(j)
		j["q_rebake"] = q
		if bool(j.get("limited", true)):
			node.set("angular_limit/lower", q-float(j["hi"]))
			node.set("angular_limit/upper", q-float(j["lo"]))
		var a: NodePath = node.node_a
		var b: NodePath = node.node_b
		node.node_a = NodePath()
		node.node_b = NodePath()
		node.node_a = a
		node.node_b = b


func _setup_heels() -> void:
	# Extra RigidBody3D siblings named "{foot}__heel", welded with a 0-limit hinge.
	_heel_hosts.clear()
	_heel_welds.clear()
	for key in _bodies.keys():
		var hname := str(key)
		var host_name := ""
		if hname.ends_with("__heel"):
			host_name = hname.substr(0, hname.length() - 6)
		elif hname.ends_with("__toe"):
			host_name = hname.substr(0, hname.length() - 5)
		else:
			continue
		_heel_hosts[hname] = host_name
		var heel: RigidBody3D = _bodies[hname]
		if _bodies.has(host_name):
			var host: RigidBody3D = _bodies[host_name]
			heel.add_collision_exception_with(host)
			host.add_collision_exception_with(heel)
			for j in _joints:
				if j["child"] == host and j["parent"] is RigidBody3D:
					var shin: RigidBody3D = j["parent"]
					heel.add_collision_exception_with(shin)
					shin.add_collision_exception_with(heel)
		if _robot != null:
			var weld := _robot.find_child("weld_" + hname, true, false)
			if weld is HingeJoint3D:
				_heel_welds.append(weld)
	_rebake_welds()


func _setup_soles() -> void:
	# Snap the 3 foot spheres onto the currently lowest mesh verts (MJ
	# contact reduction: 1–3 points, migrating as the foot pitches).
	_sole.clear()
	for binfo in _spec.get("bodies", []):
		if typeof(binfo) != TYPE_DICTIONARY:
			continue
		var bname := str(binfo.get("name", ""))
		if not binfo.has("sole_verts") or not _bodies.has(bname):
			continue
		var raw: Array = binfo.get("sole_verts", [])
		var packed := PackedVector3Array()
		for v in raw:
			if typeof(v) != TYPE_ARRAY or v.size() < 3:
				continue
			packed.append(Vector3(float(v[0]), float(v[1]), float(v[2])))
		if packed.is_empty():
			continue
		var body: RigidBody3D = _bodies[bname]
		var shapes: Array = []
		for child in body.get_children():
			if child is CollisionShape3D and str(child.name).contains("foot_collision"):
				shapes.append(child)
		if shapes.is_empty():
			continue
		var radius := float(binfo.get("sole_radius", 0.001))
		_sole[bname] = {
			"verts": packed,
			"shapes": shapes,
			"radius": radius,
			"last_idx": PackedInt32Array(),
			"pinned": false,
		}
	_update_sole_spheres()


func _fps_from_idx(world_y: PackedFloat32Array, verts: PackedVector3Array, pool: Array, k: int) -> PackedInt32Array:
	var out := PackedInt32Array()
	if pool.is_empty() or k <= 0:
		return out
	var seed_i := int(pool[0])
	var seed_y := world_y[seed_i]
	for pi in pool:
		var i0 := int(pi)
		if world_y[i0] < seed_y:
			seed_y = world_y[i0]
			seed_i = i0
	out.append(seed_i)
	while out.size() < mini(k, pool.size()):
		var best_i := int(pool[0])
		var best_d := -1.0
		for pi in pool:
			var i2 := int(pi)
			if out.has(i2):
				continue
			var md := 1.0e9
			for cj in out:
				var d: float = verts[i2].distance_squared_to(verts[int(cj)])
				if d < md:
					md = d
			if md > best_d:
				best_d = md
				best_i = i2
		if out.has(best_i):
			break
		out.append(best_i)
	return out


func _update_sole_spheres() -> void:
	# MJ contact reduction: 1–3 points on the current 0.4 mm lowest band.
	# No posterior pin — that either froze the heel (dump) or unlatched as
	# the same toe verts moved in world x.
	const BAND := 0.0004
	for bname in _sole.keys():
		if not _bodies.has(bname):
			continue
		var info: Dictionary = _sole[bname]
		var body: RigidBody3D = _bodies[bname]
		var verts: PackedVector3Array = info["verts"]
		var shapes: Array = info["shapes"]
		var radius: float = float(info["radius"])
		if verts.is_empty() or shapes.is_empty():
			continue
		var xf := body.global_transform
		var n := verts.size()
		var world_y := PackedFloat32Array()
		world_y.resize(n)
		var ymin := 1.0e9
		for i in range(n):
			world_y[i] = (xf * verts[i]).y
			if world_y[i] < ymin:
				ymin = world_y[i]
		var k := shapes.size()
		var band: Array = []
		var band_y := ymin + BAND
		for i in range(n):
			if world_y[i] <= band_y:
				band.append(i)
		if band.size() < k:
			var order: Array = []
			for i in range(n):
				order.append(i)
			order.sort_custom(func(a, b): return world_y[a] < world_y[b])
			band = order.slice(0, mini(k, order.size()))
		var chosen := _fps_from_idx(world_y, verts, band, k)
		info["last_idx"] = chosen
		var local_up := xf.basis.inverse() * Vector3.UP
		if local_up.length_squared() < 1e-12:
			local_up = Vector3(0, 1, 0)
		else:
			local_up = local_up.normalized()
		for si in range(k):
			if chosen.is_empty():
				break
			var vi: int = int(chosen[si % chosen.size()])
			(shapes[si] as CollisionShape3D).position = verts[vi] + local_up * radius


func _sync_heels() -> void:
	for heel_name in _heel_hosts.keys():
		var host_name: String = _heel_hosts[heel_name]
		if not _bodies.has(host_name):
			continue
		var host: RigidBody3D = _bodies[host_name]
		var heel: RigidBody3D = _bodies[heel_name]
		heel.global_transform = host.global_transform
		heel.force_update_transform()
		heel.linear_velocity = host.linear_velocity
		heel.angular_velocity = host.angular_velocity


func _rebake_welds() -> void:
	for node in _heel_welds:
		var a: NodePath = node.node_a
		var b: NodePath = node.node_b
		node.node_a = NodePath()
		node.node_b = NodePath()
		node.node_a = a
		node.node_b = b


func _exclude_ancestors(child: RigidBody3D, start: PhysicsBody3D) -> void:
	# MuJoCo excludes parent–child only. 2-hop hulls (hip vs shin) still
	# overlap after downsampling, so skip parent and grandparent. hops=16
	# also dropped trunk↔foot and made the C3 pile ~15 mm too short.
	var cur: Node = start
	var hops := 0
	while cur != null and cur is RigidBody3D and hops < 2:
		var rb: RigidBody3D = cur as RigidBody3D
		if rb == child:
			break
		child.add_collision_exception_with(rb)
		rb.add_collision_exception_with(child)
		var nxt: PhysicsBody3D = null
		for j in _joints:
			if j["child"] == rb:
				nxt = j["parent"]
				break
		cur = nxt
		hops += 1


func _find_joint_node(jname: String) -> HingeJoint3D:
	if _robot == null:
		return null
	var n := _robot.find_child(jname, true, false)
	return n as HingeJoint3D


func _mujoco_body_basis(body: PhysicsBody3D, body_name: String) -> Basis:
	if body == null or not (body is RigidBody3D):
		return Basis.IDENTITY
	var qI := _basis_to_m_quat((body as RigidBody3D).global_transform.basis)
	var rI := Basis(qI)
	var iq: Vector4 = _body_iquat.get(body_name, Vector4(1, 0, 0, 0))
	var rIQ := Basis(Quaternion(iq.y, iq.z, iq.w, iq.x))
	return rI * rIQ.transposed()


func _refresh_mj_basis() -> void:
	for key in _bodies.keys():
		var name := str(key)
		_mj_basis[name] = _mujoco_body_basis(_bodies[name], name)


func _mj_basis_of(body: PhysicsBody3D, body_name: String) -> Basis:
	if _mj_basis.has(body_name):
		return _mj_basis[body_name]
	return _mujoco_body_basis(body, body_name)


func _body_rel(parent: PhysicsBody3D, parent_name: String, child: RigidBody3D, child_name: String) -> Basis:
	var rp := _mj_basis_of(parent, parent_name)
	var rc := _mj_basis_of(child, child_name)
	return rp.transposed() * rc


func _twist_about(rrel: Basis, axis: Vector3) -> float:
	axis = axis.normalized()
	var q := rrel.get_rotation_quaternion()
	var v := Vector3(q.x, q.y, q.z)
	var proj := axis * v.dot(axis)
	var tq := Quaternion(proj.x, proj.y, proj.z, q.w)
	if tq.length_squared() < 1e-16:
		return 0.0
	tq = tq.normalized()
	var imag := Vector3(tq.x, tq.y, tq.z)
	var ang := 2.0 * atan2(imag.length(), tq.w)
	if imag.dot(axis) < 0.0:
		ang = -ang
	while ang > PI:
		ang -= TAU
	while ang < -PI:
		ang += TAU
	return ang


func _joint_q(j: Dictionary) -> float:
	var rrel := _body_rel(j["parent"], str(j["parent_name"]), j["child"], str(j["child_name"]))
	var rest_t: Basis = j["rest_rel_t"] if j.has("rest_rel_t") else (j["rest_rel"] as Basis).transposed()
	# MuJoCo hinge: R_rel(q) = Rot(axis_parent, q) @ R_rel(0)
	var delta: Basis = rrel * rest_t
	return _twist_about(delta, j["axis_parent_body"])


func _axis_godot(j: Dictionary) -> Vector3:
	var node: Node3D = j["node"]
	var hz := node.global_transform.basis.z.normalized()
	# Keep the same sense as axis_parent_body (MuJoCo right-hand).
	var expected := _m2g(_mj_basis_of(j["parent"], str(j["parent_name"])) * (j["axis_parent_body"] as Vector3))
	if hz.dot(expected) < 0.0:
		hz = -hz
	return hz


func _joint_qd(j: Dictionary) -> float:
	# Hinge rate from _joint_q finite difference (MuJoCo-consistent angle),
	# not RigidBody3D.angular_velocity (Jolt solver vel ≠ pose rate).
	return float(j.get("qd_kin", 0.0))


func _basis_omega(r_new: Basis, r_old: Basis, dt: float) -> Vector3:
	if dt <= 1e-12:
		return Vector3.ZERO
	var q := (r_new * r_old.transposed()).get_rotation_quaternion()
	# Shortest-arc: q and -q are the same rotation.
	if q.w < 0.0:
		q = Quaternion(-q.x, -q.y, -q.z, -q.w)
	var imag := Vector3(q.x, q.y, q.z)
	var n := imag.length()
	if n < 1e-10:
		return Vector3.ZERO
	var angle := 2.0 * atan2(n, q.w)
	return imag * (angle / (n * dt))


func _phys_tick_dt() -> float:
	var hz := float(Engine.physics_ticks_per_second)
	if hz <= 1.0:
		return 0.005
	return 1.0 / hz


func _kin_ema_alpha(tick_dt: float) -> float:
	if KIN_VEL_TAU <= 1e-12:
		return 1.0
	return 1.0 - exp(-tick_dt / KIN_VEL_TAU)


func _snapshot_kinematic_pose() -> void:
	_kin_basis.clear()
	_kin_omega.clear()
	_kin_after_tick = false
	_refresh_mj_basis()
	for key in _bodies.keys():
		var name := str(key)
		var b: RigidBody3D = _bodies[name]
		_kin_basis[name] = b.global_transform.basis
		_kin_omega[name] = Vector3.ZERO
	for j in _joints:
		var q := _joint_q(j)
		j["q_kin_old"] = q
		j["qd_kin"] = 0.0


func _update_kinematic_vel(_delta: float) -> void:
	var tick_dt := _phys_tick_dt()
	if tick_dt <= 1e-12:
		return
	var a := _kin_ema_alpha(tick_dt)
	_refresh_mj_basis()
	for key in _bodies.keys():
		var name := str(key)
		var b: RigidBody3D = _bodies[name]
		var r_new: Basis = b.global_transform.basis
		var r_old: Basis = _kin_basis[name] if _kin_basis.has(name) else r_new
		var w_tick := _basis_omega(r_new, r_old, tick_dt)
		var w_prev: Vector3 = _kin_omega[name] if _kin_omega.has(name) else Vector3.ZERO
		_kin_omega[name] = w_prev.lerp(w_tick, a)
		_kin_basis[name] = r_new
	for j in _joints:
		var q_new := _joint_q(j)
		var q_old := float(j.get("q_kin_old", q_new))
		var qd_tick := wrapf(q_new - q_old, -PI, PI) / tick_dt
		var qd_prev := float(j.get("qd_kin", 0.0))
		j["qd_kin"] = lerpf(qd_prev, qd_tick, a)
		j["q_kin_old"] = q_new


func _setup_sole_springs() -> void:
	_sole_springs.clear()
	if not SOLE_SPRINGS:
		return
	var floor_body := get_node_or_null("World/Floor") as StaticBody3D
	if floor_body == null:
		push_warning("sole_springs: missing World/Floor")
		return
	var world := get_node("World") as Node3D
	for binfo in _spec.get("bodies", []):
		if typeof(binfo) != TYPE_DICTIONARY:
			continue
		var bname := str(binfo.get("name", ""))
		if not binfo.has("sole_corners") or not _bodies.has(bname):
			continue
		var foot: RigidBody3D = _bodies[bname]
		foot.add_collision_exception_with(floor_body)
		floor_body.add_collision_exception_with(foot)
		var raw: Array = binfo.get("sole_corners", [])
		for i in range(raw.size()):
			var v: Array = raw[i]
			if typeof(v) != TYPE_ARRAY or v.size() < 3:
				continue
			var local := Vector3(float(v[0]), float(v[1]), float(v[2]))
			var j := Generic6DOFJoint3D.new()
			j.name = "sole_spring_%s_%d" % [bname, i]
			world.add_child(j)
			for axis in ["x", "y", "z"]:
				j.set("linear_limit_%s/enabled" % axis, false)
				j.set("angular_limit_%s/enabled" % axis, false)
				j.set("linear_spring_%s/enabled" % axis, true)
				j.set("angular_spring_%s/enabled" % axis, false)
			j.set("linear_spring_x/stiffness", 0.0)
			j.set("linear_spring_x/damping", 0.0)
			j.set("linear_spring_z/stiffness", 0.0)
			j.set("linear_spring_z/damping", 0.0)
			j.set("linear_spring_y/stiffness", 0.0)
			j.set("linear_spring_y/damping", 0.0)
			j.set("linear_spring_y/equilibrium_point", 0.0)
			j.node_a = j.get_path_to(floor_body)
			j.node_b = j.get_path_to(foot)
			_sole_springs.append({"joint": j, "foot": foot, "local": local, "name": bname})
	print("sole_springs n=%d k=%.0f" % [_sole_springs.size(), SOLE_SPRING_K])


func _rebake_sole_springs() -> void:
	for s in _sole_springs:
		var foot: RigidBody3D = s["foot"]
		var j: Generic6DOFJoint3D = s["joint"]
		var local: Vector3 = s["local"]
		var corner: Vector3 = foot.global_transform * local
		j.global_transform = Transform3D(Basis.IDENTITY, Vector3(corner.x, 0.0, corner.z))
		var saved := foot.global_transform
		foot.global_transform.origin = j.global_position - saved.basis * local
		foot.force_update_transform()
		var a: NodePath = j.node_a
		var b: NodePath = j.node_b
		j.node_a = NodePath()
		j.node_b = NodePath()
		j.node_a = a
		j.node_b = b
		foot.global_transform = saved
		foot.force_update_transform()
	_update_sole_springs()


func _update_sole_springs() -> void:
	for s in _sole_springs:
		var foot: RigidBody3D = s["foot"]
		var j: Generic6DOFJoint3D = s["joint"]
		var local: Vector3 = s["local"]
		var py: float = (foot.global_transform * local).y
		var on := py < 0.0
		j.set("linear_spring_y/stiffness", SOLE_SPRING_K if on else 0.0)
		j.set("linear_spring_y/damping", SOLE_SPRING_C if on else 0.0)
		# Viscous XZ only — stiffness would nail the 4 corners and lock pitch.
		var cxz := 200.0 if on else 0.0
		j.set("linear_spring_x/stiffness", 0.0)
		j.set("linear_spring_x/damping", cxz)
		j.set("linear_spring_z/stiffness", 0.0)
		j.set("linear_spring_z/damping", cxz)


func _setup_jaw_spring() -> void:
	_jaw_spring = null
	_jaw_body = null
	_jaw_pad_y = 0.0
	if not JAW_SPRINGS:
		return
	var floor_body := get_node_or_null("World/Floor") as StaticBody3D
	if floor_body == null or not _bodies.has("jaw_soft"):
		push_warning("jaw_spring: missing Floor or jaw_soft")
		return
	_jaw_body = _bodies["jaw_soft"]
	_jaw_body.add_collision_exception_with(floor_body)
	floor_body.add_collision_exception_with(_jaw_body)
	var world := get_node("World") as Node3D
	var j := Generic6DOFJoint3D.new()
	j.name = "jaw_spring"
	world.add_child(j)
	for axis in ["x", "y", "z"]:
		j.set("linear_limit_%s/enabled" % axis, false)
		j.set("angular_limit_%s/enabled" % axis, false)
		j.set("linear_spring_%s/enabled" % axis, true)
		j.set("angular_spring_%s/enabled" % axis, false)
	j.set("linear_spring_x/stiffness", 0.0)
	j.set("linear_spring_x/damping", 0.0)
	j.set("linear_spring_z/stiffness", 0.0)
	j.set("linear_spring_z/damping", 0.0)
	j.set("linear_spring_y/stiffness", 0.0)
	j.set("linear_spring_y/damping", 0.0)
	j.set("linear_spring_y/equilibrium_point", 0.0)
	j.node_a = j.get_path_to(floor_body)
	j.node_b = j.get_path_to(_jaw_body)
	_jaw_spring = j
	print("jaw_spring on k=%.0f c=%.0f" % [JAW_SPRING_K, JAW_SPRING_C])


func _jaw_lowest_world() -> Vector3:
	if _jaw_body == null:
		return Vector3(0.0, 1.0e9, 0.0)
	var best := Vector3(0.0, 1.0e9, 0.0)
	for c in _jaw_body.get_children():
		if not (c is CollisionShape3D):
			continue
		var cs := c as CollisionShape3D
		if cs.disabled or cs.shape == null:
			continue
		var xf := cs.global_transform
		var sh := cs.shape
		if sh is ConvexPolygonShape3D:
			for p in (sh as ConvexPolygonShape3D).points:
				var w: Vector3 = xf * p
				if w.y < best.y:
					best = w
		elif sh is SphereShape3D:
			var w: Vector3 = xf.origin
			w.y -= (sh as SphereShape3D).radius
			if w.y < best.y:
				best = w
		elif sh is CapsuleShape3D:
			var cap := sh as CapsuleShape3D
			var half := 0.5 * cap.height
			for s in [-1.0, 1.0]:
				var local := Vector3(0.0, s * half, 0.0)
				var w: Vector3 = xf * local
				w.y -= cap.radius
				if w.y < best.y:
					best = w
	return best


func _recapture_jaw_spring() -> void:
	if _jaw_spring == null:
		return
	var a: NodePath = _jaw_spring.node_a
	var b: NodePath = _jaw_spring.node_b
	_jaw_spring.node_a = NodePath()
	_jaw_spring.node_b = NodePath()
	_jaw_spring.node_a = a
	_jaw_spring.node_b = b


func _rebake_jaw_spring() -> void:
	if _jaw_spring == null:
		return
	var p := _jaw_lowest_world()
	_jaw_spring.global_transform = Transform3D(Basis.IDENTITY, Vector3(p.x, 0.0, p.z))
	_recapture_jaw_spring()
	_update_jaw_spring()


func _update_jaw_spring() -> void:
	if _jaw_spring == null:
		return
	var p := _jaw_lowest_world()
	_jaw_pad_y = p.y
	var on := p.y < 0.0
	var cur := _jaw_spring.global_position
	var dxz := Vector2(p.x - cur.x, p.z - cur.z).length()
	var was_on: bool = float(_jaw_spring.get("linear_spring_y/stiffness")) > 0.0
	if on and ((not was_on) or dxz > 0.002):
		_jaw_spring.global_transform = Transform3D(Basis.IDENTITY, Vector3(p.x, 0.0, p.z))
		_recapture_jaw_spring()
	_jaw_spring.set("linear_spring_y/stiffness", JAW_SPRING_K if on else 0.0)
	_jaw_spring.set("linear_spring_y/damping", JAW_SPRING_C if on else 0.0)


func _setup_sprung_floor() -> void:
	if not SPRUNG_FLOOR:
		return
	var static_floor := get_node_or_null("World/Floor") as StaticBody3D
	if static_floor != null:
		for c in static_floor.get_children():
			if c is CollisionShape3D:
				(c as CollisionShape3D).disabled = true
		static_floor.collision_layer = 0
		static_floor.collision_mask = 0
	var world := get_node("World") as Node3D
	var anchor := StaticBody3D.new()
	anchor.name = "FloorAnchor"
	anchor.position = Vector3.ZERO
	anchor.collision_layer = 0
	anchor.collision_mask = 0
	world.add_child(anchor)
	_floor_plate = RigidBody3D.new()
	_floor_plate.name = "SprungFloor"
	_floor_plate.mass = SPRUNG_FLOOR_MASS
	_floor_plate.gravity_scale = 0.0
	_floor_plate.can_sleep = false
	_floor_plate.position = Vector3.ZERO
	_floor_plate.center_of_mass_mode = RigidBody3D.CENTER_OF_MASS_MODE_CUSTOM
	_floor_plate.center_of_mass = Vector3.ZERO
	_floor_plate.collision_layer = 1
	_floor_plate.collision_mask = 1
	_floor_plate.contact_monitor = true
	_floor_plate.max_contacts_reported = 24
	var mat := PhysicsMaterial.new()
	mat.friction = 1.0
	mat.bounce = 0.0
	_floor_plate.physics_material_override = mat
	var csh := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = Vector3(4.0, 0.02, 4.0)
	csh.shape = box
	csh.position = Vector3(0.0, -0.01, 0.0)
	_floor_plate.add_child(csh)
	world.add_child(_floor_plate)
	_floor_spring = Generic6DOFJoint3D.new()
	_floor_spring.name = "FloorSpring"
	anchor.add_child(_floor_spring)
	_floor_spring.node_a = _floor_spring.get_path_to(anchor)
	_floor_spring.node_b = _floor_spring.get_path_to(_floor_plate)
	for axis in ["x", "y", "z"]:
		_floor_spring.set("linear_limit_%s/enabled" % axis, false)
		_floor_spring.set("angular_limit_%s/enabled" % axis, false)
		_floor_spring.set("linear_spring_%s/enabled" % axis, true)
		_floor_spring.set("angular_spring_%s/enabled" % axis, true)
		_floor_spring.set("angular_spring_%s/stiffness" % axis, 1.0e5)
		_floor_spring.set("angular_spring_%s/damping" % axis, 100.0)
	_floor_spring.set("linear_spring_x/stiffness", 1.0e5)
	_floor_spring.set("linear_spring_x/damping", 200.0)
	_floor_spring.set("linear_spring_z/stiffness", 1.0e5)
	_floor_spring.set("linear_spring_z/damping", 200.0)
	_floor_spring.set("linear_spring_y/stiffness", SPRUNG_FLOOR_K)
	_floor_spring.set("linear_spring_y/damping", 89.0)
	_floor_spring.set("linear_spring_y/equilibrium_point", 0.0)
	print("sprung_floor on k=%.0f mass=%.2f" % [SPRUNG_FLOOR_K, SPRUNG_FLOOR_MASS])


func _reset_sprung_floor() -> void:
	if _floor_plate == null:
		return
	_floor_plate.global_transform = Transform3D.IDENTITY
	_floor_plate.linear_velocity = Vector3.ZERO
	_floor_plate.angular_velocity = Vector3.ZERO


func _snapshot_velocities() -> void:
	_saved_lv.clear()
	_saved_av.clear()
	for key in _bodies.keys():
		var b: RigidBody3D = _bodies[key]
		_saved_lv[key] = b.linear_velocity
		_saved_av[key] = b.angular_velocity
	if _floor_plate != null:
		_saved_lv["__floor"] = _floor_plate.linear_velocity
		_saved_av["__floor"] = _floor_plate.angular_velocity


func _restore_velocities() -> void:
	for key in _bodies.keys():
		if _pinned.has(key):
			continue
		if not _saved_lv.has(key):
			continue
		var b: RigidBody3D = _bodies[key]
		b.linear_velocity = _saved_lv[key]
		b.angular_velocity = _saved_av[key]
		b.sleeping = false
	if _floor_plate != null and _saved_lv.has("__floor"):
		_floor_plate.linear_velocity = _saved_lv["__floor"]
		_floor_plate.angular_velocity = _saved_av["__floor"]
		_floor_plate.sleeping = false


func _freeze(v: bool) -> void:
	if v:
		_snapshot_velocities()
	_frozen = v
	for key in _bodies.keys():
		if (not v) and _pinned.has(key):
			continue
		(_bodies[key] as RigidBody3D).freeze = v
	if _floor_plate != null:
		_floor_plate.freeze = v
	if not v:
		_restore_velocities()


func _refresh_after_physics(delta: float) -> void:
	# Jolt integrated after the previous _physics_process. Refresh kinematic
	# ω/qd from the new pose before PD or the step reply reads them.
	if _kin_after_tick:
		_update_kinematic_vel(delta)
		_kin_after_tick = false
		if _report_mode == "research":
			for bname in _research_bodies:
				var key := str(bname)
				if _bodies.has(key):
					var report := _body_kinematic_report(key)
					var events: Array = _research_contact_events.get(key, [])
					for contact in report["contacts"]:
						# Keep each body/shape pair once within the 20 ms action.
						var found := false
						for old in events:
							if old["body"] == contact["body"] and old["shape"] == contact["shape"]:
								found = true
								break
						if not found:
							events.append(contact)
					_research_contact_events[key] = events
	if not _headless:
		Engine.max_physics_steps_per_frame = 32
		# Camera ticks on the wall clock, not the fixed step: --fixed-fps 200
		# (lockstep pacing) would otherwise slow the damping ~12x, and a blocked
		# lockstep recv stretches dt so a resumed snap catches up in one frame.
		var cam_dt := minf(_cam_wall_seconds() - _cam_last_sec, 0.1)
		_cam_last_sec = _cam_wall_seconds()
		_follow_camera(cam_dt)


func _physics_process(delta: float) -> void:
	_refresh_after_physics(delta)
	if _peer == null:
		_try_accept()
		return
	if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		_peer = null
		_buf = PackedByteArray()
		return
	if _remaining <= 0:
		if _pending_send:
			if _timing_enabled and _timing_phys_t0 > 0:
				_timing_phys_usec = Time.get_ticks_usec() - _timing_phys_t0
			_send_state("step")
			_pending_send = false
		# Hold this physics tick until Python replies. Do not freeze-as-static:
		# Jolt zeros velocity on freeze, and even save/restore loses contact
		# warmstart — walk worked, run still dumped at t≈9 s.
		var wait_t0 := Time.get_ticks_usec()
		var wait_iters := 0
		var cmd = _recv_line()
		while cmd == null:
			if _peer == null:
				_freeze(true)
				return
			_peer.poll()
			if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
				_freeze(true)
				return
			if _headless:
				# StreamPeer.get_data() blocks via poll(IN, -1) until 1 byte.
				# No StreamPeerTCP timeout API; Python recv_timeout kills us.
				var got: Array = _peer.get_data(1)
				wait_iters += 1
				if int(got[0]) != OK:
					_freeze(true)
					return
				_buf.append_array(got[1])
			else:
				OS.delay_usec(SPIN_DELAY_USEC)
				wait_iters += 1
				var cd := minf(_cam_wall_seconds() - _cam_last_sec, 0.1)
				_cam_last_sec = _cam_wall_seconds()
				_follow_camera(cd)
			cmd = _recv_line()
		if _timing_enabled:
			_timing_wait_usec = Time.get_ticks_usec() - wait_t0
			_timing_wait_iters = wait_iters
		_handle(cmd)
		# _handle may have teleported bodies (reset). Advance the camera with
		# wall-clock time before leaving the pump, or the snap stalls ~16 ms
		# until the next _physics_process entry (zero movement while idle).
		var cd2 := minf(_cam_wall_seconds() - _cam_last_sec, 0.1)
		_cam_last_sec = _cam_wall_seconds()
		_follow_camera(cd2)
		if _remaining <= 0:
			return
		_timing_phys_t0 = Time.get_ticks_usec()
		_timing_pd_usec = 0
	_advance_physics_tick(delta)


func _advance_physics_tick(delta: float) -> void:
	var pd_t0 := Time.get_ticks_usec()
	_apply_pd()
	if not _sole.is_empty():
		_update_sole_spheres()
	if SOLE_SPRINGS:
		_update_sole_springs()
	if JAW_SPRINGS:
		_update_jaw_spring()
	_timing_pd_usec += Time.get_ticks_usec() - pd_t0
	_t += delta
	_remaining -= 1
	_kin_after_tick = true
	if _remaining <= 0:
		_pending_send = true


func _try_accept() -> void:
	if _server != null and _server.is_connection_available():
		_peer = _server.take_connection()
		_peer.set_no_delay(true)
		print("sim2sim_physics_server client connected")


func _recv_line() -> Variant:
	if _peer == null:
		return null
	# Godot 4.7 StreamPeerTCP: get_available_bytes() is the kernel socket
	# buffer (no internal ring). poll() is still required to notice FIN.
	_peer.poll()
	if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return null
	var n := _peer.get_available_bytes()
	if n > 0:
		var got: Array = _peer.get_partial_data(n)
		if int(got[0]) != OK:
			return null
		_buf.append_array(got[1])
	var idx := _buf.find(10)  # \n
	if idx < 0:
		return null
	var line := _buf.slice(0, idx).get_string_from_utf8()
	_buf = _buf.slice(idx + 1)
	if line.ends_with("\r"):
		line = line.substr(0, line.length() - 1)
	if line.is_empty():
		return _recv_line()
	return JSON.parse_string(line)


func _send_dict(d: Dictionary) -> void:
	if _peer == null:
		return
	var t0 := Time.get_ticks_usec()
	var s := JSON.stringify(d) + "\n"
	var raw := s.to_utf8_buffer()
	_timing_json_bytes = raw.size()
	_peer.put_data(raw)
	_timing_send_usec = Time.get_ticks_usec() - t0


func _handle(cmd: Variant) -> void:
	if typeof(cmd) != TYPE_DICTIONARY:
		_send_dict({"ok": false, "error": "not an object"})
		return
	var name := str(cmd.get("cmd", ""))
	if name == "hello":
		var inertias: Array = []
		for j in _joints:
			var ch: RigidBody3D = j["child"]
			var node: HingeJoint3D = j["node"]
			inertias.append({
				"joint": j["name"],
				"body": j["child_name"],
				"inertia": [ch.inertia.x, ch.inertia.y, ch.inertia.z],
				"armature": j["armature"],
				"n_i": [j["n_i"].x, j["n_i"].y, j["n_i"].z],
				"limited": bool(j.get("limited", true)),
				"hinge_enable": node.get("angular_limit/enable"),
				"hinge_lo": node.get("angular_limit/lower"),
				"hinge_hi": node.get("angular_limit/upper"),
			})
		_send_dict({
			"ok": true,
			"cmd": "hello",
			"ticks_per_second": Engine.physics_ticks_per_second,
			"nu": _ctrl.size(),
			"n_joints": _joints.size(),
			"n_bodies": _bodies.size(),
			"robot_scene": _robot_scene,
			"spec_path": _spec_path,
			"window_title": _window_title,
			"inertias": inertias,
		})
	elif name == "reset":
		_do_reset(cmd)
	elif name == "step":
		var ctrl: Array = cmd.get("ctrl", [])
		if ctrl.size() != _ctrl.size():
			_send_dict({
				"ok": false,
				"cmd": "step",
				"err": "ctrl_len",
				"got": ctrl.size(),
				"want": _ctrl.size(),
			})
			return
		if _frozen:
			_freeze(false)
		if cmd.has("hud") and _hud != null and _hud.has_method("set_status"):
			_hud.call("set_status", str(cmd.get("hud", "")))
		for i in range(ctrl.size()):
			_ctrl[i] = float(ctrl[i])
		# Optional free-ball placement shares the next counted physics step,
		# matching an episodic skill trigger without resetting the robot.
		if cmd.has("place_ball") and _bodies.has("ball"):
			var ball: RigidBody3D = _bodies["ball"]
			var bp: Array = cmd["place_ball"]
			ball.global_position = _m2g(Vector3(float(bp[0]),float(bp[1]),float(bp[2])))
			ball.force_update_transform()
			ball.linear_velocity = Vector3.ZERO
			ball.angular_velocity = Vector3.ZERO
		_report_mode = str(cmd.get("report", ""))
		_research_contact_events.clear()
		_research_capture_path = str(cmd.get("capture_path", ""))
		_timing_enabled = bool(cmd.get("timing", false))
		_remaining = int(cmd.get("n_substeps", 1))
		if _remaining < 1:
			_remaining = 1
	elif name == "set_tau_limit":
		var lim := float(cmd.get("limit", 0.0))
		var n := 0
		for j in _joints:
			if int(j["act_index"]) < 0:
				continue
			j["fmin"] = -lim
			j["fmax"] = lim
			n += 1
		_send_dict({"ok": true, "cmd": "set_tau_limit", "limit": lim, "n": n})
	elif name == "pin":
		var names: Array = cmd.get("names", [])
		for n in names:
			var key := str(n)
			if _bodies.has(key):
				_pinned[key] = true
				(_bodies[key] as RigidBody3D).freeze = true
		_send_dict({"ok": true, "cmd": "pin"})
	elif name == "unpin":
		_freeze(false)
		_send_dict({"ok": true, "cmd": "unpin"})
	elif name == "nudge":
		if _base != null:
			var lin: Array = cmd.get("linvel", [0.0, 0.0, 0.0])
			_base.linear_velocity += _m2g(Vector3(float(lin[0]), float(lin[1]), float(lin[2])))
		_send_dict({"ok": true, "cmd": "nudge"})
	elif name == "screenshot":
		var path := str(cmd.get("path", "/tmp/sim2sim-play.png"))
		var tex: ViewportTexture = get_viewport().get_texture()
		if tex == null:
			_send_dict({"ok": false, "error": "no viewport texture", "cmd": "screenshot"})
		else:
			var img: Image = tex.get_image()
			var err := img.save_png(path)
			_send_dict({"ok": err == OK, "path": path, "cmd": "screenshot", "w": img.get_width(), "h": img.get_height()})
	elif name == "camera_state":
		# Headless camera integration test (tests/test_camera_follow.py).
		_send_dict({"ok": true, "cmd": "camera_state", "camera": _camera_json_state(), "held_order": _held_press_order.duplicate()})
	elif name == "key":
		# Inject a keyboard event (visual driver): parse_input_event feeds
		# the real input pipeline — physical-key tracking for _sample_held
		# and _unhandled_input for tap actions.
		var ev := InputEventKey.new()
		ev.physical_keycode = int(cmd.get("keycode", 0))
		ev.pressed = bool(cmd.get("pressed", true))
		Input.parse_input_event(ev)
		if not ev.pressed:
			# Release: poll until physical-key tracking clears (bounded) so
			# _sample_held on the next tick does not report a stale press.
			for _i in range(2000):
				if not Input.is_physical_key_pressed(ev.physical_keycode):
					break
				OS.delay_usec(500)
		else:
			# Press: ensure the queue drained into tracking before acking.
			for _i in range(2000):
				if Input.is_physical_key_pressed(ev.physical_keycode):
					break
				OS.delay_usec(500)
		_send_dict({"ok": true, "cmd": "key"})
	elif name == "camera_zoom":
		# Headless test for wheel zoom (same path as _unhandled_input).
		_cam_dist = clampf(_cam_dist * (1.0 + CAM_DIST_DRAG * 40.0 * float(cmd.get("d", 0.0))), CAM_DIST_MIN, CAM_DIST_MAX)
		_send_dict({"ok": true, "cmd": "camera_zoom", "dist": _cam_dist})
	elif name == "close":
		_send_dict({"ok": true, "cmd": "close"})
		get_tree().quit(0)
	else:
		_send_dict({"ok": false, "error": "unknown cmd", "cmd": name})


func _do_reset(cmd: Dictionary) -> void:
	_research_bodies = cmd.get("report_bodies", [])
	_research_contact_events.clear()
	_research_capture_path = ""
	_remaining = 0
	_pending_send = false
	_t = 0.0
	_pinned.clear()
	# Normal user resets glide the camera to the new shot. Integration tests
	# may disable the snap to exercise live-turn deadzone/trailing behavior.
	_cam_snap = bool(cmd.get("camera_snap", true))
	if _cam_snap:
		_cam_snap_started = _cam_wall_seconds()
	var ctrl: Array = cmd.get("ctrl", [])
	for i in range(mini(ctrl.size(), _ctrl.size())):
		_ctrl[i] = float(ctrl[i])
	_freeze(true)
	var poses: Array = cmd.get("bodies", [])
	var applied: Array = []
	var missing: Array = []
	for p in poses:
		var bname := str(p.get("name", ""))
		if not _bodies.has(bname):
			missing.append(bname)
			continue
		var body: RigidBody3D = _bodies[bname]
		var pos_m: Array = p.get("pos", [0, 0, 0])
		var quat_wxyz: Array = p.get("quat", [1, 0, 0, 0])
		var lin_m: Array = p.get("linvel", [0, 0, 0])
		var ang_m: Array = p.get("angvel", [0, 0, 0])
		var xf := Transform3D()
		xf.origin = _m2g(Vector3(pos_m[0], pos_m[1], pos_m[2]))
		xf.basis = _m_quat_to_basis(quat_wxyz)
		body.global_transform = xf
		body.force_update_transform()
		body.linear_velocity = _m2g(Vector3(lin_m[0], lin_m[1], lin_m[2]))
		body.angular_velocity = _m2g(Vector3(ang_m[0], ang_m[1], ang_m[2]))
		applied.append(bname)
	# Reset wrote velocities while frozen; refresh the lockstep snapshot
	# so the first unfreeze does not restore a previous episode.
	_snapshot_velocities()
	_sync_heels()
	_reset_sprung_floor()
	_rebake_sole_springs()
	_rebake_jaw_spring()
	for bname in _sole.keys():
		var d: Dictionary = _sole[bname]
		d["last_idx"] = PackedInt32Array()
		d["pinned"] = false
	_update_sole_spheres()
	_rebake_joints()
	_rebake_welds()
	var dump: Array = []
	for key in _bodies.keys():
		var b: RigidBody3D = _bodies[key]
		var pm := _g2m(b.global_transform.origin)
		var qm := _basis_to_m_quat(b.global_transform.basis)
		dump.append({"name": key, "pos": [pm.x, pm.y, pm.z], "quat": [qm.w, qm.x, qm.y, qm.z]})
	_reset_dump = dump
	_reset_applied = applied
	_reset_missing = missing
	_snapshot_kinematic_pose()
	_send_state("reset")


func _apply_pd() -> void:
	_refresh_mj_basis()
	var lite := _report_mode == "lite"
	for j in _joints:
		# Unactuated wheels: XML frictionloss=0, hinge unlimited. Do not apply
		# Coulomb as body torque — on I≈5e-7 it overpowers tire-floor contact
		# and the robot stands on locked wheels.
		var ai: int = int(j["act_index"])
		if ai < 0:
			j["tau"] = 0.0
			if not lite:
				j["axis_dot"] = 0.0
			continue
		var q := _joint_q(j)
		var qd := _joint_qd(j)
		var target := 0.0
		if ai < _ctrl.size():
			target = _ctrl[ai]
		var tau: float = float(j["kp"]) * (target - q) - float(j["kv"]) * qd
		tau = clampf(tau, float(j["fmin"]), float(j["fmax"]))
		tau -= float(j["damping"]) * qd
		var fl := float(j["frictionloss"])
		if fl > 0.0:
			tau -= fl * tanh(qd / 0.05)
		var axis := _axis_godot(j)
		j["tau"] = tau
		if not lite:
			var expected := _m2g(_mj_basis_of(j["parent"], str(j["parent_name"])) * (j["axis_parent_body"] as Vector3))
			if expected.length_squared() > 1e-12:
				expected = expected.normalized()
			j["axis_dot"] = axis.dot(expected)
		var child: RigidBody3D = j["child"]
		child.apply_torque(axis * tau)
		var parent = j["parent"]
		if parent is RigidBody3D:
			(parent as RigidBody3D).apply_torque(-axis * tau)


func _wheel_dump() -> Array:
	var out: Array = []
	for j in _joints:
		if not str(j["name"]).begins_with("passive_"):
			continue
		var node: HingeJoint3D = j["node"]
		var ax := _g2m(_axis_godot(j))
		var hz := _g2m(node.global_transform.basis.z.normalized())
		var cyl := Vector3.ZERO
		var child: RigidBody3D = j["child"]
		for ch in child.get_children():
			if ch is CollisionShape3D and (ch as CollisionShape3D).shape is CylinderShape3D:
				cyl = _g2m((ch as CollisionShape3D).global_transform.basis.y.normalized())
				break
		out.append({
			"name": j["name"],
			"body": j["child_name"],
			"q": _joint_q(j),
			"qd": _joint_qd(j),
			"tau": float(j.get("tau", 0.0)),
			"kp": float(j["kp"]),
			"kv": float(j["kv"]),
			"damping": float(j["damping"]),
			"frictionloss": float(j.get("frictionloss", 0.0)),
			"fmin": float(j["fmin"]),
			"fmax": float(j["fmax"]),
			"limited": bool(j.get("limited", true)),
			"hinge_enable": node.get("angular_limit/enable"),
			"motor_enable": node.get("motor/enable"),
			"axis_m": [ax.x, ax.y, ax.z],
			"hinge_z_m": [hz.x, hz.y, hz.z],
			"cyl_y_m": [cyl.x, cyl.y, cyl.z],
			"cyl_dot_hinge": absf(cyl.dot(Vector3(hz.x, hz.y, hz.z))),
		})
	return out


func _is_foot_body(key: String) -> bool:
	if key.contains("foot"):
		return true
	if not _bodies.has(key):
		return false
	var b: RigidBody3D = _bodies[key]
	for child in b.get_children():
		if child is CollisionShape3D and str(child.name).contains("foot"):
			return true
	for g in _spec.get("geoms", []):
		if typeof(g) != TYPE_DICTIONARY:
			continue
		if str(g.get("body", "")) == key and str(g.get("name", "")).contains("foot"):
			return true
	return false


func _foot_body_names() -> Array:
	var names: Array = []
	for key in _bodies.keys():
		if _is_foot_body(str(key)):
			names.append(str(key))
	names.sort()
	return names


func _body_kinematic_report(key: String) -> Dictionary:
	var b: RigidBody3D = _bodies[key]
	var n_contacts := 0
	var impulse_sum := 0.0
	var ground_contacts := 0
	var contacts: Array = []
	var dst := PhysicsServer3D.body_get_direct_state(b.get_rid())
	if dst != null:
		n_contacts = dst.get_contact_count()
		for ci in range(n_contacts):
			impulse_sum += dst.get_contact_impulse(ci).length()
			var collider := dst.get_contact_collider_object(ci)
			var collider_name := str(collider.name) if collider is Node else "unknown"
			var is_ground := collider is StaticBody3D
			if is_ground:
				ground_contacts += 1
			var shape_idx := dst.get_contact_local_shape(ci)
			var shape_name := ""
			if shape_idx >= 0:
				var owner_id := b.shape_find_owner(shape_idx)
				var owner = b.shape_owner_get_owner(owner_id)
				if owner is Node:
					shape_name = str(owner.name)
			contacts.append({"body": collider_name, "ground": is_ground, "shape": shape_name, "impulse": dst.get_contact_impulse(ci).length()})
	var pm := _g2m(b.global_transform.origin)
	var lm := _g2m(b.linear_velocity)
	var qm := _basis_to_m_quat(b.global_transform.basis)
	return {
		"name": key,
		"contact": n_contacts > 0,
		"n_contacts": n_contacts,
		"impulse": impulse_sum,
		"pos": [pm.x, pm.y, pm.z],
		"linvel": [lm.x, lm.y, lm.z],
		"quat": [qm.w, qm.x, qm.y, qm.z],
		"ground_contact": ground_contacts > 0,
		"contacts": contacts,
	}


func _feet_report() -> Array:
	var names: Array = _foot_names if not _foot_names.is_empty() else _foot_body_names()
	var out: Array = []
	for key in names:
		if _bodies.has(key):
			out.append(_body_kinematic_report(str(key)))
	return out


func _extra_bodies_report() -> Array:
	var out: Array = []
	for key in ["jaw_soft", "top_head_shell"]:
		if _bodies.has(key):
			out.append(_body_kinematic_report(key))
	return out


func _send_state(which: String) -> void:
	_refresh_mj_basis()
	var q: Array = []
	var qd: Array = []
	# actuator order
	var nu := _ctrl.size()
	q.resize(nu)
	qd.resize(nu)
	for i in range(nu):
		q[i] = 0.0
		qd[i] = 0.0
	for j in _joints:
		var ai: int = int(j["act_index"])
		if ai >= 0 and ai < nu:
			q[ai] = _joint_q(j)
			qd[ai] = _joint_qd(j)
	var base_pos := [0.0, 0.0, 0.0]
	var base_quat := [1.0, 0.0, 0.0, 0.0]
	var base_lin := [0.0, 0.0, 0.0]
	var base_ang_local := [0.0, 0.0, 0.0]
	if _base != null:
		# Origin/basis are the inertial COM frame (matches converter). Python
		# maps to body frame for obs using spec ipos/iquat.
		var pm := _g2m(_base.global_transform.origin)
		base_pos = [pm.x, pm.y, pm.z]
		var qm := _basis_to_m_quat(_base.global_transform.basis)
		base_quat = [qm.w, qm.x, qm.y, qm.z]
		var lm := _g2m(_base.linear_velocity)
		base_lin = [lm.x, lm.y, lm.z]
		# Kinematic ω (pose FD). Jolt angular_velocity includes constraint
		# correction and is kept as base_angvel_jolt on the full reply.
		var w_world_g: Vector3 = _kin_omega.get(_base_name, Vector3.ZERO)
		var w_i: Vector3 = _base.global_transform.basis.transposed() * w_world_g
		var iq: Vector4 = _body_iquat.get(_base_name, Vector4(1, 0, 0, 0))
		var w_body: Vector3 = _quat_rotate_wxyz(iq, w_i)
		base_ang_local = [w_body.x, w_body.y, w_body.z]
		var w_jolt_g: Vector3 = _base.angular_velocity
		var w_jolt_i: Vector3 = _base.global_transform.basis.transposed() * w_jolt_g
		var w_jolt_body: Vector3 = _quat_rotate_wxyz(iq, w_jolt_i)
		_dbg_ang_world = [w_jolt_g.x, w_jolt_g.y, w_jolt_g.z]
		_dbg_ang_jolt_local = [w_jolt_body.x, w_jolt_body.y, w_jolt_body.z]
	var tau: Array = []
	tau.resize(nu)
	for i in range(nu):
		tau[i] = 0.0
	for j in _joints:
		var ai: int = int(j["act_index"])
		if ai >= 0 and ai < nu:
			tau[ai] = float(j.get("tau", 0.0))
	var lite := which == "step" and _report_mode in ["lite", "research"]
	var payload: Dictionary
	if lite:
		payload = {
			"ok": true,
			"cmd": which,
			"t": _t,
			"q": q,
			"qd": qd,
			"tau": tau,
			"base_pos": base_pos,
			"base_quat": base_quat,
			"base_linvel": base_lin,
			"base_angvel_local": base_ang_local,
			"feet": _feet_report(),
			"bodies": _extra_bodies_report(),
		}
	else:
		var dump: Array = []
		# Play viewer: skip per-body contact dump (large JSON every 20 ms). Calib is headless.
		if DisplayServer.get_name() == "headless":
			for key in _bodies.keys():
				var b: RigidBody3D = _bodies[key]
				var dpm := _g2m(b.global_transform.origin)
				var dqm := _basis_to_m_quat(b.global_transform.basis)
				var com_local := b.center_of_mass
				var com_world_m := _g2m(b.global_transform * com_local)
				var n_contacts := 0
				var impulse_sum := 0.0
				var cpos: Array = []
				var cshape: Array = []
				var dst := PhysicsServer3D.body_get_direct_state(b.get_rid())
				if dst != null:
					com_local = dst.center_of_mass_local
					com_world_m = _g2m(b.global_transform.origin + dst.center_of_mass)
					n_contacts = dst.get_contact_count()
					for ci in range(n_contacts):
						impulse_sum += dst.get_contact_impulse(ci).length()
						var wp := _g2m(dst.get_contact_collider_position(ci))
						cpos.append([wp.x, wp.y, wp.z])
						cshape.append(dst.get_contact_local_shape(ci))
				dump.append({
					"name": key,
					"pos": [dpm.x, dpm.y, dpm.z],
					"quat": [dqm.w, dqm.x, dqm.y, dqm.z],
					"com": [com_world_m.x, com_world_m.y, com_world_m.z],
					"com_local": [com_local.x, com_local.y, com_local.z],
					"com_prop": [b.center_of_mass.x, b.center_of_mass.y, b.center_of_mass.z],
					"com_mode": int(b.center_of_mass_mode),
					"mass": b.mass,
					"inertia": [b.inertia.x, b.inertia.y, b.inertia.z],
					"n_contacts": n_contacts,
					"impulse": impulse_sum,
					"linvel": [_g2m(b.linear_velocity).x, _g2m(b.linear_velocity).y, _g2m(b.linear_velocity).z],
					"angvel": [_g2m(b.angular_velocity).x, _g2m(b.angular_velocity).y, _g2m(b.angular_velocity).z],
					"cpos": cpos,
					"cshape": cshape,
				})
		var axis_dot: Array = []
		axis_dot.resize(nu)
		for i in range(nu):
			axis_dot[i] = 0.0
		for j in _joints:
			var ai: int = int(j["act_index"])
			if ai >= 0 and ai < nu:
				axis_dot[ai] = float(j.get("axis_dot", 0.0))
		payload = {
			"ok": true,
			"cmd": which,
			"t": _t,
			"q": q,
			"qd": qd,
			"tau": tau,
			"axis_dot": axis_dot,
			"base_pos": base_pos,
			"base_quat": base_quat,
			"base_linvel": base_lin,
			"base_angvel_local": base_ang_local,
			"base_angvel_jolt": _dbg_ang_jolt_local,
			"dbg_ang_world": _dbg_ang_world,
			"applied": _reset_applied,
			"missing": _reset_missing,
			"body_names": _bodies.keys(),
			"tile_dy": 0.0 if _floor_plate == null else _floor_plate.global_position.y,
			"tile_pitch": 0.0 if _floor_plate == null else rad_to_deg(_floor_plate.global_rotation.z),
			"sole_n_on": _sole_n_on(),
			"sole_ymin": _sole_ymin(),
			"sole_cxmin": _sole_cxmin(),
			"sole_cxmax": _sole_cxmax(),
			"jaw_pad_y": _jaw_pad_y,
			"held": _held_now.duplicate(),
			"held_order": _held_press_order.duplicate(),
			"taps": _taps.duplicate(),
			"time_scale": _play_time_scale(),
			"dump": dump,
			"wheels": _wheel_dump(),
		}
	if not _research_bodies.is_empty():
		var research_states: Array = []
		for bname in _research_bodies:
			var key := str(bname)
			if _bodies.has(key):
				var report := _body_kinematic_report(key)
				report["contact_events"] = _research_contact_events.get(key, [])
				research_states.append(report)
		payload["body_states"] = research_states
	if _timing_enabled:
		payload["timing"] = {
			"phys_usec": _timing_phys_usec,
			"pd_usec": _timing_pd_usec,
			"wait_usec": _timing_wait_usec,
			"wait_iters": _timing_wait_iters,
			"prev_send_usec": _timing_send_usec,
			"json_bytes": _timing_json_bytes,
		}
	# Capture on the existing step reply: a separate screenshot command adds
	# an uncounted physics tick and must not be used inside measured rollouts.
	if which == "step" and _research_capture_path != "":
		var texture := get_viewport().get_texture()
		if texture != null:
			var captured := texture.get_image()
			if captured != null:
				captured.save_png(_research_capture_path)
		_research_capture_path = ""
	_send_dict(payload)
	_taps.clear()


func _sole_n_on() -> int:
	var n := 0
	for s in _sole_springs:
		var foot: RigidBody3D = s["foot"]
		var local: Vector3 = s["local"]
		if (foot.global_transform * local).y < 0.0:
			n += 1
	return n


func _sole_ymin() -> float:
	var ymin := 0.0
	var any := false
	for s in _sole_springs:
		var foot: RigidBody3D = s["foot"]
		var local: Vector3 = s["local"]
		var py: float = (foot.global_transform * local).y
		if not any or py < ymin:
			ymin = py
			any = true
	return ymin


func _sole_cxmin() -> float:
	var xmin := 0.0
	var any := false
	for s in _sole_springs:
		var foot: RigidBody3D = s["foot"]
		var local: Vector3 = s["local"]
		var p: Vector3 = foot.global_transform * local
		if p.y >= 0.0:
			continue
		if not any or p.x < xmin:
			xmin = p.x
			any = true
	return xmin


func _sole_cxmax() -> float:
	var xmax := 0.0
	var any := false
	for s in _sole_springs:
		var foot: RigidBody3D = s["foot"]
		var local: Vector3 = s["local"]
		var p: Vector3 = foot.global_transform * local
		if p.y >= 0.0:
			continue
		if not any or p.x > xmax:
			xmax = p.x
			any = true
	return xmax


func _m2g(p: Vector3) -> Vector3:
	return Vector3(p.x, p.z, -p.y)


func _g2m(p: Vector3) -> Vector3:
	return Vector3(p.x, -p.z, p.y)


func _g_local_as_m(v: Vector3) -> Vector3:
	# Godot body local == MuJoCo inertial local (same XYZ meaning).
	return v


func _m_quat_to_basis(q: Array) -> Basis:
	var bm := Basis(Quaternion(float(q[1]), float(q[2]), float(q[3]), float(q[0])))
	var cx := _m2g(bm.x)
	var cy := _m2g(bm.y)
	var cz := _m2g(bm.z)
	return Basis(cx, cy, cz)


func _basis_to_m_quat(b: Basis) -> Quaternion:
	var bm := Basis(_g2m(b.x), _g2m(b.y), _g2m(b.z))
	return bm.get_rotation_quaternion()


func _quat_rotate_wxyz(q: Vector4, v: Vector3) -> Vector3:
	# q = (w,x,y,z), rotate v by R(q)
	var w := q.x
	var u := Vector3(q.y, q.z, q.w)
	var t := u.cross(v) * 2.0
	return v + w * t + u.cross(t)


func _paint_robot_visuals() -> void:
	## Official Graphite colourway from press kit + launch photo.
	## Mesh nodes are vis_unnamed_N_N; roles keyed by N from mjcf2godot MAPPING_REPORT
	## (top_head_shell / jaw / bottom_head_shell). Visual-only.
	if _robot == null:
		return
	if DisplayServer.get_name() == "headless" and OS.get_environment("SIM2SIM_SHOT") == "":
		return
	# Graphite shell #6c6a68
	var shell := StandardMaterial3D.new()
	shell.albedo_color = Color(0.30, 0.29, 0.28)  # darkened Graphite so it reads under bright sky
	shell.roughness = 0.58
	shell.metallic = 0.08
	# Yellow beak / trim
	var trim := StandardMaterial3D.new()
	trim.albedo_color = Color(0.98, 0.82, 0.1)
	trim.roughness = 0.5
	trim.metallic = 0.0
	# Purple accent (eye rim / foot panels on Graphite photo)
	var accent := StandardMaterial3D.new()
	accent.albedo_color = Color(0.52, 0.32, 0.7)
	accent.roughness = 0.55
	accent.metallic = 0.0
	# Black mechanical frame
	var mech := StandardMaterial3D.new()
	mech.albedo_color = Color(0.07, 0.07, 0.07)
	mech.roughness = 0.72
	mech.metallic = 0.3
	_paint_mesh_recursive(_robot, shell, trim, accent, mech)


func _mesh_id(mi: MeshInstance3D) -> int:
	var parts := str(mi.name).split("_")
	if parts.size() > 0 and parts[-1].is_valid_int():
		return int(parts[-1])
	return -1


func _paint_role_for_node(mi: MeshInstance3D) -> String:
	var id := _mesh_id(mi)
	# Explicit geom ids from MAPPING_REPORT collision mesh= names
	# top_head_shell visual ~50, jaw visual ~57, bottom_head_shell visual ~59
	if id == 57:
		return "trim"  # jaw / beak
	if id in [50, 59]:
		return "shell"  # head shells
	# eye / camera ring often near head internals under jaw_soft
	if id in [51, 52, 53]:
		return "accent"
	# small black bits / screws under head
	if id in [43, 44, 45, 46, 47, 48, 54, 55, 60, 61]:
		return "mech"
	# foot panels
	if id in [28, 30, 31, 32, 77, 78, 80, 81]:
		return "accent" if id % 2 == 1 else "trim"
	var p := mi.get_parent()
	var pname := ""
	while p != null and p != _robot:
		pname = str(p.name).to_lower()
		if p is RigidBody3D:
			break
		p = p.get_parent()
	if pname == "jaw_soft":
		# Head shells live under jaw_soft; only mesh 57 is the yellow jaw.
		return "shell"
	if pname in ["neck", "neck_pitch"]:
		return "mech"
	if pname in ["yaw_roll_motion", "yaw2roll", "bearing_roll"]:
		# head yaw/roll housings: graphite shell like photo
		return "shell"
	if pname in ["ankle_left", "ankle_right"]:
		return "accent"
	if pname in ["trunk_base", "hip_l", "hip_l_2", "upper_leg_left", "upper_leg_right", "leg", "leg_2"]:
		return "shell"
	return "shell"


func _paint_mesh_recursive(
	node: Node,
	shell: StandardMaterial3D,
	trim: StandardMaterial3D,
	accent: StandardMaterial3D,
	mech: StandardMaterial3D,
) -> void:
	if node is MeshInstance3D:
		var mi := node as MeshInstance3D
		var role := _paint_role_for_node(mi)
		var mat := shell
		match role:
			"trim":
				mat = trim
			"accent":
				mat = accent
			"mech":
				mat = mech
			_:
				mat = shell
		var sc := mi.mesh.get_surface_count() if mi.mesh != null else 1
		for s in range(maxi(sc, 1)):
			mi.set_surface_override_material(s, mat)
	for child in node.get_children():
		_paint_mesh_recursive(child, shell, trim, accent, mech)


func _setup_floor_checker() -> void:
	## Paint the floor MeshInstance3D with a procedural checkerboard so you can
	## see motion. Works whether or not a texture file exists.
	if DisplayServer.get_name() == "headless":
		return
	var mesh_inst: MeshInstance3D = get_node_or_null("World/Floor/FloorMesh")
	if mesh_inst == null:
		return

	# Build a tiny 2×2 RGBA8 checker image then tile it via UV scale on the material.
	var img := Image.create(2, 2, false, Image.FORMAT_RGBA8)
	img.set_pixel(0, 0, Color(0.18, 0.32, 0.18))
	img.set_pixel(1, 1, Color(0.18, 0.32, 0.18))
	img.set_pixel(1, 0, Color(0.85, 0.82, 0.72))
	img.set_pixel(0, 1, Color(0.85, 0.82, 0.72))
	var tex := ImageTexture.create_from_image(img)

	var mat := StandardMaterial3D.new()
	mat.albedo_texture = tex
	mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST
	mat.uv1_scale = Vector3(24, 24, 1)
	mat.roughness = 0.9
	mat.metallic = 0.0
	mat.cull_mode = BaseMaterial3D.CULL_BACK
	mesh_inst.set_surface_override_material(0, mat)


func _play_time_scale() -> float:
	if _hud == null:
		return 1.0
	var v = _hud.get("time_scale")
	if typeof(v) != TYPE_FLOAT and typeof(v) != TYPE_INT:
		return 1.0
	return clampf(float(v), 0.25, 3.0)


func _setup_play_ui() -> void:
	if DisplayServer.get_name() == "headless":
		return
	var hud_script: Script = load("res://play_hud.gd")
	if hud_script == null:
		push_warning("play_hud.gd missing")
		return
	_hud = CanvasLayer.new()
	_hud.set_script(hud_script)
	add_child(_hud)
	if _hud.has_signal("tap"):
		_hud.connect("tap", Callable(self, "_on_hud_tap"))


func _on_hud_tap(action: String) -> void:
	_add_tap(action)


func _add_tap(action: String) -> void:
	if not _taps.has(action):
		_taps.append(action)


func _process(delta: float) -> void:
	if _headless:
		return
	_sample_held()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		var now := _cam_wall_seconds()
		if mb.button_index == MOUSE_BUTTON_RIGHT:
			_cam_dragging = mb.pressed
			if not mb.pressed:
				_cam_manual_until = now + CAM_RECOVER_DELAY
			get_viewport().set_input_as_handled()
			return
		if mb.button_index == MOUSE_BUTTON_WHEEL_UP and mb.pressed:
			_cam_dist = clampf(_cam_dist / (1.0 + CAM_DIST_DRAG * 40.0), CAM_DIST_MIN, CAM_DIST_MAX)
			_cam_manual_until = now + CAM_RECOVER_DELAY
			get_viewport().set_input_as_handled()
			return
		if mb.button_index == MOUSE_BUTTON_WHEEL_DOWN and mb.pressed:
			_cam_dist = clampf(_cam_dist * (1.0 + CAM_DIST_DRAG * 40.0), CAM_DIST_MIN, CAM_DIST_MAX)
			_cam_manual_until = now + CAM_RECOVER_DELAY
			get_viewport().set_input_as_handled()
			return
	elif event is InputEventMouseMotion and _cam_dragging:
		var mm := event as InputEventMouseMotion
		_cam_yaw = fmod(_cam_yaw + mm.relative.x * CAM_YAW_DRAG, TAU)
		_cam_pitch = clampf(_cam_pitch - mm.relative.y * CAM_PITCH_DRAG, CAM_PITCH_MIN, CAM_PITCH_MAX)
		_cam_manual_until = _cam_wall_seconds() + CAM_RECOVER_DELAY
		return
	if not (event is InputEventKey):
		return
	var k := event as InputEventKey
	if not k.pressed or k.echo:
		return
	match k.physical_keycode:
		KEY_G, KEY_1:
			_add_tap("pick")
		KEY_Y, KEY_2:
			_add_tap("sit")
		KEY_K, KEY_3:
			_add_tap("kick_left")
		KEY_L, KEY_4:
			_add_tap("kick_right")
		KEY_R, KEY_5:
			_add_tap("roulade")
		KEY_6:
			_add_tap("switch_robot")
		KEY_7:
			_add_tap("stand")
		KEY_0, KEY_BACKSPACE:
			_add_tap("reset")
		KEY_P:
			_add_tap("push")
		KEY_ESCAPE:
			_add_tap("quit")
			get_viewport().set_input_as_handled()


func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.physical_keycode == KEY_SHIFT and event.location == KEY_LOCATION_LEFT:
		_left_shift_down = event.pressed

func _notification(what: int) -> void:
	if what == NOTIFICATION_APPLICATION_FOCUS_OUT:
		_left_shift_down = false

func _sample_held() -> void:
	var held: Array = []
	if _left_shift_down:
		held.append("sprint")
	if Input.is_physical_key_pressed(KEY_W) or Input.is_physical_key_pressed(KEY_UP):
		held.append("fwd")
	if Input.is_physical_key_pressed(KEY_S) or Input.is_physical_key_pressed(KEY_DOWN):
		held.append("back")
	if Input.is_physical_key_pressed(KEY_A) or Input.is_physical_key_pressed(KEY_LEFT):
		held.append("left")
	if Input.is_physical_key_pressed(KEY_D) or Input.is_physical_key_pressed(KEY_RIGHT):
		held.append("right")
	if Input.is_physical_key_pressed(KEY_Q):
		held.append("strafe_l")
	if Input.is_physical_key_pressed(KEY_E):
		held.append("strafe_r")
	if Input.is_physical_key_pressed(KEY_SPACE):
		held.append("idle")
	if _hud != null:
		var extra = _hud.get("extra_held")
		if extra is Array:
			for bit in extra:
				var s := str(bit)
				if not held.has(s):
					held.append(s)
	# Newest-wins order for opposing pairs (docs §B): a bit keeps its slot
	# while held; a re-press moves it to newest. Release removes it.
	var now_set := {}
	for bit in held:
		now_set[bit] = true
	for bit in now_set.keys():
		if not _prev_held_set.has(bit):
			_held_press_order.erase(bit)
			_held_press_order.append(bit)
	_prev_held_set = now_set
	_held_press_order = _held_press_order.filter(func(b): return now_set.has(b))
	_held_now = held


func _cam_wall_seconds() -> float:
	## Wall seconds. Time.get_ticks_usec() was verified real-time even under
	## --fixed-fps (Godot 4.7), so this is the engine's monotonic clock.
	return float(Time.get_ticks_usec()) / 1e6


func _duck_forward_godot() -> Vector3:
	## MuJoCo body-local axes apply as-is in Godot: for trunk_base the
	## character forward is local +X (local -Z is the vertical), so the old
	## (0,0,-1) probe was the duck's belly, not its gaze.
	if _base == null:
		return Vector3.ZERO
	return _base.global_transform.basis * Vector3(1, 0, 0)


func _follow_camera(frame_dt: float) -> void:
	## Cinemachine-style orbit follow. Keeps the duck composed from a
	## smoothly damped orbit around a look target that leads the duck's
	## motion; the duck's own yaw only drags the camera past a deadzone, so
	## in-place turns don't spin the view. Right-drag orbits freely (auto
	## follow resumes after CAM_RECOVER_DELAY idle), wheel zooms, reset
	## glides in with a fast snap instead of a hard cut.
	if _base == null:
		return
	var cam := get_node_or_null("World/Camera3D") as Camera3D
	if cam == null:
		return
	var dt := clampf(frame_dt, 0.0, 0.1)

	if not _cam_inited:
		_cam_look = _base.global_position + Vector3(0, 0.08, 0)
		_cam_inited = true
	var base_pos: Vector3 = _base.global_position
	var target: Vector3 = base_pos + Vector3(0, 0.08, 0)
	# Camera-behind azimuth in the _orbit_offset convention (yaw from +Z
	# toward +X). Local +X is the duck's MuJoCo forward. For a Godot-world
	# forward (x, z), the point behind the duck is (-x, -z), hence atan2(-x,
	# -z). This is the orbit yaw that should track the duck after deadzone.
	var heading_now := CAM_DEFAULT_YAW
	var fwd := _duck_forward_godot()
	fwd.y = 0.0
	if fwd.length_squared() > 1e-6:
		heading_now = atan2(-fwd.x, -fwd.z)
	# Horizontal look-ahead lead so the duck walks into the frame, not off it.
	var lv := _base.linear_velocity
	lv.y = 0.0
	if not _cam_snap and lv.length() > 0.02:
		var lead := lv.normalized() * minf(lv.length() * CAM_LEAD_DIST, CAM_LEAD_MAX)
		lead.y = 0.0
		target += lead
	if _cam_snap:
		var k := 1.0 - exp(-CAM_SNAP_K * dt)
		_cam_look = _cam_look.lerp(target, k)
	else:
		# Split horizontal / vertical exponential follow (frame-rate independent).
		var kh := 1.0 - exp(-CAM_POS_SMOOTH_H * dt)
		var kv := 1.0 - exp(-CAM_POS_SMOOTH_V * dt)
		_cam_look.x = lerpf(_cam_look.x, target.x, kh)
		_cam_look.z = lerpf(_cam_look.z, target.z, kh)
		_cam_look.y = lerpf(_cam_look.y, target.y, kv)
	if _cam_snap:
		# During a reset snap, chase the duck exactly: heading_now already is
		# the orbit yaw directly *behind* the duck, with no deadzone trail.
		# The trailing goal only applies to live turning, so after the snap
		# the camera stays parked straight behind until the duck turns past
		# the deadzone again. Release by wall clock so a low frame rate
		# can't strand the snap on forever.
		var ks := 1.0 - exp(-CAM_SNAP_K * dt)
		_cam_yaw = wrapf(_cam_yaw + wrapf(heading_now - _cam_yaw, -PI, PI) * ks, -PI, PI)
		_cam_pitch = clampf(_cam_pitch + (CAM_DEFAULT_PITCH - _cam_pitch) * ks, CAM_PITCH_MIN, CAM_PITCH_MAX)
		var want0 := _cam_look + _orbit_offset(_cam_yaw, _cam_pitch, _cam_dist)
		cam.look_at_from_position(want0, _cam_look, Vector3.UP)
		var snap_now := _cam_wall_seconds()
		if snap_now - _cam_snap_started > 0.35:
			_cam_snap = false  # snap glide done: normal trailing follow resumes
		return
	var now := _cam_wall_seconds()
	var manual := _cam_dragging or now < _cam_manual_until
	if not manual:
		# Trailing yaw (Cinemachine orbital follow): the duck's own yaw only
		# drags the camera once the heading leaves the current camera
		# azimuth by more than the deadzone; the goal sits one deadzone
		# short of the heading so the camera trails the action.
		var diff := wrapf(heading_now - _cam_yaw, -PI, PI)
		if absf(diff) > CAM_YAW_DEADZONE:
			var goal := wrapf(heading_now - signf(diff) * CAM_YAW_DEADZONE, -PI, PI)
			var derr := wrapf(goal - _cam_yaw, -PI, PI)
			if absf(derr) > 0.001:
				_cam_yaw = wrapf(_cam_yaw + derr * (1.0 - exp(-CAM_YAW_SMOOTH * dt)), -PI, PI)
		var kd := 1.0 - exp(-CAM_DIST_SMOOTH * dt)
		_cam_pitch += (CAM_DEFAULT_PITCH - _cam_pitch) * kd
	_cam_pitch = clampf(_cam_pitch, CAM_PITCH_MIN, CAM_PITCH_MAX)
	var offset := _orbit_offset(_cam_yaw, _cam_pitch, _cam_dist)
	var want := _cam_look + offset
	if cam.global_position.distance_to(want) < 0.015 and not manual:
		return  # settled: skip redundant look_at (avoids micro-jitter)
	# Mild rig lag: orbit point eases in (TPS spring feel), then look at target.
	var kr := 1.0 - exp(-CAM_RIG_SMOOTH * dt)
	var pos := cam.global_position.lerp(want, kr)
	cam.look_at_from_position(pos, _cam_look, Vector3.UP)


func _orbit_offset(yaw: float, pitch: float, dist: float) -> Vector3:
	## Yaw measured from world +Z toward +X; pitch up from horizontal.
	## yaw=45°, pitch=0.5, dist=1.0 ≈ the legacy fixed (0.65, 0.42, 0.65) corner.
	var cp := cos(pitch)
	return Vector3(sin(yaw) * cp, sin(pitch), cos(yaw) * cp) * dist


func _camera_json_state() -> Dictionary:
	## Exposed for the headless camera integration test (sim2sim/camera_test.py).
	if _base == null:
		return {}
	var cam := get_node_or_null("World/Camera3D") as Camera3D
	if cam == null:
		return {}
	var look: Vector3 = _base.global_position + Vector3(0, 0.08, 0)
	var fwd_g := _duck_forward_godot()
	var fwd := _g2m(fwd_g)  # MuJoCo-world forward
	var base_m := _g2m(_base.global_position)  # MuJoCo-world base pos (ground x/y)
	return {
		"cam_pos": [cam.global_position.x, cam.global_position.y, cam.global_position.z],
		"look": [look.x, look.y, look.z],
		"base": [float(base_m.x), float(base_m.y), float(base_m.z)],
		"duck_fwd": [float(fwd.x), float(fwd.y)],
		"yaw": _cam_yaw,
		"pitch": _cam_pitch,
		"dist": _cam_dist,
		"cam_look": [_cam_look.x, _cam_look.y, _cam_look.z],
	}
