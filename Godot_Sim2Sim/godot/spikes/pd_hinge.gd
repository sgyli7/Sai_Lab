extends Node3D
## S3: hinge + apply_torque PD. Protocol reports q, qd in MuJoCo Z-up hinge-about-Y.

const KP := 0.55
const KV := 0.0
const DAMPING := 0.053
const FMIN := -0.96
const FMAX := 0.96

var _server: TCPServer
var _peer: StreamPeerTCP
var _buf: PackedByteArray = PackedByteArray()
var _port: int = 9876
var _remaining: int = 0
var _pending_send: bool = false
var _t: float = 0.0
var _ctrl: float = 0.0
var _link: RigidBody3D
var _hinge: HingeJoint3D
var _rest: float = 0.0
var _child_ref_local: Vector3

func _ready() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--port="):
			_port = int(arg.substr(7))
	Engine.max_physics_steps_per_frame = 1
	_link = $Link
	_hinge = $Anchor/Hinge
	_link.can_sleep = false
	_child_ref_local = _link.global_transform.basis.transposed() * _hinge.global_transform.basis.x
	_rest = _raw_angle()
	_server = TCPServer.new()
	var err := _server.listen(_port, "127.0.0.1")
	if err != OK:
		push_error("S3 listen failed")
		get_tree().quit(1)
		return
	print("spike_pd_hinge listening 127.0.0.1:%s" % _port)


func _raw_angle() -> float:
	var axis := _hinge.global_transform.basis.z.normalized()
	var p_ref := _hinge.global_transform.basis.x
	p_ref = (p_ref - axis * p_ref.dot(axis)).normalized()
	var c_ref := _link.global_transform.basis * _child_ref_local
	c_ref = (c_ref - axis * c_ref.dot(axis)).normalized()
	return atan2(p_ref.cross(c_ref).dot(axis), p_ref.dot(c_ref))


func _q() -> float:
	return _raw_angle() - _rest


func _qd() -> float:
	var axis := _hinge.global_transform.basis.z.normalized()
	return _link.angular_velocity.dot(axis)


func _physics_process(delta: float) -> void:
	if _peer == null:
		if _server.is_connection_available():
			_peer = _server.take_connection()
			_peer.set_no_delay(true)
		return
	if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		_peer = null
		return
	if _remaining <= 0:
		if _pending_send:
			_send({"ok": true, "cmd": "step", "t": _t, "q": [_q()], "qd": [_qd()]})
			_pending_send = false
		var cmd = _recv_line()
		if cmd == null:
			return
		var name := str(cmd.get("cmd", ""))
		if name == "hello":
			_send({"ok": true, "cmd": "hello", "ticks_per_second": Engine.physics_ticks_per_second})
		elif name == "reset":
			_link.freeze = true
			_link.global_transform = Transform3D(Basis.IDENTITY, Vector3(0, 0.4, 0))
			_link.linear_velocity = Vector3.ZERO
			_link.angular_velocity = Vector3.ZERO
			_ctrl = 0.0
			_t = 0.0
			_link.freeze = false
			_send({"ok": true, "cmd": "reset", "t": 0.0, "q": [_q()], "qd": [_qd()]})
		elif name == "step":
			var ctrl: Array = cmd.get("ctrl", [0.0])
			_ctrl = float(ctrl[0]) if ctrl.size() > 0 else 0.0
			_remaining = int(cmd.get("n_substeps", 1))
		elif name == "close":
			_send({"ok": true, "cmd": "close"})
			get_tree().quit(0)
			return
		if _remaining <= 0:
			return
	var q := _q()
	var qd := _qd()
	var tau: float = KP * (_ctrl - q) - KV * qd
	tau = clampf(tau, FMIN, FMAX)
	tau -= DAMPING * qd
	var axis := _hinge.global_transform.basis.z.normalized()
	_link.apply_torque(axis * tau)
	_t += delta
	_remaining -= 1
	if _remaining <= 0:
		_pending_send = true


func _recv_line() -> Variant:
	while true:
		if _peer == null or _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			return null
		var n := _peer.get_available_bytes()
		if n > 0:
			var got: Array = _peer.get_data(n)
			if int(got[0]) != OK:
				return null
			_buf.append_array(got[1])
		var idx := _buf.find(10)
		if idx >= 0:
			var line := _buf.slice(0, idx).get_string_from_utf8()
			_buf = _buf.slice(idx + 1)
			if line.is_empty():
				continue
			return JSON.parse_string(line)
		OS.delay_usec(200)
	return null


func _send(d: Dictionary) -> void:
	_peer.put_data((JSON.stringify(d) + "\n").to_utf8_buffer())
