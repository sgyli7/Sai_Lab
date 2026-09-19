extends Node3D
## S2: custom inertia free spin + collision layer/mask overlap test.

var _server: TCPServer
var _peer: StreamPeerTCP
var _buf: PackedByteArray = PackedByteArray()
var _port: int = 9876
var _remaining: int = 0
var _pending_send: bool = false
var _t: float = 0.0
var _spin: RigidBody3D
var _a: RigidBody3D
var _b: RigidBody3D
var _c: RigidBody3D
var _d: RigidBody3D

func _ready() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--port="):
			_port = int(arg.substr(7))
	Engine.max_physics_steps_per_frame = 1
	_spin = $SpinBody
	_a = $SphereA
	_b = $SphereB
	_c = $SphereC
	_d = $SphereD
	_a.max_contacts_reported = 8
	_b.max_contacts_reported = 8
	_c.max_contacts_reported = 8
	_d.max_contacts_reported = 8
	_a.freeze = false
	_b.freeze = false
	_c.freeze = false
	_d.freeze = false
	_spin.angular_velocity = Vector3(0, 5, 0)  # Godot Y = MuJoCo Z
	_server = TCPServer.new()
	var err := _server.listen(_port, "127.0.0.1")
	if err != OK:
		push_error("S2 listen failed")
		get_tree().quit(1)
		return
	print("spike_mass_layers listening 127.0.0.1:%s" % _port)


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
			_send(_state("step"))
			_pending_send = false
		var cmd = _recv_line()
		if cmd == null:
			return
		var name := str(cmd.get("cmd", ""))
		if name == "hello":
			_send({"ok": true, "cmd": "hello", "ticks_per_second": Engine.physics_ticks_per_second})
		elif name == "step":
			_remaining = int(cmd.get("n_substeps", 1))
		elif name == "close":
			_send({"ok": true, "cmd": "close"})
			get_tree().quit(0)
			return
		if _remaining <= 0:
			return
	_t += delta
	_remaining -= 1
	if _remaining <= 0:
		_pending_send = true


func _state(which: String) -> Dictionary:
	var w: Vector3 = _spin.angular_velocity
	return {
		"ok": true,
		"cmd": which,
		"t": _t,
		"omega": [w.x, w.y, w.z],
		"pos": [_spin.global_position.x, _spin.global_position.y, _spin.global_position.z],
		"contacts_ab": _a.get_contact_count(),
		"contacts_cd": _c.get_contact_count(),
		"layer_a": _a.collision_layer,
		"mask_a": _a.collision_mask,
		"layer_b": _b.collision_layer,
		"mask_b": _b.collision_mask,
	}


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
