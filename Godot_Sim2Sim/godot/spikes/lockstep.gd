extends Node
## S1: blocking TCP lockstep. Counts physics ticks vs requested substeps.

var _server: TCPServer
var _peer: StreamPeerTCP
var _buf: PackedByteArray = PackedByteArray()
var _port: int = 9876
var _ticks: int = 0
var _remaining: int = 0
var _pending_send: bool = false
var _t: float = 0.0
var _last_dt: float = 0.0

func _ready() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--port="):
			_port = int(a.substr(7))
	Engine.max_physics_steps_per_frame = 1
	_server = TCPServer.new()
	var err := _server.listen(_port, "127.0.0.1")
	if err != OK:
		push_error("S1 listen failed %s" % err)
		get_tree().quit(1)
		return
	print("spike_lockstep listening 127.0.0.1:%s ticks_per_second=%s" % [_port, Engine.physics_ticks_per_second])


func _physics_process(delta: float) -> void:
	_last_dt = delta
	if _peer == null:
		if _server.is_connection_available():
			_peer = _server.take_connection()
			_peer.set_no_delay(true)
			print("spike_lockstep client connected")
		return
	if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		_peer = null
		return
	if _remaining <= 0:
		if _pending_send:
			_send({"ok": true, "cmd": "step", "ticks": _ticks, "t": _t, "dt": _last_dt})
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
			_send({"ok": true, "cmd": "close", "ticks": _ticks})
			get_tree().quit(0)
			return
		if _remaining <= 0:
			return
	_ticks += 1
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
