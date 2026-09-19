"""JSON-line TCP client used by GodotBackend and spike runners."""

from __future__ import annotations

import json
import socket
import time
from typing import Any


class JsonLineClient:
    def __init__(self, host: str, port: int, timeout: float = 120.0) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout)
        self._buf = b""
        self.t_send = 0.0
        self.t_recv_wait = 0.0
        self.t_loads = 0.0
        self.n_recv = 0
        self.last_nbytes = 0

    def send(self, obj: dict[str, Any]) -> None:
        t0 = time.perf_counter()
        self.sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))
        self.t_send += time.perf_counter() - t0

    def recv(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                line, self._buf = self._buf[:nl], self._buf[nl + 1 :]
                if not line.strip():
                    t0 = time.perf_counter()
                    continue
                t1 = time.perf_counter()
                obj = json.loads(line)
                t2 = time.perf_counter()
                self.t_recv_wait += t1 - t0
                self.t_loads += t2 - t1
                self.n_recv += 1
                self.last_nbytes = len(line)
                return obj
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("Godot closed the TCP connection")
            self._buf += chunk

    def call(self, obj: dict[str, Any]) -> dict[str, Any]:
        self.send(obj)
        return self.recv()

    def call_expect(self, obj: dict[str, Any], cmd: str) -> dict[str, Any]:
        """Like call(), but drops async messages (step_result etc.) whose
        'cmd' differs, so probe commands work while the sim is running."""
        self.send(obj)
        while True:
            msg = self.recv()
            if msg.get("cmd") == cmd:
                return msg

    def close(self) -> None:
        try:
            self.send({"cmd": "close"})
            try:
                self.recv()
            except Exception:
                pass
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


def wait_connect(
    host: str, port: int, timeout: float = 20.0, recv_timeout: float = 120.0
) -> JsonLineClient:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            return JsonLineClient(host, port, timeout=recv_timeout)
        except OSError as e:
            last = e
            time.sleep(0.05)
    raise ConnectionError(f"could not connect to {host}:{port}: {last}")
