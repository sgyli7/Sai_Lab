"""Godot/Jolt lockstep backend. Protocol is MuJoCo Z-up; Godot converts internally."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim2sim.backends import SimState
from sim2sim.coords import mat_to_quat_wxyz, quat_wxyz_to_mat
from sim2sim.godot_proc import GODOT_PROJECT, spawn_godot, stop_godot


def inertial_to_body(
    pos_i: np.ndarray, quat_i: np.ndarray, ipos: np.ndarray, iquat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r_i = quat_wxyz_to_mat(quat_i)
    r_iq = quat_wxyz_to_mat(iquat)
    r_b = r_i @ r_iq.T
    p_b = np.asarray(pos_i, dtype=np.float64) - r_b @ np.asarray(ipos, dtype=np.float64)
    return p_b, mat_to_quat_wxyz(r_b)


def _robot_user_args(spec_path: Path) -> list[str]:
    spec = Path(spec_path).resolve()
    root = GODOT_PROJECT.resolve()
    try:
        rel = spec.relative_to(root)
    except ValueError:
        return []
    tscn = rel.with_name("robot.tscn").as_posix()
    return [f"--spec=res://{rel.as_posix()}", f"--robot-scene=res://{tscn}"]


class GodotBackend:
    name = "godot"

    def __init__(
        self,
        spec_path: Path,
        *,
        timestep: float = 0.005,
        headless: bool = True,
        scene: str = "res://main.tscn",
        base_body: str = "trunk_base",
        current_limit_a: float = 0.0,
        recv_timeout: float = 120.0,
    ) -> None:
        self.spec_path = Path(spec_path)
        self.spec = json.loads(self.spec_path.read_text())
        self.dt = float(timestep)
        self.nu = int(self.spec["nu"])
        self.base_body = base_body
        bodies = {b["name"]: b for b in self.spec["bodies"]}
        self._base_meta = bodies[base_body]
        self._ipos = np.asarray(self._base_meta["ipos"], dtype=np.float64).reshape(3)
        self._iquat = np.asarray(self._base_meta["iquat_wxyz"], dtype=np.float64).reshape(4)
        extra = _robot_user_args(self.spec_path)
        self._proc, self._port, self._client = spawn_godot(
            scene, headless=headless, extra_args=extra, recv_timeout=recv_timeout
        )
        try:
            hello = self._client.call({"cmd": "hello"})
            if not hello.get("ok"):
                raise RuntimeError(f"Godot hello failed: {hello}")
            self._hello = hello
            if current_limit_a and current_limit_a > 0:
                from sim2sim.backends.mujoco_backend import XL330_M6_KT

                lim = XL330_M6_KT * float(current_limit_a)
                ack = self._client.call({"cmd": "set_tau_limit", "limit": lim})
                if not ack.get("ok"):
                    raise RuntimeError(f"set_tau_limit failed: {ack}")
        except Exception as error:
            log=stop_godot(self._proc,self._client)
            raise RuntimeError(f'Godot startup handshake failed: {error}\n{log}') from error

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def reset(
        self,
        *,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
        keyframe: str | None = None,
        ctrl: np.ndarray | None = None,
        bodies: list[dict] | None = None,
        pin_base: bool = False,
        report_bodies: list[str] | None = None,
    ) -> SimState:
        payload: dict = {"cmd": "reset"}
        if ctrl is not None:
            payload["ctrl"] = np.asarray(ctrl, dtype=float).tolist()
        if bodies is not None:
            payload["bodies"] = bodies
        if report_bodies is not None:
            payload["report_bodies"] = report_bodies
        msg = self._client.call(payload)
        if pin_base:
            self._client.call({"cmd": "pin", "names": [self.base_body]})
        return self._parse(msg)

    def send_step(
        self,
        ctrl: np.ndarray,
        n_substeps: int = 1,
        *,
        hud: str | None = None,
        report: str | None = None,
        timing: bool = False,
        capture_path: str | None = None,
        place_ball: list[float] | None = None,
    ) -> None:
        payload: dict = {
            "cmd": "step",
            "ctrl": np.asarray(ctrl, dtype=float).reshape(-1).tolist(),
            "n_substeps": int(n_substeps),
        }
        if hud is not None:
            payload["hud"] = hud
        if report is not None:
            payload["report"] = report
        if timing:
            payload["timing"] = True
        if capture_path is not None:
            payload["capture_path"] = str(capture_path)
        if place_ball is not None:
            payload["place_ball"] = place_ball
        self._client.send(payload)

    def recv_step(self) -> SimState:
        return self._parse(self._client.recv())

    def step(
        self,
        ctrl: np.ndarray,
        n_substeps: int = 1,
        *,
        hud: str | None = None,
        report: str | None = None,
        timing: bool = False,
        place_ball: list[float] | None = None,
    ) -> SimState:
        self.send_step(ctrl, n_substeps, hud=hud, report=report, timing=timing, place_ball=place_ball)
        return self.recv_step()

    def nudge(self, linvel_mujoco: np.ndarray) -> None:
        ack = self._client.call(
            {"cmd": "nudge", "linvel": np.asarray(linvel_mujoco, dtype=float).reshape(3).tolist()}
        )
        if not ack.get("ok"):
            raise RuntimeError(f"nudge failed: {ack}")

    def _parse(self, msg: dict) -> SimState:
        if not msg.get("ok"):
            raise RuntimeError(f"Godot error: {msg}")
        q = np.asarray(msg["q"], dtype=np.float64)
        qd = np.asarray(msg["qd"], dtype=np.float64)
        pos_i = np.asarray(msg["base_pos"], dtype=np.float64)
        quat_i = np.asarray(msg["base_quat"], dtype=np.float64)
        pos_b, quat_b = inertial_to_body(
            pos_i,
            quat_i,
            self._ipos,
            self._iquat,
        )
        extra: dict = {"inertial_pos": pos_i, "inertial_quat": quat_i, "raw": msg}
        if "feet" in msg:
            extra["feet"] = msg["feet"]
        if "bodies" in msg:
            extra["bodies"] = msg["bodies"]
        if "body_states" in msg:
            extra["body_states"] = msg["body_states"]
        return SimState(
            t=float(msg.get("t", 0.0)),
            q=q,
            qd=qd,
            base_pos=pos_b,
            base_quat_wxyz=quat_b,
            base_linvel=np.asarray(msg["base_linvel"], dtype=np.float64),
            base_angvel_local=np.asarray(msg["base_angvel_local"], dtype=np.float64),
            extra=extra,
        )

    def close(self) -> None:
        stop_godot(self._proc, self._client)
