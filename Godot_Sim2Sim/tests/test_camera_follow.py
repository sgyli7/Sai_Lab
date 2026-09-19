"""Headless Godot camera-geometry integration (docs/research_3c_camera.md §C).

Drives the orbit-follow camera through the TCP protocol only (no window):
spawn, teleport/reset, big turns, small turns, wheel zoom. All ground
math runs in MuJoCo coordinates: the protocol's `base` is the MuJoCo
world duck position, `duck_fwd` the MuJoCo world forward — so the
expected orbit yaw is derived from the *converted* duck heading, never
the raw teleport yaw. Convention (physics_server._follow_camera /
_orbit_offset): a camera at Godot orbit yaw y sits at Godot world
(sin y, cos y) from its look target; after conversion to MuJoCo ground
coordinates that is (sin y, -cos y). The exact-behind orbit yaw is
therefore MuJoCo_heading - 90°; live follow may trail by the deadzone.

The client never goes idle: `camera_state` probes are fire-and-forget and
drained by hand, which keeps the server's lockstep poll pump — and
therefore the camera damping — running.
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

from sim2sim.godot_proc import GODOT_PROJECT, spawn_godot, stop_godot
from sim2sim.runner import apply_home_qpos, load_robot_cfg
from sim2sim.paths import sim2sim_root

ROOT = sim2sim_root()
CFG_PATH = ROOT / "robots/microduck.json"


def _home_poses():
    """Body poses at home qpos (MuJoCo frame), from the reference backend."""
    from sim2sim.backends.mujoco_backend import MujocoBackend

    cfg = load_robot_cfg(CFG_PATH)
    be = MujocoBackend(ROOT / cfg["mjcf"])
    apply_home_qpos(be, np.asarray(cfg.get("home", []), dtype=np.float64), z=0.125)
    poses = be.body_poses_mujoco()
    be.close()
    return poses


def _teleport_poses(base_xy=(0.0, 0.0), yaw_rad=0.0):
    """Rigidly transform the complete home pose around MuJoCo world Z."""
    base_xy = np.asarray(base_xy, dtype=np.float64)
    poses = _home_poses()
    home_base = np.asarray(
        next(p["pos"] for p in poses if p["name"] == "trunk_base"),
        dtype=np.float64,
    )
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    qz = (math.cos(yaw_rad * 0.5), 0.0, 0.0, math.sin(yaw_rad * 0.5))
    out = []
    for p in poses:
        pos = np.asarray(p["pos"], dtype=np.float64)
        rel = pos - home_base
        pos3 = np.array([
            base_xy[0] + c * rel[0] - s * rel[1],
            base_xy[1] + s * rel[0] + c * rel[1],
            pos[2],
        ])
        # Apply the world-space yaw to every body's world orientation: qz * q.
        qw, qx, qy, qzz = (float(v) for v in p["quat"])
        zw, zx, zy, zz = qz
        q = [
            zw * qw - zx * qx - zy * qy - zz * qzz,
            zw * qx + zx * qw + zy * qzz - zz * qy,
            zw * qy - zx * qzz + zy * qw + zz * qx,
            zw * qzz + zx * qy - zy * qx + zz * qw,
        ]
        out.append({
            "name": p["name"],
            "pos": [float(pos3[0]), float(pos3[1]), float(pos3[2])],
            "quat": [float(v) for v in q],
            "linvel": [0.0, 0.0, 0.0],
            "angvel": [0.0, 0.0, 0.0],
        })
    return out


class CameraHarness:
    """Fire-and-forget pump client (see module docstring)."""

    def __init__(self) -> None:
        self.proc, self.port, self.client = spawn_godot(
            "res://main.tscn", headless=True, cwd=GODOT_PROJECT
        )
        self._pending = 0

    def _req(self, obj: dict) -> None:
        self.client.send(obj)
        self._pending += 1

    def _next(self, cmd: str = "camera_state") -> dict:
        while True:
            msg = self.client.recv()
            self._pending -= 1
            if msg.get("cmd") == cmd:
                return msg

    def cam(self) -> dict:
        self._req({"cmd": "camera_state"})
        return self._next()["camera"]

    def reset(self, xy=(0.0, 0.0), yaw=0.0, *, camera_snap: bool = True) -> dict:
        self._req({
            "cmd": "reset",
            "ctrl": [0.0] * 14,
            "bodies": _teleport_poses(xy, yaw),
            "camera_snap": camera_snap,
        })
        return self._next("reset")

    def zoom(self, d: float):
        self._req({"cmd": "camera_zoom", "d": d})
        return self._next("camera_zoom")

    def settle(self, timeout: float = 8.0, still_needed: int = 8) -> dict:
        """Pump at ~60 Hz until cam_pos stops moving for `still_needed` polls."""
        t0 = time.time()
        prev: np.ndarray | None = None
        still = 0
        cam: dict = self.cam()
        while time.time() - t0 < timeout:
            cp = np.array(cam["cam_pos"])
            ground = cp[[0, 2]]
            prev_ground = prev[[0, 2]] if prev is not None else None
            if prev_ground is not None and float(np.linalg.norm(ground - prev_ground)) < 0.002 and not cam.get("manual", False):
                still += 1
                if still >= still_needed:
                    return cam
            elif prev_ground is not None and float(np.linalg.norm(ground - prev_ground)) < 0.002:
                still += 1  # manual (drag/zoom): accept stillness anyway after min dwell
                if still >= max(3, still_needed // 2):
                    return cam
            else:
                still = 0
            prev = cp
            time.sleep(1.0 / 60.0)
            cam = self.cam()
        return cam

    def close(self) -> None:
        try:
            self.client.close()
        finally:
            stop_godot(self.proc, None)


def heading_from_state(cam: dict) -> float:
    """MuJoCo-world heading (rad) derived from the protocol's converted fwd."""
    fx, fy = cam["duck_fwd"]
    return math.atan2(fy, fx)


def expected_trailing_orbit_yaw(cam: dict, from_yaw: float) -> float:
    """Deadzone-edge goal chosen from the shortest turn from ``from_yaw``."""
    behind = heading_from_state(cam) - math.pi / 2
    diff = (behind - from_yaw + math.pi) % (2 * math.pi) - math.pi
    return behind - math.copysign(math.radians(35), diff)


class TestCameraFollow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.h = CameraHarness()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.h.close()

    # ---- helpers -------------------------------------------------------

    def assertBehindDuck(self, cam: dict, tol_deg: float = 8.0):
        """Orbit geometry check in Godot ground coords: the rendered camera
        (cam_pos, which lerps toward the orbit target) and the orbit target
        itself (cam_look + yaw/pitch/dist) must both sit behind the duck and
        at the orbit radius — the rig lag only trails along the follow path,
        it never opens a gap off-axis."""
        look = np.array(cam["cam_look"])
        base = np.array(cam["look"])  # duck look-at point (base + 0.08 up)
        fx, fy = cam["duck_fwd"]
        f = np.array([fx, -fy]); f /= max(np.linalg.norm(f), 1e-9)

        yaw, pitch, dist = cam["yaw"], cam["pitch"], cam["dist"]
        target = look + np.array([math.sin(yaw) * math.cos(pitch),
                                  math.sin(pitch),
                                  math.cos(yaw) * math.cos(pitch)]) * dist
        for tag, cp in (("target", target), ("rendered", np.array(cam["cam_pos"]))):
            d = cp[[0, 2]] - look[[0, 2]]
            n = float(np.linalg.norm(d))
            self.assertGreater(n, 0.3, f"{tag} camera collapsed onto the duck")
            ang = math.degrees(math.acos(float(np.clip(np.dot(d / n, -f), -1, 1))))
            self.assertLess(ang, tol_deg, f"{tag} camera not behind duck: off by {ang:.1f}°")
        # rendered camera height tracks orbit height within rig-lag slack
        cp = np.array(cam["cam_pos"])
        self.assertGreater(cp[1] - base[1], 0.15, "rendered camera below duck eye level")
        # orbit target sits exactly at the orbit radius (ground projection)
        self.assertAlmostEqual(float(np.linalg.norm((target - look)[[0, 2]])),
                               dist * math.cos(pitch), delta=0.02)

    def assertAtBehindAzimuth(self, cam: dict, tol_deg: float = 6.0):
        """Orbit yaw equals the camera-behind azimuth: with duck_fwd given
        in MuJoCo ground (x, y), the behind azimuth in the orbit convention
        (yaw from +Z toward +X, offset = (sin, cos)*dist) is
        atan2(-f.x, -f.z) = wrap(duck_yaw - 90°)."""
        fx, fy = cam["duck_fwd"]
        want = math.atan2(-fx, -fy)
        err = abs(math.degrees((cam["yaw"] - want + math.pi) % (2 * math.pi) - math.pi))
        self.assertLess(err, tol_deg,
                        f"orbit yaw {math.degrees(cam['yaw']):.1f}° != behind azimuth {math.degrees(want):.1f}°")

    # ---- tests ----------------------------------------------------------

    def test_01_spawn_home_shot(self) -> None:
        self.h.reset((0.0, 0.0), 0.0)
        cam = self.h.settle()
        self.assertAlmostEqual(cam["dist"], 1.0, delta=0.05)
        cp = np.array(cam["cam_pos"])
        self.assertGreater(float(cp[1]), 0.2, "camera above the duck")
        self.assertBehindDuck(cam)
        print(f"PASS spawn: cam={cp.round(2).tolist()} behind duck")

    def test_02_teleport_fast_snap(self) -> None:
        self.h.reset((2.0, 1.5), 0.0)
        t0 = time.time()
        cam = self.h.settle(timeout=6.0)
        dt = time.time() - t0
        bx, by, _ = cam["base"]
        cp = np.array(cam["cam_pos"])
        self.assertLess(dt, 2.5, f"reset snap too slow: {dt:.2f}s")
        self.assertBehindDuck(cam)
        # camera rests one orbit radius behind the duck, near the teleported
        # duck on the ground plane
        gx, gz = cam["base"][0], cam["base"][2]
        cp = np.array(cam["cam_pos"])
        self.assertLess(float(np.hypot(cp[0] - gx, cp[2] - gz)), cam["dist"] + 0.3)
        print(f"PASS teleport snap in {dt:.2f}s: cam={cp.round(2).tolist()}")

    def test_03_yaw_deadzone_small_turn(self) -> None:
        self.h.reset((0.0, 0.0), 0.0)
        cam = self.h.settle()
        yaw0 = cam["yaw"]
        self.h.reset((0.0, 0.0), math.radians(15.0), camera_snap=False)
        cam = self.h.settle()
        # 15° turn is inside the 35° deadzone: camera azimuth must not chase.
        err = abs(math.degrees((cam["yaw"] - yaw0 + math.pi) % (2 * math.pi) - math.pi))
        self.assertLess(err, 8.0, f"small turn dragged the camera by {err:.1f}°")
        print(f"PASS deadzone: duck 15° turn -> camera moved {err:.1f}°")

    def test_04_yaw_follows_big_turn(self) -> None:
        self.h.reset((0.0, 0.0), 0.0)
        start = self.h.settle()
        self.h.reset((0.0, 0.0), math.radians(200.0), camera_snap=False)
        cam = self.h.settle()
        goal = expected_trailing_orbit_yaw(cam, start["yaw"])
        err = abs(math.degrees((cam["yaw"] - goal + math.pi) % (2 * math.pi) - math.pi))
        self.assertLess(err, 10.0, f"camera yaw {math.degrees(cam['yaw']):.0f}° != trailing goal {math.degrees(goal):.0f}°")
        print(f"PASS big turn: cam yaw {math.degrees(cam['yaw']):.1f}° trails heading by {math.degrees(heading_from_state(cam)) + 90 - math.degrees(cam['yaw']):.0f}°")

    def test_05_held_order_protocol(self) -> None:
        msg = self.h.cam()
        self.h.client.send({"cmd": "camera_state"})
        raw = self.h.client.recv()
        while raw.get("cmd") != "camera_state":
            raw = self.h.client.recv()
        self.assertIn("held_order", raw)
        print("PASS held_order echoed")

    def test_06_reset_snaps_across_map(self) -> None:
        self.h.reset((0.0, 0.0), 0.0)
        self.h.settle()
        self.h.reset((-3.0, -3.0), math.radians(45.0))
        t0 = time.time()
        cam = self.h.settle(timeout=6.0)
        dt = time.time() - t0
        self.assertLess(dt, 2.5)
        self.assertBehindDuck(cam)
        print(f"PASS reset snap {dt:.2f}s")

    def test_07_zoom_clamps(self) -> None:
        self.h.reset((0.0, 0.0), 0.0)
        self.h.settle()
        for _ in range(8):
            self.h.zoom(0.2)
            self.h.settle(still_needed=3)
        cam = self.h.settle()
        self.assertGreaterEqual(cam["dist"], 0.5 - 1e-6)
        self.assertLessEqual(cam["dist"], 3.0 + 1e-6)
        self.assertBehindDuck(cam)
        print(f"PASS zoom clamp: dist={cam['dist']:.2f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
