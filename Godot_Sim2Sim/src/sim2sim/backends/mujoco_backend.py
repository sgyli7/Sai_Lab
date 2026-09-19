"""MuJoCo physics backend aligned with microduck_rl/scripts/infer_policy.py."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from sim2sim.backends import SimState

# bam xl330 m6 kt measured from microduck_rl venv; avoids a bam dependency here.
XL330_M6_KT = 0.36601349688984386


class MujocoBackend:
    name = "mujoco"

    def __init__(
        self,
        mjcf: Path,
        *,
        timestep: float = 0.005,
        current_limit_a: float = 0.0,
        base_body: str = "trunk_base",
        imu_sensor: str = "imu_ang_vel",
    ) -> None:
        self.mjcf = Path(mjcf)
        spec = mujoco.MjSpec.from_file(str(self.mjcf))
        pin_eq = spec.add_equality()
        pin_eq.type = mujoco.mjtEq.mjEQ_WELD
        pin_eq.name = "sim2sim_pin_base"
        pin_eq.name1 = base_body
        pin_eq.name2 = ""
        pin_eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        pin_eq.active = False
        # Tight weld ≈ Godot freeze. Default contact solref (0.02) is too soft
        # to compare against a kinematic pin.
        pin_eq.solref = np.array([0.002, 1.0], dtype=np.float64)
        pin_eq.solimp = np.array([0.99, 0.999, 0.0005, 0.5, 2.0], dtype=np.float64)
        self.model = spec.compile()
        self.model.opt.timestep = float(timestep)
        self.dt = float(self.model.opt.timestep)
        if current_limit_a and current_limit_a > 0:
            lim = XL330_M6_KT * float(current_limit_a)
            self.model.actuator_forcerange[:, 0] = -lim
            self.model.actuator_forcerange[:, 1] = lim
            self.model.actuator_forcelimited[:] = 1
        self.data = mujoco.MjData(self.model)
        self._pin_eq = None
        for i in range(self.model.neq):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
            if name == "sim2sim_pin_base":
                self._pin_eq = i
                break
        self.nu = int(self.model.nu)
        self.joint_qpos_indices = [
            int(self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]) for i in range(self.nu)
        ]
        self.joint_qvel_indices = [
            int(self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]) for i in range(self.nu)
        ]
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        if self.base_body_id < 0:
            raise ValueError(f"body {base_body!r} not in model")
        self.imu_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, imu_sensor)
        free_id = -1
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) and int(
                self.model.jnt_bodyid[j]
            ) == self.base_body_id:
                free_id = j
                break
        self.free_qposadr = int(self.model.jnt_qposadr[free_id]) if free_id >= 0 else 0
        self.free_dofadr = int(self.model.jnt_dofadr[free_id]) if free_id >= 0 else 0
        self._pin_base = False
        self._key_ids = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, k): k
            for k in range(self.model.nkey)
        }
        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
            for i in range(self.nu)
        ]
        self.body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"
            for i in range(self.model.nbody)
        ]

    def body_poses_mujoco(self) -> list[dict]:
        poses = []
        for b in range(1, self.model.nbody):
            vel = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
            # mjOBJ_BODY is the inertial frame, already centered at xipos.
            # mjOBJ_XBODY would instead report the regular body origin.
            omega = vel[0:3]
            v_com = vel[3:6]
            poses.append(
                {
                    "name": self.body_names[b],
                    "pos": self.data.xipos[b].tolist(),
                    "quat": _mat_to_quat_wxyz(self.data.ximat[b].reshape(3, 3)).tolist(),
                    "linvel": v_com.tolist(),
                    "angvel": omega.tolist(),
                }
            )
        return poses

    def reset(
        self,
        *,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
        keyframe: str | None = None,
        ctrl: np.ndarray | None = None,
        pin_base: bool = False,
    ) -> SimState:
        mujoco.mj_resetData(self.model, self.data)
        if keyframe is not None:
            if keyframe not in self._key_ids:
                raise KeyError(f"keyframe {keyframe!r} not in {list(self._key_ids)}")
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._key_ids[keyframe])
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)
        self._pin_base = bool(pin_base)
        self._set_pin_weld(self._pin_base)
        return self._state()

    def _set_pin_weld(self, on: bool) -> None:
        """Weld trunk to world at the current pose. Overwriting freejoint qpos
        after mj_step is free-fall (joints feel no gravity) and is not a pin."""
        if self._pin_eq is None:
            if on:
                raise RuntimeError("sim2sim_pin_base weld missing from compiled model")
            return
        self.data.eq_active[self._pin_eq] = bool(on)
        if not on:
            return
        b = self.base_body_id
        ed = self.model.eq_data[self._pin_eq]
        # eq_data: anchor in body2 (world), anchor in body1, relquat, torquescale.
        # Translation satisfied when xpos_body = anchor_world; orientation when
        # relquat = neg(xquat_body) with world quat = identity.
        ed[0:3] = self.data.xpos[b]
        ed[3:6] = 0.0
        q = self.data.xquat[b]
        ed[6] = q[0]
        ed[7] = -q[1]
        ed[8] = -q[2]
        ed[9] = -q[3]
        ed[10] = 1.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: np.ndarray, n_substeps: int = 1) -> SimState:
        self.data.ctrl[:] = np.asarray(ctrl, dtype=np.float64).reshape(self.nu)
        for _ in range(int(n_substeps)):
            mujoco.mj_step(self.model, self.data)
        # mj_step integrates qpos/qvel last; xpos/cvel/sensordata still describe
        # the pre-step pose until mj_forward. Godot reports the post-step pose,
        # and IMU/projected-gravity obs must match that same instant.
        mujoco.mj_forward(self.model, self.data)
        return self._state()

    def _state(self) -> SimState:
        q = self.data.qpos[self.joint_qpos_indices].copy()
        qd = self.data.qvel[self.joint_qvel_indices].copy()
        pos = self.data.xpos[self.base_body_id].copy()
        quat = self.data.xquat[self.base_body_id].copy()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.base_body_id, velocity, 0)
        lin = velocity[3:6].copy()
        if self.imu_id >= 0:
            adr = int(self.model.sensor_adr[self.imu_id])
            ang_local = self.data.sensordata[adr : adr + 3].copy()
        else:
            w = self.data.cvel[self.base_body_id][0:3]
            r = self.data.xmat[self.base_body_id].reshape(3, 3)
            ang_local = r.T @ w
        return SimState(
            t=float(self.data.time),
            q=q.astype(np.float64),
            qd=qd.astype(np.float64),
            base_pos=pos.astype(np.float64),
            base_quat_wxyz=quat.astype(np.float64),
            base_linvel=lin.astype(np.float64),
            base_angvel_local=np.asarray(ang_local, dtype=np.float64),
            extra={"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()},
        )

    def close(self) -> None:
        return None


def _mat_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    from sim2sim.coords import mat_to_quat_wxyz

    return mat_to_quat_wxyz(r)
