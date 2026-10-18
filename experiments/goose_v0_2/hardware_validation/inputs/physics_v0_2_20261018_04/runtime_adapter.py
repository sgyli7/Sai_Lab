"""Standalone Goose V0.2 physical-data adapter: 50Hz commands, 200Hz drive.

No rewards, tasks, policy, hidden root support, or training repository imports.
The saved model contains 18 bounded effort actuators. Simulator-native users
must implement the same immutable drive contract; use this reference to check it.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco


class FrozenPhysics:
    def __init__(self, package: str | Path, *, bus_voltage_v=None):
        self.package = Path(package).resolve()
        self.contract = json.loads((self.package / 'physical_contract.json').read_text())
        self.drive = json.loads((self.package / 'actuator_contract.json').read_text())
        self.model = mujoco.MjModel.from_xml_path(str(self.package / 'robot.xml'))
        self.data = mujoco.MjData(self.model)
        self.joints = self.contract['joints']
        self.names = self.contract['joint_order']
        if self.names != [j['name'] for j in self.joints] or self.model.nu != 18:
            raise ValueError('18-axis public interface mismatch')
        self.dt = self.contract['physics_dt_s']
        self.drive_dt = self.contract['torque_dt_s']
        self.policy_dt = self.contract['policy_dt_s']
        self.substeps = round(self.drive_dt / self.dt)
        self.ticks = round(self.policy_dt / self.drive_dt)
        if self.ticks != 4 or abs(self.substeps * self.dt - self.drive_dt) > 1e-12:
            raise ValueError('Invalid frozen 200Hz/50Hz clock')
        if abs(self.model.opt.timestep - self.dt) > 1e-12:
            raise ValueError('Model physics clock mismatch')
        self.qidx = np.array([int(self.model.joint(j['name']).qposadr[0]) for j in self.joints])
        self.vidx = np.array([int(self.model.joint(j['name']).dofadr[0]) for j in self.joints])
        self.drive_vidx = np.array([int(self.model.joint(j.get('actuation_joint', j['name'])).dofadr[0]) for j in self.joints])
        self.neutral = np.array([j['q_neutral_rad'] for j in self.joints])
        self.scale = np.array([j['action_scale_rad'] for j in self.joints])
        self.ranges = np.array([j['range_rad'] for j in self.joints])
        self.kp = np.array([j['kp_nm_rad'] for j in self.joints])
        self.kd = np.array([j['kd_nm_s_rad'] for j in self.joints])
        self.cont = np.array([j['continuous_design_limit_nm'] for j in self.joints])
        self.speed = np.array([j['speed_limit_rad_s'] for j in self.joints])
        self.entries = self.drive['axes']
        self.bus_voltage_v = float(self.drive['power_supply']['nominal_bus_v'] if bus_voltage_v is None else bus_voltage_v)
        if not 20. <= self.bus_voltage_v <= 25.2:
            raise ValueError('Bus voltage outside frozen design window20..25.2V')
        self.voltage_speed_factor = min(self.bus_voltage_v / 24., 1.)
        self.slew_speed = self.speed * np.array([1. if j['reference_voltage_v'] == 5. else self.voltage_speed_factor for j in self.entries])
        if [j['name'] for j in self.entries] != self.names:
            raise ValueError('Drive ordering mismatch')
        self.target = self.neutral.copy()
        self.last_tau = np.zeros(18)
        self.last_action = np.zeros(18)
        self.controller_updates = 0
        self.policy_updates = 0
        self.reset()

    def reset(self, named_q=None, root_pose=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qidx] = self.neutral
        if root_pose is not None:
            root_pose = np.asarray(root_pose, dtype=float)
            if root_pose.shape != (7,) or not np.isfinite(root_pose).all():
                raise ValueError('Root pose is xyz + quaternion wxyz')
            if abs(np.linalg.norm(root_pose[3:]) - 1) > 1e-8:
                raise ValueError('Root quaternion must be unit')
            self.data.qpos[:7] = root_pose
        if named_q:
            for nm, q in named_q.items():
                if nm not in self.names:
                    raise ValueError('Unknown public axis: ' + nm)
                self.data.qpos[int(self.model.joint(nm).qposadr[0])] = float(q)
        for j in self.contract['passive_linkage_joints']:
            qi = int(self.model.joint(j['name']).qposadr[0])
            source = int(self.model.joint(j['mimic_joint']).qposadr[0])
            self.data.qpos[qi] = j['mimic_multiplier'] * self.data.qpos[source] + j['mimic_offset_rad']
        self.target = self.data.qpos[self.qidx].copy()
        self.last_tau[:] = 0
        self.last_action[:] = 0
        self.data.ctrl[:] = 0
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.controller_updates = self.policy_updates = 0
        return self.state()

    def limit_torque(self, requested, velocity):
        requested = np.asarray(requested, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        if requested.shape != (18,) or velocity.shape != (18,) or not np.isfinite(requested).all() or not np.isfinite(velocity).all():
            raise ValueError('Drive inputs must be finite 18-vectors')
        caps = self.cont.copy()
        for i, row in enumerate(self.entries):
            curve = np.asarray(row['selected_torque_speed_curve_rad_s_nm'], dtype=float).copy()
            if row['reference_voltage_v'] != 5.:
                curve[:, 0] *= self.voltage_speed_factor
            caps[i] = min(caps[i], float(np.interp(abs(velocity[i]), curve[:, 0], curve[:, 1], right=0)))
        tau = np.clip(requested, -caps, caps)
        positive = tau * velocity > 0
        power = float(np.sum(tau[positive] * velocity[positive]))
        ceiling = self.drive['positive_mechanical_power_limit_w']
        if power > ceiling:
            tau[positive] *= ceiling / power
        return tau

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (18,) or not np.isfinite(action).all():
            raise ValueError('Action must be 18 finite normalized joint targets')
        action = np.clip(action, -1, 1)
        desired = np.clip(self.neutral + self.scale * action, self.ranges[:, 0], self.ranges[:, 1])
        before = float(self.data.time)
        for _ in range(self.ticks):
            self.target += np.clip(desired - self.target, -self.slew_speed * self.drive_dt, self.slew_speed * self.drive_dt)
            q = self.data.qpos[self.qidx]
            qd = self.data.qvel[self.vidx]
            requested = self.kp * (self.target - q) - self.kd * qd
            self.last_tau = self.limit_torque(requested, self.data.qvel[self.drive_vidx])
            self.data.ctrl[:] = self.last_tau
            # Only ordinary articulated rigid-body physics and its saved constraints.
            # Never project/reset the linkage or apply floating-root forces here.
            mujoco.mj_step(self.model, self.data, nstep=self.substeps)
            self.controller_updates += 1
            if not all(np.isfinite(getattr(self.data, k)).all() for k in ('qpos', 'qvel', 'qacc')):
                raise FloatingPointError('Nonfinite physical state; no automatic recovery')
            if any(w.number for w in self.data.warning):
                raise FloatingPointError('MuJoCo warning; preserve state for inspection')
        self.policy_updates += 1
        self.last_action[:] = action
        if abs(float(self.data.time) - before - self.policy_dt) > 1e-10:
            raise RuntimeError('Physical clock did not advance exactly 20ms')
        return self.state()

    def observations(self, commands=None, phase_rad=0.):
        commands = np.zeros(3) if commands is None else np.asarray(commands, dtype=float)
        if commands.shape != (3,) or not np.isfinite(commands).all():
            raise ValueError('Commands must be three finite values')
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        bid = self.model.body('torso').id
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, bid, velocity, 1)
        R = self.data.xmat[bid].reshape(3, 3)
        return np.r_[velocity[:3] * .25, R.T @ [0., 0., -1.], commands,
                     self.data.qpos[self.qidx] - self.neutral, self.data.qvel[self.vidx] * .1,
                     self.last_action, np.sin(phase_rad), np.cos(phase_rad)].astype(np.float32)

    def state(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return dict(time_s=float(self.data.time), joint_names=list(self.names),
                    q_rad=self.data.qpos[self.qidx].copy(), qd_rad_s=self.data.qvel[self.vidx].copy(),
                    applied_drive_torque_nm=self.last_tau.copy(), root_xyz_quat_wxyz=self.data.qpos[:7].copy(),
                    contacts=int(self.data.ncon), controller_updates=self.controller_updates,
                    policy_updates=self.policy_updates)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('package', type=Path)
    p.add_argument('--steps', type=int, default=5)
    a = p.parse_args()
    env = FrozenPhysics(a.package)
    for _ in range(a.steps):
        env.step(np.zeros(18))
    print(json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in env.state().items()}, indent=2))


if __name__ == '__main__':
    main()
