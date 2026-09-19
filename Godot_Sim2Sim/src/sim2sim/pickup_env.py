"""MuJoCo-first physical pickup task for the articulated MicroDuck beak.

The actor observes 76 float32 values and controls 14 original joints plus one
beak hinge. The point assist is activated only at measured two-sided contact;
its anchor is the actual contact point, so activation cannot pull remote sites
together. This is an assisted-grip benchmark, not pure friction closure.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import math

import mujoco
import numpy as np

from sim2sim.backends.mujoco_backend import MujocoBackend, XL330_M6_KT
from sim2sim.obs import build_obs, DEFAULT_HOME


DT = 0.02
MAX_STEPS = 200  # Source GroundPick cycle is four seconds.
OBS_DIM = 76
ACTION_DIM = 15
GRIP_FORCE_LIMIT_N = 1.5


class PickupEnv:
    def __init__(self, scene: str | Path, *, max_steps: int = MAX_STEPS):
        self.sim = MujocoBackend(Path(scene))
        self.max_steps = int(max_steps)
        self.model, self.data = self.sim.model, self.sim.data
        m = self.model
        if m.nu not in (ACTION_DIM, ACTION_DIM+1):
            raise ValueError(f"Expected 15 or 16 actuators, found {m.nu}")
        self.adhesion_mode = m.nu == ACTION_DIM+1
        provenance = Path(self.sim.mjcf).parent / "provenance.json"
        gripper = json.loads(provenance.read_text()).get("gripper", "clamp") if provenance.is_file() else "clamp"
        self.contact_assist_mode = gripper.startswith("contact_assist")
        self.suction_request = False
        self.suction_contact = False
        self.suction_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "suction_contact")
        self.suction_geoms = {self.suction_geom} if self.suction_geom >= 0 else set()
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "suction_contact_lower") >= 0:
            self.suction_geoms.add(m.geom("suction_contact_lower").id)
        limit = 1.75 * XL330_M6_KT
        m.actuator_forcerange[:14] = [-limit, limit]
        m.actuator_forcelimited[:14] = 1
        self.home = DEFAULT_HOME.astype(np.float64)
        self.robot = m.body("trunk_base").id
        self.head = m.body("jaw_soft").id
        self.lower = m.body("lower_beak").id
        self.item = m.body("pickup_item").id
        self.tip = m.site("mouth_tip").id
        self.lower_tip = m.site("lower_mouth_tip").id
        self.item_grip_site = m.site("pickup_grip").id
        self.upper_geom = m.geom("upper_grip_pad").id
        self.lower_geom = m.geom("lower_grip_pad").id
        self.beak_joint = m.joint("beak_pitch").id
        self.item_joint = m.joint("pickup_item_freejoint").id
        self.assist = m.equality("assist_grip").id
        self.shape = m.geom("pickup_item_geom").type
        self.dimensions = self._dimensions()
        self.item_mass = float(m.body_mass[self.item])
        self.rng = np.random.default_rng(0)
        self.state = None
        self.last = np.zeros(14, np.float32)
        self.contacts = (False, False)
        self.contact_force = (0., 0.)
        self.events: list[dict] = []
        self.t = 0.
        self.step_count = 0
        self.initial_item_height = 0.
        self.initial_robot_x = 0.
        self.peak_lift = 0.
        self.min_gap = math.inf
        self.max_contact_force = 0.
        self.min_suction_contact_dist = math.inf
        self.min_object_robot_dist = math.inf
        self.max_commanded_adhesion_force = 0.
        self.max_pad_contact_force = 0.
        self.pad_contact_steps = 0
        self.contact_ever = False
        self.grip_ever = False
        self.grip_reward_paid = False
        self.opposed_contact_ever = False
        self.stable_hold_steps = 0
        self.stable_physical_steps = 0
        self.grip_overload_steps = 0
        self.max_grip_force = 0.
        self.grip_broken = False

    def _dimensions(self) -> np.ndarray:
        provenance = Path(self.sim.mjcf).parent / "provenance.json"
        if provenance.is_file():
            overall = json.loads(provenance.read_text()).get("overall_dimensions_xyz_m")
            if overall is not None:
                return np.asarray(overall, np.float32)
        geom = self.model.geom("pickup_item_geom")
        if geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return np.repeat(2 * geom.size[0], 3).astype(np.float32)
        if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return np.array([geom.size[0]*2, geom.size[0]*2, geom.size[1]*2], np.float32)
        return (geom.size[:3] * 2).astype(np.float32)

    def reset(self, seed: int, *, x: float | None = None, lateral: float | None = None,
              joint_noise: float = 0., base_x: float = 0., base_y: float = 0.) -> np.ndarray:
        self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        target_x = float(x if x is not None else self.rng.uniform(.10, .14))
        target_y = float(lateral if lateral is not None else self.rng.uniform(-.015, .015))
        qpos = np.zeros(m.nq)
        qpos[:7] = [base_x, base_y, .12, 1., 0., 0., 0.]
        for i, value in enumerate(self.home):
            jid = int(m.actuator_trnid[i, 0])
            qpos[int(m.jnt_qposadr[jid])] = value + self.rng.normal(0, joint_noise)
        adr = int(m.jnt_qposadr[self.item_joint])
        qpos[adr:adr+7] = [target_x, target_y, self.dimensions[2]*.5, 1., 0., 0., 0.]
        self.state = self.sim.reset(qpos=qpos, ctrl=np.r_[self.home, 0., 0.] if self.adhesion_mode else np.r_[self.home, 0.])
        d.eq_active[self.assist] = 0
        self.suction_request = False
        self.suction_contact = False
        for contact in d.contact:
            bodies = (int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2]))
            if self.item in bodies and 0 not in bodies and contact.dist < -.001:
                raise ValueError(f"Initial object-robot penetration {contact.dist:.4f} m at x={target_x:.3f}, y={target_y:.3f}")
        self.last.fill(0)
        self.contacts = (False, False)
        self.contact_force = (0., 0.)
        self.events = [{"event": "reset", "target": [target_x, target_y],
                        "base_xy": [base_x,base_y], "seed": seed}]
        self.t = 0.
        self.step_count = 0
        self.initial_item_height = float(d.xpos[self.item, 2])
        self.initial_robot_x = float(d.xpos[self.robot, 0])
        self.peak_lift = 0.
        self.min_gap = math.inf
        self.max_contact_force = 0.
        self.min_suction_contact_dist = math.inf
        self.min_object_robot_dist = math.inf
        self.max_commanded_adhesion_force = 0.
        self.max_pad_contact_force = 0.
        self.pad_contact_steps = 0
        self.contact_ever = False
        self.grip_ever = False
        self.grip_reward_paid = False
        self.opposed_contact_ever = False
        self.stable_hold_steps = 0
        self.stable_physical_steps = 0
        self.grip_overload_steps = 0
        self.max_grip_force = 0.
        self.grip_broken = False
        return self.observation()

    def _body_velocity(self, body: int) -> np.ndarray:
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 body, vel, 0)
        return vel[3:6]

    def observation(self) -> np.ndarray:
        state = replace(self.state, q=self.state.q[:14], qd=self.state.qd[:14])
        phase = self.t / 4.
        cmd = np.zeros(13, np.float32)
        cmd[:2] = [math.cos(math.tau*phase), math.sin(math.tau*phase)]
        base = build_obs(state, self.last, cmd, self.home)
        m, d = self.model, self.data
        rotation = d.xmat[self.head].reshape(3, 3)
        target = d.site_xpos[self.item_grip_site]
        rel_pos = rotation.T @ (target - d.site_xpos[self.tip])
        rel_vel = rotation.T @ (self._body_velocity(self.item) - self._body_velocity(self.head))
        qadr = int(m.jnt_qposadr[self.beak_joint])
        vadr = int(m.jnt_dofadr[self.beak_joint])
        extra = np.r_[rel_pos, self.dimensions, rel_vel,
                      d.qpos[qadr], d.qvel[vadr], float(self.contacts[0]),
                      float(self.contacts[1]), self.item_mass, float(d.eq_active[self.assist])]
        result = np.r_[base, extra].astype(np.float32)
        assert result.shape == (OBS_DIM,)
        return result

    def _contacts(self) -> tuple[list[dict], list[dict]]:
        upper, lower = [], []
        for i, contact in enumerate(self.data.contact):
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if self.model.geom_bodyid[g1] != self.item and self.model.geom_bodyid[g2] != self.item:
                continue
            where = upper if self.upper_geom in (g1, g2) else lower if self.lower_geom in (g1, g2) else None
            if where is None:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            where.append({"point": contact.pos.copy(), "force": float(force[0]),
                          "normal": contact.frame[:3].copy() * (1 if self.model.geom_bodyid[g1] == self.item else -1),
                          "geom": g2 if self.model.geom_bodyid[g1] == self.item else g1})
        return upper, lower

    def _try_assist(self, upper: list, lower: list, jaw_command: float) -> None:
        d, m = self.data, self.model
        if self.adhesion_mode:
            return
        if d.eq_active[self.assist] or self.grip_broken:
            return
        if self.contact_assist_mode:
            # Sai-style assisted grip requires *measured* pad/object contact.
            # The equality anchor is the contact point, not a distant target
            # site; it is monitored and released above the force limit.
            if jaw_command > .18:
                return
            qadr = int(m.jnt_qposadr[self.beak_joint])
            vadr = int(m.jnt_dofadr[self.beak_joint])
            if d.qpos[qadr] > .30 or d.qvel[vadr] > .3:
                return
            if np.linalg.norm(self._body_velocity(self.item)-self._body_velocity(self.lower)) > .6:
                return
            candidates = []
            for i, contact in enumerate(d.contact):
                geoms = (int(contact.geom1), int(contact.geom2))
                if not self.suction_geoms.intersection(geoms):
                    continue
                if self.item not in (int(m.geom_bodyid[geoms[0]]),int(m.geom_bodyid[geoms[1]])):
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(m,d,i,force)
                if force[0] >= .05 and np.linalg.norm(contact.pos-d.site_xpos[self.tip]) <= .055:
                    candidates.append((float(force[0]),contact.pos.copy()))
            if candidates:
                force, point = max(candidates,key=lambda pair:pair[0])
                self._activate_assist(point, "contact_assist", {"pad_normal_force_n":force})
            return
        if not upper or not lower:
            return
        # The two surfaces must squeeze OPPOSITE sides of the object. Contacts
        # on its same top patch were previously mistaken for a grasp.
        valid = [(a, b) for a in upper for b in lower
                 if a["force"] >= .05 and b["force"] >= .05
                 and np.dot(a["normal"], b["normal"]) < -.5
                 and np.linalg.norm(a["point"]-b["point"]) >= .003]
        if not valid or jaw_command > .18:
            return
        self.opposed_contact_ever = True
        qadr = int(m.jnt_qposadr[self.beak_joint])
        vadr = int(m.jnt_dofadr[self.beak_joint])
        if d.qpos[qadr] > .30 or d.qvel[vadr] > .3:
            return
        candidate = max(valid, key=lambda pair: min(pair[0]["force"],pair[1]["force"]))
        point = .5*(candidate[0]["point"]+candidate[1]["point"])
        rel_speed = np.linalg.norm(self._body_velocity(self.item)-self._body_velocity(self.lower))
        if rel_speed > .6 or np.linalg.norm(point-d.site_xpos[self.tip]) > .055:
            return
        self._activate_assist(point, "two_sided_assist", {
            "contact_separation_m": float(np.linalg.norm(candidate[0]["point"]-candidate[1]["point"])),
            "opposing_normal_dot": float(np.dot(candidate[0]["normal"],candidate[1]["normal"])),
            "jaw_angle_rad": float(d.qpos[qadr]),
            "relative_speed_mps": float(rel_speed),
            "upper_force_n": candidate[0]["force"], "lower_force_n": candidate[1]["force"]})

    def _activate_assist(self, point: np.ndarray, kind: str, details: dict) -> None:
        d, m = self.data, self.model
        for slot, body in ((slice(0, 3), self.lower), (slice(3, 6), self.item)):
            m.eq_data[self.assist, slot] = d.xmat[body].reshape(3, 3).T @ (point-d.xpos[body])
        d.eq_active[self.assist] = 1
        self.grip_ever = True
        self.events.append({"event": kind, "t": self.t,
                            "contact_point": point.tolist(), **details})

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, np.float32)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("Pickup action must be finite float32[15]")
        body = np.clip(action[:14], -3., 3.)
        jaw = float(np.clip(action[14], 0., .48))
        ctrl = np.r_[self.home + body, jaw, float(self.suction_request)] if self.adhesion_mode else np.r_[self.home + body, jaw]
        self.state = self.sim.step(ctrl, 4)
        self.last = body.copy()
        self.step_count += 1
        self.t = self.step_count * DT
        if self.data.eq_active[self.assist]:
            rows = [j for j in range(self.data.nefc)
                    if self.data.efc_type[j] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
                    and self.data.efc_id[j] == self.assist]
            force = float(np.linalg.norm(self.data.efc_force[rows])) if rows else 0.
            self.max_grip_force = max(self.max_grip_force, force)
            self.grip_overload_steps = self.grip_overload_steps+1 if force > GRIP_FORCE_LIMIT_N else 0
            if self.grip_overload_steps >= 3:
                self.data.eq_active[self.assist] = 0
                self.grip_broken = True
                self.events.append({"event": "grip_overload_release", "t": self.t,
                                    "force_n": force, "limit_n": GRIP_FORCE_LIMIT_N})
        else:
            self.grip_overload_steps = 0
        upper, lower = self._contacts()
        self.contacts = (bool(upper), bool(lower))
        self.contact_force = (max((x["force"] for x in upper), default=0.),
                              max((x["force"] for x in lower), default=0.))
        self.max_contact_force = max(self.max_contact_force, *self.contact_force)
        if upper or lower:
            self.contact_ever = True
        self._try_assist(upper, lower, jaw)
        d = self.data
        self.suction_contact = False
        if self.adhesion_mode:
            # This actuator value is the commanded bound. It can be nonzero
            # with zero contact, so never report it as transmitted load.
            self.max_commanded_adhesion_force = max(
                self.max_commanded_adhesion_force, abs(float(d.actuator_force[15])))
        pad_this_step = False
        for contact_index, contact in enumerate(d.contact):
            geoms = (int(contact.geom1), int(contact.geom2))
            bodies = (int(self.model.geom_bodyid[geoms[0]]),
                      int(self.model.geom_bodyid[geoms[1]]))
            if self.item in bodies and 0 not in bodies:
                self.min_object_robot_dist = min(self.min_object_robot_dist,
                                                 float(contact.dist))
                if self.suction_geoms.intersection(geoms):
                    self.min_suction_contact_dist = min(self.min_suction_contact_dist,
                                                        float(contact.dist))
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, d, contact_index, force)
                    self.max_pad_contact_force = max(self.max_pad_contact_force,
                                                      float(np.linalg.norm(force[:3])))
                    pad_this_step = True
        self.pad_contact_steps += int(pad_this_step)
        if self.adhesion_mode:
            for contact in d.contact:
                geoms = (int(contact.geom1),int(contact.geom2))
                if self.suction_geoms.intersection(geoms) and self.item in (
                        int(self.model.geom_bodyid[geoms[0]]),int(self.model.geom_bodyid[geoms[1]])):
                    self.suction_contact = True
                    break
            if self.suction_request and self.suction_contact and not self.grip_ever:
                self.grip_ever = True
                self.events.append({"event":"adhesion_contact","t":self.t,
                                    "force_limit_n":float(self.model.actuator_gainprm[15,0])})
        tip_gap = float(np.linalg.norm(d.site_xpos[self.tip]-d.xpos[self.item]))
        self.min_gap = min(self.min_gap, tip_gap)
        lift = float(d.xpos[self.item, 2]-self.initial_item_height)
        self.peak_lift = max(self.peak_lift, lift)
        held = bool(self.suction_request and self.suction_contact) if self.adhesion_mode else bool(d.eq_active[self.assist])
        trunk_z = float(d.xpos[self.robot, 2])
        tilt = math.degrees(math.acos(float(np.clip(d.xmat[self.robot].reshape(3, 3)[2, 2], -1, 1))))
        upright = trunk_z > .105 and tilt < 25.
        if held and lift >= .08 and upright:
            self.stable_hold_steps += 1
        else:
            self.stable_hold_steps = 0
        if lift >= .08 and upright and tip_gap <= .10:
            self.stable_physical_steps += 1
        else:
            self.stable_physical_steps = 0
        # Dense shaping stays about the real object and measured contact; the
        # terminal grade is separate and does not reuse this reward.
        approach = math.exp(-tip_gap/.065) if self.t < 1.6 else 0.
        reward = DT*(2.*approach + .6*self.contacts[0] + .6*self.contacts[1]
                     + 6.*held*max(0., min(.18, lift))/.15
                     + 2.*held*float(upright) - .1*abs(jaw-float(d.qpos[mujoco_joint_qpos(self.model,self.beak_joint)])))
        if held and not self.grip_reward_paid:
            reward += 5.
            self.grip_reward_paid = True
            self.events.append({"event": "grip_reward_paid", "t": self.t})
        if self.t > 2.5 and held and lift >= .08 and upright:
            reward += DT*10.
        item_xy = np.linalg.norm(d.xpos[self.item, :2])
        fell = trunk_z < .045 or tilt > 70.
        escaped = bool(item_xy > .6 and not held)
        done = self.step_count >= self.max_steps or fell or escaped
        success = bool(held and self.stable_hold_steps >= 50 and self.t >= 3.5)
        physical_success = bool(self.stable_physical_steps >= 50 and self.t >= 3.5)
        if done:
            reward += (20. if success else -10. if fell else -3.)
            self.events.append({"event": "terminal", "t": self.t, "success": success,
                                "physical_success": physical_success,
                                "lift_m": lift, "upright": upright, "fell": fell,
                                "escaped": escaped, "stable_hold_s": self.stable_hold_steps*DT,
                                "min_suction_contact_dist_m": None if math.isinf(self.min_suction_contact_dist) else self.min_suction_contact_dist,
                                "min_object_robot_dist_m": None if math.isinf(self.min_object_robot_dist) else self.min_object_robot_dist,
                                "max_commanded_adhesion_force_n": self.max_commanded_adhesion_force,
                                "max_pad_contact_force_n": self.max_pad_contact_force,
                                "pad_contact_s": self.pad_contact_steps*DT})
        info = {"t": self.t, "held": held, "contacts": self.contacts,
                "suction_contact":self.suction_contact,
                "suction_force_n":float(d.actuator_force[15]) if self.adhesion_mode else 0.,
                "lift_m": lift,
                "upright": upright, "tilt_deg": tilt, "fell": fell, "escaped": escaped,
                "success": success if done else False, "peak_lift_m": self.peak_lift,
                "physical_success":physical_success if done else False,
                "stable_physical_s":self.stable_physical_steps*DT,
                "max_contact_force_n": self.max_contact_force,
                "min_suction_contact_dist_m": None if math.isinf(self.min_suction_contact_dist) else self.min_suction_contact_dist,
                "min_object_robot_dist_m": None if math.isinf(self.min_object_robot_dist) else self.min_object_robot_dist,
                "max_commanded_adhesion_force_n": self.max_commanded_adhesion_force,
                "max_pad_contact_force_n": self.max_pad_contact_force,
                "pad_contact_s": self.pad_contact_steps*DT,
                "max_grip_force_n": self.max_grip_force,
                "stable_hold_s": self.stable_hold_steps*DT}
        return self.observation(), float(reward), done, info

    def close(self) -> None:
        self.sim.close()


def mujoco_joint_qpos(model: mujoco.MjModel, joint_id: int) -> int:
    return int(model.jnt_qposadr[joint_id])
