"""Interactive scene pickup: release locomotion/IK plus explicit game assistance.

Only motor targets are returned. Godot owns the proximity-gated grip constraint,
all object motion, release, and the final physical containment verdict.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sim2sim.sai_controller import MotionController
from sai_agent.manipulation import Task


class WorkshopController(MotionController):
    def __init__(self, root, stair_profile=None):
        super().__init__(root, stair_profile=stair_profile)
        self.grab_serial = -1
        self.phase = "idle"
        self.task = None
        self.arm_hold = np.zeros(6)
        self.cancel_start = None

    def command(self, state):
        request = state.get("workshop_grab", {})
        now = float(state["time"])
        if request.get("serial", -1) != self.grab_serial:
            self.grab_serial = request.get("serial", -1)
            if request.get("request") == "pick":
                self.phase = "docking"
                self.started = now
                self.task = None
                self.cancel_start = None
            elif request.get("request") == "cancel":
                self.phase = "idle"
                self.cancel_start = now
                self.arm_hold = np.asarray(state["q"][16:22]).copy()

        if self.phase == "docking":
            rotation = np.asarray(state["base_rotation_columns"]).T
            delta = rotation.T @ (np.asarray(request["target_m"]) - state["base_position"])
            distance = float(np.linalg.norm(delta[:2]))
            angle = float(np.arctan2(delta[1], delta[0]))
            if now - self.started > 24.:
                self.phase = "idle"
                result = self._drive(state, [0., 0., 0.])
                result.update(grab_stage="failed", grab_message="靠近超时 · 请驾驶到物件前方再重试")
                return result
            # Nominal release pickup is 240 mm ahead of the chassis origin.
            if .20 <= distance <= .27 and abs(angle) < .10:
                if np.linalg.norm(np.asarray(state["base_linear_world"])[:2]) > .04:
                    result = self._drive(state, [0., 0., 0.])
                    result.update(grab_stage="docking", grab_message="正在停稳底盘")
                    return result
                self.task = Task()
                self.phase = "manipulation"
                self.started = now
                self.anchor_rotation = rotation
                self.anchor_translation = np.asarray(state["base_position"]) - rotation @ np.asarray(self.spec["bodies"]["chassis"]["origin_m"])
                self.pick_delta = rotation.T @ (np.asarray(request["target_m"]) - self.anchor_translation) - np.array([.24, 0., .020373])
                self.pick_height = float(request["target_m"][2])
                # Wheel joint positions are unbounded after driving. Retain
                # their present turns when switching from speed to pose control.
                self.wheel_origin = np.asarray(state["q"][:16])[3::4].copy()
                self.waited = 0.
                self.last_time = now
                self.arm_hold = np.asarray(state["q"][16:22]).copy()
            else:
                speed = float(np.clip((distance - .24) * .7, -.06, .12)) if abs(angle) < .3 else 0.
                result = self._drive(state, [speed, float(np.clip(angle * 1.4, -.45, .45)), 0.])
                result.update(grab_stage="docking", grab_message="正在靠近所选物件")
                return result

        if self.phase == "manipulation":
            return self._manipulate(state, request)

        result = super().command(state)
        # Smoothly return the arm after a completed/cancelled interaction.
        if self.cancel_start is not None:
            alpha = float(np.clip((now - self.cancel_start) / 3., 0., 1.))
            result["target_arm"] = ((1. - alpha) * self.arm_hold).tolist()
            if alpha >= 1.: self.cancel_start = None
        return result

    def _drive(self, state, command):
        return super().command(dict(state, command=command))

    def _manipulate(self, state, request):
        task = self.task
        now = float(state["time"])
        elapsed = now - self.started - self.waited
        # Hold the close phase until the actual game-side grip exists.
        if 10.8 <= elapsed < 12. and not request["held"]:
            self.waited += now - self.last_time
            elapsed = min(elapsed, 10.8)
            if self.waited > 4.:
                self.phase = "idle"
                self.cancel_start = now
                self.arm_hold = np.asarray(state["q"][16:22]).copy()
                result = self._drive(state, [0., 0., 0.])
                result.update(grab_stage="failed", grab_message="夹爪未够到物件 · 请换个位置重试")
                return result
        self.last_time = now
        rotation = np.asarray(state["base_rotation_columns"]).T
        data = task.data
        data.qpos[:3] = state["base_position"]
        data.qpos[3:7] = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
        data.qpos[task.lq] = state["q"][:16]
        data.qpos[task.aq] = state["q"][16:22]
        data.qvel[:3] = state["base_linear_world"]
        data.qvel[3:6] = rotation.T @ state["base_angular_world"]
        data.qvel[task.lv] = state["v"][:16]
        data.qvel[task.av] = state["v"][16:22]
        mujoco.mj_forward(task.model, data)
        q, drop, label = task.path(elapsed)
        # Open wider for varied scene shapes; grip stability is game assistance.
        q[5] = 74. if request["held"] and elapsed < 34. else 60.
        point = task.arm.tool(q, drop)[0] * .001
        blend = float(np.clip((elapsed - 15.8) / (29. - 15.8), 0., 1.))
        blend = blend * blend * (3. - 2. * blend)
        release_height = .259 + request["rest_height_m"] + .001
        destination_delta = np.array([-.030, -.04 if request["slot"] == 0 else .04, release_height - .27923])
        chassis_origin = np.asarray(self.spec["bodies"]["chassis"]["origin_m"])
        current_translation = np.asarray(state["base_position"]) - rotation @ chassis_origin
        pickup_point = self.anchor_rotation @ (point + self.pick_delta) + self.anchor_translation
        if request["held"]:
            # Contact can capture a tall item before the tool reaches its centre.
            # Preserve that offset during the remaining approach, so the arm
            # cannot drive the constrained object through a thin terrain surface.
            pickup_point[2] = max(pickup_point[2], self.pick_height - request["held_offset_m"][2])
        cargo_point = rotation @ (point + destination_delta) + current_translation
        # The floor object is world-fixed; the receiving bin moves with the
        # chassis, including the small displacement caused by arm reactions.
        point = (1. - blend) * pickup_point + blend * cargo_point
        if request["held"]:
            point -= blend * np.asarray(request["held_offset_m"])
        nominal = np.deg2rad(q - task.home)
        task.ik.qpos[:] = data.qpos
        # Seed from the audited path on every tick. Accumulating corrections
        # across the wrist's half-turn can select the opposite IK branch.
        task.ik.qpos[task.aq] = nominal
        task.ik.qpos[task.aq[-1]] = nominal[-1]
        jp = np.zeros((3, task.model.nv))
        # Five axes cannot preserve an arbitrary 6D pose after retargeting.
        # For the assisted point grip, prioritize reaching the item and retain
        # the release posture only as a gentle null-space preference.
        for _ in range(40):
            mujoco.mj_fwdPosition(task.model, task.ik)
            mujoco.mj_jacSite(task.model, task.ik, jp, None, task.site)
            errp = point - task.ik.site_xpos[task.site]
            jac = jp[:, task.av[:5]]
            preference = nominal[:5] - task.ik.qpos[task.aq[:5]]
            dq = np.linalg.solve(jac.T @ jac + np.eye(5) * 1e-6, jac.T @ errp + 1e-6 * preference)
            task.ik.qpos[task.aq[:5]] += np.clip(dq, -.10, .10)
            task.ik.qpos[task.aq] = np.clip(task.ik.qpos[task.aq], task.model.jnt_range[task.arm_joints, 0] + .001,
                                          task.model.jnt_range[task.arm_joints, 1] - .001)
        target_arm = task.ik.qpos[task.aq].copy()
        mujoco.mj_fwdPosition(task.model, task.ik)
        release_point = rotation @ np.array([-.095, -.04 if request["slot"] == 0 else .04, release_height]) + current_translation
        ready_to_release = bool(np.linalg.norm(np.asarray(request["target_m"]) - release_point) < .025)
        target_leg = task.leg_reference(drop)
        target_leg[3::4] += self.wheel_origin
        result = dict(mode="manipulation", stage=label, target_leg=target_leg.tolist(),
                      target_arm=target_arm.tolist(), arm_bias=data.qfrc_bias[task.av].tolist(),
                      grip_cap=1.4, cargo_target_rad=0., physics_advanced_by_controller=False,
                      grab_stage="manipulation", assist_grip=6. <= elapsed < 12., assist_release=elapsed >= 34.5 and ready_to_release,
                      grab_elapsed=elapsed, tool_target_m=point.tolist(),
                      IK_target_error_m=float(np.linalg.norm(task.ik.site_xpos[task.site] - point)),
                      FK_tool_error_m=float(np.linalg.norm(data.site_xpos[task.site] - state["tool_m"])))
        if elapsed >= task.end:
            result.update(grab_stage="complete")
            self.phase = "idle"
            self.cancel_start = now
            self.arm_hold = target_arm
        return result
