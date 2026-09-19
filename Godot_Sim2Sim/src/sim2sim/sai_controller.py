"""Sai release control contract with the independently validated flat actor.

The flat ONNX, terrain gate and trained suspension parameters are project-owned. Observation
construction, history and heading use alpha.3. Ascending keeps its step targets;
validated descents use contact-following support.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation
from sai_agent.godot_controller import GodotController
from sim2sim.sai_terrain import step_in_path, descending_in_path
from sim2sim.sai_suspension import Suspension
from sim2sim.sai_stair_v2 import observation_numpy as stair_observation_v2, targets_numpy as stair_targets_v2
from sim2sim.sai_stair_v3 import observation_numpy as stair_observation_v3
from sim2sim.sai_stair_v4 import observation_numpy as stair_observation_v4
from sim2sim.sai_stair_v5 import targets_numpy as stair_targets_v5
from sim2sim.sai_stair_v6 import (observation_numpy as stair_observation_v6,
                                  project_action_safety,
                                  project_target_safety,
                                  targets_numpy as stair_targets_v6)


class MotionController(GodotController):
    def __init__(self, root, stair_profile=None, suspension_profile=None):
        direct_profile = None
        if stair_profile is not None:
            candidate = json.loads(Path(stair_profile).read_text())
            if candidate.get("schema_version") == 2:
                direct_profile = candidate
        super().__init__(root, stair_profile=None if direct_profile is not None else stair_profile)
        directory = Path(__file__).with_name("assets") / "sai"
        manifest = json.loads((directory / "flat-motion-v1.json").read_text())
        policy_path = directory / "flat-motion-v1.onnx"
        if hashlib.sha256(policy_path.read_bytes()).hexdigest() != manifest["onnx_sha256"]:
            raise ValueError("Sai flat motion policy hash mismatch")
        if manifest["observation_size"] != 82 or manifest["action_size"] != 16:
            raise ValueError("Sai flat motion policy contract mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.policy = ort.InferenceSession(str(policy_path), options, providers=["CPUExecutionProvider"])
        profile = suspension_profile if suspension_profile is not None else os.environ.get("SIM2SIM_SAI_SUSPENSION_PROFILE", str(directory / "suspension-v2.json"))
        self.impedance = None
        self.roll_descent = False
        self.suspension = None
        self.suspension_id = None
        if profile != "off":
            suspension = json.loads(Path(profile).read_text())
            if suspension.get("schema_version") not in (1, 2):
                raise ValueError("Unsupported Sai suspension profile")
            if suspension.get("flat_policy_sha256", manifest["onnx_sha256"]) != manifest["onnx_sha256"]:
                raise ValueError("Suspension profile does not match the flat actor")
            if suspension["schema_version"] == 2:
                from sim2sim.sai_compliance import StanceImpedance
                self.impedance = StanceImpedance(self, suspension["parameters"], suspension.get("turn_support_blend", .25))
                self.roll_descent = suspension.get("descent_control") == "contact_following"
                self.suspension = Suspension(suspension["geometry_parameters"])
            else:
                self.suspension = Suspension(suspension["parameters"])
            self.suspension_id = suspension["id"]
        self.flat_policy_id = manifest["id"]
        self.flat_policy_sha256 = manifest["onnx_sha256"]
        self.direct_stair_profile = direct_profile
        self.direct_stair_contract = None
        self.high_obstacle_latched = False
        self.high_obstacle_threshold_m = .05
        self.medium_obstacle_threshold_m = .03
        self.recovery_active = False
        self.recovery_stall_s = 0.
        self.recovery_steps = 0
        self.recovery_best_x = None
        self.stair_suspension = None
        self.payload_clamp_target_rad = 0.
        self.direct_target = np.zeros(16)
        self.direct_target_initialized = False
        self.skill_residual = np.zeros(16)
        self.skill_previous = np.zeros(16)
        self.joint_action_scales = None
        self.target_slew_rad_s = None
        self.task_impedance = None
        if direct_profile is not None:
            contract = direct_profile.get("contract")
            expected_observation = (258 if contract == "sai-task-space-skills-v6" else 243 if contract == "sai-phase-free-stairs-v4"
                                    else 242 if contract in ("sai-phase-free-stairs-v3", "sai-safe-residual-stairs-v5") else 104)
            if (contract not in ("sai-phase-free-stairs-v2", "sai-phase-free-stairs-v3", "sai-phase-free-stairs-v4", "sai-safe-residual-stairs-v5", "sai-task-space-skills-v6")
                    or direct_profile.get("observation_size") != expected_observation
                    or direct_profile.get("action_size") != 16):
                raise ValueError("Invalid phase-free stair policy contract")
            actor = (Path(stair_profile).parent / direct_profile["actor"]).resolve()
            if hashlib.sha256(actor.read_bytes()).hexdigest() != direct_profile["onnx_sha256"]:
                raise ValueError("Phase-free stair actor hash mismatch")
            options = ort.SessionOptions();options.intra_op_num_threads = 2;options.inter_op_num_threads = 1
            self.stair_policy = ort.InferenceSession(str(actor), options, providers=["CPUExecutionProvider"])
            self.stair_profile_id = direct_profile["id"]
            self.direct_stair_contract = contract
            self.experimental_profile = True
            self.stair_settings.update(direct_profile.get("control", {}))
            self.payload_clamp_target_rad = float(direct_profile.get("payload_clamp_target_rad", 0.))
            if contract == "sai-safe-residual-stairs-v5":
                self.joint_action_scales = np.asarray(direct_profile.get("joint_action_scales", []), dtype=float)
                self.target_slew_rad_s = np.asarray(direct_profile.get("target_slew_rad_s", []), dtype=float)
                if self.joint_action_scales.shape != (3,) or self.target_slew_rad_s.shape != (3,):
                    raise ValueError("Invalid safe-residual stair envelope")
            if contract == "sai-phase-free-stairs-v4":
                self.high_obstacle_threshold_m = float(direct_profile.get("high_obstacle_threshold_m", .05))
                self.medium_obstacle_threshold_m = float(direct_profile.get("medium_obstacle_threshold_m", .03))
                self.stair_suspension = Suspension(direct_profile["stair_suspension_geometry"], apply_on_stairs=True)
            # v6 is a residual policy: the accepted suspension and stance
            # impedance above remain the low-level controller.  Replacing that
            # layer made a near-zero residual stall at a 40 mm edge even though
            # the unchanged rolling prior completed the same course.

    def command(self, state):
        if self.direct_stair_contract == "sai-task-space-skills-v6":
            return self._task_space_skill_command(state)
        stair_actor = self.stair_policy
        relevant = self._step_in_wheel_path(state)
        request = state["command"]
        if self.direct_stair_contract == "sai-phase-free-stairs-v4":
            if float(request[0]) <= .015:
                self.high_obstacle_latched = False
                self._clear_recovery()
            dense = np.asarray(state.get("terrain_edge_heights", []), dtype=float)
            if dense.shape == (138,):
                rises = dense.reshape(46, 3)[1:] - dense.reshape(46, 3)[:-1]
                maximum_rise = float(np.max(rises))
                if maximum_rise > self.high_obstacle_threshold_m:
                    self.high_obstacle_latched = True
                    self._clear_recovery()
                else:
                    edge_x = -.30 + .02 * np.arange(45)
                    near = (edge_x >= -.22) & (edge_x <= .20)
                    medium_edge_near = bool(np.any(rises[near] > self.medium_obstacle_threshold_m))
                    if (maximum_rise > self.medium_obstacle_threshold_m and medium_edge_near
                            and float(request[0]) > .015):
                        self._update_recovery(float(state["base_position"][0]))
                    else:
                        self._clear_recovery()
            expert_high = self.high_obstacle_latched or self.recovery_active
            state = dict(state, high_obstacle_latched=float(expert_high))
        descending = self.roll_descent and descending_in_path(state) and float(request[0]) > .015
        if descending:
            state = dict(state, command=[min(.16, float(request[0])), request[1], request[2]])
        crouch_blocked = (float(request[2]) > .5 and float(request[0]) > .015
                          and np.ptp(state["terrain_heights"]) > .008
                          and self._step_in_wheel_path(state, honor_course=False))
        if crouch_blocked:
            # Stay low at a real edge instead of repeatedly trying a stair gait
            # with insufficient clearance. Turning/reversing away remain live.
            state = dict(state, command=[0., request[1], request[2]])
        direct = (self.direct_stair_profile is not None and not descending and not crouch_blocked
                  and relevant and np.ptp(state["terrain_heights"]) > .004 and float(request[0]) > .015)
        if direct:
            result = self._direct_stair_command(state)
            result["flat_policy_id"] = self.flat_policy_id
            result["flat_policy_sha256"] = self.flat_policy_sha256
            result["driving_profile"] = "sai-driving-20260913"
            result["terrain_step_relevant"] = relevant
            suspension = self.stair_suspension if (self.high_obstacle_latched or self.recovery_active) else self.suspension
            if suspension is not None:
                result = suspension.apply(result, state)
                result["suspension_profile"] = self.suspension_id
            if self.impedance is not None and len(state.get("wheel_ground_heights", [])) == 4:
                result = self.impedance.apply(result, state)
            return result
        # alpha.3 has no selection hook. Gate its optional actor for this one
        # sequential request, without altering raw observations or target math.
        if not relevant:
            self.stair_policy = None
        try:
            result = super().command(state)
        finally:
            self.stair_policy = stair_actor
        result["flat_policy_id"] = self.flat_policy_id
        result["flat_policy_sha256"] = self.flat_policy_sha256
        result["driving_profile"] = "sai-driving-20260913"
        result["terrain_step_relevant"] = relevant
        if descending:
            result["stage"] = "descending"
        if crouch_blocked:
            result["stage"] = "crouch_blocked"
        if self.suspension is not None:
            result = self.suspension.apply(result, state)
            result["suspension_profile"] = self.suspension_id
        if self.impedance is not None and len(state.get("wheel_ground_heights", [])) == 4:
            result = self.impedance.apply(result, state)
        return result

    def _task_space_skill_command(self, state):
        """Run the rolling prior continuously and add one learned endpoint skill."""
        stair_actor = self.stair_policy
        self.stair_policy = None
        prior_state = dict(state)
        # Keep the accepted flat actor responsible only for rolling.  The v6
        # suspension and task skill own rough terrain and edges respectively.
        # Zeroing its terrain channels also makes an isolated edge and the
        # same edge in a staircase share the same rolling prior.
        prior_state["terrain_heights"] = [float(state["base_position"][2]) - .2192] * 24
        try:
            result = super().command(prior_state)
        finally:
            self.stair_policy = stair_actor
        base_target = np.asarray(result["target_leg"], dtype=float)
        # The rolling actor may bias front versus rear traction, but the
        # deterministic heading loop exclusively owns left/right differential.
        flat_action = np.asarray(result["policy_action"], dtype=float)
        wheel_action = flat_action[3::4].copy()
        wheel_action[:2] = wheel_action[:2].mean()
        wheel_action[2:] = wheel_action[2:].mean()
        command = np.asarray(state["command"], dtype=float)
        sides = np.array([1., -1., 1., -1.])
        base_target[3::4] = sides * ((command[0] - command[1] * sides * .146) / .048
                                    + 6. * wheel_action)
        rotation = np.asarray(state["base_rotation_columns"], dtype=float).T
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        yaw_rate = float(np.dot(rotation[:, 2], np.asarray(state["base_angular_world"], dtype=float)))
        desired = yaw if self.heading.desired is None else float(self.heading.desired)
        error = float(np.arctan2(np.sin(desired - yaw), np.cos(desired - yaw)))
        correction = float(np.clip(1.5 * error - .25 * (yaw_rate - command[1]),
                                   -self.heading.max_correction, self.heading.max_correction))
        base_target[3::4] -= correction * .146 / .048
        observation = stair_observation_v6(state, state["command"], self.skill_previous, base_target)
        raw = self.stair_policy.run(None, {self.stair_policy.get_inputs()[0].name: observation[None]})[0][0]
        action = np.clip(raw, -1., 1.) if np.linalg.norm(np.asarray(state["command"])[:2]) > 1e-5 else np.zeros(16)
        columns = np.asarray(state["base_rotation_columns"], dtype=float)
        action = project_action_safety(action, columns[:, 2])
        skill_relevant = self._step_in_wheel_path(state, honor_course=False)
        if not skill_relevant:
            # Rough terrain belongs to the continuously running suspension.
            # The learned residual is enabled only for a discrete edge in the
            # wheel path, keeping out-of-distribution road texture from
            # becoming an invented stair maneuver.
            action[15] = 0.
        target = stair_targets_v6(action, base_target, self.skill_residual)
        self.skill_residual = target - base_target
        self.direct_target = target.copy()
        self.skill_previous = action.copy()
        result.update(
            stage="task_skill" if action[15] > .05 else result["stage"],
            target_leg=target.tolist(), wheel_speed=target[3::4].tolist(),
            policy_action=action.tolist(), policy_observation=observation.tolist(),
            stair_profile=self.stair_profile_id, contract_id=self.direct_stair_contract,
            skill_intensity=float(max(0., action[15])),
            terrain_observation_contract="godot-oracle-terrain",
            flat_prior_terrain_mode="level",
            terrain_step_relevant=skill_relevant,
        )
        if self.suspension is not None:
            result = self.suspension.apply(result, state)
            result["suspension_profile"] = self.suspension_id
        # Suspension is the last geometric contribution.  Re-project the
        # composed v6 command before impedance applies forces.
        safe_target = project_target_safety(result["target_leg"])
        result["target_leg"] = safe_target.tolist()
        result["wheel_speed"] = safe_target[3::4].tolist()
        if self.impedance is not None and len(state.get("wheel_ground_heights", [])) == 4:
            result = self.impedance.apply(result, state)
        return result

    def _clear_recovery(self):
        self.recovery_active = False
        self.recovery_stall_s = 0.
        self.recovery_steps = 0
        self.recovery_best_x = None

    def _update_recovery(self, forward_x):
        if self.recovery_active:
            self.recovery_steps += 1
            if self.recovery_steps >= 25:
                self.recovery_active = False
                self.recovery_steps = 0
                self.recovery_stall_s = 0.
                self.recovery_best_x = forward_x
        elif self.recovery_best_x is None or forward_x > self.recovery_best_x + .01:
            self.recovery_best_x = forward_x
            self.recovery_stall_s = 0.
        else:
            self.recovery_stall_s += .02
            if self.recovery_stall_s >= .6:
                self.recovery_active = True
                self.recovery_steps = 0

    def _direct_stair_command(self, state):
        q = np.asarray(state["q"], dtype=float);v = np.asarray(state["v"], dtype=float)
        rotation = np.asarray(state["base_rotation_columns"], dtype=float).T
        self.data.qpos[:3] = state["base_position"]
        self.data.qpos[3:7] = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
        self.data.qvel[:3] = state["base_linear_world"]
        self.data.qvel[3:6] = rotation.T @ np.asarray(state["base_angular_world"])
        self.data.qpos[self.qadr] = q;self.data.qvel[self.vadr] = v
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        command = np.asarray(state["command"], dtype=float).copy()
        command[0] = min(command[0], float(self.stair_settings.get("speed", .16)))
        self.crouch += np.clip(command[2] - self.crouch, -.04, .04)
        observation = (stair_observation_v4(state, command, self.previous)
                       if self.direct_stair_contract == "sai-phase-free-stairs-v4"
                       else stair_observation_v3(state, command, self.previous)
                       if self.direct_stair_contract in ("sai-phase-free-stairs-v3", "sai-safe-residual-stairs-v5")
                       else stair_observation_v2(state, command, self.previous))
        action = np.clip(self.stair_policy.run(None, {self.stair_policy.get_inputs()[0].name: observation[None]})[0][0], -1., 1.)
        target = (stair_targets_v5(action, command, self.direct_target,
                    wheel_residual_scale=float(self.stair_settings.get("wheel_residual_scale", 6.)),
                    joint_action_scales=self.joint_action_scales,
                    target_slew_rad_s=self.target_slew_rad_s)
                  if self.direct_stair_contract == "sai-safe-residual-stairs-v5" else
                  stair_targets_v2(action, command, self.crouch,
                    wheel_residual_scale=float(self.stair_settings.get("wheel_residual_scale", 6.))))
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        target = self.heading.apply(target, command, yaw, self.data.qvel[5])
        self.direct_target = np.asarray(target, dtype=float)
        self.previous = action
        return dict(mode="transport", stage="stairs", target_leg=target.tolist(), wheel_speed=target[3::4].tolist(),
            target_arm=[0.] * 6, arm_bias=self.data.qfrc_bias[self.armv].tolist(), grip_cap=1.4,
            cargo_target_rad=self.payload_clamp_target_rad, policy_action=action.tolist(), policy_observation=observation.tolist(),
            physics_advanced_by_controller=False, effective_crouch=float(self.crouch),
            stair_profile=self.stair_profile_id, contract_id=self.direct_stair_contract)

    def _step_in_wheel_path(self, state, honor_course=True):
        if self.roll_descent and descending_in_path(state):
            return False
        if honor_course and (state.get("stair_course") or self.experimental_profile):
            return True
        return step_in_path(state, crouch=float(state["command"][2]) > .5)
