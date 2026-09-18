#!/usr/bin/env python3
"""Train the Sai v6 task-space skill residual in MuJoCo-Warp.

The accepted rolling prior, terrain suspension, task-space residual, inverse
kinematics, stance impedance and actuator projection run in the same order used
by native deployment.  The policy never owns an absolute joint pose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco_warp as mw
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
import torch
import torch.nn.functional as F
from tensordict import TensorDict
import warp as wp
from rsl_rl.runners import OnPolicyRunner

from sai_agent.paths import resource_root
from sim2sim.research.torch_walking import inverse_rotate, rotate
from sim2sim.sai_stair_v6 import ACTION_SIZE, OBSERVATION_SIZE, continuous_stair_heights
from sim2sim.sai_stair_v6 import LOCAL_STEP_THRESHOLD_SCALED
from sim2sim.sai_stair_v6 import STAIR_CLEARANCE_M
from sim2sim.sai_stair_v6 import PITCH_GUARD_START_RAD, PITCH_GUARD_FULL_RAD


ROOT = Path(__file__).resolve().parents[1]
LEG = torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14], device="cuda")
WHEEL = torch.tensor([3, 7, 11, 15], device="cuda")
SIDES = torch.tensor([1., -1., 1., -1.], device="cuda")
FRONTS = torch.tensor([1., 1., -1., -1.], device="cuda")
TARGET_SLEW_RAD_S = torch.tensor([2., 4., 6., 20.] * 4, device="cuda")
JOINT_LIMITS = torch.tensor([.45, .70, 1.20] * 4, device="cuda")
SCAN_X = torch.tensor(np.repeat(np.arange(8) * .18 - .36, 3), device="cuda", dtype=torch.float32)
PATH_X = torch.tensor(np.repeat([-.18, 0., .18, .36, .54], 3), device="cuda", dtype=torch.float32)
DENSE_X = torch.tensor(np.repeat(np.arange(46) * .02 - .30, 3), device="cuda", dtype=torch.float32)


def _absolute_meshes(xml: ET.Element, root: Path) -> None:
    compiler = xml.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str(root / "models/full/assets"))
    for mesh in xml.findall("./asset/mesh"):
        value = mesh.get("file")
        if value and not Path(value).is_absolute():
            candidate = root / "models/full" / value
            if not candidate.exists():
                candidate = root / "models/full/assets" / value
            mesh.set("file", str(candidate))


class SaiStairWorld:
    """Vectorized, physically articulated Sai and a bank of true box stairs."""

    def __init__(self, count: int, seed: int, max_steps: int = 1200, min_rise_mm: float = 0.,
                 teacher_weight_low: float = 0., teacher_weight_high: float = 0., cargo_accel_weight: float = .012,
                 action_rate_weight: float = .05, reference_weight: float = 0.,
                 lift_event_weight: float = 0., position_objective: bool = False,
                 payload_clamp: bool = False, position_reward_scale: float = 400.,
                 domain_randomization: bool = False, joint_limit_weight: float = 12.,
                 joint_speed_weight: float = .01, upright_weight: float = 12.,
                 support_weight: float = 3., terrain_stage: str = "low",
                 isolated_step: bool = False, fixed_curriculum: bool = False):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for MuJoCo-Warp training")
        self.device = "cuda"
        self.num_envs = count
        self.num_actions = ACTION_SIZE
        self.max_episode_length = max_steps
        self.episode_length_buf = torch.zeros(count, device="cuda", dtype=torch.long)
        self.cfg = {"physics": "MuJoCo-Warp", "control_hz": 50, "phase_free": True}
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        self.teacher_weight_low = teacher_weight_low
        self.teacher_weight_high = teacher_weight_high
        self.cargo_accel_weight = cargo_accel_weight
        self.action_rate_weight = action_rate_weight
        self.reference_weight = reference_weight
        self.lift_event_weight = lift_event_weight
        self.position_objective = position_objective
        self.position_reward_scale = position_reward_scale
        self.payload_clamp = payload_clamp
        self.domain_randomization = domain_randomization
        self.joint_limit_weight = joint_limit_weight
        self.joint_speed_weight = joint_speed_weight
        self.upright_weight = upright_weight
        self.support_weight = support_weight
        self.terrain_stage = terrain_stage
        self.isolated_step = isolated_step
        self.fixed_curriculum = fixed_curriculum
        self.reference_parameters = None
        self.total_steps = 0
        self.lane_centers_cpu = np.linspace(-2.75, 2.75, 12).astype(np.float32)
        self.rises_cpu = np.array([0., .010, .015, .020, .025, .030, .035, .040, .045, .050, .055, .060], np.float32)
        self.treads_cpu = np.array([.18, .18, .16, .20, .14, .18, .22, .16, .20, .14, .18, .22], np.float32)
        self.starts_cpu = np.array([-.12, -.10, -.14, -.08, -.16, -.12, -.06, -.15, -.10, -.14, -.08, -.12], np.float32)
        # Every lane contains measured-contact terrain before the stairs.  A
        # raised 4 mm nominal strip with 0--8 mm tiles represents both bumps
        # and shallow potholes relative to the local rolling surface.
        self.rough_x_min = -.86
        self.rough_dx = .15
        self.rough_x_max = 1.30
        rough_centers = np.arange(self.rough_x_min + self.rough_dx / 2,
                                  self.rough_x_max, self.rough_dx)
        rough_rng = np.random.default_rng(20260917)
        rough = rough_rng.normal(size=(len(self.rises_cpu), len(rough_centers)))
        rough = np.stack([np.convolve(row, np.array([1., 2., 3., 2., 1.]) / 9., mode="same")
                          for row in rough])
        rough /= np.maximum(np.max(np.abs(rough), axis=1, keepdims=True), 1e-6)
        self.rough_centers_cpu = rough_centers.astype(np.float32)
        self.rough_heights_cpu = np.clip(.004 + .004 * rough, 0., .008).astype(np.float32)
        self.terrain_x_min = -1.0
        self.terrain_dx = .02
        self.terrain_x_cpu = np.arange(self.terrain_x_min, 1.401, self.terrain_dx).astype(np.float32)
        self.course_heights_cpu = np.zeros((len(self.rises_cpu), len(self.terrain_x_cpu)), np.float32)
        for lane, (rise, tread, start) in enumerate(zip(self.rises_cpu, self.treads_cpu, self.starts_cpu)):
            rough_index = np.floor((self.terrain_x_cpu - self.rough_x_min) / self.rough_dx).astype(int)
            rough_valid = (rough_index >= 0) & (rough_index < self.rough_heights_cpu.shape[1])
            rough_height = np.zeros_like(self.terrain_x_cpu)
            rough_height[rough_valid] = self.rough_heights_cpu[lane, rough_index[rough_valid]]
            if rise <= 0:
                self.course_heights_cpu[lane] = rough_height
            else:
                if isolated_step:
                    # A one-edge curriculum needs enough landing surface for
                    # the complete 240 mm wheelbase.  A normal stair tread is
                    # only 140--220 mm, so exposing the second riser made the
                    # nominal one-edge task mechanically a two-edge task.
                    stairs = np.where(self.terrain_x_cpu >= start, rise, 0.)
                else:
                    stairs = continuous_stair_heights(
                        self.terrain_x_cpu, float(start), float(tread), float(rise))
                self.course_heights_cpu[lane] = np.where(self.terrain_x_cpu < start, rough_height, stairs)
        stage_mask = {
            "rough": self.rises_cpu == 0.,
            "low": self.rises_cpu <= .040,
            "high": self.rises_cpu >= .040,
            "mixed": np.ones_like(self.rises_cpu, dtype=bool),
        }.get(terrain_stage)
        if stage_mask is None:
            raise ValueError(f"Unknown terrain stage: {terrain_stage}")
        lane_pool = np.flatnonzero(stage_mask & (self.rises_cpu * 1000 >= min_rise_mm - 1e-6))
        if not len(lane_pool):
            raise ValueError("Minimum rise excludes every training lane")
        self.lane_pool_cpu = lane_pool
        self.lane_pool = torch.tensor(lane_pool, device="cuda", dtype=torch.long)
        self.lane_centers = torch.tensor(self.lane_centers_cpu, device="cuda")
        self.rises = torch.tensor(self.rises_cpu, device="cuda")
        self.treads = torch.tensor(self.treads_cpu, device="cuda")
        self.starts = torch.tensor(self.starts_cpu, device="cuda")
        self.course_heights = torch.tensor(self.course_heights_cpu, device="cuda")
        # Introduce one new stair height at a time.  Sampling the first four
        # high-step lanes together yielded no successful rollouts and taught
        # the actor to repeat a front-wheel lift at the first edge.
        self.curriculum_count = 1
        self.curriculum_edges = 1
        self.model = self._build_model()
        self.model.opt.timestep = .002
        m = self.model
        self.chassis = m.body("chassis").id
        self.wheel_bodies = torch.tensor([m.body(x + "_wheel").id for x in
            ("front_left", "front_right", "rear_left", "rear_right")], device="cuda")
        self.payload_body = m.body("payload").id
        free = next(i for i in range(m.njnt) if m.jnt_bodyid[i] == self.chassis and m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
        self.qa, self.va = int(m.jnt_qposadr[free]), int(m.jnt_dofadr[free])
        self.qi = torch.tensor([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(16)], device="cuda")
        self.vi = torch.tensor([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(16)], device="cuda")
        self.arm_qi = torch.tensor([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(16, 22)], device="cuda")
        self.arm_vi = torch.tensor([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(16, 22)], device="cuda")
        pj = m.joint("payload_free").id
        self.payload_qa, self.payload_va = int(m.jnt_qposadr[pj]), int(m.jnt_dofadr[pj])
        cargo_joint = m.joint("cargo_drive").id
        self.cargo_qi = int(m.jnt_qposadr[cargo_joint])
        self.cargo_vi = int(m.jnt_dofadr[cargo_joint])
        self.cargo_target = .067 / .01909859317102744 if payload_clamp else 0.
        initial = mujoco.MjData(m)
        mujoco.mj_forward(m, initial)
        self.initial_qpos = torch.tensor(initial.qpos, device="cuda", dtype=torch.float32)
        self.initial_qpos[self.cargo_qi] = self.cargo_target
        wp.init()
        self.stream = wp.Stream("cuda:0")
        self.torch_stream = wp.stream_to_torch(self.stream)
        with wp.ScopedStream(self.stream):
            self.wm = mw.put_model(m)
            self.wd = mw.put_data(m, initial, nworld=count, nconmax=384, njmax=8192)
            self.qpos = wp.to_torch(self.wd.qpos)
            self.qvel = wp.to_torch(self.wd.qvel)
            self.ctrl = wp.to_torch(self.wd.ctrl)
        self.last_action = torch.zeros((count, 16), device="cuda")
        self.last_target = torch.zeros((count, 16), device="cuda")
        self.flat_previous = torch.zeros((count, 16), device="cuda")
        self.suspension_offset = torch.zeros((count, 4), device="cuda")
        self.stance_contact = torch.zeros((count, 4), device="cuda")
        self.stance_weights = torch.zeros((count, 4), device="cuda")
        self.support_force = torch.zeros(count, device="cuda")
        self.height_reference = torch.zeros(count, device="cuda")
        self.feedforward = torch.zeros((count, 16), device="cuda")
        self._load_flat_prior(ROOT / "src/sim2sim/assets/sai/flat-motion-v1.onnx")
        self.action_history = torch.zeros((count, 3, 16), device="cuda")
        self.action_delay = torch.zeros(count, device="cuda", dtype=torch.long)
        self.motor_strength = torch.ones((count, 1), device="cuda")
        self.last_payload_velocity = torch.zeros((count, 3), device="cuda")
        self.filtered_cargo_accel = torch.zeros((count, 3), device="cuda")
        self.previous_airborne = torch.zeros((count, 4), device="cuda", dtype=torch.bool)
        self.high_expert_phase = torch.zeros(count, device="cuda", dtype=torch.long)
        self.lane = torch.zeros(count, device="cuda", dtype=torch.long)
        self.goal_edges = torch.ones(count, device="cuda", dtype=torch.long)
        self.start_x = torch.full((count,), -.68, device="cuda")
        self.command = torch.zeros((count, 3), device="cuda")
        self.command[:, 0] = .16
        self.heading_reference = torch.zeros(count, device="cuda")
        self.last_goal_distance = torch.zeros(count, device="cuda")
        self.successes = self.falls = self.cargo_losses = 0
        self.safety_violations = self.joint_violations = self.tilt_violations = 0
        self.numerical_failures = 0
        self.last_numerical_diagnostic = {}
        self.lane_attempts = torch.zeros(len(self.rises_cpu), device="cuda", dtype=torch.long)
        self.lane_successes = torch.zeros(len(self.rises_cpu), device="cuda", dtype=torch.long)
        self._last_metrics = {}
        self.last_done = torch.zeros(count, device="cuda", dtype=torch.bool)
        self.last_success = torch.zeros(count, device="cuda", dtype=torch.bool)
        self.last_fall = torch.zeros(count, device="cuda", dtype=torch.bool)
        self.last_safety_violation = torch.zeros(count, device="cuda", dtype=torch.bool)
        self.last_unsafe_joint = torch.zeros(count, device="cuda", dtype=torch.bool)
        self.last_unsafe_tilt = torch.zeros(count, device="cuda", dtype=torch.bool)
        self.last_up = torch.ones(count, device="cuda")
        self.last_body_x = torch.zeros(count, device="cuda")
        self.last_body_z = torch.zeros(count, device="cuda")
        self.last_quat = torch.zeros((count, 4), device="cuda")
        self.last_wheel_x = torch.zeros((count, 4), device="cuda")
        self.last_wheel_z = torch.zeros((count, 4), device="cuda")
        self.last_cargo_accel_norm = torch.zeros(count, device="cuda")
        self.last_stance_weights = torch.zeros((count, 4), device="cuda")
        self.last_support_force = torch.zeros(count, device="cuda")
        self.last_joint_position = torch.zeros((count, 16), device="cuda")
        self.last_joint_velocity = torch.zeros((count, 16), device="cuda")
        self.last_joint_target = torch.zeros((count, 16), device="cuda")
        self.last_joint_torque = torch.zeros((count, 16), device="cuda")
        self.reset(torch.arange(count, device="cuda"))

    def _load_flat_prior(self, path: Path) -> None:
        graph = onnx.load(str(path))
        values = {item.name: torch.from_numpy(numpy_helper.to_array(item).copy()).to("cuda")
                  for item in graph.graph.initializer}
        self.flat_mean = values["obs_normalizer._mean"]
        self.flat_scale = values["onnx::Div_20"]
        self.flat_layers = [(values[f"mlp.{index}.weight"], values[f"mlp.{index}.bias"])
                            for index in (0, 2, 4)]

    def _build_model(self) -> mujoco.MjModel:
        root = resource_root()
        xml = ET.parse(root / "models/full/locomotion-articulated.xml").getroot()
        _absolute_meshes(xml, root)
        world = xml.find("worldbody")
        for lane, (center, rise, tread, start) in enumerate(zip(
                self.lane_centers_cpu, self.rises_cpu, self.treads_cpu, self.starts_cpu)):
            for tile, (x, top) in enumerate(zip(self.rough_centers_cpu, self.rough_heights_cpu[lane])):
                if rise > 0 and x >= start:
                    continue
                ET.SubElement(world, "geom", name=f"lane_{lane}_rough_{tile}", type="box",
                    pos=f"{x} {center} {float(top) - .20}",
                    size=f"{self.rough_dx / 2} .225 .20", contype="2", conaffinity="5", group="5")
            if rise <= 0:
                continue
            levels = range(1, 2) if self.isolated_step else range(1, 5)
            for level in levels:
                left = start + (level - 1) * tread
                right = (1.4 if self.isolated_step else
                         start + level * tread if level < 4 else 1.4)
                top = level * rise
                ET.SubElement(world, "geom", name=f"lane_{lane}_step_{level}", type="box",
                    pos=f"{(left + right) / 2} {center} {top - .20}",
                    size=f"{(right - left) / 2} .225 .20", contype="2", conaffinity="5", group="5")
        payload = ET.SubElement(world, "body", name="payload", pos="-.09 0 .284")
        ET.SubElement(payload, "freejoint", name="payload_free")
        mass = .10
        inertia = mass / 12 * np.array([.03**2 + .04**2, .04**2 + .04**2, .04**2 + .03**2])
        ET.SubElement(payload, "inertial", pos="0 0 0", mass=str(mass), diaginertia=" ".join(map(str, inertia)))
        ET.SubElement(payload, "geom", name="payload", type="box", size=".02 .015 .02", mass=str(mass),
                      friction="2 .05 .005", contype="4", conaffinity="3", group="4")
        # The source masks intentionally disable robot/robot collision.  v6 adds
        # the minimum critical non-adjacent pairs so the learner cannot solve a
        # stair by folding a lower leg or wheel through the chassis.
        contact = xml.find("contact")
        if contact is None:
            contact = ET.SubElement(xml, "contact")
        chassis_geoms = [g.get("name") for g in xml.findall(".//body[@name='chassis']/geom") if g.get("name")]
        for corner in ("front_left", "front_right", "rear_left", "rear_right"):
            for moving in (f"{corner}_lower_contact_0", f"{corner}_wheel_contact_0"):
                for chassis_geom in chassis_geoms:
                    ET.SubElement(contact, "pair", geom1=moving, geom2=chassis_geom,
                                  condim="3", margin=".008", gap=".004")
        return mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))

    def _sync_forward(self) -> None:
        caller = torch.cuda.current_stream()
        self.torch_stream.wait_stream(caller)
        with wp.ScopedStream(self.stream):
            mw.forward(self.wm, self.wd)
        caller.wait_stream(self.torch_stream)

    def reset(self, ids: torch.Tensor) -> None:
        if not len(ids):
            return
        active_pool = self.lane_pool[:self.curriculum_count]
        lane = active_pool[torch.randint(len(active_pool), (len(ids),), generator=self.generator, device="cuda")]
        self.lane[ids] = lane
        if self.fixed_curriculum:
            self.goal_edges[ids] = self.curriculum_edges
        else:
            self.goal_edges[ids] = torch.randint(1, self.curriculum_edges + 1, (len(ids),),
                                                 generator=self.generator, device="cuda")
        self.qpos[ids] = self.initial_qpos
        self.qvel[ids] = 0.
        self.ctrl[ids] = 0.
        self.last_action[ids] = 0.
        self.last_target[ids] = 0.
        self.flat_previous[ids] = 0.
        self.suspension_offset[ids] = 0.
        self.stance_contact[ids] = 1.
        self.stance_weights[ids] = .25
        self.support_force[ids] = 0.
        self.feedforward[ids] = 0.
        self.action_history[ids] = 0.
        if self.domain_randomization:
            self.action_delay[ids] = torch.randint(3, (len(ids),), generator=self.generator, device="cuda")
            self.motor_strength[ids] = .85 + .30 * torch.rand((len(ids), 1), generator=self.generator, device="cuda")
        else:
            self.action_delay[ids] = 0.;self.motor_strength[ids] = 1.
        self.filtered_cargo_accel[ids] = 0.
        self.previous_airborne[ids] = False
        self.high_expert_phase[ids] = 0
        self.episode_length_buf[ids] = 0
        x = self.start_x[ids] + (torch.rand(len(ids), generator=self.generator, device="cuda") - .5) * .05
        y = self.lane_centers[lane] + (torch.rand(len(ids), generator=self.generator, device="cuda") - .5) * .025
        yaw = (torch.rand(len(ids), generator=self.generator, device="cuda") - .5) * .08
        self.heading_reference[ids] = yaw
        self.qpos[ids, self.qa] = x
        self.qpos[ids, self.qa + 1] = y
        local_ground = self.terrain_height(x, lane)
        self.qpos[ids, self.qa + 2] = self.initial_qpos[self.qa + 2] + local_ground
        self.qpos[ids, self.qa + 3] = torch.cos(yaw / 2)
        self.qpos[ids, self.qa + 4:self.qa + 6] = 0.
        self.qpos[ids, self.qa + 6] = torch.sin(yaw / 2)
        self.qpos[ids, self.payload_qa] = x - .09 * torch.cos(yaw)
        self.qpos[ids, self.payload_qa + 1] = y - .09 * torch.sin(yaw)
        self.qpos[ids, self.payload_qa + 2] = .284 + local_ground
        payload_yaw = yaw + (np.pi / 2 if self.payload_clamp else 0.)
        self.qpos[ids, self.payload_qa + 3] = torch.cos(payload_yaw / 2)
        self.qpos[ids, self.payload_qa + 4:self.payload_qa + 6] = 0.
        self.qpos[ids, self.payload_qa + 6] = torch.sin(payload_yaw / 2)
        if self.domain_randomization:
            self.qvel[ids, self.va:self.va + 2] = ((torch.rand((len(ids), 2), generator=self.generator,
                device="cuda") - .5) * .10)
        wp.to_torch(self.wd.qacc_warmstart)[ids] = 0.
        self._sync_forward()
        wheel_xyz = wp.to_torch(self.wd.xpos)[ids][:, self.wheel_bodies]
        wheel_ground = self.terrain_height(wheel_xyz[:, :, 0], lane)
        self.height_reference[ids] = wheel_ground.mean(-1) + .2192
        self.last_payload_velocity[ids] = self.qvel[ids, self.payload_va:self.payload_va + 3]
        goal_x = (self.starts[lane] + (self.goal_edges[ids] - 1) * self.treads[lane]
                  + STAIR_CLEARANCE_M)
        self.last_goal_distance[ids] = (goal_x - x).clamp(min=0.)

    def terrain_height(self, x: torch.Tensor, lane: torch.Tensor | None = None) -> torch.Tensor:
        if lane is None:
            lane = self.lane[:, None] if x.ndim == 2 else self.lane
        elif x.ndim == 2 and lane.ndim == 1:
            lane = lane[:, None]
        index = torch.floor((x - self.terrain_x_min) / self.terrain_dx).long()
        valid = (index >= 0) & (index < self.course_heights.shape[1])
        height = self.course_heights[lane, index.clamp(0, self.course_heights.shape[1] - 1)]
        return torch.where(valid, height, torch.zeros_like(height))

    def _state(self):
        pos = self.qpos[:, self.qa:self.qa + 3]
        quat = self.qpos[:, self.qa + 3:self.qa + 7]
        angular_body = self.qvel[:, self.va + 3:self.va + 6]
        linear_world = self.qvel[:, self.va:self.va + 3]
        return pos, quat, angular_body, linear_world

    def _flat_target(self) -> torch.Tensor:
        pos, quat, angular, linear = self._state()
        up_world = torch.zeros_like(pos);up_world[:, 2] = 1.
        projected_up = inverse_rotate(quat, up_world)
        body_linear = inverse_rotate(quat, linear)
        q, v = self.qpos[:, self.qi], self.qvel[:, self.vi]
        phase = self.episode_length_buf.float() * .02 * (2 * np.pi / 2.4)
        flat_observation = torch.cat((
            projected_up, body_linear, angular, self.command,
            q[:, LEG], .1 * v[:, LEG], .1 * v[:, WHEEL] * SIDES,
            self.flat_previous, torch.sin(phase)[:, None], torch.cos(phase)[:, None],
            # v6 assigns terrain rejection to suspension and discrete edges to
            # the task-space skill.  Feeding a whole staircase into the flat
            # actor makes that prior invent a second, conflicting stair gait.
            torch.zeros((self.num_envs, 24), device="cuda"),
        ), dim=1)
        value = (flat_observation - self.flat_mean) / self.flat_scale
        for index, (weight, bias) in enumerate(self.flat_layers):
            value = F.linear(value, weight, bias)
            if index + 1 < len(self.flat_layers):
                value = F.elu(value)
        action = value.clamp(-1., 1.)
        moving = self.command[:, :2].norm(dim=1) > 1e-5
        action = torch.where(moving[:, None], action, torch.zeros_like(action))
        down = torch.full((self.num_envs, 4), .172812737, device="cuda")
        beta = -FRONTS[None] * torch.acos(((down.square() - .09**2 - .11**2) / (2 * .09 * .11)).clamp(-1., 1.))
        theta = -torch.atan2(.11 * torch.sin(beta), .09 + .11 * torch.cos(beta))
        theta0 = FRONTS * np.arctan2(.05, .074833147)
        beta0 = -FRONTS * (np.arctan2(.05, .09797959) + np.arctan2(.05, .074833147))
        target = .18 * action
        target = target.clone()
        target[:, 1::4] += SIDES[None] * (theta0[None] - theta)
        target[:, 2::4] += SIDES[None] * (beta0[None] - beta)
        wheel_action = action[:, WHEEL].clone()
        wheel_action[:, :2] = wheel_action[:, :2].mean(-1, keepdim=True)
        wheel_action[:, 2:] = wheel_action[:, 2:].mean(-1, keepdim=True)
        target[:, WHEEL] = SIDES[None] * (self.command[:, :1] / .048 + 6. * wheel_action)
        # Match the bounded heading hold used by CPU and Godot deployment.
        # Each episode preserves its initial heading; requested yaw rate would
        # advance this reference, but the focused stair command is zero-yaw.
        w, x, y, z = quat.unbind(-1)
        yaw = torch.atan2(2. * (w * z + x * y), 1. - 2. * (y.square() + z.square()))
        error = torch.atan2(torch.sin(self.heading_reference - yaw),
                            torch.cos(self.heading_reference - yaw))
        correction = (1.5 * error - .25 * (angular[:, 2] - self.command[:, 1])).clamp(-.4, .4)
        target[:, WHEEL] -= correction[:, None] * (.146 / .048)
        self._current_flat_action = action
        return target

    def _task_targets(self, actions: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        intensity = actions[:, 15:16].clamp(0., 1.)
        theta0 = np.arctan2(.05, .074833147)
        beta0 = np.arctan2(.05, .09797959) + theta0
        hip, knee = base[:, 1::4], base[:, 2::4]
        theta = FRONTS[None] * theta0 - hip / SIDES[None]
        beta = -FRONTS[None] * beta0 - knee / SIDES[None]
        dx = .09 * torch.sin(theta) + .11 * torch.sin(theta + beta)
        down = .09 * torch.cos(theta) + .11 * torch.cos(theta + beta)
        dx = dx + intensity * .060 * actions[:, 0:4]
        vertical = actions[:, 4:8]
        lift = torch.where(vertical >= 0., .080 * vertical, .010 * vertical)
        down = down - intensity * lift
        down = down + intensity * .025 * actions[:, 12:13]
        down = down + intensity * SIDES[None] * .146 * torch.tan(np.deg2rad(8.) * actions[:, 13:14])
        down = down + intensity * FRONTS[None] * .115 * torch.tan(np.deg2rad(12.) * actions[:, 14:15])
        radius = torch.sqrt(dx.square() + down.square())
        scale = torch.minimum(.194 / radius.clamp(min=1e-8), torch.ones_like(radius))
        dx = dx * scale;down = (down * scale).clamp(.105, .194)
        cosine = ((dx.square() + down.square() - .09**2 - .11**2) / (2 * .09 * .11)).clamp(-1., 1.)
        beta = -FRONTS[None] * torch.acos(cosine)
        theta = torch.atan2(dx, down) - torch.atan2(.11 * torch.sin(beta), .09 + .11 * torch.cos(beta))
        result = base.clone()
        result[:, 0::4] = base[:, 0::4].clamp(-.25, .25)
        result[:, 1::4] = (SIDES[None] * (FRONTS[None] * theta0 - theta)).clamp(-.50, .50)
        result[:, 2::4] = (SIDES[None] * (-FRONTS[None] * beta0 - beta)).clamp(-1.02, 1.02)
        wheel = actions[:, 8:12].clone()
        wheel[:, :2] = wheel[:, :2].mean(-1, keepdim=True)
        wheel[:, 2:] = wheel[:, 2:].mean(-1, keepdim=True)
        result[:, WHEEL] = base[:, WHEEL] + intensity * 6. * wheel * SIDES[None]
        desired_residual = result - base
        delta = (desired_residual - self.last_target).clamp(
            min=-TARGET_SLEW_RAD_S[None] * .02, max=TARGET_SLEW_RAD_S[None] * .02)
        return base + self.last_target + delta

    def _suspension_targets(self, target: torch.Tensor, ground: torch.Tensor,
                            projected_up: torch.Tensor) -> torch.Tensor:
        xy = torch.tensor([[.115, .146], [.115, -.146], [-.115, .146], [-.115, -.146]], device="cuda")
        tilt = projected_up[:, :2] @ xy.T
        deviation = ground - ground.mean(-1, keepdim=True)
        desired = -1.077687564550128 * deviation + .4125468028688285 * tilt
        blend = ((ground.amax(-1) - ground.amin(-1)) / .004).clamp(0., 1.)
        desired = (desired * blend[:, None]).clamp(-.025, .020)
        update = (desired - self.suspension_offset) * (.02 / (.0841540838419023 + .02))
        self.suspension_offset += update.clamp(-.003, .003)
        nominal = torch.full_like(self.suspension_offset, .172812737)
        down = (nominal + self.suspension_offset).clamp(.12, .195)
        def angles(value):
            b = -FRONTS[None] * torch.acos(((value.square() - .09**2 - .11**2) / (2 * .09 * .11)).clamp(-1., 1.))
            t = -torch.atan2(.11 * torch.sin(b), .09 + .11 * torch.cos(b))
            return -SIDES[None] * t, -SIDES[None] * b
        hip, knee = angles(down);hip0, knee0 = angles(nominal)
        result = target.clone()
        result[:, 1::4] += hip - hip0
        result[:, 2::4] += knee - knee0
        return result

    def _stance_impedance(self, ground: torch.Tensor, wheel_xyz: torch.Tensor,
                          pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gap = wheel_xyz[:, :, 2] - ground - .048
        contact = (1. - (gap - .002) / .01).clamp(0., 1.)
        self.stance_contact += (contact - self.stance_contact).clamp(-.25, .25)
        contact = self.stance_contact
        kp = torch.full((self.num_envs, 16), 80., device="cuda")
        kd = torch.full((self.num_envs, 16), 2., device="cuda")
        flex_contact = contact.repeat_interleave(2, dim=1)
        flex = torch.tensor([1, 2, 5, 6, 9, 10, 13, 14], device="cuda")
        kp[:, flex] = 80. + (45.05744684106064 - 80.) * flex_contact
        kd[:, flex] = 2. + (1.1532485502878331 - 2.) * flex_contact
        desired_height = ground.mean(-1) + .2192
        delta = (desired_height - self.height_reference) * (.02 / (.15023543636516334 + .02))
        self.height_reference += delta
        robot_mass = float(self.model.body_subtreemass[self.chassis])
        force = (robot_mass * 9.81 * .9497251199685409
                 + 473.84859697921144 * (self.height_reference - pos[:, 2])
                 + 49.991230469877635 * (delta / .02 - self.qvel[:, self.va + 2])).clamp(0., robot_mass * 9.81 * 1.5)
        com_xy = wp.to_torch(self.wd.subtree_com)[:, self.chassis, :2]
        xy = wheel_xyz[:, :, :2] - com_xy[:, None, :]
        matrix_a = torch.cat((torch.ones((self.num_envs, 1, 4), device="cuda"),
                              xy.transpose(1, 2)), dim=1)
        active = contact > 1e-4
        weights = torch.zeros_like(contact)
        unresolved = active.any(-1)
        rhs = torch.zeros((self.num_envs, 3, 1), device="cuda")
        rhs[:, 0] = 1.
        ridge = torch.tensor([1e-8, 1e-6, 1e-6], device="cuda").diag()[None]
        rows = torch.arange(self.num_envs, device="cuda")
        for _ in range(4):
            masked_contact = contact * active.float()
            gram = (matrix_a * masked_contact[:, None]) @ matrix_a.transpose(1, 2) + ridge
            solution = torch.linalg.solve(gram, rhs)
            values = masked_contact * (matrix_a.transpose(1, 2) @ solution).squeeze(-1)
            invalid = (values < -1e-8) & active
            accepted = unresolved & active.any(-1) & ~invalid.any(-1)
            weights = torch.where(accepted[:, None], values.clamp(min=0.), weights)
            unresolved &= ~accepted
            removable = torch.where(active & unresolved[:, None], values,
                                      torch.full_like(values, torch.inf))
            remove = removable.argmin(-1)
            active[rows[unresolved], remove[unresolved]] = False
        weight_sum = weights.sum(-1, keepdim=True)
        scale = torch.minimum(torch.ones_like(weight_sum), contact.sum(-1, keepdim=True))
        weights *= torch.where(weight_sum > 0., scale / weight_sum.clamp(min=1e-8), 0.)
        self.stance_weights.copy_(weights)
        q = self.qpos[:, self.qi]
        theta0 = np.arctan2(.05, .074833147);beta0 = np.arctan2(.05, .09797959) + theta0
        theta = FRONTS[None] * theta0 - q[:, 1::4] / SIDES[None]
        beta = -FRONTS[None] * beta0 - q[:, 2::4] / SIDES[None]
        dx = .09 * torch.sin(theta) + .11 * torch.sin(theta + beta)
        jac_hip = -dx / SIDES[None]
        jac_knee = -.11 * torch.sin(theta + beta) / SIDES[None]
        ff = torch.zeros((self.num_envs, 16), device="cuda")
        ff[:, 1::4] = -jac_hip * force[:, None] * weights
        ff[:, 2::4] = -jac_knee * force[:, None] * weights
        bias = wp.to_torch(self.wd.qfrc_bias)[:, self.vi]
        ff[:, LEG] += .9497251199685409 * bias[:, LEG] * contact.repeat_interleave(3, dim=1)
        self.feedforward += (ff - self.feedforward) * (.02 / .06)
        self.support_force.copy_(force)
        return kp, kd, self.feedforward

    def _features(self, base_target: torch.Tensor | None = None) -> torch.Tensor:
        if base_target is None:
            base_target = self._flat_target()
        pos, quat, angular, linear = self._state()
        up_world = torch.zeros_like(pos); up_world[:, 2] = 1.
        projected_up = inverse_rotate(quat, up_world)
        body_linear = inverse_rotate(quat, linear)
        q, v = self.qpos[:, self.qi], self.qvel[:, self.vi]
        wheel_xyz = wp.to_torch(self.wd.xpos)[:, self.wheel_bodies]
        ground = self.terrain_height(wheel_xyz[:, :, 0])
        local_ground = ground.mean(-1, keepdim=True)
        terrain = self.terrain_height(pos[:, :1] + SCAN_X[None])
        path = self.terrain_height(pos[:, :1] + PATH_X[None])
        dense = self.terrain_height(pos[:, :1] + DENSE_X[None])
        clearance = wheel_xyz[:, :, 2] - ground - .048
        height_error = pos[:, 2:3] - local_ground - .2192
        obs = torch.cat((projected_up, body_linear, angular, self.command,
            q[:, LEG], .1 * v[:, LEG], .1 * v[:, WHEEL] * SIDES,
            self.last_action,
            ((terrain - local_ground) * 5.).clamp(-2., 2.),
            ((path - local_ground) * 5.).clamp(-2., 2.),
            ((ground - local_ground) * 5.).clamp(-1., 1.),
            (clearance * 20.).clamp(-1., 2.),
            height_error * 10.,
            ((dense - local_ground) * 5.).clamp(-2., 2.), base_target), dim=1)
        if obs.shape[1] != OBSERVATION_SIZE:
            raise RuntimeError(f"Observation contract drift: {obs.shape}")
        local = torch.cat((obs[:, 56:95], obs[:, 104:242]), dim=1)
        higher = torch.where(local > LOCAL_STEP_THRESHOLD_SCALED, local,
                             torch.full_like(local, torch.inf)).amin(-1, keepdim=True)
        lower = torch.where(local < -LOCAL_STEP_THRESHOLD_SCALED, local,
                            torch.full_like(local, -torch.inf)).amax(-1, keepdim=True)
        local = torch.where(torch.isfinite(higher), torch.minimum(local, higher), local)
        local = torch.where(torch.isfinite(lower), torch.maximum(local, lower), local)
        obs[:, 56:95] = local[:, :39]
        obs[:, 104:242] = local[:, 39:]
        return obs

    def get_observations(self) -> TensorDict:
        policy = self._features()
        if not self.position_objective:
            return TensorDict({"policy": policy, "critic": policy}, batch_size=[self.num_envs])
        pos, _, _, _ = self._state()
        wheel_xyz = wp.to_torch(self.wd.xpos)[:, self.wheel_bodies]
        ground = self.terrain_height(wheel_xyz[:, :, 0])
        payload_relative = (wp.to_torch(self.wd.xpos)[:, self.payload_body]
                            - wp.to_torch(self.wd.xpos)[:, self.chassis])
        goal_x = (self.starts[self.lane] + (self.goal_edges - 1) * self.treads[self.lane]
                  + STAIR_CLEARANCE_M)
        privileged = torch.cat((
            self.rises[self.lane, None], self.treads[self.lane, None],
            (self.starts[self.lane] - pos[:, 0])[:, None], (goal_x - pos[:, 0])[:, None],
            self.goal_edges[:, None].float() / 4.,
            wheel_xyz[:, :, 2] - ground - .048,
            payload_relative,
        ), dim=1)
        return TensorDict({"policy": policy, "critic": torch.cat((policy, privileged), dim=1)},
                          batch_size=[self.num_envs])

    def _next_edge(self, wheel_x: torch.Tensor):
        rise = self.rises[self.lane, None]
        tread = self.treads[self.lane, None]
        start = self.starts[self.lane, None]
        edge_index = torch.floor((wheel_x - start) / tread).long() + 1
        valid = (rise > 0.) & (edge_index >= 0) & (edge_index < 4)
        edge = start + edge_index.clamp(0, 3) * tread
        return edge - wheel_x, valid

    def teacher_action(self) -> torch.Tensor:
        """Safe endpoint reference used only for behavior-cloning initialization."""
        wheel_x = wp.to_torch(self.wd.xpos)[:, self.wheel_bodies, 0]
        distance, valid = self._next_edge(wheel_x)
        active = valid & (distance <= .09) & (distance >= -.06)
        phase = ((.09 - distance) / .15).clamp(0., 1.)
        rise = self.rises[self.lane, None]
        lift = (rise + .025).clamp(max=.075) * 4 * phase * (1 - phase)
        dx = -.025 + .065 * phase
        action = torch.zeros((self.num_envs, 16), device="cuda")
        action[:, 0:4] = torch.where(active, dx / .060, 0.)
        action[:, 4:8] = torch.where(active, lift / .080, 0.)
        action[:, 15] = active.any(-1).float()
        return action.clamp(-1., 1.)

    def high_step_expert_action(self) -> torch.Tensor:
        """Privileged state-feedback demonstrator for an isolated high edge.

        This controller is confined to dataset generation.  The deployed ONNX
        receives no phase variable; it must infer the maneuver from the same
        state and terrain observations available during policy training.
        """
        if not self.isolated_step:
            raise RuntimeError("High-step expert requires the isolated-step curriculum")
        wheel_x = wp.to_torch(self.wd.xpos)[:, self.wheel_bodies, 0]
        edge = self.starts[self.lane]
        front = wheel_x[:, :2].amin(-1)
        rear = wheel_x[:, 2:].amin(-1)
        phase = self.high_expert_phase
        phase.copy_(torch.where((phase == 0) & (front >= edge - .10), 1, phase))
        phase.copy_(torch.where((phase == 1) & (front > edge + .03), 2, phase))
        phase.copy_(torch.where((phase == 2) & (rear >= edge - .03), 3, phase))
        phase.copy_(torch.where((phase == 3) & (rear > edge + .03), 4, phase))
        action = torch.zeros((self.num_envs, 16), device="cuda")
        p1, p2, p3, p4 = (phase == index for index in range(1, 5))
        action[p1, 0:2] = .25
        action[p1, 4:6] = .40
        action[p1, 8:12] = .60
        action[p1, 12] = .50
        action[p1, 14] = .25
        action[p2, 8:12] = .60
        action[p2, 12] = .50
        action[p2, 14] = -1.
        action[p3, 2:4] = -.50
        action[p3, 6:8] = .10
        action[p3, 8:12] = 1.
        action[p3, 12] = .50
        action[p3, 14] = -1.
        action[p4, 8:12] = .30
        action[p4, 12] = .20
        engaged = phase > 0
        # Equal open-loop wheel commands succeeded in only 2/16 contact
        # perturbations; closing this measured progress loop reached 16/16.
        side_error = ((wheel_x[:, 0] + wheel_x[:, 2])
                      - (wheel_x[:, 1] + wheel_x[:, 3])) * .5
        correction = (4. * side_error).clamp(-.6, .6)
        action[engaged, 8] -= correction[engaged]
        action[engaged, 10] -= correction[engaged]
        action[engaged, 9] += correction[engaged]
        action[engaged, 11] += correction[engaged]
        action[:, 8:12].clamp_(-1., 1.)
        action[engaged, 15] = 1.
        return action

    def reference_action(self, observation: torch.Tensor) -> torch.Tensor:
        if self.reference_parameters is None:
            return torch.zeros_like(self.last_action)
        value = observation
        for index, (weight, bias) in enumerate(self.reference_parameters):
            value = F.linear(value, weight, bias)
            if index + 1 < len(self.reference_parameters):
                value = F.elu(value)
        return value.clamp(-1., 1.)

    def step(self, actions: torch.Tensor):
        actions = actions.clamp(-1., 1.)
        _, guard_quat, _, _ = self._state()
        guard_up_world = torch.zeros((self.num_envs, 3), device="cuda")
        guard_up_world[:, 2] = 1.
        guard_up = inverse_rotate(guard_quat, guard_up_world)
        guard_pitch = torch.atan2(-guard_up[:, 0], guard_up[:, 2])
        guard_correction = ((guard_pitch.abs() - PITCH_GUARD_START_RAD)
                            / (PITCH_GUARD_FULL_RAD - PITCH_GUARD_START_RAD)).clamp(0., 1.)
        actions = actions.clone()
        guard_active = guard_correction > 0.
        actions[:, 14] = torch.where(
            guard_active & (guard_pitch > 0.), torch.maximum(actions[:, 14], guard_correction),
            torch.where(guard_active & (guard_pitch < 0.), torch.minimum(actions[:, 14], -guard_correction),
                        actions[:, 14]))
        self.action_history.copy_(torch.roll(self.action_history, 1, dims=1))
        self.action_history[:, 0] = actions
        actions = self.action_history[torch.arange(self.num_envs, device="cuda"), self.action_delay]
        previous = self.last_action.clone()
        teacher = self.teacher_action()
        base_target = self._flat_target()
        reference = self.reference_action(self._features(base_target))
        target = self._task_targets(actions, base_target)
        pos0, quat0, _, _ = self._state()
        up_world = torch.zeros_like(pos0);up_world[:, 2] = 1.
        projected_up = inverse_rotate(quat0, up_world)
        wheel_xyz0 = wp.to_torch(self.wd.xpos)[:, self.wheel_bodies]
        ground0 = self.terrain_height(wheel_xyz0[:, :, 0])
        actual_target = self._suspension_targets(target, ground0, projected_up)
        # Suspension is composed after task-space IK, so enforce the reserved
        # joint margin on the final command rather than only on the skill.
        actual_target[:, 0::4].clamp_(-.30, .30)
        actual_target[:, 1::4].clamp_(-.50, .50)
        actual_target[:, 2::4].clamp_(-1.02, 1.02)
        kp, kd, feedforward = self._stance_impedance(ground0, wheel_xyz0, pos0)
        wheel_target = actual_target[:, WHEEL]
        torque = torch.zeros((self.num_envs, self.model.nu), device="cuda")
        caller = torch.cuda.current_stream()
        self.torch_stream.wait_stream(caller)
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            for _ in range(10):
                q, v = self.qpos[:, self.qi], self.qvel[:, self.vi]
                torque[:, :16] = (kp * (actual_target - q) - kd * v + feedforward).clamp(-8., 8.)
                torque[:, WHEEL] = (.4 * (wheel_target - v[:, WHEEL])).clamp(-1.3, 1.3)
                arm_q, arm_v = self.qpos[:, self.arm_qi], self.qvel[:, self.arm_vi]
                bias = wp.to_torch(self.wd.qfrc_bias)[:, self.arm_vi]
                torque[:, 16:22] = (-998.22 * arm_q - 2.731 * arm_v + bias).clamp(-2.94, 2.94)
                torque[:, 21] = torque[:, 21].clamp(-1.4, 1.4)
                torque[:, 22] = (.25 * (self.cargo_target - self.qpos[:, self.cargo_qi])
                                  - .015 * self.qvel[:, self.cargo_vi]).clamp(-.12, .12)
                torque *= self.motor_strength
                self.ctrl.copy_(torque)
                mw.step(self.wm, self.wd)
        caller.wait_stream(self.torch_stream)
        xpos = wp.to_torch(self.wd.xpos)
        bias_force = wp.to_torch(self.wd.qfrc_bias)
        numerical_fields = {
            "qpos": ~torch.isfinite(self.qpos).all(-1),
            "qvel": ~torch.isfinite(self.qvel).all(-1),
            "ctrl": ~torch.isfinite(self.ctrl).all(-1),
            "xpos": ~torch.isfinite(xpos).all((-1, -2)),
            "qfrc_bias": ~torch.isfinite(bias_force).all(-1),
        }
        numerical_failure = torch.zeros(self.num_envs, device="cuda", dtype=torch.bool)
        for invalid in numerical_fields.values():
            numerical_failure |= invalid
        if numerical_failure.any():
            self.numerical_failures += int(numerical_failure.sum())
            self.last_numerical_diagnostic = {
                "control_step": self.total_steps + 1,
                "environment_ids": numerical_failure.nonzero(as_tuple=False).flatten()[:16].tolist(),
                "field_counts": {name: int(invalid.sum()) for name, invalid in numerical_fields.items()},
            }
        self.last_action.copy_(actions)
        self.last_target.copy_(target - base_target)
        self.flat_previous.copy_(self._current_flat_action)
        self.episode_length_buf += 1
        self.total_steps += 1

        pos, quat, angular, linear = self._state()
        body_velocity = inverse_rotate(quat, linear)
        up = 1. - 2. * (quat[:, 1].square() + quat[:, 2].square())
        wheel_xyz = xpos[:, self.wheel_bodies]
        ground = self.terrain_height(wheel_xyz[:, :, 0])
        clearance = wheel_xyz[:, :, 2] - ground - .048
        distance, valid_edge = self._next_edge(wheel_xyz[:, :, 0])
        near = valid_edge & (distance > -.07) & (distance < .10)
        premature = torch.relu(clearance - .010) * (~near)
        airborne = clearance > .015
        lift_events = airborne & ~self.previous_airborne
        premature_lift_events = lift_events & ~near
        self.previous_airborne.copy_(airborne)
        base_ground = self.terrain_height(pos[:, 0])
        height_error = pos[:, 2] - base_ground - .2192
        payload_v = self.qvel[:, self.payload_va:self.payload_va + 3]
        cargo_accel = (payload_v - self.last_payload_velocity) / .02
        self.filtered_cargo_accel += (cargo_accel - self.filtered_cargo_accel) * (.02 / (.015 + .02))
        self.last_payload_velocity.copy_(payload_v)
        relative_payload = xpos[:, self.payload_body] - xpos[:, self.chassis]
        # MuJoCo-Warp's tangential payload contact creeps more than CPU MuJoCo.
        # A closed clamp therefore uses the physical tray envelope for rollout
        # termination; independent CPU/Godot acceptance retains the stricter
        # no-slip boundary and remains authoritative.
        forward_limit = .015 if self.payload_clamp else -.025
        rear_limit = -.20 if self.payload_clamp else -.17
        cargo_lost = ((relative_payload[:, 2] < .02) | (relative_payload[:, 1].abs() > .12)
                      | (relative_payload[:, 0] > forward_limit) | (relative_payload[:, 0] < rear_limit))
        if self.payload_clamp:
            cargo_lost = torch.zeros_like(cargo_lost)
        # Early training terminates after one independently solved edge.  As
        # learning proceeds, the same policy must chain up to all four edges.
        finish = (self.starts[self.lane] + (self.goal_edges - 1) * self.treads[self.lane]
                  + STAIR_CLEARANCE_M)
        goal_distance = (finish - pos[:, 0]).clamp(min=0.)
        goal_progress = (self.last_goal_distance - goal_distance).clamp(-.01, .012)
        self.last_goal_distance.copy_(goal_distance)
        success = wheel_xyz[:, :, 0].amin(-1) > finish
        joint_q = self.qpos[:, self.qi][:, LEG]
        joint_v = self.qvel[:, self.vi][:, LEG]
        joint_utilization = joint_q.abs() / JOINT_LIMITS[None]
        soft_limit_cost = torch.relu(joint_utilization - .75).square().sum(-1)
        speed_cost = torch.relu(joint_v.abs() - 6.).square().sum(-1)
        unsafe_joint = (joint_utilization >= .96).any(-1) | (joint_v.abs() > 12.).any(-1)
        # Match the independent motion-safety acceptance envelope.  A looser
        # training threshold lets the actor exploit states deployment rejects.
        unsafe_tilt = up < float(np.cos(np.deg2rad(15.)))
        fall = (up < .55) | ((pos[:, 2] - base_ground) < .105) | numerical_failure
        safety_violation = unsafe_joint | unsafe_tilt
        timeout = self.episode_length_buf >= self.max_episode_length
        done = fall | safety_violation | cargo_lost | success | timeout
        no_edge_near = ~near.any(-1)
        rise = self.rises[self.lane]
        teacher_weight = torch.where(rise <= .030, self.teacher_weight_low, self.teacher_weight_high)
        teacher_error = (actions - teacher).square().sum(-1)
        reference_error = (actions - reference).square().sum(-1)
        skill_intensity = actions[:, 15].clamp(0., 1.)
        residual_effort = actions[:, :15].square().sum(-1) * skill_intensity
        saturated = (torque[:, LEG].abs() > 7.95).float().mean(-1)
        support_count = (clearance <= .008).sum(-1)
        low_step_support_loss = (rise <= .040).float() * torch.relu(2. - support_count.float())
        lateral_error = pos[:, 1] - self.lane_centers[self.lane]
        task_reward = (self.position_reward_scale * goal_progress if self.position_objective else
                       1.4 * (body_velocity[:, 0] / .16).clamp(-1., 1.5)
                       + .35 * torch.exp(-((body_velocity[:, 0] - .16) / .12).square()))
        reward = (
            .08 + task_reward
            - self.upright_weight * (1. - up).clamp(min=0.)
            - 2.0 * height_error.abs()
            - self.action_rate_weight * (actions - previous).square().sum(-1)
            - .025 * residual_effort
            - 1.0 * (residual_effort + skill_intensity.square()) * no_edge_near
            - teacher_weight * teacher_error
            - self.reference_weight * reference_error
            - 10.0 * premature.sum(-1)
            - self.lift_event_weight * lift_events.float().sum(-1)
            - 2.0 * self.lift_event_weight * premature_lift_events.float().sum(-1)
            - 5.0 * torch.relu(clearance - .075).sum(-1)
            - self.cargo_accel_weight * self.filtered_cargo_accel.norm(dim=-1).clamp(max=80.)
            - self.joint_limit_weight * soft_limit_cost
            - self.joint_speed_weight * speed_cost
            - self.support_weight * low_step_support_loss
            - 8.0 * lateral_error.square()
            - .05 * angular[:, :2].square().sum(-1)
            - 8.0 * saturated
            - .00015 * (torque[:, :16] * self.qvel[:, self.vi]).abs().sum(-1)
            + 30.0 * success.float() - 20.0 * fall.float() - 45.0 * safety_violation.float()
            - 20.0 * cargo_lost.float()
        )
        reward_failure = ~torch.isfinite(reward)
        if reward_failure.any():
            newly_bad = reward_failure & ~numerical_failure
            if newly_bad.any():
                self.numerical_failures += int(newly_bad.sum())
                self.last_numerical_diagnostic = {
                    "control_step": self.total_steps,
                    "environment_ids": reward_failure.nonzero(as_tuple=False).flatten()[:16].tolist(),
                    "field_counts": {"reward_only": int(newly_bad.sum()),
                                     **{name: int(invalid.sum()) for name, invalid in numerical_fields.items()}},
                }
            done |= reward_failure
            fall |= reward_failure
            reward = torch.nan_to_num(reward, nan=-100., posinf=-100., neginf=-100.)
        self.successes += int(success.sum())
        self.falls += int(fall.sum())
        self.cargo_losses += int(cargo_lost.sum())
        self.safety_violations += int(safety_violation.sum())
        self.joint_violations += int(unsafe_joint.sum())
        self.tilt_violations += int(unsafe_tilt.sum())
        terminal = done.nonzero(as_tuple=False).flatten()
        if len(terminal):
            terminal_lanes = self.lane[terminal]
            self.lane_attempts.scatter_add_(0, terminal_lanes, torch.ones_like(terminal_lanes))
            self.lane_successes.scatter_add_(0, terminal_lanes, success[terminal].long())
            hardest = int(self.lane_pool[self.curriculum_count - 1])
            attempts = int(self.lane_attempts[hardest])
            rate = float(self.lane_successes[hardest] / max(attempts, 1))
            if not self.fixed_curriculum and attempts >= 20 and rate >= .65:
                if self.curriculum_edges < 4:
                    self.curriculum_edges += 1
                elif self.curriculum_count < len(self.lane_pool):
                    self.curriculum_count += 1
                    self.curriculum_edges = 1
                self.lane_attempts.zero_();self.lane_successes.zero_()
        self._last_metrics = {
            "/curriculum/max_rise_mm": float(self.rises_cpu[self.lane_pool_cpu[self.curriculum_count - 1]] * 1000),
            "/curriculum/max_edges": float(self.curriculum_edges),
            "/motion/vx": body_velocity[:, 0].mean(),
            "/motion/premature_clearance_m": premature.sum(-1).mean(),
            "/motion/lift_events": lift_events.float().sum(-1).mean(),
            "/motion/premature_lift_events": premature_lift_events.float().sum(-1).mean(),
            "/motion/cargo_accel": cargo_accel.norm(dim=-1).mean(),
            "/motion/cargo_accel_filtered": self.filtered_cargo_accel.norm(dim=-1).mean(),
            "/motion/teacher_error": teacher_error.mean(),
            "/motion/reference_error": reference_error.mean(),
            "/motion/saturation": saturated.mean(),
            "/motion/joint_soft_limit_cost": soft_limit_cost.mean(),
            "/motion/joint_speed_cost": speed_cost.mean(),
            "/motion/support_count": support_count.float().mean(),
            "/motion/skill_intensity": skill_intensity.mean(),
            "/motion/lateral_error": lateral_error.abs().mean(),
            "/outcome/safety_violation": safety_violation.float().mean(),
            "/outcome/success": success.float().mean(),
            "/outcome/fall": fall.float().mean(),
            "/outcome/numerical_failure": numerical_failure.float().mean(),
        }
        self.last_done.copy_(done)
        self.last_success.copy_(success)
        self.last_fall.copy_(fall)
        self.last_safety_violation.copy_(safety_violation)
        self.last_unsafe_joint.copy_(unsafe_joint)
        self.last_unsafe_tilt.copy_(unsafe_tilt)
        self.last_up.copy_(up)
        self.last_body_x.copy_(pos[:, 0])
        self.last_body_z.copy_(pos[:, 2])
        self.last_quat.copy_(quat)
        self.last_wheel_x.copy_(wheel_xyz[:, :, 0])
        self.last_wheel_z.copy_(wheel_xyz[:, :, 2])
        self.last_cargo_accel_norm.copy_(cargo_accel.norm(dim=-1))
        self.last_stance_weights.copy_(self.stance_weights)
        self.last_support_force.copy_(self.support_force)
        self.last_joint_position.copy_(self.qpos[:, self.qi])
        self.last_joint_velocity.copy_(self.qvel[:, self.vi])
        self.last_joint_target.copy_(actual_target)
        self.last_joint_torque.copy_(torque[:, :16])
        ids = terminal
        time_outs = timeout.clone()
        if len(ids):
            self.reset(ids)
        return self.get_observations(), reward, done, {"time_outs": time_outs, "log": self._last_metrics}

    def close(self):
        torch.cuda.synchronize()
        self.qpos = self.qvel = self.ctrl = None
        self.wm = self.wd = None
        torch.cuda.empty_cache()


def train_config(steps: int, init_std: float, learning_rate: float) -> dict:
    return {
        "num_steps_per_env": steps,
        "save_interval": 50,
        "logger": "tensorboard",
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {"class_name": "rsl_rl.models.MLPModel", "hidden_dims": [256, 192, 128],
                  "activation": "elu", "obs_normalization": False,
                  "distribution_cfg": {"class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                                       "init_std": init_std, "std_type": "scalar"}},
        "critic": {"class_name": "rsl_rl.models.MLPModel", "hidden_dims": [256, 192, 128],
                   "activation": "elu", "obs_normalization": False},
        "algorithm": {"class_name": "rsl_rl.algorithms.PPO", "num_learning_epochs": 5,
                      "num_mini_batches": 8, "clip_param": .2, "gamma": .99, "lam": .95,
                      "value_loss_coef": 1., "entropy_coef": .0005, "learning_rate": learning_rate,
                      "max_grad_norm": 1., "use_clipped_value_loss": True,
                      "schedule": "adaptive", "desired_kl": .012,
                      "normalize_advantage_per_mini_batch": False, "rnd_cfg": None, "symmetry_cfg": None},
    }


def imitation_warm_start(env: SaiStairWorld, runner: OnPolicyRunner, steps: int, epochs: int) -> dict:
    initial_count, initial_edges = env.curriculum_count, env.curriculum_edges
    observations, targets = [], []
    with torch.inference_mode():
        for _ in range(steps):
            observations.append(env.get_observations()["policy"].clone())
            action = env.teacher_action()
            targets.append(action.clone())
            env.step(action)
    x, y = torch.cat(observations), torch.cat(targets)
    actor = runner.alg.get_policy()
    optimizer = torch.optim.Adam(actor.parameters(), lr=8e-4)
    final = None
    for _ in range(epochs):
        order = torch.randperm(len(x), device="cuda")
        for ids in order.split(8192):
            estimate = actor(TensorDict({"policy": x[ids]}, batch_size=[len(ids)]))
            loss = (estimate - y[ids]).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            final = float(loss.detach())
    with torch.inference_mode():
        estimate = actor(TensorDict({"policy": x[::17]}, batch_size=[len(x[::17])]))
        mae = float((estimate - y[::17]).abs().mean())
    env.total_steps = 0
    env.curriculum_count = initial_count
    env.curriculum_edges = initial_edges
    env.lane_attempts.zero_();env.lane_successes.zero_()
    env.successes = env.falls = env.cargo_losses = 0
    env.reset(torch.arange(env.num_envs, device="cuda"))
    return {"samples": len(x), "epochs": epochs, "final_mse": final, "held_stride_mae": mae,
            "teacher_scope": "network initialization only; PPO retains full action authority"}


def rolling_prior_warm_start(env: SaiStairWorld, runner: OnPolicyRunner,
                             steps: int, epochs: int) -> dict:
    """Initialize the residual actor while preserving the feasible base motion."""
    initial_count, initial_edges = env.curriculum_count, env.curriculum_edges
    observations, targets = [], []
    with torch.inference_mode():
        for _ in range(steps):
            observations.append(env.get_observations()["policy"].clone())
            action = torch.zeros((env.num_envs, env.num_actions), device="cuda")
            targets.append(action)
            env.step(action)
    x, y = torch.cat(observations), torch.cat(targets)
    actor = runner.alg.get_policy()
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    final = None
    for _ in range(epochs):
        order = torch.randperm(len(x), device="cuda")
        for ids in order.split(8192):
            estimate = actor(TensorDict({"policy": x[ids]}, batch_size=[len(ids)]))
            loss = (estimate - y[ids]).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            final = float(loss.detach())
    with torch.inference_mode():
        estimate = actor(TensorDict({"policy": x[::17]}, batch_size=[len(x[::17])]))
        mae = float(estimate.abs().mean())
    env.total_steps = 0
    env.curriculum_count = initial_count
    env.curriculum_edges = initial_edges
    env.lane_attempts.zero_(); env.lane_successes.zero_()
    env.successes = env.falls = env.cargo_losses = env.numerical_failures = 0
    env.reset(torch.arange(env.num_envs, device="cuda"))
    return {"samples": len(x), "epochs": epochs, "final_mse": final,
            "held_stride_mae": mae,
            "method": "zero task-space residual over rolling prior",
            "teacher_scope": "initialization only; no scripted lift target"}


def high_step_expert_warm_start(env: SaiStairWorld, runner: OnPolicyRunner,
                                steps: int, epochs: int) -> dict:
    """Distill the robust state-feedback demonstrator into the phase-free actor."""
    observations, targets = [], []
    successes_before = env.successes
    with torch.inference_mode():
        for _ in range(steps):
            observations.append(env.get_observations()["policy"].clone())
            action = env.high_step_expert_action()
            targets.append(action.clone())
            env.step(action)
    x, y = torch.cat(observations), torch.cat(targets)
    actor = runner.alg.get_policy()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=6e-4, weight_decay=1e-6)
    final = None
    for _ in range(epochs):
        order = torch.randperm(len(x), device="cuda")
        for ids in order.split(8192):
            estimate = actor(TensorDict({"policy": x[ids]}, batch_size=[len(ids)]))
            loss = (estimate - y[ids]).square().mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
            optimizer.step(); final = float(loss.detach())
    with torch.inference_mode():
        estimate = actor(TensorDict({"policy": x[::17]}, batch_size=[len(x[::17])]))
        mae = float((estimate - y[::17]).abs().mean())
    demonstrations = env.successes - successes_before
    env.total_steps = 0
    env.successes = env.falls = env.cargo_losses = env.numerical_failures = 0
    env.reset(torch.arange(env.num_envs, device="cuda"))
    return {"samples": len(x), "epochs": epochs, "final_mse": final,
            "held_stride_mae": mae, "demonstration_successes": demonstrations,
            "method": "state-feedback high-step expert distilled to phase-free task-space actor",
            "teacher_scope": "training data only; deployment forbidden"}


def learned_teacher_warm_start(env: SaiStairWorld, runner: OnPolicyRunner, source: Path,
                               steps: int, epochs: int) -> dict:
    """Compress terrain-conditioned experts into one actor before position-task PPO."""
    options = ort.SessionOptions();options.intra_op_num_threads = 2;options.inter_op_num_threads = 1
    teacher = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    input_name = teacher.get_inputs()[0].name
    input_size = int(teacher.get_inputs()[0].shape[-1])
    observations, targets = [], []
    with torch.inference_mode():
        for _ in range(steps):
            observation = env.get_observations()["policy"]
            teacher_observation = observation
            if input_size == observation.shape[1] + 1:
                high = (env.rises[env.lane] > .05).float()[:, None]
                teacher_observation = torch.cat((observation, high), dim=1)
            if input_size != teacher_observation.shape[1]:
                raise ValueError(f"Teacher observation mismatch: {input_size} vs {teacher_observation.shape[1]}")
            teacher_numpy = teacher_observation.cpu().numpy()
            action_numpy = np.concatenate([teacher.run(None, {input_name: row[None]})[0]
                                           for row in teacher_numpy], axis=0)
            action = torch.from_numpy(action_numpy).to("cuda")
            observations.append(observation.clone());targets.append(action.clone())
            env.step(action)
    x, y = torch.cat(observations), torch.cat(targets)
    actor = runner.alg.get_policy()
    optimizer = torch.optim.Adam(actor.parameters(), lr=8e-4)
    final = None
    for _ in range(epochs):
        order = torch.randperm(len(x), device="cuda")
        for ids in order.split(8192):
            estimate = actor(TensorDict({"policy": x[ids]}, batch_size=[len(ids)]))
            loss = (estimate - y[ids]).square().mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
            final = float(loss.detach())
    with torch.inference_mode():
        estimate = actor(TensorDict({"policy": x[::17]}, batch_size=[len(x[::17])]))
        mae = float((estimate - y[::17]).abs().mean())
    env.reference_parameters = [(actor.mlp[i].weight.detach().clone(), actor.mlp[i].bias.detach().clone())
                                for i in (0, 2, 4, 6)]
    env.total_steps = 0;env.curriculum_count = min(4, len(env.lane_pool_cpu));env.curriculum_edges = 1
    env.reset(torch.arange(env.num_envs, device="cuda"))
    return {"source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "samples": len(x), "epochs": epochs, "final_mse": final, "held_stride_mae": mae,
            "method": "terrain-conditioned expert compression before position-task PPO"}


def initialize_actor(env: SaiStairWorld, runner: OnPolicyRunner, source: Path) -> dict:
    """Load a feasible phase-free actor before constrained PPO refinement."""
    model = onnx.load(str(source), load_external_data=True)
    values = {item.name: numpy_helper.to_array(item).copy() for item in model.graph.initializer}
    actor = runner.alg.get_policy()
    with torch.no_grad():
        reference_parameters = []
        for index in (0, 2, 4, 6):
            layer = actor.mlp[index]
            source_weight = torch.from_numpy(values[f"mlp.{index}.weight"]).to(layer.weight)
            if source_weight.shape == layer.weight.shape:
                layer.weight.copy_(source_weight)
            elif index == 0 and source_weight.shape[0] == layer.weight.shape[0] and source_weight.shape[1] == 104:
                layer.weight.zero_()
                layer.weight[:, :104].copy_(source_weight)
            else:
                raise ValueError(f"Cannot expand source layer {index}: {source_weight.shape} -> {layer.weight.shape}")
            layer.bias.copy_(torch.from_numpy(values[f"mlp.{index}.bias"]).to(layer.bias))
            reference_parameters.append((layer.weight.detach().clone(), layer.bias.detach().clone()))
    env.reference_parameters = reference_parameters
    env.reset(torch.arange(env.num_envs, device="cuda"))
    return {"source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "method": "feasible actor initialization before constrained PPO"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=480)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--bc-steps", type=int, default=240)
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=160916)
    parser.add_argument("--min-rise-mm", type=float, default=0.)
    parser.add_argument("--terrain-stage", choices=("rough", "low", "high", "mixed"), default="low")
    parser.add_argument("--initialize-actor", type=Path)
    parser.add_argument("--teacher-actor", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path,
                        help="Resume PPO and optimizer state; --iterations remains the total target")
    parser.add_argument("--post-init-bc-steps", type=int, default=0,
                        help="After --initialize-actor, collect this many task-space teacher steps")
    parser.add_argument("--post-init-bc-epochs", type=int, default=0)
    parser.add_argument("--rolling-prior-init", action="store_true",
                        help="Initialize at zero residual over the feasible rolling prior")
    parser.add_argument("--high-step-expert-init", action="store_true",
                        help="Distill the isolated-edge state-feedback expert before PPO")
    parser.add_argument("--init-std", type=float, default=.12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--teacher-weight-low", type=float, default=0.)
    parser.add_argument("--teacher-weight-high", type=float, default=0.)
    parser.add_argument("--cargo-accel-weight", type=float, default=.012)
    parser.add_argument("--action-rate-weight", type=float, default=.05)
    parser.add_argument("--reference-weight", type=float, default=0.)
    parser.add_argument("--lift-event-weight", type=float, default=0.)
    parser.add_argument("--position-objective", action="store_true")
    parser.add_argument("--position-reward-scale", type=float, default=400.)
    parser.add_argument("--payload-clamp", action="store_true")
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--joint-limit-weight", type=float, default=12.)
    parser.add_argument("--joint-speed-weight", type=float, default=.01)
    parser.add_argument("--upright-weight", type=float, default=12.)
    parser.add_argument("--support-weight", type=float, default=3.)
    parser.add_argument("--isolated-step", action="store_true",
                        help="Use one riser followed by a long landing for stage-one skill learning")
    parser.add_argument("--fixed-curriculum", action="store_true",
                        help="Hold the first selected lane and one edge during a focused screen")
    parser.add_argument("--fixed-goal-edges", type=int, choices=(1, 2, 3, 4), default=1,
                        help="Number of edges required when --fixed-curriculum is active")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    started = time.time()
    env = SaiStairWorld(args.envs, args.seed, min_rise_mm=args.min_rise_mm,
                        teacher_weight_low=args.teacher_weight_low,
                        teacher_weight_high=args.teacher_weight_high,
                        cargo_accel_weight=args.cargo_accel_weight,
                        action_rate_weight=args.action_rate_weight,
                        reference_weight=args.reference_weight,
                        lift_event_weight=args.lift_event_weight,
                        position_objective=args.position_objective,
                        position_reward_scale=args.position_reward_scale,
                        payload_clamp=args.payload_clamp,
                        domain_randomization=args.domain_randomization,
                        joint_limit_weight=args.joint_limit_weight,
                        joint_speed_weight=args.joint_speed_weight,
                        upright_weight=args.upright_weight,
                        support_weight=args.support_weight,
                        terrain_stage=args.terrain_stage,
                        isolated_step=args.isolated_step,
                        fixed_curriculum=args.fixed_curriculum)
    if args.fixed_curriculum:
        env.curriculum_edges = args.fixed_goal_edges
        env.reset(torch.arange(env.num_envs, device="cuda"))
    config = train_config(args.steps, args.init_std, args.learning_rate)
    # Keep logging self-contained: the training host does not require optional
    # TensorBoard packages, and every chunk below emits a recoverable checkpoint.
    runner = OnPolicyRunner(env, config, log_dir=None, device="cuda")
    selected_initializers = (sum(x is not None for x in
                                 (args.initialize_actor, args.teacher_actor, args.resume_checkpoint))
                             + int(args.rolling_prior_init) + int(args.high_step_expert_init))
    if selected_initializers > 1:
        raise ValueError("Choose rolling-prior, direct, learned-teacher, or resume initialization")
    resume_infos = None
    if args.resume_checkpoint:
        resume_infos = runner.load(str(args.resume_checkpoint), map_location="cuda")
        warm = {"method": "resume_v6_checkpoint", "source": str(args.resume_checkpoint.resolve()),
                "sha256": hashlib.sha256(args.resume_checkpoint.read_bytes()).hexdigest(),
                "checkpoint_infos": resume_infos}
        env.successes = int(resume_infos.get("success_events", 0))
        env.falls = int(resume_infos.get("falls", 0))
        env.cargo_losses = int(resume_infos.get("cargo_losses", 0))
        env.safety_violations = int(resume_infos.get("safety_violations", 0))
        env.joint_violations = int(resume_infos.get("joint_violations", 0))
        env.tilt_violations = int(resume_infos.get("tilt_violations", 0))
        env.numerical_failures = int(resume_infos.get("numerical_failures", 0))
    else:
        if args.rolling_prior_init:
            warm = rolling_prior_warm_start(env, runner, args.bc_steps, args.bc_epochs)
        elif args.high_step_expert_init:
            warm = high_step_expert_warm_start(env, runner, args.bc_steps, args.bc_epochs)
        elif args.initialize_actor:
            initialized = initialize_actor(env, runner, args.initialize_actor)
            if args.post_init_bc_steps > 0 and args.post_init_bc_epochs > 0:
                warm = {"initialization": initialized,
                        "post_initialize_task_space_bc": imitation_warm_start(
                            env, runner, args.post_init_bc_steps, args.post_init_bc_epochs)}
            else:
                warm = initialized
        else:
            warm = (learned_teacher_warm_start(env, runner, args.teacher_actor, args.bc_steps, args.bc_epochs)
                    if args.teacher_actor else imitation_warm_start(env, runner, args.bc_steps, args.bc_epochs))
    (args.output / "warm_start.json").write_text(json.dumps(warm, indent=2) + "\n")
    runner.export_policy_to_onnx(str(args.output), "warm_start.onnx")
    try:
        if args.checkpoint_every <= 0:
            raise ValueError("--checkpoint-every must be positive")
        completed_iterations = int(resume_infos.get("completed_iterations", 0)) if resume_infos else 0
        while completed_iterations < args.iterations:
            chunk = min(args.checkpoint_every, args.iterations - completed_iterations)
            runner.learn(chunk)
            # rsl_rl records the final zero-based iteration, whereas a second
            # learn() call expects the next iteration as its starting point.
            runner.current_learning_iteration += 1
            completed_iterations += chunk
            stem = f"checkpoint-{completed_iterations:04d}"
            runner.export_policy_to_onnx(str(args.output), stem + ".onnx")
            runner.save(str(args.output / (stem + ".pt")), infos={
                "completed_iterations": completed_iterations,
                "success_events": env.successes,
                "falls": env.falls,
                "cargo_losses": env.cargo_losses,
                "safety_violations": env.safety_violations,
                "joint_violations": env.joint_violations,
                "tilt_violations": env.tilt_violations,
                "numerical_failures": env.numerical_failures,
                "last_numerical_diagnostic": env.last_numerical_diagnostic,
            })
            progress = {
                "completed_iterations": completed_iterations,
                "requested_iterations": args.iterations,
                "elapsed_s": time.time() - started,
                "success_events": env.successes,
                "falls": env.falls,
                "cargo_losses": env.cargo_losses,
                "safety_violations": env.safety_violations,
                "joint_violations": env.joint_violations,
                "tilt_violations": env.tilt_violations,
                "numerical_failures": env.numerical_failures,
                "last_numerical_diagnostic": env.last_numerical_diagnostic,
                "curriculum_max_rise_mm": float(env.rises_cpu[env.lane_pool_cpu[env.curriculum_count - 1]] * 1000),
                "latest_checkpoint": stem + ".onnx",
            }
            (args.output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps({"training_progress": progress}), flush=True)
        runner.export_policy_to_onnx(str(args.output), "final.onnx")
        actor_path = args.output / "final.onnx"
        # torch 2.9 defaults to external tensor data even for this small model.
        # Godot loads ONNX from bytes, so deployment requires one self-contained
        # protobuf with no filesystem-relative sidecar.
        exported = onnx.load(str(actor_path), load_external_data=True)
        onnx.save_model(exported, str(actor_path), save_as_external_data=False)
        sidecar = actor_path.with_name(actor_path.name + ".data")
        if sidecar.exists():
            sidecar.unlink()
        manifest = {
            "schema_version": 2, "id": "sai-task-space-skills-v6", "actor": "final.onnx",
            "onnx_sha256": hashlib.sha256(actor_path.read_bytes()).hexdigest(),
            "observation_size": OBSERVATION_SIZE, "action_size": ACTION_SIZE,
            "contract": "sai-task-space-skills-v6", "algorithm": "hierarchical task-space residual PPO over a frozen rolling prior and active suspension",
            "terrain_observation_contract": "godot-oracle-terrain",
            "deployment_scope": "simulation-only",
            "payload_clamp_target_rad": 3.508111796508603 if args.payload_clamp else 0.0,
            "training": {"envs": args.envs, "iterations": args.iterations, "steps": args.steps,
                         "checkpoint_every": args.checkpoint_every,
                         "seed": args.seed, "min_rise_mm": args.min_rise_mm, "warm_start": warm,
                         "post_init_bc_steps": args.post_init_bc_steps,
                         "post_init_bc_epochs": args.post_init_bc_epochs,
                         "terrain_stage": args.terrain_stage,
                         "isolated_step": args.isolated_step,
                         "fixed_curriculum": args.fixed_curriculum,
                         "fixed_goal_edges": args.fixed_goal_edges,
                         "init_std": args.init_std, "learning_rate": args.learning_rate,
                         "teacher_weight_low": args.teacher_weight_low,
                         "teacher_weight_high": args.teacher_weight_high,
                         "cargo_accel_weight": args.cargo_accel_weight,
                         "action_rate_weight": args.action_rate_weight,
                         "reference_weight": args.reference_weight,
                         "lift_event_weight": args.lift_event_weight,
                         "position_objective": args.position_objective,
                         "position_reward_scale": args.position_reward_scale,
                         "payload_clamp": args.payload_clamp,
                         "domain_randomization": args.domain_randomization,
                         "joint_limit_weight": args.joint_limit_weight,
                         "joint_speed_weight": args.joint_speed_weight,
                         "upright_weight": args.upright_weight,
                         "support_weight": args.support_weight},
            "control": {"speed": .16, "wheel_residual_scale": 6.0, "yaw_correction_limit": .4},
        }
        (args.output / "profile.json").write_text(json.dumps(manifest, indent=2) + "\n")
        completed = {"status": "trained_not_yet_accepted", "elapsed_s": time.time() - started,
                     "success_events": env.successes, "falls": env.falls, "cargo_losses": env.cargo_losses,
                     "safety_violations": env.safety_violations,
                     "joint_violations": env.joint_violations,
                     "tilt_violations": env.tilt_violations,
                     "numerical_failures": env.numerical_failures,
                     "last_numerical_diagnostic": env.last_numerical_diagnostic,
                     "curriculum_max_rise_mm": float(env.rises_cpu[env.lane_pool_cpu[env.curriculum_count - 1]] * 1000),
                     "onnx_sha256": manifest["onnx_sha256"]}
        (args.output / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")
        print(json.dumps(completed), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
