#!/usr/bin/env python3
"""Train a phase-free Sai stair actor with PPO in MuJoCo-Warp.

The policy owns all twelve leg targets and four wheel-speed residuals.  A
per-wheel event teacher is used only to initialize the network; PPO then
optimizes the same deployable observation/action contract without a gait clock.
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
from sim2sim.sai_stair_v5 import ACTION_SIZE, OBSERVATION_SIZE


ROOT = Path(__file__).resolve().parents[1]
LEG = torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14], device="cuda")
WHEEL = torch.tensor([3, 7, 11, 15], device="cuda")
SIDES = torch.tensor([1., -1., 1., -1.], device="cuda")
FRONTS = torch.tensor([1., 1., -1., -1.], device="cuda")
JOINT_ACTION_SCALES = torch.tensor([.30, .60, 1.02, 1.] * 4, device="cuda")
TARGET_SLEW_RAD_S = torch.tensor([3., 4., 6., 20.] * 4, device="cuda")
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
                 teacher_weight_low: float = .70, teacher_weight_high: float = .015, cargo_accel_weight: float = .012,
                 action_rate_weight: float = .05, reference_weight: float = 0.,
                 lift_event_weight: float = 0., position_objective: bool = False,
                 payload_clamp: bool = False, position_reward_scale: float = 400.,
                 domain_randomization: bool = False, joint_limit_weight: float = 12.,
                 joint_speed_weight: float = .01, upright_weight: float = 12.,
                 support_weight: float = 3.):
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
                stairs = np.floor(np.clip((self.terrain_x_cpu - start) / tread, 0., 4.)) * rise
                self.course_heights_cpu[lane] = np.where(self.terrain_x_cpu < start, rough_height, stairs)
        lane_pool = np.flatnonzero(self.rises_cpu * 1000 >= min_rise_mm - 1e-6)
        if not len(lane_pool):
            raise ValueError("Minimum rise excludes every training lane")
        self.lane_pool_cpu = lane_pool
        self.lane_pool = torch.tensor(lane_pool, device="cuda", dtype=torch.long)
        self.lane_centers = torch.tensor(self.lane_centers_cpu, device="cuda")
        self.rises = torch.tensor(self.rises_cpu, device="cuda")
        self.treads = torch.tensor(self.treads_cpu, device="cuda")
        self.starts = torch.tensor(self.starts_cpu, device="cuda")
        self.course_heights = torch.tensor(self.course_heights_cpu, device="cuda")
        self.curriculum_count = 4
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
        self.action_history = torch.zeros((count, 3, 16), device="cuda")
        self.action_delay = torch.zeros(count, device="cuda", dtype=torch.long)
        self.motor_strength = torch.ones((count, 1), device="cuda")
        self.last_payload_velocity = torch.zeros((count, 3), device="cuda")
        self.filtered_cargo_accel = torch.zeros((count, 3), device="cuda")
        self.previous_airborne = torch.zeros((count, 4), device="cuda", dtype=torch.bool)
        self.lane = torch.zeros(count, device="cuda", dtype=torch.long)
        self.goal_edges = torch.ones(count, device="cuda", dtype=torch.long)
        self.start_x = torch.full((count,), -.68, device="cuda")
        self.command = torch.zeros((count, 3), device="cuda")
        self.command[:, 0] = .16
        self.last_goal_distance = torch.zeros(count, device="cuda")
        self.successes = self.falls = self.cargo_losses = 0
        self._last_metrics = {}
        self.reset(torch.arange(count, device="cuda"))

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
            for level in range(1, 5):
                left = start + (level - 1) * tread
                right = start + level * tread if level < 4 else 1.4
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
                      contype="4", conaffinity="3", group="4")
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
        lane = self.lane_pool[torch.randint(len(self.lane_pool), (len(ids),), generator=self.generator, device="cuda")]
        self.lane[ids] = lane
        self.goal_edges[ids] = torch.randint(1, self.curriculum_edges + 1, (len(ids),),
                                             generator=self.generator, device="cuda")
        self.qpos[ids] = self.initial_qpos
        self.qvel[ids] = 0.
        self.ctrl[ids] = 0.
        self.last_action[ids] = 0.
        self.last_target[ids] = 0.
        self.action_history[ids] = 0.
        if self.domain_randomization:
            self.action_delay[ids] = torch.randint(3, (len(ids),), generator=self.generator, device="cuda")
            self.motor_strength[ids] = .85 + .30 * torch.rand((len(ids), 1), generator=self.generator, device="cuda")
        else:
            self.action_delay[ids] = 0.;self.motor_strength[ids] = 1.
        self.filtered_cargo_accel[ids] = 0.
        self.previous_airborne[ids] = False
        self.episode_length_buf[ids] = 0
        x = self.start_x[ids] + (torch.rand(len(ids), generator=self.generator, device="cuda") - .5) * .05
        y = self.lane_centers[lane] + (torch.rand(len(ids), generator=self.generator, device="cuda") - .5) * .025
        yaw = (torch.rand(len(ids), generator=self.generator, device="cuda") - .5) * .08
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
        self.last_payload_velocity[ids] = self.qvel[ids, self.payload_va:self.payload_va + 3]
        goal_x = self.starts[lane] + (self.goal_edges[ids] - 1) * self.treads[lane] + .18
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

    def _features(self) -> torch.Tensor:
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
            ((dense - local_ground) * 5.).clamp(-2., 2.)), dim=1)
        if obs.shape[1] != OBSERVATION_SIZE:
            raise RuntimeError(f"Observation contract drift: {obs.shape}")
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
        goal_x = self.starts[self.lane] + (self.goal_edges - 1) * self.treads[self.lane] + .18
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
        """Per-wheel contact-distance teacher used for initialization only."""
        wheel_x = wp.to_torch(self.wd.xpos)[:, self.wheel_bodies, 0]
        distance, valid = self._next_edge(wheel_x)
        active = valid & (distance <= .09) & (distance >= -.06)
        phase = ((.09 - distance) / .15).clamp(0., 1.)
        rise = self.rises[self.lane, None]
        lift = (rise + .025).clamp(max=.075) * 4 * phase * (1 - phase)
        dx = -.025 + .065 * phase
        down = .172812737 - lift
        numerator = down.square() + dx.square() - .09**2 - .11**2
        beta = -FRONTS[None] * torch.acos((numerator / (2 * .09 * .11)).clamp(-1., 1.))
        theta = torch.atan2(dx, down) - torch.atan2(.11 * torch.sin(beta), .09 + .11 * torch.cos(beta))
        theta0 = np.arctan2(.05, .074833147)
        beta0 = np.arctan2(.05, .09797959) + theta0
        hip = SIDES[None] * (FRONTS[None] * theta0 - theta)
        knee = SIDES[None] * (-FRONTS[None] * beta0 - beta)
        action = torch.zeros((self.num_envs, 16), device="cuda")
        for leg in range(4):
            action[:, leg * 4 + 1] = torch.where(active[:, leg], hip[:, leg] / .60, 0.)
            action[:, leg * 4 + 2] = torch.where(active[:, leg], knee[:, leg] / 1.02, 0.)
        return action.clamp(-1., 1.)

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
        self.action_history.copy_(torch.roll(self.action_history, 1, dims=1))
        self.action_history[:, 0] = actions
        actions = self.action_history[torch.arange(self.num_envs, device="cuda"), self.action_delay]
        previous = self.last_action.clone()
        teacher = self.teacher_action()
        reference = self.reference_action(self._features())
        q = self.qpos[:, self.qi]
        desired_target = actions * JOINT_ACTION_SCALES
        desired_target[:, WHEEL] = SIDES[None] * (self.command[:, :1] / .048 + 6. * actions[:, WHEEL])
        target_delta = (desired_target - self.last_target).clamp(
            min=-TARGET_SLEW_RAD_S[None] * .02, max=TARGET_SLEW_RAD_S[None] * .02)
        target = self.last_target + target_delta
        wheel_target = target[:, WHEEL]
        torque = torch.zeros((self.num_envs, self.model.nu), device="cuda")
        caller = torch.cuda.current_stream()
        self.torch_stream.wait_stream(caller)
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            for _ in range(10):
                q, v = self.qpos[:, self.qi], self.qvel[:, self.vi]
                torque[:, :16] = (80. * (target - q) - 2. * v).clamp(-8., 8.)
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
        self.last_action.copy_(actions)
        self.last_target.copy_(target)
        self.episode_length_buf += 1
        self.total_steps += 1
        # All rises remain represented throughout PPO.  The curriculum grows
        # route length only, so mastering 60 mm cannot erase smooth 20 mm use.
        self.curriculum_count = len(self.rises_cpu)
        self.curriculum_edges = min(4, 1 + self.total_steps // 960)

        pos, quat, angular, linear = self._state()
        body_velocity = inverse_rotate(quat, linear)
        up = 1. - 2. * (quat[:, 1].square() + quat[:, 2].square())
        wheel_xyz = wp.to_torch(self.wd.xpos)[:, self.wheel_bodies]
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
        relative_payload = wp.to_torch(self.wd.xpos)[:, self.payload_body] - wp.to_torch(self.wd.xpos)[:, self.chassis]
        cargo_lost = (relative_payload[:, 2] < .02) | (relative_payload[:, 1].abs() > .12) | (relative_payload[:, 0] > -.025) | (relative_payload[:, 0] < -.17)
        # Early training terminates after one independently solved edge.  As
        # learning proceeds, the same policy must chain up to all four edges.
        finish = self.starts[self.lane] + (self.goal_edges - 1) * self.treads[self.lane] + .18
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
        unsafe_tilt = up < float(np.cos(np.deg2rad(20.)))
        fall = (up < .55) | ((pos[:, 2] - base_ground) < .105) | ~torch.isfinite(pos).all(-1)
        safety_violation = unsafe_joint | unsafe_tilt
        timeout = self.episode_length_buf >= self.max_episode_length
        done = fall | safety_violation | cargo_lost | success | timeout
        no_edge_near = ~near.any(-1)
        rise = self.rises[self.lane]
        teacher_weight = torch.where(rise <= .030, self.teacher_weight_low, self.teacher_weight_high)
        teacher_error = (actions - teacher).square().sum(-1)
        reference_error = (actions - reference).square().sum(-1)
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
            - .025 * actions[:, LEG].square().sum(-1)
            - .20 * actions[:, LEG].square().sum(-1) * no_edge_near
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
        self.successes += int(success.sum())
        self.falls += int(fall.sum())
        self.cargo_losses += int(cargo_lost.sum())
        self._last_metrics = {
            "/curriculum/max_rise_mm": float(self.rises_cpu[self.lane_pool_cpu[-1]] * 1000),
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
            "/motion/lateral_error": lateral_error.abs().mean(),
            "/outcome/safety_violation": safety_violation.float().mean(),
            "/outcome/success": success.float().mean(),
            "/outcome/fall": fall.float().mean(),
        }
        ids = done.nonzero(as_tuple=False).flatten()
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
    # The initialization prior must see every deployable rise.  PPO may refine
    # high steps later, but it should not discover 45--60 mm behavior by first
    # destroying the smooth low-step solution.
    env.curriculum_count = len(env.rises_cpu)
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
    env.curriculum_count = len(env.rises_cpu)
    env.curriculum_edges = 1
    env.reset(torch.arange(env.num_envs, device="cuda"))
    return {"samples": len(x), "epochs": epochs, "final_mse": final, "held_stride_mae": mae,
            "teacher_scope": "network initialization only; PPO retains full action authority"}


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
    env.total_steps = 2880;env.curriculum_count = len(env.rises_cpu);env.curriculum_edges = 4
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
    env.total_steps = 2880
    env.curriculum_count = len(env.rises_cpu)
    env.curriculum_edges = 4
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
    parser.add_argument("--initialize-actor", type=Path)
    parser.add_argument("--teacher-actor", type=Path)
    parser.add_argument("--init-std", type=float, default=.12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--teacher-weight-low", type=float, default=.70)
    parser.add_argument("--teacher-weight-high", type=float, default=.015)
    parser.add_argument("--cargo-accel-weight", type=float, default=.012)
    parser.add_argument("--action-rate-weight", type=float, default=.05)
    parser.add_argument("--reference-weight", type=float, default=0.)
    parser.add_argument("--lift-event-weight", type=float, default=0.)
    parser.add_argument("--position-objective", action="store_true")
    parser.add_argument("--payload-clamp", action="store_true")
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--joint-limit-weight", type=float, default=12.)
    parser.add_argument("--joint-speed-weight", type=float, default=.01)
    parser.add_argument("--upright-weight", type=float, default=12.)
    parser.add_argument("--support-weight", type=float, default=3.)
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
                        payload_clamp=args.payload_clamp,
                        domain_randomization=args.domain_randomization,
                        joint_limit_weight=args.joint_limit_weight,
                        joint_speed_weight=args.joint_speed_weight,
                        upright_weight=args.upright_weight,
                        support_weight=args.support_weight)
    config = train_config(args.steps, args.init_std, args.learning_rate)
    # Keep logging self-contained: the training host does not require optional
    # TensorBoard packages, and every chunk below emits a recoverable checkpoint.
    runner = OnPolicyRunner(env, config, log_dir=None, device="cuda")
    if args.initialize_actor and args.teacher_actor:
        raise ValueError("Choose either direct actor initialization or learned-teacher compression")
    warm = (initialize_actor(env, runner, args.initialize_actor) if args.initialize_actor else
            learned_teacher_warm_start(env, runner, args.teacher_actor, args.bc_steps, args.bc_epochs)
            if args.teacher_actor else imitation_warm_start(env, runner, args.bc_steps, args.bc_epochs))
    (args.output / "warm_start.json").write_text(json.dumps(warm, indent=2) + "\n")
    runner.export_policy_to_onnx(str(args.output), "warm_start.onnx")
    try:
        if args.checkpoint_every <= 0:
            raise ValueError("--checkpoint-every must be positive")
        completed_iterations = 0
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
            })
            progress = {
                "completed_iterations": completed_iterations,
                "requested_iterations": args.iterations,
                "elapsed_s": time.time() - started,
                "success_events": env.successes,
                "falls": env.falls,
                "cargo_losses": env.cargo_losses,
                "curriculum_max_rise_mm": float(env.rises_cpu[env.lane_pool_cpu[-1]] * 1000),
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
            "schema_version": 2, "id": "sai-stairs-safe-residual-v5", "actor": "final.onnx",
            "onnx_sha256": hashlib.sha256(actor_path.read_bytes()).hexdigest(),
            "observation_size": OBSERVATION_SIZE, "action_size": ACTION_SIZE,
            "contract": "sai-safe-residual-stairs-v5", "algorithm": "safety-bounded phase-free PPO with dense edge observations and asymmetric critic",
            "joint_action_scales": [.30, .60, 1.02],
            "target_slew_rad_s": [3.0, 4.0, 6.0],
            "training": {"envs": args.envs, "iterations": args.iterations, "steps": args.steps,
                         "checkpoint_every": args.checkpoint_every,
                         "seed": args.seed, "min_rise_mm": args.min_rise_mm, "warm_start": warm,
                         "init_std": args.init_std, "learning_rate": args.learning_rate,
                         "teacher_weight_low": args.teacher_weight_low,
                         "teacher_weight_high": args.teacher_weight_high,
                         "cargo_accel_weight": args.cargo_accel_weight,
                         "action_rate_weight": args.action_rate_weight,
                         "reference_weight": args.reference_weight,
                         "lift_event_weight": args.lift_event_weight,
                         "position_objective": args.position_objective,
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
                     "curriculum_max_rise_mm": float(env.rises_cpu[env.lane_pool_cpu[-1]] * 1000),
                     "onnx_sha256": manifest["onnx_sha256"]}
        (args.output / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")
        print(json.dumps(completed), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
