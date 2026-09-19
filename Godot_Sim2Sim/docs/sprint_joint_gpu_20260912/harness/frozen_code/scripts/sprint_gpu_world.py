"""GPU training proxy from the frozen game's source asset, with explicit limits.

The Jolt motor equation and diagonal rotor map are matched. MuJoCo/Warp is
still a different constraint/contact solver. This module never edits Godot.
"""
from pathlib import Path
import json
import hashlib
import re
import gc

import mujoco
import mujoco_warp as mw
import numpy as np
import torch
import warp as wp

from sim2sim.paths import load_robot_json
from sim2sim.research.torch_walking import WalkingControl, rotate, inverse_rotate
from sprint_gpu_proxy import patch_game_inertia
from sim2sim.coords import quat_wxyz_to_mat


def patch_game_feet(spec, cfg):
    """Use the actual exported Jolt foot hulls, in their principal-axis frame.

    Body-local Godot coordinates are NOT world-axis converted. The converter
    aligns each rigid body's local axes with the MuJoCo inertial principal axes.
    Reject unsupported transforms rather than guessing a different geometry.
    """
    path = Path(cfg['godot_spec']).with_name('robot.tscn')
    raw = path.read_bytes()
    blocks = re.split(r'\n(?=\[)', raw.decode())
    bodies = {x['name']: x for x in json.loads(Path(cfg['godot_spec']).read_text())['bodies']}
    record = {}
    for geom_name, body_name in [('left_foot_collision', 'ankle_left'), ('right_foot_collision', 'ankle_right')]:
        nodes = [x for x in blocks if x.startswith('[node ') and f'name="col_{geom_name}_' in x.splitlines()[0]]
        if len(nodes) != 1 or f'parent="{body_name}"' not in nodes[0].splitlines()[0]:
            raise ValueError('Expected a single frozen foot collider')
        node = nodes[0]
        if 'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)' not in node:
            raise ValueError('Unsupported foot-local transform')
        shape = re.search(r'shape = SubResource\("([^"]+)"\)', node).group(1)
        resource = next(x for x in blocks if x.startswith('[sub_resource ') and f'id="{shape}"' in x.splitlines()[0])
        if 'type="ConvexPolygonShape3D"' not in resource or 'margin = 0.0' not in resource:
            raise ValueError('Unsupported foot shape or collision margin')
        points = np.fromstring(re.search(r'points = PackedVector3Array\(([^)]+)\)', resource).group(1), sep=',').reshape(-1, 3)
        body = bodies[body_name]
        vertices = points @ quat_wxyz_to_mat(body['iquat_wxyz']).T + np.array(body['ipos'])
        if not np.isfinite(vertices).all():
            raise ValueError('Nonfinite frozen foot geometry')
        name = 'jolt_' + geom_name
        filename = name + '.obj'
        mesh = spec.add_mesh()
        mesh.name = name
        mesh.file = filename
        spec.assets = {**spec.assets, filename: '\n'.join('v ' + ' '.join(format(v, '.17g') for v in row) for row in vertices)}
        geom = spec.geom(geom_name)
        geom.meshname = name
        geom.pos = [0., 0., 0.]
        geom.quat = [1., 0., 0., 0.]
        record[geom_name] = dict(body=body_name,vertices_body=vertices.tolist(),count=len(vertices))
    return dict(scene_sha256=hashlib.sha256(raw).hexdigest(),feet=record)


class GpuWorld:
    def __init__(self, count, settings, *, contact='source', inertia=True, feet='source', graphs=False):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA physics required; CPU fallback is disabled')
        if contact not in ('source', 'two_tick'):
            raise ValueError('Unknown contact proxy')
        self.count = count
        self.cfg = load_robot_json(Path('robots/microduck_ball_stand_fix.json'))
        spec = mujoco.MjSpec.from_file(self.cfg['mjcf'])
        if feet not in ('source', 'jolt'):
            raise ValueError('Unknown foot geometry proxy')
        self.feet = patch_game_feet(spec, self.cfg) if feet == 'jolt' else None
        limit = 1.75 * .36601349688984386
        self.limit = limit
        for actuator in spec.actuators:
            joint = spec.joint(str(actuator.target))
            joint.armature = .0018
            joint.damping = np.zeros((3, 1))
            joint.frictionloss = 0.
            actuator.set_to_motor()
            actuator.gear = [1., 0., 0., 0., 0., 0.]
            actuator.forcelimited = False
            actuator.ctrllimited = False
        if contact == 'two_tick':
            for geom in spec.geoms:
                if geom.contype or geom.conaffinity:
                    geom.solref = [.01, 1.]
        self.model = spec.compile()
        self.model.opt.timestep = .005
        self.inertia = patch_game_inertia(self.model, self.cfg['godot_spec']) if inertia else None
        m = self.model
        if self.feet:
            for name, entry in self.feet['feet'].items():
                gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
                mesh = m.geom_dataid[gid]
                start = m.mesh_vertadr[mesh]
                local = m.mesh_vert[start:start + m.mesh_vertnum[mesh]]
                actual = local @ quat_wxyz_to_mat(m.geom_quat[gid]).T + m.geom_pos[gid]
                expected = np.array(entry['vertices_body'])
                error = np.linalg.norm(expected[:, None] - actual[None], axis=2).min(axis=1).max()
                if error > 1e-7:
                    raise ValueError('Compiled foot geometry moved')
                entry['compiled_vertex_max_error_m'] = float(error)
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'trunk_base')
        joint = next(i for i in range(m.njnt) if m.jnt_bodyid[i] == self.base and m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
        self.qa = int(m.jnt_qposadr[joint])
        self.va = int(m.jnt_dofadr[joint])
        self.qi = torch.tensor([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)], device='cuda', dtype=torch.long)
        self.vi = torch.tensor([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)], device='cuda', dtype=torch.long)
        self.home = torch.tensor(self.cfg['home'], device='cuda', dtype=torch.float32)
        self.ipos = torch.tensor(m.body_ipos[self.base], device='cuda', dtype=torch.float32)
        initial = mujoco.MjData(m)
        initial.qpos[self.qa:self.qa + 7] = [0., 0., self.cfg['reset_z'], 1., 0., 0., 0.]
        initial.qpos[self.qi.cpu().numpy()] = self.cfg['home']
        for i in range(m.njnt):
            if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and i != joint:
                p = int(m.jnt_qposadr[i])
                initial.qpos[p:p + 7] = [5., 5., .035, 1., 0., 0., 0.]
        mujoco.mj_forward(m, initial)
        self.initial_qpos = torch.tensor(initial.qpos, device='cuda', dtype=torch.float32)
        wp.init()
        # Graph capture requires a Warp-owned non-default stream. Bridge it
        # explicitly to Torch instead of trying to capture legacy stream 0.
        self.stream = wp.Stream('cuda:0')
        self.torch_stream = wp.stream_to_torch(self.stream)
        with wp.ScopedStream(self.stream):
            self.wm = mw.put_model(m)
            self.wd = mw.put_data(m, initial, nworld=count, nconmax=128, njmax=2048)
            self.qpos = wp.to_torch(self.wd.qpos)
            self.qvel = wp.to_torch(self.wd.qvel)
            self.ctrl = wp.to_torch(self.wd.ctrl)
            mw.forward(self.wm, self.wd)
        torch.cuda.current_stream().wait_stream(self.torch_stream)
        self.step_graph = self.forward_graph = None
        if graphs:
            # Same graph mechanism used by mjlab.sim.Simulation. Keep every
            # captured array allocated; reset changes values, never pointers.
            if not wp.is_mempool_enabled(self.stream.device):
                raise RuntimeError('CUDA graph capture requires the Warp memory pool')
            enabled = gc.isenabled()
            gc.disable()
            try:
                with wp.ScopedStream(self.stream):
                    with wp.ScopedCapture(stream=self.stream) as capture:
                        mw.step(self.wm, self.wd)
                    self.step_graph = capture.graph
                    with wp.ScopedCapture(stream=self.stream) as capture:
                        mw.forward(self.wm, self.wd)
                    self.forward_graph = capture.graph
            finally:
                if enabled:
                    gc.enable()
        self.last = torch.zeros((count, 14), device='cuda')
        self.control = WalkingControl(count, settings, 'cuda')
        self.reset(torch.arange(count, device='cuda'))

    def reset(self, ids, perturbation=None):
        self.qpos[ids] = self.initial_qpos
        self.qvel[ids] = 0.
        self.ctrl[ids] = 0.
        self.last[ids] = 0.
        if perturbation is not None:
            self.qpos[ids[:, None], self.qi[None, :]] += perturbation
        # No previous solver acceleration should cross an episode boundary.
        wp.to_torch(self.wd.qacc_warmstart)[ids] = 0.
        self.control.reset(ids)
        self.forward()

    def forward(self):
        caller = torch.cuda.current_stream()
        self.torch_stream.wait_stream(caller)
        with wp.ScopedStream(self.stream):
            if self.forward_graph is None:
                mw.forward(self.wm, self.wd)
            else:
                wp.capture_launch(self.forward_graph)
        caller.wait_stream(self.torch_stream)

    def state(self):
        pos = self.qpos[:, self.qa:self.qa + 3]
        quat = self.qpos[:, self.qa + 3:self.qa + 7]
        angular = self.qvel[:, self.va + 3:self.va + 6]
        velocity = self.qvel[:, self.va:self.va + 3] + torch.cross(rotate(quat, angular), rotate(quat, self.ipos.expand(self.count, -1)), dim=-1)
        return pos, quat, angular, velocity

    def observe(self, requested, sprint):
        pos, quat, angular, velocity = self.state()
        command = self.control.command(requested, pos, quat, velocity, sprint)
        gravity = torch.zeros_like(pos)
        gravity[:, 2] = -1.
        obs = torch.cat((angular, inverse_rotate(quat, gravity), self.qpos[:, self.qi] - self.home,
                         self.qvel[:, self.vi], self.last, command), dim=1)
        return obs, command

    def step(self, action):
        caller = torch.cuda.current_stream()
        self.torch_stream.wait_stream(caller)
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            target = self.home + action
            for _ in range(4):
                v = self.qvel[:, self.vi]
                self.ctrl.copy_((.55 * (target - self.qpos[:, self.qi])).clamp(-self.limit, self.limit) - .053 * v - .0048 * torch.tanh(v / .05))
                if self.step_graph is None:
                    mw.step(self.wm, self.wd)
                else:
                    wp.capture_launch(self.step_graph)
            self.last.copy_(action)
        caller.wait_stream(self.torch_stream)

    def close(self):
        torch.cuda.synchronize()
        self.step_graph = self.forward_graph = None
        self.qpos = self.qvel = self.ctrl = None
        self.wm = self.wd = None
        torch.cuda.empty_cache()
