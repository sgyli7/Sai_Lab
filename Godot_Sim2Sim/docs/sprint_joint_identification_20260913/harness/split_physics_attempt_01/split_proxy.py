"""Experimental generalized-coordinate split position correction, GPU only.

This approximates Jolt's separation of momentum and position stabilization. It
is not a port of Jolt's ordered maximal-coordinate constraint/contact solver.
"""
from pathlib import Path
import sys
import numpy as np
import torch
import warp as wp
import mujoco
import mujoco_warp as mw
from mujoco_warp._src.forward import _next_position
sys.path.insert(0,str(Path('scripts').resolve()))
from sprint_gpu_world import GpuWorld


class SplitWorld(GpuWorld):
    def prepare_projection(self, alpha):
        self.projection_alpha=float(alpha)
        self.slide_q=torch.tensor([self.model.jnt_qposadr[i] for i in range(self.model.njnt) if self.model.jnt_type[i]==mujoco.mjtJoint.mjJNT_SLIDE],device='cuda')
        self.slide_v=torch.tensor([self.model.jnt_dofadr[i] for i in range(self.model.njnt) if self.model.jnt_type[i]==mujoco.mjtJoint.mjJNT_SLIDE],device='cuda')
        assert len(self.slide_q)==42 and self.wm.is_sparse
        rows=[];cols=[];indices=[]
        for i in range(self.model.nv):
            j=i;k=int(self.model.dof_Madr[i])
            while j>=0:
                rows.append(i);cols.append(j);indices.append(k)
                j=int(self.model.dof_parentid[j]);k+=1
        self.mrows=torch.tensor(rows,device='cuda');self.mcols=torch.tensor(cols,device='cuda');self.mindices=torch.tensor(indices,device='cuda')
        self.selector=torch.eye(self.model.nv,device='cuda')[:,self.slide_v].expand(self.count,-1,-1).contiguous()
        self.mass=torch.zeros((self.count,self.model.nv,self.model.nv),device='cuda')
        self.delta=torch.zeros_like(self.qvel);self.next_q=torch.zeros_like(self.qpos)
        self.delta_wp=wp.from_torch(self.delta);self.next_q_wp=wp.from_torch(self.next_q)
        self.solve_info=torch.zeros((),device='cuda',dtype=torch.int32)

    def mass_matrix(self):
        sparse=wp.to_torch(self.wd.qM)[:,0]
        self.mass.zero_()
        values=sparse[:,self.mindices]
        self.mass[:,self.mrows,self.mcols]=values
        self.mass[:,self.mcols,self.mrows]=values
        return self.mass

    def project(self):
        if self.projection_alpha==0:return
        mass=self.mass_matrix()
        mobility,info1=torch.linalg.solve_ex(mass,self.selector)
        effective=mobility[:,self.slide_v]
        impulse,info2=torch.linalg.solve_ex(effective,-self.projection_alpha*self.qpos[:,self.slide_q,None])
        self.solve_info.add_((info1!=0).sum().to(torch.int32)+(info2!=0).sum().to(torch.int32))
        self.delta.copy_(torch.bmm(mobility,impulse).squeeze(-1)/.005)
        # Reuse the installed backend's quaternion-aware integration primitive;
        # integrate only into a separate pose array, without writing qvel.
        wp.launch(_next_position,dim=(self.count,self.model.njnt),inputs=[self.wm.opt.timestep,self.wm.jnt_type,self.wm.jnt_qposadr,self.wm.jnt_dofadr,self.wd.qpos,self.delta_wp,1.],outputs=[self.next_q_wp])
        self.qpos.copy_(self.next_q)

    def step(self, action):
        caller=torch.cuda.current_stream();self.torch_stream.wait_stream(caller)
        with torch.cuda.stream(self.torch_stream),wp.ScopedStream(self.stream):
            target=self.home+action
            for _ in range(4):
                v=self.qvel[:,self.vi] if self.observer is None else self.observer.qd
                self.ctrl.copy_((.55*(target-self.qpos[:,self.qi])).clamp(-self.limit,self.limit)-.053*v-.0048*torch.tanh(v/.05))
                if self.step_graph is None:mw.step(self.wm,self.wd)
                else:wp.capture_launch(self.step_graph)
                self.project()
                if self.observer is not None:self.observer.update(self.qpos[:,self.qi],self.qpos[:,self.qa+3:self.qa+7])
            self.last.copy_(action)
        caller.wait_stream(self.torch_stream)

    def canary(self):
        # Static matrix assembly check against independent CPU MuJoCo, then
        # validate correction, quaternion integration and untouched qvel.
        old_q=self.qpos.clone();old_v=self.qvel.clone()
        with torch.cuda.stream(self.torch_stream),wp.ScopedStream(self.stream):
            self.torch_stream.wait_stream(torch.cuda.default_stream())
            matrix=self.mass_matrix().clone()
        torch.cuda.current_stream().wait_stream(self.torch_stream)
        d=mujoco.MjData(self.model);d.qpos[:]=self.qpos[0].cpu().numpy();mujoco.mj_forward(self.model,d)
        expected=np.empty((self.model.nv,self.model.nv));mujoco.mj_fullM(self.model,d,expected)
        mass_error=float(np.max(np.abs(matrix[0].cpu().numpy()-expected)))
        assert mass_error<1e-6,mass_error
        with torch.cuda.stream(self.torch_stream),wp.ScopedStream(self.stream):
            self.qpos[:,self.slide_q]=torch.linspace(-.0001,.0001,42,device='cuda')
            displaced=self.qpos.clone();self.project();delta=self.delta.clone()
        torch.cuda.current_stream().wait_stream(self.torch_stream)
        reference=displaced[0].cpu().numpy().astype(np.float64);mujoco.mj_integratePos(self.model,reference,delta[0].cpu().numpy().astype(np.float64),.005)
        integration_error=float(np.max(np.abs(self.qpos[0].cpu().numpy()-reference)))
        slide_error=float((self.qpos[:,self.slide_q]-(1-self.projection_alpha)*displaced[:,self.slide_q]).abs().max())
        velocity_error=float((self.qvel-old_v).abs().max())
        assert integration_error<1e-7 and slide_error<1e-8 and velocity_error==0.,(integration_error,slide_error,velocity_error)
        self.qpos.copy_(old_q);self.qvel.copy_(old_v);self.forward()
        return dict(mass_max_abs=mass_error,integration_max_abs=integration_error,slide_residual_max_abs=slide_error,qvel_max_abs=velocity_error)
