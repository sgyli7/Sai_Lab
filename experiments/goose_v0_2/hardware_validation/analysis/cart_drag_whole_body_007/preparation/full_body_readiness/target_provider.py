"""OFFLINE handoff module. Import/use does not integrate physics or write live qpos.
Next separately authorized executor calls provider.action(env,t,rod_world) then
ONLY env.step(action). The returned root reference is a measurement target only.
"""
from pathlib import Path
import json
import numpy as np

class DoubleStanceBodyBackTeacher:
    def __init__(self,config=Path(__file__).with_name('next_trial_input.json'),*,loaded=False,static_bias=False):
        self.config=json.loads(Path(config).read_text());self.frames=self.config['times_and_targets'];self.loaded=loaded;self.static_bias=static_bias;self.scratch=None;self.last_neck=None
    def interpolate(self,t):
        # One second actual free-root settling/grip preload precedes the existing10s program.
        t=max(0.,float(t)-1.);f=self.frames
        a,b=f[-1],f[-1]
        for aa,bb in zip(f,f[1:]):
            if t<=bb['time_s']:a,b=aa,bb;break
        dt=b['time_s']-a['time_s'];u=0. if dt<=0 else float(np.clip((t-a['time_s'])/dt,0,1));s=u*u*(3-2*u)
        interp=lambda key:(1-s)*np.asarray(a[key],float)+s*np.asarray(b[key],float)
        return interp('q_target18_rad'),interp('tau_static_0N18_Nm'),interp('expected_root_xyz_offline_reference_only'),interp('expected_TCP_world_reference_only')
    def action(self,env,t,rod_world=None):
        import mujoco as mj
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        q,tau,root_reference,tcp_reference=self.interpolate(t)
        target=np.asarray(tcp_reference if rod_world is None else rod_world,float)
        if self.scratch is None:self.scratch=mj.MjData(env.model)
        m=env.model;d=self.scratch;d.qpos[:]=env.data.qpos;d.qvel[:]=env.data.qvel
        idx=env.qidx[:5];bounds=env.ranges[:5];head=m.body('head_roll').id;local=np.array([.13750001972253326,0,-.03849996489687674])
        def f(z):
            d.qpos[idx]=z;mj.mj_kinematics(m,d);H=d.xmat[head].reshape(3,3);e=Rotation.from_matrix(H).as_euler('ZYX');return np.r_[d.xpos[head]+H@local-target,.08*(e[1]-.174532925),.08*e[2]]
        seed=env.data.qpos[idx] if self.last_neck is None else self.last_neck
        sol=least_squares(f,np.clip(seed,bounds[:,0]+.10001,bounds[:,1]-.10001),bounds=(bounds[:,0]+.1,bounds[:,1]-.1),max_nfev=35,ftol=1e-8,xtol=1e-8,gtol=1e-8);self.last_neck=sol.x.copy();q[:5]=sol.x
        if self.static_bias:q+=tau/env.kp
        # Loaded birth is real cold contact only. Jaw compression MUST develop via original slew/drive.
        # Initial1s remains body-neutral; do not overwrite env.target to pre-load.
        if self.loaded:q[5]=0.
        q=np.clip(q,env.ranges[:,0],env.ranges[:,1]);action=(q-env.neutral)/env.scale
        return action,{'requested_joint_targets18':q.tolist(),'expected_root_reference_diagnostics_only':root_reference.tolist(),'tcp_target_world':target.tolist(),'scratch_IK_error_equivalent_m':float(np.linalg.norm(f(sol.x))),'static_bias_declared':self.static_bias,'loaded_birth_clamp_only_not_active_grasp':self.loaded,'no_live_state_write':True}
