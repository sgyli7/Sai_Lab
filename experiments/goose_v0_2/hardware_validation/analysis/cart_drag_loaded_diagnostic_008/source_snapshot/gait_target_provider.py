"""008 alternating gait target interface: pure desired18 output, never live-state writes."""
from pathlib import Path
import json,importlib.util
import numpy as np
class AlternatingBackwardTeacher:
    def __init__(self,program,cold_rod_world,initial_q18):
        self.program=json.loads(Path(program).read_text());self.frames=self.program['frames'];last_tau=[0.]*18
        for frame in self.frames:
            if frame.get('static_tau18_Nm') is None:frame['static_tau18_Nm']=last_tau.copy();frame['bias_fallback']='previous_feasible_double_support_tau_held'
            else:last_tau=list(frame['static_tau18_Nm'])
        self.cold=np.asarray(cold_rod_world,float).copy();self.initial=np.asarray(initial_q18,float).copy();self.scratch=None;self.last_neck=None
        self.period=float(self.program['period_s']);self.cycle_back=float(self.program['cycle_back_m']);self.cycles=int(np.ceil(.55/self.cycle_back));self.warmup=float(self.program.get('warmup_s',2.));self.walk_time=self.period*self.cycles;self.hold_s=3.;self.duration=self.warmup+self.walk_time+self.hold_s
    def phase(self,t):
        if t<self.warmup:return {'u':0.,'cycle':0,'progress_m':0.,'phase':'loaded_prepare','stance':['left','right']}
        w=max(0.,t-self.warmup);cycle=min(int(w/self.period),self.cycles-1);u=(w/self.period)%1.
        if w>=self.walk_time:return {'u':1.,'cycle':self.cycles,'progress_m':self.cycles*self.cycle_back,'phase':'stop_hold','stance':['left','right']}
        prior=self.frames[0]
        for row in self.frames:
            if row['u']<=u:prior=row
        return {'u':u,'cycle':cycle,'progress_m':self.cycle_back*(cycle+u),'phase':prior.get('phase','alternating_step'),'stance':prior['stance']}
    def action(self,env,t,actual_rod_world=None):
        import mujoco as mj
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        state=self.phase(t);u=state['u'];a,b=self.frames[-1],self.frames[-1]
        for aa,bb in zip(self.frames,self.frames[1:]):
            if u<=bb['u']:a,b=aa,bb;break
        ds=b['u']-a['u'];v=0 if ds<=0 else float(np.clip((u-a['u'])/ds,0,1));v=v*v*(3-2*v)
        q=(1-v)*np.array(a['q_target18_rad'])+v*np.array(b['q_target18_rad']);tau=(1-v)*np.array(a.get('static_tau18_Nm',[0]*18))+v*np.array(b.get('static_tau18_Nm',[0]*18))
        if t<self.warmup:
            v=float(np.clip(t/self.warmup,0,1));v=v*v*(3-2*v);q=(1-v)*self.initial+v*q;tau*=v
        nominal=q.copy();q+=tau/env.kp
        desired_tcp=self.cold+[-state['progress_m'],0,0] # anchored forever, no actual-rod chasing
        if self.scratch is None:self.scratch=mj.MjData(env.model)
        d=self.scratch;m=env.model;d.qpos[:]=env.data.qpos;d.qvel[:]=env.data.qvel;idx=env.qidx[:5];bounds=env.ranges[:5];head=m.body('head_roll').id;local=np.array([.13750001972253326,0,-.03849996489687674])
        def f(z):
            d.qpos[idx]=z;mj.mj_kinematics(m,d);H=d.xmat[head].reshape(3,3);e=Rotation.from_matrix(H).as_euler('ZYX');return np.r_[d.xpos[head]+H@local-desired_tcp,.08*(e[1]-.174532925),.08*e[2]]
        seed=env.data.qpos[idx] if self.last_neck is None else self.last_neck
        sol=least_squares(f,np.clip(seed,bounds[:,0]+.05,bounds[:,1]-.05),bounds=(bounds[:,0]+.05,bounds[:,1]-.05),max_nfev=25,ftol=1e-7,xtol=1e-7,gtol=1e-7);self.last_neck=sol.x.copy();q[:5]=sol.x+tau[:5]/env.kp[:5];q[5]=0.;q=np.clip(q,env.ranges[:,0],env.ranges[:,1]);action=(q-env.neutral)/env.scale
        return action,{'requested_joint_targets18':q.tolist(),'nominal_target18':nominal.tolist(),'static_tau_bias18':tau.tolist(),'tcp_target_world':desired_tcp.tolist(),'cold_rod_anchor_world':self.cold.tolist(),'actual_rod_monitor_only':None if actual_rod_world is None else np.asarray(actual_rod_world).tolist(),'scratch_IK_error_equivalent_m':float(np.linalg.norm(f(sol.x))),'does_not_reanchor_to_current_rod':True,**state}
