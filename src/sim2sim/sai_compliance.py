"""Experimental stance impedance and support forces, transmitted as motor torques."""
import mujoco
import numpy as np
from sai_agent.control import STAND_HEIGHT, CROUCH_DROP
from sim2sim.sai_controller import MotionController
from sim2sim.sai_suspension import Suspension
from sim2sim.sai_task_impedance import support_weights

BASE_GEOMETRY=[1.077687564550128,.4125468028688285,.0841540838419023]
LEG_AXES=np.array([i for i in range(16) if i%4!=3])
FLEX_AXES=np.array([i for i in range(16) if i%4 in (1,2)])


class StanceImpedance:
    """Support the chassis without requiring stiff stance position tracking."""
    def __init__(self,controller,parameters,turn_blend=.25):
        if not 0 <= turn_blend <= 1:
            raise ValueError("Invalid turning support blend")
        self.turn_blend=turn_blend
        self.turn_support=0.
        self.model=controller.model
        self.data=controller.data
        self.adapter=controller.adapter
        self.parameters=None if parameters is None else np.asarray(parameters,dtype=float)
        if self.parameters is not None:
            p=self.parameters
            if p.shape!=(6,) or not np.isfinite(p).all() or np.any(p<[12.,.5,0.,0.,0.,.03]) or np.any(p>[80.,4.,1.3,3000.,250.,.6]):
                raise ValueError('Invalid loaded-suspension parameters')
        self.wheel_ids=[self.model.body(n+'_wheel').id for n in ('front_left','front_right','rear_left','rear_right')]
        self.robot_mass=float(self.model.body_subtreemass[self.model.body('chassis').id])
        self.jac=np.zeros((3,self.model.nv));self.jacr=np.zeros_like(self.jac)
        self.height_reference=None
        self.stance=None
        self.previous_feedforward=np.zeros(16)

    def apply(self,result,state):
        if self.parameters is None:return result
        kp,kd,gravity,heave,damper,tau=self.parameters
        ground=np.asarray(state['wheel_ground_heights'])
        gap=self.data.xpos[self.wheel_ids,2]-ground-.048
        contact=np.clip(1-(gap-.002)/.01,0.,1.)
        if self.stance is None:self.stance=contact.copy()
        self.stance+=np.clip(contact-self.stance,-.25,.25)
        contact=self.stance
        gains=np.full(16,80.);damping=np.full(16,2.);feedforward=np.zeros(16)
        for leg in range(4):
            gains[leg*4+1:leg*4+3]=80+(kp-80)*contact[leg]
            damping[leg*4+1:leg*4+3]=2+(kd-2)*contact[leg]
        desired_height=float(ground.mean()+STAND_HEIGHT-CROUCH_DROP*result['effective_crouch'])
        if self.height_reference is None:self.height_reference=desired_height
        delta=(desired_height-self.height_reference)*.02/(tau+.02)
        self.height_reference+=delta
        force=self.robot_mass*9.81*gravity + np.clip(heave*(self.height_reference-self.data.qpos[2])
            +damper*(delta/.02-self.data.qvel[2]),-self.robot_mass*9.81*.5,self.robot_mass*9.81*.5)
        force=np.clip(force,0,self.robot_mass*9.81*1.5)
        # Match net support and its moment around the actual robot CoM.
        # Active-set elimination prevents a foot from pulling on the ground.
        xy=self.data.xpos[self.wheel_ids,:2]-self.data.subtree_com[self.model.body('chassis').id,:2]
        weights=support_weights(contact,xy)
        for i,body in enumerate(self.wheel_ids):
            mujoco.mj_jacBody(self.model,self.data,self.jac,self.jacr,body)
            feedforward[LEG_AXES]-=self.jac[2,self.adapter.vadr[LEG_AXES]]*force*weights[i]
        feedforward[LEG_AXES]+=gravity*self.data.qfrc_bias[self.adapter.vadr[LEG_AXES]]*np.repeat(contact,3)
        self.previous_feedforward+=(feedforward-self.previous_feedforward)*(.02/.06)
        feedforward=self.previous_feedforward.copy()
        # Turning needs more lateral support than the vertical load allocation.
        # Blend smoothly toward the original motor contract, preserving authority.
        goal=self.turn_blend*min(1.,abs(float(state['command'][1]))/.35)
        self.turn_support+=(goal-self.turn_support)*(.02/.12)
        if self.turn_support > 0:
            w=self.turn_support
            gains=(1-w)*gains+80*w
            damping=(1-w)*damping+2*w
            feedforward=(1-w)*feedforward
        result.update(leg_kp=gains.tolist(),leg_kd=damping.tolist(),leg_feedforward=feedforward.tolist(),
                      impedance_contract='sai-joint-impedance-v1',support_force_N=float(force),stance_weights=weights.tolist())
        return result


class CompliantController(MotionController):
    """Explicit experimental entry point; the packaged default cannot affect trials."""
    def __init__(self,root,parameters=None,stair_profile=None):
        super().__init__(root,stair_profile=stair_profile,suspension_profile="off")
        self.suspension=Suspension(BASE_GEOMETRY)
        self.impedance=None if parameters is None else StanceImpedance(self,parameters)
        self.roll_descent=parameters is not None
        self.wheel_ids=[self.model.body(n+'_wheel').id for n in ('front_left','front_right','rear_left','rear_right')]
        self.robot_mass=float(self.model.body_subtreemass[self.model.body('chassis').id])


def apply_impedance(adapter,data,result):
    """Same target/PD/feedforward/saturation order as the native motor contract."""
    adapter.apply(data,np.asarray(result['target_leg']))
    if result.get('impedance_contract')!='sai-joint-impedance-v1':return
    q=data.qpos[adapter.qadr];v=data.qvel[adapter.vadr]
    torque=np.asarray(result['leg_kp'])*(np.asarray(result['target_leg'])-q)-np.asarray(result['leg_kd'])*v+result['leg_feedforward']
    data.ctrl[np.array(adapter.aids)[LEG_AXES]]=np.clip(torque[LEG_AXES],-8.,8.)
