"""Task-specific training objectives; never used by the physical evaluator."""
import math
from collections import deque
import numpy as np
from sim2sim.train.rewards import sit_target_q
from .tasks import DT,CROUCH_STAND,CROUCH_DOWN,crouch_blend

EXTRA_DIM=26

class Objective:
    def __init__(self,w,weights=None,params=None,roller_objective='legacy',walking_objective='legacy'):
        if walking_objective not in ('legacy','sprint_v1','sprint_v2') or (walking_objective!='legacy' and w.task.name!='walking'):
            raise ValueError('Invalid walking objective')
        self.walking_objective=walking_objective
        if roller_objective not in ('legacy','command_heading_v1','stop_hold_v1'):
            raise ValueError('Unknown roller objective: '+roller_objective)
        if roller_objective!='legacy' and (w.task.name!='roller' or not getattr(w,'roller_contract',False)):
            raise ValueError('Versioned roller objective requires the native roller contract')
        self.roller_objective=roller_objective
        self.w=w;self.weights=weights or {};self.params=params or {};self.reset()
        if set(self.params)-{"velocity_variance","yaw_variance","ball_speed_target","brake_velocity_variance"}:raise ValueError("Unknown reward parameter")
        if 'brake_velocity_variance' in self.params and (w.task.name!='roller' or not getattr(w,'roller_contract',False)):
            raise ValueError('Brake variance requires the native roller task')
        if any(not math.isfinite(float(v)) or float(v)<=0 for v in self.params.values()):
            raise ValueError("Reward parameters must be positive and finite")
        self.motion=None
        if any(k.startswith("motion_") for k in self.weights):
            if w.task.name!="roulade" or not getattr(w,"time_input_s",0.):raise ValueError("Motion-reference rewards require an explicit timed roll")
            from .roll_motion import reference
            self.motion=reference()

    def reset(self):
        w=self.w
        self.smooth=np.zeros(3);self.net=0.;self.frontier=0.;self.pivot=False
        self.inverted=False;self.touch=False;self.wrong=False;self.impact=False
        self.ball_best=0.;self.yaw0=w.features["yaw"];self.start_xy=w.features["xy"].copy()
        self.previous_score=0.
        self.sprint_heading_target=self.yaw0
        self.was_idle=False
        self.brake_started=None
        self.brake_speeds=deque(maxlen=10)
        self.low_speed_stability=deque(maxlen=50)
        self.stop_paid=False
        if getattr(w,"roll_start",None) is not None:
            self.net,self.frontier,pivot,inverted=w.roll_start
            self.pivot=bool(pivot);self.inverted=bool(inverted)

    def extra(self):
        w=self.w;f=w.features;rot=f["rot"]
        relative=rot.T@(f["ball_pos"]-w.state.base_pos) if w.task.name.startswith("kick") else np.zeros(3)
        x=np.r_[f["vel"],f["contact"],f["foot_pos"][:,2]*10,f["z"]*10,
                np.nan_to_num([f["mouth_pos"][2]*10,f["mouth_down"]]),relative*10,
                rot.T@f["ball_vel"],w.t/w.task.seconds,self.net/6.283,self.pivot,self.inverted,
                self.touch,self.wrong,self.impact,self.smooth]
        assert x.shape==(EXTRA_DIM,),x.shape
        if self.walking_objective in ('sprint_v1','sprint_v2'):
            error=self.sprint_heading_target-f['yaw']
            # Privileged critic only: replace two unused mouth channels.
            x[8:10]=[math.sin(error),math.cos(error)]
        return x.astype(np.float32)

    def compute(self):
        w=self.w;f=w.features;t=w.t;task=w.task;name=task.name;cmd=w.executed_command
        self.smooth=.9*self.smooth+.1*np.r_[f["vel"][:2],f["gyro"][2]]
        up=float(np.exp(-((1-f["up"])/.06)**2)) # inverted receives zero
        rate=float(np.sum((w.last-w.old_last)**2))
        calm=float(np.exp(-np.sum(f["gyro"][:2]**2)/4))
        stand=float(np.exp(-((f["z"]-.115)/.03)**2))*up
        pose=float(np.exp(-np.mean((w.state.q-w.home)**2)/.12))
        terms={"action_rate":-.08*rate,"joint_speed":-.00002*float(np.sum(w.state.qd**2))}
        if self.walking_objective in ('sprint_v1','sprint_v2'):
            target=cmd[:3]
            if getattr(w,'motion',None) is not None:self.sprint_heading_target=w.executed_heading_target
            else:self.sprint_heading_target+=float(target[2])*DT
            error=self.sprint_heading_target-f['yaw']
            error=math.atan2(math.sin(error),math.cos(error))
            moving=target[0]>.01
            # A linear tracking slope remains informative below a new speed
            # cell; no saturated wheel or exponential far-target reward.
            track=1.-abs(float(self.smooth[0]-target[0]))/max(.15,float(target[0]))
            # v1 retained for exact experiment replay. With path feedback the
            # actor must follow the corrective lateral request, not resist it.
            lateral_target=float(target[1]) if self.walking_objective=='sprint_v2' else 0.
            terms.update(velocity=6*max(-2.,track),
                lateral=2*math.exp(-float((self.smooth[1]-lateral_target)**2)/.01),
                yaw=3*math.exp(-float((self.smooth[2]-target[2])**2)/.25),
                heading=2*math.cos(error),upright=2*up,height=.5*stand,
                pose=(.15 if moving else .5)*pose)
        elif name=='roller' and getattr(w,'roller_contract',False):
            target=w.roller_target_yaw if self.roller_objective=='legacy' else w.executed_heading_target
            delta=target-f['yaw'];error=math.atan2(math.sin(delta),math.cos(delta))
            throttle=float(cmd[0]);speed=float(np.linalg.norm(self.smooth[:2]))
            terms.update(push=10*max(0.,throttle)*np.tanh(max(0.,self.smooth[0])/.3),
                brake=8*max(0.,-throttle)*np.exp(-speed**2/self.params.get('brake_velocity_variance',.09)),
                heading=3*np.exp(-error**2/.25),upright=2*up,height=stand,
                coast_calm=float(abs(throttle)<.01)*np.exp(-np.sum(w.state.qd[np.r_[0:5,9:14]]**2)/25),
                reverse=-6*max(0.,-self.smooth[0]),excess_speed=-3*max(0.,self.smooth[0]-.7)**2,
                yaw_rate_cost=-.02*float(f['gyro'][2]**2),pose=.2*pose)
            if 'brake_speed_cost' in self.weights:
                terms['brake_speed_cost']=-speed*float(throttle<-.01)
            if self.roller_objective=='stop_hold_v1':
                braking=throttle<-.01
                if not braking:
                    self.brake_started=None;self.brake_speeds.clear()
                    self.low_speed_stability.clear();self.stop_paid=False
                elif self.brake_started is None:
                    self.brake_started=t-DT
                elapsed=0. if self.brake_started is None else t-self.brake_started
                self.brake_speeds.append(float(np.linalg.norm(f['vel'][:2])))
                average=float(np.mean(self.brake_speeds))
                stable=f['tilt']<15 and f['z']>.08 and np.sum(f['contact'])>0
                if braking and len(self.brake_speeds)==10 and average<.05:
                    self.low_speed_stability.append(bool(stable))
                else:self.low_speed_stability.clear()
                hold=len(self.low_speed_stability)/50.
                stable_fraction=float(np.mean(self.low_speed_stability)) if hold else 0.
                confirmed=hold>=1. and stable_fraction>=.9
                # A linear speed objective retains a gradient near zero. The
                # hold phase keeps running after confirmation; no early reset.
                terms['brake']=8*max(0.,-throttle)*max(-2.,1.-average/.3)
                terms['brake_clock']=-2*float(braking)*min(3.,elapsed)*min(1.,average/.05)
                terms['stop_hold']=8*float(braking)*hold*stable_fraction
                terms['stop_confirmed']=20*float(confirmed and not self.stop_paid and elapsed<=2.+1e-9)
                if confirmed:self.stop_paid=True
        elif name in ("standing","walking","roller"):
            target=cmd[:3] if name!="standing" else np.zeros(3)
            v_err=float(np.sum((self.smooth[:2]-target[:2])**2))
            yaw_err=float((self.smooth[2]-target[2])**2)
            terms.update(velocity=4*np.exp(-v_err/self.params.get("velocity_variance",.025)),
                         yaw=3*np.exp(-yaw_err/self.params.get("yaw_variance",.18)),upright=2*up,height=.5*stand)
            idle=np.linalg.norm(target)<.01
            if idle:
                if not self.was_idle:self.yaw0=f["yaw"];self.start_xy=f["xy"].copy()
                dy=math.atan2(math.sin(f["yaw"]-self.yaw0),math.cos(f["yaw"]-self.yaw0))
                terms.update(idle_position=2*np.exp(-float(np.sum((f["xy"]-self.start_xy)**2))/.0025),idle_heading=2*np.exp(-dy*dy/.03))
            self.was_idle=idle
            if 'straight_heading' in self.weights:
                if getattr(w,'yaw_memory',None) is None:raise ValueError('Straight-heading objective requires declared IMU memory')
                terms['straight_heading']=2*np.exp(-w.yaw_memory.error**2/.01)*float(abs(target[2])<=.05)
            terms["pose"]=(.3 if not idle else 1.)*pose
            if name=="standing":terms["calm"]=calm
        elif name=="sitstand":
            sitting=cmd[0]>.5;target=sit_target_q(w.home) if sitting else w.home
            z=.06 if sitting else .115
            terms.update(posture=3*np.exp(-np.mean((w.state.q-target)**2)/.06),height=4*np.exp(-((f["z"]-z)/.02)**2),upright=2*up,calm=.5*calm)
        elif name=="ground_pick":
            phase=(t%task.period)/task.period
            if phase<.375:blend=phase/.375
            elif phase<.425:blend=1.
            elif phase<.8:blend=(.8-phase)/.375
            else:blend=0.
            # Source uses phase weights, not a prescribed tip trajectory.
            # Driving a linear target height would oppose its earlier reach.
            return_gate=float(np.clip((phase-.425)/.375,0,1))
            down=max(0.,f["mouth_down"])
            terms.update(tip_proximity=6*blend*np.exp(-(max(0.,f["mouth_pos"][2])/.06)**2),mouth_down=2*blend*down,return_stand=5*return_gate*stand,return_pose=3*return_gate*pose,feet_grounded=2*float(np.mean(f["contact"])))
            self.impact|=f["head_contact"]
            terms["head_impact"]=-15*float(f["head_contact"])
            terms["side_tilt"]=-float(f["rot"][2,1]**2)*3
        elif name.startswith("kick"):
            foot="ankle_left" if task.foot==0 else "ankle_right"
            other="ankle_right" if task.foot==0 else "ankle_left"
            newtouch=foot in f["kick_contacts"] and not self.touch
            self.touch|=foot in f["kick_contacts"];self.wrong|=other in f["kick_contacts"]
            ball_forward=float(f["ball_vel"][:2]@w.heading)
            ball_side=float(f["ball_vel"][:2]@np.array([-w.heading[1],w.heading[0]]))
            speed_target=self.params.get('ball_speed_target',1.)
            maxv=max(0.,min(ball_forward,speed_target));progress=max(0.,maxv-self.ball_best)
            self.ball_best=max(self.ball_best,maxv)
            terms.update(ball_progress=80*progress,ball_forward=8*maxv*float(self.touch),ball_side=-2*abs(ball_side),touch=15*float(newtouch),upright=2*up,stand=(4 if self.touch else 1)*stand,pose=.3*pose)
            terms["overspeed"]=-4*max(0.,ball_forward-speed_target)
            terms["wrong_foot"]=-8*float(other in f["kick_contacts"])
            terms["support"]=float(f["contact"][1-task.foot])*up
            if "heading" in self.weights:
                dy=math.atan2(math.sin(f["yaw"]-self.yaw0),math.cos(f["yaw"]-self.yaw0))
                terms["heading"]=-dy*dy
        elif name=="roller_crouch":
            blend=crouch_blend((t%task.period)/task.period)
            target=CROUCH_STAND*(1-blend)+CROUCH_DOWN*blend
            z=.12*(1-blend)+.065*blend
            terms.update(posture=5*np.exp(-np.mean((w.state.q-target)**2)/.15),height=3*np.exp(-((f["z"]-z)/.025)**2),upright=3*up,glide=.5*np.exp(-((f["vel"][0]-.2)/.3)**2))
        elif name=="roulade":
            self.net+=float(f["gyro"][1])*DT
            sagittal=float(np.clip((.866-abs(f["rot"][2,1]))/.366,0,1))
            headvalid=f["head_contact"] and f["head_up"]<-.3 and .35<self.net<2.97
            newpivot=headvalid and not self.pivot;self.pivot|=headvalid
            self.inverted|=self.pivot and f["up"]<-.7
            frontier=min(2*np.pi,max(self.frontier,self.net))
            newangle=max(0.,frontier-self.frontier);self.frontier=frontier
            supported=float(f["supported"])
            headgate=float(self.net<np.pi or self.pivot)
            terms.update(forward_progress=8*min(newangle/DT,5.)/5*sagittal*supported*headgate,pivot=10*float(newpivot),sagittal=1.5*sagittal,support=supported)
            landgate=float(self.inverted and self.net>4.5)
            terms["landing"]=15*landgate*stand*calm
            terms["land_pose"]=3*landgate*pose
            terms["land_bootstrap"]=4*landgate*(max(0.,f["up"])+np.exp(-((f["z"]-.115)/.05)**2))
            terms["stand_tax"]=-30*landgate*max(0.,.115-f["z"])
            terms["overspeed"]=-.1*max(0.,abs(f["gyro"][1])-7.)**2
            terms["stall"]=-2*float(t>2. and not (landgate and stand>.5))
            terms["reverse"]=-2*max(0.,-f["gyro"][1])*float(self.net<5.8)
            if "over_rotation" in self.weights:
                terms["over_rotation"]=-min(1.,(max(0.,self.net-2*np.pi)/np.pi)**2)
            if "land_leg_pose" in self.weights:
                legs=np.r_[0:5,9:14]
                terms["land_leg_pose"]=3*landgate*np.exp(-np.mean((w.state.q[legs]-w.home[legs])**2)/.12)
            if "land_heading" in self.weights:
                targetyaw=math.atan2(w.heading[1],w.heading[0])
                terms["land_heading"]=3*landgate*max(0.,f["up"])*math.cos(f["yaw"]-targetyaw)
            if self.motion is not None:
                ref=self.motion.sample(t+w.time_offset)
                terms["motion_pose"]=10*np.exp(-np.mean((w.state.q-ref["q"])**2)/.15)
                terms["motion_orientation"]=8*np.exp(-np.sum((f["rot"][2,:]-ref["gravity"])**2)/.25)
                terms["motion_height"]=3*np.exp(-((f["z"]-ref["z"])/.04)**2)
                terms["motion_rotation"]=4*np.exp(-((self.net-ref["net"])/.75)**2)
        else:raise ValueError(name)
        if 'yaw_spin' in self.weights and (name=='roulade' or name.startswith('kick')):
            world_yaw_rate=float(f['rot'][2,:]@f['gyro'])
            terms['yaw_spin']=-min(4.,world_yaw_rate**2)
        # Every trial records any adjusted term weights; the evaluator stays fixed.
        terms={k:float(v)*self.weights.get(k,1.) for k,v in terms.items()}
        reward=DT*sum(terms.values())
        fell=bool(f["z"]<.04 or f["tilt"]>75)
        if self.walking_objective in ('sprint_v1','sprint_v2'):fell=bool(f['z']<.055 or f['tilt']>60)
        terminate=fell and name in ("standing","walking","roller","kick_left","kick_right","ground_pick")
        if terminate:reward-=3.
        if not np.isfinite(reward):raise FloatingPointError("nonfinite task reward")
        return float(reward),terminate,terms
