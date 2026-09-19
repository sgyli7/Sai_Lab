"""Physical backends and task telemetry, independent of training rewards."""
import math,hashlib
from pathlib import Path
import mujoco
import numpy as np

from sim2sim.paths import load_robot_json,sim2sim_root
from sim2sim.obs import build_obs
from sim2sim.coords import quat_wxyz_to_mat, mat_to_quat_wxyz
from sim2sim.backends.godot_backend import GodotBackend, inertial_to_body
from sim2sim.backends.mujoco_backend import XL330_M6_KT
from sim2sim.train.reset_poses import HomePoseSampler
from sim2sim.train.rewards import sit_target_q
from .tasks import DT, command


def roller_support_groups(model,names):
    """Wheel ownership follows the articulated ankle, including during a fall."""
    groups=[[],[]]
    for name in names:
        if not name.startswith('tire'):continue
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        side=None
        while body>0:
            ancestor=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body)
            if ancestor in ('ankle_l_v1','ankle_r_v1'):
                side=0 if ancestor=='ankle_l_v1' else 1;break
            body=int(model.body_parentid[body])
        if side is None:raise RuntimeError('Wheel has no known ankle ancestor: '+name)
        groups[side].append(name)
    if not all(groups):raise RuntimeError('Both articulated roller support groups must be present')
    return groups

class World:
    def __init__(self, task, backend="godot", headless=True, reference_profile="xml",time_input_s=0.,heading_input=False,entry_source=None,roller_contract=False,yaw_memory_input=False,state_input='',task_input='',motion_settings=None,scene_override=None):
        self.task, self.backend_name = task, backend
        from sim2sim.motion_control import MotionControl
        if motion_settings is not None and task.name!='walking':raise ValueError('Training motion feedback is currently walking only')
        self.motion=None if motion_settings is None else MotionControl(motion_settings)
        self._command_stamp=None
        self.time_input_s=float(time_input_s);self.time_offset=0.
        self.heading_input=bool(heading_input)
        from sim2sim.policy_memory import YawDriftMemory
        if yaw_memory_input and task.name not in ('walking','kick_left','kick_right'):raise ValueError('Yaw memory requires walking or a kick')
        self.yaw_memory=YawDriftMemory() if yaw_memory_input else None
        self.entry_source=None if entry_source is None else Path(entry_source)
        self.state_input=state_input
        from sim2sim.policy_task_state import BrakeTaskState, task_input as validate_task_input, TASK_STATE_KEY
        from sim2sim.policy_state import BRAKE_STATE_V1
        validate_task_input({TASK_STATE_KEY:task_input})
        if task_input and (task.name!='roller' or state_input!=BRAKE_STATE_V1):
            raise ValueError('Brake task observation requires the roller velocity-state input')
        self.task_state=BrakeTaskState() if task_input else None
        if state_input and (state_input!=BRAKE_STATE_V1 or task.name not in ('walking','roller') or
                            (task.name=='roller' and not roller_contract) or time_input_s or heading_input or yaw_memory_input):
            raise ValueError('Residual state input requires walking or the native roller contract')
        self.roller_contract=bool(roller_contract)
        if self.roller_contract and task.name!='roller':raise ValueError('Native roller contract requires the roller task')
        if self.heading_input and not self.time_input_s:raise ValueError("Relative heading requires a timed maneuver")
        if self.time_input_s and (task.name not in ('roulade','kick_left','kick_right') or self.time_input_s!=task.seconds):
            raise ValueError("Time input requires the full declared roll or kick duration")
        self.cfg = load_robot_json(task.robot_path)
        files={"robot":task.robot_path,"mjcf":Path(self.cfg["mjcf"])}
        if backend=="godot":files.update(server=sim2sim_root()/"godot/physics_server.gd",spec=Path(self.cfg["godot_spec"]))
        if backend=="godot":files['robot_scene']=Path(self.cfg['godot_spec']).with_name('robot.tscn')
        self.physics={"backend":backend,"joint_limits":"signed_fresh_reset_v1","dt":DT,"current_limit_a":1.75,
                      "state_transfer":"inertial_com_velocity_v2","velocity_metric":"trunk_inertial_com_v2",
                      "files":{k:{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in files.items()}}
        self.sampler = HomePoseSampler(self.cfg)
        self.mj = self.sampler.mj
        godot_scene='res://main.tscn'
        if reference_profile!="xml":
            if reference_profile!="source_play" or (backend!='mujoco' and task.robot!='microduck_roller'):raise ValueError("Unknown reference profile")
            if task.robot=="microduck_roller":
                for j in range(self.mj.model.njnt):
                    name=mujoco.mj_id2name(self.mj.model,mujoco.mjtObj.mjOBJ_JOINT,j) or ""
                    if name.startswith("passive_"):
                        self.mj.model.dof_frictionloss[self.mj.model.jnt_dofadr[j]]=.003
                self.physics["source_play_overrides"]={"passive_wheel_frictionloss":.003}
                if backend=='godot':
                    godot_scene='res://research/roller_source_play.tscn'
                    for name in ['roller_source_play.gd','roller_source_play.tscn']:
                        path=sim2sim_root()/'godot/research'/name
                        self.physics['files'][name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    self.physics['source_play_overrides']['method']='zero_speed_bounded_hinge_constraint'
            self.physics["reference_profile"]=reference_profile
        self.home = self.sampler.home
        self.meta = {mujoco.mj_id2name(self.mj.model,mujoco.mjtObj.mjOBJ_BODY,i): i for i in range(1,self.mj.model.nbody)}
        names = ["trunk_base", "jaw_soft", "ball", "ankle_left", "ankle_right", "ankle_l_v1", "ankle_r_v1"]
        names += [n for n in self.meta if n.startswith("tire")]
        self.report_names = [n for n in names if n in self.meta]
        if task.name == "roulade": self.report_names = list(self.meta)
        self.site_id = mujoco.mj_name2id(self.mj.model,mujoco.mjtObj.mjOBJ_SITE,"mouth_tip")
        if task.name == "ground_pick" and self.site_id < 0:
            raise RuntimeError("ground_pick requires the real mouth_tip site")
        if scene_override is not None:
            if reference_profile != "xml":
                raise ValueError("Presentation scene override requires the unchanged XML physics profile")
            godot_scene = str(scene_override)
        self.backend = self.mj if backend == "mujoco" else GodotBackend(Path(self.cfg["godot_spec"]),headless=headless,scene=godot_scene,current_limit_a=1.75,recv_timeout=15)
        if backend == "mujoco":
            limit = 1.75*XL330_M6_KT
            self.mj.model.actuator_forcerange[:] = [-limit,limit]
            self.mj.model.actuator_forcelimited[:] = 1
        self.state = None
        self.features = None
        self.last = np.zeros(14,np.float32)
        self.pending_ball=None

    def reset(self, seed, condition="default", randomize=True, entry_speed=None, phase_start=0., q_override=None):
        if self.motion is not None:self.motion.reset()
        self._command_stamp=None
        if self.yaw_memory is not None:self.yaw_memory.reset()
        if self.task_state is not None:self.task_state.reset()
        self.pending_ball=None
        self.roll_start=None
        self.time_offset=0.
        self.rng = np.random.default_rng(seed)
        self.condition = condition
        self.command_schedule=None
        self.command_tape=None
        self.sprint_selection=None
        if condition.startswith('sprint_'):
            if self.task.name!='walking':raise ValueError('Sprint applies to walking only')
            from .sprint_tasks import commands
            limits={} if self.motion is None else self.motion.settings.get('twist_limits',{})
            self.command_tape,self.sprint_selection=commands(condition,DT,self.task.seconds,include_selection=True,twist_limits=limits)
        if condition.startswith('keyboard_'):
            from .schedules import keyboard_commands,roller_keyboard_commands
            if self.task.name=='walking':self.command_tape=keyboard_commands(condition.removeprefix('keyboard_'),DT)
            elif self.task.name=='roller' and self.roller_contract:
                self.command_tape=roller_keyboard_commands(condition.removeprefix('keyboard_'),DT)
            else:raise ValueError('Keyboard training tapes require walking or native roller')
        if condition=="random_seq":
            if self.task.name not in ("walking","roller"):raise ValueError("Random twist schedule requires locomotion")
            from .schedules import random_schedule
            self.command_schedule=random_schedule(self.task,seed)
        self.t = float(phase_start)
        sitting = self.task.name == "sitstand" and condition in ("rise","sit_hold")
        q0 = sit_target_q(self.home) if sitting else q_override
        z = .06 if sitting else None
        yaw = float(self.rng.uniform(-math.pi,math.pi)) if randomize else 0.
        if self.roller_contract:
            from .roller_tasks import initial_speed,target_offset
            self.roller_target_yaw=yaw+target_offset(condition)
            if entry_speed is None:entry_speed=initial_speed(condition)
        poses,_,_ = self.sampler.sample(self.rng,yaw_range=(yaw,yaw),joint_noise_rad=.015 if randomize else 0.,q_base=q0,z=z)
        d,m = self.mj.data,self.mj.model
        if entry_speed is None: entry_speed = .3 if self.task.name == "roller_crouch" else 0.
        self.heading = np.array([math.cos(yaw),math.sin(yaw)],np.float64)
        if entry_speed:
            d.qvel[self.mj.free_dofadr:self.mj.free_dofadr+2] = self.heading*entry_speed
        if "ball" in self.meta:
            bid=self.meta["ball"]; jid=int(m.body_jntadr[bid]); adr=int(m.jnt_qposadr[jid])
            off=np.array([.09,.042 if self.task.foot==0 else -.042])
            if randomize: off += self.rng.uniform(-.015,.015,2)
            rot=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
            d.qpos[adr:adr+3]=[*((rot@off)+d.qpos[self.mj.free_qposadr:self.mj.free_qposadr+2]),.036]
            d.qpos[adr+3:adr+7]=[1,0,0,0]
            if not self.task.name.startswith('kick'):d.qpos[adr:adr+3]=[5.,5.,.035]
        mujoco.mj_forward(m,d)
        poses=self.mj.body_poses_mujoco()
        if self.backend_name == "mujoco":
            self.state=self.mj.reset(qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=self.home)
        else:
            self.state=self.backend.reset(ctrl=self.home,bodies=poses,report_bodies=self.report_names)
        self.last[:] = 0
        # Reset reports contain exact teleported poses; contact is not established yet.
        self.features=self.measure(reset=True)
        self.initial_xy=np.array(self.state.base_pos[:2],copy=True)
        self.initial_ball=self.features["ball_pos"].copy()
        return self.obs()

    def reset_from_roll_state(self,qpos,qvel,last,heading,progress,source_time=0.):
        """Training-only mid-roll state; subsequent dynamics remain native."""
        if self.task.name!="roulade":raise ValueError("Roll starts only apply to roulade")
        self.pending_ball=None;self.t=0.;self.condition="default";self.command_schedule=None
        self.time_offset=float(source_time) if self.time_input_s else 0.
        self.heading=np.array(heading,copy=True);self.last=np.array(last,np.float32,copy=True)
        ctrl=self.home+self.last
        state=self.mj.reset(qpos=np.array(qpos),qvel=np.array(qvel),ctrl=ctrl)
        if self.backend_name=="mujoco":self.state=state
        else:self.state=self.backend.reset(ctrl=ctrl,bodies=self.mj.body_poses_mujoco(),report_bodies=self.report_names)
        self.features=self.measure(reset=True)
        self.initial_xy=np.array(self.state.base_pos[:2],copy=True)
        self.initial_ball=self.features["ball_pos"].copy()
        self.roll_start=np.array(progress,copy=True)
        return self.obs()

    def obs(self):
        obs=build_obs(self.state,self.last,self.command(),self.home)
        from sim2sim.policy_state import inject_state
        obs=inject_state(obs,self.state,self.state_input)
        if self.task_state is not None:obs=self.task_state.observe(obs,self.features['contact'],self.t)
        return obs if self.yaw_memory is None else self.yaw_memory.observe(obs,stamp=self.t)

    def command(self):
        if self.motion is None:return self.requested_command()
        if self._command_stamp!=self.t:
            skill='sprint' if getattr(self,'sprint_composed',False) and self.sprint_active() else 'walking'
            self._controlled_command=self.motion.command(self.requested_command(),self.state,skill,DT)
            self._command_stamp=self.t
        return self._controlled_command.copy()

    def sprint_active(self):
        if self.sprint_selection is None:raise ValueError('Composed training requires a sprint selection tape')
        return bool(self.sprint_selection[min(int(round(self.t/DT)),len(self.sprint_selection)-1)])

    def requested_command(self):
        if self.command_tape is not None:
            return self.command_tape[min(int(round(self.t/DT)),len(self.command_tape)-1)].copy()
        if self.roller_contract:
            from .roller_tasks import command as roller_command
            return roller_command(self)
        if self.time_input_s:
            from sim2sim.policy_time import time_command
            return time_command(self.t+self.time_offset,self.time_input_s,
                self.features['rot'] if self.heading_input else None,self.heading)
        if self.command_schedule is not None:
            from .schedules import scheduled_command
            return scheduled_command(self.command_schedule,self.t)
        return command(self.task,self.t,self.condition)

    def send(self, action, capture_path=None):
        if not np.isfinite(action).all(): raise FloatingPointError("nonfinite policy action")
        self.executed_command=self.command()
        if self.motion is not None:self.executed_heading_target=self.motion.target_yaw
        if self.roller_contract:
            # The actor receives a relative heading error, including keyboard
            # commands. Capture its world target before applying this action.
            self.executed_heading_target=self.features['yaw']+float(self.executed_command[2])
        self.old_last=self.last.copy()
        self.last=np.asarray(action,np.float32).copy()
        ctrl=self.home+self.last
        if self.backend_name == "godot": self.backend.send_step(ctrl,n_substeps=4,report="research",capture_path=capture_path,place_ball=self.pending_ball)
        else:
            if self.pending_ball is not None:
                m,d=self.mj.model,self.mj.data;bid=self.meta["ball"];jid=int(m.body_jntadr[bid]);adr=int(m.jnt_qposadr[jid]);vadr=int(m.jnt_dofadr[jid])
                d.qpos[adr:adr+7]=[*self.pending_ball,1,0,0,0];d.qvel[vadr:vadr+6]=0.;mujoco.mj_forward(m,d)
            self.contact_events={n:[] for n in self.report_names}
            for _ in range(4):
                self.state=self.mj.step(ctrl,n_substeps=1)
                for n,b in self._mj_bodies().items():self.contact_events[n].extend(b["contacts"])
        self.pending_ball=None

    def recv(self):
        if self.backend_name == "godot": self.state=self.backend.recv_step()
        self.t += DT
        self.features=self.measure()
        return self.obs()

    def step(self,action):
        self.send(action)
        return self.recv()

    def _mj_bodies(self):
        m,d=self.mj.model,self.mj.data
        reports={}
        for name in self.report_names:
            bid=self.meta[name];vel=np.zeros(6)
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,bid,vel,0)
            reports[name]={"pos":d.xpos[bid].copy(),"rot":d.xmat[bid].reshape(3,3).copy(),"linvel":vel[3:].copy(),"ground_contact":False,"contacts":[]}
        for c in d.contact:
            b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            for a,b,geom in [(b1,b2,c.geom1),(b2,b1,c.geom2)]:
                name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,a)
                if name not in reports: continue
                other=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b) or "world"
                reports[name]["contacts"].append({"body":other,"ground":b==0,"shape":mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,geom) or f"geom_{geom}","impulse":0.})
                reports[name]["ground_contact"] |= b==0
        return reports

    def _godot_bodies(self,reset=False):
        raw=(self.state.extra or {}).get("raw",{})
        items=raw.get("body_states")
        if reset and items is None:items=raw.get("dump",[])
        if items is None: raise RuntimeError("research telemetry absent")
        out={}
        for b in items:
            name=b["name"]
            if name not in self.meta or "quat" not in b: continue
            bid=self.meta[name]
            pos,quat=inertial_to_body(np.asarray(b["pos"]),np.asarray(b["quat"]),self.mj.model.body_ipos[bid],self.mj.model.body_iquat[bid])
            out[name]={**b,"pos":pos,"rot":quat_wxyz_to_mat(quat),"linvel":np.asarray(b.get("linvel",[0,0,0])),"ground_contact":bool(b.get("ground_contact",False)),"contacts":b.get("contacts",[])}
        missing=set(self.report_names)-set(out)
        if missing: raise RuntimeError(f"missing task bodies: {missing}")
        return out

    def measure(self,reset=False):
        s=self.state
        b=self._mj_bodies() if self.backend_name=="mujoco" else self._godot_bodies(reset)
        for n,body in b.items():
            body["contact_events"] = [] if reset else (self.contact_events.get(n,[]) if self.backend_name=="mujoco" else body.get("contact_events",body["contacts"]))
        rot=quat_wxyz_to_mat(s.base_quat_wxyz)
        yaw=math.atan2(rot[1,0],rot[0,0]); cy,sy=math.cos(yaw),math.sin(yaw)
        # Evaluate the same trunk inertial-COM velocity in both simulators;
        # MuJoCo data.cvel is expressed at the subtree COM and is not this value.
        linear=np.asarray(s.base_linvel)
        if self.backend_name=="mujoco":
            bv=np.zeros(6);bid=self.mj.base_body_id
            mujoco.mj_objectVelocity(self.mj.model,self.mj.data,mujoco.mjtObj.mjOBJ_BODY,bid,bv,0)
            linear=bv[3:]
        vel=np.array([[cy,sy,0],[-sy,cy,0],[0,0,1]])@linear
        supports = [["ankle_left"],["ankle_right"]]
        if self.task.robot == "microduck_roller":
            # Current trunk-relative positions change when the duck tips over.
            # Reclassifying each frame can empty a group and create NaN critic
            # features precisely at a terminal transition. Ownership is fixed.
            if not hasattr(self,'_roller_supports'):
                self._roller_supports=roller_support_groups(self.mj.model,self.meta)
            supports=self._roller_supports
        contact=np.array([any(b[n]["ground_contact"] for n in group) for group in supports],np.float32)
        foot_pos=np.stack([np.mean([b[n]["pos"] for n in group],axis=0) if group else np.full(3,np.nan) for group in supports])
        foot_vel=np.stack([np.mean([b[n]["linvel"] for n in group],axis=0) if group else np.full(3,np.nan) for group in supports])
        jaw=b.get("jaw_soft")
        mouth_pos=np.full(3,np.nan);mouth_down=np.nan
        if self.site_id>=0:
            sid=self.site_id;bid=int(self.mj.model.site_bodyid[sid]);name=mujoco.mj_id2name(self.mj.model,mujoco.mjtObj.mjOBJ_BODY,bid)
            sb=b[name];mouth_pos=sb["pos"]+sb["rot"]@self.mj.model.site_pos[sid]
            sr=sb["rot"]@quat_wxyz_to_mat(self.mj.model.site_quat[sid]);mouth_down=float(-sr[2,0])
        ball=b.get("ball")
        kick_contacts=[] if ball is None else [c["body"] for c in ball["contact_events"] if not c["ground"]]
        f={"z":float(s.base_pos[2]),"xy":np.asarray(s.base_pos[:2]),"rot":rot,"yaw":yaw,
           "up":float(rot[2,2]),"tilt":float(np.degrees(np.arccos(np.clip(rot[2,2],-1,1)))),
           "vel":vel,"gyro":np.asarray(s.base_angvel_local),"contact":contact,"foot_pos":foot_pos,"foot_vel":foot_vel,
           "mouth_pos":mouth_pos,"mouth_down":mouth_down,"head_contact":False if jaw is None else any(c["ground"] for c in jaw["contact_events"]),
           "head_up":1. if jaw is None else float((jaw["rot"]@np.array([.882,0,.471]))[2]),
           "supported":any(v["ground_contact"] for k,v in b.items() if k!="ball"),
           "ball_pos":np.zeros(3) if ball is None else np.asarray(ball["pos"]),
           "ball_vel":np.zeros(3) if ball is None else np.asarray(ball["linvel"]),
           "kick_contacts":kick_contacts,"bodies":b}
        if not all(np.isfinite(x).all() for x in [s.q,s.qd,s.base_pos,s.base_quat_wxyz,vel,s.base_angvel_local]):
            raise FloatingPointError("nonfinite physical state")
        return f

    def nudge(self,velocity):
        if self.backend_name=="godot":self.backend.nudge(np.asarray(velocity))
        else:
            self.mj.data.qvel[self.mj.free_dofadr:self.mj.free_dofadr+3]+=velocity
            mujoco.mj_forward(self.mj.model,self.mj.data)

    def prepare_standing_entry(self):
        from .models import NativeAnchor
        from .tasks import TASKS
        teacher_name="roller" if self.task.robot=="microduck_roller" else "standing"
        seated=self.task.name=="sitstand" and self.condition in ("rise","sit_hold")
        if seated:teacher_name="sitstand"
        source=self.entry_source if self.entry_source is not None and not seated else TASKS[teacher_name].source
        key=str(source.resolve())
        if not hasattr(self,"_entry_teachers"):self._entry_teachers={}
        if key not in self._entry_teachers:self._entry_teachers[key]=NativeAnchor(source)
        teacher=self._entry_teachers[key]
        if teacher.time_input_s:raise ValueError('Entry actor must accept a zero idle command')
        cmd=np.zeros(13,np.float32)
        if seated:cmd[0]=1.
        if self.task.robot=="microduck_roller" and self.task.name=="roller_crouch":cmd[0]=.3
        if "ball" in self.meta:self.pending_ball=[5.,5.,.035]
        return teacher,cmd

    def finish_standing_entry(self,reset_motion=True):
        self.t=0.;self.time_offset=0.
        if reset_motion and self.motion is not None:self.motion.reset()
        self._command_stamp=None
        if self.yaw_memory is not None:self.yaw_memory.reset()
        if self.task_state is not None:self.task_state.reset()
        self.initial_xy=self.features["xy"].copy()
        yaw=self.features["yaw"];self.heading=np.array([math.cos(yaw),math.sin(yaw)])
        if self.roller_contract:
            from .roller_tasks import target_offset
            self.roller_target_yaw=yaw+target_offset(self.condition)
        if "ball" in self.meta and self.task.name.startswith('kick'):
            off=np.array([.09,.042 if self.task.foot==0 else -.042])+self.rng.uniform(-.015,.015,2)
            rotation=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
            self.pending_ball=[*(self.initial_xy+rotation@off),.035]
            self.initial_ball=np.asarray(self.pending_ball)
            self.features["ball_pos"]=self.initial_ball.copy();self.features["ball_vel"]=np.zeros(3)
        return self.obs()

    def enter_from_standing(self,seconds=1.):
        """A real policy handoff, retaining last_action as upstream play does."""
        teacher,cmd=self.prepare_standing_entry()
        for _ in range(round(seconds/DT)):
            obs=build_obs(self.state,self.last,cmd,self.home)
            self.step(teacher(obs[None])[0])
        return self.finish_standing_entry()

    def close(self):
        if self.backend_name=="godot":self.backend.close()
        self.sampler.close()
