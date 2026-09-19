"""Full MuJoCo articulated suspension/cargo episodes, sampled at every physics step."""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from sai_agent.paths import resource_root
from sai_agent.runtime import JointAdapter
from sim2sim.sai_compliance import CompliantController,apply_impedance,LEG_AXES
from sim2sim.sai_stair_v2 import observation_numpy as phase_free_observation, actions_from_targets_numpy
from sim2sim.sai_stair_v6 import STAIR_CLEARANCE_M
from sai_suspension_experiment import Terrain,SCAN,PATH,DENSE


class LoadedTerrain(Terrain):
    def __init__(self,seed,kind,mass=.1,tread=.18,terrain_scale=1.):
        self.mass=mass;self.tread=tread
        super().__init__(seed,kind)
        self.z*=terrain_scale

    def function(self,x,y):
        envelope=np.clip((x-.25)/.4,0.,1.);envelope=envelope**2*(3-2*envelope)
        if self.kind in ('up20','down20','up40','down40','up60','down60','flat'):return np.zeros_like(x+y)
        if self.kind=='washboard':z=.009*np.sin(24*x)+.004*np.sin(41*x+2*y)
        elif self.kind=='potholes':z=-.025*np.exp(-((x-.9)/.13)**2-((y-.10)/.17)**2)-.022*np.exp(-((x-1.6)/.16)**2-((y+.10)/.16)**2)
        elif self.kind=='bumps':z=.025*np.exp(-((x-.9)/.16)**2-((y-.09)/.18)**2)+.02*np.exp(-((x-1.6)/.15)**2-((y+.09)/.18)**2)
        elif self.kind=='cross':z=.08*y+.012*np.sin(9*x)*np.sin(9*y)
        elif self.kind=='slope':z=.08*np.minimum(x,2.)+.006*np.sin(13*x+4*y)
        else:z=self.amplitude*np.sin(self.wave*x+3*y+self.phase[0])+.009*np.sin(12*x-7*y+self.phase[1])
        return envelope*z

    def stairs(self,p):
        if self.kind not in ('up20','down20','up40','down40','up60','down60','mixed20','mixed40','mixed60'):return np.zeros(len(p))
        rise=.06 if '60' in self.kind else .04 if '40' in self.kind else .02
        start=1.1 if self.kind.startswith('mixed') else .45
        level=np.clip(np.floor((p[:,0]-start)/self.tread)+1,0,4)
        if self.kind.startswith('down'):level=4-level
        return np.where(np.abs(p[:,1])<=1.,level*rise,0.)

    def query(self,p):
        return np.maximum(super().query(p),self.stairs(p)) if self.kind.startswith(('up','down','mixed')) else super().query(p)

    def model(self):
        root=resource_root();tree=ET.parse(root/'models/full/locomotion-articulated.xml');xml=tree.getroot()
        for mesh in xml.findall('./asset/mesh'):
            mesh.set('file',str(root/'models/full'/mesh.get('file')))
        world=xml.find('worldbody')
        for g in list(world.findall('geom')):
            if g.get('type')=='plane':world.remove(g)
        asset=xml.find('asset');low=float(self.z.min());span=max(.001,float(np.ptp(self.z)))
        ET.SubElement(asset,'hfield',name='terrain',nrow='151',ncol='401',size=f'4 3 {span} .2')
        ET.SubElement(world,'geom',name='terrain',type='hfield',hfield='terrain',pos=f'0 0 {low}',contype='2',conaffinity='5',group='5')
        if self.kind.startswith(('up','down','mixed')):
            start=1.1 if self.kind.startswith('mixed') else .45;rise=.06 if '60' in self.kind else .04 if '40' in self.kind else .02
            bounds=[-4.,start,start+self.tread,start+2*self.tread,start+3*self.tread,4.]
            for i,(left,right) in enumerate(zip(bounds,bounds[1:])):
                top=(4-i if self.kind.startswith('down') else i)*rise
                ET.SubElement(world,'geom',name=f'stair_{i}',type='box',pos=f'{(left+right)/2} 0 {top-.25}',size=f'{(right-left)/2} 1 .25',contype='2',conaffinity='5',group='5')
        obj=ET.SubElement(world,'body',name='payload',pos='-.09 0 .284')
        ET.SubElement(obj,'freejoint',name='payload_free')
        inertia=self.mass/12*np.array([.03**2+.04**2,.04**2+.04**2,.04**2+.03**2])
        ET.SubElement(obj,'inertial',pos='0 0 0',mass=str(self.mass),diaginertia=' '.join(map(str,inertia)))
        ET.SubElement(obj,'geom',name='payload',type='box',size='.02 .015 .02',mass=str(self.mass),friction='2 .05 .005',contype='4',conaffinity='3',group='4')
        model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));model.hfield_data[:]=((self.z-low)/span).ravel()
        return model


def run(kind='rough',seed=47,mass=.1,parameters=None,out=None,seconds=None,*,clamped=False,yaw=0.,start_x=0.,tread=.18,terrain_scale=1.,drive_speed=.5,turn_rate=0.,crouch=0.,physics_hz=1000,stair_profile=None):
    if physics_hz < 50 or physics_hz % 50:
        raise ValueError('Physics rate must be an integer multiple of the 50 Hz controller')
    terrain=LoadedTerrain(seed,kind,mass,tread,terrain_scale);model=terrain.model();data=mujoco.MjData(model);adapter=JointAdapter(model)
    model.opt.timestep=1./physics_hz
    c=CompliantController(resource_root(),parameters,stair_profile=stair_profile)
    names=[b['joint']['name'] for b in c.spec['bodies'].values() if b.get('parent') is not None]
    qa=np.array([model.jnt_qposadr[model.joint(n).id] for n in names]);va=np.array([model.jnt_dofadr[model.joint(n).id] for n in names])
    wheel_names=('front_left','front_right','rear_left','rear_right')
    wheels=[model.body(n+'_wheel').id for n in wheel_names]
    wheel_geoms=[model.geom(n+'_wheel_contact_0').id for n in wheel_names];chassis=model.body('chassis').id
    payload=model.body('payload').id;payload_geom=model.geom('payload').id;pj=model.joint('payload_free').id;pq=model.jnt_qposadr[pj];pv=model.jnt_dofadr[pj]
    initial=float(terrain.query(np.array([[start_x,0.]]))[0]);data.qpos[0]=start_x;data.qpos[2]+=initial
    data.qpos[3:7]=[np.cos(yaw/2),0.,0.,np.sin(yaw/2)]
    data.qpos[pq:pq+3]=[start_x-.09*np.cos(yaw),-.09*np.sin(yaw),.284+initial]
    cargo_yaw=yaw+(np.pi/2 if clamped else 0.)
    data.qpos[pq+3:pq+7]=[np.cos(cargo_yaw/2),0.,0.,np.sin(cargo_yaw/2)]
    stairs=kind.startswith(('up','down','mixed'));duration=seconds or (35. if stairs else 8.)
    dt=model.opt.timestep;substeps=round(.02/dt);samples=[];trace=[];last_v=None;last_deck_v=None;filtered=np.zeros(3);last_filtered=np.zeros(3)
    max_force=0.;contact_force=np.zeros(6);cleared_at=None;lost=False;max_slip=0.;baseline_local=None;saturation=0;total_controls=0
    started=time.monotonic();result=None;ground=[initial]*4;cargo_supported=False;bilateral=False;distill_previous=np.zeros(16)
    for step in range(round(duration/dt)+1):
        mujoco.mj_forward(model,data);t=step*dt;rot=data.xmat[chassis].reshape(3,3)
        world_omega=rot@data.qvel[3:6];deck_offset=rot@np.array([-.09,0.,.06]);deck_v=data.qvel[:3]+np.cross(world_omega,deck_offset)
        cargo_v=data.qvel[pv:pv+3].copy();cargo_local=rot.T@(data.xpos[payload]-data.xpos[chassis])
        if t>=1. and baseline_local is None:baseline_local=cargo_local.copy()
        if baseline_local is not None:
            slip=float(np.linalg.norm((cargo_local-baseline_local)[:2]));max_slip=max(max_slip,slip)
            lost=lost or cargo_local[2]<.02 or abs(cargo_local[1])>.12 or not -.17<cargo_local[0]<-.025
        acceleration=np.zeros(3);deck_acc=np.zeros(3);jerk=np.zeros(3)
        if last_v is not None:
            acceleration=(cargo_v-last_v)/dt;deck_acc=(deck_v-last_deck_v)/dt
            filtered+=(acceleration-filtered)*dt/(.015+dt)
            jerk=(filtered-last_filtered)/dt
            if t>=1.5:
                samples.append([t,*acceleration,*filtered,*jerk,*deck_acc,rot[2,2],cargo_local[2],data.qpos[0],data.qvel[0], data.qpos[2]-np.mean(ground)-.2192, float(cargo_supported),float(bilateral)])
            last_filtered=filtered.copy()
        last_v=cargo_v;last_deck_v=deck_v.copy()
        cargo_supported=False;pad_forces=np.zeros(2);wheel_normal_forces=np.zeros(4)
        for index,ct in enumerate(data.contact):
            mujoco.mj_contactForce(model,data,index,contact_force)
            for wheel_index,wheel_geom in enumerate(wheel_geoms):
                if wheel_geom in ct.geom:wheel_normal_forces[wheel_index]+=max(0.,float(contact_force[0]))
            if payload_geom in ct.geom and t>=1.5:
                max_force=max(max_force,float(contact_force[0]))
                other=ct.geom[0] if ct.geom[1]==payload_geom else ct.geom[1]
                if model.geom_bodyid[other]==chassis and contact_force[0]>mass*9.81*.05:cargo_supported=True
                for side,name in enumerate(['cargo_slide_-1','cargo_slide_1']):
                    if model.geom_bodyid[other]==model.body(name).id:pad_forces[side]+=max(0.,float(contact_force[0]))
        bilateral=bool(np.min(pad_forces)>.25)
        if step%substeps==0:
            angle=np.arctan2(rot[1,0],rot[0,0]);cy,sy=np.cos(angle),np.sin(angle);r2=np.array([[cy,-sy],[sy,cy]])
            def scan(xy):return terrain.query(xy@r2.T+data.qpos[:2]).tolist()
            ground=[]
            for xyz in data.xpos[wheels]:
                gid=np.zeros(1,dtype=np.int32);distance=mujoco.mj_ray(model,data,xyz+[0,0,1],np.array([0.,0.,-1.]),np.array([0,0,0,0,0,1],dtype=np.uint8),1,-1,gid)
                ground.append(float(xyz[2]+1-distance) if distance>=0 else float(terrain.query(xyz[None,:2])[0]))
            finish_x=(1.1 if kind.startswith('mixed') else .45)+3*tread+STAIR_CLEARANCE_M
            if stairs and cleared_at is None and min(data.xpos[wheels,0])>finish_x:cleared_at=t
            moving=t>=1. and (cleared_at is None if stairs else t<7.)
            speed=(.16 if stairs else drive_speed) if moving else 0.
            state=dict(robot_id='Sai_Agent_001',physics_owner='Godot/Jolt',time=t,q=data.qpos[qa].tolist(),v=data.qvel[va].tolist(),
                base_position=data.qpos[:3].tolist(),base_rotation_columns=rot.T.tolist(),base_linear_world=data.qvel[:3].tolist(),base_angular_world=world_omega.tolist(),
                command=[speed,turn_rate if moving else 0.,crouch],terrain_heights=scan(SCAN),terrain_path_heights=scan(PATH),terrain_edge_heights=scan(DENSE),
                wheel_ground_heights=ground,wheel_positions=data.xpos[wheels].tolist(),stair_course=stairs)
            result=c.command(state)
            policy_observation=result.get('policy_observation')
            if policy_observation is not None and len(policy_observation)==104:
                distill_observation=np.asarray(policy_observation,dtype=np.float32)
            else:
                distill_observation=phase_free_observation(state,state['command'],distill_previous)
            distill_target=result.get('distill_target_leg',result['target_leg'])
            distill_action=actions_from_targets_numpy(distill_target,state['command'])
            distill_previous=distill_action
            wheel_gaps=[position[2]-height-.048 for position,height in zip(data.xpos[wheels],ground)]
            trace.append(dict(time=t,position=data.qpos[:3].tolist(),base_rotation=rot.tolist(),
                base_linear_world=data.qvel[:3].tolist(),base_angular_world=world_omega.tolist(),
                cargo_position=data.xpos[payload].tolist(),cargo_local=cargo_local.tolist(),cargo_velocity=cargo_v.tolist(),
                cargo_acceleration=acceleration.tolist(),cargo_filtered_acceleration=filtered.tolist(),cargo_jerk=jerk.tolist(),deck_acceleration=deck_acc.tolist(),
                speed=float((rot.T@data.qvel[:3])[0]),command=speed,stage=result['stage'],upright=float(rot[2,2]),qpos=data.qpos.tolist(),
                q=state['q'],v=state['v'],wheels_supported=int(np.sum(wheel_normal_forces>.05)),
                wheel_gap_supported=int(sum(gap<=.006 for gap in wheel_gaps)),wheel_normal_forces=wheel_normal_forces.tolist(),
                policy_action=result.get('policy_action'),target_leg=result.get('target_leg'),wheel_positions=data.xpos[wheels].tolist(),wheel_ground_heights=ground,
                policy_observation=result.get('policy_observation'),
                skill_intensity=result.get('skill_intensity',0.),suspension_offset_m=result.get('suspension_offset_m'),
                distill_observation=distill_observation.tolist(),distill_action=distill_action.tolist()))
        apply_impedance(adapter,data,result)
        if step%substeps==0 and trace:
            trace[-1]['motor_torque']=data.ctrl[:16].tolist()
            trace[-1]['leg_kp']=result.get('leg_kp')
            trace[-1]['leg_kd']=result.get('leg_kd')
            trace[-1]['leg_feedforward']=result.get('leg_feedforward')
        if clamped:
            cj=model.joint('cargo_drive').id;act=model.actuator('cargo_drive_motor').id
            data.ctrl[act]=np.clip(4*(.067/.01909859317102744-data.qpos[model.jnt_qposadr[cj]])-.06*data.qvel[model.jnt_dofadr[cj]],-.12,.12)
        saturation+=int(np.any(np.abs(data.ctrl[np.array(adapter.aids)[LEG_AXES]])>=7.999));total_controls+=1
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all() or rot[2,2]<.6:break
        if stairs and cleared_at is not None and t-cleared_at>=3.:break
    a=np.array(samples);moving=a[(a[:,0]<(cleared_at or 7.) if stairs else a[:,0]<7.)]
    if not len(moving):raise RuntimeError('Episode lacks settled motion measurements')
    norm=np.linalg.norm(moving[:,1:4],axis=1);smooth=np.linalg.norm(moving[:,4:7],axis=1);j=np.linalg.norm(moving[:,7:10],axis=1)
    moving_trace=[r for r in trace if abs(r['command'])>0 and r['time']>=1.5]
    m=dict(cargo_accel_rms=float(np.sqrt(np.mean(smooth**2))),cargo_accel_p95=float(np.quantile(smooth,.95)),cargo_accel_peak=float(norm.max()),
        cargo_jerk_rms=float(np.sqrt(np.mean(j**2))),deck_accel_rms=float(np.sqrt(np.mean(np.linalg.norm(moving[:,10:13],axis=1)**2))),
        contact_force_peak_N=max_force,max_cargo_slip_m=max_slip,cargo_lost=bool(lost),min_upright=float(a[:,13].min()),
        height_error_mean=float(np.mean(moving[:,17])),height_error_p95=float(np.quantile(np.abs(moving[:,17]),.95)),cargo_supported_fraction=float(np.mean(moving[:,18])),bilateral_clamp_fraction=float(np.mean(moving[:,19])),
        speed=float(np.sign(.16 if stairs else drive_speed)*np.mean([r['speed'] for r in moving_trace])),distance=float(data.qpos[0]),cleared_at=cleared_at,
        completed=bool(cleared_at is not None if stairs else t>=duration-dt),duration=t,saturation_fraction=saturation/total_controls)
    report=dict(drive_speed=drive_speed,turn_rate=turn_rate,crouch=crouch,kind=kind,seed=seed,mass=mass,clamped=clamped,yaw=yaw,start_x=start_x,tread=tread,terrain_scale=terrain_scale,parameters=None if parameters is None else list(parameters),metrics=m,wall_seconds=time.monotonic()-started,
        physics_hz=1/dt,controller_hz=50,comfort_filter_tau_s=.015,payload='freejoint, existing tray contacts; physical clamp' if clamped else 'freejoint, existing tray contacts, open clamp; no object constraint')
    if out:
        out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        np.savez_compressed(out/'physics.npz',samples=a,columns=np.array(['t','cargo_ax','cargo_ay','cargo_az','filtered_ax','filtered_ay','filtered_az','jerk_x','jerk_y','jerk_z','deck_ax','deck_ay','deck_az','upright','cargo_local_z','base_x','base_vx','height_error','cargo_supported','bilateral_clamp']));(out/'trace.json').write_text(json.dumps(trace)+'\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--kind',default='rough');p.add_argument('--seed',type=int,default=47)
    p.add_argument('--mass',type=float,default=.1);p.add_argument('--parameters',type=float,nargs=6)
    p.add_argument('--physics-hz',type=int,default=1000);p.add_argument('--seconds',type=float)
    p.add_argument('--clamped',action='store_true');p.add_argument('--drive-speed',type=float,default=.5)
    p.add_argument('--stair-profile',type=Path)
    a=p.parse_args()
    print(json.dumps(run(a.kind,a.seed,a.mass,a.parameters,a.out,seconds=a.seconds,
                         clamped=a.clamped,drive_speed=a.drive_speed,
                         physics_hz=a.physics_hz,stair_profile=a.stair_profile),indent=2),flush=True)

if __name__=='__main__':main()
