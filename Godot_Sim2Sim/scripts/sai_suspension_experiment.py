"""Physical low-dimensional suspension training with frozen actors and held-out terrain."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from sai_agent.paths import resource_root
from sai_agent.runtime import JointAdapter
from sim2sim.sai_controller import MotionController
from sim2sim.sai_suspension import Suspension
from sim2sim.sai_terrain import EDGE_X, EDGE_Y

ROOT=Path(__file__).resolve().parents[1]
LEGS=('front_left','front_right','rear_left','rear_right')
SCAN=np.array([(x,y) for x in np.arange(8)*.18-.36 for y in [-.24,0.,.24]])
PATH=np.array([(x,y) for x in [-.18,0.,.18,.36,.54] for y in [-.16,0.,.16]])
DENSE=np.array([(x,y) for x in EDGE_X for y in EDGE_Y])


class Terrain:
    """Shared sampled terrain; MuJoCo heightfield and Jolt triangles use these vertices."""
    def __init__(self, seed, kind='rough'):
        self.seed,self.kind=seed,kind
        rng=np.random.default_rng(seed)
        self.phase=rng.uniform(-np.pi,np.pi,3)
        self.amplitude=rng.uniform(.008,.018)
        self.wave=rng.uniform(4.,7.)
        self.x=np.linspace(-4,4,401)
        self.y=np.linspace(-3,3,151)
        xx,yy=np.meshgrid(self.x,self.y)
        self.z=self.function(xx,yy)

    def function(self,x,y):
        if self.kind=='flat':return np.zeros_like(x+y)
        if self.kind=='ramp':return .10*np.clip(x-.3,0.,2.)
        if self.kind=='cross':return .07*y + .010*np.sin(5*x)*np.sin(5*y)
        envelope=np.clip((x-.30)/.45,0.,1.)
        envelope=envelope**2*(3-2*envelope)
        return envelope*(self.amplitude*np.sin(self.wave*x+2*y+self.phase[0])
                         +.007*np.sin(10*x-3*y+self.phase[1]))

    def query(self, points):
        # Piecewise-linear support heights on the shared collision grid.
        x=np.clip((points[:,0]+4)/.02,0,399.999999)
        y=np.clip((points[:,1]+3)/.04,0,149.999999)
        i,j=x.astype(int),y.astype(int);u,v=x-i,y-j
        a,b,c,d=self.z[j,i],self.z[j,i+1],self.z[j+1,i],self.z[j+1,i+1]
        return np.where(u+v<=1,a*(1-u-v)+b*u+c*v,b*(1-v)+c*(1-u)+d*(u+v-1))

    def model(self):
        root=resource_root()
        tree=ET.parse(root/'models/full/locomotion-articulated.xml')
        xml=tree.getroot()
        compiler=xml.find('compiler')
        if compiler is not None:compiler.set('meshdir',str(root/'models/full/assets'))
        for mesh in xml.findall('./asset/mesh'):
            f=mesh.get('file')
            if f and not Path(f).is_absolute():
                file=root/'models/full'/f
                if not file.exists():file=root/'models/full/assets'/f
                mesh.set('file',str(file))
        world=xml.find('worldbody')
        for g in list(world.findall('geom')):
            if g.get('type')=='plane':world.remove(g)
        asset=xml.find('asset')
        if asset is None:asset=ET.SubElement(xml,'asset')
        low=float(self.z.min());span=max(.001,float(np.ptp(self.z)))
        ET.SubElement(asset,'hfield',name='terrain',nrow='151',ncol='401',size=f'4 3 {span} .2')
        ET.SubElement(world,'geom',name='terrain',type='hfield',hfield='terrain',pos=f'0 0 {low}',contype='2',conaffinity='5',group='5')
        model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
        model.hfield_data[:]=((self.z-low)/span).ravel()
        return model


def metrics(rows):
    steady=[r for r in rows if 2.<=r['time']<6.]
    v=np.array([r['velocity'] for r in steady])
    angle=np.array([r['angular'][:2] for r in steady])
    gap=np.array([r['gap'] for r in steady])
    contact=np.array([r['supported'] for r in steady])
    active=np.array([r['command'][0] for r in steady])
    acceleration=np.diff(v[:,2])/.02
    score=dict(speed=float(v[:,0].mean()),speed_mae=float(np.abs(v[:,0]-active).mean()),
        angular_rms=float(np.sqrt(np.mean(angle**2))),vertical_accel_rms=float(np.sqrt(np.mean(acceleration**2))),
        gap_rms=float(np.sqrt(np.mean(np.maximum(gap-.003,0.)**2))),gap_p95=float(np.quantile(gap,.95)),
        four_contact_fraction=float(np.mean(contact==4)),mean_contacts=float(contact.mean()),
        min_upright=float(min(r['upright'] for r in rows)),stair_fraction=float(np.mean([r['stage']=='stairs' for r in steady])),
        distance=float(rows[-1]['position'][0]-rows[0]['position'][0]),samples=len(rows))
    score['loss']=(20*score['speed_mae']+2*score['angular_rms']+.03*score['vertical_accel_rms']
                   +60*score['gap_rms']+.20*(4-score['mean_contacts'])+50*(score['min_upright']<.8))
    return score


def rollout(seed=47,kind='rough',parameters=None,mode='gate',seconds=7.,speed=.5,yaw=0.,crouch=0.,trace=False):
    terrain=Terrain(seed,kind)
    model=terrain.model();data=mujoco.MjData(model)
    data.qpos[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    data.qpos[2]+=float(terrain.query(np.array([[0.,0.]]))[0])
    c=MotionController(resource_root());c.suspension=Suspension(parameters) if parameters is not None else None
    if mode=='original':
        c._step_in_wheel_path=lambda state,honor_course=True: bool(np.ptp(state['terrain_path_heights'])>.004)
    adapter=JointAdapter(model)
    names=[b['joint']['name'] for b in c.spec['bodies'].values() if b.get('parent') is not None]
    qadr=np.array([model.jnt_qposadr[model.joint(n).id] for n in names]);vadr=np.array([model.jnt_dofadr[model.joint(n).id] for n in names])
    wheels=[model.body(n+'_wheel').id for n in LEGS];chassis=model.body('chassis').id
    rows=[]
    for k in range(round(seconds/.02)+1):
        mujoco.mj_forward(model,data)
        t=k*.02;rotation=data.xmat[chassis].reshape(3,3)
        angle=np.arctan2(rotation[1,0],rotation[0,0]);cy,sy=np.cos(angle),np.sin(angle)
        rot=np.array([[cy,-sy],[sy,cy]])
        def scan(p):return terrain.query(p@rot.T+data.qpos[:2]).tolist()
        wheel_xyz=data.xpos[wheels].copy()
        # Ground-only rays determine actual clearance independently of interpolation.
        ground=[]
        for point in wheel_xyz:
            gid=np.zeros(1,dtype=np.int32)
            distance=mujoco.mj_ray(model,data,point+np.array([0,0,1.]),np.array([0.,0.,-1.]),np.array([0,0,0,0,0,1],dtype=np.uint8),1,-1,gid)
            if distance<0:raise RuntimeError(f'Wheel left terrain bounds t={t} xyz={point.tolist()} base={data.qpos[:3].tolist()}')
            ground.append(float(point[2]+1-distance))
        vx=speed if 1.<=t<6. else 0.
        state=dict(robot_id='Sai_Agent_001',physics_owner='Godot/Jolt',time=t,q=data.qpos[qadr].tolist(),v=data.qvel[vadr].tolist(),
            base_position=data.qpos[:3].tolist(),base_rotation_columns=rotation.T.tolist(),base_linear_world=data.qvel[:3].tolist(),
            base_angular_world=(rotation@data.qvel[3:6]).tolist(),command=[vx,0.,crouch],
            terrain_heights=scan(SCAN),terrain_path_heights=scan(PATH),terrain_edge_heights=scan(DENSE),wheel_ground_heights=ground)
        result=c.command(state)
        contact_bodies={int(model.geom_bodyid[g]) for ct in data.contact for g in ct.geom if ct.dist<.001}
        rows.append(dict(time=t,position=data.qpos[:3].tolist(),velocity=(rotation.T@data.qvel[:3]).tolist(),angular=data.qvel[3:6].tolist(),
            upright=float(rotation[2,2]),gap=(wheel_xyz[:,2]-np.array(ground)-.048).tolist(),supported=len(contact_bodies.intersection(wheels)),
            stage=result['stage'],command=state['command'],effective_speed=result['policy_observation'][9]))
        for _ in range(round(.02/model.opt.timestep)):
            adapter.apply(data,np.asarray(result['target_leg']));mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all():raise RuntimeError('Non-finite physics')
    report=dict(seed=seed,kind=kind,mode=mode,parameters=parameters,metrics=metrics(rows),physics='CPU MuJoCo full articulated')
    if trace:report['rows']=rows
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--train',action='store_true');p.add_argument('--seed',type=int,default=47);p.add_argument('--kind',default='rough')
    p.add_argument('--parameters',nargs=3,type=float);p.add_argument('--mode',default='gate');p.add_argument('--profile',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if not a.train:
        parameters=json.loads(a.profile.read_text())['parameters'] if a.profile else a.parameters
        result=rollout(a.seed,a.kind,parameters,a.mode,trace=True)
        (a.out/'rollout.json').write_text(json.dumps(result)+'\n');print(result['metrics'],flush=True);return
    # Predeclared training set; separate held-out seeds are never used to select parameters.
    seeds=[47,71,109];rng=np.random.default_rng(20260915)
    candidates=[[0.,0.,.08],[.7,.7,.08],[1.,.7,.08]]
    mean=np.array([.7,.7,.08]);std=np.array([.25,.25,.04]);best=None
    started=time.monotonic();records=[]
    for generation in range(3):
        if generation:
            candidates=[best['parameters']]+[np.clip(rng.normal(mean,std),[0,0,.02],[1.2,1.2,.30]).tolist() for _ in range(5)]
        batch=[]
        for parameters in candidates:
            reports=[rollout(seed,parameters=parameters) for seed in seeds]
            loss=float(np.mean([r['metrics']['loss'] for r in reports]))
            entry=dict(index=len(records),generation=generation,parameters=parameters,loss=loss,reports=reports)
            records.append(entry);batch.append(entry)
            with (a.out/'trials.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            print(json.dumps({k:entry[k] for k in ('index','generation','parameters','loss')}),flush=True)
        best=min(records,key=lambda r:r['loss']);elite=sorted(batch,key=lambda r:r['loss'])[:3]
        mean=np.mean([e['parameters'] for e in elite],axis=0);std=np.maximum(np.std([e['parameters'] for e in elite],axis=0),[.08,.08,.015])
    result=dict(schema_version=1,id='sai-suspension-20260915',parameters=best['parameters'],training_loss=best['loss'],
        method='bounded cross-entropy parameter optimization in full CPU MuJoCo; frozen ONNX actors',training_seeds=seeds,
        trials=len(records),elapsed_seconds=time.monotonic()-started,status='candidate_unvalidated',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'candidate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
