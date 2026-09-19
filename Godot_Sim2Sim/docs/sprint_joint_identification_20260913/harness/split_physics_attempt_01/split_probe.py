"""Check whether passive joint translation improves the native response model."""
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np
import mujoco
import torch
from sim2sim.paths import load_robot_json
from sim2sim.research.queue import atomic_json
sys.path.insert(0,str(Path('scripts').resolve()))
from sprint_gpu_world import GpuWorld
sys.path.insert(0,str(Path('results/sprint_contact_calibration_20260913').resolve()))
from contact_probe import initialize

R=Path('results/sprint_joint_identification_20260913')


def main():
    if Path(os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR','/nonexistent')).resolve()!=R.resolve():
        raise RuntimeError('Active-budget supervision required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    out=R/'split_physics';out.mkdir(exist_ok=False)
    summary=json.loads(Path('results/sprint_stop_state_20260912/native_joint_fd/suite/summary.json').read_text())
    selected=sorted([e for e in summary['episodes'] if e['case']=='sprint_alternate' and 927000<=e['seed']<=927015],key=lambda e:e['seed'])
    assert len(selected)==16
    rows=[json.loads(Path(e['trace']).read_text())['rows'] for e in selected]
    settings=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text())['walk']
    action=torch.tensor([[r[i]['action'] for r in rows] for i in range(100)],device='cuda')
    native_pos=np.array([[r[i]['body']['base_pos'] for r in rows] for i in range(101)])
    native_vel=np.array([[r[i]['body']['base_linvel'] for r in rows] for i in range(101)])
    native_q=np.array([[r[i]['raw']['q'] for r in rows] for i in range(101)])
    native_vec=np.load(R/'joint_residuals.npz')['foot_rigid'][:101]
    profiles=[('rigid','source',None),('uniform20k','two_tick',(20000.,200.)),('velocity_only','two_tick',(1.,200.)),('split_projection','two_tick',(1.,200.))]
    protocol=dict(profiles=profiles,seeds=[e['seed'] for e in selected],train_seed_count=8,validation_seed_count=8,
        physics_hz=200,decision_hz=50,horizon_seconds=2,
        gate='Same 20% landing/moving response improvement, no component/standing >10% regression, both splits; also >=20% reduction of actual foot-vs-rigid-manifold vector error on landing and moving. Pre-contact joint RMS <=baseline+1e-5 rad.',
        response_normalization=dict(velocity=.05,joint=.02,position=.002),
        source='Frozen a402 native recorded actions; no policy inference, training or game physics edits',
        assumptions='Extra equalities are an experimental approximation of translation compliance, not an identified Jolt solver.',
        rationale='A bounded architecture test: remove equality positional acceleration (k=0,b=200), with/without generalized mass-weighted pose-only correction alpha=1-(1-.35)^2=.5775. Uses start-of-step mass, simultaneous linear projection, no contact position correction; not a full Jolt port. All original physical/manifold gates unchanged.',
        reference='https://raw.githubusercontent.com/godotengine/godot/ed1daf0bf/thirdparty/jolt_physics/Jolt/Physics/Constraints/ConstraintPart/PointConstraintPart.h',
        code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'split_proxy.py',Path('scripts/sprint_gpu_world.py'),Path('src/sim2sim/research/joint_compliance.py')]})
    atomic_json(out/'protocol.json',protocol)
    cfg=load_robot_json(Path('robots/microduck_ball_stand_fix.json'))
    rigid_model=mujoco.MjModel.from_xml_path(cfg['mjcf']);rigid_data=mujoco.MjData(rigid_model)
    base=mujoco.mj_name2id(rigid_model,mujoco.mjtObj.mjOBJ_BODY,'trunk_base')
    bj=next(i for i in range(rigid_model.njnt) if rigid_model.jnt_bodyid[i]==base and rigid_model.jnt_type[i]==0)
    rqa=rigid_model.jnt_qposadr[bj];rqi=[rigid_model.jnt_qposadr[rigid_model.actuator_trnid[i,0]] for i in range(14)]
    names=['ankle_left','ankle_right'];rfeet=[mujoco.mj_name2id(rigid_model,mujoco.mjtObj.mjOBJ_BODY,n) for n in names]
    results={}
    for name,contact,compliance in profiles:
        from contextlib import nullcontext
        from unittest.mock import patch
        from sim2sim.research.joint_compliance import add_joint_compliance_direct
        sys.path.insert(0,str(R.resolve()))
        from split_proxy import SplitWorld
        special=name in ['velocity_only','split_projection']
        def velocity_connections(spec,godot_spec,stiffness,damping):
            result=add_joint_compliance_direct(spec,godot_spec,stiffness,damping)
            for eq in spec.equalities:eq.solref=[0.,-200.]
            result.update(solref=[0.,-200.],stiffness=0.,scope='Prototype velocity damping only; pose projection separately applied')
            return result
        context=patch('sim2sim.research.joint_compliance.add_joint_compliance_direct',velocity_connections) if special else nullcontext()
        with context:
            w=(SplitWorld if special else GpuWorld)(16,settings,contact=contact,feet='jolt',graphs=True,velocity_observer='joint_fd',joint_compliance_direct=compliance)
        if special:w.prepare_projection(1-(1-.35)**2 if name=='split_projection' else 0.)
        try:
            initialize(w,rows)
            if special and name=='split_projection':
                atomic_json(out/'projection_canary.json',w.canary())
                initialize(w,rows)
            state=torch.empty((101,16,13),device='cuda');joint=torch.empty((101,16,14),device='cuda');qpos=torch.empty((101,16,w.model.nq),device='cuda')
            start=time.monotonic()
            with torch.inference_mode():
                for i in range(101):
                    pos,quat,ang,vel=w.state();state[i]=torch.cat((pos,quat,ang,vel),dim=-1);joint[i]=w.qpos[:,w.qi];qpos[i]=w.qpos
                    if i<100:w.step(action[i])
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            if special and int(w.solve_info)!=0:raise RuntimeError('Projection linear solve failed')
            state=state.cpu().numpy();joint=joint.cpu().numpy();qpos=qpos.cpu().numpy()
            if not np.isfinite(qpos).all():raise RuntimeError('Nonfinite compliant proxy')
            own_data=mujoco.MjData(w.model);feet=[mujoco.mj_name2id(w.model,mujoco.mjtObj.mjOBJ_BODY,n) for n in names]
            vectors=np.empty((101,16,2,3))
            for i in range(101):
                for j in range(16):
                    own_data.qpos[:]=qpos[i,j];mujoco.mj_kinematics(w.model,own_data)
                    rigid_data.qpos[rqa:rqa+7]=state[i,j,:7];rigid_data.qpos[rqi]=joint[i,j];mujoco.mj_kinematics(rigid_model,rigid_data)
                    vectors[i,j]=own_data.xipos[feet]-rigid_data.xipos[rfeet]
            error=dict(velocity=np.mean((state[:,:,10:13]-native_vel)**2,axis=-1)/.05**2,
                       joint=np.mean((joint-native_q)**2,axis=-1)/.02**2,
                       position=np.mean((state[:,:,:3]-native_pos)**2,axis=-1)/.002**2)
            manifold_error=np.mean((vectors-native_vec)**2,axis=(-2,-1))
            metrics={};t=np.arange(101)*.02
            for split,ids in [('train',slice(0,8)),('validation',slice(8,16))]:
                metrics[split]={}
                for phase,(lo,hi) in [('landing',(.04,.3)),('standing',(.3,1.)),('moving',(1.,1.3))]:
                    mask=(t>=lo-1e-8)&(t<hi-1e-8)
                    values={k:float(v[mask,ids].mean()) for k,v in error.items()}
                    metrics[split][phase]=dict(components=values,objective=float(np.mean(list(values.values()))),
                        manifold_mse_m2=float(manifold_error[mask,ids].mean()))
            result=dict(metrics=metrics,elapsed_physics_s=elapsed,nq=w.model.nq,nv=w.model.nv,neq=w.model.neq,nu=w.model.nu,
                compliance=w.joint_compliance,precontact_joint_rms=float(np.sqrt(np.mean((joint[1]-native_q[1])**2))),
                minimum_body_height=float(state[:,:,2].min()))
            np.savez_compressed(out/(name+'.npz'),states=state,joints=joint,qpos=qpos,foot_manifold_vector=vectors)
            atomic_json(out/(name+'.json'),result);results[name]=result
            print(json.dumps(dict(profile=name,**result)),flush=True)
        finally:w.close()
    decisions={};baseline=results['rigid']
    for name,result in results.items():
        if name=='rigid':continue
        checks=[]
        for split in ['train','validation']:
            for phase in ['landing','moving','standing']:
                a=baseline['metrics'][split][phase];b=result['metrics'][split][phase]
                ratio=b['objective']/a['objective'];manifold=b['manifold_mse_m2']/a['manifold_mse_m2']
                component=max(b['components'][k]/a['components'][k] for k in a['components'])
                checks.append(dict(split=split,phase=phase,objective_ratio=ratio,manifold_ratio=manifold,max_component_ratio=component,
                    passed=bool(ratio<=(1.1 if phase=='standing' else .8) and component<=1.1 and manifold<=(1.1 if phase=='standing' else .8))))
        precontact=result['precontact_joint_rms']<=baseline['precontact_joint_rms']+1e-5
        decisions[name]=dict(checks=checks,precontact_passed=precontact,passed=precontact and all(c['passed'] for c in checks))
    atomic_json(out/'completed.json',dict(results=results,decisions=decisions,eligible=[k for k,v in decisions.items() if v['passed']],completed=True))


if __name__=='__main__':main()
