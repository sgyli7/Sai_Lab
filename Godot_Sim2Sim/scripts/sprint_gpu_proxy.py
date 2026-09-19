"""Isolated GPU training-domain experiment; run with microduck_rl's venv.

The optional torque proxy matches the game's motor torque expression, not its
contact solver or articulated inertia. No Godot physics or upstream files change.
Both GPU physics sampling and network updates stay in this bounded process.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import get_base_metadata, attach_metadata_to_onnx


class JoltTorqueProxy(IdealPdActuator):
    def edit_spec(self,spec,target_names):
        # The source XML already contains position actuators. Reuse their
        # identity/order, as BAM does, instead of adding duplicate motors.
        found=set()
        for actuator in spec.actuators:
            target=str(actuator.target)
            if target not in target_names:continue
            actuator.set_to_motor();actuator.gear=[1.,0.,0.,0.,0.,0.]
            actuator.forcelimited=False;actuator.ctrllimited=False
            joint=spec.joint(target);joint.armature=.0018
            joint.damping=np.zeros((3,1));joint.frictionloss=0.
            self._mjs_actuators.append(actuator);found.add(target)
        if found!=set(target_names):raise ValueError('Unexpected source actuator targets')

    def compute(self,cmd):
        drive=(.55*(cmd.position_target-cmd.pos)).clamp(-1.75*.36601349688984386,1.75*.36601349688984386)
        return drive-.053*cmd.vel-.0048*torch.tanh(cmd.vel/.05)


@dataclass(kw_only=True)
class JoltTorqueProxyCfg(IdealPdActuatorCfg):
    def build(self,entity,target_ids,target_names):
        return JoltTorqueProxy(self,entity,target_ids,target_names)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_game_inertia(model,spec_path):
    """Experimental post-compile tensors; Jolt's rotor map is not a solid body.

    MuJoCo's XML triangle check rejects these positive diagonal tensors. Never
    enable balanceinertia: it would silently replace the requested values.
    Apply before mjwarp.put_model and retain an explicit compatibility audit.
    """
    import mujoco
    spec=json.loads(Path(spec_path).read_text())
    bodies={b['name']:b for b in spec['bodies']}
    tensors={};rotation=np.empty(9)
    for joint in spec['joints']:
        armature=float(joint['armature'])
        if armature<=0:continue
        body=bodies[joint['body']]
        mujoco.mju_quat2Mat(rotation,np.array(body['iquat_wxyz']))
        axis=rotation.reshape(3,3).T@np.array(joint['axis_child_body'])
        axis=axis/np.linalg.norm(axis)
        inertia=tensors.get(body['name'],np.array(body['inertia'])).copy()
        inertia+=armature*axis**2
        tensors[body['name']]=np.maximum(inertia,inertia.max()/10.)
    def unique_id(kind,count,name):
        matches=[i for i in range(count) if (mujoco.mj_id2name(model,kind,i) or '').split('/')[-1]==name]
        if len(matches)!=1:raise ValueError(f'Expected one compiled {name}, got {matches}')
        return matches[0]
    report={}
    for name,inertia in tensors.items():
        index=unique_id(mujoco.mjtObj.mjOBJ_BODY,model.nbody,name);body=bodies[name]
        if not np.isclose(model.body_mass[index],body['mass'],rtol=1e-6,atol=1e-9):
            raise ValueError('Training/game body mass differs: '+name)
        model.body_inertia[index]=inertia
        model.body_iquat[index]=body['iquat_wxyz'];model.body_ipos[index]=body['ipos']
        report[name]=dict(index=index,inertia=inertia.tolist(),triangle_inequality=bool(2*inertia.max()<=inertia.sum()))
    for joint in spec['joints']:
        if joint['type']=='free':continue
        index=unique_id(mujoco.mjtObj.mjOBJ_JOINT,model.njnt,joint['name'])
        model.dof_armature[model.jnt_dofadr[index]]=0.
    data=mujoco.MjData(model);mujoco.mj_setConst(model,data);mujoco.mj_forward(model,data)
    maximum=max(float(np.max(np.abs(model.body_inertia[v['index']]-v['inertia']))) for v in report.values())
    if maximum>1e-12 or not np.isfinite(data.qacc).all():raise RuntimeError('Inertia patch did not survive constant recomputation')
    return dict(spec_sha256=sha(spec_path),bodies=report,max_abs_after_set_const=maximum,
                supported_physical_solid=False,scope='experimental Jolt diagonal rotor-map proxy; distinct contact solver')


def export_checked(runner,vec,output,name):
    import onnxruntime as ort
    runner.export_policy_to_onnx(str(output),name+'.onnx')
    path=output/(name+'.onnx')
    attach_metadata_to_onnx(str(path),get_base_metadata(vec.unwrapped,'sprint_gpu_proxy'))
    model=runner.alg.get_policy().as_onnx(verbose=False).cpu().eval()
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    if session.get_inputs()[0].shape!=[1,61] or session.get_outputs()[0].shape!=[1,14]:
        raise RuntimeError('GPU proxy changed the policy observation/action contract')
    inputs=np.random.default_rng(921999).normal(0,.5,(128,61)).astype(np.float32)
    inputs[:,3:6]=[0,0,-1]
    real=vec.unwrapped.observation_manager.compute()['actor'].detach().cpu().numpy()
    inputs=np.concatenate([inputs,real[:128]])
    error=0.
    with torch.inference_mode():
        for obs in inputs:
            expected=model(torch.from_numpy(obs[None])).numpy()
            actual=session.run(None,{session.get_inputs()[0].name:obs[None]})[0]
            error=max(error,float(np.max(np.abs(expected-actual))))
    report=dict(max_abs=error,passed=error<1e-5,cases=len(inputs),sha256=sha(path))
    (output/(name+'_parity.json')).write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise RuntimeError('GPU policy export failed numerical equivalence')
    path.with_suffix('.manifest.json').write_text(json.dumps(dict(
        sim2sim=dict(skill='sprint',use_stand_policy=False),
        status='experimental_gpu_source_requires_jolt_acceptance',source_precision='original_fp32',
        sha256=report['sha256']),indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['bam','jolt_torque','jolt_inertia'],required=True)
    p.add_argument('--jolt-spec',type=Path)
    p.add_argument('--fixed-inertia',action='store_true',help='Disable mass/COM/inertia/armature DR for a matched proxy control')
    p.add_argument('--envs',type=int,default=64);p.add_argument('--iterations',type=int,default=5)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--seed',type=int,default=921101)
    args=p.parse_args()
    if args.kind=='jolt_inertia' and args.jolt_spec is None:raise ValueError('An explicit frozen Godot spec is required')
    budget=os.environ.get('SIM2SIM_ACTIVE_BUDGET_DIR')
    if not budget or not args.output.resolve().is_relative_to(Path(budget).resolve()):
        raise RuntimeError('GPU experiments require the session supervisor and an owned output directory')
    if not 1<=args.envs<=512 or not 1<=args.iterations<=1000:raise ValueError('Unbounded GPU experiment')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(min(4,len(os.sched_getaffinity(0))))
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; do not fall back to CPU')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    task='Mjlab-Velocity-Flat-MicroDuck';ec,ac=load_env_cfg(task),load_rl_cfg(task)
    ec.scene.num_envs=args.envs;ec.seed=ac.seed=args.seed
    ec.sim.mujoco.timestep=.005;ec.decimation=4
    command=ec.commands['twist']
    command.ranges.lin_vel_x=(.20,.45);command.ranges.lin_vel_y=(0.,0.);command.ranges.ang_vel_z=(-.8,.8)
    command.rel_standing_envs=.10;command.rel_forward_envs=.40;command.rel_turn_in_place_envs=.10
    ec.curriculum.clear();ec.rewards['action_rate_l2'].weight=-1.
    if args.kind in ('jolt_torque','jolt_inertia'):
        ec.scene.entities['robot'].articulation.actuators=(JoltTorqueProxyCfg(
            target_names_expr=(r'^(?!passive_).*',),stiffness=.55,damping=0.,effort_limit=float('inf'),
            frictionloss=0.,viscous_damping=0.,armature=.0018,delay_min_lag=0,delay_max_lag=0),)
        for event in ['expand_bam_friction_fields','randomize_joint_friction','randomize_motor_gains']:
            ec.events.pop(event,None)
    if args.kind=='jolt_inertia' or args.fixed_inertia:
        # pseudo_inertia uses a physical-solid Cholesky factorization that is
        # invalid for Jolt's rotor-added tensors. Avoid reintroducing armature.
        for event in ['base_com','randomize_com','randomize_head_com','randomize_mass_inertia','randomize_armature']:
            ec.events.pop(event,None)
    ac.logger='tensorboard';ac.upload_model=False;ac.save_interval=25
    ac.algorithm.learning_rate=3e-5;ac.algorithm.desired_kl=.005;ac.algorithm.entropy_coef=.005
    record=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        checkpoint_sha256=sha(args.checkpoint),script_sha256=sha(__file__),
        environment=asdict(ec),algorithm=asdict(ac),torch=torch.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),game_physics_changed=False,
        proxy_scope=('motor torque plus experimental Jolt diagonal rotor inertia; distinct contact solver'
                     if args.kind=='jolt_inertia' else 'motor torque only; distinct MuJoCo contact and armature'),
        inertia_randomization=not (args.kind=='jolt_inertia' or args.fixed_inertia),started_unix=time.time())
    (args.output/'config.json').write_text(json.dumps(record,indent=2,default=str)+'\n')
    env=None;runner=None;start=time.monotonic()
    from mjlab.scene import Scene
    original_compile=Scene.compile
    if args.kind=='jolt_inertia':
        def compile_with_game_inertia(scene):
            model=original_compile(scene)
            audit=patch_game_inertia(model,args.jolt_spec)
            (args.output/'inertia_patch.json').write_text(json.dumps(audit,indent=2)+'\n')
            return model
        Scene.compile=compile_with_game_inertia
    try:
        env=ManagerBasedRlEnv(cfg=ec,device='cuda:0')
        Scene.compile=original_compile
        vec=RslRlVecEnvWrapper(env,clip_actions=ac.clip_actions)
        runner=load_runner_cls(task)(vec,asdict(ac),str(args.output),'cuda:0')
        runner.load(str(args.checkpoint),load_cfg=dict(actor=True,critic=True,optimizer=False,iteration=False,rnd=False),map_location='cuda:0')
        initial=export_checked(runner,vec,args.output,'initial')
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=True)
        runner.save(str(args.output/'final.pt'))
        final=export_checked(runner,vec,args.output,'final')
        (args.output/'completed.json').write_text(json.dumps(dict(status='completed',
            elapsed_s=time.monotonic()-start,iterations=args.iterations,envs=args.envs,
            initial_parity=initial,final_parity=final),indent=2)+'\n')
    except BaseException:
        (args.output/'error.txt').write_text(traceback.format_exc())
        if runner is not None:
            try:runner.save(str(args.output/'interrupted.pt'))
            except Exception:pass
        raise
    finally:
        Scene.compile=original_compile
        if env is not None:env.close()


if __name__=='__main__':main()
