from pathlib import Path
import dataclasses,hashlib,json,sys
import numpy as np
import mujoco
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.scene import Scene
sys.path.insert(0,str(Path('scripts').resolve()))
from sprint_gpu_proxy import JoltTorqueProxyCfg,patch_game_inertia
from sim2sim.paths import load_robot_json
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_gpu_match_20260912');out=r/'asset_audit';out.mkdir(exist_ok=False)
cfg=load_env_cfg('Mjlab-Velocity-Flat-MicroDuck');cfg.scene.num_envs=1
robot=cfg.scene.entities['robot']
source_name=str(robot.spec_fn)
robot.articulation.actuators=(JoltTorqueProxyCfg(target_names_expr=(r'^(?!passive_).*',),stiffness=.55,damping=0.,effort_limit=float('inf'),frictionloss=0.,viscous_damping=0.,armature=.0018,delay_min_lag=0,delay_max_lag=0),)
scene=Scene(cfg.scene,device='cpu');model=scene.compile()
gamecfg=load_robot_json(Path('robots/microduck_ball_stand_fix.json'));game=mujoco.MjModel.from_xml_path(gamecfg['mjcf'])
def extract(m):
 def name(kind,i):return (mujoco.mj_id2name(m,kind,i) or str(i)).split('/')[-1]
 bodies={name(mujoco.mjtObj.mjOBJ_BODY,i):dict(mass=float(m.body_mass[i]),ipos=m.body_ipos[i].tolist(),iquat=m.body_iquat[i].tolist(),inertia=m.body_inertia[i].tolist()) for i in range(1,m.nbody)}
 joints={name(mujoco.mjtObj.mjOBJ_JOINT,i):dict(body=name(mujoco.mjtObj.mjOBJ_BODY,int(m.jnt_bodyid[i])),type=int(m.jnt_type[i]),pos=m.jnt_pos[i].tolist(),axis=m.jnt_axis[i].tolist(),range=m.jnt_range[i].tolist(),armature=float(m.dof_armature[m.jnt_dofadr[i]])) for i in range(m.njnt)}
 geoms={name(mujoco.mjtObj.mjOBJ_GEOM,i):dict(body=name(mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[i])),type=int(m.geom_type[i]),size=m.geom_size[i].tolist(),pos=m.geom_pos[i].tolist(),quat=m.geom_quat[i].tolist(),contype=int(m.geom_contype[i]),conaffinity=int(m.geom_conaffinity[i]),condim=int(m.geom_condim[i]),friction=m.geom_friction[i].tolist(),solref=m.geom_solref[i].tolist(),solimp=m.geom_solimp[i].tolist(),priority=int(m.geom_priority[i])) for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]}
 return dict(bodies=bodies,joints=joints,geoms=geoms,actuator_order=[name(mujoco.mjtObj.mjOBJ_JOINT,int(m.actuator_trnid[i,0])) for i in range(m.nu)])
a,b=extract(model),extract(game);differences={}
for kind in ['bodies','joints','geoms']:
 differences[kind]=dict(only_training=sorted(set(a[kind])-set(b[kind])),only_game=sorted(set(b[kind])-set(a[kind])),different={name:{key:dict(training=a[kind][name][key],game=b[kind][name][key]) for key in a[kind][name] if a[kind][name][key]!=b[kind][name][key]} for name in a[kind].keys()&b[kind].keys() if a[kind][name]!=b[kind][name]})
atomic_json(out/'compiled_training.json',a);atomic_json(out/'game_source.json',b)
inertia=patch_game_inertia(model,gamecfg['godot_spec']);atomic_json(out/'inertia.json',inertia)
record=dict(status='completed',source_spec=source_name,differences=differences,actuator_order_equal=a['actuator_order']==b['actuator_order'],actions=dataclasses.asdict(cfg.actions['joint_pos']),observations=dataclasses.asdict(cfg.observations['actor']),commands={k:dataclasses.asdict(v) for k,v in cfg.commands.items()},limitations='CPU compilation audit of configured GPU task before events, no physics steps. Source MJCF is not a proof of identical generated Jolt hulls or collision exclusions.')
(out/'completed.json').write_text(json.dumps(record,indent=2,default=str)+'\n')
print(dict(actuator_order_equal=record['actuator_order_equal'],differences=differences),flush=True)
