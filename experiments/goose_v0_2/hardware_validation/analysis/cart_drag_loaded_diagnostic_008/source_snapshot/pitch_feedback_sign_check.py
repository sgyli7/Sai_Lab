import os,sys
sys.dont_write_bytecode=True
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,gzip,importlib.util,numpy as np,mujoco as mj
from scipy.spatial.transform import Rotation
R=Path(__file__).parent;P=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots/robots/goose_v0_2/models/physics_v0_2_20261018_04')
def module(n,p):
 sp=importlib.util.spec_from_file_location(n,p);z=importlib.util.module_from_spec(sp);sp.loader.exec_module(z);return z
ad=module('ad',P/'runtime_adapter.py');ik=module('ik',R/'narrow_stance_readiness/foot_ik_reference.py');env=ad.FrozenPhysics(R/'scene');m=env.model;rows=[json.loads(x) for x in gzip.open(R/'attempt_005/substeps.jsonl.gz','rt')];row=min(rows,key=lambda x:abs(x['pre_time_s']-.4));cold=np.load(R/'attempt_005/cold_state.npz');ref=mj.MjData(m);ref.qpos[:]=cold['qpos'];mj.mj_forward(m,ref);out=[]
for pitch in [-.01,.01]:
 scratch=mj.MjData(m);scratch.qpos[:]=row['pre_qpos'];scratch.qvel[:]=row['pre_qvel'];scratch.qpos[:3]=row['command_metadata']['virtual_body_reference'];qq=Rotation.from_euler('y',pitch).as_quat();scratch.qpos[3:7]=np.r_[qq[3],qq[:3]];target=np.array(row['actual_slew_target18']);
 for side in ['left','right']:
  g=m.geom(side+'_actual_sole').id;names=[side+'_'+n for n in ik.LEG_JOINT_SUFFIXES];ix=[env.names.index(n) for n in names];sol=ik.solve_foot_ik(m,scratch,side,ref.geom_xpos[g],ref.geom_xmat[g].reshape(3,3),target[ix]);target[ix]=sol['q_target6_rad']
 scratch.qpos[:]=row['pre_qpos'];scratch.qvel[:]=row['pre_qvel'];tau=env.limit_torque(env.kp*(target-scratch.qpos[env.qidx])-env.kd*scratch.qvel[env.vidx],scratch.qvel[env.drive_vidx]);scratch.ctrl[:]=tau;mj.mj_forward(m,scratch);out.append({'virtual_pitch_rad':pitch,'root_angular_Y_qacc_rad_s2':float(scratch.qacc[4]),'actual_tau18':tau.tolist()})
(R/'pitch_feedback_sign_check.json').write_text(json.dumps({'physics_steps':0,'forward_only_on_separate_scratch':True,'pre_time_s':row['pre_time_s'],'checks':out},indent=2));print(json.dumps(out))
