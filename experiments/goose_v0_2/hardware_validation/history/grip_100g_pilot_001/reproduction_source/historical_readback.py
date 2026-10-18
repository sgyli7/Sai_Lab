"""READ-ONLY historical binary identity; not the active pilot engine."""
from pathlib import Path
import os, sys, json, resource, time, hashlib
ROOT=Path(__file__).resolve().parent
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
START=time.time()
sys.path.insert(0,'/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_native_discrete_admission_001/upstream315')
import mujoco as mj
import numpy as np
assert mj.__version__=='3.15.0'
old=Path('/home/ethan/ProjectBackups/2026-10-13/Sai_Lab/goose_picking_001/crown_ribs_final100g_210')
m=mj.MjModel.from_binary_path(str(old/'model.mjb'))
robot_b=[i for i in range(m.nbody) if m.body(i).name.startswith('robot/')]
collisions=[i for i in range(m.ngeom) if m.geom(i).name.startswith('robot/') and (m.geom_contype[i] or m.geom_conaffinity[i])]
jaws=[i for i in collisions if any(x in m.geom(i).name for x in ['bill','rib','jaw','beak','pad'])]
opts={k:getattr(m.opt,k) for k in ['timestep','impratio','iterations','noslip_iterations','solver','integrator','cone','enableflags','disableflags']}
out={'purpose':__doc__,'engine':mj.__version__,'physics_integrations':0,'NN':0,'training':0,'source_model':str(old/'model.mjb'),'source_model_SHA256':hashlib.sha256((old/'model.mjb').read_bytes()).hexdigest(),'robot_body_mass_kg':float(sum(m.body_mass[i] for i in robot_b)),'robot_collision_leaves':len(collisions),'robot_nu':int(m.nu),'robot_njnt':int(m.njnt-1),'robot_body_names':[m.body(i).name for i in robot_b],'equalities':[m.equality(i).name for i in range(m.neq)],'options':opts,'jaw_related_geoms':[{'name':m.geom(i).name,'condim':int(m.geom_condim[i]),'friction':m.geom_friction[i].tolist(),'solref':m.geom_solref[i].tolist(),'solimp':m.geom_solimp[i].tolist()} for i in jaws],'actuation_mapping':[{'name':m.actuator(i).name,'joint':m.joint(int(m.actuator_trnid[i,0])).name,'joint_qposadr':int(m.jnt_qposadr[int(m.actuator_trnid[i,0])]),'joint_dofadr':int(m.jnt_dofadr[int(m.actuator_trnid[i,0])])} for i in range(m.nu)],'interpretation':'Historical only: coupled geometry/engine/solver/drive/scene changes; not a causal A/B and no qualification transfer'}
(ROOT/'historical_readback.json').write_text(json.dumps(out,indent=2)+'\n')
r=resource.getrusage(resource.RUSAGE_SELF)
(ROOT/'resource_exit_historical_readback.json').write_text(json.dumps({'engine':mj.__version__,'purpose':__doc__,'physics_integrations':0,'CPU_affinity':sorted(os.sched_getaffinity(0)),'max_memory_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],'maxRSS_KiB':r.ru_maxrss,'CPU_user_s':r.ru_utime,'CPU_system_s':r.ru_stime,'process_wall_s':time.time()-START,'GPU':0,'training':0},indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['engine','physics_integrations','robot_body_mass_kg','robot_collision_leaves','equalities','options']},indent=2))
