import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,gzip,numpy as np,mujoco as mj,hashlib
R=Path(__file__).parent;P=R.parent/'pd_plant_rebalanced_candidate/attempt_001'
def lock(m):
 changes={}
 for name in dir(m):
  if 'solref' not in name:continue
  a=getattr(m,name)
  if not isinstance(a,np.ndarray) or a.ndim!=2 or a.shape[-1]!=2:continue
  ids=np.where((a[:,0]>0)&(a[:,0]<.01))[0]
  if len(ids):changes[name]={'indices':ids.tolist(),'before':a[ids].tolist(),'after_first':.01};a[ids,0]=.01
 if 0<m.opt.o_solref[0]<.01:changes['option_o_solref']=m.opt.o_solref.tolist();m.opt.o_solref[0]=.01
 return changes
if __name__=='__main__':
 old=mj.MjModel.from_binary_path(str(P/'actual_scene.mjb'));new=mj.MjModel.from_binary_path(str(P/'actual_scene.mjb'));changes=lock(new);a=[json.loads(x) for x in gzip.open(P/'substeps.jsonl.gz','rt')];out=[]
 for t in [0,.1,1.,2.]:
  r=min(a,key=lambda x:abs(x['pre_time_s']-t));ds=[]
  for m in [old,new]:
   d=mj.MjData(m);d.qpos[:]=r['pre_qpos'];d.qvel[:]=r['pre_qvel'];d.ctrl[:]=r['actual_torque18'];mj.mj_forward(m,d);ds.append(d)
  d,e=ds;assert d.nefc==e.nefc;out.append({'time_s':r['pre_time_s'],'qacc_max_abs_diff':float(max(abs(d.qacc-e.qacc))),'efc_force_max_abs_diff':float(max(abs(d.efc_force[:d.nefc]-e.efc_force[:e.nefc]))),'efc_R_max_abs_diff':float(max(abs(d.efc_R[:d.nefc]-e.efc_R[:e.nefc]))),'efc_aref_max_abs_diff':float(max(abs(d.efc_aref[:d.nefc]-e.efc_aref[:e.nefc]))),'contacts_equal':[(c.geom1,c.geom2) for c in d.contact]==[(c.geom1,c.geom2) for c in e.contact]})
 assert max(x['qacc_max_abs_diff'] for x in out)<1e-7 and max(x['efc_force_max_abs_diff'] for x in out)<1e-7
 result={'physics_steps':0,'pass':True,'source_dt':.005,'locked_effective_refsafe_timeconst_s':.01,'changes':changes,'states':out,'original_source_MJB_SHA':hashlib.sha256((P/'actual_scene.mjb').read_bytes()).hexdigest(),'meaning':'Positive reference time constants explicitly clamped to original max(tc,2*.005); negative direct stiffness/damping unchanged; original200Hz qacc/constraintforces/R/aref equal before finer integration'};(R/'refsafe_equivalence.json').write_text(json.dumps(result,indent=2));mj.mj_saveModel(new,str(R/'locked_dt005.mjb'));print(json.dumps(result,indent=2))
