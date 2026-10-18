import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_GL']='disable';os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
from pathlib import Path
import json,xml.etree.ElementTree as E,importlib.util,hashlib
import numpy as np,mujoco as mj
from scipy.optimize import least_squares
R=Path(__file__).parent;W=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots');P=W/'robots/goose_v0_2/models/physics_v0_2_20261018_04';L=Path('/home/ethan/Projects/TempWorktree/Sai_Lab/goose_v02_hardware_validation/experiments/goose_v0_2/hardware_validation');A=L/'assets/low_profile_stock_cart_03';S=R/'scene';S.mkdir(exist_ok=True)
X=E.parse(A/'goose_cart_scene.xml');root=X.getroot();root.find('compiler').set('meshdir',str(A));baseX=E.parse(P/'robot.xml').getroot();root.find('./worldbody/geom[@name="ground"]').set('conaffinity',baseX.find('./worldbody/geom[@name="ground"]').get('conaffinity'))
for mesh in root.findall('./asset/mesh'):mesh.set('file',str((A/mesh.get('file')).resolve()))
E.SubElement(root.find('equality'),'weld',name='stationary_local_torso_fixture',body1='world',body2='torso',relpose='0 0 0.005000002805734566 1 0 0 0',solref='.005 1',solimp='.99 .9999 .0001')
b=E.SubElement(root.find('worldbody'),'body',name='probe_cargo',pos='.9711259770181133 0 .2565');E.SubElement(b,'freejoint',name='probe_cargo_free');E.SubElement(b,'inertial',pos='0 0 0',mass='20',diaginertia='.3041666666666667 .6375 .8666666666666667');E.SubElement(b,'geom',name='probe_cargo_box',type='box',size='.3 .2 .075',density='0',contype='8',conaffinity='7',friction='.65 .01 .002',condim='3',solref='.005 1',rgba='.6 .35 .15 1')
for j in root.findall('.//joint'):
 if (j.get('name') or '').startswith('cart_wheel_spin'):j.set('frictionloss','.125')
j=root.find('.//joint[@name="cart_pitch_hinge"]');j.set('stiffness','20');j.set('springref','.0276930475');key=root.find('./keyframe/key');key.set('qpos',key.get('qpos')+' .9711259770181133 0 .2565 1 0 0 0');X.write(S/'robot.xml')
for n in ['physical_contract.json','actuator_contract.json']:(S/n).write_bytes((P/n).read_bytes())
m=mj.MjModel.from_xml_path(str(S/'robot.xml'));d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);mj.mj_forward(m,d);head=m.body('head_roll').id;rod=m.geom('cart_bite_bar').id;local=d.xmat[head].reshape(3,3).T@(d.geom_xpos[rod]-d.xpos[head]);axes=['neck_pitch','neck_mid_pitch','head_pitch'];qi=np.array([int(m.joint(n).qposadr[0]) for n in axes]);lo=m.jnt_range[[m.joint(n).id for n in axes],0];hi=m.jnt_range[[m.joint(n).id for n in axes],1];cold=d.qpos.copy();bar=d.geom_xpos[rod].copy();pitch0=sum(cold[qi]);jq=int(m.joint('beak_hinge').qposadr[0]);jrot=int(m.joint('beak_input_rotor').qposadr[0]);jcouple=int(m.joint('beak_coupler').qposadr[0]) if 'beak_coupler' in [m.joint(i).name for i in range(m.njnt)] else None
C=json.loads((P/'physical_contract.json').read_text());candidates=[]
def setjaw(q):
 d.qpos[jq]=q
 for j in C['passive_linkage_joints']:
  d.qpos[int(m.joint(j['name']).qposadr[0])]=j['mimic_multiplier']*d.qpos[int(m.joint(j['mimic_joint']).qposadr[0])]+j['mimic_offset_rad']
def ik(target,pitch,seed):
 def f(q):
  d.qpos[qi]=q;mj.mj_kinematics(m,d);return np.r_[(d.xpos[head]+d.xmat[head].reshape(3,3)@local-target)[[0,2]],.08*(sum(q)-pitch)]
 sol=least_squares(f,np.clip(seed,lo+.100001,hi-.100001),bounds=(lo+.1,hi-.1),max_nfev=60,gtol=1e-10,ftol=1e-10,xtol=1e-10);return sol.x,np.linalg.norm(f(sol.x))
for pitch in [pitch0,.25,.30,.40]:
 for offx,offz in [(0,.002),(0,.003),(0,.004),(0,.005),(0,.006),(.01,.004),(-.01,.004)]:
  d.qpos[:]=cold;setjaw(.28);q,e=ik(bar+[offx,0,offz],pitch,cold[qi]);d.qpos[qi]=q;mj.mj_forward(m,d);dist=[];bad=[]
  for c in d.contact:
   a=m.geom(c.geom1).name;b=m.geom(c.geom2).name
   if ('cart' in a and ('bill' in b or 'neck' in b)) or ('cart' in b and ('bill' in a or 'neck' in a)):dist.append({'a':a,'b':b,'dist':float(c.dist)})
   if c.dist<-.0005:bad.append([a,b,float(c.dist)])
  # actual minimum mouth-cart geometry distance via mj_geomDistance, even without active contacts
  distances=[];mouth=[i for i in range(m.ngeom) if any(s in m.geom(i).name for s in ['bill','neck_pitch','neck_mid'])];cartgeoms=[i for i in range(m.ngeom) if m.geom(i).name.startswith('cart_')]
  for a in mouth:
   for b in cartgeoms:
    distv=mj.mj_geomDistance(m,d,a,b,1.,None)
    distances.append(float(distv))
  path=[];seed=q
  for u in [0,.25,.5,.75,1.]:
   qq,ee=ik(bar+[-.05*u,0,0],pitch,seed);seed=qq;path.append({'u':u,'q':qq.tolist(),'error':float(ee),'margin':float(np.min(np.minimum(qq-lo,hi-qq)))})
  candidates.append({'jaw':.28,'pitch':pitch,'off':[offx,0,offz],'pre_q':q.tolist(),'pre_error':float(e),'minimum_mouth_cart_distance':min(distances),'bad':bad,'active_cart_contacts':dist,'pull_path':path,'pathmaxerror':max(x['error'] for x in path)})
valid=[c for c in candidates if c['pre_error']<.0005 and c['minimum_mouth_cart_distance']>.0005 and not c['bad'] and c['pathmaxerror']<.0005]
valid.sort(key=lambda c:(-min(x['margin'] for x in c['pull_path']),c['pre_error']))
res={'physics_steps':0,'candidates':candidates,'valid_count':len(valid),'selected':valid[0] if valid else None,'cold_tcp_head':local.tolist(),'rod':bar.tolist(),'axes':axes,'cold_qpos':cold.tolist()};(R/'planning_gate.json').write_text(json.dumps(res,indent=2));print('GATE',json.dumps({k:v for k,v in res.items() if k not in ['candidates','cold_qpos']}),flush=True)
