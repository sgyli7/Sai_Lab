import json,importlib.util,numpy as np,mujoco
from scipy.optimize import linprog
from pathlib import Path
p=Path('robots/goose_v0_2/models/physics_v0_2_20261018_04');c=json.load(open(p/'physical_contract.json'));sp=importlib.util.spec_from_file_location('h',p/'verify_contacts.py');h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h);m=mujoco.MjModel.from_xml_path(str(p/'robot.xml'));d=mujoco.MjData(m);hb=m.body('head_roll').id
h.pose(m,d,c,{},root_z=0.);native_tcp=np.array(c['grip_reference_zero_m']);r0=d.xmat[hb].reshape(3,3).copy();t0=d.xpos[hb].copy();local=r0.T@(native_tcp-t0)
q=dict(neck_pitch=.464336,neck_mid_pitch=.736350,head_pitch=-1.200687,beak_hinge=.19682523003160665);h.pose(m,d,c,q,root_z=.005000003);tcp=d.xmat[hb].reshape(3,3)@local+d.xpos[hb];jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jac(m,d,jp,jr,tcp,hb)
js=c['joints'];act_indices=[int(m.joint(j.get('actuation_joint',j['name'])).dofadr[0])for j in js];A=np.zeros((m.nv,len(js)));A[act_indices,np.arange(len(js))]=1.;cap=np.array([j['continuous_design_limit_nm']for j in js]);loss=np.array([m.dof_frictionloss[k]for k in act_indices]);g=d.qfrc_bias.copy()
equality=np.asarray(d.efc_type)==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY);E=np.asarray(d.efc_J).reshape(d.nefc,m.nv)[equality].T;C=[]
for side,sy in(('left',1),('right',-1)):
 bid=m.body(side+'_ankle_roll').id;y=sy*.125
 for x,yy in [(-.0299,y),(.0061,y-.036),(.09053355,y-.036),(.12653355,y),(.09053355,y+.036),(.0061,y+.036)]:
  z=np.zeros((3,m.nv));rot=z.copy();mujoco.mj_jac(m,d,z,rot,np.array([x,yy,0.]),bid);C.append(z.T)
C=np.hstack(C);nact=len(js);nc=C.shape[1];ne=E.shape[1];Nv=nact+nc+ne+1;AA=np.c_[A,C,E,np.zeros(m.nv)];ub=[];uv=[]
for i in range(nact):
 for s in(1,-1):
  r=np.zeros(Nv);r[i]=s;r[-1]=-cap[i];ub.append(r);uv.append(-loss[i])
for k in range(nc//3):
 for sx,sy in((1,1),(1,-1),(-1,1),(-1,-1)):
  r=np.zeros(Nv);r[nact+3*k]=sx;r[nact+3*k+1]=sy;r[nact+3*k+2]=-.65;ub.append(r);uv.append(0.)
bounds=[(None,None)]*nact+[(None,None),(None,None),(0,None)]*(nc//3)+[(None,None)]*ne+[(0,None)];objective=np.zeros(Nv);objective[-1]=1
print('static_nativeTCP_ground',tcp.tolist(),'q',q,'native_model_reference_correction_local_m',local.tolist(),'neqrows',ne,'integrations',0)
for Fx,Fy,Fz in[(0,0,0),(5,0,0),(10,0,0),(20,0,0),(20,0,-3),(0,3,0),(-10,0,0)]:
 F=np.array([Fx,Fy,Fz]);r=linprog(objective,A_ub=np.array(ub),b_ub=np.array(uv),A_eq=AA,b_eq=g-jp.T@F,bounds=bounds,method='highs')
 if r.success:
  tau=r.x[:nact];u=(abs(tau)+loss)/cap;ri=r.x[nact:nact+nc].reshape(-1,3);print('whole_static',F.tolist(),'min_cont_util',float(r.x[-1]),'worst',js[int(u.argmax())]['name'],'feetF',ri[:6].sum(0).tolist(),ri[6:].sum(0).tolist(),'residual',float(abs(AA@r.x-(g-jp.T@F)).max()),'passive_lambda_max',float(abs(r.x[nact+nc:-1]).max()))
 else:print('whole_static',F.tolist(),'NO_BALANCE',r.message)
idx=[int(m.joint(j['name']).dofadr[0])for j in js[:5]];B=np.c_[jp[:,idx].T,jr[:,idx].T];gg=g[idx];caps=cap[:5];fr=np.array([m.dof_frictionloss[i]for i in idx])
print('upper_rows',[dict(name=js[i]['name'],gravity=gg[i],coef=B[i].tolist())for i in range(5)])
for k,lab in enumerate(('Fx','Fy','Fz','Mx','My','Mz')):
 for s in(1,-1):
  co=-s*B[:,k];lo=0.;hi=1e9
  for i in range(5):
   if abs(co[i])<1e-10:
    if abs(gg[i])>caps[i]-fr[i]:hi=-1.
   else:
    a,b=sorted(((-caps[i]+fr[i]-gg[i])/co[i],(caps[i]-fr[i]-gg[i])/co[i]));lo=max(lo,a);hi=min(hi,b)
  print('upper1D',lab,s,lo,hi)
