"""Conservative point budget from frozen M/r plus kinematics only, no forward."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import mujoco as mj

def ident(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main(root,arrays,out):
    assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    a=np.load(arrays);m=mj.MjModel.from_binary_path(str(root/'attempt_001/actual_scene.mjb'));d=mj.MjData(m)
    d.qpos[:]=a['qpos'];d.qvel[:]=a['qvel'];mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
    M=a['M'];r=a['residual'];Mi=np.linalg.inv(M);rho=float(np.sqrt(r@Mi@r));rows=[]
    def point(label,b,p):
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,p,b)
        scalar=rho*np.sqrt(np.maximum(0,np.diag(jp@Mi@jp.T)))
        vector=rho*np.sqrt(max(0,float(np.linalg.eigvalsh(jp@Mi@jp.T)[-1])))
        ang=rho*np.sqrt(max(0,float(np.linalg.eigvalsh(jr@Mi@jr.T)[-1])))
        return dict(label=label,body=m.body(b).name,world_point_m=p.tolist(),coordinate_acceleration_bounds_m_s2=scalar.tolist(),euclidean_acceleration_bound_m_s2=float(vector),euclidean_angular_acceleration_bound_rad_s2=float(ang),local_zero_error_1ms_displacement_bound_m=float(.5e-6*vector),local_zero_error_5ms_displacement_bound_m=float(.5*25e-6*vector))
    for side in ['left','right']:
        g=m.geom(side+'_actual_sole').id;b=int(m.geom_bodyid[g]);mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh]);ps=m.mesh_vert[start:start+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
        allrows=[point(side+'_sole_vertex_'+str(i),b,p) for i,p in enumerate(ps)]
        rows.append(dict(label=side+'_all_63_sole_vertices',vertex_count=n,worst=max(allrows,key=lambda x:x['euclidean_acceleration_bound_m_s2'])))
    b=m.body('head_roll').id;p=d.xpos[b]+d.xmat[b].reshape(3,3)@np.array([.13750001972253326,0,-.03849996489687674]);rows.append(point('true_mouth_TCP',b,p))
    b=m.body('beak_coupler_link').id;eq=m.equality('beak_distal_pin').id
    assert int(m.eq_obj2id[eq])==b
    endpoints=[('coupler_proximal_pin',np.zeros(3)),('coupler_distal_pin',m.eq_data[eq,3:6]),('coupler_COM',m.body_ipos[b])]
    for name,local in endpoints:rows.append(point(name,b,d.xpos[b]+d.xmat[b].reshape(3,3)@local))
    joint=[]
    for k in range(6,28):
        bound=float(rho*np.sqrt(Mi[k,k]));jid=int(m.dof_jntid[k]);joint.append(dict(dof=k,name=m.joint(jid).name,acceleration_bound_rad_s2=bound,angle_bound_1ms_rad=.5e-6*bound,angle_bound_5ms_rad=.5*25e-6*bound))
    result=dict(source=ident(Path(__file__)),inputs=[ident(arrays),ident(root/'attempt_001/actual_scene.mjb')],physics_steps=0,LP_calls=0,pure_forward_calls=0,kinematics_calls=1,comPos_calls=1,r_mass_inverse_norm=rho,derivation='For fixed q,v, the native convex acceleration objective has Hessian >= M. Its gradient r gives ||a-a*||_M <= sqrt(rT M^-1 r). Thus ||Jp(a-a*)||2 <= sqrt(lambda_max(Jp M^-1 JpT))*sqrt(rT M^-1 r). Scalar coordinates use sqrt(j M^-1 jT). Local zero-initial-error displacement is approximated conservatively from this instantaneous acceleration budget as 0.5 dt^2 times the acceleration bound, holding q,v,M,J fixed.',points=rows,robot_hinge_acceleration_bounds=joint,scope_limits=['Only instantaneous same-state numerical optimization error, no full-trajectory integration error bound.','The dt^2 local Taylor budget is not a rigorous finite-horizon bound across moving M/J, changing contact sets, or feedback.','M^-1 r alone is a force-frozen algebraic correction, not the exact constrained optimum displacement.','No actual or hypothetical trajectory has been stepped or reset.'],official_basis_url='https://mujoco.readthedocs.io/en/stable/computation/#algorithms')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=ident(out),points=rows,worst_joint=max(joint,key=lambda x:x['angle_bound_5ms_rad'])),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arrays',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.arrays,a.output)
