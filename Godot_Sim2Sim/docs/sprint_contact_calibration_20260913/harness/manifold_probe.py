"""Reconstruct feet from simultaneous native base pose and measured hinges.

Only forward kinematics is run; no MuJoCo physics step or policy evaluation.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import mujoco
from sim2sim.paths import load_robot_json
from sim2sim.coords import quat_wxyz_to_mat
from sim2sim.research.queue import atomic_json

R=Path('results/sprint_contact_calibration_20260913')


def main():
    cfg=load_robot_json(Path('robots/microduck_ball_stand_fix.json'))
    model=mujoco.MjModel.from_xml_path(cfg['mjcf']);data=mujoco.MjData(model)
    base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'trunk_base')
    joint=next(i for i in range(model.njnt) if model.jnt_bodyid[i]==base and model.jnt_type[i]==0)
    qa=model.jnt_qposadr[joint];qi=[model.jnt_qposadr[model.actuator_trnid[i,0]] for i in range(model.nu)]
    names=['ankle_left','ankle_right'];bids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in names]
    summary=json.loads(Path('results/sprint_stop_state_20260912/native_joint_fd/suite/summary.json').read_text())
    episodes=sorted([e for e in summary['episodes'] if e['case']=='sprint_alternate' and 927000<=e['seed']<=927015],key=lambda e:e['seed'])
    positions=[];angles=[];vectors=[];contacts=[];sources=[]
    for e in episodes:
        raw=Path(e['trace']).read_bytes();rows=json.loads(raw)['rows'][:451]
        per_pos=[];per_angle=[];per_vec=[];per_contact=[]
        for row in rows:
            data.qpos[qa:qa+3]=row['body']['base_pos'];data.qpos[qa+3:qa+7]=row['body']['base_quat'];data.qpos[qi]=row['raw']['q']
            mujoco.mj_kinematics(model,data)
            feet={f['name']:f for f in row['raw']['feet']}
            pp=[];pa=[];pv=[];pc=[]
            for name,bid in zip(names,bids):
                foot=feet[name]
                vector=np.asarray(foot['pos'])-data.xipos[bid]
                quat=np.asarray(foot['quat']);quat=quat/np.linalg.norm(quat)
                rotation=quat_wxyz_to_mat(quat)
                angle=np.rad2deg(np.arccos(np.clip((np.trace(rotation.T@data.ximat[bid].reshape(3,3))-1)/2,-1,1)))
                pp.append(np.linalg.norm(vector));pa.append(angle);pv.append(vector);pc.append(foot['ground_contact'])
            per_pos.append(pp);per_angle.append(pa);per_vec.append(pv);per_contact.append(pc)
        positions.append(per_pos);angles.append(per_angle);vectors.append(per_vec);contacts.append(per_contact)
        sources.append(dict(seed=e['seed'],trace=e['trace'],sha256=hashlib.sha256(raw).hexdigest()))
    pos=np.asarray(positions).transpose(1,0,2);ang=np.asarray(angles).transpose(1,0,2)
    vector=np.asarray(vectors).transpose(1,0,2,3);contact=np.asarray(contacts).transpose(1,0,2)
    assert pos[0].max()<1e-6,'Initial coordinate mapping invalid'
    t=np.arange(len(pos))*.02
    metrics={}
    for name,(lo,hi) in {'initial':(0,.001),'precontact':(0,.04),'landing':(.04,.3),'standing':(.3,1.),'forward':(1,3),'left':(3,6),'reverse':(6,9.01)}.items():
        mask=(t>=lo-1e-7)&(t<hi-1e-7)
        metrics[name]=dict(position_m=dict(median=float(np.median(pos[mask])),p95=float(np.quantile(pos[mask],.95)),max=float(pos[mask].max())),
            angle_deg=dict(median=float(np.median(ang[mask])),p95=float(np.quantile(ang[mask],.95)),max=float(ang[mask].max())),
            mean_vector_m=vector[mask].mean((0,1,2)).tolist())
    np.savez_compressed(R/'manifold_residuals.npz',position_error=pos,rotation_error_deg=ang,position_vector=vector,ground_contact=contact,t=t)
    result=dict(metrics=metrics,initial_position_max_m=float(pos[0].max()),
        sources=sources,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sampling_audit='physics_server._send_state refreshes _mj_basis before q; raw feet and base transforms are read in the same call. No stale-basis assumption.',
        scope='Simultaneous native root/hinge coordinates do not exactly determine actual foot COM poses. Residual may include both constrained-axis rotation and translation compliance; does not identify each joint or prove a policy remedy.')
    atomic_json(R/'manifold_probe.json',result);print(json.dumps(result['metrics']),flush=True)


if __name__=='__main__':main()
