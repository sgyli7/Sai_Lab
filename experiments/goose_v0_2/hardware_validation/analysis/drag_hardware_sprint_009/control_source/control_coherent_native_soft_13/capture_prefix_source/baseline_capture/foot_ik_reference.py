"""Offline foot IK only. No model load, stepping, I/O or search occurs on import.
Caller must supply a separate MjData scratch object, never the live environment data.
This API does not grant source- or dynamic-safety admission to any resulting pose.
"""
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

LEG_JOINT_SUFFIXES = ('hip_yaw','hip_roll','hip_pitch','knee_pitch','ankle_pitch','ankle_roll')

def solve_foot_ik(model, scratch_data, side, target_geom_center_world_m,
                  target_geom_rotation_world, seed6_rad, *, max_nfev=40):
    """Mutate only caller-owned scratch qpos, return original six leg joint targets.

    Target is actual_sole GEOM center/orientation, NOT ankle center or sole contact.
    Root and other joints must already contain the intended OFFLINE reference state.
    Does not solve COM, force balance, closed-mouth force, collision or trajectory.
    The existing 1:1 hip-yaw/input-rotor relation is synchronized in scratch only.
    """
    if side not in ('left','right'):
        raise ValueError('side must be left or right')
    names=[side+'_'+suffix for suffix in LEG_JOINT_SUFFIXES]
    indices=np.array([int(model.joint(n).qposadr[0]) for n in names])
    bounds=np.array([model.jnt_range[model.joint(n).id] for n in names])
    gid=model.geom(side+'_actual_sole').id
    target=np.asarray(target_geom_center_world_m,float)
    rotation=np.asarray(target_geom_rotation_world,float).reshape(3,3)
    residual_calls=0
    def residual(q):
        nonlocal residual_calls
        residual_calls+=1
        scratch_data.qpos[indices]=q
        mj.mj_kinematics(model,scratch_data)
        return np.r_[scratch_data.geom_xpos[gid]-target,
                     .08*Rotation.from_matrix(rotation.T@scratch_data.geom_xmat[gid].reshape(3,3)).as_rotvec()]
    sol=least_squares(residual,np.clip(seed6_rad,bounds[:,0]+1e-6,bounds[:,1]-1e-6),
                      bounds=(bounds[:,0]+1e-7,bounds[:,1]-1e-7),max_nfev=int(max_nfev),
                      gtol=1e-9,ftol=1e-9,xtol=1e-9)
    scratch_data.qpos[indices]=sol.x
    scratch_data.qpos[int(model.joint(side+'_yaw_input_rotor').qposadr[0])]=sol.x[0]
    err=residual(sol.x)
    return {'joint_names6':names,'q_target6_rad':sol.x.tolist(),
            'equivalent_IK_error_m':float(np.linalg.norm(err)),
            'position_error_world_m':err[:3].tolist(),'weighted_rotation_error':err[3:].tolist(),
            'nfev':int(sol.nfev),'FK_residual_calls':residual_calls,
            'offline_only':True,'physics_steps':0}
