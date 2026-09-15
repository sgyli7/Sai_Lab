"""Read recorded MuJoCo states and detect swings far before the first riser.

This is a read-only audit, not a new rollout or a change to the motor policy.
"""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
from sai_loaded_mujoco import LoadedTerrain

LEGS=['front_left','front_right','rear_left','rear_right']

def audit(episode):
    report=json.loads((episode/'report.json').read_text());rows=json.loads((episode/'trace.json').read_text())
    terrain=LoadedTerrain(report['seed'],report['kind'],report['mass'],report.get('tread',.18),report.get('terrain_scale',1.))
    model=terrain.model();data=mujoco.MjData(model)
    wheels=[model.body(n+'_wheel').id for n in LEGS]
    geoms=[next(i for i in range(model.ngeom) if model.geom_bodyid[i]==b and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_CYLINDER) for b in wheels]
    first_edge=1.1 if report['kind'].startswith('mixed') else .45
    events=[];active=np.zeros(4,dtype=bool);before=[]
    for row in rows:
        data.qpos[:]=row['qpos'];mujoco.mj_forward(model,data)
        centers=data.geom_xpos[geoms];axes=data.geom_xmat[geoms].reshape(4,3,3)[:,:,2]
        sizes=model.geom_size[geoms]
        extent_z=sizes[:,0]*np.sqrt(np.maximum(0.,1-axes[:,2]**2))+sizes[:,1]*abs(axes[:,2])
        extent_x=sizes[:,0]*np.sqrt(np.maximum(0.,1-axes[:,0]**2))+sizes[:,1]*abs(axes[:,0])
        ground=terrain.query(centers[:,:2]);gap=centers[:,2]-ground-extent_z
        supported=np.zeros(4,dtype=bool)
        for ct in data.contact:
            for i,g in enumerate(geoms):
                if g in ct.geom:
                    other=ct.geom[1] if ct.geom[0]==g else ct.geom[0]
                    if model.geom_group[other]==5:supported[i]=True
        # 15 mm clearance and 100 mm wheel-envelope-to-edge margin isolate
        # substantial air motion well before climbing is geometrically necessary.
        far=(first_edge-(centers[:,0]+extent_x))>.10
        lifted=(gap>.015)&far&(row['time']>=1.)&~supported
        for i in range(4):
            if lifted[i] and not active[i]:events.append(dict(leg=LEGS[i],time=row['time'],gap_m=float(gap[i]),distance_to_first_riser_m=float(first_edge-centers[i,0]-extent_x[i])))
        active=lifted
        if row['time']>=1. and np.all(far):before.append(dict(time=row['time'],gap=gap.tolist(),stage=row['stage']))
    return dict(episode=str(episode),kind=report['kind'],passed_no_premature_lift=not events,premature_lift_events=events,
                all_wheels_far_samples=len(before),max_gap_while_all_far_m=max((max(r['gap']) for r in before),default=0.),
                definition='actual cylinder geometry orientation/extents; no wheel-terrain contact, >15 mm vertical ground clearance and front envelope >100 mm before first riser; after 1 s settling')

def main():
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--assert-no-premature',action='store_true');a=p.parse_args()
    result=audit(a.episode);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if a.assert_no_premature and not result['passed_no_premature_lift']:raise SystemExit(1)
if __name__=='__main__':main()
