"""Real native walking at the new wall and through the open depot doorway.

One explicit test start per case. The doorway harness issues ordinary yaw
commands to steer toward the opening; no mid-run pose changes or direct forces.
"""
from dataclasses import replace
import itertools
import json
import math
import os
from pathlib import Path
import mujoco
import numpy as np
from showcase_resource_guard import preview_lease, preflight
import sim2sim.godot_proc as godot_proc
from sim2sim.research.bundle import bundle_paths
from sim2sim.research.models import NativeAnchor
from sim2sim.research.tasks import TASKS
from sim2sim.research.world import World
from sim2sim.research.evaluate import record
from sim2sim.obs import build_obs
from capture_showcase import serial

ROOT=Path(__file__).resolve().parents[1]
def main():
    cpus=sorted(os.sched_getaffinity(0))[-2:];os.sched_setaffinity(0,set(cpus));os.nice(10)
    godot_proc._CORE_SEQ=itertools.cycle(cpus)
    os.environ.update(MD_WORKSHOP_COLLISIONS='1',MD_STATIC_COURSE='0')
    paths=bundle_paths(ROOT/'bundles/delivery_v3');actor=NativeAnchor(paths['walking'])
    reports={}
    out=ROOT/'results/atmosphere/policy';out.mkdir(parents=True,exist_ok=True)
    with preview_lease():
        preflight()
        for label,xy,yaw in [('wall',[2.97,2.22],math.pi),('doorway',[2.18,1.40],math.pi/2)]:
            world=World(replace(TASKS['walking'],robot='microduck_ball_stand_fix'),
                entry_source=paths['standing'],yaw_memory_input=actor.yaw_memory_input,
                scene_override='res://atelier/atelier.tscn')
            try:
                world.reset(61000,randomize=False)
                world.sampler.sample(world.rng,yaw_range=(yaw,yaw),joint_noise_rad=0.)
                world.mj.data.qpos[world.mj.free_qposadr:world.mj.free_qposadr+2]=xy
                mujoco.mj_forward(world.mj.model,world.mj.data)
                world.state=world.backend.reset(ctrl=world.home,bodies=world.mj.body_poses_mujoco(),report_bodies=world.report_names)
                world.features=world.measure(reset=True)
                teacher,cmd=world.prepare_standing_entry()
                for _ in range(50):world.step(teacher(build_obs(world.state,world.last,cmd,world.home)[None])[0])
                world.finish_standing_entry()
                steps=500 if label=='doorway' else 250
                world.command_tape=np.zeros((steps,13),np.float32);world.command_tape[:,0]=.12
                rows=[];contacts=[]
                for k in range(steps):
                    if label=='doorway':
                        # Test-only steering, equivalent to issuing turn commands.
                        # Record it explicitly; this is not autonomous pathfinding.
                        error=math.atan2(math.sin(yaw-world.features['yaw']),math.cos(yaw-world.features['yaw']))
                        world.command_tape[k,2]=np.clip(error*1.5,-.5,.5)
                    action=actor(world.obs()[None])[0];world.step(action);rows.append(record(world,action))
                    for name,body in world.features['bodies'].items():
                        for contact in body['contact_events']:
                            if 'WorkshopCollisions' in contact['body']:contacts.append(dict(time=world.t,body=name,contact=contact))
                report=dict(initial_mujoco_xy=xy,initial_yaw=yaw,final_xy=rows[-1]['xy'],
                    min_x=min(float(r['xy'][0]) for r in rows),max_y=max(float(r['xy'][1]) for r in rows),
                    max_tilt=max(r['tilt'] for r in rows),fell=any(r['tilt']>70 or r['z']<.045 for r in rows),
                    contacts=contacts,model_sha256=actor.sha256,seconds=1+steps*.02,
                    steering="test-only yaw command toward initial heading" if label=="doorway" else "none")
                reports[label]=report
                np.savez_compressed(out/(label+'.npz'),**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
            finally:
                proc=world.backend._proc;world.close()
                (out/(label+'.log')).write_text(Path(proc._sim2sim_log_path).read_text())
    (out/'report.json').write_text(json.dumps(reports,default=serial,indent=2)+'\n')
    print(json.dumps({k:{**{a:b for a,b in v.items() if a!='contacts'},'contacts':len(v['contacts'])} for k,v in reports.items()},default=serial,indent=2))
    assert reports['wall']['contacts'] and reports['wall']['min_x']>2.65,'Wall penetration or missing real contact'
    assert reports['doorway']['max_y']>1.99,'Robot did not enter doorway'
if __name__=='__main__':main()
