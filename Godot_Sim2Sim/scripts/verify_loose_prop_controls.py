"""Exercise the real B-key handler and reset on the workshop physics server."""
import json
import os
from dataclasses import replace
from pathlib import Path
import numpy as np
from showcase_resource_guard import preflight, preview_lease
from sim2sim.research.world import World
from sim2sim.research.tasks import TASKS

ROOT=Path(__file__).resolve().parents[1]
def main():
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));os.nice(10)
    os.environ.update(MD_WORKSHOP_COLLISIONS='1',MD_STATIC_COURSE='0',MD_PROP_TELEMETRY='1')
    with preview_lease():
        preflight()
        world=World(replace(TASKS['standing'],robot='microduck_ball_stand_fix'),scene_override='res://atelier/atelier.tscn')
        try:
            client=world.backend._client
            world.reset(61000,randomize=False)
            seen=[]
            for i in range(8):
                seen.append(client.call(dict(cmd='atelier_contract'))['prop_target'])
                if i<7:
                    client.call(dict(cmd='key',keycode=66,pressed=True))
                    client.call(dict(cmd='key',keycode=66,pressed=False))
            assert seen==[0,1,2,3,4,5,6,0],seen
            # The real reset command preserves selection but restores all free bodies.
            client.call(dict(cmd='atelier_prop_target',index=2))
            world.reset(61000,randomize=False)
            initial=world.state.extra['raw']['workshop_props']
            for _ in range(10):world.step(np.zeros(14,np.float32))
            world.reset(61000,randomize=False)
            final=world.state.extra['raw']['workshop_props']
            assert [(p['position'],p['quaternion_xyzw']) for p in initial]==[(p['position'],p['quaternion_xyzw']) for p in final]
            assert all(np.linalg.norm(p['linear_velocity'])==0 and np.linalg.norm(p['angular_velocity'])==0 for p in final)
            assert client.call(dict(cmd='atelier_contract'))['prop_target']==2
            report=dict(b_key_cycle=seen,reset_transforms_exact=True,reset_velocities_zero=True,selection_preserved=True)
            (ROOT/'results/workshop_validation/prop_controls.json').write_text(json.dumps(report,indent=2)+'\n')
            print(report)
        finally:world.close()
if __name__=='__main__':main()
