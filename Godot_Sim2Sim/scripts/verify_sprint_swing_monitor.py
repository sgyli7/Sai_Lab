"""GPU integration check: swing instrumentation is passive and reads real hulls."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sim2sim.research.torch_anchor import TorchAnchor
from sprint_gpu_world import GpuWorld


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--control',type=Path,required=True);parser.add_argument('--actor',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    settings=json.loads(args.control.read_text())['walk'];actor=TorchAnchor(args.actor).cuda().eval()
    values=[];samples=[];read_errors=[]
    with torch.inference_mode():
        for enabled in [False,True]:
            world=GpuWorld(8,settings,feet='jolt',graphs=True,velocity_observer='joint_fd')
            try:
                if enabled:world.enable_swing_monitor()
                states=[];command=torch.zeros(8,13,device='cuda');sprint=torch.ones(8,device='cuda',dtype=torch.bool)
                for i in range(250):
                    command[:,0]=0. if i<50 else .45
                    obs,_=world.observe(command,sprint);world.step(actor(obs));states.append(torch.cat([world.qpos,world.qvel,world.last],dim=-1).clone())
                    if enabled:
                        samples.append(torch.stack([world.swing.air_time,world.swing.clearance,world.swing.contact.float()],dim=-1).clone())
                        before=torch.cat([world.qpos,world.qvel,world.ctrl,world.last],dim=-1).clone()
                        world.foot_contacts();world.foot_clearance()
                        after=torch.cat([world.qpos,world.qvel,world.ctrl,world.last],dim=-1)
                        read_errors.append((before-after).abs().amax())
                values.append(torch.stack(states).cpu().numpy())
                if enabled:
                    world.qpos[:,world.qa+2]+=1.;world.forward()
                    assert not world.foot_contacts().any(),'Contacts must disappear when lifted above floor'
                    assert (world.foot_clearance()>.8).all(),'Clearance must track actual hull height'
            finally:world.close()
    error=float(np.max(np.abs(values[0]-values[1])))
    data=torch.stack(samples).cpu().numpy();np.savez_compressed(args.output/'sensors.npz',samples=data)
    read_error=float(torch.stack(read_errors).amax())
    assert read_error==0.,'Sensor reads changed physical/action state'
    assert data[...,2].any() and (~data[...,2].astype(bool)).any(),'Need actual stance and swing samples'
    assert data[...,1].max()>.005,'No valid clearance samples'
    result=dict(passed=True,same_state_read_max_abs=read_error,separate_run_max_abs=error,
                repeatability_note='Separate Warp worlds are not assumed bit deterministic. Both training arms use the same enabled monitor. This check establishes read purity on the same state, not cross-run trajectory equivalence.',
                rows=250,worlds=8,contact_fraction=float(data[...,2].mean()),air_time_max=float(data[...,0].max()),clearance_range=[float(data[...,1].min()),float(data[...,1].max())],lifted_floor_check=True)
    (args.output/'completed.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
