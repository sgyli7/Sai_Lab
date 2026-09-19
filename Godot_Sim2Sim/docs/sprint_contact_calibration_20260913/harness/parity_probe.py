from pathlib import Path
import json,sys
import torch
from sim2sim.research.queue import atomic_json
sys.path.insert(0,str(Path('results/sprint_contact_calibration_20260913').resolve()))
from contact_probe import initialize,microstep,GpuWorld,R

def main():
    torch.set_num_threads(4)
    selected=sorted([e for e in json.loads(Path('results/sprint_stop_state_20260912/native_joint_fd/suite/summary.json').read_text())['episodes'] if e['case']=='sprint_alternate' and 927000<=e['seed']<=927015],key=lambda e:e['seed'])
    rows=[json.loads(Path(e['trace']).read_text())['rows'] for e in selected]
    settings=json.loads(Path('results/sprint_joint_20260912/delivery/control.json').read_text())['walk']
    a=torch.tensor([r[0]['action'] for r in rows],device='cuda')
    states=[];out=[]
    for fresh in range(2):
        w=GpuWorld(16,settings,feet='jolt',graphs=True,velocity_observer='joint_fd')
        try:
            for kind in ['normal','normal','micro','micro']:
                initialize(w,rows)
                before=[w.qpos.clone(),w.qvel.clone(),w.observer.qd.clone()]
                if kind=='normal':w.step(a)
                else:
                    for _ in range(4):microstep(w,a)
                after=[w.qpos.clone(),w.qvel.clone(),w.observer.qd.clone()]
                if not states:states=[before,after]
                record=dict(fresh=fresh,kind=kind,before=[float((x-y).abs().max()) for x,y in zip(before,states[0])],after=[float((x-y).abs().max()) for x,y in zip(after,states[1])])
                out.append(record);print(record,flush=True)
        finally:w.close()
    atomic_json(R/'parity_probe.json',dict(results=out,scope='Diagnose microstep equivalence and reset repeatability, no contact tuning'))

if __name__=='__main__':main()
