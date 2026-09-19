"""Single-variable turning support probe, preserving straight suspension unchanged."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import sai_loaded_mujoco as harness
from sim2sim.sai_compliance import CompliantController
class TurnController(CompliantController):
    blend=0.
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.turn_support=0.
    def command(self,state):
        r=super().command(state)
        if 'leg_kp' not in r:return r
        goal=self.blend*min(1.,abs(state['command'][1])/.35);self.turn_support+=(goal-self.turn_support)*.02/.12
        w=self.turn_support
        r['leg_kp']=((1-w)*np.asarray(r['leg_kp'])+80*w).tolist()
        r['leg_kd']=((1-w)*np.asarray(r['leg_kd'])+2*w).tolist()
        r['leg_feedforward']=((1-w)*np.asarray(r['leg_feedforward'])).tolist()
        return r

def job(args):
    blend,case,p,out=args;TurnController.blend=blend;harness.CompliantController=TurnController
    r=harness.run(parameters=p,out=out,**case);r['blend']=blend;return r

def main():
    root=Path('results/sai-cargo-suspension-20260915');out=root/'turn-probe';out.mkdir(exist_ok=False);p=json.loads((root/'descent-selection/candidate.json').read_text())['parameters']
    cases=[dict(kind='rough',seed=3907,mass=.1,clamped=True,drive_speed=.3,turn_rate=y) for y in [.35,-.35]]
    jobs=[(b,c,p,out/f'{b}-{i}') for b in [.25,.5,1.] for i,c in enumerate(cases)]
    (out/'protocol.json').write_text(json.dumps(dict(hypothesis='Turning requires lateral support absent from vertical load model; interpolate toward original stance during commanded turns, with 120 ms transition. Only this blend varies.',blends=[.25,.5,1.],cases=cases),indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=3) as pool:reports=list(pool.map(job,jobs))
    (out/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    for r in reports:print(r['blend'],r['turn_rate'],r['metrics'],flush=True)
if __name__=='__main__':main()
