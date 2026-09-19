"""Falsify whether upward stair swing causes loaded descent pitch/slip."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from sim2sim.sai_compliance import CompliantController
from sim2sim.sai_terrain import EDGE_X
import sai_loaded_mujoco as harness

class DescentController(CompliantController):
    strategy='rolling'
    @staticmethod
    def descending(state):
        if 'terrain_edge_heights' not in state:return False
        h=np.asarray(state['terrain_edge_heights']).reshape(46,3);delta=np.diff(h,axis=0);p=np.pad(delta,((2,2),(0,0)),mode='edge')
        e=delta-np.median(np.stack([p[i:i+45] for i in (0,1,3,4)]),axis=0);e=e[(EDGE_X[:-1]>=-.20)&(EDGE_X[:-1]<.54)]
        return bool(np.any(e<-.008) and not np.any(e>.008))
    def _step_in_wheel_path(self,state,honor_course=True):
        if self.strategy=='rolling' and self.descending(state):return False
        return super()._step_in_wheel_path(state,honor_course)
    def command(self,state):
        old=self.stair_settings['lift_height']
        if self.strategy=='low_lift' and self.descending(state):self.stair_settings['lift_height']=.015
        try:return super().command(state)
        finally:self.stair_settings['lift_height']=old

def job(args):
    strategy,case,params,out=args
    DescentController.strategy=strategy;harness.CompliantController=DescentController
    r=harness.run(parameters=params,out=out,**case);r['strategy']=strategy;return r

def main():
    root=Path('results/sai-cargo-suspension-20260915');out=root/'descent-probe';out.mkdir(exist_ok=False)
    p=json.loads((root/'screening/candidate.json').read_text())['parameters']
    cases=[dict(kind='down40',seed=2003,mass=.05,yaw=-.06,start_x=-.02,tread=.175,terrain_scale=.95),dict(kind='down40',seed=2203,mass=.1,yaw=.03,start_x=.01,tread=.185,terrain_scale=1.05),dict(kind='down40',seed=47,mass=.1),dict(kind='down20',seed=47,mass=.1)]
    (out/'protocol.json').write_text(json.dumps(dict(hypothesis='Upward swing removes support during descent. Compare contact-following roll versus 15 mm swing; gains, speeds, masses and terrain unchanged.',parameters=p,cases=cases),indent=2)+'\n')
    jobs=[(s,c,p,out/f'{s}-{i}') for s in ['rolling','low_lift'] for i,c in enumerate(cases)]
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows=[]
        for r in pool.map(job,jobs):rows.append(r);print(json.dumps(r),flush=True);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
