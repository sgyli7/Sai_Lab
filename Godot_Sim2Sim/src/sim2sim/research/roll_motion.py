"""Training reward reference from successful original MuJoCo forward rolls.

The reference never produces deployment actions or changes native body poses.
Time-aware PPO can use its pose/orientation/progress errors as dense feedback.
"""
from functools import lru_cache
import hashlib,json
from pathlib import Path
import numpy as np
from .tasks import TASKS,SESSION,DT
from .world import World
from .models import NativeAnchor
from .evaluate import record,summarize


def build(path=None):
    path=Path(path or SESSION/'roulade_source_motion.npz');task=TASKS['roulade'];teacher=NativeAnchor(task.source)
    if path.exists():
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['source_sha256']!=teacher.sha256:raise RuntimeError('Roll reference parent changed')
        return path
    w=World(task,'mujoco');samples=[];reports=[]
    try:
        for seed in range(40000,40008):
            obs=w.reset(seed);rows=[];pose=[];gravity=[];height=[];progress=[];net=0.
            for k in range(round(task.seconds/DT)):
                a=teacher(obs[None])[0];obs=w.step(a);rows.append(record(w,a));net+=w.features['gyro'][1]*DT
                pose.append(w.state.q.copy());gravity.append(w.features['rot'][2,:].copy())
                height.append(w.features['z']);progress.append(net)
            report=summarize(task,rows,w.heading);reports.append(dict(seed=seed,**report))
            if report['success']:samples.append(dict(q=pose,gravity=gravity,z=height,net=progress))
    finally:w.close()
    if len(samples)<6:raise RuntimeError('Insufficient successful source rolls for a motion reference')
    arrays={k:np.median(np.array([s[k] for s in samples]),axis=0) for k in samples[0]}
    arrays['gravity']/=np.linalg.norm(arrays['gravity'],axis=1,keepdims=True)
    arrays['time']=np.arange(1,round(task.seconds/DT)+1)*DT
    np.savez_compressed(path,**arrays)
    path.with_suffix('.json').write_text(json.dumps(dict(source_sha256=teacher.sha256,physics=w.physics,episodes=reports,
        aggregation='elementwise median over successful training seeds 40000-40007'),indent=2))
    return path


class RollMotion:
    def __init__(self):
        self.path=build();self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path) as a:self.data={k:a[k] for k in a.files}

    def sample(self,seconds):
        index=np.clip(float(seconds)/DT-1,0,len(self.data['time'])-1);lo=int(index);hi=min(lo+1,len(self.data['time'])-1);f=index-lo
        return {k:a[lo]*(1-f)+a[hi]*f for k,a in self.data.items() if k!='time'}


@lru_cache(maxsize=1)
def reference():return RollMotion()


if __name__=='__main__':print(build())
