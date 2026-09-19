"""Training-only physical starts sampled from successful source forward rolls.

The actor receives its usual 61 observations and retained action history.
Evaluation never imports or applies this reset curriculum.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from .tasks import TASKS,SESSION,DT
from .world import World
from .models import NativeAnchor
from .evaluate import record,summarize


def build_roll_starts(path=None):
    path=Path(path or SESSION/"roulade_source_starts.npz")
    task=TASKS["roulade"];policy=NativeAnchor(task.source)
    if path.exists():
        meta=json.loads(path.with_suffix(".json").read_text())
        if meta["source_sha256"]!=policy.sha256:raise RuntimeError("Curriculum source changed")
        return path
    w=World(task,"mujoco");samples=[];reports=[]
    try:
        for seed in range(40000,40008):
            obs=w.reset(seed);rows=[];episode_samples=[]
            net=0.;frontier=0.;pivot=False;inverted=False
            for k in range(round(task.seconds/DT)):
                action=policy(obs[None])[0];obs=w.step(action);rows.append(record(w,action))
                f=w.features;net+=float(f["gyro"][1])*DT;frontier=max(frontier,net)
                pivot|=bool(f["head_contact"] and f["head_up"]<-.3 and .35<net<2.97)
                inverted|=bool(pivot and f["up"]<-.7)
                # Include the difficult inversion/recovery and the final hold.
                if k%5==4 and .4<net<7.5 and .1<w.t<2.5:
                    episode_samples.append(dict(qpos=w.mj.data.qpos.copy(),qvel=w.mj.data.qvel.copy(),
                        last=w.last.copy(),progress=np.array([net,frontier,pivot,inverted]),
                        heading=w.heading.copy(),source_time=w.t,seed=seed))
            result=summarize(task,rows,w.heading);reports.append(dict(seed=seed,**result))
            if result["success"]:samples.extend(episode_samples)
    finally:w.close()
    if len(samples)<20:raise RuntimeError("Insufficient successful source roll states")
    np.savez_compressed(path,**{key:np.asarray([s[key] for s in samples]) for key in samples[0]})
    path.with_suffix(".json").write_text(json.dumps(dict(source_sha256=policy.sha256,
        source=str(task.source),physics=w.physics,count=len(samples),episodes=reports),indent=2))
    return path


class RollStarts:
    def __init__(self,path=None):
        self.path=build_roll_starts(path);self.data=np.load(self.path)
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()

    def reset(self,w,rng):
        i=int(rng.integers(len(self.data["qpos"])))
        w.reset_from_roll_state(self.data["qpos"][i],self.data["qvel"][i],
            self.data["last"][i],self.data["heading"][i],self.data["progress"][i],source_time=self.data["source_time"][i])


if __name__=="__main__":print(build_roll_starts())
