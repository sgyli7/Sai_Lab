"""Re-score archived physical traces without replacing the original results."""
import argparse,json,math
from pathlib import Path
import numpy as np
from .tasks import TASKS,SESSION
from .evaluate import summarize,PROTOCOL_VERSION


def rescore(path):
    path=Path(path);old=json.loads(path.read_text());episodes=[]
    for original in old["episodes"]:
        episode=original.copy()
        if "error" not in episode:
            if episode.get("physics",{}).get("joint_limits")!="signed_fresh_reset_v1":
                raise ValueError("Only invariant-physics traces may enter this comparison")
            if episode.get("backend")=="mujoco" and episode.get("physics",{}).get("velocity_metric")!="trunk_inertial_com_v2":
                raise ValueError("MuJoCo COM velocity requires fresh reference rollouts")
            with np.load(episode["trace"]) as archive:
                a={k:archive[k] for k in archive.files if k!="obs"}
            n=len(a["time"])
            if "heading" in episode:heading=np.array(episode["heading"])
            else:
                if episode.get("entry","reset")!="reset":raise ValueError("Missing handoff heading")
                yaw=np.random.default_rng(episode["seed"]).uniform(-math.pi,math.pi)
                heading=np.array([math.cos(yaw),math.sin(yaw)])
            rows=[{k:value[i] for k,value in a.items()} for i in range(n)]
            episode.update(summarize(TASKS[old["skill"]],rows,heading))
            episode.update(protocol=PROTOCOL_VERSION,rescored_from=original.get("protocol"))
        episodes.append(episode)
    result=old.copy();result.update(protocol=PROTOCOL_VERSION,rescored_from=str(path),episodes=episodes,
        success_rate=float(np.mean([e["success"] for e in episodes])),score=float(np.mean([e["score"] for e in episodes])))
    target=path.with_name("summary."+PROTOCOL_VERSION.rsplit("_",1)[-1]+".json")
    target.write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument("paths",nargs="*",type=Path)
    args=p.parse_args();paths=args.paths or list(SESSION.rglob("summary.json"))
    for path in paths:
        if "conditioning_probe" in path.parts:continue
        d=json.loads(path.read_text())
        if d.get("protocol") not in ("physical_tasks_v3","physical_tasks_v4","physical_tasks_v5","physical_tasks_v6"):continue
        try:
            r=rescore(path);print(path,r["success_rate"],round(r["score"],4),flush=True)
        except (KeyError,ValueError,FileNotFoundError) as e:print("SKIP",path,str(e),flush=True)


if __name__=="__main__":main()
