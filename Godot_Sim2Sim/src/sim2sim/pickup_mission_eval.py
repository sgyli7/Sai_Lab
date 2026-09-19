"""MuJoCo A→B pilot: learned pickup, original walking, guarded setdown.

Reports physical stage failures. Navigation/setdown presently use privileged
world pose; this is a controller prototype, not the final camera-only robot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from sim2sim.pickup_env import PickupEnv
from sim2sim.policy import PolicyBundle


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",type=Path,required=True)
    p.add_argument("--pick-model",type=Path,required=True)
    p.add_argument("--walk-model",type=Path,required=True)
    p.add_argument("--ground-model",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--count",type=int,default=50)
    p.add_argument("--seed",type=int,default=445501)
    p.add_argument("--goal-x",type=float,default=.38)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    env=PickupEnv(args.scene,max_steps=900)
    pick=ort.InferenceSession(str(args.pick_model),providers=["CPUExecutionProvider"])
    input_name=pick.get_inputs()[0].name
    walk=PolicyBundle(args.walk_model)
    ground=PolicyBundle(args.ground_model)
    rows=[]
    for n in range(args.count):
        x=float(rng.uniform(.045,.115));y=float(rng.uniform(-.012,.012))
        obs=env.reset(args.seed+n,x=x,lateral=y)
        stage="pick"
        fell=False
        for k in range(200):
            action=np.asarray(pick.run(None,{input_name:obs[None,:]})[0],np.float32).reshape(-1)
            obs,_,_,info=env.step(action)
            if info["fell"]:fell=True;break
        pick_ok=bool(not fell and info["held"] and info["stable_hold_s"]>=1.)
        walk_steps=0;transport_ok=False;released=False;setdown_steps=0
        if pick_ok:
            stage="transport"
            for k in range(300):
                command=obs.copy();command[48:61]=0.;command[48]=.2
                action=walk.infer(command[:61])
                obs,_,_,info=env.step(np.r_[action,0.])
                walk_steps=k+1
                if info["fell"] or not info["held"]:fell=info["fell"];break
                if env.data.xpos[env.item,0]>=args.goal_x+.01:
                    transport_ok=True;break
        if transport_ok:
            stage="setdown"
            for k in range(200):
                command=obs.copy();command[48:61]=0.
                command[48]=math.cos(math.tau*k/200)
                command[49]=math.sin(math.tau*k/200)
                action=ground.infer(command[:61])
                obs,_,_,info=env.step(np.r_[action,.48 if released else 0.])
                speed=float(np.linalg.norm(env._body_velocity(env.item)))
                if (not released and k>=30 and env.data.xpos[env.item,2]<=.022
                        and speed<=.035):
                    env.data.eq_active[env.assist]=0
                    released=True
                    env.events.append({"event":"guarded_release","t":env.t,
                                       "item_xyz":env.data.xpos[env.item].tolist(),
                                       "speed_mps":speed})
                setdown_steps=k+1
                if info["fell"]:fell=True;break
        item_pos=env.data.xpos[env.item].copy()
        item_speed=float(np.linalg.norm(env._body_velocity(env.item)))
        robot_pos=env.data.xpos[env.robot].copy()
        goal_distance=float(np.linalg.norm(item_pos[:2]-np.array([args.goal_x,0.])))
        complete=bool(pick_ok and transport_ok and released and not fell
                      and not env.data.eq_active[env.assist] and goal_distance<=.07
                      and item_pos[2]<=.02 and item_speed<=.05 and info["upright"])
        row={"episode":n,"seed":args.seed+n,"start_xy":[x,y],"goal_xy":[args.goal_x,0.],
             "pick_ok":pick_ok,"transport_ok":transport_ok,"released":released,
             "complete":complete,"fell":fell,"walk_steps":walk_steps,
             "setdown_steps":setdown_steps,"item_xyz":item_pos.tolist(),
             "robot_xyz":robot_pos.tolist(),"goal_distance_m":goal_distance,
             "item_speed_mps":item_speed,"opposed_contact":env.opposed_contact_ever,
             "events":env.events}
        rows.append(row)
        if (n+1)%10==0:
            print(json.dumps({"episodes":n+1,"pick":sum(r["pick_ok"] for r in rows),
                              "transport":sum(r["transport_ok"] for r in rows),
                              "released":sum(r["released"] for r in rows),
                              "complete":sum(r["complete"] for r in rows)}),flush=True)
    (args.out/"episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    summary={"schema_version":1,"kind":"privileged_pose_mission_pilot",
             "pick_model_sha256":hashlib.sha256(args.pick_model.read_bytes()).hexdigest(),
             "walk_model_sha256":hashlib.sha256(args.walk_model.read_bytes()).hexdigest(),
             "ground_model_sha256":hashlib.sha256(args.ground_model.read_bytes()).hexdigest(),
             "scene_sha256":hashlib.sha256(args.scene.read_bytes()).hexdigest(),
             "count":len(rows),"seed":args.seed,"goal_x":args.goal_x,
             "pick":sum(r["pick_ok"] for r in rows),
             "transport":sum(r["transport_ok"] for r in rows),
             "released":sum(r["released"] for r in rows),
             "complete":sum(r["complete"] for r in rows),
             "limitations":"Uses privileged MuJoCo object pose/robot world pose; release uses assisted grasp. No camera student or Godot deployment."}
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
