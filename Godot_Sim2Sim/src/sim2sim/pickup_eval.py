"""Evaluate a pickup ONNX actor in fresh MuJoCo episodes and save failures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from sim2sim.pickup_env import PickupEnv, MAX_STEPS
from sim2sim.policy import PolicyBundle


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=991301)
    p.add_argument("--xmin", type=float, default=.045)
    p.add_argument("--xmax", type=float, default=.115)
    p.add_argument("--ablate-target", action="store_true",
                   help="Zero privileged object position/dimensions/velocity at inference")
    p.add_argument("--spoof-target", action="store_true",
                   help="Shift only the relative target position to another valid workspace point")
    p.add_argument("--fixed-source", type=Path,
                   help="Evaluate the frozen original actor with a fixed .46 s beak close")
    p.add_argument("--suction-on", action="store_true",
                   help="Command the finite-force contact adhesion actuator throughout pickup")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"]) if not args.fixed_source else None
    input_name = session.get_inputs()[0].name if session else None
    source = PolicyBundle(args.fixed_source) if args.fixed_source else None
    env = PickupEnv(args.scene)
    rng = np.random.default_rng(args.seed)
    rows = []
    for n in range(args.count):
        x = float(rng.uniform(args.xmin,args.xmax))
        y = float(rng.uniform(-.012,.012))
        try:
            obs = env.reset(args.seed+n,x=x,lateral=y)
        except ValueError as exc:
            if "Initial object-robot penetration" not in str(exc):
                raise
            rows.append({"episode":n,"seed":args.seed+n,"target_x":x,"target_y":y,
                         "invalid_initial_state":True,"reason":str(exc),
                         "success":False,"physical_success":False,
                         "opposed_contact":False,"grip":False})
            continue
        if args.suction_on:
            if not env.adhesion_mode:
                raise ValueError("--suction-on requires an adhesion scene")
            env.suction_request=True
        jaw_trace=[]
        for k in range(MAX_STEPS):
            actor_obs = obs.copy()
            if args.ablate_target:
                actor_obs[61:70] = 0.
            if args.spoof_target:
                # A physically plausible alternate x, transformed into the
                # current head frame; shape, mass, velocity stay untouched.
                delta_x = .025 if x < .08 else -.025
                rotation = env.data.xmat[env.head].reshape(3,3)
                actor_obs[61:64] = rotation.T @ (
                    env.data.site_xpos[env.item_grip_site] + np.array([delta_x,0.,0.])
                    - env.data.site_xpos[env.tip])
            if source is not None:
                action = np.r_[source.infer(actor_obs[:61]), .48 if k*.02 < .46 else 0.].astype(np.float32)
            else:
                action = np.asarray(session.run(None,{input_name:actor_obs[None,:]})[0],np.float32).reshape(-1)
            if action.shape != (15,) or not np.isfinite(action).all():
                raise RuntimeError(f"Bad ONNX output at episode {n}, step {k}")
            jaw_trace.append(float(action[14]))
            obs,reward,done,info=env.step(action)
            if done:break
        row={"episode":n,"seed":args.seed+n,"target_x":x,"target_y":y,
             "invalid_initial_state":False,
             "success":bool(info["success"]),
             "clearance_success_1mm":bool(info["success"] and
                 (info["min_object_robot_dist_m"] is None or
                  info["min_object_robot_dist_m"] >= -.001)),
             "physical_success":bool(info["physical_success"]),
             "min_suction_contact_dist_m":info["min_suction_contact_dist_m"],
             "min_object_robot_dist_m":info["min_object_robot_dist_m"],
             "max_commanded_adhesion_force_n":info["max_commanded_adhesion_force_n"],
             "max_pad_contact_force_n":info["max_pad_contact_force_n"],
             "pad_contact_s":info["pad_contact_s"],
             "max_contact_force_n":info["max_contact_force_n"],
             "opposed_contact":env.opposed_contact_ever,
             "grip":env.grip_ever,"peak_lift_m":float(env.peak_lift),
             "stable_hold_s":float(info["stable_hold_s"]),
             "jaw_first_s":jaw_trace[:50],"events":env.events}
        rows.append(row)
        if (n+1)%20==0:
            print(json.dumps({"episodes":n+1,"successes":sum(r["success"] for r in rows),
                              "opposed_contacts":sum(r["opposed_contact"] for r in rows)}),flush=True)
    (args.out/"episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    summary={"schema_version":1,"kind":"fresh_mujoco_rollout",
             "model_sha256":hashlib.sha256((args.fixed_source or args.model).read_bytes()).hexdigest(),
             "scene_sha256":hashlib.sha256(args.scene.read_bytes()).hexdigest(),
             "robot_sha256":hashlib.sha256((args.scene.parent/"robot_pickable.xml").read_bytes()).hexdigest(),
             "seed":args.seed,"count":args.count,"x_range":[args.xmin,args.xmax],
             "invalid_initial_states":sum(r["invalid_initial_state"] for r in rows),
             "target_ablation":args.ablate_target,
             "target_spoofed":args.spoof_target,"fixed_source":bool(args.fixed_source),
             "suction_on":args.suction_on,
             "successes":sum(r["success"] for r in rows),
             "clearance_successes_1mm":sum(r.get("clearance_success_1mm",False) for r in rows),
             "physical_successes":sum(r["physical_success"] for r in rows),
             "opposed_contacts":sum(r["opposed_contact"] for r in rows),
             "grips":sum(r["grip"] for r in rows)}
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
