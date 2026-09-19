"""Replay the deployed Sai controller against CPU MuJoCo's articulated robot.

The controller still receives the Godot wire contract; only this test harness
advances MuJoCo. Runtime Godot physics remains independent.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from sai_agent.paths import resource_root
from sai_agent.runtime import JointAdapter


CASES = {"stop": (0., 0.), "W": (.16, 0.), "S": (-.16, 0.),
         "A": (0., .45), "D": (0., -.45), "WA": (.14, .3),
         "shift": (0., 0.), "W_shift": (.14, 0.)}


def rollout(controller, case="W", seconds=12., yaw=0., command_at=None):
    root = resource_root()
    model = mujoco.MjModel.from_xml_path(str(root / "models/full/locomotion-articulated.xml"))
    data = mujoco.MjData(model)
    data.qpos[3:7] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
    adapter = JointAdapter(model)
    names = [b["joint"]["name"] for b in controller.spec["bodies"].values() if b.get("parent") is not None]
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    vadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in names])
    wheels = [model.body(n + "_wheel").id for n in ("front_left", "front_right", "rear_left", "rear_right")]
    chassis = model.body("chassis").id
    samples = []
    for k in range(round(seconds/.02) + 1):
        mujoco.mj_forward(model, data)
        t = k*.02
        rotation = data.xmat[chassis].reshape(3, 3)
        vx, wz = CASES[case] if t >= 1 else (0., 0.)
        state = dict(robot_id="Sai_Agent_001", physics_owner="Godot/Jolt", time=t,
                     q=data.qpos[qadr].tolist(), v=data.qvel[vadr].tolist(),
                     base_position=data.qpos[:3].tolist(), base_rotation_columns=rotation.T.tolist(),
                     base_linear_world=data.qvel[:3].tolist(),
                     base_angular_world=(rotation @ data.qvel[3:6]).tolist(),
                     command=[vx, wz, float("shift" in case and 3 <= t < 8)],
                     terrain_heights=[0.]*24)
        if command_at is not None:
            state["command"] = list(command_at(t))
        command = controller.command(state)
        contacts = {int(model.geom_bodyid[g]) for contact in data.contact for g in contact.geom}
        state.update(policy_action=command["policy_action"], policy_observation=command["policy_observation"],
                     target_leg=command["target_leg"], upright=float(rotation[2, 2]),
                     wheel_positions=data.xpos[wheels].tolist(), wheels_supported=len(contacts.intersection(wheels)),
                     controller_stage=command["stage"])
        samples.append(state)
        for _ in range(round(.02/model.opt.timestep)):
            adapter.apply(data, np.asarray(command["target_leg"]))
            mujoco.mj_step(model, data)
    return {"simulator": "CPU MuJoCo", "body_count": 26, "hinges": 23, "sliders": 2,
            "case": case, "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="W")
    parser.add_argument("--seconds", type=float, default=5.)
    parser.add_argument("--yaw", type=float, default=0.)
    parser.add_argument("--release-policy", action="store_true", help="Reproduce the original alpha.3 defect")
    parser.add_argument("--policy", type=Path, help="Evaluate a candidate ONNX without installing it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.release_policy or args.policy:
        from sai_agent.godot_controller import GodotController
        controller = GodotController(resource_root())
        if args.policy:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            controller.policy = ort.InferenceSession(str(args.policy), options, providers=["CPUExecutionProvider"])
    else:
        from sim2sim.sai_controller import MotionController
        controller = MotionController(resource_root())
    result = rollout(controller, args.case, args.seconds, args.yaw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
