"""Derive a MuJoCo pickup scene with a physical fifteenth beak actuator.

The upstream robot XML and assets are inputs. This module writes a separate
experiment scene and never changes the nine released robot/policy contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim2sim.paths import load_robot_json


BEAK_PIVOT_BODY = np.array([-0.00112156, -0.00057860, 0.01230156])
BEAK_AXIS_BODY = np.array([0.0055038, 0.99916835, 0.04040203])
MOVING_MESHES = {"jaw", "bottom_head_shell"}
SHAPES = {
    # The two 4 mm grip pads have about 35 mm net clearance at full opening.
    # A 70 mm ball cannot be enclosed
    # by that geometry, regardless of policy quality. Keep it as a negative
    # reachability control and start training with physically graspable items.
    "ball": {"type": "sphere", "size": "0.035", "mass": 0.010, "height": 0.035},
    "bead": {"type": "sphere", "size": "0.009", "mass": 0.002, "height": 0.009},
    "bottle": {"type": "cylinder", "size": "0.008 0.035", "mass": 0.004, "height": 0.035},
    "box": {"type": "box", "size": "0.008 0.008 0.035", "mass": 0.006, "height": 0.035},
    "bottle_full": {"type": "composite", "mass": 0.008, "height": 0.039,
                    "dimensions": [0.036, 0.036, 0.078], "grip_z": 0.025},
    "bin_full": {"type": "composite", "mass": 0.012, "height": 0.025,
                 "dimensions": [0.090, 0.064, 0.050],
                 "grip_xyz": [-0.0435, 0., 0.0235]},
}


def _vector(values: np.ndarray) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def generate(out: Path, shape: str = "ball", gripper: str = "clamp") -> dict:
    if shape not in SHAPES:
        raise ValueError(f"Unknown pickup shape: {shape}")
    if gripper not in ("clamp", "adhesion", "adhesion_dual", "contact_assist", "contact_assist_dual"):
        raise ValueError(f"Unknown gripper: {gripper}")
    source_scene = Path(load_robot_json("robots/microduck.json")["mjcf"])
    source_robot = source_scene.with_name("robot_allcollisions.xml")
    output = Path(out).resolve()
    output.mkdir(parents=True, exist_ok=True)

    robot = ET.parse(source_robot)
    jaw = robot.find(".//body[@name='jaw_soft']")
    if jaw is None:
        raise RuntimeError("Upstream jaw_soft is missing")
    moving = [geom for geom in jaw.findall("geom") if geom.get("mesh") in MOVING_MESHES]
    if len(moving) != 4:
        raise RuntimeError(f"Expected jaw and bottom shell visual/collision pairs, found {len(moving)}")
    for geom in moving:
        jaw.remove(geom)
    lower = ET.SubElement(jaw, "body", name="lower_beak", pos=_vector(BEAK_PIVOT_BODY))
    ET.SubElement(lower, "joint", name="beak_pitch", type="hinge",
                  axis=_vector(BEAK_AXIS_BODY), range="0 0.48", damping="0.002",
                  armature="0.00002", frictionloss="0.0001")
    # The original jaw_soft mass included these moving shells. The 20 g split
    # is a versioned starting estimate; physical tuning uses measured parts.
    jaw.find("inertial").set("mass", "0.168766")
    ET.SubElement(lower, "inertial", pos="0 0 -0.035", mass="0.020",
                  diaginertia="0.000010 0.000010 0.000010")
    for geom in moving:
        geom.set("pos", _vector(np.fromstring(geom.get("pos"), sep=" ") - BEAK_PIVOT_BODY))
        if geom.get("class") == "collision":
            geom.set("name", "lower_beak_collision" if geom.get("mesh") == "jaw" else "lower_shell_collision")
            geom.set("priority", "2")
            geom.set("solref", "0.01 1")
            geom.set("solimp", "0.99 0.999 0.001 0.5 2")
        lower.append(geom)
    upper_collision = next((g for g in jaw.findall("geom")
                            if g.get("mesh") == "top_head_shell" and g.get("class") == "collision"), None)
    if upper_collision is None:
        raise RuntimeError("Upper beak collision surface is missing")
    upper_collision.set("name", "upper_beak_collision")
    upper_collision.set("priority", "2")
    upper_collision.set("solref", "0.01 1")
    upper_collision.set("solimp", "0.99 0.999 0.001 0.5 2")
    original_tip = jaw.find("site[@name='mouth_tip']")
    if original_tip is None:
        raise RuntimeError("Upstream mouth_tip is missing")
    tip_local = original_tip.get("pos")
    # Small compliant gripping surfaces at the actual mouth tips. The source
    # mesh collision hulls model the external shell, not the interior bite.
    ET.SubElement(jaw, "geom", name="upper_grip_pad", type="sphere", pos=tip_local,
                  size="0.004", mass="0.0005", friction="1.2 0.01 0.001",
                  priority="2", solref="0.01 1", solimp="0.99 0.999 0.001 0.5 2",
                  rgba="0.15 0.15 0.15 1")
    lower_tip_local = np.fromstring(tip_local, sep=" ") - BEAK_PIVOT_BODY
    ET.SubElement(lower, "site", name="lower_mouth_tip",
                  pos=_vector(lower_tip_local),
                  size="0.002", rgba="1 0.6 0 1")
    ET.SubElement(lower, "geom", name="lower_grip_pad", type="sphere",
                  pos=_vector(lower_tip_local), size="0.004", mass="0.0005",
                  friction="1.2 0.01 0.001", rgba="0.15 0.15 0.15 1")
    lower.find("geom[@name='lower_grip_pad']").set("priority", "2")
    lower.find("geom[@name='lower_grip_pad']").set("solref", "0.01 1")
    lower.find("geom[@name='lower_grip_pad']").set("solimp", "0.99 0.999 0.001 0.5 2")
    if gripper.startswith("adhesion") or gripper.startswith("contact_assist"):
        # A separate contact-only suction surface lets MuJoCo's adhesion
        # actuator apply its bounded force only to this pad, never to the
        # whole head shell or remotely to the object.
        suction = ET.SubElement(lower, "body", name="suction_cup",
                                pos=_vector(lower_tip_local))
        ET.SubElement(suction, "geom", name="suction_contact", type="sphere",
                      size="0.008", mass="0.001", friction="1.2 0.01 0.001",
                      contype="2", conaffinity="2", priority="2",
                      solref="0.01 1", solimp="0.99 0.999 0.001 0.5 2",
                      rgba="0.12 0.12 0.12 1")
        if gripper.endswith("_dual"):
            ET.SubElement(suction, "geom", name="suction_contact_lower", type="sphere",
                          pos="0 0 -0.015", size="0.010", mass="0.001",
                          friction="1.2 0.01 0.001", contype="2", conaffinity="2",
                          priority="2", solref="0.01 1",
                          solimp="0.99 0.999 0.001 0.5 2",
                          rgba="0.12 0.12 0.12 1")
    # Upstream's camera points backward through an opaque shell. The pickup
    # experiment needs a real forward/downward image, not privileged geometry
    # presented as vision. Keep a fixed head mount and record its calibration.
    head_camera = jaw.find("camera[@name='head_camera']")
    if head_camera is None:
        raise RuntimeError("Upstream head camera is missing")
    head_camera.set("pos", "0.0155 0 -0.105")
    head_camera.set("quat", "0.793353 0 0.608761 0")  # sees floor targets 5–15 cm ahead
    head_camera.set("fovy", "90")
    actuator = robot.getroot().find("actuator")
    if actuator is None:
        raise RuntimeError("Upstream actuators are missing")
    ET.SubElement(actuator, "position", name="beak_pitch", joint="beak_pitch",
                  kp="0.5", kv="0.01", forcerange="-0.08 0.08", ctrlrange="0 0.48")
    if gripper.startswith("adhesion"):
        ET.SubElement(actuator, "adhesion", name="beak_suction", body="suction_cup",
                      ctrlrange="0 1", gain="1.5" if gripper == "adhesion_dual" else "1.0")
    (output / "robot_pickable.xml").write_bytes(ET.tostring(robot.getroot(), encoding="utf-8"))

    scene = ET.parse(source_scene)
    include = scene.find("include")
    if include is None:
        raise RuntimeError("Upstream scene include is missing")
    include.set("file", "robot_pickable.xml")
    params = SHAPES[shape]
    world = scene.getroot().find("worldbody")
    obj = ET.SubElement(world, "body", name="pickup_item",
                        pos=f"0.105 0 {params['height']}")
    ET.SubElement(obj, "freejoint", name="pickup_item_freejoint")
    if shape == "bottle_full":
        # Original Godot bottle: 36 mm maximum diameter, 78 mm tall, 8 g.
        # The neck is 16 mm wide, but the broad base and inertia remain.
        parts = [
            ("pickup_item_geom", "cylinder", "0.018 0.0225", "0 0 -0.0165", .0065),
            ("bottle_shoulder", "cylinder", "0.013 0.0065", "0 0 0.0125", .0007),
            ("pickup_neck_geom", "cylinder", "0.008 0.006", "0 0 0.025", .0004),
            ("bottle_cap", "cylinder", "0.010 0.004", "0 0 0.035", .0004),
        ]
        for name, kind, size, pos, mass in parts:
            ET.SubElement(obj, "geom", name=name, type=kind, size=size, pos=pos,
                          mass=str(mass), friction="0.8 0.01 0.001", rgba="0.95 0.65 0.08 1",
                          contype="3" if gripper != "clamp" else "1",
                          conaffinity="3" if gripper != "clamp" else "1")
    elif shape == "bin_full":
        # Open 90×64×50 mm tub with 3 mm floor and walls, matching Godot.
        parts = [("pickup_item_geom", "0.045 0.032 0.0015", "0 0 -0.0235", .002),
                 ("bin_x_left", "0.0015 0.032 0.0235", "-0.0435 0 0.0015", .0025),
                 ("bin_x_right", "0.0015 0.032 0.0235", "0.0435 0 0.0015", .0025),
                 ("bin_y_left", "0.042 0.0015 0.0235", "0 -0.0305 0.0015", .0025),
                 ("bin_y_right", "0.042 0.0015 0.0235", "0 0.0305 0.0015", .0025)]
        for name, size, pos, mass in parts:
            ET.SubElement(obj, "geom", name=name, type="box", size=size, pos=pos,
                          mass=str(mass), friction="0.8 0.01 0.001", rgba="0.95 0.65 0.08 1",
                          contype="3" if gripper != "clamp" else "1",
                          conaffinity="3" if gripper != "clamp" else "1")
    else:
        ET.SubElement(obj, "geom", name="pickup_item_geom", type=params["type"],
                      size=params["size"], mass=str(params["mass"]),
                      friction="0.8 0.01 0.001", rgba="0.95 0.65 0.08 1",
                      contype="3" if gripper != "clamp" else "1",
                      conaffinity="3" if gripper != "clamp" else "1")
    grip_xyz = params.get("grip_xyz", [0., 0., params.get("grip_z", params["height"]*.85)])
    ET.SubElement(obj, "site", name="pickup_grip", pos=_vector(np.asarray(grip_xyz)),
                  size="0.002")
    equality = ET.SubElement(scene.getroot(), "equality")
    # Body-body anchors are rewritten at the measured contact point immediately
    # before activation, so the assist does not pull distant sites together.
    ET.SubElement(equality, "connect", name="assist_grip", body1="lower_beak",
                  body2="pickup_item", anchor="0 0 0", active="false", solref="0.01 1")
    (output / "scene_pickable.xml").write_bytes(ET.tostring(scene.getroot(), encoding="utf-8"))
    assets = source_scene.parent / "assets"
    link = output / "assets"
    if link.exists() and link.resolve() != assets.resolve():
        raise FileExistsError(f"Unexpected asset link: {link}")
    if not link.exists():
        link.symlink_to(assets, target_is_directory=True)
    model = mujoco.MjModel.from_xml_path(str(output / "scene_pickable.xml"))
    if model.nu != (16 if gripper.startswith("adhesion") else 15):
        raise RuntimeError(f"Expected {'16' if gripper.startswith('adhesion') else '15'} controls, got {model.nu}")
    info = {
        "schema_version": 1,
        "shape": shape,
        "gripper": gripper,
        "suction_force_limit_n": (1.5 if gripper == "adhesion_dual" else 1.0)
        if gripper.startswith("adhesion") else None,
        "contact_assist_force_break_n": 1.5 if gripper.startswith("contact_assist") else None,
        "source_scene": str(source_scene),
        "source_scene_sha256": hashlib.sha256(source_scene.read_bytes()).hexdigest(),
        "source_robot": str(source_robot),
        "source_robot_sha256": hashlib.sha256(source_robot.read_bytes()).hexdigest(),
        "scene_sha256": hashlib.sha256((output / "scene_pickable.xml").read_bytes()).hexdigest(),
        "robot_sha256": hashlib.sha256((output / "robot_pickable.xml").read_bytes()).hexdigest(),
        "actuators": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)],
        "pickup_mass_kg": params["mass"],
        "overall_dimensions_xyz_m": params.get("dimensions"),
        "nominal_grip_site_xyz_m": grip_xyz,
        "assist_grip_initially_active": bool(model.eq_active0[model.equality("assist_grip").id]),
    }
    (output / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--shape", choices=SHAPES, default="ball")
    p.add_argument("--gripper", choices=("clamp", "adhesion", "adhesion_dual",
                                          "contact_assist", "contact_assist_dual"), default="clamp")
    args = p.parse_args()
    print(json.dumps(generate(args.out, args.shape, args.gripper), indent=2))


if __name__ == "__main__":
    main()
