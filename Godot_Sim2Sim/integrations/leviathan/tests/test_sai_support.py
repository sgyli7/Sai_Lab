"""Support-frame control invariants; native contact acceptance is separate."""
import copy
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
from sai_agent.paths import resource_root

from sim2sim.sai_controller import MotionController
from sim2sim.workshop_grab import WorkshopController


def state_for(controller, time=0., rotation=None, origin=None, moving=True, command=None):
    rotation = np.eye(3) if rotation is None else rotation
    origin = np.zeros(3) if origin is None else origin
    robot_origin = np.asarray(controller.spec["bodies"]["chassis"]["origin_m"])
    state = dict(robot_id="Sai_Agent_001", physics_owner="Godot/Jolt", time=time,
                 base_position=(rotation @ robot_origin+origin).tolist(),
                 base_rotation_columns=rotation.T.tolist(), base_linear_world=[.2, -.1, .03] if moving else [0.]*3,
                 base_angular_world=[.001, .002, .03] if moving else [0.]*3,
                 q=[0.]*25, v=[0.]*25, command=[0.,0.,0.] if command is None else command,
                 terrain_heights=[float(origin[2])]*24, tool_m=[0.]*3)
    state["support_frame"] = dict(body_id="front", base_position=robot_origin.tolist(),
                                  base_rotation_columns=np.eye(3).tolist(), base_linear_world=[0.]*3,
                                  base_angular_world=[0.]*3, projected_up_body=rotation[2].tolist(),
                                  terrain_heights=[0.]*24, terrain_path_heights=[0.]*15)
    state["support_frames"] = {"front": dict(origin_m=origin.tolist(), rotation_columns=rotation.T.tolist())}
    return state


def test_turning_platform_removes_carried_twist_and_keeps_real_gravity():
    root = resource_root()
    c = MotionController(root)
    rotation = Rotation.from_euler("xyz", [.06,-.04,.3]).as_matrix()
    state = state_for(c, rotation=rotation, origin=np.array([145.,18.,12.]))
    with patch("mujoco.mj_step", side_effect=AssertionError("Controller advanced physics")):
        result = c.command(state)
    np.testing.assert_allclose(result["policy_observation"][:3], rotation[2], atol=1e-7)
    np.testing.assert_allclose(result["policy_observation"][3:9], 0., atol=1e-8)
    # Bias is computed from actual world attitude/twist, not level fake gravity.
    reference = MotionController(root).command({k:v for k,v in state.items() if k not in ("support_frame","support_frames")})
    np.testing.assert_allclose(result["arm_bias"], reference["arm_bias"], atol=1e-10)
    assert result["mode"] == "braked"
    assert not result["physics_advanced_by_controller"]


def test_flat_world_and_measured_level_support_have_identical_moving_actor_input():
    root = resource_root()
    a, b = MotionController(root), MotionController(root)
    for index, command in enumerate(([.16,0.,0.],[.10,.2,0.],[-.1,0.,1.])):
        measured = state_for(a,index*.02,moving=False,command=command)
        world = {k:v for k,v in measured.items() if k not in ("support_frame","support_frames")}
        world["terrain_path_heights"] = [0.]*15
        x,y = a.command(measured),b.command(world)
        np.testing.assert_array_equal(x["policy_observation"],y["policy_observation"])
        np.testing.assert_array_equal(x["target_leg"],y["target_leg"])


def test_park_brake_holds_real_encoder_angles_and_releases_for_drive():
    c = MotionController(resource_root())
    state = state_for(c)
    turns = np.array([12.,-8.,6.,-2.])
    for i,value in enumerate(turns):state["q"][i*4+3]=value
    first = c.command(state)
    for i in range(4):state["q"][i*4+3]+=.02
    second = c.command(state)
    np.testing.assert_array_equal(np.array(first["target_leg"])[3::4],turns)
    np.testing.assert_array_equal(np.array(second["target_leg"])[3::4],turns)
    moving = c.command(dict(state,command=[.1,0.,0.]))
    assert moving["mode"] == "transport"
    last = c.command(state)
    np.testing.assert_allclose(np.array(last["target_leg"])[3::4],turns+.02)


def test_pickup_target_follows_measured_deck_after_docking():
    root=resource_root()
    still,moving=WorkshopController(root),WorkshopController(root)
    for time in [0.,2.,4.,8.]:
        rotation=Rotation.from_euler("z",time*.03).as_matrix()
        origin=np.array([145.+time*.2,-5.,12.])
        packets=[]
        for c,r,p in [(still,np.eye(3),np.zeros(3)),(moving,rotation,origin)]:
            state=state_for(c,time,rotation=r,origin=p)
            state["workshop_grab"]=dict(serial=1,request="pick",target_m=(r@np.array([.24,0.,.03])+p).tolist(),
                target_support_body_id="front",object="Bottle6g",rest_height_m=.032,held=False,
                held_offset_m=[0.,0.,-.02],slot=0,busy=True)
            packets.append(c.command(state))
        a,b=packets
        assert b["mode"]=="manipulation"  # .2 m/s world speed must not prevent docking.
        np.testing.assert_allclose(rotation@a["tool_target_m"]+origin,b["tool_target_m"],atol=1e-7)
        assert b["task_support_body_id"]=="front"
        assert np.isfinite(b["target_arm"]).all()
