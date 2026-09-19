"""Controller regression checks; physical placement is tested by the Godot replay."""
import importlib.util
import json
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("sai_agent"), "Install the optional Sai extra")
class WorkshopGrabTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from sai_agent.paths import resource_root
        from sim2sim.workshop_grab import WorkshopController
        self.np = np
        self.root = resource_root()
        self.controller = WorkshopController(self.root)

    def state(self, time=0., rotation=None, offset=None, request=None):
        np = self.np
        rotation = np.eye(3) if rotation is None else rotation
        offset = np.zeros(3) if offset is None else offset
        origin = np.array(self.controller.spec["bodies"]["chassis"]["origin_m"])
        state = dict(robot_id="Sai_Agent_001", physics_owner="Godot/Jolt", time=time,
                     base_position=(rotation @ origin + offset).tolist(), base_rotation_columns=rotation.T.tolist(),
                     base_linear_world=[0.] * 3, base_angular_world=[0.] * 3, q=[0.] * 25, v=[0.] * 25,
                     command=[0., 0., 0.], terrain_heights=[0.] * 24, tool_m=[0.] * 3)
        if request is not None:
            state["workshop_grab"] = dict(serial=1, request=request, target_m=(rotation @ np.array([.24, 0., .03]) + offset).tolist(),
                                         object="Bottle6g", rest_height_m=.032, held=False, held_offset_m=[0.,0.,-.02], slot=0, busy=request == "pick")
        return state

    def test_idle_commands_preserve_deployed_locomotion(self):
        from sim2sim.sai_controller import MotionController
        baseline = MotionController(self.root)
        for i, command in enumerate(([0., 0., 0.], [.16, 0., 0.], [0., -.45, 1.], [-.16, 0., 0.])):
            state = self.state(i * .02, request="idle")
            state["command"] = command
            self.assertEqual(baseline.command(state), self.controller.command(state))

    def test_ik_targets_follow_translated_rotated_chassis_without_physics_steps(self):
        from scipy.spatial.transform import Rotation
        from sim2sim.workshop_grab import WorkshopController
        np = self.np
        rotated = WorkshopController(self.root)
        rotation = Rotation.from_euler("z", 1.1).as_matrix()
        offset = np.array([.8, -.6, 0.])
        with patch("mujoco.mj_step", side_effect=AssertionError("Controller advanced physics")):
            for time in [0., 2., 4., 8.]:
                a = self.controller.command(self.state(time, request="pick"))
                b = rotated.command(self.state(time, rotation, offset, "pick"))
                np.testing.assert_allclose(rotation @ a["tool_target_m"] + offset, b["tool_target_m"], atol=1e-7)
                self.assertTrue(np.isfinite(b["target_arm"]).all())
                self.assertFalse(b["physics_advanced_by_controller"])

    def test_cancel_releases_control_and_repeated_serial_does_not_restart_task(self):
        self.controller.command(self.state(0., request="pick"))
        task = self.controller.task
        self.controller.command(self.state(1., request="pick"))
        self.assertIs(self.controller.task, task)
        state = self.state(2., request="cancel")
        state["workshop_grab"]["serial"] = 2
        result = self.controller.command(state)
        self.assertEqual(self.controller.phase, "idle")
        self.assertEqual(result["mode"], "transport")
        self.assertNotIn("assist_grip", result)

    def test_unreachable_target_times_out_without_grip(self):
        state = self.state(0., request="pick")
        state["workshop_grab"]["target_m"] = [.9, 0., .03]
        self.controller.command(state)
        state["time"] = 25.
        result = self.controller.command(state)
        self.assertEqual(result["grab_stage"], "failed")
        self.assertNotIn("assist_grip", result)

    def test_manipulation_keeps_wheel_turns_accumulated_while_docking(self):
        np = self.np
        state = self.state(0., request="pick")
        turns = [4.2, -3.7, 4.1, -3.6]
        for i, value in enumerate(turns): state["q"][4 * i + 3] = value
        result = self.controller.command(state)
        np.testing.assert_allclose(np.array(result["target_leg"])[3::4], turns, atol=1e-6)

    def test_retargeted_transfer_remains_reachable_at_wrist_limit(self):
        self.controller.command(self.state(0., request="pick"))
        for time in [18., 22., 25., 30., 34.5]:
            state = self.state(time, request="pick")
            state["workshop_grab"]["held"] = True
            result = self.controller.command(state)
            json.dumps(result, allow_nan=False)
            self.assertLess(result["IK_target_error_m"], .015, f"Unreachable transfer at {time} s")


if __name__ == "__main__":
    unittest.main()
