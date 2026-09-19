"""Replay terrain patterns through the deployed controller, including mode recovery."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("sai_agent"), "Install the optional Sai extra")
class SaiTerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sai_agent.paths import resource_root
        from sim2sim.sai_controller import MotionController
        from scripts.sai_mujoco_motion import rollout
        cls.root = resource_root()
        cls.state = rollout(MotionController(cls.root), "stop", .02)["samples"][-1]

    def controller(self):
        from sim2sim.sai_controller import MotionController
        return MotionController(self.root)

    def request(self, scan, crouch=1., **extra):
        return dict(self.state, command=[.5, 0., crouch], terrain_heights=scan,
                    terrain_path_heights=extra.pop("terrain_path_heights", [0.] * 15), **extra)

    def test_distant_side_and_behind_steps_do_not_interrupt_crouch_cruise(self):
        for indices in ([21, 22, 23], [12], [0, 1, 2]):
            with self.subTest(indices=indices):
                scan = [0.] * 24
                for i in indices: scan[i] = .02
                result = self.controller().command(self.request(scan))
                self.assertNotEqual(result["stage"], "stairs")
                self.assertAlmostEqual(result["policy_observation"][9], .5)

    def test_crouch_rolls_small_seams_but_keeps_real_step_response(self):
        controller = self.controller()
        seam = [0.] * 9 + [.006] * 15
        result = controller.command(self.request(seam, terrain_path_heights=[0.] * 6 + [.006] * 9))
        self.assertNotEqual(result["stage"], "stairs")
        real_step = [0.] * 9 + [.02] * 15
        result = controller.command(self.request(real_step, terrain_path_heights=[0.] * 6 + [.02] * 9))
        self.assertEqual(result["stage"], "crouch_blocked")
        self.assertEqual(result["policy_observation"][9], 0.)
        near = self.request(real_step, terrain_path_heights=[0.] * 6 + [.02] * 9)
        retreat = controller.command(dict(near, command=[-.5, 0., 1.]))
        self.assertNotEqual(retreat["stage"], "crouch_blocked")
        self.assertAlmostEqual(retreat["policy_observation"][9], -.5)
        turn = controller.command(dict(near, command=[.5, .45, 1.]))
        self.assertAlmostEqual(turn["policy_observation"][10], .45, places=6)
        result = controller.command(self.request(real_step, crouch=0., terrain_path_heights=[0.] * 6 + [.02] * 9))
        self.assertEqual(result["stage"], "stairs")
        self.assertLessEqual(result["policy_observation"][9], .121)
        # Do not retain a disabled or stale stair actor after switching modes.
        result = controller.command(self.request([0.] * 24, time=.04))
        self.assertNotEqual(result["stage"], "stairs")
        self.assertAlmostEqual(result["policy_observation"][9], .5)

    def test_explicit_stair_course_retains_published_lookahead(self):
        scan = [0.] * 21 + [.02] * 3
        result = self.controller().command(self.request(scan, crouch=0., stair_course=True))
        self.assertEqual(result["stage"], "stairs")


if __name__ == "__main__":
    unittest.main()
