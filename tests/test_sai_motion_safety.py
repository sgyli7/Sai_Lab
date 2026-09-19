import unittest

from sim2sim.sai_motion_safety import assess_motion_safety


def sample(index, *, unsafe=False):
    q = [0.0] * 16
    v = [0.0] * 16
    action = [0.1] * 16
    target = [0.001 * index] * 16
    if unsafe:
        q[0] = 0.449
        q[5] = 0.699
        v[0] = 15.0
        for axis in (0, 4, 8, 12):
            action[axis] = 1.0
        target[0] = 0.4 if index % 2 else -0.4
    return {"time": index * 0.02, "controller_stage": "stairs", "q": q, "v": v,
            "policy_action": action, "controller_command": {"target_leg": target},
            "upright": 0.94 if unsafe else 0.999, "wheels_supported": 1 if unsafe else 4}


class SaiMotionSafetyTests(unittest.TestCase):
    def test_nominal_trace_passes(self):
        report = assess_motion_safety([sample(i) for i in range(50)], riser_m=0.04)
        self.assertTrue(report["passed"], report)

    def test_twisted_limit_saturated_trace_fails(self):
        report = assess_motion_safety([sample(i, unsafe=True) for i in range(50)], riser_m=0.04)
        self.assertFalse(report["passed"])
        for check in ("hard_stop_fraction", "haa_soft_stop_fraction", "action_saturation",
                      "target_slew", "measured_joint_speed", "upright", "support"):
            self.assertFalse(report["checks"][check], check)


if __name__ == "__main__":
    unittest.main()
