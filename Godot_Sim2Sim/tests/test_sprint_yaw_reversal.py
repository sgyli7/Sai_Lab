"""Sprint reversal regression: retain the transition through its zero crossing."""
import unittest

import numpy as np

from sim2sim.play_input import PlayBrain, TwistLimits


class TestSprintYawReversal(unittest.TestCase):
    def brain(self, duration=.2):
        return PlayBrain(has_sprint=True, lim=TwistLimits(
            sprint_vmax_x=.3, sprint_vmax_ang=.8, sprint_yaw_reversal_s=duration))

    def settled_left(self, brain, sprint=True):
        held={'fwd', 'left'} | ({'sprint'} if sprint else set())
        for _ in range(20):
            brain.tick(held, [], .02)

    def test_reversal_takes_ten_ticks_and_preserves_forward_speed(self):
        brain=self.brain()
        self.settled_left(brain)
        yaw=[]
        for _ in range(10):
            out=brain.tick({'fwd', 'sprint', 'right'}, [], .02)
            yaw.append(float(brain.vel[2]))
            self.assertAlmostEqual(float(brain.vel[0]), .3, places=6)
            self.assertTrue(out.sprint)
        np.testing.assert_allclose(yaw, [.64,.48,.32,.16,0.,-.16,-.32,-.48,-.64,-.8], atol=1e-6)
        self.assertFalse(brain.ramp.yaw_reversing)

    def test_ordinary_sequence_is_identical_when_enabled(self):
        old,new=self.brain(0.),self.brain()
        for held in [{'fwd','left'}, {'fwd','right'}, {'fwd'}, set()]*3:
            for _ in range(20):
                a,b=old.tick(held, [], .02),new.tick(held, [], .02)
                np.testing.assert_array_equal(a.command,b.command)
                self.assertEqual(a.policy,b.policy)

    def test_release_and_reset_cancel_pending_reversal(self):
        for release in [{'fwd','sprint'}, {'fwd','right'}, set()]:
            brain=self.brain();self.settled_left(brain)
            brain.tick({'fwd','sprint','right'}, [], .02)
            self.assertTrue(brain.ramp.yaw_reversing)
            brain.tick(release, [], .02)
            self.assertFalse(brain.ramp.yaw_reversing)
        brain=self.brain();self.settled_left(brain)
        brain.tick({'fwd','sprint','right'}, [], .02)
        brain.reset_motion()
        self.assertFalse(brain.ramp.yaw_reversing)
        np.testing.assert_array_equal(brain.ramp.vel, [0,0,0])

    def test_changing_direction_again_settles_at_latest_key(self):
        brain=self.brain();self.settled_left(brain)
        for _ in range(3):
            brain.tick({'fwd','sprint','right'}, [], .02)
        for _ in range(10):
            brain.tick({'fwd','sprint','left'}, [], .02)
        self.assertAlmostEqual(float(brain.vel[2]), .8, places=6)
        self.assertFalse(brain.ramp.yaw_reversing)

    def test_invalid_duration_rejected(self):
        for value in [-.1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                self.brain(value)


if __name__ == '__main__':
    unittest.main()
