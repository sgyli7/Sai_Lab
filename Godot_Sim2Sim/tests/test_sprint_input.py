"""Sprint input must preserve normal skills, opposing keys, and stop semantics."""
import unittest
import numpy as np
from sim2sim.play_input import PlayBrain, TwistLimits, keys_to_held


class SprintInput(unittest.TestCase):
    def settle(self, brain, held, order=None):
        for _ in range(20):out=brain.tick(set(held),[],.02,press_order=order)
        return out

    def test_only_left_shift_maps_to_sprint(self):
        self.assertEqual(keys_to_held({'SHIFT_LEFT','W'}),{'sprint','fwd'})
        self.assertEqual(keys_to_held({'SHIFT_RIGHT','W'}),{'fwd'})

    def test_optional_actor_and_modifier_alone(self):
        ordinary=self.settle(PlayBrain(),['fwd','sprint'])
        self.assertFalse(ordinary.sprint)
        self.assertAlmostEqual(float(ordinary.command[0]),.3)
        brain=PlayBrain(has_sprint=True,has_stand_hold=True)
        brain.tick(set(),['stand'],.02)
        out=self.settle(brain,['sprint'])
        self.assertFalse(out.sprint);self.assertTrue(brain.stand_hold)
        np.testing.assert_array_equal(out.command,np.zeros(13))

    def test_sprint_turn_release_and_opposing_key(self):
        brain=PlayBrain(has_sprint=True,lim=TwistLimits(sprint_vmax_x=.5,sprint_vmax_ang=.8))
        for turn,sign in [('left',1),('right',-1)]:
            out=self.settle(brain,['sprint','fwd',turn],['sprint','fwd',turn])
            self.assertTrue(out.sprint)
            np.testing.assert_allclose(out.command[:3],[.5,0,.8*sign],atol=1e-6)
        out=self.settle(brain,['fwd'],['fwd'])
        self.assertFalse(out.sprint);self.assertAlmostEqual(float(out.command[0]),.3)
        out=self.settle(brain,['sprint','fwd','back'],['fwd','sprint','back'])
        self.assertFalse(out.sprint);self.assertAlmostEqual(float(out.command[0]),-.3)

    def test_modifier_does_not_override_space_or_reset(self):
        brain=PlayBrain(has_sprint=True)
        out=self.settle(brain,['sprint','fwd','idle'],['fwd','idle','sprint'])
        self.assertFalse(out.sprint);np.testing.assert_array_equal(out.command,np.zeros(13))
        self.settle(brain,['sprint','fwd'])
        out=brain.tick({'sprint','fwd'},['reset'],.02)
        self.assertFalse(out.sprint);np.testing.assert_array_equal(out.command,np.zeros(13))
        out=brain.tick({'sprint','fwd'},['pick'],.02)
        self.assertFalse(out.sprint);self.assertEqual(out.policy,'ground_pick')
