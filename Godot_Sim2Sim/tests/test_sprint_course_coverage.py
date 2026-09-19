import unittest
from sim2sim.play_input import PlayBrain,TwistLimits
from sim2sim.research.sprint_courses import keyboard_courses


class SprintCourseCoverage(unittest.TestCase):
    def coverage(self,paired):
        counts={'ordinary_left':0,'ordinary_right':0,'sprint_left':0,'sprint_right':0}
        for name,case in keyboard_courses(paired):
            brain=PlayBrain(has_standing=False,has_sprint=True,lim=TwistLimits(vmax_x=.25,vmax_ang=.8,sprint_vmax_x=.3,sprint_vmax_ang=.8))
            for i in range(round(case['seconds']/.02)):
                held=next(s['held'] for s in reversed(case['segments']) if s['at']<=i*.02+1e-9)
                step=brain.tick(set(held),[],.02,press_order=sorted(held))
                if name.endswith('_ordinary'):self.assertFalse(step.sprint)
                if step.command[0]>.01 and abs(step.command[2])>.05:
                    counts[('sprint' if step.sprint else 'ordinary')+('_left' if step.command[2]>0 else '_right')]+=1
        return counts

    def test_shared_actor_receives_both_ordinary_turn_directions(self):
        legacy=self.coverage(False);paired=self.coverage(True)
        self.assertEqual(legacy['ordinary_right'],0)
        self.assertTrue(all(count>0 for count in paired.values()))
        self.assertEqual(paired['sprint_right'],legacy['sprint_right'])
        self.assertEqual(paired['sprint_left'],legacy['sprint_left'])
        self.assertEqual(keyboard_courses(True)[:8],keyboard_courses(False))
