import math
import unittest

import numpy as np

from sim2sim.standalone.score import brake_metrics


def trajectory(velocity,seconds=6.,tilt=0.,yaw=0.):
    rows=[];xy=np.zeros(2)
    for index in range(round(seconds/.02)+1):
        t=index*.02;vx=float(velocity(t));xy+=np.array([vx*math.cos(yaw),vx*math.sin(yaw)])*.02
        rows.append(dict(time=t,vel=[vx,0.,0.],xy=xy.copy(),yaw=yaw,tilt=tilt,z=.13,contact=[1.,1.]))
    return rows


class BrakeAcceptanceTests(unittest.TestCase):
    def test_smooth_standing_stop_passes(self):
        result=brake_metrics(trajectory(lambda t:max(0.,.6-t)),0.,6.)
        self.assertTrue(result['success'])
        self.assertLessEqual(result['stop_confirmed_s'],2.)

    def test_fallen_body_does_not_count_as_stopping(self):
        result=brake_metrics(trajectory(lambda t:max(0.,.6-3*t),tilt=80),0.,6.)
        self.assertFalse(result['success']);self.assertTrue(result['fell'])

    def test_short_low_speed_interval_is_not_one_second_hold(self):
        result=brake_metrics(trajectory(lambda t:0. if .3<t<1.1 else .6),0.,6.)
        self.assertFalse(result['success']);self.assertIsNone(result['stop_confirmed_s'])

    def test_late_confirmation_fails_even_if_eventually_stopped(self):
        result=brake_metrics(trajectory(lambda t:max(0.,.6-.4*t)),0.,6.)
        self.assertFalse(result['success']);self.assertGreater(result['stop_confirmed_s'],2.)

    def test_continued_reverse_after_stop_fails(self):
        result=brake_metrics(trajectory(lambda t:max(0.,.6-t) if t<3. else -.2),0.,6.)
        self.assertFalse(result['success']);self.assertTrue(result['sustained_backwards'])

    def test_excess_distance_fails_despite_timed_stop(self):
        rows=trajectory(lambda t:0. if t>.3 else 2.)
        # A low initial velocity must not grant a long distance allowance later.
        rows[0]['vel']=[.1,0.,0.]
        result=brake_metrics(rows,0.,6.)
        self.assertLess(result['stop_confirmed_s'],2.)
        self.assertFalse(result['success']);self.assertGreater(result['braking_distance'],result['distance_limit'])

    def test_rotated_world_heading_preserves_speed_and_distance(self):
        first=brake_metrics(trajectory(lambda t:max(0.,.6-t)),0.,6.)
        rotated=brake_metrics(trajectory(lambda t:max(0.,.6-t),yaw=1.7),0.,6.)
        self.assertEqual(first['success'],rotated['success'])
        self.assertAlmostEqual(first['braking_distance'],rotated['braking_distance'])
        self.assertAlmostEqual(first['minimum_forward_02s'],rotated['minimum_forward_02s'])


if __name__=='__main__':unittest.main()
