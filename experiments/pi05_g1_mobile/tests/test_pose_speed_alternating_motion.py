"""Discriminating synthetic speed test; no frozen producer/measurement edits."""
import copy
import unittest
from experiments.pi05_g1_mobile.tests import test_acceptance_pose_speed as fixtures
from experiments.pi05_g1_mobile import acceptance


class AlternatingMotionTest(unittest.TestCase):
    def test_half_mm_alternation_pose_window_passes_but_speed_gate_blocks(self):
        helper=fixtures.PoseSpeedTests();rows,config,bundle=helper.fixture()
        for row in rows:
            row['acceptance_truth_only']['brown_box']['velocity'][:3]=[0,0,0]
        for row in rows[11:]:
            row['acceptance_truth_only']['brown_box']['pose'][0]=5.+(.00025 if row['control_tick']%2 else -.00025)
        before=fixtures.deep_sha(rows)
        self.assertTrue(acceptance.stable_pose_window(rows[50:151],.005,5.))
        judged=helper.evaluate(rows,config,bundle)
        self.assertTrue(all(s['box_speed_m_s']>.0249 for s in judged['samples'][12:]))
        self.assertFalse(any(s['placement_ready'] for s in judged['samples'][11:]))
        self.assertFalse(judged['success']);self.assertIsNone(judged['qualified_stability_window'])
        self.assertEqual(before,fixtures.deep_sha(rows))

if __name__=='__main__':unittest.main()
