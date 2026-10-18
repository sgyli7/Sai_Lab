"""Synthetic CPU timeline checks; no simulator/model imports or execution."""
import unittest
from experiments.pi05_g1_round02.video_compare import caption_state
class CaptionTests(unittest.TestCase):
 def setUp(self):
  self.trace=[{'control_tick':i,'monotonic_ns':(12+i)*10**9} for i in range(3)]
  self.result={'cold_process_started_monotonic_ns':10*10**9,'success':True,'qualified_stability_window':{'end_tick':1}}
 def test_final_outcome_at_start_does_not_claim_qualification(self):
  s=caption_state(self.result,self.trace,0,True)
  self.assertEqual(s['outcome_label'],'Final outcome: SUCCESS');self.assertEqual(s['qualification_label'],'No qualification yet');self.assertFalse(s['ended'])
 def test_wall_ends_and_clamps(self):
  s=caption_state(self.result,self.trace,99,True)
  self.assertEqual(s['wall_seconds'],4);self.assertEqual(s['tick'],2);self.assertTrue(s['ended']);self.assertIn('LAST FRAME HELD',s['end_label'])
 def test_qualification_event_uses_actual_observed_wall(self):
  s=caption_state(self.result,self.trace,3,True)
  self.assertIn('QUALIFIED at sim0.02s / actual obs-wall1.00s',s['qualification_label']);self.assertFalse(s['ended'])
 def test_simulation_ends_and_holds(self):
  s=caption_state(self.result,self.trace,.04,False)
  self.assertTrue(s['ended']);self.assertEqual(s['simulation_seconds'],.04);self.assertEqual(s['wall_seconds'],4)
 def test_failed_recording_not_marked_working_after_end(self):
  r={**self.result,'success':False,'qualified_stability_window':None};s=caption_state(r,self.trace,20,True)
  self.assertEqual(s['outcome_label'],'Final outcome: FAILURE');self.assertEqual(s['qualification_label'],'No qualified window');self.assertTrue(s['ended'])
if __name__=='__main__':unittest.main()
