"""CPU-only explicit common limits intersect actual named hardware limits."""
import unittest
import numpy as np
from experiments.pi05_g1.contract import DATASET_NAMES,UPPER_NAMES
from experiments.pi05_g1_round02.simulator import intersect_action_bounds
class BoundsTests(unittest.TestCase):
 def test_named_hardware_intersection(self):
  names=list(reversed(DATASET_NAMES));hw=np.tile([-2.,2.],(43,1));hw[names.index(UPPER_NAMES[0])]=[-.4,.5]
  lo,hi=intersect_action_bounds(np.full(32,-10.),np.full(32,10.),hw,names)
  self.assertAlmostEqual(float(lo[0]),-.4,places=6);self.assertAlmostEqual(float(hi[0]),.5)
  np.testing.assert_allclose(lo[1:28],-2);np.testing.assert_allclose(hi[1:28],2)
  np.testing.assert_allclose(lo[28:],-10);np.testing.assert_allclose(hi[28:],10)
 def test_empty_rejected(self):
  with self.assertRaises(ValueError):intersect_action_bounds(np.full(32,3.),np.full(32,4.),np.tile([-2.,2.],(43,1)),DATASET_NAMES)
 def test_nonfinite_rejected(self):
  with self.assertRaises(ValueError):intersect_action_bounds(np.full(32,float('nan')),np.full(32,4.),np.tile([-2.,2.],(43,1)),DATASET_NAMES)
if __name__=='__main__':unittest.main()
