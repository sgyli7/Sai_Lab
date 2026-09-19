import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('gait_summary',Path(__file__).resolve().parents[1]/'scripts/summarize_forward_gait.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class LandingSummary(unittest.TestCase):
    def test_simultaneous_two_foot_hops_are_not_alternating_steps(self):
        times=np.arange(100)*.02
        result=module.landing_pattern([(10,0),(10,1),(30,0),(30,1),(50,0),(50,1)],times)
        self.assertEqual(result['alternating_landing_fraction'],0.)
        self.assertEqual(result['simultaneous_landing_events'],3)
        self.assertAlmostEqual(result['median_interval_frequency_hz'],2.5)

    def test_single_foot_events_and_missing_observations(self):
        times=np.arange(100)*.02
        self.assertEqual(module.landing_pattern([(10,0),(30,1),(50,0)],times)['alternating_landing_fraction'],1.)
        self.assertEqual(module.landing_pattern([(10,0),(30,0)],times)['alternating_landing_fraction'],0.)
        self.assertIsNone(module.landing_pattern([],times)['alternating_landing_fraction'])
