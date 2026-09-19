"""Persistent interactive play must not hide actual scheduled checks."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('science_guard',Path(__file__).resolve().parents[1]/'scripts/showcase_resource_guard.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)

class ClassificationTests(unittest.TestCase):
    def test_affinity_isolation_is_only_for_headless_checks(self):
        jobs=[{'pid':10},{'pid':11},{'pid':12}]
        def affinity(pid):
            if pid==12:raise ProcessLookupError()
            return {0:{14,15},10:{16,17},11:{15,16}}[pid]
        with patch.object(guard.os,'sched_getaffinity',side_effect=affinity):
            self.assertEqual(guard.competing_processes(jobs,True),[jobs[1]])
            self.assertEqual(guard.competing_processes(jobs,False),jobs)

    def test_explicit_sharing_keeps_pressure_and_fps_limits(self):
        jobs=[{'pid':10,'run':None}]
        with patch.object(guard,'training_processes',return_value=jobs), \
             patch.object(guard,'training_sample',return_value={}), \
             patch.object(guard,'pressure_sample',return_value={}), \
             patch.object(guard,'unmeasured_processes',return_value=jobs), \
             patch.object(guard,'pressure_reason',return_value=None):
            with self.assertRaises(guard.PreviewDeferred):guard.preflight()
            self.assertTrue(guard.preflight(shared_eval=True)['shared_eval'])
            with self.assertRaises(guard.PreviewDeferred):guard.preflight(60,shared_eval=True)
            with patch.object(guard,'pressure_reason',return_value='memory pressure'):
                with self.assertRaises(guard.PreviewDeferred):guard.preflight(shared_eval=True)

    def test_only_unattended_flag_free_workshop_is_interactive(self):
        base=['python','-m','sim2sim.workshop']
        self.assertTrue(guard.is_interactive_workshop(base+['--robot','sai','--task','drive']))
        for flags in [['--plan','check.json'],['--plan=check.json'],['--headless'],['--record']]:
            with self.subTest(flags=flags):self.assertFalse(guard.is_interactive_workshop(base+flags))
        for args in [['python','-m','sim2sim.research.budget'],['python','accept_workshop_grab.py'],['python','-m']]:
            self.assertFalse(guard.is_interactive_workshop(args))

if __name__=='__main__':unittest.main()
