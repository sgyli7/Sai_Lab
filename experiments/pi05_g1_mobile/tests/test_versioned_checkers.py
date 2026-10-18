"""Explicit checker injection preserves native defaults; pure CPU fixtures."""
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
from experiments.pi05_g1_mobile import benchmark as old_benchmark,benchmark_v2,audit_episodes_v2,audit_episodes_v3,acceptance
from experiments.pi05_g1_mobile.tests import test_benchmark as bf,test_audit_episodes_v2 as af


class VersionedCheckersTests(unittest.TestCase):
    def test_benchmark_default_identical_and_explicit_checker_called(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);planpath,jobs=bf.MobileBenchmarkTests().fixture(p);plan=old_benchmark.read(planpath);job=jobs[0]
            expected=old_benchmark.load_case(job,plan,planpath)
            self.assertEqual(benchmark_v2.load_case(job,plan,planpath),expected)
            with patch.object(benchmark_v2,'decoded_evidence',side_effect=AssertionError('default must not run')):
                calls=[]
                def injected(config,result):calls.append(result['case_id']);return old_benchmark.decoded_evidence(config,result)
                self.assertEqual(benchmark_v2.load_case(job,plan,planpath,decoded_checker=injected),expected)
            self.assertEqual(calls,[job['case_id']])

    def test_audit_defaults_identical_and_checker_injection_is_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixture,rows,result=af.V2EpisodeTests().fixture(p)
            with patch.object(acceptance,'geometry',return_value=fixture.geometry):
                old=audit_episodes_v2.audit_case(p);new=audit_episodes_v3.audit_case(p)
                self.assertTrue(old['passed']);self.assertTrue(new['passed']);self.assertEqual(old['checks'],new['checks'])
                checker=audit_episodes_v2.decoded_evidence
                with patch.object(audit_episodes_v3,'decoded_evidence',side_effect=AssertionError('default must not run')):
                    calls=[]
                    def injected(config,result):calls.append(result['case_id']);return checker(config,result)
                    actual=audit_episodes_v3.audit_case(p,decoded_checker=injected)
                self.assertTrue(actual['passed'],actual['anomalies']);self.assertEqual(actual['checks'],old['checks']);self.assertEqual(len(calls),1)
                def failed(config,result):raise ValueError('explicit checker rejected evidence')
                self.assertFalse(audit_episodes_v3.audit_case(p,decoded_checker=failed)['passed'])

if __name__=='__main__':unittest.main()
