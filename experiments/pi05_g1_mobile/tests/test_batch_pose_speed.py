"""Only mock launcher/temporary producer fixtures; never starts Docker or GPU."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from unittest.mock import patch,MagicMock
import unittest
from experiments.pi05_g1_mobile import batch_pose_speed as b
from experiments.pi05_g1_mobile.simulator import validate_config


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class BatchTests(unittest.TestCase):
    def fixture(self,p):
        configs=p/'configs';policies=['gr00t_n16','pi05_mobile','official_pi05_base'];ids=[f'case{i}' for i in range(20)];jobs=[]
        for case in ids:
            for policy in policies:
                cp=configs/(policy+'_'+case+'.json');output=p/'raw'/policy/case
                cfg={'mode':'policy','policy_id':policy,'case':{'case_id':case,'seed':42},'output_dir':str(output),'ticks':5,'native_action_horizon':50,'replan_interval_ticks':50,'action_limits_file':str(p/'bounds.json'),'evaluate_acceptance':True}
                write(cp,cfg);jobs.append({'policy':policy,'case_id':case,'config_path':str(cp),'config_sha256':b.sha(cp),'result_path':str(output/'result.json')})
        marker=p/'frozen.txt';marker.write_text('synthetic immutable');plan=p/'plan.json';write(plan,{'schema':'pi05_g1_mobile_benchmark_plan_v1','policies':policies,'case_ids':ids,'jobs':jobs,'frozen_files':{str(marker):b.sha(marker)}})
        token='CPU_ONLY_TEST_TOKEN';owner=p/'owned';owner.mkdir();ledger=p/'ledger.json';write(ledger,{'runs':[{'budget_token':token,'ownership_receipt_dir':str(owner),'stage':'evaluation','status':'running','started_unix':time.time(),'child_work_limit_seconds':3600,'owned_unit':'CPU_FAKE_NEVER_SYSTEMD'}]})
        environment=patch.dict(os.environ,{'PI05_G1_BUDGET_TOKEN':token,'PI05_G1_BUDGET_OWNERSHIP_DIR':str(owner)});environment.start();self.addCleanup(environment.stop)
        return plan,ledger,jobs

    def producer(self,command,log,timeout,cancel):
        self.assertEqual(command[command.index('-m')+1],'experiments.pi05_g1_mobile.sim_launcher_pose_speed')
        config=b.read(command[command.index('--config')+1]);out=Path(config['output_dir']);out.mkdir(parents=True)
        write(out/'config.json',validate_config(config));trace=out/'trajectory.jsonl';trace.write_text('SYNTHETIC_NOT_PHYSICS\n');(out/'evidence.bin').write_bytes(b'cpu fixture artifact')
        write(out/'result.json',{'schema':'pi05_g1_mobile_case_result_v1','status':'completed','source_interface_passed':True,'policy':config['policy_id'],'case_id':config['case']['case_id'],'success':False,'acceptance':{'success':False},'trace_sha256':b.sha(trace)})
        log.write('SYNTHETIC_CPU_PRODUCER\n');log.flush();return 0

    def run_segment(self,p,plan,ledger,index,resume=(),max_jobs=5):
        return b.run(plan,p/f'segment{index}.json',p/'batch_result.json',ledger=ledger,max_jobs=max_jobs,resume_receipts=resume,min_job_seconds=1)

    def test_task_failures_complete_segment_but_not_global60_and_resume_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,ledger,jobs=self.fixture(p)
            with patch.object(b,'launch',side_effect=self.producer) as launch:
                a=self.run_segment(p,plan,ledger,0);prior=b.sha(p/'segment0.json')
                c=self.run_segment(p,plan,ledger,1,[p/'segment0.json'])
            self.assertEqual(a['status'],'completed');self.assertFalse(a['benchmark_complete']);self.assertEqual(a['overall_status'],'partial')
            self.assertEqual(c['completed_total_jobs'],10);self.assertEqual([r['index'] for r in c['jobs']],list(range(5,10)))
            self.assertTrue(all(r['success'] is False for r in c['jobs']));self.assertEqual(launch.call_count,10)
            self.assertFalse((p/'batch_result.json').exists());self.assertEqual(b.sha(p/'segment0.json'),prior)

    def test_three_job_segment_keeps_one_case_all_three_and_typed_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,ledger,jobs=self.fixture(p)
            with patch.object(b,'launch',side_effect=self.producer) as launch:
                first=self.run_segment(p,plan,ledger,0,max_jobs=3)
                second=self.run_segment(p,plan,ledger,1,[p/'segment0.json'],max_jobs=3)
            self.assertEqual(first['schema'],b.SEGMENT_SCHEMA);self.assertEqual(first['status'],'completed')
            self.assertEqual(first['expected_segment_jobs'],3);self.assertFalse(first['benchmark_complete'])
            self.assertEqual({r['case_id'] for r in first['jobs']},{'case0'})
            self.assertEqual({r['policy'] for r in first['jobs']},{'gr00t_n16','pi05_mobile','official_pi05_base'})
            self.assertEqual([r['index'] for r in second['jobs']],[3,4,5]);self.assertEqual(second['completed_total_jobs'],6)
            self.assertEqual(launch.call_count,6);self.assertFalse((p/'batch_result.json').exists())

    def test_only_all_sixty_create_typed_global_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,ledger,jobs=self.fixture(p);receipts=[]
            with patch.object(b,'launch',side_effect=self.producer):
                for i in range(6):
                    report=self.run_segment(p,plan,ledger,i,receipts,max_jobs=10);receipts.append(p/f'segment{i}.json')
                    self.assertEqual((p/'batch_result.json').exists(),i==5)
            final=b.read(p/'batch_result.json');self.assertEqual(final['schema'],b.BATCH_SCHEMA);self.assertEqual(final['status'],'completed');self.assertEqual(len(final['jobs']),60)
            self.assertTrue(final['independent_physics_audit_and_benchmark_pending']);self.assertTrue(report['benchmark_complete'])

    def test_infrastructure_failure_stops_preserves_prefix_and_never_zero_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,ledger,jobs=self.fixture(p);calls=[]
            def failing(command,log,timeout,cancel):
                calls.append(command)
                return self.producer(command,log,timeout,cancel) if len(calls)==1 else 3
            with patch.object(b,'launch',side_effect=failing):
                with self.assertRaisesRegex(RuntimeError,'launcher failed'):self.run_segment(p,plan,ledger,0)
            receipt=b.read(p/'segment0.json');self.assertEqual(receipt['status'],'infrastructure_failed');self.assertEqual(receipt['completed_total_jobs'],1);self.assertEqual(len(calls),2)
            self.assertEqual(receipt['jobs'][0]['status'],'completed');self.assertFalse((p/'batch_result.json').exists());self.assertNotIn('success_rate',receipt)

    def test_resume_artifact_tamper_or_unreceipted_outputs_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,ledger,jobs=self.fixture(p)
            with patch.object(b,'launch',side_effect=self.producer):self.run_segment(p,plan,ledger,0)
            evidence=Path(jobs[0]['result_path']).parent/'evidence.bin';evidence.write_bytes(b'tampered')
            with patch.object(b,'launch') as launch:
                with self.assertRaisesRegex(ValueError,'changed'):self.run_segment(p,plan,ledger,1,[p/'segment0.json'])
                launch.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,ledger,jobs=self.fixture(p);Path(jobs[0]['result_path']).parent.mkdir(parents=True)
            with patch.object(b,'launch') as launch:
                with self.assertRaises(FileExistsError):self.run_segment(p,plan,ledger,0)
                launch.assert_not_called()

    def test_owned_deadline_partial_launches_nothing_and_outside_budget_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,ledger,jobs=self.fixture(p);doc=b.read(ledger);doc['runs'][0]['child_work_limit_seconds']=10;write(ledger,doc)
            with patch.object(b,'launch') as launch:report=self.run_segment(p,plan,ledger,0);launch.assert_not_called()
            self.assertEqual(report['status'],'partial');self.assertEqual(report['completed_total_jobs'],0)
            with patch.dict(os.environ,{'PI05_G1_BUDGET_TOKEN':''}):
                with self.assertRaisesRegex(ValueError,'outer mobile'):self.run_segment(p,plan,ledger,1)

    def test_launcher_deadline_terminates_only_owned_child_with_bounded_cleanup(self):
        child=MagicMock();child.wait.side_effect=[subprocess.TimeoutExpired('owned',1),0];cancel=b.Cancellation()
        with patch.object(b.subprocess,'Popen',return_value=child):
            with self.assertRaises(TimeoutError):b.launch(['OWNED_CPU_MOCK'],MagicMock(),1,cancel)
        child.terminate.assert_called_once();self.assertEqual(child.wait.call_args.kwargs['timeout'],20);self.assertIsNone(cancel.child)

if __name__=='__main__':unittest.main()
