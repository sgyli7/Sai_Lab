import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.pi05_g1_round02.offline_diagnostic import make_plan,aggregate,run_job,sha,write_new


class OfflineDiagnosticTests(unittest.TestCase):
    def fixture(self,p):
        manifest=p/'manifest.json';ids=list(range(20,41));train=[0,1]
        write_new(manifest,{'dataset_root':str(p/'NEVER_READ_TEST_OBSERVATIONS'),'train':train,'test':ids,
                    'files':[{'episode_id':eid,'length':51 if eid==40 else 50} for eid in ids]})
        norm=p/'norm_stats_v2.json';write_new(norm,{'actions':{}})
        bounds=p/'bounds.json';low=[0.]*32;high=[0.]*32
        high[14]=.001;high[27]=.5
        write_new(bounds,{'source_manifest_sha256':sha(manifest),'train_episodes':train,'source_split':'train_only_synthetic',
                         'lower':low,'upper':high,'action_names':[f'a{i}' for i in range(32)]})
        artifacts={}
        for policy in ('pi05_round01','pi05_round02','official_pi05_base'):
            folder=p/policy;folder.mkdir();weights=folder/'model.safetensors';weights.write_bytes(policy.encode())
            artifacts[policy]={'path':str(weights),'sha256':sha(weights),'bytes':weights.stat().st_size}
        selection=p/'selection.json';write_new(selection,{'schema':'pi05_g1_round02_final_selection_v1',
            'selected':{'path':str(p/'pi05_round02'),'validation_flow_loss':1.},'all_complete_validation_candidates':[{'validation_flow_loss':1.}],
            'data_identity':{'norm_sha256':sha(norm),'sources':[{'kind':'lerobot','manifest':str(manifest),'manifest_sha256':sha(manifest)}]},
            'frozen_files':{str(norm):sha(norm)},'model_artifacts':artifacts})
        plan=make_plan(selection,bounds,p/'offline');return plan,p/'offline/plan.json'

    def fake_result(self,job,plan,status='complete'):
        write_new(job['output'],{'status':status,'split':'test','checkpoint':job['checkpoint'],
             'valid_frames':plan['expected_valid_frames_per_policy'],'chunks_evaluated':plan['expected_chunks_per_policy'],
             'chunks_planned':plan['expected_chunks_per_policy'],'mae_per_dimension':[1.]*32,'rmse_per_dimension':[2.]*32,
             'inference_seconds_mean':.2,'inference_seconds_p95':.3,'physical_success_evaluated':False})

    def test_plan_uses_metadata_counts_and_train_only_hand_ranges_without_test_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,path=self.fixture(p)
            self.assertEqual(plan['expected_chunks_per_policy'],22);self.assertEqual(plan['expected_valid_frames_per_policy'],1051)
            self.assertEqual(plan['groups']['activehands_train_nonconstant'],[14,27])
            self.assertFalse((p/'NEVER_READ_TEST_OBSERVATIONS').exists())
            for job in plan['jobs']:
                self.assertEqual(job['argv'][1:3],['-m','experiments.pi05_g1.evaluate_offline'])
                self.assertEqual(job['argv'][job['argv'].index('--split')+1],'test')
            missing=aggregate(path,p/'missing_summary.json');self.assertFalse(missing['complete'])
            self.assertTrue(all(r['status']=='incomplete' for r in missing['policies'].values()))

    def test_bound_wrapper_and_complete_group_summary_without_running_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,path=self.fixture(p)
            for job in plan['jobs']:
                def fake_run(argv,**kwargs):
                    self.assertEqual(argv,job['argv']);self.fake_result(job,plan);return SimpleNamespace(returncode=0)
                with patch('experiments.pi05_g1_round02.offline_diagnostic.subprocess.run',side_effect=fake_run):
                    receipt=run_job(path,job['policy']);self.assertEqual(receipt['status'],'complete')
            report=aggregate(path,p/'summary.json');self.assertTrue(report['complete']);self.assertFalse(report['physical_success_evaluated'])
            for row in report['policies'].values():
                self.assertEqual(row['groups']['activehands_train_nonconstant']['dimensions'],[14,27])
                self.assertEqual(row['groups']['arms14']['mae'],1.);self.assertEqual(row['groups']['arms14']['rmse'],2.)
                for name,unit in [('arms14','rad'),('activehands_train_nonconstant','rad'),('allhands14','rad'),('height','m'),('vx','m/s'),('vy','m/s'),('yaw','rad/s')]:
                    self.assertEqual(row['groups'][name]['unit'],unit)
                    self.assertEqual(row['groups'][name]['mae'],1.);self.assertEqual(row['groups'][name]['rmse'],2.)
                self.assertIsNone(row['groups']['nav3']['unit']);self.assertFalse(row['groups']['nav3']['physical_unit_defined'])
                self.assertIn('mixed-units numeric-only',row['groups']['nav3']['interpretation'])
                self.assertIn('no cross-group ranking',row['groups']['nav3']['interpretation'])
            job=plan['jobs'][0];result=Path(job['output']);raw=json.loads(result.read_text());raw['split']='validation';result.write_text(json.dumps(raw))
            self.assertFalse(aggregate(path,p/'tampered_summary.json')['complete'])
            with self.assertRaises(FileExistsError):run_job(path,job['policy'])

    def test_partial_result_is_incomplete_and_missing_binding_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,path=self.fixture(p);job=plan['jobs'][0]
            def fake_run(*args,**kwargs):self.fake_result(job,plan,status='partial_time_budget');return SimpleNamespace(returncode=0)
            with patch('experiments.pi05_g1_round02.offline_diagnostic.subprocess.run',side_effect=fake_run):run_job(path,job['policy'])
            report=aggregate(path,p/'partial.json');self.assertFalse(report['complete']);self.assertIn('partial',report['errors'][0])
            raw=Path(job['receipt']);receipt=json.loads(raw.read_text());receipt.pop('before_sha256');raw.write_text(json.dumps(receipt))
            self.assertIn('identity',aggregate(path,p/'unbound.json')['errors'][0])


if __name__=='__main__':unittest.main()
