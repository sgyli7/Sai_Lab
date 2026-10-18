"""Synthetic temp fixtures only. No real formal outcome or GPU read/run."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from experiments.pi05_g1_mobile import benchmark as b
from experiments.pi05_g1_mobile.simulator import validate_config


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False))


class MobileBenchmarkTests(unittest.TestCase):
    def fixture(self,p):
        policies=['gr00t_n16','pi05_mobile','official_pi05_base'];cases=[{'case_id':f'p{i}_s{seed}','cluster_id':i,'seed':seed} for i in range(5) for seed in range(4)]
        norm=p/'norm.json';write(norm,{'synthetic':True});limits=p/'limits.json';write(limits,{'synthetic':True});jobs=[]
        for policy in policies:
            for case in cases:
                out=p/policy/case['case_id'];success=policy!='official_pi05_base';config={'mode':'policy','policy_id':policy,'case':case,'output_dir':str(out),'evaluate_acceptance':True,'episode_seconds':30.,'ticks':2,'native_action_horizon':50,'replan_interval_ticks':50,'action_limits_file':str(limits),'acceptance':{'stable_seconds':.02,'task_family':'SYNTHETIC_ONLY_NOT_PHYSICS_PROOF'},'norm_stats':str(norm),'policy_command':['python','-m','SYNTHETIC_SERVICE','--checkpoint',str(p/policy/'model.safetensors'),'--norm-stats',str(norm),'--decoded-evidence',str(out/'decoded.jsonl')]}
                config_path=p/'configs'/str(policy+'_'+case['case_id']+'.json');write(config_path,config);write(out/'config.json',validate_config(config))
                rows=[]
                for tick in range(3):
                    rows.append({'control_tick':tick,'physics_steps_since_reset':tick*4,'monotonic_ns':1000000000+tick*(100000000 if success else 1000000),'source_joint_names':[f'q{i}' for i in range(43)],'joint_positions':[0.]*43,'joint_velocities':[0.]*43,'root_link_pose_w':[0.,float(case['cluster_id']),0.,0.,0.,0.,1.],'root_link_velocity_w':[0.]*6,'root_com_velocity_w':[0.]*6,'acceptance_truth_only':{'brown_box':{'pose':[0.,0.,0.,0.,0.,0.,1.],'velocity':[0.]*6}}})
                trace=out/'trajectory.jsonl';trace.write_text(''.join(json.dumps(r)+'\n' for r in rows));initial={k:rows[0][k] for k in b.INITIAL_KEYS}
                acceptance={'schema':'pi05_g1_mobile_acceptance_v1','success':success,'grasped':success,'stable_placement':success,'fell':False,'dropped':False,'timeout':not success,'first_grasp_control_tick':0 if success else None,'qualified_stability_window':{'start_tick':1,'end_tick':1,'seconds':.02} if success else None,'placement_rule':config['acceptance'],'policy_external_interventions':0,'actual_control_ticks':2,'physics_integrations':8,'task_sim_seconds':.02 if success else .04,'task_wall_seconds':.1 if success else .002,'termination_reasons':[] if success else ['no_verified_grasp','no_released_stable_placement'],**{key:0. for key in b.METRICS}}
                result={'schema':'pi05_g1_mobile_case_result_v1','status':'completed','policy':policy,'case_id':case['case_id'],'case':case,'control_hz':50,'physics_hz':200,'source_interface_passed':True,'trace_sha256':b.sha(trace),'initial_physical_state_schema':'pi05_g1_mobile_initial_kinematic_v1','initial_physical_state_sha256':hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest(),'frozen_interface_sha256':'SYNTHETIC_SAME_INTERFACE','completed_control_ticks':2,'success':success,'acceptance':acceptance,'inference_calls':[{'observation_control_tick':tick,'inference_seconds':.01,'simulation_paused_wall_seconds':.02 if success else .00002,'model_load_seconds':.001 if tick==0 else None} for tick in [0,1]],'cold_process_started_monotonic_ns':900000000,'evidence_finalized_monotonic_ns':1300000000,'commanded_channels':64,'clipped_channels':0}
                event='native_decoded' if policy=='gr00t_n16' else 'pi05_decoded'
                (out/'decoded.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in [{'event':'model_loaded','load_seconds':.001},*[{'event':event,'episode_id':case['case_id'],'control_tick':tick,'actions32':[[0.]*32]*50} for tick in [0,1]]]))
                result_path=out/'result.json';write(result_path,result);jobs.append({'policy':policy,'case_id':case['case_id'],'config_path':str(config_path),'config_sha256':b.sha(config_path),'result_path':str(result_path)})
        plan=p/'plan.json';write(plan,{'schema':'pi05_g1_mobile_benchmark_plan_v1','primary_policies':policies,'case_ids':[c['case_id'] for c in cases],'cases':cases,'jobs':jobs,'frozen_files':{str(Path(b.__file__).with_name('acceptance.py').resolve()):b.sha(Path(b.__file__).with_name('acceptance.py')),str(norm):b.sha(norm),str(limits):b.sha(limits)}})
        return plan,jobs

    def change_result(self,job,mutate):
        p=Path(job['result_path']);r=b.read(p);mutate(r);write(p,r)

    def test_complete_pair_stats_failure_rates_and_speed_excludes_fast_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,jobs=self.fixture(p);r=b.aggregate(plan,p/'summary.json')
            self.assertTrue(r['complete']);self.assertTrue(r['formal_verdict_available'])
            self.assertEqual(r['policies']['pi05_mobile']['formal_success_count'],20)
            self.assertEqual(r['policies']['official_pi05_base']['formal_success_count'],0)
            self.assertEqual(r['policies']['official_pi05_base']['task_failure_rate'],1.)
            self.assertLess(r['policies']['official_pi05_base']['episode_actual_wall_seconds_all_outcomes']['median'],r['policies']['pi05_mobile']['success_only_actual_task_wall_seconds']['median'])
            pair=r['paired_comparisons']['pi05_mobile__official_pi05_base']
            self.assertFalse(pair['speed_available']);self.assertEqual(pair['matched_success_count'],0)
            self.assertIsNone(pair['matched_success_actual_task_wall_difference_seconds_second_minus_first']['median'])
            self.assertEqual(pair['physical_position_cluster']['exact_cluster_sign_flip_two_sided_p'],.0625)
            self.assertEqual(r['policies']['pi05_mobile']['warm_inference_seconds']['count'],20)
            self.assertEqual(len((p/'summary.csv').read_text().splitlines()),61)
            self.assertIn('not20 independent',r['scope']);self.assertIn('not a long-distance5m',r['scope'])

    def test_missing_result_is_incomplete_not_zero_twenty(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,jobs=self.fixture(p);Path(jobs[0]['result_path']).unlink()
            r=b.aggregate(plan,p/'summary.json');self.assertFalse(r['complete']);self.assertIsNone(r['formal_verdict'])
            self.assertIsNone(r['policies']['gr00t_n16']['formal_success_count']);self.assertEqual(r['policies']['gr00t_n16']['completed_evidence_cases'],19)
            self.assertEqual(len(r['missing_or_invalid_cases']),1);self.assertEqual(r['paired_comparisons'],{})

    def test_actual_initial_mismatch_across_policy_rejects_despite_same_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,jobs=self.fixture(p);job=jobs[20];trace=Path(job['result_path']).parent/'trajectory.jsonl';rows=[json.loads(s) for s in trace.read_text().splitlines()];rows[0]['joint_velocities'][0]=.01;trace.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            def mutate(r):
                r['trace_sha256']=b.sha(trace);r['initial_physical_state_sha256']=hashlib.sha256(json.dumps({k:rows[0][k] for k in b.INITIAL_KEYS},sort_keys=True).encode()).hexdigest()
            self.change_result(job,mutate);r=b.aggregate(plan,p/'summary.json')
            self.assertFalse(r['complete']);self.assertTrue(any('Actual initial state differs' in e for e in r['errors']))

    def test_trace_tamper_config_change_and_wall_nan_are_incomplete(self):
        for kind in ['trace','config','wall']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);plan,jobs=self.fixture(p);job=jobs[0]
                if kind=='trace':
                    trace=Path(job['result_path']).parent/'trajectory.jsonl';trace.write_text(trace.read_text()+'\n')
                elif kind=='config':
                    path=Path(job['config_path']);c=b.read(path);c['ticks']=1;write(path,c)
                else:self.change_result(job,lambda r:r['acceptance'].update(task_wall_seconds=.001))
                r=b.aggregate(plan,p/'summary.json');self.assertFalse(r['complete']);self.assertTrue(r['missing_or_invalid_cases'])

    def test_nonfinite_trace_and_missing_cold_load_cannot_qualify(self):
        for kind in ['nonfinite','coldload']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);plan,jobs=self.fixture(p);job=jobs[0]
                if kind=='nonfinite':
                    trace=Path(job['result_path']).parent/'trajectory.jsonl';rows=[json.loads(line) for line in trace.read_text().splitlines()];rows[1]['joint_positions'][0]=float('nan');trace.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                    self.change_result(job,lambda r:r.update(trace_sha256=b.sha(trace)))
                else:self.change_result(job,lambda r:r['inference_calls'][0].update(model_load_seconds=None))
                report=b.aggregate(plan,p/'summary.json');self.assertFalse(report['complete']);self.assertTrue(report['missing_or_invalid_cases'])

    def test_real_pi_service_evidence_output_and_case_owned_guard(self):
        import base64,io
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from experiments.pi05_g1_mobile import contract,pi05_service,model_runtime
        q=[0.]*43;names=list(contract.DATASET_NAMES);rgb=np.zeros((480,640,3),np.uint8)
        req={'schema':'pi05_g1_observation_v1','episode_id':'producer-case','control_tick':0,'rgb_shape':[480,640,3],'rgb_uint8_b64':base64.b64encode(rgb.tobytes()).decode(),'state':contract.encode_state(q,names).tolist(),'prompt':contract.PROMPT,'seed':42,'measured_joint_positions43':q,'measured_joint_names43':names}
        fake=SimpleNamespace(metadata={'synthetic_cpu':True},infer=lambda *args:{'actions':np.zeros((50,32),np.float32),'inference_seconds':.01})
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'case';out.mkdir();evidence=out/'decoded.jsonl';argv=['service','--checkpoint','FAKE','--norm-stats','FAKE','--decoded-evidence',str(evidence)]
            with patch.object(pi05_service.sys,'argv',argv),patch.object(pi05_service.sys,'stdin',io.StringIO(json.dumps(req)+'\n')),patch.object(pi05_service.sys,'stdout',io.StringIO()),patch.object(model_runtime,'ModelRuntime',return_value=fake):pi05_service.main()
            config={'policy_command':argv,'policy_id':'pi05_mobile','output_dir':str(out),'case':{'case_id':'producer-case'}}
            path,records=b.decoded_evidence(config,{'inference_calls':[{'observation_control_tick':0}]})
            self.assertEqual(path,evidence);self.assertEqual([r['event'] for r in records],['model_loaded','pi05_decoded'])
            config['output_dir']=str(out/'OTHER_CASE')
            with self.assertRaisesRegex(ValueError,'inside its case'):b.decoded_evidence(config,{'inference_calls':[{'observation_control_tick':0}]})

    def test_exact_cluster_all_positive_and_zero_difference_and_known_mcnemar(self):
        clusters=[i//4 for i in range(20)]
        self.assertEqual(b.cluster_statistics([0]*20,[1]*20,clusters)['exact_cluster_sign_flip_two_sided_p'],.0625)
        self.assertEqual(b.cluster_statistics([1]*20,[1]*20,clusters)['exact_cluster_sign_flip_two_sided_p'],1.)
        self.assertEqual(b.paired_statistics([0,0],[1,1])['exact_mcnemar_p'],.5)
        self.assertAlmostEqual(b.wilson(0,20)[1],.16112515805281938)

    def test_different_pi_runtime_and_incomplete_model_timing_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,jobs=self.fixture(p);job=jobs[20];cp=Path(job['config_path']);c=b.read(cp);c['policy_command'][0]='DIFFERENT_RUNTIME';write(cp,c);write(Path(job['result_path']).parent/'config.json',validate_config(c));doc=b.read(plan);doc['jobs'][20]['config_sha256']=b.sha(cp);write(plan,doc)
            r=b.aggregate(plan,p/'summary.json');self.assertFalse(r['complete']);self.assertTrue(any('norm/runtime' in e for e in r['errors']))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);plan,jobs=self.fixture(p)
            self.change_result(jobs[0],lambda r:r['inference_calls'][0].pop('simulation_paused_wall_seconds'))
            r=b.aggregate(plan,p/'summary.json');self.assertFalse(r['complete'])


if __name__=='__main__':unittest.main()
