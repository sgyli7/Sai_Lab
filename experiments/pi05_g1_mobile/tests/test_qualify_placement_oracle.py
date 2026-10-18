import hashlib,json,shutil,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.pi05_g1_mobile import qualify_placement_oracle as subject
from experiments.pi05_g1_mobile.tests import test_acceptance_pose_speed as fixtures


class QualifierTests(unittest.TestCase):
    def fixture(self,root):
        helper=fixtures.PoseSpeedTests();rows,config,bundle=helper.fixture()
        for row in rows:
            row['acceptance_truth_only']['brown_box']['pose']=[5,0,.02,0,0,0,1]
            c=row['acceptance_contacts_only'];m=np.zeros((54,3));m[0,2]=1
            c.update(box_filter_force_matrix_world_n=[m.tolist()],box_robot_force_matrix_world_n=m[1:].tolist(),box_hand_force_matrix_world_n=m[1:17].tolist(),box_bin_force_world_n=m[0].tolist(),points=[{'filter_index':0,'normal_world':[0,0,1],'normal_force_n':1.,'separation_m':0.}])
        case=root/'case';case.mkdir();trace=case/'trajectory.jsonl';trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        config.update(mode='replay',ticks=150,evaluate_acceptance=True,case={'case_id':'synthetic_oracle'},policy_id='nonformal_oracle',
            native_policy_trial=False,full_task_trial=False,teacher_trial=False,formal_benchmark_trial=False,
            frozen_files={str(Path(subject.measurement.__file__).resolve()):subject.sha(subject.measurement.__file__),str(Path(subject.measurement.body_ids.__file__).resolve()):subject.sha(subject.measurement.body_ids.__file__)})
        (case/'config.json').write_text(json.dumps(config));(case/'runtime_identity.json').write_text('{}')
        judged=helper.evaluate(rows,config,bundle)
        result={'schema':'pi05_g1_mobile_case_result_v1','status':'completed','success':False,'source_interface_passed':True,
            'case_id':'synthetic_oracle','policy':'nonformal_oracle','completed_control_ticks':150,'actual_sdk_integrations':600,'control_hz':50,'physics_hz':200,
            'body_names':config['audit_body_names'],'contact_view':config['audit_contact_metadata'],'trace_sha256':subject.sha(trace),'acceptance':judged}
        head=case/'head.mp4';observer=case/'observer.mp4'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=blue:s=64x64:r=50','-frames:v','151','-c:v','libx264','-threads','2',str(head)],check=True)
        shutil.copyfile(head,observer)
        result.update(head_video_path=str(head),observer_video_path=str(observer))
        (case/'result.json').write_text(json.dumps(result))
        bound={str(p.resolve()):subject.sha(p) for p in case.iterdir() if p.is_file()}
        streams={key:{'path':str(p),'sha256':subject.sha(p),**subject.independent.video_probe(p,151)} for key,p in [('head_video_path',head),('observer_video_path',observer)]}
        checks={k:{'passed':True} for k in subject.REQUIRED_CHECKS}
        checks['physical_counter_timing'].update(controls=150,observations=151,physics_integrations=600)
        checks['actual_runtime_identity'].update(full_roster_verified=True,frozen_dependency_count=2)
        checks['videos'].update(streams=streams,pair_consistency=subject.independent.check_video_pair(streams))
        audit={'schema':'pi05_g1_mobile_episode_cpu_audit_v2','passed':True,'anomalies':[],'autonomous_policy_trial':False,'result_path':str(case/'result.json'),
            'tool_sha256':subject.sha(subject.independent.__file__),'checks':checks,'inputs_sha256':bound,'recomputed_acceptance':{k:v for k,v in judged.items() if k!='samples'}}
        self.audit_write(root/'audit.json',audit)
        return helper,case,root/'audit.json',bundle,audit

    def audit_write(self,path,audit):
        audit.pop('audit_sha256',None);audit['audit_sha256']=hashlib.sha256(json.dumps(audit,sort_keys=True,allow_nan=False).encode()).hexdigest();path.write_text(json.dumps(audit))

    @unittest.skipUnless(shutil.which('ffmpeg'),'CPU ffmpeg required')
    def test_success_is_primitive_only_with_real_tiny_video_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);helper,case,ap,bundle,audit=self.fixture(root)
            original={str(p):subject.sha(p) for p in case.iterdir()}
            with patch.object(subject.measurement.body_ids,'_ORIGINAL_GEOMETRY',return_value=bundle) as fake,patch.object(subject.measurement.original,'geometry',fake):
                result=subject.qualify(case,ap,root/'qualification.json')
            self.assertEqual(result['status'],'completed');self.assertTrue(result['passed']);self.assertEqual(result['qualified_stable_2s_windows'],50)
            for key in ('full_task_success','autonomous','teacher'):self.assertIs(result[key],False)
            self.assertEqual(original,{str(p):subject.sha(p) for p in case.iterdir()})

    @unittest.skipUnless(shutil.which('ffmpeg'),'CPU ffmpeg required')
    def test_failed_producer_stale_sha_and_wrong_audit_tool_rejected(self):
        for mode in ('failed','stale','oldtool'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);helper,case,ap,bundle,audit=self.fixture(root)
                if mode=='failed':
                    p=case/'result.json';r=json.loads(p.read_text());r['status']='failed';p.write_text(json.dumps(r))
                if mode=='stale':(case/'runtime_identity.json').write_text('{"changed":true}')
                if mode=='oldtool':audit['tool_sha256']='0'*64;self.audit_write(ap,audit)
                result=subject.qualify(case,ap,root/'reject.json')
                self.assertFalse(result['passed']);self.assertEqual(result['status'],'failed')

    @unittest.skipUnless(shutil.which('ffmpeg'),'CPU ffmpeg required')
    def test_pose_only_or_readyspan_cannot_bypass_allpairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);helper,case,ap,bundle,audit=self.fixture(root)
            with patch.object(subject.measurement.body_ids,'_ORIGINAL_GEOMETRY',return_value=bundle) as fake,patch.object(subject.measurement.original,'geometry',fake),patch.object(subject.acceptance,'stable_pose_window',return_value=False):
                result=subject.qualify(case,ap,root/'reject.json')
            self.assertFalse(result['passed']);self.assertIn('all-pairs',result['error'])

if __name__=='__main__':unittest.main()
