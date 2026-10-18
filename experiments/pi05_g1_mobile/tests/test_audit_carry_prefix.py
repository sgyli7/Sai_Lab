"""Synthetic admission-only fixtures; never create real training data."""
import hashlib,json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from experiments.pi05_g1_mobile import audit_carry_prefix as a


def write(p,v):p.write_text(json.dumps(v))


class CarryAdmissionTests(unittest.TestCase):
    def fixture(self,p):
        folder=p/'case';folder.mkdir();(folder/'head_rgb').mkdir();case={'case_id':'SYNTHETIC_DEV_NATIVE201','seed':201}
        rgb=np.zeros((480,640,3),np.uint8);pixel=hashlib.sha256(rgb.tobytes()).hexdigest();rows=[]
        for tick in range(2):
            Image.fromarray(rgb).save(folder/'head_rgb'/f'{tick:05d}.png');rows.append({'control_tick':tick,'physics_steps_since_reset':tick*4,'sim_time_seconds':tick/50,'head_rgb_sha256':pixel})
        (folder/'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        score={'carry_qualified_control_tick':1,'success':False,'fell':False,'dropped':False,'max_carried_box_displacement_m':4.,'carry_turn_degrees':60.,'source_target_distance_m':5.,'termination_reasons':['no_released_stable_placement']}
        write(folder/'result.json',{'case':case,'policy':'gr00t_n16','status':'completed','success':False,'acceptance':{**score,'samples':[{'fell':False,'ground_drop':False}]*2}})
        write(folder/'config.json',{'case':case,'mode':'policy','policy_id':'gr00t_n16','prompt':a.PROMPT});write(folder/'runtime_identity.json',{'wbc_config':{'enable_waist':False}})
        streams={k:{'sha256':'SYNTHETIC_VIDEO_SHA','frames':2,'first_pts_seconds':.04} for k in ['head_video_path','observer_video_path']}
        child={'schema':'pi05_g1_mobile_episode_cpu_audit_v2','result_path':str(folder/'result.json'),'passed':True,'anomalies':[],'autonomous_policy_trial':True,'checks':{'actual_runtime_identity':{'passed':True,'frozen_dependency_count':1},'videos':{'passed':True,'streams':streams}},'inputs_sha256':{str(folder/name):a.sha(folder/name) for name in ['result.json','trajectory.jsonl','config.json','runtime_identity.json']},'recomputed_acceptance':score}
        child['audit_sha256']=hashlib.sha256(json.dumps(child,sort_keys=True,allow_nan=False).encode()).hexdigest()
        v2=p/'v2.json';write(v2,{'schema':'pi05_g1_mobile_episode_audits_v2','passed':True,'episodes':[child]})
        legacy=p/'legacy.json';write(legacy,{'schema':'pi05_g1_mobile_episode_audits_v1','passed':False,'episodes':[{'result_path':str(folder/'result.json'),'passed':False,'anomalies':['ValueError: Video PTS/time origin mismatch']}]})
        diagnostic=p/'diagnostic.json';write(diagnostic,{'schema':'pi05_g1_mobile_native_dev04_physical_diagnostic_v1','acceptance_recomputed_exact_all_fields_with_1e6_float_tolerance':True,'videos':{key:{'sha256':'SYNTHETIC_VIDEO_SHA','frames':2,'first_pts_seconds':.04,'relative_50hz_contiguous':True} for key in ['head_50fps.mp4','observer_50fps.mp4']}})
        return folder,v2,legacy,diagnostic,p/'admission.json'

    def test_scope_boundary_and_original_failed_task_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));original=a.sha(args[0]/'result.json');r=a.admit(*args)
            self.assertEqual(r['status'],'passed');self.assertFalse(r['source_task_success']);self.assertFalse(r['legacy_v1_audit_passed']);self.assertTrue(r['independent_v2_audit_passed'])
            self.assertEqual([f['control_tick'] for f in r['raw_rgb_frames']],[0]);self.assertEqual(r['boundary_frame']['control_tick'],1);self.assertEqual(r['audited_raw_png_count'],2);self.assertFalse(r['data_conversion_executed']);self.assertEqual(a.sha(args[0]/'result.json'),original)
            with self.assertRaises(FileExistsError):a.admit(*args)

    def test_failed_v2_raw_tamper_or_boundary_pixel_change_refuses(self):
        for kind in ['v2failed','result','png','legacy']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                args=self.fixture(Path(tmp))
                if kind=='v2failed':o=a.read(args[1]);o['passed']=False;write(args[1],o)
                elif kind=='result':(args[0]/'result.json').write_text('{}')
                elif kind=='png':Image.fromarray(np.ones((480,640,3),np.uint8)).save(args[0]/'head_rgb/00001.png')
                else:o=a.read(args[2]);o['passed']=True;write(args[2],o)
                with self.assertRaises((ValueError,KeyError)):a.admit(*args)
                self.assertFalse(args[-1].exists())

if __name__=='__main__':unittest.main()
