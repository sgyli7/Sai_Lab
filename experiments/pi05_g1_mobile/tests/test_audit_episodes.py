"""Synthetic geometry/contacts plus real CPU PNG/MP4 codec fixtures; no GPU."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from experiments.pi05_g1_mobile import audit_episodes as a,acceptance,contract
from experiments.pi05_g1_mobile.tests import test_acceptance as acceptance_fixture


def write(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,allow_nan=False))

def video(path,rgb,n):
    import av
    with av.open(str(path),'w') as out:
        stream=out.add_stream('libx264',rate=50);stream.width=640;stream.height=480;stream.pix_fmt='yuv420p'
        for _ in range(n):
            frame=av.VideoFrame.from_ndarray(rgb,format='rgb24')
            for packet in stream.encode(frame):out.mux(packet)
        for packet in stream.encode():out.mux(packet)


class MobileEpisodeAuditTests(unittest.TestCase):
    def fixture(self,p):
        fixture=acceptance_fixture.AcceptanceAdversarialTests();fixture.setUp();config=copy.deepcopy(fixture.config)
        for key in ['geometry_definition','robot_geometry']:
            path=p/(key+'.json');write(path,{'synthetic':True});config[key]=str(path);config[key+'_sha256']=a.sha(path)
        meta=config.pop('audit_contact_metadata');meta['filter_paths']=[meta['filter_paths']];body_names=config.pop('audit_body_names')
        limits=p/'limits.json';lower=[-.2]*28+[.7,-.4,-.2,-.4];upper=[.2]*28+[.8,.4,.2,.4];write(limits,{'lower':lower,'upper':upper})
        config.update(mode='policy',policy_id='pi05_mobile',case={'case_id':'SYNTHETIC_CPU','seed':42},output_dir=str(p),ticks=2,episode_seconds=30.,evaluate_acceptance=True,prompt=contract.PROMPT,replan_interval_ticks=50,native_action_horizon=50,action_limits_file=str(limits),policy_command=['python','service','--decoded-evidence',str(p/'decoded.jsonl')])
        names=list(contract.DATASET_NAMES);rgb=np.zeros((480,640,3),np.uint8);rgb[:,:,0]=111;rgbsha=hashlib.sha256(rgb.tobytes()).hexdigest()
        chunk=np.full((50,32),.1,np.float32);chunk[:,0]=.5;chunk[:,28]=.75
        rows=[]
        for tick in range(3):
            row=fixture.row(tick,[0.,0.,.1]);row.update(sim_time_seconds=tick/50,monotonic_ns=1_000_000_000+tick*100_000_000,source_joint_names=names,joint_positions=(np.arange(43,dtype=np.float32)/1000).tolist(),joint_velocities=[0.]*43,root_link_velocity_w=[0.]*6,root_com_velocity_w=[0.]*6,head_rgb_sha256=rgbsha,issued_action32_raw=None,applied_action32=None,applied_source_action50=None)
            if tick:
                raw=chunk[tick-1];clipped=np.clip(raw,np.asarray(lower,np.float32),np.asarray(upper,np.float32));row.update(issued_action32_raw=raw.tolist(),applied_action32=clipped.tolist(),applied_source_action50=contract.decode_to_source(clipped,rows[-1]['joint_positions'],names).tolist())
            rows.append(row)
        context={**config,'audit_body_names':body_names,'audit_contact_metadata':meta}
        with patch.object(acceptance,'geometry',return_value=fixture.geometry):score=acceptance.evaluate_episode(rows,context)
        self.assertFalse(score['success'])
        from experiments.pi05_g1_mobile.simulator import validate_config
        config=validate_config(config)
        write(p/'config.json',config);trace=p/'trajectory.jsonl';trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        result={'schema':'pi05_g1_mobile_case_result_v1','status':'completed','policy':'pi05_mobile','case_id':'SYNTHETIC_CPU','source_interface_passed':True,'source_joint_names':names,'control_hz':50,'physics_hz':200,'completed_control_ticks':2,'actual_sdk_integrations':8,'trace_sha256':a.sha(trace),'initial_physical_state_schema':'pi05_g1_mobile_initial_kinematic_v1','initial_physical_state_sha256':hashlib.sha256(json.dumps({k:rows[0][k] for k in a.INITIAL_KEYS},sort_keys=True).encode()).hexdigest(),'actual_action_bounds':{'lower':lower,'upper':upper},'action_limits_sha256':a.sha(limits),'commanded_channels':64,'clipped_channels':2,'inference_calls':[{'observation_control_tick':0,'request_started_monotonic_ns':1_010_000_000,'reply_received_monotonic_ns':1_020_000_000}],'success':False,'acceptance':score,'body_names':body_names,'contact_view':meta,'head_video_path':str(p/'head_50fps.mp4'),'observer_video_path':str(p/'observer_50fps.mp4')}
        write(p/'result.json',result);captures=p/'policy_observations';captures.mkdir();np.save(captures/'actions_00000.npy',chunk,allow_pickle=False);Image.fromarray(rgb).save(captures/'head_00000.png')
        write(captures/'observation_00000.json',{'control_tick':0,'prompt':contract.PROMPT,'measured_joint_names43':names,'measured_joint_positions43':rows[0]['joint_positions'],'state':contract.encode_state(rows[0]['joint_positions'],names).tolist(),'rgb_sha256':rgbsha})
        (p/'decoded.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in [{'event':'model_loaded','load_seconds':.01},{'event':'pi05_decoded','control_tick':0,'episode_id':'SYNTHETIC_CPU','actions32':chunk.tolist()}]))
        for name in ['head_50fps.mp4','observer_50fps.mp4']:video(p/name,rgb,3)
        return fixture,rows,result

    def audit(self,p,fixture,**kwargs):
        with patch.object(acceptance,'geometry',return_value=fixture.geometry):return a.audit_case(p,**kwargs)

    def test_failed_task_with_nested_real_sdk_meta_and_full_codec_action_evidence_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixture,rows,result=self.fixture(p);before=a.sha(p/'result.json');report=self.audit(p,fixture)
            self.assertTrue(report['passed'],report['anomalies']);self.assertFalse(report['checks']['acceptance']['success'])
            self.assertEqual(report['checks']['policy_feedback']['queries'][0]['executed_prefix'],2)
            self.assertEqual(report['checks']['videos']['streams']['head_video_path']['frames'],3)
            self.assertEqual(report['checks']['actions']['clipped_channels'],2);self.assertEqual(a.sha(p/'result.json'),before)
            config_report=self.audit(p/'config.json',fixture);self.assertTrue(config_report['passed'],config_report['anomalies'])

    def test_raw_mapping_query_pixel_and_missing_chunk_reject(self):
        for kind in ['source','rgb','missingchunk','querytime','state','extra_truth']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);fixture,rows,result=self.fixture(p)
                if kind=='source':
                    rows[1]['applied_source_action50'][43]+=.1;trace=p/'trajectory.jsonl';trace.write_text(''.join(json.dumps(r)+'\n' for r in rows));result['trace_sha256']=a.sha(trace);write(p/'result.json',result)
                elif kind=='rgb':Image.fromarray(np.zeros((480,640,3),np.uint8)).save(p/'policy_observations/head_00000.png')
                elif kind=='missingchunk':(p/'policy_observations/actions_00000.npy').unlink()
                else:
                    op=p/'policy_observations/observation_00000.json';o=a.read(op)
                    if kind=='querytime':o['control_tick']=1
                    elif kind=='extra_truth':o['object_world_pose']=[1,2,3]
                    else:o['state'][0]+=.01
                    write(op,o)
                report=self.audit(p,fixture);self.assertFalse(report['passed']);self.assertTrue(report['anomalies'])

    def test_changed_acceptance_wall_and_wrong_video_length_reject(self):
        for kind in ['wall','video']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);fixture,rows,result=self.fixture(p)
                if kind=='wall':result['acceptance']['task_wall_seconds']+=.001;write(p/'result.json',result)
                else:video(p/'observer_50fps.mp4',np.zeros((480,640,3),np.uint8),2)
                report=self.audit(p,fixture);self.assertFalse(report['passed']);self.assertTrue(report['anomalies'])

    def test_failed_producer_remains_nonformal_even_with_available_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixture,rows,result=self.fixture(p);result['status']='failed';result['success']=None;result.pop('acceptance');write(p/'result.json',result)
            report=self.audit(p,fixture,allow_nonformal=True)
            self.assertFalse(report['passed']);self.assertTrue(report['checks']['actions']['passed']);self.assertTrue(report['checks']['videos']['passed'])
            self.assertIsNotNone(report['recomputed_acceptance']);self.assertFalse(report['checks']['acceptance']['passed'])

if __name__=='__main__':unittest.main()
