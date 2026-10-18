import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from experiments.pi05_g1_mobile import report


class ReportTests(unittest.TestCase):
    def fixture(self,root):
        # Tiny media reused by60 synthetic results, never real benchmark evidence.
        media=root/'tiny.mp4'
        (root/'trajectory.jsonl').write_text('\n'.join(json.dumps({'monotonic_ns':n}) for n in (1000000000,1500000000))+'\n')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=blue:s=80x60:r=50',
            '-frames:v','2','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(media)],check=True)
        summary={'schema':'pi05_g1_mobile_benchmark_summary_v1','status':'complete','complete':True,
            'evidence_qualified':True,'errors':[],'policy_order':['a','b','c'],
            'case_ids':['p'+str(i) for i in range(20)],'episodes':[],'inputs_sha256':{},'policies':{},
            'paired_comparisons':{'ab':{'first':'a','second':'b','matched_success_count':0}}}
        audits=[]
        for policy in summary['policy_order']:
            summary['policies'][policy]={'formal_success_count':0,'success_wilson95ci':[0.,.16],
                'counts_observed':{'dropped':0,'fell':0,'timeout':20}}
            for case in summary['case_ids']:
                path=root/f'{policy}_{case}.json'
                raw={'status':'completed','policy':policy,'case_id':case,'completed_control_ticks':1,
                    'head_video_path':str(media),'observer_video_path':str(media)}
                path.write_text(json.dumps(raw))
                flags={k:False for k in report.FLAGS};flags['timeout']=True
                metrics={k:0. for k in report.METRICS}
                ep={'policy':policy,'case_id':case,**flags,**metrics,'result_path':str(path),
                    'RPC_pause_wall_seconds':[.2,.3],'model_load_seconds':[.1],
                    'task_sim_seconds':.02,'task_wall_seconds':.5,'episode_sim_seconds':.02,
                    'episode_wall_seconds':.5,'cold_process_to_frame0_seconds':.1}
                summary['episodes'].append(ep);summary['inputs_sha256'][str(path)]=report.sha(path)
                videos={field:{'path':str(media),'sha256':report.sha(media),'frames':2,'fps':50}
                        for field in ('head_video_path','observer_video_path')}
                audit={'schema':'pi05_g1_mobile_episode_cpu_audit_v1','passed':True,
                    'autonomous_policy_trial':True,'result_path':str(path),'inputs_sha256':{str(path):report.sha(path)},
                    'checks':{'videos':{'passed':True,'streams':videos}},'recomputed_acceptance':{**flags,**metrics}}
                audit['audit_sha256']=report.audit_identity(audit);audits.append(audit)
        sp=root/'summary.json';sp.write_text(json.dumps(summary));ap=root/'audits.json'
        ap.write_text(json.dumps({'schema':'pi05_g1_mobile_episode_audits_v1','passed':True,'episodes':audits}))
        return sp,ap,summary,audits

    @unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg required')
    def test_complete_report_and_raw_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sp,ap,summary,audits=self.fixture(root);before={str(p):report.sha(p) for p in root.iterdir()}
            out=root/'report';receipt=report.generate(sp,[ap],out)
            self.assertEqual(receipt['completed_cases'],60);self.assertEqual(receipt['raw_video_streams'],120)
            text=(out/'index.html').read_text();self.assertIn('N/A: no both-success',text)
            self.assertIn('no physical robot capability claim',text)
            self.assertEqual(len(json.loads((out/'video_index.json').read_text())['videos']),120)
            self.assertTrue((out/'outcomes.png').is_file())
            for p,digest in before.items():self.assertEqual(report.sha(p),digest)
            with self.assertRaises(FileExistsError):report.generate(sp,[ap],out)

    @unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg required')
    def test_reject_partial_bad_audit_changed_media_and_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sp,ap,summary,audits=self.fixture(root)
            bad=copy.deepcopy(summary);bad['complete']=False;sp.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'completed'):report.load_evidence(sp,[ap])
            sp.write_text(json.dumps(summary));bad=copy.deepcopy(audits);bad[0]['passed']=False
            ap.write_text(json.dumps({'schema':'pi05_g1_mobile_episode_audits_v1','passed':True,'episodes':bad}))
            with self.assertRaisesRegex(ValueError,'audit identity'):report.load_evidence(sp,[ap])
            ap.write_text(json.dumps({'schema':'pi05_g1_mobile_episode_audits_v1','passed':True,'episodes':audits}))
            bad=copy.deepcopy(summary);bad['episodes'][0]['carry_turn_degrees']=9;sp.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'recomputation'):report.load_evidence(sp,[ap])
            sp.write_text(json.dumps(summary));(root/'tiny.mp4').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'video changed'):report.load_evidence(sp,[ap])

    def test_representative_same_case_frozen_order_then_lexical_common(self):
        summary={'case_ids':['z','b','a'],'policy_order':['a','b','c']}
        episodes={(p,c):{'success':c!='z'} for p in summary['policy_order'] for c in summary['case_ids']}
        self.assertEqual(report.representatives(summary,episodes),[
            {'kind':'first_frozen_case','case_id':'z'},
            {'kind':'smallest_case_id_all_three_success','case_id':'a'}])
        episodes[('c','a')]['success']=False;episodes[('c','b')]['success']=False
        self.assertEqual(len(report.representatives(summary,episodes)),1)

    def test_caption_holds_both_observed_clocks_and_event_is_retrospective(self):
        row={'success':True,'first_grasp_sim_seconds':0.,'qualified_stability_endpoint_sim_seconds':.02}
        caption,event=report.frame_caption('p','head',row,1,2,[0.,4.])
        self.assertIn('Final outcome: success',caption)
        self.assertEqual(event,'QUALIFIED at sim 0.02s')
        caption,event=report.frame_caption('p','head',row,1000,2,[0.,4.])
        self.assertIn('sim 0.02s | actual obs-wall 4.00s',caption)
        self.assertIn('END OF RECORDING / LAST FRAME HELD',event)
        self.assertIn('Stopped sim 0.02s; obs-wall 4.00s',event)

    @unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg required')
    def test_tiny_montage_real_decode_and_explicit_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,summary,_=self.fixture(root)
            episodes={}
            for i,p in enumerate(summary['policy_order']):
                raw=copy.deepcopy(summary['episodes'][0]);raw.update(policy=p)
                raw['_result']={'head_video_path':str(root/'tiny.mp4'),'observer_video_path':str(root/'tiny.mp4'),
                    'completed_control_ticks':1 if i else 0}
                episodes[(p,'case')]=raw
            # Use an actual one-frame stream for the short policy.
            short=root/'short.mp4';subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(root/'tiny.mp4'),
                '-frames:v','1','-c:v','libx264','-threads','1',str(short)],check=True)
            episodes[('a','case')]['_result'].update(head_video_path=str(short),observer_video_path=str(short))
            path=root/'comparison.mp4';receipt=report.montage('case',summary['policy_order'],episodes,path,width=480,height=360)
            self.assertEqual(receipt['frames'],2);self.assertEqual(receipt['fps'],50)
            self.assertIn('omitted',receipt['timing'])
            self.assertIn('END OF RECORDING',receipt['short_stream_semantics'])
            self.assertIn('Final outcome',receipt['selection'])
            decoded=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-f','null','-'],capture_output=True)
            self.assertEqual(decoded.returncode,0,decoded.stderr)


if __name__=='__main__':unittest.main()
