"""Native v2 producer/consumer seam fixtures; real fragmented H264, no GPU."""
import copy,hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.pi05_g1_mobile import report_v2 as report,archive_mobile_v2 as archive
from experiments.pi05_g1_mobile.tests import test_report as legacy_report,test_archive_mobile as legacy_archive,test_audit_episodes_v2 as codec
from experiments.pi05_g1_mobile.audit_episodes_v2 import video_probe


def write(p,o):p.write_text(json.dumps(o))

def upgrade_audit(audit,common_inputs,streams):
    row=copy.deepcopy(audit);row['schema']='pi05_g1_mobile_episode_cpu_audit_v2';row['anomalies']=[]
    for name in ['physical_counter_timing','initial_kinematic_sha','actions','policy_feedback','acceptance']:row.setdefault('checks',{}).setdefault(name,{'passed':True})
    row['checks']['actual_runtime_identity']={'passed':True,'full_roster_verified':True,'frozen_dependency_count':1}
    row['checks']['videos']={'passed':True,'streams':streams,'pair_consistency':{'passed':True}}
    row['inputs_sha256'].update(common_inputs);row['audit_sha256']=report.audit_identity(row);return row


class DeliveryV2Tests(unittest.TestCase):
    def report_fixture(self,p):
        sp,ap,summary,audits=legacy_report.ReportTests().fixture(p)
        media=p/'tiny.mp4';media.unlink();codec.encode(media,fragmented=True,n=6)
        common={}
        for name in ['config.json','runtime_identity.json']:
            path=p/name;write(path,{'synthetic_cpu_only':True});common[str(path)]=report.sha(path)
        common[str(p/'trajectory.jsonl')]=report.sha(p/'trajectory.jsonl');common[str(media)]=report.sha(media)
        for audit in audits:
            path=Path(audit['result_path']);raw=report.read(path);raw['completed_control_ticks']=5;write(path,raw);summary['inputs_sha256'][str(path)]=report.sha(path);audit['inputs_sha256'][str(path)]=report.sha(path)
        write(sp,summary)
        probe=video_probe(media,6);streams={key:{'path':str(media),'sha256':report.sha(media),**probe} for key in ['head_video_path','observer_video_path']}
        audits=[upgrade_audit(audit,common,streams) for audit in audits]
        write(ap,{'schema':'pi05_g1_mobile_episode_audits_v2','passed':True,'episodes':audits})
        return sp,ap,summary,audits

    def test_real_fragmented_guard_to_complete_report_no_schema_spoof(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);sp,ap,summary,audits=self.report_fixture(p);raw=report.sha(p/'tiny.mp4')
            # Rendering reuses the unchanged code from the explicitly versioned
            # report; only dependency availability is isolated by caller env.
            receipt=report.generate(sp,[ap],p/'delivery')
            self.assertEqual(receipt['schema'],'pi05_g1_mobile_report_receipt_v2');self.assertEqual(receipt['raw_video_streams'],120);self.assertEqual(report.sha(p/'tiny.mp4'),raw)
            index=report.read(p/'delivery/video_index.json');self.assertGreater(index['videos'][0]['first_pts_seconds'],0.);self.assertEqual(index['schema'],'pi05_g1_mobile_report_video_index_v2')
            self.assertEqual(report.sha(report.LEGACY_SOURCE_PATH),report.LEGACY_SOURCE_SHA256)

    def test_failed_missing_runtime_origin_and_old_schema_are_rejected(self):
        for kind in ['failed','missing','origin','schema']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);sp,ap,summary,audits=self.report_fixture(p);bad=copy.deepcopy(audits)
                if kind=='failed':bad[0]['checks']['actions']['passed']=False
                elif kind=='missing':bad[0]['checks'].pop('policy_feedback')
                elif kind=='origin':bad[0]['checks']['videos']['streams']['head_video_path']['first_pts_seconds']=.2
                else:bad[0]['schema']='pi05_g1_mobile_episode_cpu_audit_v1'
                bad[0]['audit_sha256']=report.audit_identity(bad[0]);write(ap,{'schema':'pi05_g1_mobile_episode_audits_v2','passed':True,'episodes':bad})
                with self.assertRaises(ValueError):report.load_evidence(sp,[ap])

    def test_archive_final_native_v2_gate_and_failure_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);helper=legacy_archive.ArchiveTests();plan,out,payload=helper.fixture(p);self.addCleanup(helper.doCleanups);training,summary,ap=helper.add_final(p,plan)
            document=archive.read(ap);upgraded=[]
            for old in document['episodes']:
                folder=Path(old['result_path']).parent;common={}
                for name in ['config.json','trajectory.jsonl','runtime_identity.json','head.mp4','observer.mp4']:
                    f=folder/name;f.write_bytes(b'SYNTHETIC_ONLY');common[str(f)]=archive.sha(f)
                streams={key:{'path':str(folder/name),'sha256':common[str(folder/name)],'frames':2,'fps':50,'first_pts_seconds':.04,'last_pts_seconds':.06,'relative_pts_seconds':[0,.02],'origin_rule':'MP4 packaging'} for key,name in [('head_video_path','head.mp4'),('observer_video_path','observer.mp4')]}
                upgraded.append(upgrade_audit(old,common,streams))
            write(ap,{'schema':'pi05_g1_mobile_episode_audits_v2','passed':True,'episodes':upgraded})
            with patch.object(archive,'BACKUP_ROOT',p/'backups'):
                receipt=archive.archive(plan,out,'final')
                self.assertEqual(receipt['schema'],'pi05_g1_mobile_archive_receipt_v2');self.assertTrue(receipt['experiment_complete_claim']);self.assertEqual(receipt['reserve_bytes'] if 'reserve_bytes' in receipt else receipt['disk_preflight']['reserve_bytes'],32*1024**3)
                bad=archive.read(ap);bad['episodes'][0]['checks'].pop('actions');bad['episodes'][0]['audit_sha256']=report.audit_identity(bad['episodes'][0]);write(ap,bad)
                with self.assertRaises(ValueError):archive.archive(plan,p/'backups/rejected','final')

    def test_v3_cli_false_receipt_has_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
            completed=subprocess.run([sys.executable,'-m','experiments.pi05_g1_mobile.audit_episodes_v3',str(p/'missing'),'--output',str(p/'audit.json')],capture_output=True,text=True,env=env)
            self.assertNotEqual(completed.returncode,0);self.assertFalse(json.loads((p/'audit.json').read_text())['passed'])

if __name__=='__main__':unittest.main()
