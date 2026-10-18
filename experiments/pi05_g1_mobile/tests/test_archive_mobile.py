"""Tiny CPU retention fixtures only: no real archive, GPU, or weight reads."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.pi05_g1_mobile import archive_mobile as a


def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


class ArchiveTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'sources';source.mkdir();sources=[]
        for category in sorted(a.CATEGORIES):
            directory=source/category;directory.mkdir();(directory/'record.json').write_text('{"original":"failed or interrupted evidence preserved"}');sources.append({'category':category,'path':str(directory),'destination':category,'immutable':category=='official_base','reason':'Synthetic immutable CPU toy'})
        payload=source/'official_base/model.safetensors';payload.write_bytes(b'TOY WEIGHTS ONLY')
        cache=source/'round03/venv';cache.mkdir();(cache/'excluded').write_bytes(b'NO ENV COPY')
        ledger=root/'ledger.json';write(ledger,{'runs':[{'status':'failed','accounting_open':False}]})
        evidence=root/'probe.txt';evidence.write_text('OWNED CLOSED; external game permitted')
        quiet=root/'quiet.json';write(quiet,{'schema':'pi05_g1_mobile_owned_quiescence_v1','passed':True,'owned_closed':True,'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'ledger':str(ledger),'ledger_sha256':a.sha(ledger),'evidence_files':{str(evidence):a.sha(evidence)},'global_idle':False,'non_display_compute_processes':[{'process_name':'external_game'}]})
        for file in [ledger,evidence,quiet]:sources.append({'category':'provenance','path':str(file),'destination':'gate/'+file.name})
        plan=root/'plan.json';write(plan,{'schema':'pi05_g1_mobile_archive_plan_v1','sources':sources,'required_files':[{'path':str(payload),'sha256':a.sha(payload)}],'reconstruction_commands':['docker build -f Dockerfile -t PINNED .'],'docker_image_id':'sha256:TOY','quiescence':str(quiet)})
        pin=patch.object(a,'BACKUP_ROOT',root/'backups');pin.start();self.addCleanup(pin.stop)
        return plan,root/'backups/new',payload

    def add_final(self,root,plan):
        training=root/'training.json';write(training,{'schema':'pi05_g1_mobile_training_result_v1','status':'completed','training_stopped':{'complete_optimizer_update_boundary':True}})
        episodes=[];audited=[];bindings={}
        for policy in ['gr00t_n16','pi05_mobile','official_pi05_base']:
            for case in range(20):
                result=root/'results'/policy/str(case)/'result.json';write(result,{'status':'completed','success':False});bindings[str(result)]=a.sha(result)
                episodes.append({'policy':policy,'case_id':str(case),'result_path':str(result)})
                row={'schema':'pi05_g1_mobile_episode_cpu_audit_v1','passed':True,'anomalies':[],'autonomous_policy_trial':True,'result_path':str(result),'inputs_sha256':{str(result):a.sha(result)}}
                row['audit_sha256']=hashlib.sha256(json.dumps(row,sort_keys=True,allow_nan=False).encode()).hexdigest();audited.append(row)
        summary=root/'summary.json';write(summary,{'schema':'pi05_g1_mobile_benchmark_summary_v1','complete':True,'evidence_qualified':True,'formal_verdict_available':True,'errors':[],'case_ids':[str(i) for i in range(20)],'policies':{p:{'completed_evidence_cases':20} for p in ['gr00t_n16','pi05_mobile','official_pi05_base']},'episodes':episodes,'inputs_sha256':bindings})
        audit=root/'audit.json';write(audit,{'schema':'pi05_g1_mobile_episode_audits_v1','passed':True,'episodes':audited})
        doc=a.read(plan);doc.update(training_result=str(training),benchmark=str(summary),episode_audit=str(audit))
        for file in [training,summary,audit]:doc['sources'].append({'category':'provenance','path':str(file),'destination':'final_gate/'+file.name})
        doc['sources'].append({'category':'provenance','path':str(root/'results'),'destination':'final_results'})
        write(plan,doc)
        return training,summary,audit

    def test_partial_retains_failure_bytes_metadata_copy_and_external_game(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,out,payload=self.fixture(root);before=payload.read_bytes();r=a.archive(plan,out,'partial')
            self.assertEqual(r['status'],'retained_partial');self.assertFalse(r['experiment_complete_claim']);self.assertFalse(r['readiness']['global_idle'])
            rows=[json.loads(line) for line in (out/'inventory.jsonl').read_text().splitlines()];files=[r for r in rows if r['type']!='directory']
            self.assertTrue(all(r['type']=='copy' for r in files if r['source_path'].endswith('.json')))
            weight=next(r for r in files if r['source_path']==str(payload));self.assertEqual((out/weight['relative_path']).read_bytes(),before)
            expected='hardlink' if payload.stat().st_uid!=0 and payload.stat().st_uid==os.getuid() else 'copy';self.assertEqual(weight['type'],expected)
            self.assertFalse((out/'retained/round03/venv').exists());self.assertTrue(r['omitted']);self.assertEqual(a.sha(out/'inventory.jsonl'),r['inventory_sha256'])
            with self.assertRaises(FileExistsError):a.archive(plan,out,'partial')

    def test_final_exact60_typed_gate_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,out,_=self.fixture(root);training,summary,audit=self.add_final(root,plan)
            r=a.archive(plan,out,'final');self.assertTrue(r['experiment_complete_claim']);self.assertEqual(r['status'],'retained_final')
        for kind in ['training','audit','roster','raw']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);plan,out,_=self.fixture(root);training,summary,audit=self.add_final(root,plan)
                if kind=='training':doc=a.read(training);doc['status']='controlled_interruption';write(training,doc)
                elif kind=='audit':doc=a.read(audit);doc['episodes'][0]['passed']=False;write(audit,doc)
                elif kind=='roster':doc=a.read(summary);doc['episodes'].pop();write(summary,doc)
                else:Path(a.read(summary)['episodes'][0]['result_path']).write_text('TAMPERED')
                with self.assertRaises(ValueError):a.archive(plan,out,'final')
                self.assertFalse(out.exists())

    def test_space_missing_closure_and_busy_refuse_before_copy(self):
        for kind in ['space','closure','busy','missing','escape']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);plan,out,payload=self.fixture(root);doc=a.read(plan)
                if kind=='closure':doc['required_files'][0]['path']=str(root/'nonexistent')
                elif kind=='busy':q=Path(doc['quiescence']);v=a.read(q);v['owned_closed']=False;write(q,v)
                elif kind=='missing':doc['sources'][0]['path']=str(root/'no_artifact')
                elif kind=='escape':doc['sources'][0]['destination']='../escape'
                write(plan,doc)
                context=patch.object(a.shutil,'disk_usage',return_value=type('Space',(),{'free':0})()) if kind=='space' else patch.object(a,'RESERVE',a.RESERVE)
                with context,self.assertRaises((ValueError,OSError)):a.archive(plan,out,'partial')
                self.assertFalse(out.exists())

    def test_link_fallback_root_owned_and_failed_copy_retains_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,out,_=self.fixture(root)
            with patch.object(a.os,'link',side_effect=PermissionError('root-owned/cross-FS fallback')):r=a.archive(plan,out,'partial')
            self.assertTrue(r['archive_complete'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,out,payload=self.fixture(root)
            with patch.object(a.shutil,'copyfileobj',side_effect=OSError('copy failed')),self.assertRaises(OSError):a.archive(plan,out,'partial')
            receipt=a.read(out/'archive_receipt.json');self.assertFalse(receipt['archive_complete']);self.assertEqual(receipt['status'],'archive_failed_partial_retained');self.assertEqual(payload.read_bytes(),b'TOY WEIGHTS ONLY')

if __name__=='__main__':unittest.main()
