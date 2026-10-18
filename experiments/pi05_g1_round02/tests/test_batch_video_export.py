import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.pi05_g1_round02.batch_video_export import collect_jobs,export_job,closed_ledger,run,sha,ROOT


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def fixture(root,optional=False):
    positions=((0,0),(.02,0),(-.02,0),(0,.02),(0,-.02));phases=[('primary',('gr00t_n17','pi05_round01','pi05_round02'))]
    if optional:phases.append(('optional_base',('official_pi05_base',)))
    for phase,policies in phases:
        jobs=[];records=[];frozen={}
        for policy in policies:
            for position in range(5):
                for seed in range(40,44):
                    cid=f'p{position}_seed{seed}';name=policy+'_'+cid;raw=root/policy/cid
                    config={'case':{'case_id':cid,'seed':seed,'apple_offset_xy_m':list(positions[position])},'policy_id':policy,
                            'output_dir':str(raw),'video_output_dir':str(root/'video_exports'/policy/cid)}
                    cp=root/'configs'/(name+'.json');write(cp,config);write(raw/'config.json',config)
                    write(raw/'result.json',{'case':config['case'],'policy':policy,'status':'completed'})
                    (raw/'trajectory.jsonl').write_text('{}\n')
                    digest=sha(cp);frozen[str(cp)]=digest;jobs.append({'name':name,'config_path':str(cp),'config_sha256':digest})
                    records.append({'name':name,'output':str(raw),'status':'completed','config_file_sha256':digest,'result_sha256':sha(raw/'result.json')})
        exporter=ROOT/'experiments/pi05_g1_round02/video_export.py'
        plan=root/(phase+'_plan.json');write(plan,{'schema':'pi05_g1_round02_frozen_benchmark_plan_v1','phase':phase,'policies':list(policies),'jobs':jobs,'frozen_files':{str(exporter):sha(exporter)}})
        write(root/(phase+'_batch_receipt.json'),{'schema':'pi05_g1_round02_batch_receipt_v1','status':'completed','plan':str(plan),
                                               'plan_sha256':sha(plan),'frozen_files':frozen,'jobs':records})
    return root


class BatchVideoExportTests(unittest.TestCase):
    def test_exact_completed_primary60_optional80_and_fixed_representatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=fixture(Path(tmp)/'round02/final');jobs,phases,errors=collect_jobs(root)
            self.assertEqual(len(jobs),60);self.assertEqual(sum(j['wall_clock'] for j in jobs),3);self.assertEqual(errors,[])
            fixture(root,optional=True);jobs,phases,errors=collect_jobs(root)
            self.assertEqual(len(jobs),80);self.assertEqual(sum(j['wall_clock'] for j in jobs),4)
            receipt=root/'optional_base_batch_receipt.json';r=json.loads(receipt.read_text());r['status']='infrastructure_failed';write(receipt,r)
            jobs,phases,errors=collect_jobs(root);self.assertEqual(len(jobs),60);self.assertTrue(errors)
            self.assertEqual(phases[-1]['status'],'incomplete')

    def test_gpu_ledger_and_workers_reject_unsafe_cpu_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);ledger=p/'ledger.json'
            for row in ({'status':'running'},{'status':'cleanup_pending'},{'status':'failed','accounting_open':True},{'launcher_cleanup_pending':True}):
                write(ledger,{'schema':'pi05_g1_gpu_budget_v1','runs':[row]})
                with self.assertRaises(ValueError):closed_ledger(ledger)
            for n in (0,5):
                with self.assertRaises(ValueError):run(p,ledger,n)
            with self.assertRaises(ValueError):run(p,ledger,1)

    def test_exporter_uses_cpu_env_and_compatible_complete_can_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);root=fixture(p/'round02/final');job=collect_jobs(root)[0][0];ledger=p/'ledger.json'
            write(ledger,{'schema':'pi05_g1_gpu_budget_v1','runs':[]});work=p/'logs';work.mkdir();raw=Path(job['raw'])/'result.json';before=raw.read_bytes()
            def fake_export(command,**kwargs):
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(kwargs['env']['OMP_NUM_THREADS'],'1')
                self.assertEqual(command[1:3],['-m','experiments.pi05_g1_round02.video_export']);self.assertIn('--wall-clock',command)
                write(Path(job['output'])/'video_manifest.json',{});return SimpleNamespace(returncode=0)
            with patch('experiments.pi05_g1_round02.batch_video_export.subprocess.run',side_effect=fake_export),patch('experiments.pi05_g1_round02.batch_video_export.verify_export',return_value={'complete_mock':True}):
                receipt=export_job(job,work,ledger);self.assertEqual(receipt['status'],'complete');self.assertFalse(receipt['reused'])
            nextwork=p/'logs2';nextwork.mkdir()
            with patch('experiments.pi05_g1_round02.batch_video_export.subprocess.run') as exporter,patch('experiments.pi05_g1_round02.batch_video_export.verify_export',return_value={'complete_mock':True}):
                receipt=export_job(job,nextwork,ledger);self.assertTrue(receipt['reused']);exporter.assert_not_called()
            self.assertEqual(raw.read_bytes(),before)

    def test_incompatible_partial_export_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);root=fixture(p/'round02/final');job=collect_jobs(root)[0][0];ledger=p/'ledger.json'
            write(ledger,{'schema':'pi05_g1_gpu_budget_v1','runs':[]});work=p/'logs';work.mkdir()
            old=Path(job['output'])/'partial.mp4';old.parent.mkdir(parents=True);old.write_bytes(b'preserve partial')
            with patch('experiments.pi05_g1_round02.batch_video_export.subprocess.run') as exporter:
                receipt=export_job(job,work,ledger);self.assertEqual(receipt['status'],'incomplete');exporter.assert_not_called()
            self.assertEqual(old.read_bytes(),b'preserve partial')
            config=Path(root/'configs'/(job['name']+'.json'));r=json.loads(config.read_text());r['case']['seed']=100;write(config,r)
            with self.assertRaises(ValueError):collect_jobs(root)


if __name__=='__main__':unittest.main()
