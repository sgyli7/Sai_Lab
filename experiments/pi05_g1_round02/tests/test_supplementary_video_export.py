"""Exercise the prepared patch in a temporary copy, never the live producer."""
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.pi05_g1_round02.tests.test_batch_video_export import fixture, write


REPO = Path(__file__).resolve().parents[3]
PATCH = REPO / '.scratch/pi05_g1/round02/prepared_post_training_patches/batch_video_export_supplementary.patch'
TARGET = Path('experiments/pi05_g1_round02/batch_video_export.py')
IDENTITY = PATCH.with_name(PATCH.name+'.identity.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_copy(source, root, identity):
    """Accept exact original/updated states; apply only to a disposable copy."""
    source_before = digest(source)
    if source_before not in (identity['original_sha256'], identity['patched_sha256']):
        raise AssertionError('Producer is neither the complete original nor prepared updated identity')
    if digest(PATCH) != identity['patch_sha256']:
        raise AssertionError('Prepared patch identity changed')
    target = root / TARGET
    target.parent.mkdir(parents=True)
    shutil.copy2(source, target)
    try:
        if source_before == identity['original_sha256']:
            for dry in (True, False):
                command = ['patch', '--batch', '--forward', '-p1', '-d', str(root), '-i', str(PATCH)]
                if dry:
                    command.append('--dry-run')
                subprocess.run(command, check=True, capture_output=True, text=True)
        if digest(target) != identity['patched_sha256']:
            raise AssertionError('Sandbox copy does not match complete updated identity')
    finally:
        if digest(source) != source_before:
            raise AssertionError('Source changed while testing disposable producer copy')
    return target


class SupplementaryExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.live_before = digest(REPO/TARGET)
        cls.identity = json.loads(IDENTITY.read_text())
        cls.sandbox = tempfile.TemporaryDirectory()
        root = Path(cls.sandbox.name)
        target = prepare_copy(REPO/TARGET, root, cls.identity)
        spec = importlib.util.spec_from_file_location('sandbox_supplementary_export', target)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.module.ROOT = REPO

    @classmethod
    def tearDownClass(cls):
        try:
            if digest(REPO/TARGET) != cls.live_before:
                raise AssertionError('Live producer changed across supplementary test suite')
        finally:
            cls.sandbox.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = fixture(Path(self.temp.name) / 'evaluation')

    def tearDown(self):
        self.assertEqual(digest(REPO/TARGET), self.live_before)

    def test_both_complete_source_identities_and_no_partial_acceptance(self):
        root = Path(self.temp.name)
        updated_source = root/'updated_source'/TARGET
        updated_source.parent.mkdir(parents=True)
        shutil.copy2(Path(self.sandbox.name)/TARGET, updated_source)
        original_source = root/'original_source'/TARGET
        original_source.parent.mkdir(parents=True)
        shutil.copy2(updated_source, original_source)
        subprocess.run(['patch','--batch','--reverse','-p1','-d',str(root/'original_source'),
                        '-i',str(PATCH)],check=True,capture_output=True,text=True)
        self.assertEqual(digest(original_source),self.identity['original_sha256'])
        for name, source in (('original',original_source),('updated',updated_source)):
            source_before = digest(source)
            target = prepare_copy(source,root/('copy_'+name),self.identity)
            self.assertEqual(digest(target),self.identity['patched_sha256'])
            self.assertEqual(digest(source),source_before)
        partial = root/'partial.py'
        partial.write_text('SUPPLEMENTARY=()\n')
        with self.assertRaises(AssertionError):
            prepare_copy(partial,root/'copy_partial',self.identity)

    def add_supplementary(self, status='completed'):
        # Reuse synthetic exact-final20 data, then relocate only synthetic base
        # identity into the newly predeclared policy/phase.
        other = fixture(Path(self.temp.name) / 'synthetic_source', optional=True)
        source = json.loads((other/'optional_base_plan.json').read_text())
        receipt = json.loads((other/'optional_base_batch_receipt.json').read_text())
        jobs, records, frozen = [], [], {}
        for job, record in zip(source['jobs'], receipt['jobs']):
            config = json.loads(Path(job['config_path']).read_text())
            cid = config['case']['case_id']
            policy = 'pi05_fresh_validation'
            raw = self.root/policy/cid
            name = policy+'_'+cid
            config.update(policy_id=policy, output_dir=str(raw),
                          video_output_dir=str(self.root/'video_exports'/policy/cid))
            cp = self.root/'configs'/(name+'.json')
            write(cp, config)
            write(raw/'config.json', config)
            write(raw/'result.json', {'case':config['case'],'policy':policy,'status':'completed'})
            (raw/'trajectory.jsonl').write_text('{}\n')
            digest = self.module.sha(cp)
            jobs.append({'name':name,'config_path':str(cp),'config_sha256':digest})
            records.append({'name':name,'output':str(raw),'status':'completed',
                            'config_file_sha256':digest,'result_sha256':self.module.sha(raw/'result.json')})
            frozen[str(cp)] = digest
        plan_path = self.root/'supplementary_fresh_plan.json'
        source.update(phase='supplementary_fresh',policies=['pi05_fresh_validation'],jobs=jobs)
        write(plan_path,source)
        receipt.update(status=status,plan=str(plan_path),plan_sha256=self.module.sha(plan_path),
                       jobs=records,frozen_files=frozen)
        write(self.root/'supplementary_fresh_batch_receipt.json',receipt)

    def status(self, jobs, phases, errors):
        value = {'jobs':[{'job':j,'status':'complete'} for j in jobs],
                 'phases':phases,'errors':errors}
        self.module.delivery_status(value)
        return value

    def test_unexecuted_fallback_does_not_block_complete80(self):
        fixture(self.root,optional=True)
        (self.root/'optional_base_batch_receipt.json').unlink()
        self.add_supplementary()
        jobs, phases, errors = self.module.collect_jobs(self.root)
        self.assertEqual((len(jobs),sum(j['wall_clock'] for j in jobs),errors),(80,4,[]))
        self.assertEqual({j['policy'] for j in jobs},set(self.module.PRIMARY)|{'pi05_fresh_validation'})
        self.assertEqual(next(p for p in phases if p['phase']=='optional_base')['status'],'not_executed')
        result = self.status(jobs,phases,errors)
        self.assertTrue(result['primary_complete'])
        self.assertEqual(result['deliverable_cases'],80)
        self.assertEqual(result['additional_incomplete'],[])

    def test_partial_extra_not_promoted_main60_independently_deliverable(self):
        self.add_supplementary('infrastructure_failed')
        jobs, phases, errors = self.module.collect_jobs(self.root)
        self.assertEqual(len(jobs),60)
        self.assertTrue(errors)
        result = self.status(jobs,phases,errors)
        self.assertTrue(result['primary_complete'])
        self.assertEqual(result['deliverable_policies'],list(self.module.PRIMARY))
        self.assertEqual(result['deliverable_cases'],60)
        self.assertEqual(result['additional_incomplete'][0]['policy'],'pi05_fresh_validation')
        self.assertEqual(result['additional_incomplete'][0]['status'],'incomplete')

    def test_mutually_exclusive_actual_execution_even_if_failed(self):
        fixture(self.root,optional=True)
        self.add_supplementary('infrastructure_failed')
        with self.assertRaisesRegex(ValueError,'mutually exclusive'):
            self.module.collect_jobs(self.root)

    def test_no_supplement_main60_and_failed_export_qualification(self):
        jobs, phases, errors = self.module.collect_jobs(self.root)
        self.assertEqual((len(jobs),errors),(60,[]))
        self.assertTrue(self.status(jobs,phases,errors)['primary_complete'])
        self.add_supplementary()
        jobs, phases, errors = self.module.collect_jobs(self.root)
        result = self.status(jobs,phases,errors)
        next(r for r in result['jobs'] if r['job']['policy']=='pi05_fresh_validation')['status']='incomplete'
        self.module.delivery_status(result)
        self.assertTrue(result['primary_complete'])
        self.assertEqual(result['deliverable_cases'],60)
        self.assertEqual(result['additional_incomplete'][0]['complete_exports'],19)

    def test_completed_partial20_and_hash_tampering_rejected(self):
        self.add_supplementary()
        receipt_path = self.root/'supplementary_fresh_batch_receipt.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['jobs'].pop()
        write(receipt_path,receipt)
        with self.assertRaisesRegex(ValueError,'Incomplete/duplicate'):
            self.module.collect_jobs(self.root)
        self.add_supplementary()
        receipt = json.loads(receipt_path.read_text())
        receipt['plan_sha256']='0'*64
        write(receipt_path,receipt)
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            self.module.collect_jobs(self.root)

    def test_full_run_preserves_schema_and_marks_additional_failure(self):
        self.add_supplementary('infrastructure_failed')
        ledger = Path(self.temp.name)/'ledger.json'
        write(ledger,{'schema':'pi05_g1_gpu_budget_v1','runs':[]})
        def fake_export(job, workdir, ledger):
            return {'status':'complete','job':job}
        with patch.object(self.module,'export_job',side_effect=fake_export), patch('builtins.print'):
            _, result = self.module.run(self.root,ledger,workers=1)
        self.assertEqual(result['schema'],'pi05_g1_round02_cpu_video_export_batch_v1')
        self.assertEqual(result['status'],'incomplete')
        self.assertFalse(result['complete'])
        self.assertTrue(result['primary_complete'])
        self.assertEqual(result['deliverable_cases'],60)


if __name__ == '__main__':
    unittest.main()
