"""CPU sandbox of unapplied runtime dependency pin; never reads real token/model bytes."""
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
PATCH_DIR=ROOT/'.scratch/pi05_g1/round02/prepared_post_training_patches'
PATCH=PATCH_DIR/'runtime_tokenizer_openpi_provenance.patch'
IDENTITY=PATCH_DIR/'runtime_tokenizer_openpi_provenance.identity.json'
PACKAGE='experiments.pi05_g1_round02'


def patched_sources():
    identity=json.loads(IDENTITY.read_text());content=PATCH.read_text()
    assert hashlib.sha256(content.encode()).hexdigest()==identity['patch_sha256']
    result={}
    for section in content.split('--- a/')[1:]:
        path,rest=section.split('\n',1);lines=rest.splitlines(True)[1:]
        original=(ROOT/path).read_text()
        digest=hashlib.sha256(original.encode()).hexdigest();entry=identity['files'][path]
        if digest==entry['after_sha256']:
            result[path]=original;continue
        assert digest==entry['before_sha256'], 'Unexpected source version: '+path
        base=original.splitlines(True);output=[];cursor=0;i=0
        while i<len(lines):
            match=re.match(r'@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@',lines[i]);assert match
            start=int(match[1])-1;output.extend(base[cursor:start]);cursor=start;i+=1
            while i<len(lines) and not lines[i].startswith('@@ '):
                marker,text=lines[i][0],lines[i][1:]
                if marker in ' -':assert base[cursor]==text;cursor+=1
                if marker in ' +':output.append(text)
                i+=1
        output.extend(base[cursor:]);source=''.join(output)
        assert hashlib.sha256(source.encode()).hexdigest()==entry['after_sha256']
        result[path]=source
    return result


class RuntimeProvenancePatchTests(unittest.TestCase):
    def setUp(self):
        self.sources=patched_sources();self.modules={}
        for filename in ('make_benchmark_plan','offline_diagnostic'):
            name=PACKAGE+'.'+filename;original=importlib.import_module(name)
            module=types.ModuleType(name);module.__file__=original.__file__;module.__package__=PACKAGE
            source=self.sources['experiments/pi05_g1_round02/'+filename+'.py']
            exec(compile(source,module.__file__,'exec'),module.__dict__);self.modules[name]=module
        package=importlib.import_module(PACKAGE)
        previous={name:sys.modules[name] for name in self.modules}
        for name,module in self.modules.items():sys.modules[name]=module
        self.addCleanup(lambda:sys.modules.update(previous))
        for name,module in self.modules.items():
            setter=patch.object(package,name.rsplit('.',1)[1],module);setter.start();self.addCleanup(setter.stop)
        self.factory=self.modules[PACKAGE+'.make_benchmark_plan'];self.offline=self.modules[PACKAGE+'.offline_diagnostic']

    def fixture_class(self,filename,classname):
        name=PACKAGE+'.tests.'+filename;module=types.ModuleType(name);module.__file__=str(ROOT/'experiments/pi05_g1_round02/tests'/str(filename+'.py'));module.__package__=PACKAGE+'.tests'
        exec(compile(self.sources['experiments/pi05_g1_round02/tests/'+filename+'.py'],module.__file__,'exec'),module.__dict__)
        return getattr(module,classname)

    def test_factory_freezes_external_source_and_tokenizer_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixture=self.fixture_class('test_benchmark_plan','BenchmarkPlanTests')();self.addCleanup(fixture.doCleanups)
            args,_=fixture.fixture(p);plans=self.factory.make_plan(**args)
            selection=json.loads((p/'eval/selection_receipt.json').read_text())
            descriptor=selection['runtime_provenance'];files=plans['primary']['frozen_files']
            self.assertEqual(len(descriptor['source_python_files']),2)
            for path in descriptor['source_python_files']+[descriptor['tokenizer']['path']]:self.assertIn(path,files)
            self.assertNotIn('venv/bin/python',files)
            token=Path(descriptor['tokenizer']['path']);token.write_bytes(b'x'*token.stat().st_size)
            args['output']=p/'second_eval'
            with self.assertRaisesRegex(ValueError,'tokenizer'):self.factory.make_plan(**args)
            self.assertFalse(args['output'].exists())

    def test_offline_inherits_selected_runtime_shas_and_rejects_changed_or_missing_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixture=self.fixture_class('test_offline_diagnostic','OfflineDiagnosticTests')();self.addCleanup(fixture.doCleanups)
            plan,_=fixture.fixture(p);selection=json.loads((p/'selection.json').read_text())
            runtime=selection['runtime_provenance']
            for path in runtime['source_python_files']+[runtime['tokenizer']['path']]:
                self.assertEqual(plan['frozen_files'][path],selection['frozen_files'][path])
            self.assertFalse((p/'NEVER_READ_TEST_OBSERVATIONS').exists())
            source=Path(runtime['source_python_files'][0]);source.write_text('# changed source\n')
            with self.assertRaisesRegex(ValueError,'frozen file SHA'):
                self.offline.make_plan(p/'selection.json',p/'bounds.json',p/'changed_offline')
            source.unlink()
            with self.assertRaisesRegex(ValueError,'source root'):
                self.offline.make_plan(p/'selection.json',p/'bounds.json',p/'missing_offline')

    def test_patched_existing_factory_and_offline_regressions(self):
        suite=unittest.TestSuite()
        for filename,classname in [('test_benchmark_plan','BenchmarkPlanTests'),('test_offline_diagnostic','OfflineDiagnosticTests')]:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(self.fixture_class(filename,classname)))
        result=unittest.TestResult();suite.run(result)
        self.assertEqual(result.testsRun,10)
        self.assertEqual(result.errors,[]);self.assertEqual(result.failures,[])


if __name__=='__main__':unittest.main()
