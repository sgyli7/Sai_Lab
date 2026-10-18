"""Mocked read-only resource commands; no real Docker/systemd/GPU calls."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from experiments.pi05_g1_mobile.quiescence import probe


class MobileQuiescenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.ledger=self.root/'ledger.json';self.token='a'*32;self.cid='b'*64;self.compute='';self.active=False;self.query_error=False;self.container=None;self.calls=[]
        self.row={'status':'result_contract_failed','raw_supervisor_status':'completed','raw_supervisor_exit_code':0,'ended_utc':'then','elapsed_seconds':12.5,'budget_token':self.token,'owned_unit':f'pi05-g1-owned-{self.token}.service','docker_cleanup':{'complete':True,'containers':[{'container_id':self.cid}]}}

    def command(self,argv,**kwargs):
        self.calls.append(argv);code,out,error=0,'',''
        if argv[0]=='systemctl':out='ActiveState='+('active' if self.active else 'inactive')+'\nSubState=dead\nResult=success\n'
        elif argv[:2]==['docker','inspect']:
            if self.container is None:code,error=1,'Error: No such object: '+self.cid
            else:out=json.dumps([self.container])
        elif argv[0]=='nvidia-smi':
            out=self.compute
            if self.query_error:code,error=1,'query unavailable'
        return subprocess.CompletedProcess(argv,code,out,error)

    def run_probe(self,name='probe',rows=None):
        self.ledger.write_text(json.dumps({'runs':rows or [self.row]}))
        with patch('experiments.pi05_g1_mobile.quiescence.subprocess.run',side_effect=self.command):return probe(self.ledger,self.root/(name+'.json'))

    def test_new_failed_contract_terminal_is_owned_closed_not_workload_success(self):
        result=self.run_probe();self.assertTrue(result['owned_closed']);self.assertTrue(result['passed']);self.assertTrue(result['global_idle']);self.assertEqual(result['accounting_open_tokens'],[])
        self.assertEqual(json.loads(self.ledger.read_text())['runs'][0]['status'],'result_contract_failed')
        for path,digest in result['evidence_files'].items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),digest)
        self.assertTrue(all(not set(command).intersection({'run','stop','kill','rm'}) for command in self.calls))

    def test_carried_overlap_resolves_exact_closed_source_without_double_charge(self):
        source=self.root/'previous_ledger.json';source.write_text(json.dumps({'runs':[self.row]}))
        carried={'status':'carried_static_shutdown_overlap','elapsed_seconds':24.788666,'ended_utc':'then','source_ledger':str(source),'source_budget_token':self.token,'accounting_open':False}
        result=self.run_probe(rows=[carried]);self.assertTrue(result['owned_closed']);self.assertAlmostEqual(result['total_gpu_seconds'],24.788666)
        self.assertEqual(len(result['unit_checks']),1);self.assertEqual(result['accounting_open_tokens'],[])
        previous=json.loads(source.read_text());previous['runs'][0]['accounting_open']=True;source.write_text(json.dumps(previous))
        self.assertFalse(self.run_probe('source_open',rows=[carried])['owned_closed'])

    def test_foreign_compute_does_not_block_owned_closure_or_become_idle(self):
        self.compute='GPU-fixture,354,/external/UserGame,354\n'
        result=self.run_probe();self.assertTrue(result['owned_closed']);self.assertFalse(result['global_idle']);self.assertTrue(result['passed'])
        self.assertEqual(result['non_display_compute_processes'][0]['pid'],354)
        self.assertTrue(all(command[0] in ('systemctl','docker','nvidia-smi') for command in self.calls))

    def test_gpu_query_failure_is_unknown_global_not_unclosed_owned(self):
        self.query_error=True;result=self.run_probe();self.assertTrue(result['owned_closed']);self.assertFalse(result['global_idle']);self.assertFalse(result['compute_inventory_verified'])
        self.assertTrue(result['errors'])

    def test_active_unit_open_accounting_and_foreign_label_are_not_closed(self):
        self.active=True;self.assertFalse(self.run_probe('active')['owned_closed']);self.active=False
        self.row['accounting_open']=True;self.assertFalse(self.run_probe('open')['owned_closed']);self.row.pop('accounting_open')
        self.container={'Config':{'Labels':{'sai.pi05_g1.budget':'foreign'}},'State':{'Running':False,'Status':'exited'}}
        self.assertFalse(self.run_probe('foreign_label')['owned_closed'])

    def test_user_bus_permission_failure_not_collected_unit(self):
        self.ledger.write_text(json.dumps({'runs':[self.row]}));original=self.command
        def error(argv,**kwargs):
            if argv[0]=='systemctl':return subprocess.CompletedProcess(argv,1,'','Failed to connect to bus: endpoint not found')
            return original(argv,**kwargs)
        with patch('experiments.pi05_g1_mobile.quiescence.subprocess.run',side_effect=error):self.assertFalse(probe(self.ledger,self.root/'denied.json')['owned_closed'])

    def test_ledger_race_rejected_and_existing_report_never_overwritten(self):
        self.run_probe()
        with self.assertRaises(FileExistsError):probe(self.ledger,self.root/'probe.json')
        self.ledger.write_text(json.dumps({'runs':[self.row]}));original=self.command
        def race(argv,**kwargs):
            if argv[0]=='nvidia-smi':self.ledger.write_text('{}')
            return original(argv,**kwargs)
        with patch('experiments.pi05_g1_mobile.quiescence.subprocess.run',side_effect=race):result=probe(self.ledger,self.root/'raced.json')
        self.assertFalse(result['owned_closed']);self.assertFalse(result['global_idle'])


if __name__=='__main__':unittest.main()
