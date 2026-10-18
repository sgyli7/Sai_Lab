"""Small synthetic ledgers/mock legacy entrypoint only; no real supervisor."""
import copy,datetime,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.pi05_g1_mobile import budget_allocated as b

def utc(seconds):return datetime.datetime.fromtimestamp(seconds,datetime.timezone.utc).isoformat()

class AllocatedTests(unittest.TestCase):
    def fixture(self,root,current=None):
        initial=dict(b.INITIAL);current=current or {'preflight':3000,'training':9000,'evaluation':31200}
        authority={'schema':'pi05_g1_mobile_new_budget_authorization_v1','total_limit_seconds':43200,'stage_limits_seconds':initial,'accounting_start_utc':utc(0)}
        ap=root/'authority.json';ap.write_text(json.dumps(authority))
        row={'stage':'preflight','elapsed_seconds':100.,'status':'completed','started_utc':utc(10),'ended_utc':utc(110),'last_unit_properties':{'ActiveState':'inactive'},'docker_cleanup':{'complete':True}}
        amendment={'schema':'pi05_g1_mobile_stage_allocation_amendment_v1','utc':utc(200),'total_limit_seconds_unchanged':43200,'previous_stage_limits_seconds':initial,'new_stage_limits_seconds':current,'consumed_seconds_unchanged':100.,'prior_ledger_sha256':'a'*64}
        doc={'schema':'pi05_g1_gpu_budget_v1','total_limit_seconds':43200,'stage_limits_seconds':current,'authorization_receipt':str(ap),'runs':[row],'allocation_amendments':[amendment]}
        path=root/'ledger.json';path.write_text(json.dumps(doc));path.with_suffix('.lock').write_text('')
        return path,doc,authority
    def test_current_caps_delegated_unchanged_and_globals_restored(self):
        for limits in ({'preflight':3000,'training':9000,'evaluation':31200},{'preflight':2600,'training':9000,'evaluation':31600}):
            with self.subTest(limits=limits),tempfile.TemporaryDirectory() as tmp,patch.object(b.legacy,'LEDGER_ROOT',Path(tmp)):
                root=Path(tmp);path,doc,authority=self.fixture(root,limits);old=b.legacy.LIMITS;argv=['--ledger',str(path),'--result-file',str(root/'result.json'),'--result-schema','typed_v1','--stage','training','--max-seconds','100','--','fake']
                def main(received):self.assertEqual(received,argv);self.assertEqual(b.legacy.LIMITS,limits);raise SystemExit(0)
                with patch.object(b.legacy,'main',side_effect=main):
                    with self.assertRaises(SystemExit) as exit:b.invoke(argv)
                self.assertEqual(exit.exception.code,0);self.assertIs(b.legacy.LIMITS,old)
    def test_changed_total_chain_open_usage_and_active_amendment_rejected(self):
        for kind in ('total','chain','open','overstage','amendactive','missingcleanup','nan'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                path,doc,authority=self.fixture(Path(tmp))
                if kind=='total':doc['total_limit_seconds']=50000
                elif kind=='chain':doc['allocation_amendments'][0]['previous_stage_limits_seconds']={'preflight':5400,'training':16200,'evaluation':21600}
                elif kind=='open':doc['runs'][0]['accounting_open']=True
                elif kind=='overstage':doc['runs'][0]['elapsed_seconds']=3100
                elif kind=='amendactive':doc['allocation_amendments'][0]['utc']=utc(50)
                elif kind=='missingcleanup':doc['runs'][0]['docker_cleanup']={}
                else:doc['stage_limits_seconds']['training']=float('nan')
                with self.assertRaises(ValueError):b.validate_document(doc,authority)
    def test_future_closed_reallocation_continuous_chain_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,doc,authority=self.fixture(Path(tmp));previous=dict(doc['stage_limits_seconds']);new={'preflight':2600,'training':9000,'evaluation':31600}
            second=copy.deepcopy(doc['allocation_amendments'][0]);second.update(utc=utc(300),previous_stage_limits_seconds=previous,new_stage_limits_seconds=new)
            doc['allocation_amendments'].append(second);doc['stage_limits_seconds']=new
            self.assertEqual(b.validate_document(doc,authority)['limits'],new)
    def test_allocation_change_during_delegate_rejected_and_global_restored(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(b.legacy,'LEDGER_ROOT',Path(tmp)):
            path,doc,authority=self.fixture(Path(tmp));old=b.legacy.LIMITS
            def changed(argv):doc['stage_limits_seconds']={'preflight':2600,'training':9000,'evaluation':31600};path.write_text(json.dumps(doc));raise SystemExit(0)
            with patch.object(b.legacy,'main',side_effect=changed):
                with self.assertRaises(ValueError):b.invoke(['--ledger',str(path)])
            self.assertIs(b.legacy.LIMITS,old)
    def test_mandatory_result_still_enforced_by_real_legacy_parser(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(b.legacy,'LEDGER_ROOT',Path(tmp)):
            path,doc,authority=self.fixture(Path(tmp))
            with patch.object(b.legacy.supervisor,'main',side_effect=AssertionError('No supervisor may launch')):
                with self.assertRaises(SystemExit) as exit:b.invoke(['--ledger',str(path)])
            self.assertEqual(exit.exception.code,2)
    def test_readonly_cli_does_not_change_ledger_or_call_legacy(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(b.legacy,'LEDGER_ROOT',Path(tmp)):
            path,doc,authority=self.fixture(Path(tmp));before=path.read_bytes()
            with patch.object(b.legacy,'main',side_effect=AssertionError('readonly')):b.main(['--check-only','--ledger',str(path)])
            self.assertEqual(path.read_bytes(),before)

if __name__=='__main__':unittest.main()
