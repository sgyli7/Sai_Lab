"""CPU mocked systemd/docker only; no real service/container/GPU command runs."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.pi05_g1_mobile import budget,sim_launcher


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class MobileBudgetTests(unittest.TestCase):
    def ledger(self,root):
        path=root/'gpu_budget.json';write(path,{'total_limit_seconds':43200,'stage_limits_seconds':dict(budget.LIMITS),'runs':[]});return path

    def test_duration_stage_limits_and_carried_overlap_not_forgiven(self):
        ledger={'runs':[{'stage':'preflight','elapsed_seconds':24.788666}]}
        self.assertEqual(sum(budget.LIMITS.values()),43200)
        self.assertEqual(budget.LIMITS,{'preflight':5400,'training':16200,'evaluation':21600})
        self.assertAlmostEqual(budget.remaining_seconds(ledger,'preflight'),5400-24.788666)
        self.assertEqual(budget.remaining_seconds(ledger,'training'),16200)
        self.assertEqual(budget.remaining_seconds(ledger,'evaluation'),21600)

    def run_wrapper(self,root,*,result_body=None,raw_code=0,active=False):
        ledger=self.ledger(root);result=root/'producer_result.json';old_limits=budget.supervisor.LIMITS;old_total=budget.supervisor.TOTAL
        def fake_supervisor():
            import sys
            self.assertEqual(budget.supervisor.TOTAL,43200);self.assertEqual(budget.supervisor.LIMITS,budget.LIMITS)
            self.assertIn('10000',sys.argv);self.assertNotIn('--result-file',sys.argv)
            doc=json.loads(ledger.read_text());doc['runs'].append({'supervisor_pid':os.getpid(),'status':'completed' if raw_code==0 else 'failed','exit_code':raw_code,'elapsed_seconds':1,'last_unit_properties':{'ActiveState':'active' if active else 'inactive'},'docker_cleanup':{'complete':True}});write(ledger,doc)
            if result_body is not None:write(result,result_body)
            raise SystemExit(raw_code)
        with patch.object(budget,'LEDGER_ROOT',root),patch.object(budget.supervisor,'main',side_effect=fake_supervisor):
            with self.assertRaises(SystemExit) as exit:
                budget.main(['--ledger',str(ledger),'--result-file',str(result),'--result-schema','mobile_completed_v1','--stage','preflight','--max-seconds','10000','--','synthetic_command'])
        self.assertIs(budget.supervisor.LIMITS,old_limits);self.assertEqual(budget.supervisor.TOTAL,old_total)
        return exit.exception.code,json.loads(ledger.read_text())['runs'][-1]

    def test_sigint_style_raw_zero_missing_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            code,row=self.run_wrapper(Path(directory))
            self.assertEqual(code,2);self.assertEqual(row['status'],'result_contract_failed')
            self.assertEqual(row['raw_supervisor_status'],'completed');self.assertEqual(row['raw_supervisor_exit_code'],0)
            self.assertFalse(row['workload_completion_contract']['valid'])

    def test_completed_task_failure_is_valid_infrastructure_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            code,row=self.run_wrapper(Path(directory),result_body={'schema':'mobile_completed_v1','status':'completed','success':False})
            self.assertEqual(code,0);self.assertEqual(row['status'],'completed');self.assertIs(row['workload_completion_contract']['task_success'],False)

    def test_nonzero_cannot_be_forgiven_by_valid_result(self):
        with tempfile.TemporaryDirectory() as directory:
            code,row=self.run_wrapper(Path(directory),raw_code=7,result_body={'schema':'mobile_completed_v1','status':'completed'})
            self.assertEqual(code,7);self.assertEqual(row['raw_supervisor_exit_code'],7)

    def test_wrong_schema_or_unfinished_status_rejected(self):
        for body in ({'schema':'wrong','status':'completed'},{'schema':'mobile_completed_v1','status':'model_initialized'}):
            with tempfile.TemporaryDirectory() as directory:
                code,row=self.run_wrapper(Path(directory),result_body=body)
                self.assertEqual(code,2);self.assertEqual(row['status'],'result_contract_failed')

    def test_preexisting_result_wrong_ledger_or_unclosed_resources_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ledger=self.ledger(root);result=root/'result.json';write(result,{'schema':'old'})
            with patch.object(budget,'LEDGER_ROOT',root),patch.object(budget.supervisor,'main') as main:
                with self.assertRaises(FileExistsError):budget.main(['--ledger',str(ledger),'--result-file',str(result),'--result-schema','anything','--stage','preflight','--','unused'])
                main.assert_not_called()
            with patch.object(budget,'LEDGER_ROOT',root/'different'):
                with self.assertRaises(ValueError):budget.finalize_workload(ledger,0,{},0)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(RuntimeError,'Resources not confirmed closed'):self.run_wrapper(root,result_body={'schema':'mobile_completed_v1','status':'completed'},active=True)
            row=json.loads((root/'gpu_budget.json').read_text())['runs'][-1]
            self.assertNotIn('raw_supervisor_status',row)

    def test_stale_contract_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.json';write(path,{'schema':'r','status':'completed'});os.utime(path,ns=(1,1))
            self.assertFalse(budget.completion_contract(path,'r',100)['valid'])


class MobileLauncherTests(unittest.TestCase):
    def test_readonly_mounts_new_evidence_memory_and_frozen_image(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(sim_launcher,'ROOT',Path(directory)):
            root=Path(directory);command=sim_launcher.docker_command(root/'config.json',probe_only=True,launch_id='testlaunch')
            volumes=[command[i+1] for i,item in enumerate(command[:-1]) if item=='--volume']
            evidence=root/'.scratch/pi05_g1/round03_mobile'
            self.assertIn(f'{root}:{root}:ro',volumes);self.assertIn('/home/ethan/models:/home/ethan/models:ro',volumes)
            self.assertIn(f'{evidence}:{evidence}:rw',volumes);self.assertFalse(any('/round02' in item for item in command))
            self.assertIn('experiments.pi05_g1_mobile.sim_launcher',command);self.assertIn('--probe',command)
            self.assertIn(sim_launcher.OWNED_IMAGE,command);self.assertEqual(command[command.index('--memory')+1],'90g')
            self.assertNotIn('--gpus',command);self.assertNotIn('--device',command)
            self.assertEqual(sim_launcher.BUILD,'6.0.0-rc.22+release.33481.407f3ea1.gl')

    def test_owned_registration_cleanup_identity_and_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ownership=root/'ownership';cid='a'*64;holder={};calls=[]
            def fake_popen(command):
                holder['command']=command;Path(command[command.index('--cidfile')+1]).write_text(cid)
                return SimpleNamespace(wait=lambda **kwargs:0,poll=lambda:0)
            def fake_run(command,**kwargs):
                calls.append(command)
                if len(calls)==1:
                    launch=next(item.split('=',1)[1] for item in holder['command'] if item.startswith('sai.pi05_g1.launch='))
                    return SimpleNamespace(returncode=0,stdout=json.dumps([{'Id':cid,'Config':{'Labels':{'sai.pi05_g1.launch':launch}},'State':{'Running':False}}]),stderr='')
                return SimpleNamespace(returncode=1,stdout='',stderr='No such container')
            with patch.object(sim_launcher,'ROOT',root),patch.dict(os.environ,{'PI05_G1_BUDGET_TOKEN':'our-token','PI05_G1_BUDGET_OWNERSHIP_DIR':str(ownership)}),patch.object(sim_launcher.subprocess,'Popen',side_effect=fake_popen),patch.object(sim_launcher.subprocess,'run',side_effect=fake_run),patch.object(sim_launcher.time,'time',side_effect=[100.,104.5]):
                sim_launcher.run_owned_docker(root/'config.json',probe_only=True)
            registration=json.loads(next(ownership.glob('launch-*.json')).read_text());receipt=json.loads(Path(registration['cleanup_receipt']).read_text())
            self.assertEqual(registration['budget_token'],'our-token');self.assertEqual(registration['status'],'finished');self.assertIs(registration['removed'],True)
            self.assertEqual(receipt['finished_unix_s']-receipt['started_unix_s'],4.5);self.assertFalse(receipt['gpu_requested'])
            self.assertFalse(any(command[1] in ('stop','kill') for command in calls))

    def test_missing_paired_owner_identity_refused_before_popen(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(sim_launcher,'ROOT',Path(directory)),patch.dict(os.environ,{'PI05_G1_BUDGET_TOKEN':'token'},clear=True),patch.object(sim_launcher.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(ValueError,'supplied together'):sim_launcher.run_owned_docker(Path(directory)/'config.json',probe_only=True)
            popen.assert_not_called()

    def test_foreign_container_launch_label_is_never_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cid='b'*64;calls=[]
            def fake_popen(command):
                Path(command[command.index('--cidfile')+1]).write_text(cid)
                return SimpleNamespace(wait=lambda **kwargs:0,poll=lambda:0)
            def fake_run(command,**kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0,stdout=json.dumps([{'Id':cid,'Config':{'Labels':{'sai.pi05_g1.launch':'foreign'}},'State':{'Running':True}}]),stderr='')
            with patch.object(sim_launcher,'ROOT',root),patch.dict(os.environ,{},clear=True),patch.object(sim_launcher.subprocess,'Popen',side_effect=fake_popen),patch.object(sim_launcher.subprocess,'run',side_effect=fake_run):
                sim_launcher.run_owned_docker(root/'config.json',probe_only=True)
            self.assertFalse(any(command[1] in ('stop','kill') for command in calls))
            receipt=json.loads(next((root/'.scratch/pi05_g1/round03_mobile/sim_launchers').glob('*/receipt.json')).read_text())
            self.assertFalse(receipt['container_ownership_verified'])

    def test_copy_only_expected_namespace_and_evidence_changes(self):
        root=Path(__file__).resolve().parents[3]
        original=(root/'experiments/pi05_g1_round02/sim_launcher.py').read_text()
        expected=original.replace('.scratch/pi05_g1/round02','.scratch/pi05_g1/round03_mobile').replace('experiments.pi05_g1_round02.sim_launcher','experiments.pi05_g1_mobile.sim_launcher').replace('experiments.pi05_g1_round02.simulator','experiments.pi05_g1_mobile.simulator')
        self.assertEqual(expected,(root/'experiments/pi05_g1_mobile/sim_launcher.py').read_text())


if __name__=='__main__':unittest.main()
