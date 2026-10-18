"""Mock queue only; never imports/calls the actual FIFO or GPU supervisor."""
import copy,json,tempfile,unittest,os,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
from experiments.pi05_g1_mobile import round03_driver as d

class Queue:
    def __init__(self,mode='success'):self.mode=mode;self.row=None;self.calls=[];self.cancelled=0
    def request(self,owner,stage,seconds):
        self.calls.append(('request',owner,stage,seconds));self.row={'ticket':'t000001','owner':owner,'status':'waiting'};return dict(self.row)
    def snapshot(self,ticket):return dict(self.row)
    def cancel(self,ticket,owner):self.cancelled+=1;self.row['status']='cancelled'
    def run(self,ticket,ledger,command,wait):
        self.calls.append(('run',command));self.command=command
        if self.mode=='cancelwaiting':return 130
        if self.mode=='cancelled':self.row['status']='cancelled';return 130
        self.row.update(status='cleanup_unknown' if self.mode=='cleanupunknown' else 'completed',cleanup_confirmed=self.mode!='cleanupunknown')
        def option(key):return command[command.index(key)+1]
        path=Path(option('--result-file'));path.parent.mkdir(parents=True,exist_ok=True)
        result={'schema':option('--result-schema'),'status':'completed','success':False,'acceptance':{'max_ready_span_seconds':0 if self.mode=='partial' else 2}}
        if self.mode=='failed':result['status']='failed'
        path.write_text(json.dumps(result))
        row={'stage':option('--stage'),'status':'completed','exit_code':0,'elapsed_seconds':5,'last_unit_properties':{'ActiveState':'inactive'},'docker_cleanup':{'complete':True},'workload_completion_contract':{'path':str(path),'expected_schema':result['schema'],'valid':True}}
        Path(ledger).write_text(json.dumps({'runs':[row]}));return 0

class DriverTests(unittest.TestCase):
    def fixture(self,root):
        plan={'schema':d.SCHEMA,'root_reviewed':True,'owner':'ROOT_OWNER','coord_root':str(root/'mockcoord'),'ledger':str(root/'gpu_budget.json'),'supervisor_python':'/mock/python','stages':[{'id':'oracle','kind':'gpu','role':'oracle_gate','budget_stage':'preflight','max_seconds':120,'wait_seconds':30,'argv':['/mock/producer','--fixed','value'],'result':{'path':str(root/'oracle.json'),'schema':'oracle_v1','predicates':[{'path':['status'],'op':'eq','value':'completed'},{'path':['acceptance','max_ready_span_seconds'],'op':'gte','value':2}]}}]}
        p=root/'plan.json';p.write_text(json.dumps(plan));return p,plan
    def test_success_gpu_then_explicit_cpu_receipt(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,'MOBILE',Path(tmp)):
            root=Path(tmp);path,plan=self.fixture(root);cpu=root/'cpu.json'
            plan['stages'].append({'id':'audit','kind':'cpu','max_seconds':5,'argv':['/mock/audit'],'result':{'path':str(cpu),'schema':'cpu_v1','predicates':[{'path':['passed'],'op':'eq','value':True}]}});path.write_text(json.dumps(plan))
            def runner(argv,seconds):cpu.write_text(json.dumps({'schema':'cpu_v1','passed':True}));return 0
            queue=Queue();result=d.execute(path,root/'journal',queue=queue,cpu_runner=runner)
            self.assertEqual(result['status'],'completed');self.assertEqual(len(result['completed_stages']),2);self.assertIn('experiments.pi05_g1_mobile.budget',queue.command);self.assertIn('--result-schema',queue.command)
            self.assertEqual(queue.command[-3:],['/mock/producer','--fixed','value']);self.assertEqual(queue.cancelled,0)
    def test_failed_partial_cancelled_and_cleanup_unknown_never_continue(self):
        for mode in ('failed','partial','cancelled','cleanupunknown','cancelwaiting'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp,patch.object(d,'MOBILE',Path(tmp)):
                root=Path(tmp);path,plan=self.fixture(root);queue=Queue(mode)
                result=d.execute(path,root/'journal',queue=queue,cpu_runner=lambda *a:self.fail('must not proceed'))
                self.assertEqual(result['status'],'attention');self.assertEqual(result['completed_stages'],[]);self.assertFalse(result['automatic_retry'])
                self.assertEqual(queue.cancelled,1 if mode=='cancelwaiting' else 0)
    def test_dryrun_does_not_touch_queue_or_run_command(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,'MOBILE',Path(tmp)):
            root=Path(tmp);path,plan=self.fixture(root);queue=Queue()
            result=d.execute(path,root/'journal',dry_run=True,queue=queue,cpu_runner=lambda *a:self.fail('no commands'))
            self.assertEqual(result['status'],'dry_run');self.assertFalse(queue.calls);self.assertFalse((root/'oracle.json').exists())
    def test_existing_result_and_nonzero_cpu_exit_refused(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,'MOBILE',Path(tmp)):
            root=Path(tmp);path,plan=self.fixture(root);(root/'oracle.json').write_text('{}');queue=Queue()
            r=d.execute(path,root/'journal',queue=queue);self.assertEqual(r['status'],'attention');self.assertFalse(queue.calls)
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,'MOBILE',Path(tmp)):
            root=Path(tmp);path,plan=self.fixture(root);plan['stages'][0]['kind']='cpu';path.write_text(json.dumps(plan))
            r=d.execute(path,root/'journal',cpu_runner=lambda *a:1);self.assertEqual(r['status'],'attention')
    def test_training_requires_real_optimizer_boundary_and_final_validation(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,'MOBILE',Path(tmp)):
            root=Path(tmp);path,plan=self.fixture(root);stage=plan['stages'][0];stage.update(kind='cpu',role='training');stage['result']['schema']='train_v1';stage['result']['predicates']=[{'path':['status'],'op':'eq','value':'completed'}]
            document={'schema':'train_v1','status':'completed','step':800,'training_stopped':{'complete_optimizer_update_boundary':True,'step':800},'final_validation':{'complete':True,'flow_loss':.1},'final_checkpoint':'final800','selected_checkpoint':'step400'}
            resultpath=Path(stage['result']['path']);resultpath.write_text(json.dumps(document));d.check_result(stage,0)
            for key in ('training_stopped','final_validation'):
                bad=copy.deepcopy(document);bad.pop(key);resultpath.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):d.check_result(stage,0)
    def test_frozen_startup_and_poststage_changes_fail_without_refresh(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,'MOBILE',Path(tmp)):
            root=Path(tmp);path,plan=self.fixture(root);source=root/'source.py';source.write_text('approved')
            plan['frozen_files']={str(source):d.sha(source)};path.write_text(json.dumps(plan));source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'frozen'):d.execute(path,root/'badstartup',dry_run=True)
            source.write_text('approved');queue=Queue();original=queue.run
            def changed(*args):
                result=original(*args);source.write_text('changed during FIFO work');return result
            queue.run=changed
            result=d.execute(path,root/'journal',queue=queue)
            self.assertEqual(result['status'],'attention');self.assertEqual(result['completed_stages'],[])
            snapshot=json.loads((root/'journal/plan_snapshot.json').read_text());self.assertEqual(snapshot['frozen_files'],plan['frozen_files'])
            captured=list((root/'journal/frozen_snapshot').glob('*source.py'));self.assertEqual(captured[0].read_text(),'approved')
    def test_real_cpu_timeout_cleans_only_own_process_group_grandchild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pidfile=root/'grandchild.pid'
            # Grandchild ignores TERM. Parent's group is separate from this test.
            script="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); open(sys.argv[1],'w').write(str(p.pid)); time.sleep(60)"
            started=time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):d.cpu_execute([sys.executable,'-c',script,str(pidfile)],.5)
            self.assertLess(time.monotonic()-started,5)
            pid=int(pidfile.read_text());status=Path(f'/proc/{pid}/status')
            for _ in range(40):
                if not status.exists() or '\nState:\tZ' in status.read_text():break
                time.sleep(.025)
            self.assertTrue(not status.exists() or '\nState:\tZ' in status.read_text(),'grandchild still executing')
            self.assertTrue(Path(f'/proc/{os.getpid()}').exists())

    def test_no_implicit_resume_shell_or_unbounded_plan(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,'MOBILE',Path(tmp)):
            root=Path(tmp);path,plan=self.fixture(root);plan['stages'][0]['argv']='python command'
            with self.assertRaises(ValueError):d.validate_plan(plan)
            plan['stages'][0]['argv']=['/mock'];plan['stages'][0]['max_seconds']=float('inf')
            with self.assertRaises(ValueError):d.validate_plan(plan)

if __name__=='__main__':unittest.main()
