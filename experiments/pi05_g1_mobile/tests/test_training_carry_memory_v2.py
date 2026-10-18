"""CPU toy/mocked allocator tests only; no real CUDA context or large model."""
import ast,json,tempfile,types,sys,unittest,random
from pathlib import Path
from unittest.mock import patch,Mock
import numpy as np
import torch
# Initialize optional optimizer Python modules outside mocked sys.modules scopes.
import torch._dynamo
from experiments.pi05_g1_mobile import training_carry_memory_v2 as t
from experiments.pi05_g1_mobile import training_carry as old
from experiments.pi05_g1_mobile.tests.test_training import Runtime,Dataset

class MemoryTests(unittest.TestCase):
    def test_numerical_update_and_rng_unchanged_by_cpu_cache_boundary(self):
        a=Runtime();b=Runtime();b.model.load_state_dict(a.model.state_dict());oa=t.MasterAdamW(a.model.parameters(),lr=1e-5);ob=t.MasterAdamW(b.model.parameters(),lr=1e-5);events=[]
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        result_a=old.update(a,Dataset(),oa,torch.ones(32),float('inf'));state=torch.get_rng_state().clone();python=random.getstate();numpy=np.random.get_state()
        boundary=t.IdleMemoryBoundaries(a,oa,lambda *a,**kw:events.append((a,kw)))
        with patch.object(t,'memory_snapshot',return_value={}),patch.object(t,'release_cpu_staging',return_value={'mock':True}),patch.object(torch.cuda,'empty_cache',side_effect=AssertionError('CPU cannot CUDA')):boundary.run('optimizer_boundary')
        self.assertTrue(torch.equal(state,torch.get_rng_state()));self.assertEqual(python,random.getstate());self.assertTrue(np.array_equal(numpy[1],np.random.get_state()[1]))
        random.seed(42);np.random.seed(42);torch.manual_seed(42);result_b=t.update(b,Dataset(),ob,torch.ones(32),float('inf'))
        self.assertEqual(result_a,result_b)
        for x,y in zip(a.model.parameters(),b.model.parameters()):self.assertTrue(torch.equal(x,y))
        for x,y in zip(oa.master_parameters,ob.master_parameters):self.assertTrue(torch.equal(x,y))
        self.assertEqual(events[0][0],('idle_memory_boundary',))
    def test_rejects_live_accumulated_grads_instead_of_discarding_them(self):
        runtime=Runtime();optimizer=t.MasterAdamW(runtime.model.parameters(),lr=1e-5);parameter=optimizer.model_parameters[0];parameter.grad=torch.ones_like(parameter)
        boundary=t.IdleMemoryBoundaries(runtime,optimizer,lambda *a,**k:None)
        with patch.object(t,'release_cpu_staging') as cleanup:
            with self.assertRaisesRegex(RuntimeError,'cleared'):boundary.run('unsafe_mid_microbatch')
        cleanup.assert_not_called();self.assertTrue(torch.equal(parameter.grad,torch.ones_like(parameter)))
    def test_mock_cuda_empty_cache_only_after_cleared_grads_and_sync(self):
        calls=[];cuda=types.SimpleNamespace(is_initialized=lambda:True,synchronize=lambda device:calls.append('sync'),empty_cache=lambda:calls.append('empty_cache'))
        runtime=types.SimpleNamespace(torch=types.SimpleNamespace(cuda=cuda),device=types.SimpleNamespace(type='cuda'))
        optimizer=types.SimpleNamespace(model_parameters=[types.SimpleNamespace(grad=None)],master_parameters=[types.SimpleNamespace(grad=None)])
        with patch.object(t,'memory_snapshot',side_effect=[{'cuda_allocated_bytes':40,'cuda_reserved_bytes':70},{'cuda_allocated_bytes':40,'cuda_reserved_bytes':41}]),patch.object(t,'release_cpu_staging',return_value={'mock':True}):result=t.IdleMemoryBoundaries(runtime,optimizer,lambda *a,**k:None).run('optimizer_boundary')
        self.assertEqual(calls,['sync','empty_cache','sync']);self.assertEqual(result['before']['cuda_allocated_bytes'],result['after']['cuda_allocated_bytes'])
    def test_math_checkpoint_precision_deadlines_and_data_admission_ast_unchanged(self):
        functions=lambda source:{n.name:ast.dump(n) for n in ast.parse(Path(source).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=functions(old.__file__),functions(t.__file__)
        for name in ('update','validate','checkpoint','preflight','precision','StopRequest','parser','disk_estimate'):self.assertEqual(a[name],b[name])
        self.assertEqual(t.parser().parse_args(['--output','UNUSED']).save_optimizer,False)
    def test_full_toy_training_hooks_preserve_original_validation_selection_and_complete_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';base.mkdir();(base/'model.safetensors').write_bytes(b'tinyfake')
            args=t.parser().parse_args(['--output',str(root/'run'),'--checkpoint',str(base),'--max-steps','2'])
            admission={'disk_estimate':{'model_bytes':10000,'final_optimizer_bytes_upper_bound':0,'reserve_bytes':0},'manifest_sha256':'manifest','norm_sha256':'norm','official_base':{'sha256':t.sha(base/'model.safetensors')},'frozen_files':{}}
            runtime=Runtime();runtime.dtype_summary={};module=types.ModuleType('experiments.pi05_g1_mobile.model_runtime');module.ModelRuntime=lambda *a,**kw:runtime;valset=Dataset()
            values=[{'flow_loss':1.,'complete':True,'chunks':190,'frames':9240,'episodes':10},{'flow_loss':.9,'complete':True,'chunks':190,'frames':9240,'episodes':10}]
            with patch.dict(sys.modules,{module.__name__:module}),patch.object(t,'preflight',return_value=admission),patch.object(t,'QuantileNormalizer',return_value=None),patch.object(t,'EpisodeDataset',side_effect=[Dataset(),valset]),patch.object(t,'validate',side_effect=values) as validate,patch.object(t,'memory_snapshot',return_value={}),patch.object(t,'release_cpu_staging',return_value={'mock':True}):report=t.train(args)
            self.assertEqual(report['status'],'completed');self.assertEqual(report['step'],2);self.assertEqual(report['final_validation']['chunks'],190);self.assertTrue(report['training_stopped']['complete_optimizer_update_boundary'])
            self.assertTrue(all(x.args[1] is valset for x in validate.call_args_list));self.assertFalse(report['test_loaded']);self.assertFalse(report['optimizer_payload_saved'])
            rows=[json.loads(x) for x in (root/'run/progress.jsonl').read_text().splitlines()];phases=[x['phase'] for x in rows if x['event']=='idle_memory_boundary']
            self.assertEqual(phases.count('optimizer_boundary'),2);self.assertEqual(phases.count('before_validation'),2);self.assertEqual(phases.count('after_validation'),2);self.assertEqual(phases.count('before_checkpoint'),1);self.assertEqual(phases.count('after_checkpoint'),1)
            self.assertIn('memory_policy',report['training_configuration'])

if __name__=='__main__':unittest.main()
