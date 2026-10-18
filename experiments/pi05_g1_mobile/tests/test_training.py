"""Synthetic CPU-only schedule, prompt, validation, and atomic-save regressions."""
import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.pi05_g1_mobile import training as t


class Dataset:
    def __init__(self):self.indices=list(range(8));self.episodes=[{'id':0}];self.calls=0
    def sample(self,index):
        self.calls+=1
        return np.zeros((2,2,3),np.uint8),np.zeros(32,np.float32),np.ones((50,32),np.float32),np.ones(50,bool)
    def validation_indices(self):return [0,1,2]


class Runtime:
    def __init__(self):
        self.torch=torch;self.device=torch.device('cpu');self.model=torch.nn.Linear(32,32,bias=False);self.prompts=[]
        self.model.weight.data.fill_(.01)
        original=self.model.forward
        self.model.forward=lambda observation,target:(original(target)-target)**2
    def observation(self,rgb,state,prompt):self.prompts.append(prompt);return torch.zeros(32)


class MobileTrainingTests(unittest.TestCase):
    def test_fixed_defaults_and_signal_only_records_request(self):
        args=t.parser().parse_args(['--output','UNUSED'])
        self.assertEqual((args.max_steps,args.checkpoint_every,args.hours),(1800,400,5.8))
        self.assertEqual((args.save_reserve_seconds,args.validation_reserve_seconds,args.disk_reserve_gib),(450,240,32))
        request=t.StopRequest();request.handler(15,None)
        self.assertTrue(request.requested);self.assertEqual(request.signal,15)
        self.assertFalse(args.save_optimizer)
        self.assertIn('brown box',t.PROMPT);self.assertNotIn('apple',t.PROMPT)

    def test_update_has_exact_eight_microbatches_and_explicit_mobile_prompt(self):
        runtime=Runtime();data=Dataset();optimizer=t.MasterAdamW(runtime.model.parameters(),lr=1e-5)
        before=runtime.model.weight.detach().clone()
        result=t.update(runtime,data,optimizer,torch.ones(32),float('inf'))
        self.assertEqual(result['microbatches'],8);self.assertEqual(data.calls,8)
        self.assertEqual(runtime.prompts,[t.PROMPT]*8);self.assertFalse(torch.equal(before,runtime.model.weight))
        state=optimizer.state_dict();self.assertEqual(state['schema'],'fp32_master_adamw_v1')
        self.assertTrue(all(p.dtype==torch.float32 for p in state['master_parameters']))

    def test_partial_accumulation_discards_gradients_and_never_updates(self):
        runtime=Runtime();data=Dataset();optimizer=t.MasterAdamW(runtime.model.parameters(),lr=1e-5)
        before=runtime.model.weight.detach().clone()
        with patch.object(t.time,'monotonic',side_effect=[0,0,0,11]):
            result=t.update(runtime,data,optimizer,torch.ones(32),10)
        self.assertIsNone(result);self.assertEqual(data.calls,3);self.assertTrue(torch.equal(before,runtime.model.weight))
        self.assertTrue(all(p.grad is None for p in runtime.model.parameters()));self.assertEqual(len(optimizer.optimizer.state),0)

    def test_signal_during_microbatches_completes_one_optimizer_boundary(self):
        request=t.StopRequest();runtime=Runtime();dataset=Dataset();original=dataset.sample
        def sample(index):
            if dataset.calls==2:request.handler(15,None)
            return original(index)
        dataset.sample=sample;optimizer=t.MasterAdamW(runtime.model.parameters(),lr=1e-5)
        result=t.update(runtime,dataset,optimizer,torch.ones(32),float('inf'))
        self.assertTrue(request.requested);self.assertEqual(dataset.calls,8);self.assertEqual(result['microbatches'],8)

    def test_validation_complete_fixed_rng_and_partial_never_selectable(self):
        runtime=Runtime();data=Dataset();state=torch.get_rng_state().clone()
        a=t.validate(runtime,data,torch.ones(32),float('inf'));b=t.validate(runtime,data,torch.ones(32),float('inf'))
        self.assertEqual(a,b);self.assertEqual(a['chunks'],3);self.assertEqual(a['frames'],150)
        self.assertTrue(torch.equal(state,torch.get_rng_state()));self.assertEqual(runtime.prompts,[t.PROMPT]*6)
        self.assertTrue(runtime.model.training)
        with patch.object(t.time,'monotonic',side_effect=[0,11]):
            self.assertIsNone(t.validate(runtime,data,torch.ones(32),10))
        self.assertTrue(runtime.model.training)

    def test_disk_estimate_retains_all_models_final_master_and_reserve(self):
        import struct
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);header=json.dumps({'weight':{'dtype':'BF16','shape':[10,20],'data_offsets':[0,400]}}).encode()
            (p/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'x'*400)
            disk=t.disk_estimate(p,1800,400,32,save_optimizer=True)
            self.assertEqual(disk['periodic_checkpoint_count_upper_bound'],4)
            self.assertGreaterEqual(disk['final_optimizer_bytes_upper_bound'],200*12)
            self.assertEqual(disk['required_free_bytes'],disk['all_payload_bytes_upper_bound']+32*1024**3)
            with patch.object(t.shutil,'disk_usage',return_value=SimpleNamespace(free=disk['required_free_bytes']-1)):
                with self.assertRaisesRegex(RuntimeError,'Disk admission rejected'):
                    t.require_disk(p,disk['all_payload_bytes_upper_bound'],disk['reserve_bytes'])

    def test_default_disk_estimate_omits_unselected_optimizer_and_preserves_all_models(self):
        import struct
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);header=json.dumps({'weight':{'dtype':'BF16','shape':[10,20],'data_offsets':[0,400]}}).encode()
            (p/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'x'*400)
            default=t.disk_estimate(p,1600,400,32);optin=t.disk_estimate(p,1600,400,32,save_optimizer=True)
            self.assertEqual(default['final_optimizer_bytes_upper_bound'],0)
            self.assertEqual(default['periodic_checkpoint_count_upper_bound'],4)
            self.assertEqual(default['baseline_checkpoint_count'],1)
            self.assertEqual(default['required_free_bytes'],optin['required_free_bytes']-optin['final_optimizer_bytes_upper_bound'])
            self.assertEqual(default['reserve_bytes'],32*1024**3)
            args=t.parser().parse_args(['--output','UNUSED','--save-optimizer']);self.assertTrue(args.save_optimizer)

    def test_checkpoint_metadata_sha_and_final_fp32_optimizer_preserved_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'final_000001';runtime=Runtime();optimizer=t.MasterAdamW(runtime.model.parameters(),lr=1e-5)
            disk={'model_bytes':10000,'final_optimizer_bytes_upper_bound':100000,'reserve_bytes':0}
            metadata={'step':1,'identity':{'synthetic':True}}
            with patch.object(torch.cuda,'get_rng_state_all',return_value=[]):
                result=t.checkpoint(runtime,optimizer,path,metadata,disk,include_optimizer=True)
            self.assertTrue(result['optimizer_payload_saved'])
            self.assertEqual(json.loads((path/'metadata_sha256.json').read_text())['metadata_sha256'],t.sha(path/'metadata.json'))
            payload=torch.load(path/'training_state.pt',weights_only=False,map_location='cpu')
            self.assertEqual(payload['optimizer']['schema'],'fp32_master_adamw_v1')
            self.assertTrue(all(p.dtype==torch.float32 for p in payload['optimizer']['master_parameters']))
            with self.assertRaises(FileExistsError):t.checkpoint(runtime,optimizer,path,metadata,disk,include_optimizer=True)

    def test_synthetic_full_training_selects_complete_minimum_and_writes_real_stop(self):
        import sys,types
        for optin in [False,True]:
            with self.subTest(save_optimizer=optin),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);base=p/'official';base.mkdir();(base/'model.safetensors').write_bytes(b'synthetic official payload')
                args=t.parser().parse_args(['--output',str(p/'run'),'--checkpoint',str(base),'--max-steps','2'])
                args.save_optimizer=optin
                args.norm_stats='not-read-synthetic';admission={'schema':'synthetic','disk_estimate':{'model_bytes':10000,'final_optimizer_bytes_upper_bound':100000,'reserve_bytes':0},'manifest_sha256':'manifest','norm_sha256':'norm','official_base':{'sha256':t.sha(base/'model.safetensors')},'frozen_files':{}}
                runtime=Runtime();runtime.dtype_summary={'torch.float32':{'parameter_count':1024}}
                module=types.ModuleType('experiments.pi05_g1_mobile.model_runtime');module.ModelRuntime=lambda *a,**kw:runtime
                values=[{'flow_loss':1.,'complete':True,'chunks':3,'frames':150,'episodes':1},{'flow_loss':.8,'complete':True,'chunks':3,'frames':150,'episodes':1}]
                with patch.dict(sys.modules,{module.__name__:module}),patch.object(t,'preflight',return_value=admission),patch.object(t,'QuantileNormalizer',return_value=None),patch.object(t,'EpisodeDataset',side_effect=lambda *a,**kw:Dataset()),patch.object(t,'validate',side_effect=values),patch.object(t,'memory_snapshot',return_value={}),patch.object(torch.cuda,'get_rng_state_all',return_value=[]):
                    report=t.train(args)
                self.assertEqual(report['status'],'completed');self.assertEqual(report['step'],2)
                self.assertEqual(report['training_stopped']['reason'],'max_steps')
                self.assertTrue(report['selected_checkpoint'].endswith('final_000002'))
                self.assertEqual((p/'run/final_000002/training_state.pt').exists(),optin)
                self.assertEqual(report['optimizer_payload_saved'],optin);self.assertEqual(report['final_optimizer_includes_fp32_master'],optin)
                self.assertFalse(report['exact_optimizer_resume_supported'])
                metadata=json.loads((p/'run/final_000002/metadata.json').read_text());self.assertEqual(metadata['optimizer_payload_saved'],optin)
                self.assertEqual((p/'run/step_000000/model.safetensors').stat().st_ino,(base/'model.safetensors').stat().st_ino)
                events=[json.loads(line) for line in (p/'run/progress.jsonl').read_text().splitlines()]
                self.assertEqual(sum(row['event']=='optimizer_step' for row in events),2)
                self.assertFalse(report['test_loaded'])

    def test_synthetic_baseline_partial_records_failure_without_any_update(self):
        import sys,types
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);args=t.parser().parse_args(['--output',str(p/'run'),'--max-steps','2'])
            admission={'disk_estimate':{},'manifest_sha256':'manifest','norm_sha256':'norm','official_base':{'sha256':'base'},'frozen_files':{}}
            runtime=Runtime();runtime.dtype_summary={}
            module=types.ModuleType('experiments.pi05_g1_mobile.model_runtime');module.ModelRuntime=lambda *a,**kw:runtime
            with patch.dict(sys.modules,{module.__name__:module}),patch.object(t,'preflight',return_value=admission),patch.object(t,'QuantileNormalizer',return_value=None),patch.object(t,'EpisodeDataset',side_effect=lambda *a,**kw:Dataset()),patch.object(t,'validate',return_value=None),patch.object(t,'update') as update:
                with self.assertRaisesRegex(RuntimeError,'Baseline full validation incomplete'):t.train(args)
                update.assert_not_called()
            report=json.loads((p/'run/training_result.json').read_text())
            self.assertEqual(report['status'],'failed');self.assertIsNone(report['training_stopped']);self.assertEqual(report['step'],0)
            self.assertFalse((p/'run/best_checkpoint.json').exists())

    def test_reject_apple_warmstart_and_insufficient_reserve_before_model_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=t.parser().parse_args(['--output',str(Path(tmp)/'out'),'--checkpoint','/tmp/APPLE_MODEL'])
            with self.assertRaisesRegex(ValueError,'official'):t.preflight(args)
            args.checkpoint=str(t.OFFICIAL_BASE);args.save_reserve_seconds=449
            with self.assertRaisesRegex(ValueError,'450'):t.preflight(args)
            self.assertFalse(Path(args.output).exists())


if __name__=='__main__':unittest.main()
