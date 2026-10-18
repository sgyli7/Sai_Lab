"""Tiny synthetic weights only: interrupted-run recovery, counters and host trim."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from experiments.pi05_g1_round02.checkpoint_lineage import (CHECKPOINT_SCHEMA,LINEAGE_SCHEMA,ROSTER_SCHEMA,
    collect_candidates,declared_training_dirs,resolve_lineage,select_global_best,validate_failed_parent,sha)
from experiments.pi05_g1_round02.training import release_cpu_staging,save_baseline_checkpoint


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class CheckpointLineageTests(unittest.TestCase):
    def fixture(self,root):
        genesis=root/'round01/step_001200';write(genesis/'metadata.json',{'step':1200,'validation_flow_loss':.1});(genesis/'model.safetensors').write_bytes(b'original tiny weights')
        norm=root/'norm.json';loss=root/'loss.json';write(norm,{'synthetic':'no real observations'});write(loss,{'actions':[1.]*32})
        declarations=[];sources=[]
        for name,kind,episodes,chunks in [('original','lerobot',21,80),('fresh_teacher','npz_video',3,24)]:
            manifest=root/(name+'.json');write(manifest,{'synthetic':name})
            declaration={'name':name,'kind':kind,'manifest':str(manifest),'train_sampling_weight':.5,'validation_weight':.5};declarations.append(declaration)
            sources.append({**declaration,'manifest_sha256':sha(manifest),'train_episodes':7,'validation_episodes':episodes,'validation_chunks':chunks,'time_sampling':{'kind':'uniform'}})
        config=root/'data.json';write(config,{'sources':declarations});identity={'data_config_sha256':sha(config),'norm_sha256':sha(norm),'loss_weights_sha256':sha(loss),'sources':sources}
        old=root/'run01';new=root/'run02';new.mkdir()
        for step,value in [(0,.8),(200,.4)]:
            folder=old/f'step_{step:06d}';folder.mkdir(parents=True)
            (folder/'model.safetensors').write_bytes(b'original tiny weights' if step==0 else b'updated tiny weights')
            metrics={s['name']:{'flow_loss':value,'frames':50,'episodes':s['validation_episodes'],'chunks':s['validation_chunks'],'validation_weight':.5} for s in sources}
            write(folder/'metadata.json',{'schema':CHECKPOINT_SCHEMA,'step':step,'warmstart':{'checkpoint':str(genesis),'weights_sha256':sha(genesis/'model.safetensors'),'source_optimizer_steps':1200,'optimizer_reset':True,'source_optimizer_payload_loaded':False},'round02_data_identity':identity,'loss_weights':[1.]*32,'validation_complete':True,'validation_flow_loss':value,'validation_sources':metrics})
        (old/'progress.jsonl').write_text(''.join(json.dumps({'event':'optimizer_step','step':i})+'\n' for i in range(1,344)))
        budget=root/'budget.json';write(budget,{'runs':[{'stage':'training','command':['python','--output',str(old)],'status':'host_memory_guard','exit_code':137,'elapsed_seconds':3303.595,'last_unit_properties':{'ActiveState':'inactive'}}]})
        return SimpleNamespace(root=root,genesis=genesis,old=old,new=new,parent=old/'step_000200',budget=budget,identity=identity,config=config,norm=norm,loss=loss)

    def recovery(self,f,local=0,value=.4):
        failure,parent=validate_failed_parent(f.parent,f.identity,f.budget,1656,f.genesis)
        receipt=f.new/'parent_failure_receipt.json'
        if not receipt.exists():write(receipt,failure)
        lineage={'schema':LINEAGE_SCHEMA,'immediate_parent':{'path':str(f.parent),'metadata_sha256':sha(f.parent/'metadata.json'),'weights_sha256':sha(f.parent/'model.safetensors')},'genesis':parent['genesis'],'inherited_round_updates':200,'local_step':local,'cumulative_round_updates':200+local,'optimizer_reset':True,'source_failure_receipt':{'path':str(receipt),'sha256':sha(receipt)}}
        meta=copy.deepcopy(json.loads((f.parent/'metadata.json').read_text()))
        meta.update(step=local,local_step=local,inherited_round_updates=200,cumulative_round_updates=200+local,checkpoint_lineage=lineage,validation_flow_loss=value)
        for metric in meta['validation_sources'].values():metric['flow_loss']=value
        meta['warmstart']={'checkpoint':str(f.parent),'weights_sha256':sha(f.parent/'model.safetensors'),'source_optimizer_steps':200,'optimizer_reset':True,'source_optimizer_payload_loaded':False}
        path=f.new/f'step_{local:06d}';write(path/'metadata.json',meta);(path/'model.safetensors').write_bytes(b'updated tiny weights' if local==0 else b'next tiny weights')
        return path

    def test_legacy_and_recovery_counters_and_chain_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));path=self.recovery(f)
            result=resolve_lineage(path,verify_weights=True)
            self.assertEqual(result['local_step'],0);self.assertEqual(result['inherited_round_updates'],200);self.assertEqual(result['cumulative_round_updates'],200)
            self.assertEqual([x['path'] for x in result['ancestor_chain']],[str(f.genesis),str(f.parent),str(path)])
            self.assertTrue(all(x['weights_sha256'] and x['metadata_sha256'] for x in result['ancestor_chain']))
            self.assertEqual(resolve_lineage(f.parent)['cumulative_round_updates'],200)

    def test_all_runs_candidate_minimum_and_direct_child_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));self.recovery(f);new=self.recovery(f,200,.3)
            result=collect_candidates([f.old,f.new],f.config,f.norm,f.loss)
            self.assertEqual(len(result['candidates']),4);self.assertEqual(select_global_best(result)['path'],str(new))
            self.assertEqual(select_global_best(result)['cumulative_round_updates'],400)
            with self.assertRaisesRegex(ValueError,'global'):select_global_best(result,{'path':str(f.parent),'validation_flow_loss':.4},f.new)
            with self.assertRaisesRegex(ValueError,'global'):select_global_best(result,{'path':str(f.root/'outsider'),'validation_flow_loss':.3},f.new)
            self.assertIn(str(f.genesis/'metadata.json'),result['metadata_hashes']);self.assertIn(str(f.new/'parent_failure_receipt.json'),result['metadata_hashes'])
            roster=f.new/'candidate_training_dirs.json';write(roster,{'schema':ROSTER_SCHEMA,'training_dirs':[str(f.old),str(f.new)],'current_training_dir':str(f.new)})
            self.assertEqual(declared_training_dirs(roster,f.new),[f.old,f.new])

    def test_no_big_payload_reads_when_verify_weights_false(self):
        import experiments.pi05_g1_round02.checkpoint_lineage as shared
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));self.recovery(f);real=shared.sha
            def small_only(path):
                if Path(path).name=='model.safetensors':raise AssertionError('Metadata-only selection must not hash weights')
                return real(path)
            with patch.object(shared,'sha',side_effect=small_only):result=collect_candidates([f.old,f.new],f.config,f.norm,f.loss)
            self.assertEqual(len(result['candidates']),3)

    def test_wrong_identity_counters_or_parent_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));path=self.recovery(f);p=path/'metadata.json';good=json.loads(p.read_text());bad=copy.deepcopy(good);bad['cumulative_round_updates']=0;write(p,bad)
            with self.assertRaisesRegex(ValueError,'accounting'):resolve_lineage(path)
            write(p,good);parentmeta=f.parent/'metadata.json';old=json.loads(parentmeta.read_text());write(parentmeta,{**old,'changed':True})
            with self.assertRaisesRegex(ValueError,'metadata SHA'):resolve_lineage(path)

    def test_actual_parent_weight_tamper_and_failed_budget_cap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));self.recovery(f)
            with self.assertRaisesRegex(ValueError,'cap1656'):validate_failed_parent(f.parent,f.identity,f.budget,1657,f.genesis)
            (f.genesis/'model.safetensors').write_bytes(b'tampered source')
            with self.assertRaisesRegex(ValueError,'weights SHA'):validate_failed_parent(f.parent,f.identity,f.budget,1656,f.genesis)
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));budget=json.loads(f.budget.read_text());budget['runs'][0]['status']='running';write(f.budget,budget)
            with self.assertRaisesRegex(ValueError,'closed host'):validate_failed_parent(f.parent,f.identity,f.budget,1656,f.genesis)

    def test_recovery_baseline_hardlink_reports_inherited200_not_roundzero(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));parentmeta=json.loads((f.parent/'metadata.json').read_text())
            metadata={**parentmeta,'step':0,'local_step':0,'inherited_round_updates':200,'cumulative_round_updates':200,'warmstart':{'weights_sha256':sha(f.parent/'model.safetensors')}}
            target=f.new/'baseline';saved=save_baseline_checkpoint(f.parent,target,metadata)
            self.assertEqual(saved['round_optimizer_updates'],200);self.assertEqual(saved['local_optimizer_updates'],0)
            self.assertEqual((target/'model.safetensors').stat().st_ino,(f.parent/'model.safetensors').stat().st_ino)
            self.assertFalse((target/'training_state.pt').exists())

    def test_trim_supported_and_failure_only_diagnostic(self):
        trim=Mock(return_value=1)
        with patch('experiments.pi05_g1_round02.training.platform.system',return_value='Linux'),patch('experiments.pi05_g1_round02.training.platform.libc_ver',return_value=('glibc','2.39')),patch('experiments.pi05_g1_round02.training.gc.collect',return_value=3),patch('experiments.pi05_g1_round02.training.ctypes.CDLL',return_value=SimpleNamespace(malloc_trim=trim)):
            receipt=release_cpu_staging()
        self.assertTrue(receipt['trim_available']);self.assertEqual(receipt['trim_return'],1);self.assertEqual(receipt['gc_collected'],3);trim.assert_called_once_with(0)
        with patch('experiments.pi05_g1_round02.training.platform.system',return_value='Linux'),patch('experiments.pi05_g1_round02.training.platform.libc_ver',return_value=('glibc','2.39')),patch('experiments.pi05_g1_round02.training.ctypes.CDLL',side_effect=OSError('mock missing libc symbol')):
            receipt=release_cpu_staging()
        self.assertIn('mock missing',receipt['diagnostic']);self.assertIn('after',receipt)

    def test_real_cpu_trim_preserves_live_parameter_gradient_and_dtype(self):
        import torch
        parameter=torch.nn.Parameter(torch.tensor([1.001,-2.],dtype=torch.float32));parameter.grad=torch.tensor([.1,.2])
        before=parameter.detach().clone();grad=parameter.grad.clone();pointer=parameter.data_ptr()
        receipt=release_cpu_staging()
        self.assertEqual(parameter.data_ptr(),pointer);self.assertEqual(parameter.dtype,torch.float32)
        self.assertTrue(torch.equal(parameter,before));self.assertTrue(torch.equal(parameter.grad,grad))
        self.assertIn('after',receipt)


if __name__=='__main__':unittest.main()
