"""CPU tests for round02 data alignment, weighting and checkpoint storage policy."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.pi05_g1_round02.datasets import TeacherEpisodes, sha256
from experiments.pi05_g1_round02.training import disk_estimate, require_disk, weighted_flow_loss, save_checkpoint, save_baseline_checkpoint, validation_improves


class Round02TrainingTests(unittest.TestCase):
    def teacher(self,root):
        records=[]
        for name in ('train_case','validation_case'):
            npz=root/(name+'.npz');video=root/(name+'.mp4')
            actions=np.arange(3*32,dtype=np.float32).reshape(3,32)
            np.savez(npz,state=np.zeros((3,32),np.float32),actions=actions,timestamps=np.arange(3)/50)
            video.write_bytes(b'CPU test: decoder is mocked; no fabricated simulator evidence')
            records.append({'episode_id':name,'npz':npz.name,'video':video.name,'npz_sha256':sha256(npz),'video_sha256':sha256(video),'length':3,'fps':50})
        path=root/'teacher.json';path.write_text(json.dumps({'schema':'pi05_g1_teacher_episodes_v1','fresh_source_enabled':True,'train':['train_case'],'validation':['validation_case'],'episodes':records}))
        return path

    def test_teacher_chunks_keep_raw_contract_and_tail_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.teacher(Path(directory))
            normalizer=SimpleNamespace(normalize=lambda values,key:values)
            data=TeacherEpisodes(path,'train',normalizer)
            with patch('experiments.pi05_g1_round02.datasets.decode_video_frame',return_value=np.zeros((480,640,3),np.uint8)):
                _,_,actions,mask=data.sample(1)
            self.assertEqual(actions.shape,(50,32));self.assertEqual(int(mask.sum()),2)
            np.testing.assert_array_equal(actions[1:],np.repeat(np.arange(64,96,dtype=np.float32)[None],49,axis=0))
            self.assertEqual(len(data.episodes),1)
            altered=json.loads(path.read_text());altered['validation']=['train_case'];path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,'overlap'):TeacherEpisodes(path,'train',normalizer)

    def test_explicit_dimension_weights_and_padding(self):
        import torch
        losses=torch.ones((1,50,32));losses[:,2:]=1000.;losses[:,:,31]=1000.
        mask=torch.arange(50)<2;weights=torch.ones(32);weights[31]=0
        self.assertEqual(float(weighted_flow_loss(losses,mask,weights)),1.)

    def test_disk_model_plus_periodic_and_final_optimizer_estimate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);header=json.dumps({'weight':{'shape':[2,3],'dtype':'F32','data_offsets':[0,24]}}).encode()
            import struct
            (root/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+bytes(24))
            estimate=disk_estimate(root,600,200,32)
            self.assertEqual(estimate['periodic_checkpoint_count_upper_bound'],3)
            self.assertEqual(estimate['unique_parameter_elements'],6)
            self.assertEqual(estimate['baseline_checkpoint_count'],1)
            self.assertGreater(estimate['required_free_bytes'],32*1024**3)
            with patch('experiments.pi05_g1_round02.training.shutil.disk_usage',return_value=SimpleNamespace(free=1)):
                with self.assertRaisesRegex(RuntimeError,'Disk admission rejected'):require_disk(root,2,0)

    def test_step0_exact_hardlink_no_optimizer_and_strict_selection(self):
        import torch
        from safetensors.torch import save_file, load_file
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'frozen1200';source.mkdir()
            save_file({'stable':torch.tensor([1.001],dtype=torch.float32),'bf16':torch.tensor([2.],dtype=torch.bfloat16)},source/'model.safetensors')
            digest=sha256(source/'model.safetensors');inode=(source/'model.safetensors').stat().st_ino
            metadata={'step':0,'validation_complete':True,'validation_flow_loss':.5,'warmstart':{'weights_sha256':digest,'source_optimizer_steps':1200},'round02_data_identity':{'norm_sha256':'newnorm','sources':['original21','fresh']}}
            with patch('safetensors.torch.save_model',side_effect=AssertionError('must not stage')),patch.object(torch,'save',side_effect=AssertionError('must not save optimizer')):
                saved=save_baseline_checkpoint(source,root/'step_000000',metadata)
            self.assertEqual((root/'step_000000/model.safetensors').stat().st_ino,inode)
            self.assertEqual(sha256(source/'model.safetensors'),digest)
            self.assertFalse(saved['optimizer_payload_saved']);self.assertEqual(saved['round_optimizer_updates'],0)
            self.assertEqual(load_file(root/'step_000000/model.safetensors')['stable'].dtype,torch.float32)
            self.assertFalse((root/'step_000000/training_state.pt').exists())
            self.assertFalse(validation_improves(None,.5));self.assertFalse(validation_improves({'combined_flow_loss':.5},.5))
            self.assertFalse(validation_improves({'combined_flow_loss':.6},.5));self.assertTrue(validation_improves({'combined_flow_loss':.4},.5))

    def test_teacher_prefers_exact_lossless_png_and_rejects_changed_pixels(self):
        from PIL import Image
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=self.teacher(root);manifest=json.loads(path.read_text())
            pixels=np.arange(4*6*3,dtype=np.uint8).reshape(4,6,3)
            png=root/'raw.png';Image.fromarray(pixels).save(png)
            for record in manifest['episodes']:
                record['raw_rgb_frames']=[{'path':str(png),'byte_sha256':sha256(png),'rgb_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),'control_tick':i,'timestamp_seconds':i/50,'rgb_shape':list(pixels.shape)} for i in range(3)]
            path.write_text(json.dumps(manifest));data=TeacherEpisodes(path,'train',SimpleNamespace(normalize=lambda values,key:values))
            with patch('experiments.pi05_g1_round02.datasets.decode_video_frame',side_effect=AssertionError('PNG must take precedence')):
                rgb,_,_,_=data.sample(1)
            np.testing.assert_array_equal(rgb,pixels)
            Image.fromarray(np.zeros_like(pixels)).save(png)
            with self.assertRaisesRegex(ValueError,'byte hash'):data.sample(1)

    def test_prefix_sampling_seed_bounds_missing_candidate_and_full_validation(self):
        import random
        with tempfile.TemporaryDirectory() as directory:
            path=self.teacher(Path(directory));normalizer=SimpleNamespace(normalize=lambda values,key:values)
            data=TeacherEpisodes(path,'train',normalizer)
            policy={'kind':'prefix_mixture','prefix_probability':.8,'post_candidate_ticks':25,'episode_choice':'uniform'}
            with self.assertRaisesRegex(ValueError,'source_success_candidate_tick'):data.configure_time_sampling(policy)
            data.episodes[0]['source_success_candidate_tick']=0
            before=list(data.indices);validation_before=data.validation_indices()
            data.configure_time_sampling(policy)
            self.assertEqual(data.time_sampling['prefix_end_tick_inclusive']['train_case'],2)
            rng1,rng2=random.Random(42),random.Random(42)
            self.assertEqual([data.draw_index(rng1) for _ in range(100)],[data.draw_index(rng2) for _ in range(100)])
            self.assertEqual(data.indices,before);self.assertEqual(data.validation_indices(),validation_before)
            data.episodes[0]['source_success_candidate_tick']=301
            with self.assertRaisesRegex(ValueError,'source_success_candidate_tick'):data.configure_time_sampling(policy)
            with self.assertRaisesRegex(ValueError,'episode_choice'):data.configure_time_sampling({key:value for key,value in policy.items() if key!='episode_choice'})

    def test_prefix_draw_episode_first_and_inclusive_boundary(self):
        # Different lengths ensure episode choice is independent of frame count.
        data=TeacherEpisodes.__new__(TeacherEpisodes)
        data.episodes=[{'id':'short','actions':np.zeros((30,32)),'source_success_candidate_tick':0}, {'id':'long','actions':np.zeros((400,32)),'source_success_candidate_tick':100}]
        data.indices=[(episode,frame) for episode,length in enumerate((30,400)) for frame in range(length)]
        data.configure_time_sampling({'kind':'prefix_mixture','prefix_probability':.8,'post_candidate_ticks':25,'episode_choice':'uniform'})
        class LastFrame:
            def __init__(self,branch):self.calls=0;self.branch=branch
            def randrange(self,n):self.calls+=1;return 1 if self.calls==1 else n-1
            def random(self):return self.branch
        self.assertEqual(data.draw_index(LastFrame(.1)),(30+125,'prefix'))
        self.assertEqual(data.draw_index(LastFrame(.9)),(30+399,'full'))

    def test_periodic_model_only_and_exactly_one_final_optimizer_no_pruning(self):
        import torch
        from experiments.pi05_g1.training import MasterAdamW
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=torch.nn.Linear(2,2);optimizer=MasterAdamW(model.parameters(),lr=1e-5)
            runtime=SimpleNamespace(torch=torch,model=model)
            disk={'model_bytes':1024,'final_optimizer_bytes_upper_bound':8192}
            metadata={'step':1,'round02_data_identity':{'test':'CPU storage behavior only'}}
            with patch.object(torch.cuda,'empty_cache'),patch.object(torch.cuda,'get_rng_state_all',return_value=[]):
                save_checkpoint(runtime,optimizer,root/'step_000200',metadata,include_optimizer=False,disk=disk,reserve=0)
                original=sha256(root/'step_000200/model.safetensors')
                save_checkpoint(runtime,optimizer,root/'final_000200',metadata,include_optimizer=True,disk=disk,reserve=0)
            self.assertFalse((root/'step_000200/training_state.pt').exists())
            self.assertTrue((root/'final_000200/training_state.pt').is_file())
            self.assertEqual(original,sha256(root/'step_000200/model.safetensors'))
            self.assertFalse(json.loads((root/'step_000200/metadata.json').read_text())['optimizer_payload_saved'])
            self.assertTrue(json.loads((root/'final_000200/metadata.json').read_text())['optimizer_payload_saved'])


if __name__=='__main__':unittest.main()
