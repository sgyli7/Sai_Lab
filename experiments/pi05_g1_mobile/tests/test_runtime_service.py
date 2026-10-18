"""Real tiny CPU safetensors and contract tests; never construct a CUDA model."""
import ast
import base64
import copy
import json
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from experiments.pi05_g1_mobile import contract,gr00t_service,model_runtime,pi05_service


class MobileRuntimeServiceTests(unittest.TestCase):
    def request(self):
        names=list(contract.DATASET_NAMES);q=np.arange(43,dtype=np.float32)/100
        rgb=np.zeros((480,640,3),np.uint8);rgb[:,:,0]=211
        return {'schema':'pi05_g1_observation_v1','episode_id':'cpu-fixture','control_tick':0,'rgb_shape':[480,640,3],'rgb_uint8_b64':base64.b64encode(rgb.tobytes()).decode(),'state':contract.encode_state(q,names).tolist(),'prompt':contract.PROMPT,'seed':42,'measured_joint_positions43':q.tolist(),'measured_joint_names43':names}

    def test_real_safetensors_preserves_stability_fp32_1001_without_postcast(self):
        import torch
        from safetensors.torch import save_model
        from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.input_layernorm=torch.nn.Linear(2,2,bias=False);self.projection=torch.nn.Linear(2,2,bias=False)
            def establish(self):PaliGemmaWithExpertModel.to_bfloat16_for_selected_params(self,'bfloat16')
        source=Tiny();source.establish()
        with torch.no_grad():source.input_layernorm.weight.fill_(1.001)
        expected=source.input_layernorm.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.safetensors';save_model(source,str(path));before=path.read_bytes()
            destination=Tiny();destination.establish()
            names=model_runtime.load_preserving_parameter_dtypes(destination,path,'cpu')
            self.assertIn('input_layernorm.weight',names);self.assertEqual(destination.input_layernorm.weight.dtype,torch.float32)
            self.assertTrue(torch.equal(destination.input_layernorm.weight,expected));self.assertEqual(destination.projection.weight.dtype,torch.bfloat16)
            self.assertEqual(path.read_bytes(),before)
            destination.establish()  # The removed historical post-load call.
            self.assertFalse(torch.equal(destination.input_layernorm.weight,expected))
            self.assertEqual(float(destination.input_layernorm.weight[0,0].detach()),1.)

    def test_dtype_assertion_and_strict_missing_parameters(self):
        import torch
        from safetensors.torch import save_file
        model=torch.nn.Linear(2,2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.safetensors';save_file({'weight':torch.ones((2,2))},str(path))
            with self.assertRaises(RuntimeError):model_runtime.load_preserving_parameter_dtypes(model,path,'cpu')
        with patch('safetensors.torch.load_model',side_effect=lambda m,*args,**kwargs:m.to(dtype=torch.float64)):
            with self.assertRaisesRegex(RuntimeError,'changed established'):model_runtime.load_preserving_parameter_dtypes(model,'unused','cpu')

    def test_mobile_observation_exact_keys_and_q_state_agreement_both_services(self):
        req=self.request();rgb,state=pi05_service.observation(req);native=gr00t_service.observation(req)
        self.assertEqual(rgb.shape,(480,640,3));self.assertEqual(rgb.dtype,np.uint8)
        np.testing.assert_array_equal(rgb,native['video']['ego_view'][0,0]);self.assertEqual(state.shape,(32,))
        np.testing.assert_array_equal(native['state']['left_arm'][0,0],state[:7])
        self.assertEqual(native['language']['annotation.human.task_description'],[[contract.PROMPT]])
        for field,value in [('object_pose',[0,0,1]),('route_waypoint',[1,2]),('contacts',True)]:
            bad={**req,field:value}
            for adapter in (pi05_service.observation,gr00t_service.observation):
                with self.assertRaisesRegex(ValueError,'extra truth'):adapter(bad)
        bad=copy.deepcopy(req);bad['state'][0]+=.1
        with self.assertRaisesRegex(ValueError,'exactly encode'):pi05_service.observation(bad)
        bad=copy.deepcopy(req);bad['state'][-1]=1
        with self.assertRaises(ValueError):pi05_service.observation(bad)
        for key,value in [('prompt','Pick up the apple.'),('rgb_shape',[640,480,3]),('rgb_uint8_b64','AA==')]:
            with self.assertRaises(ValueError):pi05_service.observation({**req,key:value})

    def test_named_joint_permutation_and_source_mapping(self):
        req=self.request();names=req['measured_joint_names43'];q=np.asarray(req['measured_joint_positions43'],np.float32)
        shuffled={**req,'measured_joint_names43':names[::-1],'measured_joint_positions43':q[::-1].tolist()}
        _,left=pi05_service.observation(req);_,right=pi05_service.observation(shuffled);np.testing.assert_array_equal(left,right)
        a=np.arange(32,dtype=np.float32)/10
        source=contract.decode_to_source(a,q[::-1],names[::-1])
        for index,name in enumerate(contract.UPPER_NAMES):self.assertEqual(source[names[::-1].index(name)],a[index])
        np.testing.assert_array_equal(source[43:46],a[29:32]);self.assertEqual(source[46],a[28]);np.testing.assert_array_equal(source[47:],0)

    def test_native35_projects_effective32_with_waist_evidence(self):
        widths={**{key:len(value) for key,value in contract.GROUPS.items()},'waist':3,'base_height_command':1,'navigate_command':3}
        outputs={key:np.full((1,50,width),i+1,dtype=np.float32) for i,(key,width) in enumerate(widths.items())}
        actions,waist=contract.native_outputs_to_actions(outputs)
        self.assertEqual(actions.shape,(50,32));self.assertEqual(waist.shape,(50,3))
        np.testing.assert_array_equal(actions[:,:7],outputs['left_arm'][0]);np.testing.assert_array_equal(actions[:,28:29],outputs['base_height_command'][0]);np.testing.assert_array_equal(actions[:,29:32],outputs['navigate_command'][0]);np.testing.assert_array_equal(waist,outputs['waist'][0])
        changed={**outputs,'waist':outputs['waist']+1}
        np.testing.assert_array_equal(contract.native_outputs_to_actions(changed)[0],actions)
        with self.assertRaises(ValueError):contract.native_outputs_to_actions({**outputs,'object_truth':np.zeros((1,50,3))})

    def test_real_cpu_observation_layout_masks_and_mobile_language(self):
        import torch
        req=self.request();rgb,state=pi05_service.observation(req)
        runtime=model_runtime.ModelRuntime.__new__(model_runtime.ModelRuntime);runtime.torch=torch;runtime.device=torch.device('cpu');runtime.head_only=False
        captured={}
        def tokenize(prompt,normalized):captured['prompt']=prompt;return np.array([1,2]),np.array([True,True])
        runtime.tokenizer=SimpleNamespace(tokenize=tokenize)
        runtime.norm=SimpleNamespace(normalize=lambda value,key:value)
        obs=runtime.observation(rgb,state)
        self.assertEqual(captured['prompt'],contract.PROMPT);self.assertEqual(tuple(obs.images['base_0_rgb'].shape),(1,3,224,224))
        self.assertTrue(bool(obs.image_masks['base_0_rgb'][0]));self.assertFalse(bool(obs.image_masks['left_wrist_0_rgb'][0]));self.assertFalse(bool(obs.image_masks['right_wrist_0_rgb'][0]))
        self.assertLessEqual(float(obs.images['base_0_rgb'].max()),1.);self.assertGreaterEqual(float(obs.images['base_0_rgb'].min()),-1.)

    def test_unrelated_runtime_paths_are_ast_identical_to_static_source(self):
        root=Path(__file__).resolve().parents[3]
        old=ast.parse((root/'experiments/pi05_g1/model_runtime.py').read_text());new=ast.parse((root/'experiments/pi05_g1_mobile/model_runtime.py').read_text())
        def nodes(tree):
            result={}
            for node in tree.body:
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                    result[node.name]=node
                    if isinstance(node,ast.ClassDef):result.update({node.name+'.'+part.name:part for part in node.body if isinstance(part,ast.FunctionDef)})
            return result
        a,b=nodes(old),nodes(new)
        for name in ('resize_pad_rgb','preprocess_g1_observation','exclude_unused_expert_logits_head','ModelRuntime.observation','ModelRuntime.infer','ModelRuntime.set_head_only'):
            self.assertEqual(ast.dump(a[name]),ast.dump(b[name]))

    def test_service_jsonl_metadata_timing_and_rejection_without_cuda(self):
        req=self.request();bad={**req,'object_truth':[1,2,3]};calls=[]
        def infer(rgb,state,prompt,seed):calls.append((prompt,seed));return {'actions':np.zeros((50,32),np.float32),'inference_seconds':.25}
        fake=SimpleNamespace(metadata={'precision_rule':'mobile_preserve_preload_stable_fp32_v1'},infer=infer)
        with tempfile.TemporaryDirectory() as directory:
            evidence=Path(directory)/'decoded.jsonl';stdout=io.StringIO()
            argv=['pi05_service','--checkpoint','synthetic','--norm-stats','synthetic','--decoded-evidence',str(evidence)]
            with patch.object(pi05_service.sys,'argv',argv),patch.object(pi05_service.sys,'stdin',io.StringIO(json.dumps(req)+'\n'+json.dumps(bad)+'\n')),patch.object(pi05_service.sys,'stdout',stdout),patch.object(model_runtime,'ModelRuntime',return_value=fake) as constructor:
                pi05_service.main()
            self.assertFalse(constructor.call_args.kwargs['head_only']);self.assertEqual(calls,[(contract.PROMPT,42)])
            replies=[json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual(replies[0]['action_horizon'],50);self.assertEqual(replies[0]['inference_seconds'],.25);self.assertIn('error',replies[1])
            rows=[json.loads(line) for line in evidence.read_text().splitlines()]
            self.assertEqual([row['event'] for row in rows],['model_loaded','pi05_decoded','request_rejected'])


if __name__=='__main__':unittest.main()
