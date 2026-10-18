import ast
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.pi05_g1_mobile import simulator as subject


class InterruptionEvidenceTests(unittest.TestCase):
    def test_deadline_absolute_boundary_and_default(self):
        result={'cold_process_started_monotonic_ns':1_000_000_000}
        subject.check_wall_deadline({},result,10**18)
        subject.check_wall_deadline({'max_run_wall_seconds':None},result,10**18)
        subject.check_wall_deadline({'max_run_wall_seconds':2},result,2_999_999_999)
        with self.assertRaisesRegex(subject.InfraWallDeadlineError,'infra_wall_deadline'):
            subject.check_wall_deadline({'max_run_wall_seconds':2},result,3_000_000_000)
        base={'mode':'diagnostic','ticks':0,'case':{'case_id':'x','seed':0}}
        for bad in (True,0,-1,float('nan'),float('inf'),'840'):
            with self.assertRaises(ValueError):subject.validate_config({**base,'max_run_wall_seconds':bad})
        self.assertEqual(subject.validate_config({**base,'max_run_wall_seconds':840})['max_run_wall_seconds'],840)

    def test_video_close_before_model_even_on_error(self):
        order=[]
        def item(name,fail=False):
            def close():
                order.append(name)
                if fail:raise RuntimeError('video close error')
            return SimpleNamespace(close=close)
        errors=subject.close_recordings_before_model(item('head',True),item('observer'),item('model'))
        self.assertEqual(order,['head','observer','model']);self.assertEqual(len(errors),1)

    def test_real_finish_body_records_infra_failed_never_task_failure(self):
        # Execute actual nested finish in isolation, without importing Arena/Kit.
        tree=ast.parse(Path(subject.__file__).read_text())
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        finish=copy.deepcopy(next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name=='finish'))
        finish.body=[ast.copy_location(ast.Global(names=n.names),n) if isinstance(n,ast.Nonlocal) else n for n in finish.body]
        module=ast.fix_missing_locations(ast.Module(body=[finish],type_ignores=[]))
        with tempfile.TemporaryDirectory() as tmp:
            result={'status':'running','success':None};order=[]
            ns={'result':result,'model':SimpleNamespace(close=lambda:order.append('model')),
                'video':SimpleNamespace(close=lambda:order.append('head')),
                'observer_video':SimpleNamespace(close=lambda:order.append('observer')),
                'output':Path(tmp),'time':time,'json':json,'digest':subject.digest,
                'InfraWallDeadlineError':subject.InfraWallDeadlineError,
                'close_recordings_before_model':subject.close_recordings_before_model}
            exec(compile(module,'actual_finish_test','exec'),ns)
            ns['finish'](subject.InfraWallDeadlineError('infra_wall_deadline'))
            recorded=json.loads((Path(tmp)/'result.json').read_text())
            self.assertEqual(recorded['status'],'failed');self.assertIsNone(recorded['success'])
            self.assertEqual(recorded['termination_reason'],'infra_wall_deadline')
            self.assertTrue(recorded['infrastructure_failure']);self.assertEqual(order,['head','observer','model'])

    def test_runtime_identity_static_and_rpc_unmodified(self):
        result={'source_joint_names':['q'],'body_names':['hand'],'contact_view':{'filter_paths':['box','hand']},
            'wbc_config':{'enable_waist':False},'frozen_interface_sha256':'x'}
        identity={'physics':{'dt':.005},'head_camera':{'width':640,'height':480}}
        payload=subject.runtime_identity_payload({'frozen_files':{'file':'sha'}},result,identity)
        self.assertEqual(payload['source_joint_names'],['q'])
        self.assertEqual(payload['wbc_config'],{'enable_waist':False})
        self.assertEqual(payload['video_encoding']['output_params'],subject.VIDEO_OUTPUT_PARAMS)
        self.assertNotIn('acceptance_truth_only',payload)
        source=Path(subject.__file__).read_text()
        self.assertLess(source.index("(output/'runtime_identity.json').write_text"),source.index("for tick in range(config['ticks']+1)"))
        self.assertIn("model.infer(rgb,state,case,tick,config['prompt'],q,names)",source)
        self.assertNotIn('runtime_identity',source[source.index('model.infer('):source.index('model.infer(')+100])
        self.assertIn('default_base_moof',subject.VIDEO_OUTPUT_PARAMS[1])
        self.assertEqual(subject.VIDEO_OUTPUT_PARAMS[-2:],['-g','50'])


if __name__=='__main__':unittest.main()
