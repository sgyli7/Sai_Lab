"""Synthetic IPC tests: no real service, CUDA, simulator, or robot actions."""
import ast,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from PIL import Image
from experiments.pi05_g1_mobile import standalone_policy_probe as p
from experiments.pi05_g1_mobile.contract import DATASET_NAMES,PROMPT,encode_state

class FakePolicy:
    shape=(50,32);code=0;calls=0;failure=False
    def __init__(self,command,stderr_path,timeout):
        Path(stderr_path).write_text('synthetic load memory evidence\n');self.process=Mock();self.process.poll.return_value=self.code
    def infer(self,*args):
        type(self).calls+=1
        if self.failure:raise RuntimeError('synthetic service load failure')
        return np.zeros(self.shape,np.float32),{'inference_seconds':.1}
    def close(self):pass

class ProbeTests(unittest.TestCase):
    def fixture(self,path):
        obs=path/'obs';obs.mkdir();rgb=np.zeros((480,640,3),np.uint8);Image.fromarray(rgb).save(obs/'head_00000.png');q=np.zeros(43,np.float32)
        data={'control_tick':0,'prompt':PROMPT,'state':encode_state(q,DATASET_NAMES).tolist(),'measured_joint_positions43':q.tolist(),'measured_joint_names43':list(DATASET_NAMES),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}
        (obs/'observation_00000.json').write_text(json.dumps(data));return obs
    def setUp(self):FakePolicy.calls=0;FakePolicy.shape=(50,32);FakePolicy.code=0;FakePolicy.failure=False
    def test_one_finite_reply_clean_close_completes_and_saves_all_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report=p.run(root/'out',self.fixture(root),process_class=FakePolicy)
            self.assertEqual(report['status'],'completed');self.assertEqual(FakePolicy.calls,1)
            self.assertEqual(np.load(root/'out/actions.npy').shape,(50,32));self.assertFalse(report['actions_executed'])
            with self.assertRaises(FileExistsError):p.run(root/'out',root/'obs',process_class=FakePolicy)
    def test_short_reply_and_unclean_close_rejected(self):
        for shape,code in [((1,32),0),((50,32),-15)]:
            with self.subTest(shape=shape,code=code),tempfile.TemporaryDirectory() as tmp:
                FakePolicy.shape=shape;FakePolicy.code=code;root=Path(tmp)
                report=p.run(root/'out',self.fixture(root),process_class=FakePolicy)
                self.assertEqual(report['status'],'failed');self.assertIn('error',report)
    def test_failure_keeps_error_and_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);FakePolicy.failure=True
            report=p.run(root/'out',self.fixture(root),process_class=FakePolicy)
            self.assertEqual(report['status'],'failed');self.assertIn('synthetic service load failure',report['error']);self.assertIn('policy_stderr_sha256',report)
    def test_extra_truth_rejected_before_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);obs=self.fixture(root);f=obs/'observation_00000.json';data=json.loads(f.read_text());data['object_pose']=[0,0,0];f.write_text(json.dumps(data))
            report=p.run(root/'out',obs,process_class=FakePolicy)
            self.assertEqual(report['status'],'failed');self.assertEqual(FakePolicy.calls,0)
    def test_existing_ipc_class_copied_without_simulator_import(self):
        source=Path('experiments/pi05_g1_round02/simulator.py').read_text();new=Path(p.__file__).read_text()
        get=lambda s:ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=='PolicyProcess'))
        self.assertEqual(get(source),get(new))
        imports=[n.module or '' for n in ast.walk(ast.parse(new)) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('simulator' in x for x in imports))

if __name__=='__main__':unittest.main()
