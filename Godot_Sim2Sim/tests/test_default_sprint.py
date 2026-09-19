import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sim2sim.default_sprint import apply_default_sprint


class TestDefaultSprint(unittest.TestCase):
    def fixture(self, root):
        project=root/'project';assets=project/'runtime_assets';profile=root/'profile'
        assets.mkdir(parents=True);profile.mkdir()
        (project/'robot.json').write_bytes(b'accepted robot')
        old=dict(physics_hz=200,decimation=4,robots={'walk':{'spec':'res://robot.json'}},
                 policies={'walking':{'sha256':'old'},'roller':{'sha256':'keep'}},
                 control_config={'walk':{'old':True},'roller':{'brake':'keep'}})
        (assets/'deployment.json').write_text(json.dumps(old))
        (assets/'self_test.json').write_text(json.dumps({'cases':[{'skill':'walking','sha256':'old'},{'skill':'roller','sha256':'keep'}]}))
        policies={}
        for skill in ['walking','sprint']:
            name=skill+'.onnx';data=skill.encode();sha=hashlib.sha256(data).hexdigest()
            (profile/name).write_bytes(data)
            (profile/(skill+'.manifest.json')).write_text('{}')
            policies[skill]={'path':'res://runtime_assets/policies/'+name,'sha256':sha,'manifest':{}}
        (profile/'profile.json').write_text(json.dumps(dict(id='test',policies=policies,
            robot_spec_sha256=hashlib.sha256(b'accepted robot').hexdigest(),walk_control={'sprint_yaw_reversal_s':.2})))
        (profile/'self_test.json').write_text(json.dumps({'cases':[dict(skill=k,sha256=v['sha256']) for k,v in policies.items()]}))
        return project,profile,old

    def test_only_walking_pair_changes_and_reapplying_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            project,profile,old=self.fixture(Path(directory))
            apply_default_sprint(project,profile_directory=profile)
            path=project/'runtime_assets/deployment.json';before=path.read_bytes();new=json.loads(before)
            self.assertEqual(new['policies']['roller'],old['policies']['roller'])
            self.assertEqual(new['control_config']['roller'],old['control_config']['roller'])
            self.assertEqual(set(new['policies']),{'walking','sprint','roller'})
            apply_default_sprint(project,profile_directory=profile)
            self.assertEqual(before,path.read_bytes())

    def test_invalid_model_rejected_before_deployment_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            project,profile,_=self.fixture(Path(directory))
            path=project/'runtime_assets/deployment.json';before=path.read_bytes()
            (profile/'sprint.onnx').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                apply_default_sprint(project,profile_directory=profile)
            self.assertEqual(before,path.read_bytes())
            self.assertFalse((project/'runtime_assets/policies').exists())

    def test_different_robot_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            project,profile,_=self.fixture(Path(directory))
            (project/'robot.json').write_bytes(b'different body')
            with self.assertRaisesRegex(ValueError,'physical asset'):
                apply_default_sprint(project,profile_directory=profile)


if __name__=='__main__':unittest.main()
