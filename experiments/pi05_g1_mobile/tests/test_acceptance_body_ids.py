import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from experiments.pi05_g1_mobile import acceptance_body_ids as subject


class BodyIdTests(unittest.TestCase):
    def fixture(self):
        return {'bodies':[{'name':'a'},{'name':'b'}],
                'collisions':[{'body':1,'points':[[1,2,3]],'shape':'convex_hull'}], 'other':{'mass':2.}}
    def test_only_body_identity_changes_and_actual_pose_order_is_named(self):
        robot=self.fixture();before=copy.deepcopy(robot);mapped=subject.adapt_robot_geometry(robot)
        self.assertEqual(robot,before);self.assertEqual(mapped['collisions'][0]['body'],'b')
        actual_names=['b','a'];self.assertEqual(actual_names.index(mapped['collisions'][0]['body']),0)
        undo=copy.deepcopy(mapped);undo['collisions'][0]['body']=1;self.assertEqual(undo,before)
        self.assertIs(mapped['collisions'][0]['points'],robot['collisions'][0]['points'])
    def test_reject_wrong_integer_names_and_duplicates(self):
        for value in (-1,2,True,'b',1.0):
            robot=self.fixture();robot['collisions'][0]['body']=value
            with self.assertRaises(ValueError):subject.adapt_robot_geometry(robot)
        robot=self.fixture();robot['bodies'][1]['name']='a'
        with self.assertRaises(ValueError):subject.adapt_robot_geometry(robot)
    def test_context_restores_even_when_old_math_raises(self):
        original=subject.original.geometry
        with self.assertRaisesRegex(RuntimeError,'test'):
            with subject.geometry_body_names_context():
                self.assertIs(subject.original.geometry,subject._geometry_with_names)
                raise RuntimeError('test')
        self.assertIs(subject.original.geometry,original)
    def test_tuple_shapes_and_math_inputs_preserved(self):
        robot=self.fixture();bundle=(object(),object(),object(),object(),robot,object(),object())
        with patch.object(subject,'_ORIGINAL_GEOMETRY',return_value=bundle):
            mapped=subject._geometry_with_names('path','sha')
        for i in (0,1,2,3,5,6):self.assertIs(mapped[i],bundle[i])
        self.assertEqual(mapped[4]['collisions'][0]['body'],'b')
    def test_public_context_and_cpu_cli_dispatch_are_explicit_and_restore(self):
        previous=sys.argv
        def fake_run(module,run_name):
            self.assertEqual(module,'experiments.pi05_g1_mobile.transport_analysis')
            self.assertEqual(sys.argv[1:],['--episodes','synthetic','--output','new.json'])
            self.assertIs(subject.original.geometry,subject._geometry_with_names)
            return {'cpu':'only'}
        with tempfile.TemporaryDirectory() as tmp,patch.object(subject.runpy,'run_module',side_effect=fake_run),contextlib.redirect_stderr(io.StringIO()):
            target=Path(tmp)/'identity.json'
            result=subject.main(['--module','experiments.pi05_g1_mobile.transport_analysis','--identity-output',str(target),'--','--episodes','synthetic','--output','new.json'])
            self.assertEqual(result,{'cpu':'only'})
            self.assertTrue(json.loads(target.read_text())['cpu_module_mode'])
        self.assertIs(sys.argv,previous)
        self.assertIs(subject.original.geometry,subject._ORIGINAL_GEOMETRY)
        with subject.body_name_geometry():self.assertIs(subject.original.geometry,subject._geometry_with_names)
    def test_unapproved_module_or_missing_separator_refused(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for args in (['--module','experiments.pi05_g1_mobile.training','--','--help'],
                         ['--module','experiments.pi05_g1_mobile.transport_analysis','--help']):
                with self.assertRaises(SystemExit):subject.main(args)

if __name__=='__main__':unittest.main()
