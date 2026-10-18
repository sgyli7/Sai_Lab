"""Command-only CPU checks; no container, simulator, CUDA or queue execution."""
import ast,tempfile,sys,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.pi05_g1_mobile import sim_launcher_acceptance_v2 as old
from experiments.pi05_g1_mobile import sim_launcher_pose_speed as new

class LauncherTests(unittest.TestCase):
    def test_exact_copy_only_two_module_seams(self):
        expected=Path(old.__file__).read_text().replace("'experiments.pi05_g1_mobile.sim_launcher_acceptance_v2'","'experiments.pi05_g1_mobile.sim_launcher_pose_speed'").replace("'experiments.pi05_g1_mobile.acceptance_body_ids'","'experiments.pi05_g1_mobile.acceptance_pose_speed'")
        self.assertEqual(Path(new.__file__).read_text(),expected)
        root=Path(old.__file__).parent
        self.assertEqual((root/'batch_pose_speed.py').read_text(),(root/'batch_acceptance_v2.py').read_text().replace("'experiments.pi05_g1_mobile.sim_launcher_acceptance_v2'","'experiments.pi05_g1_mobile.sim_launcher_pose_speed'"))
    def test_docker_recursion_module_switched_mounts_owner_labels_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(old,'ROOT',Path(tmp)),patch.object(new,'ROOT',Path(tmp)):
            args=dict(config=Path(tmp)/'config.json',gpu=False,launch_id='cpufake',budget_token='ownedtoken',cidfile=Path(tmp)/'container.cid')
            oldcmd=old.docker_command(**args);newcmd=new.docker_command(**args)
            self.assertEqual(newcmd,[x.replace('experiments.pi05_g1_mobile.sim_launcher_acceptance_v2','experiments.pi05_g1_mobile.sim_launcher_pose_speed') for x in oldcmd])
            self.assertIn('sai.pi05_g1.budget=ownedtoken',newcmd);self.assertIn('sai.pi05_g1.launch=cpufake',newcmd);self.assertIn('--cidfile',newcmd)
    def test_inside_container_actual_python_command_uses_explicit_compatibility_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'config.json';python=Path('/fake/runtime/python')
            with patch.object(sys,'argv',['launcher','--python',str(python),'--config',str(config)]),patch.object(new,'probe',return_value={'ready':True}),patch.object(new.subprocess,'run') as run:
                new.main()
            run.assert_called_once_with([str(python),'-m','experiments.pi05_g1_mobile.acceptance_pose_speed','--config',str(config.resolve())],cwd=new.ROOT,check=True)
    def test_probe_still_probe_only_no_compatibility_execution(self):
        with patch.object(sys,'argv',['launcher','--probe']),patch.object(new,'probe',return_value={'ready':True}),patch.object(new.subprocess,'run') as run:new.main()
        run.assert_not_called()

if __name__=='__main__':unittest.main()
