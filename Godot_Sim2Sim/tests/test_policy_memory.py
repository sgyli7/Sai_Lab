import tempfile,unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
from sim2sim.policy_memory import YawDriftMemory


class MemoryContract(unittest.TestCase):
    def test_reflection_matches_mirrored_imu_history(self):
        from sim2sim.research.mirror import reflect_obs
        a=YawDriftMemory();b=YawDriftMemory();rng=np.random.default_rng(48)
        for i in range(200):
            obs=rng.normal(0,.2,61).astype(np.float32)
            obs[3:6]/=np.linalg.norm(obs[3:6]);obs[50]=.5 if 30<i<60 else 0.
            direct=a.observe(obs)
            reflected=b.observe(reflect_obs(obs,task='kick_right'))
            np.testing.assert_allclose(reflected,reflect_obs(direct,task='kick_right',yaw_memory_input=True),atol=1e-7)

    def test_vertical_projection_stamp_and_turn_reset(self):
        m=YawDriftMemory();obs=np.zeros(61,np.float32)
        obs[3:6]=[0,.6,-.8];obs[1]=1.
        self.assertEqual(m.observe(obs,stamp=0)[55],0.)
        first=m.observe(obs,stamp=.02)
        self.assertAlmostEqual(float(first[55]),.012,places=7)
        np.testing.assert_array_equal(m.observe(obs,stamp=.02),first)
        np.testing.assert_array_equal(first[:55],obs[:55])
        obs[50]=.5;self.assertEqual(m.observe(obs,stamp=.04)[55],0.)
        obs[50]=0.;self.assertEqual(m.observe(obs,stamp=.06)[55],0.)
        self.assertGreater(m.observe(obs,stamp=.08)[55],0.)
        m.reset();self.assertEqual(m.observe(obs,stamp=0)[55],0.)

    def test_parent_mask_and_increment_normalization(self):
        import torch
        from sim2sim.research.tasks import TASKS
        from sim2sim.research.walk_memory import prepare
        from sim2sim.research.models import NativeAnchor,Policy,export_policy,parity
        with tempfile.TemporaryDirectory() as d:
            source=TASKS['walking'].source;dest=prepare(source,Path(d)/'memory.onnx')
            obs=np.random.default_rng(711).normal(0,.2,(100,61)).astype(np.float32)
            expected=obs.copy();expected[:,55]=0
            np.testing.assert_array_equal(NativeAnchor(dest)(obs),NativeAnchor(source)(expected))
            actor=Policy(dest,'residual',template=source)
            self.assertEqual(float(actor.delta.mean[55]),0.)
            self.assertEqual(float(actor.delta.denominator[55]),1.)
            with torch.no_grad():actor.delta.net[-1].weight.add_(.001)
            self.assertTrue(parity(actor,export_policy(actor,Path(d)/'learned.onnx'),n=500)['passed'])

    def test_native_world_and_deployment_memory_agree(self):
        self._native_world_and_deployment_memory_agree('walking')
        self._native_world_and_deployment_memory_agree('kick_right')

    def _native_world_and_deployment_memory_agree(self,skill):
        from sim2sim.research.tasks import TASKS
        from sim2sim.research.walk_memory import prepare
        from sim2sim.research.world import World
        from sim2sim.policy import PolicyBundle
        from sim2sim.obs import build_obs
        with tempfile.TemporaryDirectory() as d:
            path=prepare(TASKS[skill].source,Path(d)/'memory.onnx');actor=PolicyBundle(path)
            w=World(replace(TASKS[skill],robot='microduck_ball_stand_fix'),yaw_memory_input=True)
            condition='walk_025' if skill=='walking' else 'default'
            try:
                w.reset(93001,condition)
                for _ in range(30):
                    expected=w.obs();np.testing.assert_array_equal(w.obs(),expected)
                    raw=build_obs(w.state,w.last,w.command(),w.home)
                    action=actor.infer(raw)
                    self.assertAlmostEqual(actor._yaw_memory.error,float(expected[55]),places=7)
                    w.step(action)
                actor.reset_context();w.reset(93002,condition)
                actor.infer(build_obs(w.state,w.last,w.command(),w.home))
                self.assertEqual(actor._yaw_memory.error,0.)
                self.assertEqual(w.obs()[55],0.)
            finally:w.close()


if __name__=='__main__':unittest.main()
