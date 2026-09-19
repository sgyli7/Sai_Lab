import unittest
import tempfile
from pathlib import Path
import numpy as np

from sim2sim.policy_task_state import BrakeTaskState, contacts_from_raw, wrap_anchor


class TaskStateTests(unittest.TestCase):
    def test_brake_learning_cannot_change_other_joint_targets_or_positive_commands(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from sim2sim.research.models import Increment
        record=SimpleNamespace(mean=np.zeros(61,np.float32),std=np.ones(61,np.float32))
        with patch('sim2sim.research.models.parse_mlp_onnx',return_value=record):
            delta=Increment('unused',variant='residual',input_dim=68,
                command_gate='negative_throttle',action_basis='brake_sagittal_v1')
        with torch.no_grad():delta.net[-1].bias.fill_(1.)
        x=torch.zeros(2,68);x[:,48]=torch.tensor([-.5,.6])
        action=delta(x).detach().numpy()
        self.assertTrue(np.all(action[0,[2,3,11,12]]>0))
        np.testing.assert_array_equal(action[0,[0,1,4,5,6,7,8,9,10,13]],0.)
        np.testing.assert_array_equal(action[1],0.)

    def base(self, speed=0., throttle=-.5):
        x=np.zeros(61,np.float32);x[5]=-1.;x[48]=throttle;x[58]=speed;x[60]=.115
        return x

    def test_same_body_state_has_different_brake_history_and_repeated_reads_do_not_advance(self):
        memory=BrakeTaskState();x=self.base(.3)
        first=memory.observe(x,[1,1],0.)
        np.testing.assert_array_equal(first,memory.observe(x,[1,1],0.))
        for i in range(1,51): last=memory.observe(x,[1,1],i*.02)
        np.testing.assert_array_equal(first[:61],last[:61])
        self.assertEqual(last[61],1.)
        self.assertEqual(last[62],0.)
        self.assertAlmostEqual(last[63],.3,places=6)

    def test_short_speed_dip_is_not_a_hold_and_new_brake_resets_memory(self):
        memory=BrakeTaskState()
        for i in range(70): x=memory.observe(self.base(),[1,1],i*.02)
        self.assertGreaterEqual(x[62],1.)
        memory.observe(self.base(throttle=.6),[1,1],1.4)
        x=memory.observe(self.base(.2),[1,1],1.42)
        self.assertEqual(x[61],0.);self.assertEqual(x[62],0.)
        for i in range(1,9):x=memory.observe(self.base(),[1,1],1.42+i*.02)
        self.assertEqual(x[62],0.)
        with self.assertRaises(ValueError):memory.observe(self.base(),[1,1],0.)
        memory.reset();self.assertEqual(memory.observe(self.base(),[1,1],0.)[61],0.)

    def test_contacts_use_declared_wheel_ownership_and_fail_when_missing(self):
        raw={'body_states':[{'name':'left','ground_contact':True},{'name':'right','ground_contact':False}]}
        np.testing.assert_array_equal(contacts_from_raw(raw,[['left'],['right']]),[1,0])
        with self.assertRaises(ValueError):contacts_from_raw(raw,[['missing'],['right']])

    def test_68d_wrapper_preserves_old_double_graph_and_does_not_leak_new_features(self):
        import onnx
        from onnx import helper,numpy_helper,TensorProto
        from sim2sim.research.models import NativeAnchor
        from sim2sim.policy_state import STATE_KEY,BRAKE_STATE_V1
        from sim2sim.policy import OnnxPolicy
        rng=np.random.default_rng(917122)
        tensor=numpy_helper.from_array(rng.normal(0,.1,(61,14)),'weights')
        nodes=[helper.make_node('Cast',['obs'],['double'],to=TensorProto.DOUBLE),
               helper.make_node('MatMul',['double','weights'],['y']),
               helper.make_node('Cast',['y'],['actions'],to=TensorProto.FLOAT)]
        graph=helper.make_graph(nodes,'test',[helper.make_tensor_value_info('obs',TensorProto.FLOAT,[1,61])],
              [helper.make_tensor_value_info('actions',TensorProto.FLOAT,[1,14])],[tensor])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=10)
        helper.set_model_props(model,{STATE_KEY:BRAKE_STATE_V1})
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.onnx';target=Path(directory)/'target.onnx'
            onnx.save(model,source);wrap_anchor(source,target)
            x=rng.normal(size=(24,68)).astype(np.float32)
            np.testing.assert_array_equal(NativeAnchor(source)(x[:,:61]),NativeAnchor(target)(x))
            emitted=onnx.load(target)
            self.assertEqual([n.SerializeToString() for n in emitted.graph.node[1:]],
                             [n.SerializeToString() for n in nodes])
            self.assertEqual(emitted.graph.initializer[0].SerializeToString(),tensor.SerializeToString())
            actor=OnnxPolicy(target);actor.check_dims(14);self.assertEqual(actor.obs_dim,68)
            with self.assertRaises(ValueError):wrap_anchor(target,source)


if __name__=='__main__':unittest.main()
