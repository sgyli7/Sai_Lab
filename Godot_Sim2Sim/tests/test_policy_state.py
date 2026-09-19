import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from sim2sim.policy_state import BRAKE_STATE_V1, STATE_KEY, inject_state, state_input, wrap_anchor
from sim2sim.research.models import NativeAnchor


class ResidualStateTests(unittest.TestCase):
    def test_frozen_ordinary_controller_keeps_body_commands_during_state_training(self):
        from sim2sim.obs import build_obs
        from sim2sim.research.sprint_composition import SprintComposition
        state=SimpleNamespace(base_quat_wxyz=np.array([1.,0.,0.,0.]),base_angvel_local=np.zeros(3),
            base_linvel=np.array([.3,-.1,0.]),base_pos=np.array([0.,0.,.115]),q=np.zeros(14),qd=np.zeros(14))
        command=np.zeros(13,np.float32);command[10:]=[.01,.02,.03]
        legacy=build_obs(state,np.zeros(14),command,np.zeros(14))
        world=SimpleNamespace(state_input=BRAKE_STATE_V1,state=state,last=np.zeros(14),home=np.zeros(14),
            sprint_active=lambda:False,command=lambda:command,
            obs=lambda:inject_state(legacy,state,BRAKE_STATE_V1))
        controller=object.__new__(SprintComposition);controller.learn_all=False
        controller.counts={'learned':0,'ordinary':0}
        received=[]
        def actor(x):received.append(x.copy());return np.zeros((1,14),np.float32)
        controller.actor=actor
        controller.actions([world],np.ones((1,14),np.float32))
        np.testing.assert_array_equal(received[0][0],legacy)

    def test_motion_state_ablation_hides_only_added_actor_features(self):
        from unittest.mock import patch
        import torch
        from sim2sim.research.models import Increment
        record=SimpleNamespace(mean=np.zeros(61,np.float32),std=np.ones(61,np.float32))
        with patch('sim2sim.research.models.parse_mlp_onnx',return_value=record):
            hidden=Increment('unused',variant='residual',mask_motion_state=True)
            visible=Increment('unused',variant='residual')
        with torch.no_grad():
            for parameter in hidden.net.parameters():parameter.fill_(.01)
            for target,source in zip(visible.net.parameters(),hidden.net.parameters()):target.copy_(source)
        x=torch.zeros(2,61);x[1,58:61]=torch.tensor([.3,-.1,.115])
        np.testing.assert_array_equal(hidden(x)[0].detach(),hidden(x)[1].detach())
        self.assertGreater(float((visible(x)[0]-visible(x)[1]).detach().abs().max()),1e-6)
        x[1,48]=.3
        self.assertGreater(float((hidden(x)[0]-hidden(x)[1]).detach().abs().max()),1e-6)
        missing=hidden.state_dict();del missing['observation_mask']
        with self.assertRaisesRegex(RuntimeError,'observation_mask'):hidden.load_state_dict(missing)

    def test_world_velocity_is_projected_into_heading_and_old_contract_is_unchanged(self):
        obs = np.arange(61, dtype=np.float32)
        state = SimpleNamespace(base_quat_wxyz=np.array([np.sqrt(.5), 0., 0., np.sqrt(.5)]),
                                base_linvel=np.array([.2, .6, 9.]), base_pos=np.array([2., 3., .115]))
        actual = inject_state(obs, state, BRAKE_STATE_V1)
        np.testing.assert_array_equal(actual[:58], obs[:58])
        np.testing.assert_allclose(actual[58:], [.6, -.2, .115], atol=1e-7)
        self.assertIs(inject_state(obs, state, ''), obs)
        with self.assertRaises(ValueError): state_input({STATE_KEY: 'unknown'})
        state.base_linvel[0] = np.nan
        with self.assertRaises(ValueError): inject_state(obs, state, BRAKE_STATE_V1)

    def test_anchor_nodes_and_double_tensors_are_identical_and_state_is_masked(self):
        rng = np.random.default_rng(915002)
        nodes = [helper.make_node('Cast', ['obs'], ['precise'], to=TensorProto.DOUBLE),
                 helper.make_node('MatMul', ['precise', 'weights'], ['product']),
                 helper.make_node('Cast', ['product'], ['actions'], to=TensorProto.FLOAT)]
        tensor = numpy_helper.from_array(rng.normal(0, .1, (61, 14)), 'weights')
        graph = helper.make_graph(nodes, 'double_actor',
            [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, 61])],
            [helper.make_tensor_value_info('actions', TensorProto.FLOAT, [1, 14])], [tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'old.onnx', Path(directory)/'new.onnx'
            onnx.save(model, source); wrap_anchor(source, target)
            original, wrapped = NativeAnchor(source), NativeAnchor(target)
            obs = rng.normal(size=(64, 61)).astype(np.float32)
            masked = obs.copy(); masked[:, 58:61] = 0
            np.testing.assert_array_equal(wrapped(obs), original(masked))
            self.assertEqual(wrapped.state_input, BRAKE_STATE_V1)
            emitted = onnx.load(target)
            self.assertEqual([n.SerializeToString() for n in emitted.graph.node[1:]],
                             [n.SerializeToString() for n in model.graph.node])
            self.assertEqual(emitted.graph.initializer[0].SerializeToString(), tensor.SerializeToString())
            with self.assertRaises(ValueError): wrap_anchor(target, source)


if __name__ == '__main__': unittest.main()
