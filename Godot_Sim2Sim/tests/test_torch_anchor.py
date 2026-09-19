import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import torch

from sim2sim.research.torch_anchor import TorchAnchor


class TorchAnchorTests(unittest.TestCase):
    def test_batched_mixed_precision_graph_matches_original_runtime(self):
        rng = np.random.default_rng(925010)
        tensors = [numpy_helper.from_array(rng.normal(size=(14, 61)), 'weight'),
                   numpy_helper.from_array(rng.normal(size=14), 'bias')]
        nodes = [helper.make_node('Cast', ['obs'], ['precise'], to=TensorProto.DOUBLE),
                 helper.make_node('Gemm', ['precise', 'weight', 'bias'], ['linear'], transB=1, alpha=.5, beta=2.),
                 helper.make_node('Cast', ['linear'], ['activation_input'], to=TensorProto.FLOAT),
                 helper.make_node('Elu', ['activation_input'], ['action'], alpha=.7)]
        graph = helper.make_graph(nodes, 'mixed_precision',
                                  [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, 61])],
                                  [helper.make_tensor_value_info('action', TensorProto.FLOAT, [1, 14])], tensors)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'actor.onnx'
            onnx.save(model, path)
            original = path.read_bytes()
            actor = TorchAnchor(path)
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            reference = ort.InferenceSession(original, sess_options=options, providers=['CPUExecutionProvider'])
            obs = rng.normal(size=(16, 61)).astype(np.float32)
            expected = np.concatenate([reference.run(None, {'obs': x[None]})[0] for x in obs])
            actual = actor(torch.from_numpy(obs)).numpy()
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
            self.assertEqual({x.dtype for x in actor.buffers()}, {torch.float64})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(actor.parameters()), [])

    def test_unsupported_operator_fails_before_rollout(self):
        graph = helper.make_graph([helper.make_node('Sin', ['obs'], ['action'])], 'unknown',
                                  [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, 61])],
                                  [helper.make_tensor_value_info('action', TensorProto.FLOAT, [1, 61])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'actor.onnx'
            onnx.save(model, path)
            with self.assertRaisesRegex(ValueError, 'Unsupported frozen operator: Sin'):
                TorchAnchor(path)


if __name__ == '__main__':
    unittest.main()
