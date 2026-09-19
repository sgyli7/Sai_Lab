import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto

from sim2sim.research.brake_pose_search import action_offset,adapted_action,export
from sim2sim.research.models import NativeAnchor
from sim2sim.policy_state import wrap_anchor


class BrakePoseExportTests(unittest.TestCase):
    def test_feedback_exports_keep_double_anchor_and_nonbraking_behavior(self):
        from sim2sim.research import brake_height_feedback, brake_near_stop, brake_onset_pose, brake_pitch_rate
        from sim2sim.research.brake_velocity_pose import export as velocity_export
        rng=np.random.default_rng(915005)
        weights=numpy_helper.from_array(rng.normal(0,.01,(61,14)), 'weights')
        nodes=[helper.make_node('Cast',['obs'],['precise'],to=TensorProto.DOUBLE),
               helper.make_node('MatMul',['precise','weights'],['product']),
               helper.make_node('Cast',['product'],['actions'],to=TensorProto.FLOAT)]
        graph=helper.make_graph(nodes,'double_actor',[helper.make_tensor_value_info('obs',TensorProto.FLOAT,[1,61])],
                                [helper.make_tensor_value_info('actions',TensorProto.FLOAT,[1,14])],[weights])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=10)
        probes=rng.normal(0,.2,(64,61)).astype(np.float32)
        probes[:32,48]=np.linspace(0.,.6,32)
        probes[32:,48]=-.5
        probes[:,58]=np.linspace(-.3,.3,64)
        probes[:,60]=np.linspace(.075,.14,64)
        parameters=[.03,-.04,.05,-.06,.07]
        exporters=[(module.__name__,lambda source,target,module=module:module.export(source,parameters,target))
                   for module in [brake_height_feedback,brake_near_stop,brake_onset_pose,brake_pitch_rate]]
        for dimensions,mode in [(10,'centered'),(15,'centered'),(15,'excess_forward_tilt')]:
            exporters.append((f'velocity_{dimensions}_{mode}',
                lambda source,target,dimensions=dimensions,mode=mode:velocity_export(source,parameters*(dimensions//5),target,mode)))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original=root/'original.onnx';source=root/'state.onnx'
            onnx.save(model,original);wrap_anchor(original,source)
            source_graph=onnx.load(source);anchor=NativeAnchor(source)
            for name,emit in exporters:
                with self.subTest(adapter=name):
                    target=root/(name+'.onnx');report=emit(source,target);emitted=onnx.load(target)
                    self.assertLess(report['max_abs'],1e-5)
                    self.assertEqual([n.SerializeToString() for n in emitted.graph.node[:len(source_graph.graph.node)]],
                                     [n.SerializeToString() for n in source_graph.graph.node])
                    self.assertEqual(emitted.graph.initializer[0].SerializeToString(),weights.SerializeToString())
                    actual=NativeAnchor(target)(probes)
                    np.testing.assert_array_equal(actual[:32],anchor(probes[:32]))
                    self.assertTrue(np.isfinite(actual).all())

    def test_original_double_graph_and_nonbraking_actions_are_preserved(self):
        rng=np.random.default_rng(915002)
        weights=numpy_helper.from_array(rng.normal(0,.01,(61,14)), 'weights')
        nodes=[helper.make_node('Cast',['obs'],['precise'],to=TensorProto.DOUBLE),
               helper.make_node('MatMul',['precise','weights'],['product']),
               helper.make_node('Cast',['product'],['actions'],to=TensorProto.FLOAT)]
        graph=helper.make_graph(nodes,'double_actor',[helper.make_tensor_value_info('obs',TensorProto.FLOAT,[1,61])],
                                [helper.make_tensor_value_info('actions',TensorProto.FLOAT,[1,14])],[weights])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=10)
        observations=rng.normal(0,.2,(7,61)).astype(np.float32)
        observations[:,48]=[0.,.6,.01,-.5,-.05,-.025,-1e-6]
        offset=action_offset([-.2,.1,-.15,.12,.2])
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.onnx';target=Path(directory)/'target.onnx';onnx.save(model,source)
            report=export(source,offset,target)
            old=NativeAnchor(source);new=NativeAnchor(target)
            actual=new(observations);expected=adapted_action(old,observations,offset)
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(actual[:3],old(observations[:3]))
            emitted=onnx.load(target)
            feedback=(.3,.02)
            report_feedback=export(source,offset,target,feedback)
            actual_feedback=NativeAnchor(target)(observations)
            np.testing.assert_allclose(actual_feedback,adapted_action(old,observations,offset,feedback),rtol=0,atol=1e-5)
            np.testing.assert_array_equal(actual_feedback[:3],old(observations[:3]))
            self.assertTrue(report_feedback['positive_and_neutral_exact'])
        self.assertTrue(report['positive_and_neutral_exact'])
        self.assertEqual([node.SerializeToString() for node in emitted.graph.node[:3]],
                         [node.SerializeToString() for node in nodes])
        self.assertEqual(emitted.graph.initializer[0].data_type,TensorProto.DOUBLE)
        np.testing.assert_array_equal(offset[9:],-offset[:5])
        np.testing.assert_array_equal(offset[5:9],0.)


if __name__=='__main__':unittest.main()
