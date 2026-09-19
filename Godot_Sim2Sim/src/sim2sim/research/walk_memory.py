"""Expose explicit locomotion/kick memory while preserving the exact parent."""
import hashlib
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,compose


def prepare(source,dest):
    original=onnx.load(source);meta={p.key:p.value for p in original.metadata_props}
    if meta.get('sim2sim_task','walking') not in ('walking','kick_left','kick_right'):
        raise ValueError('Yaw memory requires locomotion or a kick')
    if meta.get('sim2sim_yaw_memory'):raise ValueError('Parent already has walking memory')
    parent=compose.add_prefix(original,'memory_free_parent/')
    mask=np.ones((1,61),np.float32);mask[:,55]=0
    nodes=[helper.make_node('Mul',['obs','memory_parent_mask'],[parent.graph.input[0].name])]
    nodes+=list(parent.graph.node)+[helper.make_node('Identity',[parent.graph.output[0].name],['actions'])]
    graph=helper.make_graph(nodes,'walking_with_declared_imu_memory',
        [helper.make_tensor_value_info('obs',onnx.TensorProto.FLOAT,[1,61])],
        [helper.make_tensor_value_info('actions',onnx.TensorProto.FLOAT,[1,14])],
        initializer=list(parent.graph.initializer)+[numpy_helper.from_array(mask,'memory_parent_mask')])
    model=helper.make_model(graph,opset_imports=list(original.opset_import));model.ir_version=original.ir_version
    meta.update(sim2sim_yaw_memory='gyro_vertical_integral_v1',sim2sim_yaw_memory_index='55',
                sim2sim_memory_parent_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    for k,v in meta.items():model.metadata_props.add(key=k,value=v)
    onnx.checker.check_model(model);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,str(dest));return dest
