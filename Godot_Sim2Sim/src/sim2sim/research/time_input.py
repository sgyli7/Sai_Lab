"""Preserve a zero-command teacher while enabling a time-aware increment.

The public actor remains 61 -> 14. Its first command slot receives elapsed
time / duration, clamped at one. The frozen teacher sees its original zeros;
the trainable increment sees elapsed time and all ordinary proprioception.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import compose,helper,numpy_helper


def prepare(source,dest,seconds=5.):
    from sim2sim.policy_time import time_input_seconds
    original=onnx.load(source)
    if time_input_seconds({p.key:p.value for p in original.metadata_props}):
        raise ValueError("A time-aware actor must not be masked again")
    if not np.isfinite(seconds) or seconds<=0:raise ValueError("Invalid duration")
    m=compose.add_prefix(original,"zero_command_teacher/")
    mask=np.ones((1,61),np.float32);mask[:,48:]=0
    nodes=[helper.make_node("Mul",["obs","teacher_input_mask"],[m.graph.input[0].name])]
    nodes+=list(m.graph.node)+[helper.make_node("Identity",[m.graph.output[0].name],["actions"])]
    graph=helper.make_graph(nodes,"time_ready_teacher",[helper.make_tensor_value_info("obs",onnx.TensorProto.FLOAT,[1,61])],
        [helper.make_tensor_value_info("actions",onnx.TensorProto.FLOAT,[1,14])],
        initializer=list(m.graph.initializer)+[numpy_helper.from_array(mask,"teacher_input_mask")])
    model=helper.make_model(graph,opset_imports=list(original.opset_import));model.ir_version=original.ir_version
    meta={p.key:p.value for p in original.metadata_props}
    meta.update(sim2sim_time_input_s=str(seconds),sim2sim_command_mode="one_shot_time",sim2sim_task="roulade",
        sim2sim_time_teacher_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    for k,v in meta.items():model.metadata_props.add(key=k,value=v)
    onnx.checker.check_model(model);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,str(dest))
    return dest


def add_heading(source,dest):
    """Keep the parent exact while making heading available to a new increment."""
    from sim2sim.policy_time import time_input_seconds
    original=onnx.load(source);meta={p.key:p.value for p in original.metadata_props}
    if not time_input_seconds(meta):raise ValueError("Heading requires a timed parent")
    if meta.get('sim2sim_heading_input'):raise ValueError("A heading-aware parent must not be masked again")
    m=compose.add_prefix(original,"heading_free_parent/")
    mask=np.ones((1,61),np.float32);mask[:,49:51]=0
    nodes=[helper.make_node("Mul",["obs","parent_input_mask"],[m.graph.input[0].name])]
    nodes+=list(m.graph.node)+[helper.make_node("Identity",[m.graph.output[0].name],["actions"])]
    graph=helper.make_graph(nodes,"heading_ready_parent",[helper.make_tensor_value_info("obs",onnx.TensorProto.FLOAT,[1,61])],
        [helper.make_tensor_value_info("actions",onnx.TensorProto.FLOAT,[1,14])],
        initializer=list(m.graph.initializer)+[numpy_helper.from_array(mask,"parent_input_mask")])
    model=helper.make_model(graph,opset_imports=list(original.opset_import));model.ir_version=original.ir_version
    meta.update(sim2sim_heading_input="lateral_axis_sin_cos",sim2sim_heading_parent_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    for k,v in meta.items():model.metadata_props.add(key=k,value=v)
    onnx.checker.check_model(model);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,str(dest));return dest


def main():
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("dest",type=Path)
    a=p.parse_args();print(prepare(a.source,a.dest))


if __name__=="__main__":main()
