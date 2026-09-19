"""Small neural input-adapter experiments, contained in the exported actor.

The evaluator and public 61D observation retain the actual requested command.
Only the actor's internal command conditioning is adapted; no physical feedback,
privileged information, external controller or runtime script is introduced.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,compose


def adapt(source,dest,matrix=None,bias=None,task="walking",limits=None,forward_yaw_hinge=None,forward_command_hinge=None):
    matrix=np.asarray(np.eye(3) if matrix is None else matrix,np.float32).reshape(3,3)
    bias=np.asarray(np.zeros(3) if bias is None else bias,np.float32).reshape(3)
    original=onnx.load(source);m=compose.add_prefix(original,"conditioned/")
    arrays={"before_start":np.array([0],np.int64),"before_end":np.array([48],np.int64),
            "command_start":np.array([48],np.int64),"command_end":np.array([51],np.int64),
            "after_start":np.array([51],np.int64),"after_end":np.array([61],np.int64),
            "feature_axis":np.array([1],np.int64),"command_matrix":matrix,"command_bias":bias}
    nodes=[helper.make_node("Slice",["obs",part+"_start",part+"_end","feature_axis"],[part]) for part in ("before","command","after")]
    nodes += [helper.make_node("MatMul",["command","command_matrix"],["mapped"]),
        helper.make_node("Add",["mapped","command_bias"],["latent_command"])]
    latent="latent_command"
    if forward_yaw_hinge is not None:
        threshold,gain=map(float,forward_yaw_hinge)
        arrays.update(forward_start=np.array([0],np.int64),forward_end=np.array([1],np.int64),
            forward_threshold=np.array(threshold,np.float32),forward_yaw_coeff=np.array([0.,0.,gain],np.float32))
        nodes += [helper.make_node("Slice",["command","forward_start","forward_end","feature_axis"],["forward"]),
            helper.make_node("Sub",["forward","forward_threshold"],["above_threshold"]),
            helper.make_node("Relu",["above_threshold"],["positive_excess"]),
            helper.make_node("Mul",["positive_excess","forward_yaw_coeff"],["hinge_offset"]),
            helper.make_node("Add",[latent,"hinge_offset"],["hinged_command"])]
        latent="hinged_command"
    if forward_command_hinge is not None:
        threshold,coefficients=forward_command_hinge
        coefficients=np.asarray(coefficients,np.float32).reshape(3)
        arrays.update(speed_start=np.array([0],np.int64),speed_end=np.array([1],np.int64),
            speed_threshold=np.array(threshold,np.float32),speed_coefficients=coefficients)
        nodes += [helper.make_node('Slice',['command','speed_start','speed_end','feature_axis'],['speed_forward']),
            helper.make_node('Sub',['speed_forward','speed_threshold'],['speed_excess']),
            helper.make_node('Relu',['speed_excess'],['speed_positive_excess']),
            helper.make_node('Mul',['speed_positive_excess','speed_coefficients'],['speed_offset']),
            helper.make_node('Add',[latent,'speed_offset'],['speed_mapped_command'])]
        latent='speed_mapped_command'
    if limits is not None:
        arrays["latent_min"]=np.asarray(limits[0],np.float32);arrays["latent_max"]=np.asarray(limits[1],np.float32)
        nodes += [helper.make_node("Max",[latent,"latent_min"],["lower_bounded"]),
                  helper.make_node("Min",["lower_bounded","latent_max"],["bounded_command"])]
        latent="bounded_command"
    nodes += [helper.make_node("Concat",["before",latent,"after"],[m.graph.input[0].name],axis=1)]
    nodes += list(m.graph.node)
    nodes += [helper.make_node("Identity",[m.graph.output[0].name],["actions"])]
    graph=helper.make_graph(nodes,"command_conditioned_policy",[helper.make_tensor_value_info("obs",onnx.TensorProto.FLOAT,[1,61])],
        [helper.make_tensor_value_info("actions",onnx.TensorProto.FLOAT,[1,14])],
        initializer=list(m.graph.initializer)+[numpy_helper.from_array(v,k) for k,v in arrays.items()])
    model=helper.make_model(graph,opset_imports=list(original.opset_import));model.ir_version=original.ir_version
    replaced={"sim2sim_transform","sim2sim_task","sim2sim_source_sha256","sim2sim_command_adapter"}
    for p in original.metadata_props:
        if p.key not in replaced:model.metadata_props.add(key=p.key,value=p.value)
    for k,v in dict(sim2sim_transform="affine_command_conditioning",sim2sim_task=task,
        sim2sim_source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        sim2sim_command_adapter=json.dumps(dict(matrix=matrix.tolist(),bias=bias.tolist(),limits=limits,forward_yaw_hinge=forward_yaw_hinge,forward_command_hinge=forward_command_hinge))).items():
        model.metadata_props.add(key=k,value=v)
    onnx.checker.check_model(model);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,str(dest));return dest


def parity(source,adapted,matrix,bias=None,n=10000,limits=None,forward_yaw_hinge=None,forward_command_hinge=None):
    from .models import NativeAnchor
    x=np.random.default_rng(972).normal(0,.5,(n,61)).astype(np.float32)
    mapped=x.copy();mapped[:,48:51]=x[:,48:51]@np.asarray(matrix,np.float32)+(np.zeros(3,np.float32) if bias is None else np.asarray(bias,np.float32))
    if forward_yaw_hinge is not None:
        threshold,gain=np.asarray(forward_yaw_hinge,np.float32)
        mapped[:,50]+=np.maximum(x[:,48]-threshold,0)*gain
    if forward_command_hinge is not None:
        threshold,coefficients=forward_command_hinge
        mapped[:,48:51]+=np.maximum(x[:,48:49]-np.float32(threshold),0)*np.asarray(coefficients,np.float32)
    if limits is not None:mapped[:,48:51]=np.clip(mapped[:,48:51],limits[0],limits[1])
    error=float(np.max(np.abs(NativeAnchor(source)(mapped)-NativeAnchor(adapted)(x))))
    return dict(samples=n,max_abs=error,threshold=1e-5,passed=error<1e-5)


def output_gain(source,dest,gain):
    """Scale the neural actor's joint offsets inside ONNX, keeping PD physics fixed."""
    gain=float(gain)
    if not 0 < gain <= 1.:raise ValueError('This probe only contracts actor offsets')
    model=onnx.load(source);old=model.graph.output[0].name;renamed='gain_probe_unscaled_actions'
    for node in model.graph.node:
        for names in (node.input,node.output):
            for i,name in enumerate(names):
                if name==old:names[i]=renamed
    model.graph.initializer.append(numpy_helper.from_array(np.array(gain,np.float32),'gain_probe_factor'))
    model.graph.node.append(helper.make_node('Mul',[renamed,'gain_probe_factor'],[old]))
    metadata={p.key:p.value for p in model.metadata_props}
    metadata.update(sim2sim_output_gain=str(gain),sim2sim_output_gain_parent_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    del model.metadata_props[:]
    for k,v in metadata.items():model.metadata_props.add(key=k,value=v)
    onnx.checker.check_model(model);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,str(dest));return dest


def main():
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("dest",type=Path)
    p.add_argument("--gains",default="1,1,1");p.add_argument("--yaw-from-vx",type=float,default=0.)
    p.add_argument("--bias",default="0,0,0");p.add_argument("--task",default="walking")
    a=p.parse_args();matrix=np.diag([float(x) for x in a.gains.split(",")]);matrix[0,2]=a.yaw_from_vx
    bias=[float(x) for x in a.bias.split(",")];dest=adapt(a.source,a.dest,matrix,bias,a.task)
    report=parity(a.source,dest,matrix,bias);dest.with_suffix(".parity.json").write_text(json.dumps(report,indent=2));print(dest,report)


if __name__=="__main__":main()
