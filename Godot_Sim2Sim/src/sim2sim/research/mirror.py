"""Exact bilateral policy transform, using the upstream 61D symmetry contract."""
import argparse,hashlib
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,compose

JOINT_PERM=np.array([9,10,11,12,13,5,6,7,8,0,1,2,3,4],np.int64)
JOINT_SIGN=np.array([-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,-1],np.float32)
OBS_PERM=np.r_[np.arange(6),6+JOINT_PERM,20+JOINT_PERM,34+JOINT_PERM,np.arange(48,61)].astype(np.int64)
OBS_SIGN=np.r_[[-1,1,-1],[1,-1,1],JOINT_SIGN,JOINT_SIGN,JOINT_SIGN,[1,-1,-1],[1,1,-1,-1],[1,-1,1,-1,1,-1]].astype(np.float32)

def observation_sign(task=None,heading_input=False,yaw_memory_input=False,state_input=''):
    sign=OBS_SIGN.copy()
    if task in ('ground_pick','roller_crouch'):sign[48:50]=1. # Phase is independent of side.
    if heading_input:sign[50]=1. # Heading cosine is even.
    if yaw_memory_input:sign[55]=-1. # Accumulated yaw drift is odd under reflection.
    if state_input:
        from sim2sim.policy_state import STATE_KEY,state_input as validate_state
        validate_state({STATE_KEY:state_input})
        sign[58:61]=[1.,-1.,1.] # Forward speed and height are even; lateral speed is odd.
    return sign

def reflect_obs(x,*,task=None,heading_input=False,yaw_memory_input=False,state_input=''):return np.asarray(x)[...,OBS_PERM]*observation_sign(task,heading_input,yaw_memory_input,state_input)
def reflect_action(x):return np.asarray(x)[...,JOINT_PERM]*JOINT_SIGN

def mirror(source,dest,task="kick_right"):
    original=onnx.load(source);m=compose.add_prefix(original,"mirror_source/")
    input_name=m.graph.input[0].name
    from sim2sim.policy_time import has_heading_input
    from sim2sim.policy_memory import has_yaw_memory
    from sim2sim.policy_state import state_input
    meta={p.key:p.value for p in original.metadata_props}
    arrays={"obs_perm":OBS_PERM,"obs_sign":observation_sign(task,has_heading_input(meta),has_yaw_memory(meta),state_input(meta)),"joint_perm":JOINT_PERM,"joint_sign":JOINT_SIGN}
    nodes=[helper.make_node("Gather",["obs","obs_perm"],["permuted_obs"],axis=1),helper.make_node("Mul",["permuted_obs","obs_sign"],[input_name])]
    nodes+=list(m.graph.node)
    nodes+=[helper.make_node("Gather",[m.graph.output[0].name,"joint_perm"],["permuted_action"],axis=1),helper.make_node("Mul",["permuted_action","joint_sign"],["actions"])]
    graph=helper.make_graph(nodes,"bilateral_policy_transform",[helper.make_tensor_value_info("obs",onnx.TensorProto.FLOAT,[1,61])],[helper.make_tensor_value_info("actions",onnx.TensorProto.FLOAT,[1,14])],initializer=list(m.graph.initializer)+[numpy_helper.from_array(v,k) for k,v in arrays.items()])
    result=helper.make_model(graph,opset_imports=list(original.opset_import));result.ir_version=original.ir_version
    for p in original.metadata_props:
        if p.key not in ("sim2sim_transform","sim2sim_task","sim2sim_source_sha256"):result.metadata_props.add(key=p.key,value=p.value)
    result.metadata_props.add(key="sim2sim_transform",value="bilateral_reflection_61d")
    result.metadata_props.add(key="sim2sim_task",value=task)
    result.metadata_props.add(key="sim2sim_source_sha256",value=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    onnx.checker.check_model(result);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);onnx.save(result,str(dest));return dest

def symmetrize(source,dest,task="roulade"):
    """An exactly bilateral neural ensemble for sagittal/locomotion tasks."""
    dest=Path(dest);reflected_path=dest.with_suffix(".reflected.onnx")
    mirror(source,reflected_path,task)
    original=onnx.load(source);a=compose.add_prefix(original,"direct/")
    b=compose.add_prefix(onnx.load(reflected_path),"reflected/")
    for model in (a,b):
        old=model.graph.input[0].name
        for node in model.graph.node:
            for i,name in enumerate(node.input):
                if name==old:node.input[i]="obs"
    nodes=list(a.graph.node)+list(b.graph.node)
    nodes += [helper.make_node("Add",[a.graph.output[0].name,b.graph.output[0].name],["sum"]),
              helper.make_node("Mul",["sum","half"],["actions"])]
    graph=helper.make_graph(nodes,"bilateral_mean_actor",[helper.make_tensor_value_info("obs",onnx.TensorProto.FLOAT,[1,61])],
        [helper.make_tensor_value_info("actions",onnx.TensorProto.FLOAT,[1,14])],
        initializer=list(a.graph.initializer)+list(b.graph.initializer)+[numpy_helper.from_array(np.array(.5,np.float32),"half")])
    result=helper.make_model(graph,opset_imports=list(original.opset_import));result.ir_version=original.ir_version
    metadata={p.key:p.value for p in original.metadata_props}
    metadata['sim2sim_parent_architecture']=metadata.get('sim2sim_model_architecture','neural_actor')
    metadata['sim2sim_model_architecture']='bilateral_mean_neural_actor'
    metadata.update(sim2sim_transform="bilateral_mean_actor",sim2sim_task=task,
                    sim2sim_source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    for key,value in metadata.items():result.metadata_props.add(key=key,value=value)
    onnx.checker.check_model(result);onnx.save(result,str(dest));reflected_path.unlink();return dest

def main():
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("dest",type=Path);p.add_argument("--task",default="kick_right")
    a=p.parse_args();print(mirror(a.source,a.dest,a.task))

if __name__=="__main__":main()
