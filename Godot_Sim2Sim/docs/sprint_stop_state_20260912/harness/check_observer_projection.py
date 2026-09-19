from pathlib import Path
import json
import numpy as np
import torch
import onnxruntime as ort
from sim2sim.research.kinematic_observer import KinematicObserver
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_stop_state_20260912');data=json.loads((r/'native_sensor_trace/state.sensors.json').read_text());trace=json.loads((r/'native_sensor_trace/trace.json').read_text());ob=KinematicObserver(1,'cpu',dtype=torch.float64);q=torch.tensor([data[0]['q']],dtype=torch.float64);quat=torch.tensor([data[0]['quat']],dtype=torch.float64);ob.reset(torch.arange(1),q,quat);pred=[];error=np.zeros(2)
for i,row in enumerate(data):
 quat=torch.tensor([row['quat']],dtype=torch.float64)
 if i:ob.update(torch.tensor([row['q']],dtype=torch.float64),quat)
 w=ob.angular_local(quat).numpy()[0];qd=ob.qd.numpy()[0]
 error=np.maximum(error,[np.max(np.abs(w-row['omega'])),np.max(np.abs(qd-row['qd']))])
 if i%4==0:pred.append((w.copy(),qd.copy()))
options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
bank={k:ort.InferenceSession(str(p),sess_options=options,providers=['CPUExecutionProvider']) for k,p in [('walking',Path('results/sprint_20260912/runs/s05_native_handoff/final.onnx')),('sprint',Path('results/sprint_joint_gpu_20260912/train_tracking_split/final.onnx'))]}
max_action=0.;max_on_native=0.
for i,row in enumerate(trace['rows']):
 ses=bank[row['skill']];inp=np.array(row['obs'],np.float32)[None];actual=ses.run(None,{ses.get_inputs()[0].name:inp})[0]
 updated=inp.copy();updated[0,:3]=pred[i][0];updated[0,20:34]=pred[i][1];changed=ses.run(None,{ses.get_inputs()[0].name:updated})[0]
 max_on_native=max(max_on_native,float(np.max(np.abs(actual-row['action']))));max_action=max(max_action,float(np.max(np.abs(actual-changed))))
result=dict(completed=True,angular_velocity_error=float(error[0]),joint_velocity_error=float(error[1]),native_onnx_action_parity=max_on_native,observer_projection_action_error=max_action,original_strict_sensor_check_passed=bool(error.max()<1e-7),policy_parity_under_1e5=bool(max_action<1e-5),note='Quaternion exported by policy contract is not the internal Godot Basis used by the native observer; full orientation projection remains a measured discrepancy. No threshold changed.')
atomic_json(r/'observer_projection_audit.json',result);print(result)
