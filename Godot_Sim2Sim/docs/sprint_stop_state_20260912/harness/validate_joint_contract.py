from pathlib import Path
import json
import numpy as np
import torch
import onnxruntime as ort
from sim2sim.research.kinematic_observer import KinematicObserver
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_stop_state_20260912');samples=json.loads((r/'native_sensor_trace/state.sensors.json').read_text());trace=json.loads((r/'native_sensor_trace/trace.json').read_text());spec=json.loads(Path('godot/generated/microduck_ball_stand_fix/robot_spec.json').read_text());joints={j['name']:j for j in spec['joints']};acts=sorted(spec['actuators'],key=lambda x:x['id']);params={key:np.array([joints[x['joint']][key] for x in acts]) for key in ['damping','frictionloss']};kp=np.array([x['kp'] for x in acts]);kv=np.array([x['kv'] for x in acts]);limit=1.75*.36601349688984386
assert np.all(kp==.55) and np.all(kv==0.) and np.all(params['damping']==.053) and np.all(params['frictionloss']==.0048)
ob=KinematicObserver(1,'cpu',dtype=torch.float64,track_angular=False);ob.reset(torch.arange(1),torch.tensor([samples[0]['q']],dtype=torch.float64),torch.tensor([samples[0]['quat']],dtype=torch.float64));est=[];maximum=0.;tau_error=0.
for i,x in enumerate(samples):
 if i:ob.update(torch.tensor([x['q']],dtype=torch.float64),torch.tensor([x['quat']],dtype=torch.float64))
 qd=ob.qd.numpy()[0].copy();est.append(qd);maximum=max(maximum,float(np.max(np.abs(qd-x['qd']))))
 if i%4==3:
  step=i//4;ctrl=np.array(trace['rows'][step]['ctrl']);q=np.array(x['q']);tau=np.clip(kp*(ctrl-q),-limit,limit)-params['damping']*qd-params['frictionloss']*np.tanh(qd/.05)
  actual=(trace['rows'][step+1]['raw'] if step+1<len(trace['rows']) else trace['summary']['final_raw'])['tau'];tau_error=max(tau_error,float(np.max(np.abs(tau-actual))))
assert maximum<1e-7 and tau_error<1e-7,(maximum,tau_error)
options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
bank={k:ort.InferenceSession(str(p),sess_options=options,providers=['CPUExecutionProvider']) for k,p in [('walking',Path('results/sprint_20260912/runs/s05_native_handoff/final.onnx')),('sprint',Path('results/sprint_joint_gpu_20260912/train_tracking_split/final.onnx'))]};action_error=0.
for i,x in enumerate(trace['rows']):
 obs=np.array(x['obs'],np.float32)[None];obs[0,20:34]=est[4*i];ses=bank[x['skill']];out=ses.run(None,{ses.get_inputs()[0].name:obs})[0];action_error=max(action_error,float(np.max(np.abs(out-x['action']))))
assert action_error<1e-5,action_error
atomic_json(r/'joint_contract_validation.json',dict(passed=True,physics_substeps=2400,policy_steps=600,max_joint_velocity_error=maximum,max_pd_torque_error=tau_error,max_policy_action_error=action_error,angular_path='unchanged solver velocity in GPU proxy; full angular projection remains rejected',scope='Actual native input trace and motor outputs; no simulation or model changes'))
print(maximum,tau_error,action_error)
