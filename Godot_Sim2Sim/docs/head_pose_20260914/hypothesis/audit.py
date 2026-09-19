from pathlib import Path
import hashlib,json
import numpy as np
import mujoco
R=Path(__file__).resolve().parent
old=R.parent/'sprint_input_diagnosis_20260914';spec=json.loads((old/'robot_spec.json').read_text())
mass={b['name']:b['mass'] for b in spec['bodies'] if b['name'] not in ['world','ball']}
head=['neck','neck_pitch','yaw_roll_motion','jaw_soft'];total=sum(mass.values());hm=sum(mass[k] for k in head)
model=mujoco.MjModel.from_xml_path(spec['mjcf']);data=mujoco.MjData(model)
ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,k) for k in mass]);weights=np.array(list(mass.values()))
np.testing.assert_allclose(model.body_mass[ids],weights,atol=1e-12,rtol=0)
qix=model.jnt_qposadr[model.actuator_trnid[:,0]];assert len(qix)==14
results={};hashes={}
for label in ['forward','sprint','backward','walk_backward_025','walk_forward_030']:
 path=old/'gait'/label/'native-0.json'
 if not path.exists():path=old/'command_ceiling'/label/'native-0.json'
 hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 trace=json.loads(path.read_text());rows=[r for r in trace['rows'] if 3<=r['episode_t']<=8 and abs(r['requested_command'][0])>.01]
 values=[];sensitivity={f'{joint}_{angle:+d}':[] for joint in ['neck_pitch','head_pitch'] for angle in [-10,-5,5,10]}
 for row in rows:
  bodies={b['name']:np.array(b['pos']) for b in row['raw']['body_states']};q=row['body']['base_quat'];w,x,y,z=q;yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z));h=np.array([np.cos(yaw),np.sin(yaw),0]);com=sum(mass[k]*bodies[k] for k in mass)/total;hc=sum(mass[k]*bodies[k] for k in head)/hm;foot=(bodies['ankle_left']+bodies['ankle_right'])/2
  values.append([np.array(row['body']['base_linvel'])@h,(hc-bodies['trunk_base'])@h,(com-foot)@h,*np.rad2deg(np.array(row['raw']['q'])[[5,6]])])
  # Static ideal joint FK only: no integration, action, reward or velocity test.
  data.qpos[:]=model.qpos0;data.qpos[:3]=row['body']['base_pos'];data.qpos[3:7]=q;data.qpos[qix]=row['raw']['q'];mujoco.mj_forward(model,data)
  base_com=(weights[:,None]*data.xipos[ids]).sum(0)/total
  for index,joint in [(5,'neck_pitch'),(6,'head_pitch')]:
   original=data.qpos[qix[index]]
   for angle in [-10,-5,5,10]:
    data.qpos[qix[index]]=original+np.deg2rad(angle);mujoco.mj_forward(model,data);candidate=(weights[:,None]*data.xipos[ids]).sum(0)/total
    sensitivity[f'{joint}_{angle:+d}'].append(float((candidate-base_com)@h))
   data.qpos[qix[index]]=original
  mujoco.mj_forward(model,data)
 results[label]=dict(frames=len(rows),window_s=[rows[0]['episode_t'],rows[-1]['episode_t']],requested_vx=rows[0]['requested_command'][0],mean=dict(zip(['forward_velocity_mps','head_neck_com_minus_trunk_com_m','whole_robot_com_minus_ankle_midpoint_m','neck_pitch_deg','head_pitch_deg'],np.mean(values,axis=0).tolist())),static_fk_com_x_shift_m={k:float(np.mean(v)) for k,v in sensitivity.items()},fell=any(r['fell'] for r in rows))
result=dict(total_robot_mass_kg=total,head_neck_mass_kg=hm,head_neck_fraction=hm/total,results=results,input_sha256=hashes,scope='Read-only prior trajectories. Ball/world excluded. Godot rigid-body origins are custom inertial COM frames; ankle midpoint is NOT center of pressure/support polygon. Static FK sensitivity uses source articulated geometry/masses and fixed other joints/root, not exact Jolt compliance or proof of increased speed. Nominal command ±0.25 pair uses frozen ordinary S05; finite one-episode windows, no causal posture intervention or new simulation.')
(R/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(mass_fraction=hm/total,forward=results['forward'],backward_same_command=results['walk_backward_025']),indent=2))
