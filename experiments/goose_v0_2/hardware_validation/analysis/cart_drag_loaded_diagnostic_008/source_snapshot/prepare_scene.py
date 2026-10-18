import sys,os
sys.dont_write_bytecode=True
os.environ['MUJOCO_GL']='disable'
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as E
import mujoco as mj
R=Path(__file__).parent;source=Path('/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_probe_006/scene');S=R/'scene';S.mkdir(exist_ok=True);X=E.parse(source/'robot.xml');eq=X.getroot().find('equality')
for e in list(eq):
 if e.get('name')=='stationary_local_torso_fixture':eq.remove(e)
assert len(eq)==3
X.write(S/'robot.xml')
for n in ['physical_contract.json','actuator_contract.json']:(S/n).write_bytes((source/n).read_bytes())
m=mj.MjModel.from_xml_path(str(S/'robot.xml'));d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);mj.mj_forward(m,d);F=Path('/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/cart_drag_probe_006/full_body_readiness');program=json.loads((F/'next_trial_input.json').read_text());C=json.loads((S/'physical_contract.json').read_text());qi=[int(m.joint(n).qposadr[0]) for n in C['joint_order']]
for n,v in zip(C['joint_order'],program['times_and_targets'][0]['q_target18_rad']):d.qpos[int(m.joint(n).qposadr[0])]=v
for j in C['passive_linkage_joints']:d.qpos[int(m.joint(j['name']).qposadr[0])]=j['mimic_multiplier']*d.qpos[int(m.joint(j['mimic_joint']).qposadr[0])]+j['mimic_offset_rad']
mj.mj_forward(m,d);bar=m.geom('cart_bite_bar').id;pre={'physics_steps':0,'mode':'LOADED_FREE_ROOT_TRUE_CART_AND_FREE20KG_CARGO','neq':m.neq,'equalities':[m.equality(i).name for i in range(m.neq)],'free_joints':[m.joint(i).name for i in range(m.njnt) if m.jnt_type[i]==int(mj.mjtJoint.mjJNT_FREE)],'cargo_mass_kg':float(m.body_mass[m.body('probe_cargo').id]),'total_mass_kg':float(sum(m.body_mass)),'cold_rod_world':d.geom_xpos[bar].tolist(),'cold_root':d.qpos[:7].tolist(),'cold_qpos':d.qpos.tolist(),'wheel_frictionloss_Nm':[float(m.dof_frictionloss[int(m.joint(i).dofadr[0])]) for i in range(m.njnt) if m.joint(i).name.startswith('cart_wheel_spin')],'pitch_stiffness':float(m.jnt_stiffness[m.joint('cart_pitch_hinge').id]),'pitch_qpos_springref':float(m.qpos_spring[int(m.joint('cart_pitch_hinge').qposadr[0])]),'initial_contacts':[{'geom1':m.geom(c.geom1).name,'geom2':m.geom(c.geom2).name,'distance_m':float(c.dist)} for c in d.contact],'controller':'007 second L1 static bias + same leg/body20mm sequence; 008 new wrapper cold real rod anchor+plannedworld20mmback, actualrod monitoring only; jawtarget0 develops physical clamp through original slew.','cold_holding_only_not_complete_active_grasp':True}
assert m.neq==3 and pre['cargo_mass_kg']==20 and len(pre['free_joints'])==3
(R/'scene_preflight.json').write_text(json.dumps(pre,indent=2));print('READY_NO_STEPS',json.dumps({k:v for k,v in pre.items() if k not in ['cold_qpos','initial_contacts']}))
