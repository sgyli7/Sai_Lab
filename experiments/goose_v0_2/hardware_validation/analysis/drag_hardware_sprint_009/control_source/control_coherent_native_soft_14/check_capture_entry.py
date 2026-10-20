from pathlib import Path
import sys,json,numpy as np,mujoco as mj
R=Path(__file__).parent;sys.path.insert(0,str(R));import teacher_mit_profile as mit
from coherent_plan_sampler import CoherentPlan
from aligned_capture_plan import AlignedCapturePlan
from capture_plan_entry import CapturePlanEntry
from motion_stop_reference import MotionStopReference
from teacher_drive_segment import make_packet,sample_packet,compute_drive_tick
from world_neck_qonly import command as neck
mj.mj_step=lambda *a,**k:(_ for _ in ()).throw(AssertionError('NO PHYSICS'))
e=mit.TeacherMITDrive(R/'scene');m=e.model;ctx=json.loads((R/'actual_capture_context/capture_handoff_context.json').read_text());sm=json.loads((R/'actual_capture_context/summary.json').read_text());base=CoherentPlan(m,R/'complete_reference.npz',e.qidx);aligned=AlignedCapturePlan(base,ctx);rebuilt=CoherentPlan(m,R/'readbacks/generated_00/observed_capture_reference.npz',e.qidx);entry=CapturePlanEntry(rebuilt,ctx,sm['foot_material_slip_integral_m']);curve=MotionStopReference(entry);actual=mj.MjData(m);actual.qpos[:]=ctx['actual_qpos'];actual.qvel[:]=ctx['actual_qvel'];mj.mj_forward(m,actual);anchors=np.array([actual.geom_xpos[m.geom(s+'_actual_sole').id] for s in ['right','left']]);rows=[]
for t in [0.,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5]:
 r=curve.sample(t);d=mj.MjData(m);d.qpos[:]=r['qpos'];mj.mj_forward(m,d);feet=np.array([d.geom_xpos[m.geom(s+'_actual_sole').id] for s in ['right','left']]);rows.append(dict(t=t,sole_center_error_R_L_m=(feet-anchors).tolist(),root_reference=r['qpos'][:7].tolist(),root_reference_acceleration6=r['qacc'][:6].tolist(),max_eqres=float(max(abs(d.efc_pos[np.where(d.efc_type[:d.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY))[0]]),default=0)),body_reference_only_not_live_state=True))
packet=make_packet(curve,0.);front,frontmeta=neck(e,actual.qpos,0.,aligned,entry.adapter_target[:6])
for sg in packet['segments']:
 cc=np.array(sg['bernstein_q18']);cc[:,:6]=front;sg['bernstein_q18']=cc.tolist()
held=ctx['held_motor_command'];ff=np.array(held['tau_ff18_Nm']);target=np.array(ctx['original_env_target']);nexttarget,total,details=compute_drive_tick(e,packet,0.,target,actual.qpos,actual.qvel,ff);qdold=np.array(held['qd_ref18_rad_s']);old=e.limit_torque(e.kp*(target-actual.qpos[e.qidx])+e.kd*(qdold-actual.qvel[e.vidx])+ff,actual.qvel[e.drive_vidx]);out=dict(physics_steps=0,alignment=aligned.report,observed_XY_anchor_rebuilt=True,entry_rows=rows,maximum_sole_center_drift_R_L_m=np.max(np.linalg.norm(np.array([r['sole_center_error_R_L_m'] for r in rows]),axis=2),axis=0).tolist(),maximum_root_reference_acceleration6=np.max(abs(np.array([r['root_reference_acceleration6'] for r in rows])),axis=0).tolist(),first_entry_target_difference=float(max(abs(nexttarget-target))),first_entry_total_tau_difference=float(max(abs(total-old))),first_entry_qd_difference=float(max(abs(np.array(details['dispatch_qd18'])-qdold))),fixed_rod_anchor_not_actualbody_chasing=aligned.rod_anchor.tolist(),cumulative_slip_R_L_preserved=sm['foot_material_slip_integral_m'],first_root_reference_velocity_difference=float(max(abs(curve.sample(0.)['qvel'][:6]-np.array(ctx['actual_qvel'])[:6]))),live_state_untouched=True,entry_duration_s=.5,entry_within_4p90625_motion_clock=True,geometry_alignment_not_dynamic_plant_certificate=True)
(R/'capture_alignment_entry_readback.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='entry_rows'},indent=2))
