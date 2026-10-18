"""Recheck recorded physics evidence without stepping dynamics or running a policy."""
from pathlib import Path
import os, sys, json, gzip, resource, time, hashlib
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
START=time.time()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
import numpy as np
import mujoco as mj
assert mj.__version__=='3.10.0'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def angle(A,B):
    q=np.empty(4);mj.mju_mat2Quat(q,(A@B.T).ravel());return float(2*np.arctan2(np.linalg.norm(q[1:]),abs(q[0]))*180/np.pi)
pkg=Path((ROOT/'isolated_model_root.txt').read_text().strip())
C=json.loads((pkg/'physical_contract.json').read_text());DR=json.loads((pkg/'actuator_contract.json').read_text());pre=json.loads((ROOT/'preflight_audit.json').read_text())
J=C['joints'];neutral=np.array([j['q_neutral_rad'] for j in J]);scale=np.array([j['action_scale_rad'] for j in J]);ranges=np.array([j['range_rad'] for j in J]);kp=np.array([j['kp_nm_rad'] for j in J]);kd=np.array([j['kd_nm_s_rad'] for j in J]);cont=np.array([j['continuous_design_limit_nm'] for j in J])
mapping=pre['named_mapping'];qi=np.array([r['logical_q_index'] for r in mapping]);vi=np.array([r['logical_v_index'] for r in mapping]);drivevi=np.array([r['actual_motor_dof'] for r in mapping])
OUT=[];MAX={'source_state_continuity':0.,'post_pose_translation_m':0.,'post_pose_rotation_matrix':0.,'requested_torque_Nm':0.,'limited_and_actual_torque_Nm':0.,'target_slew_rad':0.,'payload_wrench':0.,'positive_normal_N':0.}
for n in range(1,5):
    p=ROOT/f'attempt_{n:03d}';s=json.loads((p/'summary.json').read_text());tr=np.load(p/'actual_trajectory.npz');commands=json.loads((p/'commands50Hz.json').read_text())
    with gzip.open(p/'actual_substeps.jsonl.gz','rt') as f:rr=[json.loads(l) for l in f]
    m=mj.MjModel.from_binary_path(str(p/'actual_scene.mjb'));d=mj.MjData(m);pin=mj.MjData(m);head=m.body('head_roll').id;body=m.body('pilot_payload').id
    def pose(q,d):
        d.qpos[:]=q;mj.mj_kinematics(m,d);mj.mj_comPos(m,d);H=d.xmat[head].reshape(3,3);return H.T@(d.xpos[body]-d.xpos[head]),H.T@d.xmat[body].reshape(3,3)
    coldp,coldR=pose(tr['qpos'][0],d);lasttarget=tr['qpos'][0][qi].copy();truecold=[];holdstart=None;holdp=None;holdR=None;hold=[]
    assert len(rr)==s['actual_steps']==len(tr['time'])-1==len(commands)*4
    assert sha(p/'actual_trajectory.npz')==s['trajectory_SHA256'] and sha(p/'actual_substeps.jsonl.gz')==s['raw_substeps_SHA256'] and sha(p/'actual_scene.mjb')==s['model_MJB_SHA256']
    for k,r in enumerate(rr):
        q0=np.array(r['incoming_qpos']);v0=np.array(r['incoming_qvel']);q=np.array(r['qpos']);v=np.array(r['qvel']);target=np.array(r['target18']);actual=np.array(r['actual_actuator_force18'])
        MAX['source_state_continuity']=max(MAX['source_state_continuity'],float(abs(q0-tr['qpos'][k]).max()),float(abs(v0-tr['qvel'][k]).max()),float(abs(q-tr['qpos'][k+1]).max()),float(abs(v-tr['qvel'][k+1]).max()))
        assert abs(r['time_s']-(k+1)*.005)<1e-10 and abs(r['incoming_time_s']-k*.005)<1e-10
        factor=np.array([1. if a['reference_voltage_v']==5. else min(r['bus_voltage_v']/24.,1.) for a in DR['axes']]);slew=np.array([j['speed_limit_rad_s'] for j in J])*factor
        desired=np.clip(neutral+scale*np.clip(commands[k//4]['action18'],-1,1),ranges[:,0],ranges[:,1]);expectedtarget=lasttarget+np.clip(desired-lasttarget,-slew*.005,slew*.005)
        MAX['target_slew_rad']=max(MAX['target_slew_rad'],float(abs(target-expectedtarget).max()));lasttarget=target
        requested=kp*(target-q0[qi])-kd*v0[vi];caps=cont.copy()
        for i,a in enumerate(DR['axes']):
            curve=np.array(a['selected_torque_speed_curve_rad_s_nm']);curve[:,0]*=factor[i];caps[i]=min(caps[i],float(np.interp(abs(v0[drivevi[i]]),curve[:,0],curve[:,1],right=0)))
        limited=np.clip(requested,-caps,caps);positive=limited*v0[drivevi]>0;power=float(sum(limited[positive]*v0[drivevi][positive]));
        if power>350:limited[positive]*=350/power
        MAX['requested_torque_Nm']=max(MAX['requested_torque_Nm'],float(abs(requested-r['requested_PD_torque18']).max()));MAX['limited_and_actual_torque_Nm']=max(MAX['limited_and_actual_torque_Nm'],float(abs(limited-actual).max()),float(abs(caps-r['continuous_torque_speed_caps18']).max()))
        assert np.max(abs(actual)-caps)<1e-9 and np.maximum(actual*v[drivevi],0).sum()<=350+1e-8
        rel,R=pose(q,d);MAX['post_pose_translation_m']=max(MAX['post_pose_translation_m'],float(abs(rel-r['relative_payload_head_translation_m']).max()));MAX['post_pose_rotation_matrix']=max(MAX['post_pose_rotation_matrix'],float(abs(R-r['relative_payload_head_R']).max()))
        truecold.append({'time_s':r['time_s'],'translation_m':float(np.linalg.norm(rel-coldp)),'rotation_deg':angle(R,coldR)})
        pose(q0,pin);wrench=np.zeros(6);normals=np.zeros(2)
        for c in r['contacts']:
            a,b=c['geom1'],c['geom2'];ap=a.startswith('pilot_');bp=b.startswith('pilot_')
            if not (ap or bp):continue
            f=np.array(c['actual_force_contact_frame_6D']);frame=np.array(c['frame_rows_world']).reshape(3,3);sign=1 if bp else -1;F=sign*frame.T@f[:3];T=sign*frame.T@f[3:]+np.cross(np.array(c['pos_world_m'])-pin.xipos[body],F);wrench+=np.r_[F,T]
            other=a if bp else b
            if other in ['upper_bill_envelope','lower_bill_envelope']:normals[['upper_bill_envelope','lower_bill_envelope'].index(other)]+=max(0.,f[0])
        MAX['payload_wrench']=max(MAX['payload_wrench'],float(abs(wrench-r['actual_payload_contact_wrench_about_incoming_COM_world']).max()));MAX['positive_normal_N']=max(MAX['positive_normal_N'],float(abs(normals-r['jaw_positive_normal_N']).max()))
        if holdstart is None and min(normals)>1e-6 and target[5]<=1e-10:holdstart=r['time_s'];holdp=rel.copy();holdR=R.copy()
        if holdstart is not None:
            elapsed=r['time_s']-holdstart;drift=float(np.linalg.norm(rel-holdp));ang=angle(R,holdR);failure=(drift>.002 or ang>2 or min(normals)<=1e-6 or bool(r['extra_payload_positive_contacts']));hold.append({'time_s':r['time_s'],'elapsed_s':elapsed,'drift_m':drift,'angle_deg':ang,'failed':failure})
    window=[r for r in hold if r['elapsed_s']<=5+1e-10];firstfail=next((r for r in hold if r['failed']),None)
    if s['hold5s'] is not None:
        assert window[-1]['elapsed_s']>=5-1e-10
        assert abs(max(r['drift_m'] for r in window)-s['hold5s']['max_relative_translation_m'])<1e-12
        assert abs(max(r['angle_deg'] for r in window)-s['hold5s']['max_relative_rotation_deg'])<1e-10
        assert s['hold5s']['passed']==(firstfail is None)
    if firstfail is not None:assert abs(firstfail['time_s']-s['hold_first_failure']['time_s'])<1e-10
    correction={'original_label_issue':'relative_from_cold fields in original raw/summary used the first completed step at5ms as reference, not the saved t0 birth; original files retained','correct_true_cold_final':truecold[-1],'correct_true_cold_max_translation_m':max(x['translation_m'] for x in truecold),'correct_true_cold_max_rotation_deg':max(x['rotation_deg'] for x in truecold),'true_cold_first_30mm_crossing':next((x for x in truecold if x['translation_m']>.03),None),'original_safety_latch_reference_time_s':.005,'hold_reference_affected':False}
    (p/'true_cold_reference_correction.json').write_text(json.dumps(correction,indent=2)+'\n')
    OUT.append({'attempt':n,'rows':len(rr),'commands':len(commands),'hashes_match':True,'hold5s_recomputed_from_actual_pose':s['hold5s'],'hold_first_failure_recomputed':firstfail,'true_cold_reference_correction':correction})
assert max(MAX.values())<1e-10,MAX
r=resource.getrusage(resource.RUSAGE_SELF)
out={'status':'PASS_RECORDED_DATA_CONSISTENCY_ONLY','engine':mj.__version__,'physics_integrations':0,'training':0,'GPU':0,'rows_checked':sum(x['rows'] for x in OUT),'max_independent_reconstruction_errors':MAX,'attempts':OUT,'limitations':'No new dynamics or causal A/B. Solved contact forces are actual recorded values, not independently rerun. A detected5ms cold-reference label issue is explicitly corrected separately; hold references and fail results are unchanged.','resources':{'process_wall_s':time.time()-START,'CPU_affinity':sorted(os.sched_getaffinity(0)),'max_memory_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],'maxRSS_KiB':r.ru_maxrss,'CPU_user_s':r.ru_utime,'CPU_system_s':r.ru_stime}}
(ROOT/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='attempts'},indent=2))
