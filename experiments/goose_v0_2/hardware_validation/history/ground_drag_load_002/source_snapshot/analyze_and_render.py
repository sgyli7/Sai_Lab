"""Independent recorded-state audit, ground-contact qualification, CPU1x replay."""
from pathlib import Path
import os, sys, time, datetime, resource, signal, atexit, json, gzip, hashlib
ROOT=Path(__file__).resolve().parent;START=time.time();A=json.loads((ROOT/'acceptance.json').read_text());DEADLINE=datetime.datetime.fromisoformat(A['absolute_deadline_UTC']).timestamp()
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','LP_NUM_THREADS']:os.environ[k]='1'
os.environ.update(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='osmesa',LIBGL_ALWAYS_SOFTWARE='1',GALLIUM_DRIVER='softpipe')
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
def timeout(*_):raise TimeoutError('Ground drag absolute deadline')
signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(DEADLINE-time.time()-45)))
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
assert mj.__version__=='3.10.0'
PKG=Path((ROOT/'isolated_model_root.txt').read_text().strip());C=json.loads((PKG/'physical_contract.json').read_text());DR=json.loads((PKG/'actuator_contract.json').read_text());PLAN=json.loads((ROOT/'kinematic_plan.json').read_text());COUNTS={'physics_integrations':0,'NN':0,'GPU':0,'training':0,'kinematic_queries':0,'replay_forward_calls':0,'video_frames':0,'rendered_views':0,'max_observed_threads':1};RESULTS=[];VIDEOS=[];PLOTS=[]
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mat(q):
    x=np.empty(9);mj.mju_quat2Mat(x,np.asarray(q));return x.reshape(3,3)
def threadcount():
    n=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('Threads:')));COUNTS['max_observed_threads']=max(COUNTS['max_observed_threads'],n)
def receipt():
    threadcount();r=resource.getrusage(resource.RUSAGE_SELF);ch=resource.getrusage(resource.RUSAGE_CHILDREN)
    write(ROOT/'resource_exit_postprocess.json',{'pid':os.getpid(),'purpose':'read-only analysis/verification and actual-state CPU rendering; NO mj_step','engine':mj.__version__,'finished_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_s_from_acceptance':time.time()-datetime.datetime.fromisoformat(A['received_at_UTC']).timestamp(),'process_wall_s':time.time()-START,'within_budget':time.time()<DEADLINE,'CPU_affinity':sorted(os.sched_getaffinity(0)),'memory_cap_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],'maxRSS_KiB':r.ru_maxrss,'child_maxRSS_KiB':ch.ru_maxrss,'CPU_user_s':r.ru_utime,'CPU_system_s':r.ru_stime,'child_CPU_s':ch.ru_utime+ch.ru_stime,'counts':COUNTS,'OSMesa':'softpipe LP_NUM_THREADS=1','ffmpeg_threads':1,'videos':VIDEOS})
atexit.register(receipt)
def gaps(ts,mask):
    intervals=[];start=None
    for i,(t,b) in enumerate(zip(ts,mask)):
        if b and start is None:start=i
        if start is not None and (not b or i==len(mask)-1):
            end=i-1 if not b else i;intervals.append({'start_s':float(ts[start]-.005),'end_s':float(ts[end]),'duration_s':float((end-start+1)*.005)});start=None
    return intervals
def analyze(n):
    p=ROOT/f'attempt_{n:03d}';s=json.loads((p/'summary.json').read_text());tr=np.load(p/'actual_trajectory.npz');cmd=json.loads((p/'commands50Hz.json').read_text())
    with gzip.open(p/'actual_substeps.jsonl.gz','rt') as f:rr=[json.loads(x) for x in f]
    assert len(rr)==s['actual_steps']==len(tr['time'])-1==len(cmd)*4
    assert sha(p/'actual_trajectory.npz')==s['actual_trajectory_SHA256'] and sha(p/'actual_substeps.jsonl.gz')==s['actual_raw_SHA256'] and sha(p/'actual_scene.mjb')==s['model_MJB_SHA256']
    m=mj.MjModel.from_binary_path(str(p/'actual_scene.mjb'));d=mj.MjData(m);pin=mj.MjData(m);pb=m.body('pilot_payload').id;head=m.body('head_roll').id;body=m.body('torso').id;pj=int(m.joint('pilot_payload_free').qposadr[0]);casepos=np.array([.08,0,-.115]);tip=np.array(PLAN['tip_point_head_local_m']);names=[j['name'] for j in C['joints']];qi=np.array([int(m.joint(x).qposadr[0]) for x in names]);vi=np.array([int(m.joint(x).dofadr[0]) for x in names]);drivevi=m.jnt_dofadr[m.actuator_trnid[:,0]];kp=np.array([j['kp_nm_rad'] for j in C['joints']]);kd=np.array([j['kd_nm_s_rad'] for j in C['joints']]);scale=np.array([j['action_scale_rad'] for j in C['joints']]);neutral=np.array([j['q_neutral_rad'] for j in C['joints']]);ranges=np.array([j['range_rad'] for j in C['joints']]);cont=np.array([j['continuous_design_limit_nm'] for j in C['joints']]);lasttarget=tr['qpos'][0][qi].copy();weight=s['mass_kg']*9.81
    maxerr={'source_state':0.,'relative_position_m':0.,'relative_R':0.,'target_slew_rad':0.,'requested_PD_Nm':0.,'actual_limited_torque_Nm':0.,'payload_world_wrench':0.};data=[];torques=[];saturated=[];speed_ratios=[];reqratios=[];pinerrors={};contact_mus={'jaw':set(),'object_ground':set(),'foot_ground':set()};footstance_start=np.zeros(2);prev_foot=np.zeros(2);active=np.zeros(2,dtype=bool);maxstance_slip=np.zeros(2);warnings=set();maxjoint=0.;sample=[];mincapmargin=1e9
    q=tr['qpos'][0];R=mat(q[pj+3:pj+7]);coldcase=q[pj:pj+3]+R@casepos;coldcom=q[pj:pj+3]+R@m.body_ipos[pb]
    d.qpos[:]=q;mj.mj_kinematics(m,d);mj.mj_comPos(m,d);coldtip=d.xpos[head]+d.xmat[head].reshape(3,3)@tip
    for k,r in enumerate(rr):
        t=r['time_s'];q0=np.array(r['incoming_qpos']);v0=np.array(r['incoming_qvel']);q=np.array(r['qpos']);v=np.array(r['qvel']);tar=np.array(r['target18']);tau=np.array(r['actual_torque18']);requested=kp*(tar-q0[qi])-kd*v0[vi]
        maxerr['source_state']=max(maxerr['source_state'],float(abs(q0-tr['qpos'][k]).max()),float(abs(v0-tr['qvel'][k]).max()),float(abs(q-tr['qpos'][k+1]).max()),float(abs(v-tr['qvel'][k+1]).max()));assert abs(t-(k+1)*.005)<1e-10
        factors=np.array([1. if e['reference_voltage_v']==5. else min(r['bus_voltage_v']/24.,1.) for e in DR['axes']]);slew=np.array([j['speed_limit_rad_s'] for j in C['joints']])*factors;desired=np.clip(neutral+scale*np.clip(cmd[k//4]['action18'],-1,1),ranges[:,0],ranges[:,1]);expected_tar=lasttarget+np.clip(desired-lasttarget,-slew*.005,slew*.005);lasttarget=tar
        caps=cont.copy()
        for i,e in enumerate(DR['axes']):
            curve=np.array(e['selected_torque_speed_curve_rad_s_nm']);curve[:,0]*=factors[i];caps[i]=min(caps[i],float(np.interp(abs(v0[drivevi[i]]),curve[:,0],curve[:,1],right=0)))
        limited=np.clip(requested,-caps,caps);positive=limited*v0[drivevi]>0;power=float((limited[positive]*v0[drivevi][positive]).sum())
        if power>350:limited[positive]*=350/power
        maxerr['target_slew_rad']=max(maxerr['target_slew_rad'],float(abs(expected_tar-tar).max()));maxerr['requested_PD_Nm']=max(maxerr['requested_PD_Nm'],float(abs(requested-r['requested_PD18']).max()));maxerr['actual_limited_torque_Nm']=max(maxerr['actual_limited_torque_Nm'],float(abs(limited-tau).max()),float(abs(caps-r['torque_speed_caps18']).max()));mincapmargin=min(mincapmargin,float(np.min(caps-abs(tau))));assert r['external_applied_forces_zero'] and max(r['positive_mechanical_power_incoming_w'],r['positive_mechanical_power_after_w'])<=350+1e-8
        d.qpos[:]=q;mj.mj_kinematics(m,d);mj.mj_comPos(m,d);pin.qpos[:]=q0;mj.mj_kinematics(m,pin);mj.mj_comPos(m,pin);COUNTS['kinematic_queries']+=2;H=d.xmat[head].reshape(3,3);rel=H.T@(d.xpos[pb]-d.xpos[head]);relR=H.T@d.xmat[pb].reshape(3,3);maxerr['relative_position_m']=max(maxerr['relative_position_m'],float(abs(rel-r['relative_payload_head_position_m']).max()));maxerr['relative_R']=max(maxerr['relative_R'],float(abs(relR-r['relative_payload_head_R']).max()))
        Rp=mat(q[pj+3:pj+7]);case=q[pj:pj+3]+Rp@casepos;com=q[pj:pj+3]+Rp@m.body_ipos[pb];tcp=d.xpos[head]+H@tip;Rt=d.xmat[body].reshape(3,3);pitch=float(np.arctan2(Rt[0,2],Rt[0,0])*180/np.pi);roll=float(np.arctan2(Rt[2,1],Rt[2,2])*180/np.pi);w=np.zeros(6);cop_num=0.;cop_den=0.;jaw_mu=[];floor_geom=False;floor_positive=False;maxcone=0.
        for c in r['contacts']:
            a,b=c['geom1'],c['geom2'];F6=np.array(c['actual_force_contact_6D']);frame=np.array(c['frame_rows_world']).reshape(3,3);ap=a.startswith('pilot_');bp=b.startswith('pilot_');onpayload=ap or bp
            if onpayload:
                sign=1 if bp else -1;F=sign*frame.T@F6[:3];T=sign*frame.T@F6[3:]+np.cross(np.array(c['point_world_m'])-pin.xipos[pb],F);w+=np.r_[F,T];other=a if bp else b
                if other=='ground':
                    contact_mus['object_ground'].add(tuple(c['effective_friction5']));floor_geom=floor_geom or c['distance_m']<=0;floor_positive=floor_positive or F6[0]>1e-6
                elif other in ['upper_bill_envelope','lower_bill_envelope']:
                    contact_mus['jaw'].add(tuple(c['effective_friction5']))
                    if F6[0]>1e-6:maxcone=max(maxcone,float(np.linalg.norm(F6[1:3]/np.array(c['effective_friction5'][:2]))/F6[0]))
            elif 'ground' in [a,b] and any(x in [a,b] for x in ['right_actual_sole','left_actual_sole']):
                contact_mus['foot_ground'].add(tuple(c['effective_friction5']));cop_num+=F6[0]*c['point_world_m'][0];cop_den+=F6[0]
        maxerr['payload_world_wrench']=max(maxerr['payload_world_wrench'],float(abs(w-r['all_payload_contact_wrench_about_COM_world']).max()))
        footcum=np.array(r['foot_contact_slip_integral_m']);on=np.array(r['foot_stance_on'])
        for j in range(2):
            if on[j] and not active[j]:footstance_start[j]=prev_foot[j]
            if on[j]:maxstance_slip[j]=max(maxstance_slip[j],footcum[j]-footstance_start[j])
        active=on;prev_foot=footcum
        for eq in r['equalities']:pinerrors[eq['name']]=max(pinerrors.get(eq['name'],0.),max([abs(x) for x in eq['position_residual']],default=0.))
        warnings.update(r['warnings']);maxjoint=max(maxjoint,r['joint_range_max_violation_rad']);JW=np.array(r['jaw_wrenches_about_COM_world']).sum(0);saturated.append(abs(requested)>caps+1e-8);torques.append(tau);reqratios.append(np.divide(abs(requested),caps,out=np.zeros(18),where=caps>1e-12));speed_ratios.append(abs(v[drivevi])/slew)
        data.append([t,coldcase[0]-case[0],coldcom[0]-com[0],r['object_backwards_from_cold_m'],r['torso_backwards_from_cold_m'],coldtip[0]-tcp[0],r['object_ground_normal_N'],JW[0],JW[2],pitch,roll,r['torso_up_cosine'],r['payload_lowest_z_m'],float(floor_geom),float(floor_positive),r['conservative_grasp_X_margin_m'],rel[0],rel[2],maxcone,d.subtree_com[body,0],0. if cop_den==0 else cop_num/cop_den,r['foot_ground_normal_N'][0],r['foot_ground_normal_N'][1]])
        if any(abs(t-x)<1e-8 for x in [1,2,3,3.5,4,5,6]):sample.append({'completed_time_s':t,'incoming_contact_time_s':r['incoming_time_s'],'case_center_world_m':case.tolist(),'payload_COM_world_m':com.tolist(),'actual_mouth_TCP_world_m':tcp.tolist(),'relative_rod_head_m':rel.tolist(),'actual_torque18':tau.tolist(),'requested_PD18':requested.tolist(),'caps18':caps.tolist(),'torso_pitch_deg':pitch,'torso_roll_deg':roll,'jaw_actual_wrench_about_COM_world':JW.tolist(),'ground_N':r['object_ground_normal_N'],'equalities':r['equalities']})
    assert max(maxerr.values())<1e-10,maxerr
    torques=np.asarray(torques)
    z=np.array(data);ts=z[:,0];loaded=ts>1+1e-10;pull=(ts>1+1e-10)&(ts<=3+1e-10);stop=ts>3+1e-10;ix=int(np.argmin(abs(ts-1)));case_pull=z[:,1]-z[ix,1];com_pull=z[:,2]-z[ix,2];gaps_force=gaps(ts,loaded&(z[:,6]<=.01*weight));gaps_zero=gaps(ts,loaded&(z[:,14]<.5));gaps_geom=gaps(ts,loaded&(z[:,13]<.5));maxgap=max([x['duration_s'] for x in gaps_force],default=0.);ground_duty=float(np.mean(z[loaded,6]>.01*weight));longfloat=max([x['duration_s'] for x in gaps_geom],default=0.)
    phases={}
    for name,mask in [('capture',ts<=1+1e-10),('pull_1_to3s',pull),('stop_after3s',stop),('all_loaded',loaded)]:
        if not np.any(mask):continue
        ids=np.where(mask)[0];phases[name]={'observed_s':len(ids)*.005,'mean_ground_normal_N':float(z[mask,6].mean()),'mean_ground_normal_over_mg':float(z[mask,6].mean()/weight),'mean_downward_mouth_force_N':float(np.maximum(-z[mask,8],0).mean()),'mean_mouth_backward_force_N':float((-z[mask,7]).mean()),'mean_nominal_mu_times_actual_ground_N':float(.3*z[mask,6].mean()),'ground_force_above1pct_mg_duty':float(np.mean(z[mask,6]>.01*weight)),'case_center_backward_net_m':float(z[ids[-1],1]-z[max(ids[0]-1,0),1]),'drive_clipped_duty18':np.array(saturated)[mask].mean(0).tolist()}
    classification='LOCAL_SHORT_STROKE_OBSERVED' if s['mode']=='LOCAL_TORSO_FIXED_NECK_PULL' else 'FREE_BASE_FAILED_NOT_QUALIFIED'
    if s['mode']=='LOCAL_TORSO_FIXED_NECK_PULL' and (maxgap>.1+1e-9 or ground_duty<.95):classification='LOCAL_MOTION_OBSERVED_GROUND_CONTACT_GATE_NOT_MET'
    if s['mode']=='LOCAL_TORSO_FIXED_NECK_PULL' and case_pull[-1]<.05:classification+='__LESS_THAN50MM_NET_AFTER_STOP'
    forcegating={'original_mean_only_label':s['ground_support_classification'],'authoritative_full_classification':classification,'mean_only_label_is_not_qualification':True,'actual_force_support_duty':ground_duty,'max_force_support_loss_s':maxgap,'max_zero_positive_floor_contact_loss_s':max([x['duration_s'] for x in gaps_zero],default=0.),'max_geometric_floor_contact_loss_s':longfloat,'force_loss_intervals':gaps_force,'zero_contact_loss_intervals':gaps_zero,'geometric_contact_loss_intervals':gaps_geom,'nominal_flat_average_load_condition':phases['all_loaded']['mean_ground_normal_over_mg']>=.9,'partial_contact95pct_and_max_gap100ms_gate':ground_duty>=.95 and maxgap<=.1+1e-9,'formal_whole_robot_drag':'NOT_PASSED','continuous60s':'NOT_EXECUTED'}
    out={'attempt':n,'mode':s['mode'],'mass_kg':s['mass_kg'],'classification':classification,'raw_summary_preserved':True,'source_raw_SHA256':sha(p/'actual_substeps.jsonl.gz'),'independent_reconstruction_max_errors':maxerr,'rows_checked':len(rr),'case_center_net_backward_m':float(z[-1,1]),'case_center_net_backward_from_pull_entry_m':float(case_pull[-1]),'case_center_max_backward_from_pull_entry_m':float(case_pull[loaded].max()),'full_COM_net_backward_from_pull_entry_m':float(com_pull[-1]),'handle_origin_net_backward_from_pull_entry_m':s['final_object_backwards_from_pull_entry_m'],'case_net_return_forward_after3s_m':None if not np.any(stop) else float(z[np.argmin(abs(ts-3)),1]-z[-1,1]),'max_torso_backward_m':float(z[:,4].max()),'max_abs_torso_pitch_deg':float(abs(z[:,9]).max()),'max_abs_torso_roll_deg':float(abs(z[:,10]).max()),'ground_contact_gate':forcegating,'phases':phases,'actual_effective_friction5':{k:[list(x) for x in sorted(v)] for k,v in contact_mus.items()},'drive_axis_order':names,'actual_torque_peak18_Nm':np.max(abs(torques),0).tolist(),'actual_requested_torque_over_curve_peak18':np.max(reqratios,0).tolist(),'curve_clipping_duty18':np.array(saturated).mean(0).tolist(),'selected_slew_speed_exceeded_ratio18_peak':np.max(speed_ratios,0).tolist(),'positive_mechanical_power_peak_w':s['max_positive_mechanical_power_w'],'max_per_stance_force_weighted_contact_slip_integral_m':maxstance_slip.tolist(),'foot_center_excursion_is_not_same_as_contact_slip':True,'max_internal_and_fixture_eq_coordinate_residuals':pinerrors,'max_joint_range_violation_rad':maxjoint,'warnings_present':any(warnings),'selected_state_force_samples':sample,'limitations':'Exploratory specific cold grasp and prescribed short IK request. No formal reliability, whole-body distance/endurance or maximum load qualification. Contact forces are actual records; this is read-only state/drive/wrench reconstruction, not a force rerun.'}
    write(p/'analysis.json',out);write(p/'ground_contact_correction.json',forcegating);np.savez_compressed(p/'derived_trace.npz',data=z,columns=np.array(['time_s','case_backward_m','COM_backward_m','handle_backward_m','torso_backward_m','TCP_backward_m','ground_N','jaw_Fx_N','jaw_Fz_N','torso_pitch_deg','torso_roll_deg','up_cosine','object_lowest_z_m','geometric_floor_contact','positive_floor_contact','grasp_margin_m','rod_head_x_m','rod_head_z_m','max_jaw_sliding_cone_utilization','robot_COM_world_x_m','feet_CoP_world_x_m','right_foot_N','left_foot_N']))
    RESULTS.append(out);PLOTS.append((s,z));print('EVENT analyzed',n,'mass',s['mass_kg'],'case_pull_mm',round(case_pull[-1]*1000,2),'ground_duty',round(ground_duty,4),'longest_loss_s',maxgap,'class',classification,flush=True)
def render(n):
    p=ROOT/f'attempt_{n:03d}';s=json.loads((p/'summary.json').read_text());tr=np.load(p/'actual_trajectory.npz');q=tr['qpos'];ts=tr['time'];assert (len(ts)-1)%4==0
    m=mj.MjModel.from_binary_path(str(p/'actual_scene.mjb'));d=mj.MjData(m);renderer=mj.Renderer(m,height=480,width=480)
    cam1=mj.MjvCamera();cam1.lookat[:]=[.13,0,.25];cam1.distance=1.15;cam1.azimuth=115;cam1.elevation=-12
    cam2=mj.MjvCamera();cam2.lookat[:]=[.28,0,.14];cam2.distance=.50;cam2.azimuth=100;cam2.elevation=-10
    outfile=p/'actual_continuous_1x_50fps.mp4';writer=imageio_ffmpeg.write_frames(str(outfile),(960,480),fps=50,codec='libx264',quality=8,macro_block_size=16,ffmpeg_log_level='error',output_params=['-threads','1','-preset','fast','-pix_fmt','yuv420p']);writer.send(None);font=ImageFont.load_default(size=16);indices=list(range(0,len(ts)-1,4));mode='TORSO FIXED / REAL NECK ACTUATION' if s['mode']=='LOCAL_TORSO_FIXED_NECK_PULL' else 'FREE BASE / NO WORLD FIXTURES'
    try:
        for k,idx in enumerate(indices):
            if time.time()>DEADLINE-90:raise TimeoutError('Video closeout reserve')
            d.qpos[:]=q[idx];mj.mj_forward(m,d);COUNTS['replay_forward_calls']+=1;renderer.update_scene(d,camera=cam1);left=renderer.render().copy();renderer.update_scene(d,camera=cam2);right=renderer.render().copy();COUNTS['rendered_views']+=2
            frame=Image.fromarray(np.concatenate([left,right],axis=1));draw=ImageDraw.Draw(frame);draw.rectangle((0,0,960,62),fill=(12,12,16));draw.text((8,5),f'SEALED04 | FULL {s["mass_kg"]:g}kg PAYLOAD | attempt {n:03d} | actual t={ts[idx]:.3f}s | 1x',font=font,fill='white');draw.text((8,31),mode+' | payload FREE | actual trajectory',font=font,fill=(255,201,71));draw.rectangle((0,450,960,480),fill=(12,12,16));draw.text((8,456),'Whole robot / floor',font=font,fill='white');draw.text((488,456),'Jaw / case / ground',font=font,fill='white');writer.send(np.asarray(frame));COUNTS['video_frames']+=1
            if k==0:frame.save(p/'preview_cold.png')
            if abs(ts[idx]-3.)<1e-8:frame.save(p/'preview_3s.png')
            if k==len(indices)-1:frame.save(p/'preview_final.png')
            if k%100==0:threadcount();print('EVENT render',n,k,'/',len(indices),flush=True)
    finally:writer.close();renderer.close()
    v={'attempt':n,'video_path':str(outfile),'video_SHA256':sha(outfile),'trajectory_SHA256':sha(p/'actual_trajectory.npz'),'model_MJB_SHA256':sha(p/'actual_scene.mjb'),'fps':50,'frames':len(indices),'duration_s':len(indices)/50,'actual_duration_s':float(ts[-1]),'sampling':'exact states every4 actual5ms steps, t0 included/end exclusive, last20ms held as sampled frame; no interpolation/cuts/time scaling','GPU':0,'physics_integrations':0,'renderer':'CPU OSMesa softpipe','ffmpeg_threads':1};write(p/'video_receipt.json',v);VIDEOS.append(v);print('STATUS video',n,str(outfile),flush=True)
def plot():
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for s,z in PLOTS:
        label=f'{s["mass_kg"]:g}kg '+('FREE' if s['mode']=='FREE_BASE_BACKDRAG' else 'LOCAL');t=z[:,0];ix=np.argmin(abs(t-1));ax[0,0].plot(t,(z[:,1]-z[ix,1])*1000,label=label);ax[0,1].plot(t,z[:,6]/(s['mass_kg']*9.81),label=label)
        if s['mode']=='FREE_BASE_BACKDRAG':ax[1,0].plot(t,z[:,4]*1000,label='Torso backward mm');ax[1,0].plot(t,z[:,9],label='Torso pitch deg');ax[1,1].plot(t,z[:,12]*1000,label='Lowest object point mm')
    ax[0,0].set(title='Actual BOX center displacement from pull entry',xlabel='Actual time (s)',ylabel='Backward (mm)');ax[0,0].axhline(50,color='gray',linestyle='--');ax[0,0].legend(fontsize=8)
    ax[0,1].set(title='Actual object ground load / mg (zeros are preserved)',xlabel='Actual time (s)',ylabel='Normal / mg');ax[0,1].axhline(.9,color='gray',linestyle='--');ax[0,1].legend(fontsize=8)
    ax[1,0].set(title='Free-base pilot: retreat followed by posture failure',xlabel='Actual time (s)');ax[1,0].legend(fontsize=8)
    ax[1,1].set(title='Free-base pilot: post-retreat object leaves ground',xlabel='Actual time (s)',ylabel='Lowest point (mm)');ax[1,1].legend(fontsize=8)
    for a in ax.flat:a.axvline(1,color='gray',alpha=.3);a.axvline(3,color='gray',alpha=.3);a.grid(alpha=.2)
    fig.suptitle('Sealed04 ground-drag exploratory evidence / no hardware change / no training',fontsize=12);fig.savefig(ROOT/'ground_drag_diagnostics.png',dpi=150);plt.close(fig)
try:
    for n in range(1,6):analyze(n)
    write(ROOT/'analysis_summary.json',{'physics_integrations':0,'rows_checked':sum(x['rows_checked'] for x in RESULTS),'results':[{k:v for k,v in x.items() if k!='selected_state_force_samples'} for x in RESULTS]})
    plot()
    for n in [2,1,5,4,3]:render(n)
    write(ROOT/'postprocess_status.json',{'status':'completed','counts':COUNTS,'videos':VIDEOS})
except BaseException as e:write(ROOT/'postprocess_status.json',{'status':'failed','error':repr(e),'counts':COUNTS,'videos':VIDEOS});raise
