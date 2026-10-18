"""Read-only trajectory analysis and CPU video replay. No dynamics integration."""
from pathlib import Path
import os, sys, time, json, gzip, hashlib, resource, datetime, signal, atexit
ROOT=Path(__file__).resolve().parent
START=time.time()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','LP_NUM_THREADS'):
    os.environ[key]='1'
os.environ.update(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='osmesa',LIBGL_ALWAYS_SOFTWARE='1',GALLIUM_DRIVER='softpipe')
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
ACCEPT=json.loads((ROOT/'acceptance.json').read_text())
DEADLINE=datetime.datetime.fromisoformat(ACCEPT['absolute_deadline_UTC']).timestamp()
def timeout(*_):raise TimeoutError('Accepted absolute deadline')
signal.signal(signal.SIGALRM,timeout)
signal.alarm(max(1,int(DEADLINE-time.time()-30)))
import numpy as np
import mujoco as mj
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
assert mj.__version__=='3.10.0'
COUNTS={'physics_integrations':0,'replay_mj_forward_calls':0,'rendered_views':0,'video_frames':0,'max_observed_threads':1}
RENDERS=[]
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def threads():
    n=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('Threads:')))
    COUNTS['max_observed_threads']=max(COUNTS['max_observed_threads'],n)
def receipt():
    threads();r=resource.getrusage(resource.RUSAGE_SELF);ch=resource.getrusage(resource.RUSAGE_CHILDREN)
    write(ROOT/'resource_exit_postprocess.json',{'pid':os.getpid(),'engine':mj.__version__,'purpose':'read-only analysis and actual-state replay; no mj_step, no dynamics','finished_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'process_wall_s':time.time()-START,'wall_s_from_acceptance':time.time()-datetime.datetime.fromisoformat(ACCEPT['received_at_UTC']).timestamp(),'within_wall_budget':time.time()<=DEADLINE,'CPU_affinity':sorted(os.sched_getaffinity(0)),'memory_limit_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],'self_maxRSS_KiB':r.ru_maxrss,'child_maxRSS_KiB':ch.ru_maxrss,'CPU_user_s':r.ru_utime,'CPU_system_s':r.ru_stime,'children_CPU_s':ch.ru_utime+ch.ru_stime,'counters':COUNTS,'GPU':0,'NN':0,'training_updates':0,'software_renderer':'OSMesa softpipe LP_NUM_THREADS=1','ffmpeg_threads':1,'renders':RENDERS})
atexit.register(receipt)
def rows(p):
    with gzip.open(p/'actual_substeps.jsonl.gz','rt') as f:return [json.loads(l) for l in f]
def analysis(n):
    p=ROOT/f'attempt_{n:03d}';s=json.loads((p/'summary.json').read_text());rr=rows(p)
    m=mj.MjModel.from_binary_path(str(p/'actual_scene.mjb'));d=mj.MjData(m)
    pb=m.body('pilot_payload').id;qh=int(m.joint('pilot_payload_free').qposadr[0]);jaw_names=['upper_bill_envelope','lower_bill_envelope']
    jawpen=0.;groundpen=0.;otherpen=0.;samples=[];allfixture=[];maxjoint=0.;maxcap=0.;maxforce=0.
    for r in rr:
        maxjoint=max(maxjoint,r['joint_range_max_violation_rad']);maxcap=max(maxcap,max(abs(x) for x in r['actual_actuator_force18']))
        for c in r['contacts']:
            names=[c['geom1'],c['geom2']]
            if not any(x.startswith('pilot_') for x in names):continue
            if any(x in jaw_names for x in names):jawpen=max(jawpen,-c['dist_m'])
            elif any('ground' in x or 'floor' in x for x in names):groundpen=max(groundpen,-c['dist_m'])
            else:otherpen=max(otherpen,-c['dist_m'])
        allfixture.extend(r['fixture_reactions'])
    requested_times=[.03,.095,.105,.15,.215,.27,.28,.435,.49,1.,3.,5.2,rr[-1]['time_s']]
    for t in sorted(set(requested_times)):
        if t>rr[-1]['time_s']+1e-8:continue
        r=min(rr,key=lambda z:abs(z['time_s']-t));q=np.array(r['incoming_qpos']);quat=q[qh+3:qh+7];R=np.empty(9);mj.mju_quat2Mat(R,quat);R=R.reshape(3,3)
        moment=np.cross(R@m.body_ipos[pb],np.array([0.,0.,-.981]));axis=R[:,1]
        contacts=[]
        for c in r['contacts']:
            names=[c['geom1'],c['geom2']]
            if not(any(x.startswith('pilot_') for x in names) and any(x in jaw_names for x in names)):continue
            f=np.array(c['actual_force_contact_frame_6D']);cone=None if f[0]<=1e-6 else float(np.linalg.norm(f[1:3]/np.array(c['effective_friction5'][:2]))/f[0])
            contacts.append({'geoms':names,'position_world_m':c['pos_world_m'],'frame_rows_world':c['frame_rows_world'],'actual_force_contact_frame_6D':f.tolist(),'sliding_elliptic_cone_utilization':cone,'dim':c['dim'],'distance_m':c['dist_m']})
        samples.append({'time_s':r['time_s'],'incoming_contact_time_s':r['incoming_time_s'],'hold_elapsed_s':r['hold_elapsed_s'],'jaw_positive_normal_N':r['jaw_positive_normal_N'],'head_relative_position_m':r['relative_payload_head_translation_m'],'head_relative_rotation_matrix':r['relative_payload_head_R'],'hold_relative_drift_m':r['hold_relative_drift_m'],'hold_relative_angle_deg':r['hold_relative_angle_deg'],'cold_rotation_deg':r['relative_from_cold_rotation_deg'],'gravity_moment_about_rod_origin_world_Nm':moment.tolist(),'gravity_moment_about_rod_axis_Nm':float(moment@axis),'payload_free_joint_velocity6':r['qvel'][-6:],'contacts':contacts,'jaw_actual_wrench_about_COM_world':r['actual_jaw_wrenches_about_incoming_COM_world'],'fixture_reactions':r['fixture_reactions']})
    worst=max(rr,key=lambda r:r['internal_mechanism_eq_max_abs_position_error'] or 0.)
    d.qpos[:]=worst['incoming_qpos'];d.qvel[:]=worst['incoming_qvel'];mj.mj_forward(m,d);COUNTS['replay_mj_forward_calls']+=1
    internal=[]
    for i in range(3):
        idx=np.where((d.efc_type[:d.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY))&(d.efc_id[:d.nefc]==i))[0]
        internal.append({'equality':m.equality(i).name,'position_residual':d.efc_pos[idx].tolist()})
    hold=[r for r in rr if r['hold_elapsed_s'] is not None]
    drift_vector=None if not hold else (np.array(hold[-1]['relative_payload_head_translation_m'])-hold[0]['relative_payload_head_translation_m']).tolist()
    out={'attempt':n,'summary_source_SHA256':sha(p/'summary.json'),'raw_source_SHA256':sha(p/'actual_substeps.jsonl.gz'),'contact_force_state_alignment':'contact geometry/solved forces are evaluated at incoming qpos/time; post-integration qpos and relative pose are the completed step; both are preserved','max_jaw_payload_penetration_m':max(0.,jawpen),'max_ground_payload_penetration_m':max(0.,groundpen),'max_other_payload_penetration_m':max(0.,otherpen),'hold_endpoint_translation_vector_head_m':drift_vector,'all_external_applied_forces_zero':all(r['external_applied_force_zero'] for r in rr),'max_actual_axis_torque_Nm':maxcap,'max_joint_range_violation_rad':maxjoint,'max_fixture_wrench_reconstruction_error':max(f['wrench_reconstruction_max_error'] for f in allfixture),'fixture_reaction_max_force_norm_N':max(float(np.linalg.norm(f['recovered_world_wrench_about_body_origin'][:3])) for f in allfixture),'fixture_reaction_max_torque_norm_Nm':max(float(np.linalg.norm(f['recovered_world_wrench_about_body_origin'][3:])) for f in allfixture),'worst_recorded_internal_eq_error_time_s':worst['incoming_time_s'],'worst_recorded_internal_eq_error':worst['internal_mechanism_eq_max_abs_position_error'],'internal_eq_position_reconstruction_at_worst_state':internal,'reconstruction_note':'mj_forward geometry/position residual only; recalculated forces are NOT substituted for recorded actual forces','sampled_actual_contact_and_fixture_records':samples}
    write(p/'hardware_evidence.json',out);print('EVENT analysis',n,'jaw_pen_mm',round(jawpen*1000,4),'drift_vector_mm',None if drift_vector is None else [round(v*1000,3) for v in drift_vector],flush=True)
def render(n):
    p=ROOT/f'attempt_{n:03d}';tr=np.load(p/'actual_trajectory.npz');ts=tr['time'];qs=tr['qpos'];assert len(ts)==len(qs)
    assert abs(ts[0])<1e-12 and np.allclose(np.diff(ts),.005,atol=1e-10)
    assert (len(ts)-1)%4==0
    m=mj.MjModel.from_binary_path(str(p/'actual_scene.mjb'));d=mj.MjData(m)
    cameraA=mj.MjvCamera();cameraA.lookat[:]=[.13,0.,.36];cameraA.distance=1.30;cameraA.azimuth=125.;cameraA.elevation=-14.
    cameraB=mj.MjvCamera();cameraB.lookat[:]=[.23,0.,.51];cameraB.distance=.48;cameraB.azimuth=95.;cameraB.elevation=-8.
    renderer=mj.Renderer(m,height=480,width=480)
    outfile=p/'actual_continuous_1x_50fps.mp4';ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    writer=imageio_ffmpeg.write_frames(str(outfile),(960,480),fps=50,codec='libx264',quality=8,macro_block_size=16,ffmpeg_log_level='error',output_params=['-threads','1','-preset','fast','-pix_fmt','yuv420p'])
    writer.send(None)
    indices=list(range(0,len(ts)-1,4));font=ImageFont.load_default(size=16)
    try:
        for k,idx in enumerate(indices):
            if time.time()>DEADLINE-90:raise TimeoutError('Closeout reserve for video')
            d.qpos[:]=qs[idx];mj.mj_forward(m,d);COUNTS['replay_mj_forward_calls']+=1
            renderer.update_scene(d,camera=cameraA);a=renderer.render().copy();COUNTS['rendered_views']+=1
            renderer.update_scene(d,camera=cameraB);b=renderer.render().copy();COUNTS['rendered_views']+=1
            frame=Image.fromarray(np.concatenate([a,b],axis=1));draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,960,60),fill=(12,12,16));draw.text((10,6),f'SEALED04 | FULL 100g OBJECT | attempt {n:03d} | actual t={ts[idx]:.3f}s | 1x',font=font,fill=(255,255,255))
            draw.text((10,30),'LOCAL FIXED HEAD + TORSO | payload FREE | actual trajectory, no interpolation',font=font,fill=(255,201,71))
            draw.rectangle((0,450,960,480),fill=(12,12,16));draw.text((10,456),'Overview',font=font,fill='white');draw.text((490,456),'Jaw / full object detail',font=font,fill='white')
            writer.send(np.asarray(frame));COUNTS['video_frames']+=1
            if k==0:frame.save(p/'preview_cold.png')
            if abs(ts[idx]-.5)<.0001:frame.save(p/'preview_0p5s.png')
            if k==len(indices)-1:frame.save(p/'preview_final.png')
            if k%50==0:threads();print('EVENT render',n,k,'of',len(indices),'actual_t',round(float(ts[idx]),3),flush=True)
    finally:
        writer.close();renderer.close()
    probe=resource.getrusage(resource.RUSAGE_CHILDREN)
    rec={'attempt':n,'path':str(outfile),'SHA256':sha(outfile),'actual_trajectory_SHA256':sha(p/'actual_trajectory.npz'),'model_MJB_SHA256':sha(p/'actual_scene.mjb'),'fps':50,'video_frames':len(indices),'video_duration_s':len(indices)/50,'actual_trajectory_duration_s':float(ts[-1]),'first_sample_time_s':float(ts[indices[0]]),'last_sample_time_s':float(ts[indices[-1]]),'sampling':'exact recorded states every 4 physics steps, start included, end exclusive; last frame spans final20ms; no interpolation/time scaling/cuts','CPU_software_OSMesa_only':True,'GPU':0,'physics_integrations':0}
    write(p/'video_receipt.json',rec);RENDERS.append(rec);print('STATUS video',n,rec['path'],flush=True)
try:
    for n in range(1,5):analysis(n)
    for n in [2,1,3,4]:render(n)
    write(ROOT/'postprocess_status.json',{'status':'completed','renders':RENDERS,'counts':COUNTS})
except BaseException as e:
    write(ROOT/'postprocess_status.json',{'status':'failed','error':repr(e),'renders':RENDERS,'counts':COUNTS});raise
