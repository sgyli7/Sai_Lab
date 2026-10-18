"""Recorded-state kinematics, actual force readback, and diagnostic plot only."""
from pathlib import Path
import os, json, gzip, resource, time
ROOT=Path(__file__).resolve().parent
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
START=time.time()
import numpy as np
import mujoco as mj
assert mj.__version__=='3.10.0'
ALL=[];FIGDATA=[];fk=0
for n in range(1,5):
    p=ROOT/f'attempt_{n:03d}'
    with gzip.open(p/'actual_substeps.jsonl.gz','rt') as f:rr=[json.loads(l) for l in f]
    m=mj.MjModel.from_binary_path(str(p/'actual_scene.mjb'));d=mj.MjData(m)
    names=[j['name'] for j in json.loads((ROOT/'preflight_audit.json').read_text())['named_mapping']]
    vals=[];speed=np.array([r['actual_drive_velocity_after18'] for r in rr]);sel=np.array(rr[0]['selected_voltage_derated_speed18'])
    maxidx=np.unravel_index(np.argmax(abs(speed)/sel),speed.shape)
    limids=np.where(m.jnt_limited)[0];range_worst={'violation_rad':0.}
    for r in rr:
        q=np.array(r['qpos']);jq=q[m.jnt_qposadr[limids]];violation=np.maximum(m.jnt_range[limids,0]-jq,jq-m.jnt_range[limids,1]);i=int(np.argmax(violation))
        if float(violation[i])>range_worst['violation_rad']:range_worst={'violation_rad':float(violation[i]),'time_s':r['time_s'],'joint':m.joint(int(limids[i])).name,'actual_q_rad':float(jq[i]),'range_rad':m.jnt_range[limids[i]].tolist()}
    for t in [.03,.105,.215,.27,1.,3.,5.2]:
        if t>rr[-1]['time_s']+1e-8:continue
        r=min(rr,key=lambda r:abs(r['time_s']-t));d.qpos[:]=r['incoming_qpos'];d.qvel[:]=r['incoming_qvel'];mj.mj_forward(m,d);fk+=1
        cs=[]
        for c in r['contacts']:
            if not (c['geom1']=='pilot_grip' and c['geom2'] in ['upper_bill_envelope','lower_bill_envelope']):continue
            force=np.array(c['actual_force_contact_frame_6D']);
            if force[0]<=1e-6:continue
            point=np.array(c['pos_world_m']);j1=np.zeros((3,m.nv));jr=j1.copy();j2=j1.copy()
            mj.mj_jac(m,d,j1,jr,point,int(m.geom(c['geom1']).bodyid[0]));mj.mj_jac(m,d,j2,jr,point,int(m.geom(c['geom2']).bodyid[0]))
            frame=np.array(c['frame_rows_world']).reshape(3,3);worldv=(j1-j2)@d.qvel;vc=frame@worldv
            cs.append({'geoms':[c['geom1'],c['geom2']],'actual_incoming_tangent_speed_m_s':float(np.linalg.norm(vc[1:])),'actual_incoming_relative_velocity_contact_frame_m_s':vc.tolist(),'contact_normal_inclination_from_vertical_deg':float(np.arccos(np.clip(abs(frame[0,2]),0,1))*180/np.pi),'actual_contact_force6':force.tolist(),'sliding_cone_utilization':float(np.linalg.norm(force[1:3]/np.array(c['effective_friction5'][:2]))/force[0])})
        vals.append({'time_s':r['time_s'],'contact_kinematics_time_s':r['incoming_time_s'],'jaw_logical_q_rad':r['incoming_qpos'][int(m.joint('beak_hinge').qposadr[0])],'jaw_input_q_rad':r['incoming_qpos'][int(m.joint('beak_input_rotor').qposadr[0])],'jaw_target_rad':r['target18'][5],'jaw_torque_Nm':r['actual_actuator_force18'][5],'jaw_continuous_speed_voltage_cap_Nm':r['continuous_torque_speed_caps18'][5],'contacts':cs})
    out={'attempt':n,'source':'actual_substeps.jsonl.gz; recorded force retained, kinematics from exact incoming state; no dynamics','largest_selected_speed_excursion':{'axis':names[maxidx[1]],'time_s':rr[maxidx[0]]['time_s'],'actual_drive_velocity_after_rad_s':float(speed[maxidx]),'selected_slew_speed_rad_s':float(sel[maxidx[1]]),'ratio':float(abs(speed[maxidx])/sel[maxidx[1]])},'worst_actual_joint_soft_limit_violation':range_worst,'samples':vals}
    # Correct the original raw cold-reference label for the plot without changing raw evidence.
    d.qpos[:]=rr[0]['incoming_qpos'];mj.mj_kinematics(m,d)
    head=m.body('head_roll').id;payload=m.body('pilot_payload').id
    birthR=d.xmat[head].reshape(3,3).T@d.xmat[payload].reshape(3,3)
    for r in rr:
        qt=np.empty(4);mj.mju_mat2Quat(qt,(np.array(r['relative_payload_head_R'])@birthR.T).ravel())
        r['plot_true_cold_rotation_deg']=float(2*np.arctan2(np.linalg.norm(qt[1:]),abs(qt[0]))*180/np.pi)
    (p/'kinematic_evidence.json').write_text(json.dumps(out,indent=2)+'\n');ALL.append(out);FIGDATA.append(rr)
    print('EVENT kinematics',n,'speed',out['largest_selected_speed_excursion'],'limit',range_worst,flush=True)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    colors=['#c04b3f','#137a9a','#9a67b8','#888888']
    for n,rr in enumerate(FIGDATA):
        t=np.array([r['time_s'] for r in rr]);color=colors[n]
        ax[0,0].plot(t,[r['plot_true_cold_rotation_deg'] for r in rr],label=f'Attempt {n+1}',color=color)
        hold=[r for r in rr if r['hold_elapsed_s'] is not None]
        if hold:
            h=[r['hold_elapsed_s'] for r in hold]
            ax[0,1].plot(h,[r['hold_relative_drift_m']*1000 for r in hold],color=color,label=f'Attempt {n+1}')
            ax[1,0].plot(h,[r['hold_relative_angle_deg'] for r in hold],color=color)
    r=FIGDATA[1];t=[x['time_s'] for x in r];N=np.array([x['jaw_positive_normal_N'] for x in r]);ax[1,1].plot(t,N[:,0],label='Upper jaw',color='#137a9a');ax[1,1].plot(t,N[:,1],label='Lower jaw',color='#d48a24')
    ax[0,0].set(title='Rotation from saved true t=0 birth',xlabel='Actual time (s)',ylabel='Relative rotation (deg)',ylim=(0,150));ax[0,0].legend(fontsize=9)
    ax[0,1].set(title='Hold translation after first closed bilateral contact',xlabel='Hold elapsed (s)',ylabel='Translation (mm)');ax[0,1].axhline(2,color='#c04b3f',linestyle='--',label='2 mm limit');ax[0,1].legend(fontsize=9)
    ax[1,0].set(title='Hold rotation; failed threshold is never reset',xlabel='Hold elapsed (s)',ylabel='Rotation (deg)');ax[1,0].axhline(2,color='#c04b3f',linestyle='--')
    ax[1,1].set(title='Attempt 2: positive normal force on both jaws',xlabel='Actual time (s)',ylabel='Sum of point normal forces (N)');ax[1,1].legend(fontsize=9)
    for a in ax.flat:a.grid(alpha=.2)
    fig.suptitle('Sealed04 / full 100g payload / LOCAL FIXED HEAD + TORSO / no training',fontsize=12)
    fig.savefig(ROOT/'hardware_diagnostics.png',dpi=150);plt.close(fig)
    plot='hardware_diagnostics.png'
except ImportError as e:plot={'unavailable':str(e)}
r=resource.getrusage(resource.RUSAGE_SELF)
(ROOT/'resource_exit_contact_kinematics.json').write_text(json.dumps({'purpose':'read-only kinematic analysis','physics_integrations':0,'mj_forward_calls':fk,'CPU_affinity':sorted(os.sched_getaffinity(0)),'max_memory_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],'maxRSS_KiB':r.ru_maxrss,'CPU_user_s':r.ru_utime,'CPU_system_s':r.ru_stime,'process_wall_s':time.time()-START,'GPU':0,'training':0,'plot':plot},indent=2)+'\n')
