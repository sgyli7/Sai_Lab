"""Whole robot COM and foot timing from same-frame native inertial-body poses."""
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('/home/ethan/Projects/MicroDuck/sim2sim/results/sprint_input_diagnosis_20260914')
spec=json.loads((R/'robot_spec.json').read_text())
masses={v['name']:v['mass'] for v in spec['bodies'] if v['mass']>0 and v['name']!='ball'}
total=sum(masses.values());reports={};series={};equivalence={}
for case in ['forward','sprint','backward','walk_forward_030','walk_backward_025']:
    trace=json.loads((R/'gait'/case/'native-0.json').read_text());all_rows=trace['rows']
    if case in ['forward','sprint','backward']:
        old=json.loads((R/'baseline'/case/'native-0.json').read_text())['rows']
        fields=['obs','action','last_action','ctrl','command','requested_command']
        equivalence[case]={k:float(np.max(np.abs(np.array([v[k] for v in old])-np.array([v[k] for v in all_rows])))) for k in fields}
        assert max(equivalence[case].values())==0,equivalence[case]
    rows=[v for v in all_rows if 1.5<=v['episode_t']<=4.4]
    t=np.array([v['episode_t'] for v in rows]);dt=float(np.median(np.diff(t)))
    bodies=[{b['name']:b for b in v['raw']['body_states']} for v in rows]
    assert all(set(masses)<=set(b) for b in bodies), 'Whole-body COM needs every massive robot body'
    com=np.array([sum((m*np.asarray(b[n]['pos']) for n,m in masses.items()),np.zeros(3))/total for b in bodies])
    q=np.array([v['body']['base_quat'] for v in rows]);w,x,y,z=q.T
    pitch=np.rad2deg(np.arcsin(np.clip(2*(w*y-z*x),-1,1)))
    sign=1 if com[-1,0]>com[0,0] else -1
    velocity=np.gradient(com[:,0],dt)
    feet={};landings=[];strides=[];duty=[];contacts=[];clearance=[];positions=[]
    for name in ['ankle_left','ankle_right']:
        pos=np.array([b[name]['pos'] for b in bodies]);positions.append(pos)
        contact=median_filter(np.array([b[name]['ground_contact'] for b in bodies],dtype=int),size=3,mode='nearest').astype(bool)
        starts=np.flatnonzero(contact[1:] & ~contact[:-1])+1
        landings.extend(starts.tolist());contacts.append(contact);duty.append(float(contact.mean()))
        stride=sign*np.diff(pos[starts,0]);strides.extend(stride.tolist())
        height=float(np.percentile(pos[:,2],95)-np.percentile(pos[:,2],5));clearance.append(height)
        feet[name]=dict(duty_factor=float(contact.mean()),landing_times=t[starts].tolist(),stride_lengths_m=stride.tolist(),height_range_p05_p95_m=height)
    load=np.stack(contacts,axis=1);pos=np.stack(positions,axis=1)
    support=np.sum(pos[:,:,0]*load,axis=1)/np.maximum(1,load.sum(1));supported=load.any(1)
    offset=sign*(com[:,0]-support)
    steps=sorted(set(landings));durations=np.diff(t[steps])
    changes=[sign*(np.mean(velocity[i+1:i+4])-np.mean(velocity[i-3:i])) for i in steps if 3<=i<len(t)-4]
    reports[case]=dict(speed_mps=float(sign*(com[-1,0]-com[0,0])/(t[-1]-t[0])),
        pitch_degrees_mean=float(pitch.mean()),pitch_degrees_p05_p95=np.percentile(pitch,[5,95]).tolist(),
        tilt_toward_motion_mean=float(sign*pitch.mean()),
        com_height_mean=float(com[:,2].mean()),com_height_std=float(com[:,2].std()),
        com_vs_supporting_ankle_centers_m=float(np.mean(offset[supported])),
        double_support_fraction=float(np.all(load,axis=1).mean()),no_foot_contact_fraction=float((~supported).mean()),
        observed_footfalls=len(steps),footfalls_per_second=float(1/np.median(durations)) if len(durations) else None,
        same_foot_stride_median=float(np.median(strides)) if strides else None,
        mean_step_length_m=float(np.median(strides)/2) if strides else None,
        landing_speed_change_60ms=float(np.mean(changes)) if changes else None,
        feet=feet)
    series[case]=(t,sign*velocity,pitch,offset*1000,load,com)
result=dict(robot_mass_kg=total,robot_body_count=len(masses),cases=reports,telemetry_action_equivalence=equivalence,
    definitions='Steady window 1.5–4.4 s. Body origins are forced CUSTOM/zero inertial COM by the existing engine. Whole-body COM excludes the ball. Support offset is relative to supporting ankle centers, NOT contact polygon or center of pressure. Contacts are median-filtered across 3 frames (60 ms). Footfall/stride sample counts are small. Landing velocity change is a kinematic diagnostic, NOT force attribution. No claim that short contact loss establishes a running gait.')
(R/'gait_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axs=plt.subplots(4,1,figsize=(11,10),sharex=True)
colors=['#3465a4','#eb8a23','#9b55bd']
for case,color in zip(['forward','sprint','backward'],colors):
    t,v,p,offset,load,com=series[case]
    axs[0].plot(t,v,label=case,color=color,lw=1)
    axs[1].plot(t,p,label=case,color=color,lw=1)
    axs[2].plot(t,offset,label=case,color=color,lw=1)
for i,case in enumerate(['forward','sprint','backward']):
    t,v,p,offset,load,com=series[case]
    for j,side in enumerate(['L','R']):
        y=i*2+j
        axs[3].scatter(t[load[:,j]],np.full(load[:,j].sum(),y),s=8,color=colors[i],marker='s')
axs[0].set_ylabel('Speed along travel (m/s)');axs[0].legend(ncol=3)
axs[1].set_ylabel('Torso pitch (deg)\n+ = toward head')
axs[2].set_ylabel('COM - stance ankle (mm)\n+ = toward travel')
axs[3].set_ylabel('Foot contact');axs[3].set_yticks(range(6),['W L','W R','Shift+W L','Shift+W R','S L','S R']);axs[3].set_xlabel('Simulation time (s)')
for ax in axs:ax.grid(alpha=.2)
fig.suptitle('Current desktop MD: faster walking is not proof of a running gait',fontsize=15)
fig.tight_layout();fig.savefig(R/'gait_comparison.png',dpi=160)
print(json.dumps({k:{x:v[x] for x in ['speed_mps','pitch_degrees_mean','double_support_fraction','no_foot_contact_fraction','footfalls_per_second','mean_step_length_m','landing_speed_change_60ms']} for k,v in reports.items()},indent=2))
