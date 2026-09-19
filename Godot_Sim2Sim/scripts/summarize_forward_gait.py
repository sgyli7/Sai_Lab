"""Independent native displacement/contact summaries for the frozen gait ablation."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter


def landing_pattern(landings,t):
    events={}
    for index,side in landings:events.setdefault(index,set()).add(side)
    ordered=sorted(events)
    alternating=[len(events[a])==len(events[b])==1 and events[a]!=events[b] for a,b in zip(ordered,ordered[1:])]
    intervals=np.diff(t[ordered])
    return dict(alternating_landing_fraction=float(np.mean(alternating)) if alternating else None,
                simultaneous_landing_events=sum(len(sides)>1 for sides in events.values()),
                median_interval_frequency_hz=float(1/np.median(intervals)) if len(intervals) else None,
                landing_events_per_second=len(ordered)/(t[-1]-t[0]),
                individual_foot_landings_per_second=len(landings)/(t[-1]-t[0]))


def native_gait(trace):
    payload=json.loads(Path(trace).read_text());rows=payload['rows']
    first_fall=next((r['episode_t'] for r in rows if r.get('fell')),None)
    rows=[r for r in rows if 4. <= r['episode_t'] <= 20.]
    t=np.array([r['episode_t'] for r in rows]);dt=float(np.median(np.diff(t)))
    xyz=np.array([r['body']['base_pos'] for r in rows]);q=np.array([r['body']['base_quat'] for r in rows]);w,x,y,z=q.T
    yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z));heading=np.array([np.cos(yaw[0]),np.sin(yaw[0])])
    displacement=(xyz[:,:2]-xyz[0,:2])@heading;speed=np.gradient(displacement,dt)
    bodies=[{b['name']:b for b in r['raw']['body_states']} for r in rows]
    contacts=[];stride=[];landings=[];feet={}
    for side,name in enumerate(['ankle_left','ankle_right']):
        pos=np.array([b[name]['pos'] for b in bodies]);contact=median_filter(np.array([b[name]['ground_contact'] for b in bodies],int),size=3,mode='nearest').astype(bool)
        starts=np.flatnonzero(contact[1:] & ~contact[:-1])+1
        lengths=np.diff(pos[starts,:2]@heading)
        feet[name]=dict(landings=len(starts),duty_factor=float(contact.mean()),median_same_foot_stride=float(np.median(lengths)) if len(lengths) else None,median_cycle_s=float(np.median(np.diff(t[starts]))) if len(starts)>1 else None)
        contacts.append(contact);stride.extend(lengths.tolist());landings.extend((int(i),side) for i in starts)
    load=np.stack(contacts,axis=1);pitch=np.rad2deg(np.arcsin(np.clip(2*(w*y-z*x),-1,1)))
    result=dict(net_forward_speed_mps=float(displacement[-1]/(t[-1]-t[0])),negative_speed_fraction=float((speed<-.05).mean()),step_length_estimate_m=float(np.median(stride)/2) if stride else None,double_contact_fraction=float(load.all(-1).mean()),no_foot_contact_fraction=float((~load.any(-1)).mean()),pitch_mean_deg=float(pitch.mean()),pitch_std_deg=float(pitch.std()),feet=feet,**landing_pattern(landings,t))
    result.update(first_fall_s=first_fall,valid_steady_window=first_fall is None or first_fall>20.)
    return result,dict(t=t,speed=speed,displacement=displacement,pitch=pitch,contacts=load)


def main():
    p=argparse.ArgumentParser();p.add_argument('session',type=Path);a=p.parse_args();R=a.session.resolve()
    comparison={};series={};training={}
    for path in sorted(R.glob('native_*/completed.json')):
        label=path.parent.name.removeprefix('native_');result=json.loads(path.read_text());summary=json.loads((path.parent/'suite/summary.json').read_text())
        long=[e for e in summary['episodes'] if e['case']=='sprint_long'];gait=[]
        for episode in long:
            values,data=native_gait(episode['trace']);gait.append(dict(seed=episode['seed'],**values))
            if episode['seed']==929100:series[label]=data
        comparison[label]=dict(passes=result['passes'],count=result['count'],falls=result['falls'],errors=result['errors'],long_speed_mean=float(np.mean(result['long_speed'])),gait=gait,
                               net_speed_mean=float(np.mean([x['net_forward_speed_mps'] for x in gait])),step_length_mean=float(np.mean([x['step_length_estimate_m'] for x in gait])) if all(x['valid_steady_window'] and x['step_length_estimate_m'] is not None for x in gait) else None,valid_gait_windows=sum(x['valid_steady_window'] for x in gait),max_negative_speed_fraction=max(x['negative_speed_fraction'] for x in gait))
        gpu=R/('gpu_'+label)/'completed.json'
        if gpu.exists():
            data=json.loads(gpu.read_text());speeds=[x['metrics']['sustained_mean_vx'] for x in data['results'] if x['case']=='sprint_long']
            comparison[label]['gpu']=dict(passes=data['passes'],count=data['count'],falls=data['falls'],long_speed_mean=float(np.mean(speeds)))
            indices={(x['case'],x['seed']):i for i,x in enumerate(data['results'])}
            with np.load(gpu.parent/'trajectories.npz',allow_pickle=False) as archive:
                requested,selected=archive['requested'],archive['sprint']
            command_error=0.;selection_errors=0;frames=0
            for episode in summary['episodes']:
                if episode['control_config']!=data['effective_control']:raise ValueError('GPU/native effective control differs')
                trace=json.loads(Path(episode['trace']).read_text());rows=trace['rows'];j=indices[(episode['case'],episode['seed'])]
                native=np.array([r['requested_command'] for r in rows]);command_error=max(command_error,float(np.abs(native-requested[:len(rows),j]).max()))
                selection_errors+=int((np.array([r['skill']=='sprint' for r in rows])!=selected[:len(rows),j]).sum());frames+=len(rows)
            comparison[label]['input_parity']=dict(frames=frames,requested_max_abs=command_error,selection_errors=selection_errors,passed=command_error<1e-6 and selection_errors==0)
            if not comparison[label]['input_parity']['passed']:raise ValueError('Native/GPU keyboard tapes differ')
    for path in R.glob('*_9524*/metrics.jsonl'):
        rows=[json.loads(x) for x in path.read_text().splitlines()];audits=[r['gradient_audit'] for r in rows if r.get('gradient_audit')];last=rows[-1]
        training[path.parent.name]=dict(iterations=last['iteration'],samples=last['samples'],last_terms=last['terms'],maximum_sampled_bound_fraction=max(r['residual']['bound_fraction'] for r in rows),last_residual=last['residual'],gradient_conflict_fraction=float(np.mean([x['cosine']<0 for x in audits])) if audits else None,teacher_to_ppo_norm_median=float(np.median([x['weighted_teacher_norm']/x['ppo_norm'] for x in audits])) if audits else None)
    result=dict(comparison=comparison,training=training,accepted=False,definitions='Native long cases, seconds4–20; trunk origin net displacement along its heading at4s; ankle inertial centers for same-foot stride/2 estimated step; contacts median-filtered60ms. Not whole-body COM, center of pressure, measured support force, exact sole slip, or proof of running. Finite retained development seeds, no held-out final evaluation.')
    (R/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['default030','a402','off_952410','support_952410'];fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for label in labels:
        if label not in series:continue
        d=series[label];axs[0].plot(d['t'],d['displacement'],label=label);axs[1].plot(d['t'],d['pitch'],label=label,alpha=.75,lw=.8)
    for i,label in enumerate(labels):
        if label not in series:continue
        d=series[label]
        for side in range(2):axs[2].scatter(d['t'][d['contacts'][:,side]],np.full(d['contacts'][:,side].sum(),2*i+side),s=4,marker='s')
    axs[0].set_ylabel('Net forward displacement (m)');axs[0].legend(ncol=2)
    axs[1].set_ylabel('Torso pitch (degrees)');axs[2].set_ylabel('Foot contact');axs[2].set_yticks(range(8),[label+' '+side for label in labels for side in ['L','R']]);axs[2].set_xlabel('Simulation time (s)')
    for ax in axs:ax.grid(alpha=.2)
    fig.suptitle('Native Jolt: fixed endpoints, development seed 929100');fig.tight_layout();fig.savefig(R/'native_gait_comparison.png',dpi=140)
    print(json.dumps({k:{x:v[x] for x in ['passes','count','falls','long_speed_mean','net_speed_mean','step_length_mean']} for k,v in comparison.items()},indent=2))


if __name__=='__main__':main()
