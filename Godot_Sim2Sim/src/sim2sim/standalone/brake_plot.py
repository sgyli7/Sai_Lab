"""Plot recorded brake behavior; no new physical run or policy selection occurs."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sim2sim.research.tasks import DT
from sim2sim.research.queue import atomic_json
from .score import TraceWorld, brake_metrics, moving_average


def recorded_rows(path):
    trace=json.loads(Path(path).read_text())
    if trace['summary']['error']:raise ValueError('Cannot plot an invalid trace as completed behavior')
    world=TraceWorld('roller','roller');rows=[]
    try:
        for index,row in enumerate(trace['rows']):
            raw=trace['rows'][index+1]['raw'] if index+1<len(trace['rows']) else trace['summary']['final_raw']
            rows.append(world.append_state(raw,row['action'],row.get('requested_command',row['command']),row['t']+DT))
    finally:world.close()
    return trace,rows


def plot(before,after,case,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    definition=json.loads(Path(case).read_text());at=definition['brake_times'][0]
    fig,axes=plt.subplots(4,1,figsize=(9,9),sharex=True,layout='constrained')
    evidence=dict(case=str(Path(case).resolve()),case_sha256=hashlib.sha256(Path(case).read_bytes()).hexdigest(),runs=[])
    for name,path,color in [('Frozen incumbent',before,'#b0443b'),('Development candidate',after,'#087f8c')]:
        trace,rows=recorded_rows(path)
        selected=[row for row in rows if at-1e-9<=row['time']<=definition['seconds']+1e-9]
        times=np.array([row['time']-at for row in selected])
        speed=np.array([np.linalg.norm(row['vel'][:2]) for row in selected])
        xy=np.array([row['xy'] for row in selected])
        distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        tilt=np.array([row['tilt'] for row in selected]);height=np.array([row['z'] for row in selected])
        metrics=brake_metrics(rows,at,definition['seconds'])
        axes[0].plot(times[9:],moving_average(speed,10),color=color,label=name)
        axes[1].plot(times,tilt,color=color)
        axes[2].plot(times,height,color=color)
        axes[3].plot(times,distance,color=color)
        if metrics['first_fall_s'] is not None:
            axes[1].axvline(metrics['first_fall_s'],color=color,alpha=.6,linestyle=':')
            axes[1].annotate(f"Fall: {metrics['first_fall_s']:.2f} s",(metrics['first_fall_s'],70),
                xytext=(8,5),textcoords='offset points',color=color)
        if metrics['stop_confirmed_s'] is not None and metrics['success']:
            confirmed=metrics['stop_confirmed_s']
            axes[0].axvline(confirmed,color=color,alpha=.6,linestyle=':')
            axes[0].annotate(f'Stop confirmed: {confirmed:.2f} s',(confirmed,.05),
                xytext=(8,22),textcoords='offset points',color=color)
            axes[3].scatter([confirmed],[metrics['braking_distance']],color=color,zorder=3)
        slug='before' if path==before else 'after'
        np.savetxt(output/(slug+'.csv'),np.column_stack([times,speed,tilt,height,distance]),delimiter=',',
            header='seconds_after_S,speed_m_s,tilt_deg,height_m,path_distance_m',comments='')
        evidence['runs'].append(dict(label=name,trace=str(Path(path).resolve()),
            trace_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),models=trace['summary']['models'],metrics=metrics))
    axes[0].axhline(.05,color='#666666',linestyle='--',linewidth=1)
    axes[1].axhline(15,color='#666666',linestyle='--',linewidth=1)
    axes[1].axhline(70,color='#aaaaaa',linestyle=':',linewidth=1)
    axes[2].axhline(.08,color='#666666',linestyle='--',linewidth=1)
    axes[3].axhline(evidence['runs'][0]['metrics']['distance_limit'],color='#666666',linestyle='--',linewidth=1)
    for axis,label in zip(axes,['Speed, 0.2 s average (m/s)','Trunk tilt (degrees)','Trunk height (m)','Travel after S (m)']):
        axis.set_ylabel(label);axis.grid(alpha=.2)
        axis.axvline(2.,color='#888888',linestyle='--',linewidth=.8)
    axes[0].legend(loc='upper right');axes[-1].set_xlabel('Seconds after S is pressed')
    axes[-1].set_xlim(0,definition['seconds']-at)
    title='Original brake replay: idle 1 s, W 3 s, then S' if definition['case']=='roller_brake_3s' else definition['case'].replace('_',' ')
    fig.suptitle(title+'\nUnchanged Jolt physics; development evidence')
    fig.savefig(output/'braking.png',dpi=180);fig.savefig(output/'braking.pdf');plt.close(fig)
    atomic_json(output/'evidence.json',evidence)
    return evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before');parser.add_argument('after');parser.add_argument('case');parser.add_argument('--out',required=True)
    args=parser.parse_args();plot(args.before,args.after,args.case,args.out)
