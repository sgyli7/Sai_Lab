"""CPU static PNG/PDF from a complete benchmark summary, with no raw-case reads."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCHEMA='pi05_g1_round02_multi_policy_benchmark_v2'
LABELS={'gr00t_n17':('GR00T N1.7','native ONNX Runtime'),
        'pi05_round01':('First-round pi0.5','PyTorch / common norm v2'),
        'pi05_round02':('Updated pi0.5','PyTorch / common norm v2'),
        'official_pi05_base':('Official pi0.5 base','PyTorch / common norm v2')}
FAILURES=('fell','dropped','missed_grasp','grasp_without_placement','unstable_or_unreleased_placement')
COLORS={'fell':'#A73939','dropped':'#E07A39','missed_grasp':'#6083A6',
        'grasp_without_placement':'#B39757','unstable_or_unreleased_placement':'#9B78AA'}
SHORT={'fell':'Fell','dropped':'Dropped','missed_grasp':'Missed grasp',
       'grasp_without_placement':'Grasp, no placement','unstable_or_unreleased_placement':'Unstable / unreleased'}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def validate(report):
    if report.get('schema')!=SCHEMA or report.get('complete') is not True or report.get('evidence_qualified') is not True:
        raise ValueError('Only complete, evidence-qualified formal benchmark summaries may be plotted')
    if report.get('errors')!=[] or report.get('infrastructure_incomplete')!=[]:raise ValueError('Incomplete evidence/infrastructure')
    policies=report.get('policies',{})
    if len(policies) not in (3,4) or report.get('policy_count')!=len(policies) or not set(policies)<=set(LABELS):raise ValueError('Expected three or four named benchmark policies')
    if report.get('physical_positions')!=5 or report.get('noise_repetitions_per_position')!=4:raise ValueError('Expected five position clusters with four repeats')
    checks=report.get('actual_initial_pair_checks',[])
    if len(checks)!=20 or len({c['case_id'] for c in checks})!=20 or any(c.get('all_policies_match') is not True or c.get('policies_expected')!=len(policies) for c in checks):
        raise ValueError('All twenty actual-initial-state comparisons must qualify')
    videos=report.get('video_evidence',[])
    if len(videos)!=20*len(policies) or any(v.get('complete') is not True for v in videos):raise ValueError('All videos must be evidence-qualified')
    for policy,row in policies.items():
        n=row.get('successes');ci=row.get('success_wilson_95ci');stages=row.get('mutually_exclusive_stage_counts',{})
        if row.get('completed')!=20 or row.get('expected')!=20 or row.get('missing_or_infrastructure_cases')!=[]:raise ValueError('Every policy must have the same twenty completed trials')
        if type(n) is not int or not 0<=n<=20 or row.get('success_rate')!=n/20 or row.get('task_failure_rate')!=(20-n)/20:raise ValueError('Inconsistent success/failure counts')
        if not isinstance(ci,list) or len(ci)!=2 or not all(finite(x) for x in ci) or not 0<=ci[0]<=n/20<=ci[1]<=1:raise ValueError('Invalid Wilson interval')
        if any(type(v) is not int or v<0 for v in stages.values()) or stages.get('success')!=n or sum(stages.values())!=20 or sum(stages.get(k,0) for k in FAILURES)!=20-n:
            raise ValueError('Mutually exclusive failure stages do not match outcomes')
        rpc=row.get('warm_rpc_wall_seconds')
        if not isinstance(rpc,dict) or type(rpc.get('count')) is not int or rpc['count']<=0 or not finite(rpc.get('p50')) or not finite(rpc.get('p95')) or not 0<=rpc['p50']<=rpc['p95']:
            raise ValueError('Warm RPC median/P95 evidence missing')
    comparisons=report.get('comparisons',[])
    expected={frozenset(p) for p in itertools.combinations(policies,2)}
    keys=[frozenset((c['first_policy'],c['second_policy'])) for c in comparisons]
    if len(keys)!=len(expected) or set(keys)!=expected:raise ValueError('All policy-pair comparisons must be present')
    for pair in comparisons:
        ids=pair['matched_success_case_ids'];speeds=pair['matched_success_speed']
        if len(ids)!=len(set(ids)) or {s['case_id'] for s in speeds}!=set(ids) or len(speeds)!=len(ids):raise ValueError('Matched-success speed records are incomplete')
        if not ids and any(pair.get(key) is not None for key in ('matched_success_sim_delta_seconds','matched_success_actual_task_wall_delta_seconds')):
            raise ValueError('No shared success must report N/A speed')
        if ids:
            for key in ('matched_success_sim_delta_seconds','matched_success_actual_task_wall_delta_seconds'):
                metric=pair.get(key)
                if not isinstance(metric,dict) or metric.get('count')!=len(ids) or not finite(metric.get('mean')):
                    raise ValueError('Matched-success speed summary missing/inconsistent')
    return policies


def draw_figure(report):
    """Standard Matplotlib artists retain paths/text in the PDF export."""
    policies=validate(report);names=list(policies);x=np.arange(len(names));ink='#243345';muted='#56667A'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                         'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(15,7.2));fig.patch.set_facecolor('#F5F6F8')
    fig.subplots_adjust(left=.055,right=.985,top=.76,bottom=.29,wspace=.30)
    fig.suptitle('G1 apple pick-and-place | Frozen simulation benchmark',x=.055,y=.955,ha='left',fontsize=19,fontweight='bold',color=ink)
    fig.text(.055,.898,'20 trials per policy = 5 positions x 4 seeds | Simulation only | Native ONNX Runtime vs PyTorch',color=muted,fontsize=11)
    labels=[LABELS[name][0].replace(' ', '\n',1)+'\n'+('ONNX Runtime' if name=='gr00t_n17' else 'PyTorch') for name in names]
    for ax in axes:
        ax.set_axisbelow(True);ax.grid(axis='y',color='#E0E6EC',linewidth=.8)
        ax.set_xticks(x,labels,fontsize=9);ax.tick_params(axis='x',length=0,pad=8)
        ax.spines[['top','right']].set_visible(False)
        for side in ('left','bottom'):ax.spines[side].set_color('#CDD5DF')
        ax.tick_params(colors=muted);ax.set_xlim(-.55,len(names)-.45)
    success=np.asarray([policies[name]['successes'] for name in names],float)
    ci=np.asarray([policies[name]['success_wilson_95ci'] for name in names])*20
    axes[0].bar(x,success,width=.5,color='#2B7773',zorder=3)
    axes[0].errorbar(x,success,yerr=np.vstack((success-ci[:,0],ci[:,1]-success)),fmt='none',ecolor=ink,capsize=6,linewidth=1.6,zorder=4)
    axes[0].axhline(16,color='#4F7968',linestyle='--',linewidth=1.1)
    axes[0].text(-.48,16.35,'16/20 task threshold',color='#4F7968',fontsize=8.5)
    for i,value in enumerate(success):axes[0].text(i,ci[i,1]+.45,f'{int(value)}/20',ha='center',weight='bold',color=ink)
    axes[0].set(title='Success / 20 + Wilson 95% CI',ylabel='Successful trials',ylim=(0,22),yticks=np.arange(0,21,4))
    bottoms=np.zeros(len(names))
    for stage in FAILURES:
        values=np.asarray([policies[name]['mutually_exclusive_stage_counts'].get(stage,0) for name in names])
        axes[1].bar(x,values,bottom=bottoms,width=.5,color=COLORS[stage],label=SHORT[stage],zorder=3)
        bottoms+=values
    for i,value in enumerate(bottoms):axes[1].text(i,value+.45,str(int(value)),ha='center',weight='bold',color=ink)
    axes[1].set(title='Task failures by stage',ylabel='Failed trials',ylim=(0,22),yticks=np.arange(0,21,4))
    handles,legend_labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,.125),frameon=False,fontsize=8,ncol=5,handlelength=1,handletextpad=.5)
    rpc=[policies[name]['warm_rpc_wall_seconds'] for name in names]
    for offset,key,color,label in ((-.16,'p50','#347891','Median'),(.16,'p95','#D3904A','P95')):
        values=[r[key] for r in rpc];bars=axes[2].bar(x+offset,values,width=.29,color=color,label=label,zorder=3)
        axes[2].bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=4,fontsize=8)
    ymax=max(r['p95'] for r in rpc)*1.25 or 1.
    axes[2].set(title='Warm RPC latency',ylabel='Seconds per RPC',ylim=(0,ymax))
    axes[2].legend(loc='upper left',frameon=False,fontsize=8,ncol=2)
    for i,r in enumerate(rpc):axes[2].text(i,-.235,f"n={r['count']} RPCs",transform=axes[2].get_xaxis_transform(),ha='center',fontsize=7.5,color=muted)
    fig.text(.055,.075,'Paired task speed: matched_success_speed.csv. Mutually successful cases only; no common success = N/A.',fontsize=10,color=ink)
    fig.text(.055,.04,'5 position clusters; repeated seeds are not independent scenes. Warm RPC includes IPC/preprocessing/inference; no hardware real-time claim.',fontsize=9,color=muted)
    return fig


def export(summary,output):
    summary=Path(summary).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Plot export requires a new directory; preserve previous evidence')
    before=sha(summary);report=json.loads(summary.read_text(),parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))
    validate(report);figure=draw_figure(report);output.mkdir(parents=True)
    try:
        figure.savefig(output/'benchmark.png',dpi=200,facecolor=figure.get_facecolor())
        figure.savefig(output/'benchmark.pdf',format='pdf',facecolor=figure.get_facecolor())
    finally:plt.close(figure)
    fields=['first_policy','second_policy','matched_success_count','speed_status','mean_second_minus_first_sim_seconds','mean_second_minus_first_actual_task_wall_seconds']
    with (output/'matched_success_speed.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for pair in report['comparisons']:
            count=len(pair['matched_success_case_ids'])
            writer.writerow({'first_policy':pair['first_policy'],'second_policy':pair['second_policy'],'matched_success_count':count,
               'speed_status':'paired_successes_only' if count else 'N/A: no common successful cases',
               'mean_second_minus_first_sim_seconds':pair['matched_success_sim_delta_seconds']['mean'] if count else 'N/A',
               'mean_second_minus_first_actual_task_wall_seconds':pair['matched_success_actual_task_wall_delta_seconds']['mean'] if count else 'N/A'})
    if sha(summary)!=before:raise ValueError('Input summary changed during export; preserve partial output for audit')
    files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir())}
    dependency_receipt=Path(__file__).resolve().parents[2]/'.scratch/pi05_g1/round02/plot_environment_receipt.json'
    manifest={'schema':'pi05_g1_round02_static_plot_export_v1','summary_path':str(summary),'summary_sha256':before,
              'benchmark_schema':SCHEMA,'policies':list(report['policies']),'trials_per_policy':20,'position_clusters':5,'seeds_per_position':4,
              'simulation_only':True,'render_backend':'Matplotlib '+matplotlib.__version__+' / Agg CPU; PDF retains vector artists and embedded TrueType text; PNG 200dpi',
              'dependency_receipt':str(dependency_receipt),'dependency_receipt_sha256':sha(dependency_receipt) if dependency_receipt.is_file() else None,
              'renderer_versions':{'matplotlib':matplotlib.__version__,'numpy':np.__version__},
              'raw_case_reads':False,'speed_scope':'Only matched successful cases; no common success explicitly N/A. No failure-duration ranking.',
              'tool_sha256':sha(__file__),'files':files}
    with (output/'manifest.json').open('x') as stream:stream.write(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest=export(a.summary,a.output);print(json.dumps({'summary_sha256':manifest['summary_sha256'],'files':list(manifest['files'])}))


if __name__=='__main__':main()
