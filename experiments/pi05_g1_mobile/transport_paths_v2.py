"""CPU measured-path figures for exactly60 qualified formal trials; no new verdict.

Consumes a complete benchmark,passed independentv3 audits,and existing transport
analysis. Streams raw traces once,retaining onlyXY/yaw/tick and existing events.
No simulator/model/testdata imports or generated content.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path

POLICIES=('gr00t_n16','pi05_mobile','official_pi05_base')
SCHEMA='pi05_g1_mobile_transport_paths_result_v2'
ROOT=Path(__file__).resolve().parents[2]
CAPTION=('4m: maximum held-sample XY net displacement from first bilateral contact + lift; 60deg: held-sample robot heading span.\n'
         'These do not guarantee continuous grasp. Bin center is post-hoc audit truth only; no new arrival judgment or policy input.\n'
         'Full mission failure = 1 - success; drop/fall may overlap; timeout is mutually exclusive with both. Existing scores unchanged.')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON:'+x)))


def resolve(path,base):
    value=Path(path);return value.resolve() if value.is_absolute() else (Path(base)/value).resolve()


def yaw(pose):
    if len(pose)!=7 or any(type(x) not in (int,float) or not math.isfinite(x) for x in pose):raise ValueError('Finite measured7D pose required')
    x,y,z,w=pose[3:];n=x*x+y*y+z*z+w*w
    if not math.isclose(n,1.,rel_tol=0,abs_tol=1e-3):raise ValueError('Root quaternion not unit')
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def measurement_identity():
    from .acceptance_pose_speed import measurement_identity as identity
    return identity()


def prepare_inputs(plan_path,summary_path,audits_paths,analysis_path):
    plan_path=Path(plan_path).resolve();summary_path=Path(summary_path).resolve();analysis_path=Path(analysis_path).resolve()
    plan=read(plan_path);summary=read(summary_path);analysis=read(analysis_path)
    if summary.get('schema')!='pi05_g1_mobile_benchmark_summary_v1' or summary.get('status')!='complete' or any(summary.get(key) is not True for key in ('complete','evidence_qualified','formal_verdict_available')):raise ValueError('Complete evidence-qualified formal benchmark required')
    if plan.get('schema')!='pi05_g1_mobile_benchmark_plan_v1' or set(plan.get('primary_policies',plan.get('policies',[])))!=set(POLICIES):raise ValueError('Frozen3-policy formal plan required')
    cases=plan['case_ids']
    if len(cases)!=20 or len(set(cases))!=20 or summary.get('case_ids')!=cases or set(summary.get('policy_order',[]))!=set(POLICIES):raise ValueError('Exact20 frozen case order required')
    expected={(p,c) for p in POLICIES for c in cases};jobs={(x['policy'],x['case_id']):x for x in plan['jobs']}
    if len(plan['jobs'])!=60 or set(jobs)!=expected:raise ValueError('Exactly60 unique formal jobs required')
    if summary.get('inputs_sha256',{}).get(str(plan_path))!=sha(plan_path):raise ValueError('Benchmark not bound to frozen formal plan')
    if analysis.get('schema')!='pi05_g1_mobile_transport_analysis_v1' or analysis.get('source_benchmark',{}).get('sha256')!=sha(summary_path) or Path(analysis['source_benchmark']['path']).resolve()!=summary_path:raise ValueError('Existing transport analysis benchmark identity differs')
    rows={(x['policy'],x['case_id']):x for x in summary['episodes']};transport={(x['policy'],x['case_id']):x for x in analysis['episodes']}
    if len(summary['episodes'])!=60 or len(analysis['episodes'])!=60 or set(rows)!=expected or set(transport)!=expected:raise ValueError('All60 benchmark and transport episodes required')
    audit_entries={};input_hash={str(plan_path):sha(plan_path),str(summary_path):sha(summary_path),str(analysis_path):sha(analysis_path)}
    tool=ROOT/'experiments/pi05_g1_mobile/audit_episodes_v3.py';tool_sha=sha(tool);context=measurement_identity()
    for ap in audits_paths:
        ap=Path(ap).resolve();doc=read(ap);input_hash[str(ap)]=sha(ap)
        if doc.get('schema')!='pi05_g1_mobile_episode_audits_v2' or doc.get('passed') is not True or doc.get('tool_sha256')!=tool_sha:raise ValueError('Independent full v3 audit must pass and bind exact source')
        for entry in doc['episodes']:
            key=str(Path(entry['result_path']).resolve())
            if key in audit_entries:raise ValueError('Duplicate independent audit case')
            if entry.get('passed') is not True or entry.get('tool_sha256')!=tool_sha or not entry.get('checks') or any(x.get('passed') is not True for x in entry['checks'].values()):raise ValueError('Every independent case audit must pass')
            audit_entries[key]=entry
    if len(audit_entries)!=60:raise ValueError('Independent audits must cover exactly60 cases')
    records=[]
    for policy in POLICIES:
        for cid in cases:
            key=(policy,cid);episode=rows[key];metrics=transport[key];job=jobs[key];result_path=resolve(job['result_path'],plan_path.parent)
            if Path(episode['result_path']).resolve()!=result_path or str(result_path) not in audit_entries:raise ValueError('Formal episode/audit path differs from plan')
            audit=audit_entries[str(result_path)];config_path=resolve(job['config_path'],plan_path.parent);config=read(config_path);result=read(result_path);trace=result_path.parent/'trajectory.jsonl';raw_config_path=result_path.parent/'config.json';raw_config=read(raw_config_path)
            if raw_config.get('case')!=config.get('case') or raw_config.get('policy_id')!=policy or raw_config.get('mode')!='policy':raise ValueError('Actual recorded case config differs from frozen identity')
            if config['case']['seed'] not in range(1040,1044) or config.get('mode')!='policy' or config['policy_id']!=policy or config['case']['case_id']!=cid or result.get('case')!=config['case']:raise ValueError('Development/nonformal case forbidden')
            if sha(config_path)!=job['config_sha256'] or result.get('status')!='completed' or result.get('source_interface_passed') is not True:raise ValueError('Frozenconfig or producercompletion differs')
            judged=audit['recomputed_acceptance'];acceptance=result['acceptance']
            if acceptance.get('measurement_source')!=context or judged.get('measurement_source')!=context:raise ValueError('Same explicit pose-speed measurement context required')
            for flag in ('success','grasped','stable_placement','fell','dropped','timeout'):
                if type(episode.get(flag)) is not bool or episode[flag]!=acceptance.get(flag) or judged.get(flag)!=episode[flag]:raise ValueError('Existing benchmark/audit verdict disagreement')
            if metrics.get('diagnostic_prefix_only') is not False or metrics.get('infrastructure_completed') is not True or metrics.get('full_task_verdict')!=episode['success']:raise ValueError('Diagnostic/development transport report forbidden')
            for name in ('first_grasp_control_tick','carry_qualified_control_tick'):
                if metrics.get(name)!=judged.get(name) or acceptance.get(name)!=judged.get(name):raise ValueError('Existing event identity differs')
            rule=acceptance['placement_rule']
            if rule['minimum_carried_displacement_m']!=4 or rule['minimum_carry_turn_degrees']!=60:raise ValueError('Frozen4m/60deg criteria required')
            input_hash[str(config_path)]=sha(config_path)
            if summary.get('inputs_sha256',{}).get(str(config_path))!=input_hash[str(config_path)]:raise ValueError('Planned config not bound to qualified benchmark')
            for path in (raw_config_path,result_path):
                digest=sha(path)
                if audit['inputs_sha256'].get(str(path))!=digest or metrics['inputs_sha256'].get(str(path))!=digest:raise ValueError('Bound case metadata changed')
                input_hash[str(path)]=digest
            expected_trace=result['trace_sha256']
            if audit['inputs_sha256'].get(str(trace))!=expected_trace or metrics['inputs_sha256'].get(str(trace))!=expected_trace:raise ValueError('Shared raw trace SHA identity differs')
            records.append({'policy':policy,'case_id':cid,'result':result,'trace':trace,'expected_trace_sha256':expected_trace,'trace_stat':(trace.stat().st_dev,trace.stat().st_ino,trace.stat().st_size,trace.stat().st_mtime_ns),'existing':metrics,'summary':episode})
    if {str(x['trace'].parent/'result.json') for x in records}!=set(audit_entries):raise ValueError('Audit contains an unplanned case')
    return plan,records,input_hash,context


def stream_path(record,output):
    metrics=record['existing'];result=record['result'];trace=record['trace'];first=metrics['first_grasp_control_tick'];carry=metrics['carry_qualified_control_tick'];end=result['completed_control_ticks'];diagnostics=metrics['per_tick_diagnostics']
    if len(diagnostics)!=end+1:raise ValueError('Existing diagnostic timing coverage differs')
    fields=['policy','case_id','control_tick','sim_seconds','root_x_m','root_y_m','box_x_m','box_y_m','root_yaw_rad','destination_reference_x_m','destination_reference_y_m','grasp_event','carry_qualified_event','terminal_event','existing_success','existing_termination_reasons']
    csv_path=output/'paths'/record['policy']/(record['case_id']+'.csv');csv_path.parent.mkdir(parents=True,exist_ok=True)
    kept=[];digest=hashlib.sha256();drop=None;fall=None;bin_xy=None
    with trace.open('rb') as stream,csv_path.open('x',newline='') as destination:
        writer=csv.DictWriter(destination,fieldnames=fields);writer.writeheader()
        for index,line in enumerate(stream):
            digest.update(line)
            if not line.strip():raise ValueError('Blank row breaks strict frozen timing')
            row=json.loads(line);tick=row['control_tick'];sample=diagnostics[index] if index<len(diagnostics) else {}
            if tick!=index or row['physics_steps_since_reset']!=tick*4 or sample.get('control_tick')!=tick:raise ValueError('Raw physical trace timeline mismatch')
            root=row['root_link_pose_w'];box=row['acceptance_truth_only']['brown_box']['pose'];angle=yaw(root)
            if index==0:
                bin_pose=row['acceptance_truth_only']['blue_sorting_bin']['pose']
                if len(bin_pose)!=7 or any(type(x) not in (int,float) or not math.isfinite(x) for x in bin_pose):raise ValueError('Finite measured destination bin pose required')
                bin_xy=list(bin_pose[:2])
            if len(box)!=7 or not all(type(x) in (int,float) and math.isfinite(x) for x in box):raise ValueError('Finite box pose required')
            entry={'policy':record['policy'],'case_id':record['case_id'],'control_tick':tick,'sim_seconds':tick/50,'root_x_m':root[0],'root_y_m':root[1],'box_x_m':box[0],'box_y_m':box[1],'root_yaw_rad':angle,'destination_reference_x_m':bin_xy[0],'destination_reference_y_m':bin_xy[1],'grasp_event':tick==first,'carry_qualified_event':tick==carry,'terminal_event':tick==end,'existing_success':record['summary']['success'],'existing_termination_reasons':';'.join(record['summary']['termination_reasons']) if tick==end else ''}
            writer.writerow(entry);kept.append(entry)
            if sample.get('ground_drop_this_tick') and drop is None:drop=tick
            if sample.get('fell_so_far') and fall is None:fall=tick
    if len(kept)!=end+1 or digest.hexdigest()!=record['expected_trace_sha256']:raise ValueError('Raw trace coverage/SHA differs from existing audits')
    window=result['acceptance'].get('qualified_stability_window')
    return {'policy':record['policy'],'case_id':record['case_id'],'csv_path':str(csv_path),'rows':kept,'destination_reference_xy_m':bin_xy,'box_start_xy_m':[kept[0]['box_x_m'],kept[0]['box_y_m']],'existing_success':record['summary']['success'],'events':{'grasp':first,'carry4m60deg':carry,'stable_endpoint':None if window is None else window['end_tick'],'first_existing_drop_flag':drop,'first_existing_fall_flag':fall,'trial_terminal':end},'termination_reasons':record['summary']['termination_reasons']}


def draw(paths,cases,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=plt.get_cmap('tab20');allrows=[r for p in paths for r in p['rows']]
    xs=[r[k] for r in allrows for k in ('root_x_m','box_x_m','destination_reference_x_m')];ys=[r[k] for r in allrows for k in ('root_y_m','box_y_m','destination_reference_y_m')];limits=((min(xs)-.25,max(xs)+.25),(min(ys)-.25,max(ys)+.25))
    fig,axes=plt.subplots(1,3,figsize=(15,7),sharex=True,sharey=True)
    for ax,policy in zip(axes,POLICIES):
        for path in [p for p in paths if p['policy']==policy]:
            rows=path['rows'];color=colors(cases.index(path['case_id'])%20)
            ax.plot([r['root_x_m'] for r in rows],[r['root_y_m'] for r in rows],color=color,alpha=.45,linestyle='--',linewidth=.7)
            ax.plot([r['box_x_m'] for r in rows],[r['box_y_m'] for r in rows],color=color,alpha=.8,linewidth=1,label=path['case_id'])
            ax.scatter(*path['destination_reference_xy_m'],marker='s',s=30,color='blue',alpha=.65)
            ax.scatter(*path['box_start_xy_m'],marker='^',s=18,color=color)
            end=rows[-1];ax.scatter(end['box_x_m'],end['box_y_m'],marker='o' if path['existing_success'] else 'x',s=16,color=color)
        ax.set(title=policy,xlabel='Simulation world X (m)',ylabel='Simulation world Y (m)',xlim=limits[0],ylim=limits[1]);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    fig.suptitle('All20 frozen trials per model | solid: box; dashed: robot; triangle: box start\nBlue square: bin center reference; o: success; x: failed endpoint; identical simulation metre axes')
    fig.text(.5,.01,CAPTION,ha='center',va='bottom',fontsize=7)
    fig.tight_layout(rect=(0,.10,1,.94));fig.savefig(output/'all20_paths.png',dpi=160);fig.savefig(output/'all20_paths.pdf');plt.close(fig)
    first=cases[0];subset=[next(p for p in paths if p['policy']==policy and p['case_id']==first) for policy in POLICIES]
    fig,axes=plt.subplots(2,3,figsize=(15,9),gridspec_kw={'height_ratios':[2,1]},sharex='row')
    for column,path in enumerate(subset):
        rows=path['rows'];ax=axes[0,column];ax.plot([r['root_x_m'] for r in rows],[r['root_y_m'] for r in rows],'--',label='robot');ax.plot([r['box_x_m'] for r in rows],[r['box_y_m'] for r in rows],label='box')
        ax.scatter(*path['destination_reference_xy_m'],marker='s',s=65,color='blue',label='bin center (audit reference)');ax.scatter(*path['box_start_xy_m'],marker='^',s=50,color='black',label='box start')
        ax.set(title=path['policy'],xlabel='Simulation world X (m)',ylabel='Simulation world Y (m)',xlim=limits[0],ylim=limits[1]);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2);ax.legend()
        timeline=axes[1,column];end=rows[-1]['sim_seconds'];timeline.hlines(0,0,end,color='grey');text=[]
        for i,(name,tick) in enumerate(path['events'].items()):
            if tick is None:continue
            seconds=tick/50;timeline.scatter(seconds,i,color='tab:blue');timeline.annotate(name,(seconds,i),xytext=(2,3),textcoords='offset points',fontsize=7);text.append(f'{name}={seconds:.2f}s')
        timeline.set(xlabel='Simulation time (s)',ylabel='Existing event',yticks=[],xlim=(0,max(p['rows'][-1]['sim_seconds'] for p in subset)+.2));timeline.grid(axis='x',alpha=.2)
        timeline.set_title(('SUCCESS' if path['existing_success'] else 'FAILED')+f'; terminal={end:.2f}s\n'+','.join(path['termination_reasons']),fontsize=9)
    fig.suptitle('First frozen case: '+first+' | paths and existing event times; failures are not full-task speed comparisons');fig.text(.5,.01,CAPTION,ha='center',va='bottom',fontsize=7);fig.tight_layout(rect=(0,.09,1,.96));fig.savefig(output/'first_case_path_timeline.png',dpi=160);fig.savefig(output/'first_case_path_timeline.pdf');plt.close(fig)


def run(plan_path,summary_path,audits,analysis_path,output,*,plotter=draw):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh path figure output required')
    output.mkdir(parents=True);report={'schema':SCHEMA,'status':'failed','new_task_success_judgment':False,'GPU_started':False}
    try:
        plan,records,inputs,context=prepare_inputs(plan_path,summary_path,audits,analysis_path)
        paths=[stream_path(record,output) for record in records]
        plotter(paths,plan['case_ids'],output)
        common=[cid for cid in plan['case_ids'] if all(p['existing_success'] for p in paths if p['case_id']==cid)]
        speeds=[{'case_id':cid,'sim_seconds_by_policy':{p['policy']:records[i]['summary']['task_completion_sim_seconds_success_only'] for i,p in enumerate(paths) if p['case_id']==cid}} for cid in common]
        for record in records:
            st=record['trace'].stat()
            if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=record['trace_stat']:raise ValueError('Streamed raw trace metadata changed during export')
        for path,digest in inputs.items():
            if sha(path)!=digest:raise ValueError('Bound metadata changed during path export')
        outputs={str(p):sha(p) for p in output.rglob('*') if p.is_file()}
        required=['all20_paths.png','all20_paths.pdf','first_case_path_timeline.png','first_case_path_timeline.pdf']
        if len([p for p in output.rglob('*.csv')])!=60 or any(not (output/name).is_file() for name in required):raise ValueError('All60CSV+4figureoutputs required')
        inputs.update({str(record['trace']):record['expected_trace_sha256'] for record in records})
        report.update(status='completed',inputs_sha256=inputs,outputs_sha256=outputs,measurement_source=context,episodes=60,cases=20,policies=list(POLICIES),first_frozen_case=plan['case_ids'][0],existing_case_results=[{k:p[k] for k in ('policy','case_id','existing_success','events','termination_reasons','csv_path','destination_reference_xy_m','box_start_xy_m')} for p in paths],common_success_cases_only_fulltask_speed=speeds,all60_raw_trace_SHA_stream_verified=True,figure_caption=CAPTION,destination_reference_scope='First raw trace blue_sorting_bin.pose[:2]; post-hoc audit truth only; not policy input or new arrival verdict',tool_sha256=sha(__file__),scope='Measured simulation metres/seconds;existingauditedscoresunchanged.Nofailedtrialsrankedbyfulltaskspeed.Nocausalorhardwareclaim.')
    except Exception as error:report['error']=repr(error)
    with (output/'transport_paths_result.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--benchmark',required=True);p.add_argument('--audits',nargs='+',required=True);p.add_argument('--transport-analysis',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=run(a.plan,a.benchmark,a.audits,a.transport_analysis,a.output);print(json.dumps({'status':result['status'],'output':a.output}));return 0 if result['status']=='completed' else 2

if __name__=='__main__':raise SystemExit(main())
