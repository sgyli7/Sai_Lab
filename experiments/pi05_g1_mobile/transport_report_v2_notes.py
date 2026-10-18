"""中文长程综合概览，仅消费完整60回合既有证据；不运行策略或改变判据。

--benchmark SUMMARY --audits AUDIT [AUDIT ...] --transport ANALYSIS
--output-dir NEW [--report-dir EXISTING_REPORT_V2]
Charts use Matplotlib if available; absence is explicitly recorded, never faked.
"""
from __future__ import annotations
import argparse,csv,html,itertools,json,math,os
from pathlib import Path
from collections import Counter
import numpy as np
from . import report_v2
from .benchmark_v2 import distribution,wilson,paired_statistics,cluster_statistics

LABELS={'gr00t_n16':'GR00T N1.6 原生','pi05_mobile':'π0.5 本轮微调','official_pi05_base':'π0.5 官方基座'}
sha=report_v2.sha
read=report_v2.read


def same_number(a,b,name):
    if type(a) not in (int,float) or type(b) not in (int,float) or not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>1e-6:raise ValueError('运输数值不一致: '+name)


def trace_metrics(path,event_tick):
    first=None;event=None;count=0;minimum=float('inf');final=None
    with Path(path).open() as stream:
        for line in stream:
            row=json.loads(line);tick=row['control_tick']
            if tick!=count:raise ValueError('运输原始轨迹不连续')
            if first is None:first=row['monotonic_ns']
            if tick==event_tick:event=(row['monotonic_ns']-first)/1e9
            truth=row['acceptance_truth_only'];box=np.asarray(truth['brown_box']['pose'][:2],float);target=np.asarray(truth['blue_sorting_bin']['pose'][:2],float)
            error=float(np.linalg.norm(box-target))
            if not math.isfinite(error):raise ValueError('运输物理位置非有限')
            minimum=min(minimum,error);final=error;count+=1
    return {'count':count,'event_wall':event,'final':final,'minimum':minimum}


def load(benchmark,audits,transport):
    summary,episodes,videos=report_v2.load_evidence(benchmark,audits)
    document=read(transport);source=document.get('source_benchmark')
    if document.get('schema')!='pi05_g1_mobile_transport_analysis_v1' or source!={'path':str(Path(benchmark).resolve()),'sha256':sha(benchmark)}:raise ValueError('运输分析必须绑定同一完整benchmark')
    records=document.get('episodes',[])
    if len(records)!=60:raise ValueError('运输分析必须完整60回合；不把缺失计为失败')
    joined={};bound={}
    for row in records:
        key=(row.get('policy'),row.get('case_id'))
        if key not in episodes or key in joined:raise ValueError('运输case不匹配或重复')
        base=episodes[key];audit=base['_audit'];result_path=str(Path(base['result_path']).resolve());folder=Path(result_path).parent;accept=audit['recomputed_acceptance']
        if row.get('schema')!='pi05_g1_mobile_transport_episode_v1' or row.get('infrastructure_completed') is not True or row.get('diagnostic_prefix_only') is not False or row.get('result_status')!='completed':raise ValueError('运输回合证据未完成')
        if row.get('full_task_verdict') is not base['success'] or row.get('full_task_timeout') is not base['timeout'] or row.get('original_result_success') is not base['success']:raise ValueError('运输完整任务判定与benchmark不一致')
        for name in ['result.json','config.json','trajectory.jsonl','runtime_identity.json']:
            path=result_path if name=='result.json' else str(folder/name)
            if row.get('inputs_sha256',{}).get(path)!=audit['inputs_sha256'].get(path) or path not in audit['inputs_sha256']:raise ValueError('运输/审计/benchmark原始SHA不匹配: '+name)
            if name!='runtime_identity.json' and summary['inputs_sha256'].get(path)!=audit['inputs_sha256'][path]:raise ValueError('benchmark未绑定同一原始输入: '+name)
        for path,digest in row['inputs_sha256'].items():
            if path in bound and bound[path]!=digest:raise ValueError('运输输入SHA冲突')
            bound[path]=digest
        for flag in ['grasped','stable_placement','fell','dropped']:
            if row.get('observed_physical_flags',{}).get(flag) is not base[flag]:raise ValueError('运输物理标记与独立审计不一致')
        rule={k:v for k,v in accept['placement_rule'].items() if k.startswith('minimum_')}
        if row.get('existing_carry_rule')!=rule or rule.get('minimum_carried_displacement_m')!=4 or rule.get('minimum_carry_turn_degrees')!=60 or rule.get('minimum_source_target_distance_m')!=5:raise ValueError('运输必须使用原冻结4m/60deg判据')
        T=accept['carry_qualified_control_tick']
        if row.get('first_grasp_control_tick')!=accept.get('first_grasp_control_tick'):raise ValueError('抓起事件与独立审计不一致')
        if T is not None and (not base['grasped'] or type(accept.get('first_grasp_control_tick')) is not int or accept['first_grasp_control_tick']>T):raise ValueError('携物达标缺少先行有效抓取')
        if row.get('carry_qualified_control_tick')!=T or row.get('first_4m_60deg_qualified_tick')!=T or (T is not None and (type(T) is not int or not 0<=T<=base['_result']['completed_control_ticks'])):raise ValueError('首次携物事件与独立审计不一致')
        if row.get('transport_condition_is_not_arrival_or_task_success') is not True:raise ValueError('必须明确携物阶段不等于到达或放置')
        for name in ['source_target_distance_m','max_carried_box_displacement_m','contact_qualified_carried_path_m','carry_turn_degrees']:same_number(row[name],accept[name],name)
        observed=trace_metrics(folder/'trajectory.jsonl',T)
        if observed['count']!=base['_result']['completed_control_ticks']+1 or row.get('control_ticks')!=observed['count']-1 or row.get('physics_integrations')!=4*(observed['count']-1):raise ValueError('运输完整轨迹计数不一致')
        error=row['box_target_actor_origin_xy_error_m']
        same_number(error['final'],observed['final'],'final XY');same_number(error['minimum'],observed['minimum'],'minimum XY')
        event=row.get('first_transport_event')
        if T is None:
            if event is not None:raise ValueError('未达标回合不得有达标速度')
        else:
            if not isinstance(event,dict) or event.get('control_tick')!=T:raise ValueError('首次携物时间事件缺失')
            same_number(event['sim_seconds'],T/50,'transport sim');same_number(event['actual_observation_wall_seconds'],observed['event_wall'],'transport wall')
        joined[key]={'policy':key[0],'case_id':key[1],'cluster_id':base['cluster_id'],'full_success':base['success'],'carry_qualified':T is not None,
            **{flag:base[flag] for flag in ['grasped','stable_placement','dropped','fell','timeout']},'carry_qualified_tick':T,
            'first_carry_qualified_sim_seconds':None if T is None else T/50,'first_carry_qualified_wall_seconds':observed['event_wall'],
            'minimum_actor_origin_xy_error_m':observed['minimum'],'final_actor_origin_xy_error_m':observed['final'],
            'full_task_sim_seconds_success_only':base['task_sim_seconds'] if base['success'] else None,'full_task_wall_seconds_success_only':base['task_wall_seconds'] if base['success'] else None,
            'warm_inference_seconds':base['warm_inference_seconds'],'warm_inference_median_seconds':distribution(base['warm_inference_seconds'])['median'],'warm_inference_p95_seconds':distribution(base['warm_inference_seconds'])['p95'],'result_path':result_path}
    if set(joined)!=set(episodes):raise ValueError('运输/审计/benchmark非同一完整60回合')
    for path,digest in bound.items():
        if sha(path)!=digest:raise ValueError('运输分析原始输入已改变: '+path)
    cases=summary['case_ids'];policies=summary['policy_order'];clusters={c:str(joined[(policies[0],c)]['cluster_id']) for c in cases}
    if Counter(Counter(clusters.values()).values())!={4:5}:raise ValueError('必须5位置×4种子重复')
    if any(str(joined[(p,c)]['cluster_id'])!=clusters[c] for p in policies for c in cases):raise ValueError('同case位置组不一致')
    return summary,joined,videos,clusters


def aggregate(summary,rows,clusters):
    cases=summary['case_ids'];policies=summary['policy_order'];table={};pairs={}
    for policy in policies:
        records=[rows[(policy,c)] for c in cases];success=sum(r['full_success'] for r in records);carry=sum(r['carry_qualified'] for r in records)
        table[policy]={'label':LABELS.get(policy,policy),'completed':20,'full_task_success':success,'full_task_success_rate':success/20,'full_task_wilson95ci':wilson(success,20),'full_task_16_of20_threshold_met':success>=16,
            'carry_qualified':carry,'carry_qualified_rate':carry/20,'carry_qualified_wilson95ci':wilson(carry,20),
            'counts':{flag:sum(r[flag] for r in records) for flag in ['grasped','stable_placement','dropped','fell','timeout']},
            'minimum_actor_origin_xy_error_m':distribution([r['minimum_actor_origin_xy_error_m'] for r in records]),'final_actor_origin_xy_error_m':distribution([r['final_actor_origin_xy_error_m'] for r in records]),
            'warm_inference_seconds':distribution([x for r in records for x in r['warm_inference_seconds']])}
    for first,second in itertools.combinations(policies,2):
        a=[rows[(first,c)]['carry_qualified'] for c in cases];b=[rows[(second,c)]['carry_qualified'] for c in cases]
        matched=[c for c in cases if rows[(first,c)]['carry_qualified'] and rows[(second,c)]['carry_qualified']]
        complete=[c for c in cases if rows[(first,c)]['full_success'] and rows[(second,c)]['full_success']]
        pair={'first':first,'second':second,'carry_qualified_paired_statistics':paired_statistics(a,b,seed=42),'physical_position_cluster':cluster_statistics(a,b,[clusters[c] for c in cases],seed=42),
            'matched_carry_case_ids':matched,'matched_carry_count':len(matched),'transport_speed_available':bool(matched),
            'matched_full_success_case_ids':complete,'matched_full_success_count':len(complete),'full_task_speed_available':bool(complete),
            'speed_scope':'Only same-case both-qualified transport or both-success full task. Empty=N/A; fast failure never wins. Difference is second minus first, conditional on both endpoints being reached.'}
        for field in ['sim','wall']:
            for policy,role in [(first,'first'),(second,'second')]:
                pair['matched_carry_'+role+'_'+field+'_seconds']=distribution([rows[(policy,c)]['first_carry_qualified_'+field+'_seconds'] for c in matched])
                pair['matched_full_task_'+role+'_'+field+'_seconds']=distribution([rows[(policy,c)]['full_task_'+field+'_seconds_success_only'] for c in complete])
            pair['matched_carry_'+field+'_difference_seconds_second_minus_first']=distribution([rows[(second,c)]['first_carry_qualified_'+field+'_seconds']-rows[(first,c)]['first_carry_qualified_'+field+'_seconds'] for c in matched])
            pair['matched_full_task_'+field+'_difference_seconds_second_minus_first']=distribution([rows[(second,c)]['full_task_'+field+'_seconds_success_only']-rows[(first,c)]['full_task_'+field+'_seconds_success_only'] for c in complete])
        pairs[first+'__'+second]=pair
    representatives=[{'kind':'first_frozen_case','case_id':cases[0]}]
    common=next((c for c in cases if all(rows[(p,c)]['carry_qualified'] for p in policies)),None)
    if common is not None and common!=cases[0]:representatives.append({'kind':'first_in_frozen_order_all_three_carry_qualified','case_id':common})
    return {'schema':'pi05_g1_mobile_transport_report_v1','status':'complete','complete':True,'evidence_qualified':True,'policy_order':policies,'case_ids':cases,'policies':table,'paired_comparisons':pairs,'representatives':representatives,
        'scope_zh':'仅仿真；5个物理位置×4种子重复，非20个独立场景。4m/60°携物阶段达标不等于到达目标、释放、稳定放置或完整任务成功。16/20门槛仅用于完整任务。阶段不确定性不替换原成功判据。',
        'causal_limit_zh':'不自动宣称架构或训练因果，不宣布总体赢家。单回合、最接近距离及接触代理均不可冒充完整成功。',
        'timing_zh':'首次携物时间：trace row0至既定首次carry qualified。wall包括观测后的物理/渲染/录像与推理暂停，不含row0之前冷启动；warm inference为模型推理计算延迟，非整次RPC。仿真时钟与实际墙钟分开报告。',
        'xy_error_zh':'箱体与同期目标容器actor origin的世界XY距离；报告最接近和最终连续数值，没有新增到达阈值。',
        'episodes':[rows[(p,c)] for c in cases for p in policies]}


def plot(report,output):
    try:
        import matplotlib
        matplotlib.use('Agg');import matplotlib.pyplot as plt
    except ImportError as error:return {'status':'unavailable','reason':str(error),'artifacts':[]}
    policies=report['policy_order'];fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,count,ci,title in [(axes[0],'full_task_success','full_task_wilson95ci','Complete task /20'),(axes[1],'carry_qualified','carry_qualified_wilson95ci','Existing carry stage /20 (not task success)')]:
        ax.bar(range(3),[report['policies'][p][count] for p in policies],color=['#4776aa','#d48b32','#6b9b6e'])
        for index,p in enumerate(policies):low,high=report['policies'][p][ci];ax.plot([index,index],[20*low,20*high],color='black',marker='_')
        ax.set(title=title,ylim=(0,21),ylabel='Cases; Wilson95% CI')
    for i,p in enumerate(policies):
        d=report['policies'][p]['warm_inference_seconds']
        if d['count']:
            axes[2].scatter([i],[d['median']*1000],marker='o',label='Median' if i==0 else None)
            axes[2].scatter([i],[d['p95']*1000],marker='^',label='P95' if i==0 else None)
    axes[2].legend()
    axes[0].axhline(16,color='gray',linestyle='--',linewidth=1)
    axes[2].set(title='Warm inference median / P95',ylabel='Milliseconds')
    for ax in axes:ax.set_xticks(range(3),policies,rotation=12,ha='right')
    fig.suptitle('Simulation only; 5 positions x 4 seed repeats; native GR00T vs PyTorch Pi')
    artifacts=[]
    for suffix in ['png','pdf']:
        p=output/('transport_overview.'+suffix);fig.savefig(p,dpi=160);artifacts.append({'path':p.name,'sha256':sha(p)})
    plt.close(fig);return {'status':'rendered','artifacts':artifacts,'pdf_vector':True}


def generate(benchmark,audits,transport,output,report_dir=None):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('新输出目录，禁止覆盖')
    summary,rows,videos,clusters=load(benchmark,audits,transport);report=aggregate(summary,rows,clusters)
    inputs={str(Path(p).resolve()):sha(p) for p in [benchmark,transport,*audits]};montages=[]
    if report_dir is not None:
        folder=Path(report_dir).resolve();receipt_path=folder/'report_receipt.json';receipt=read(receipt_path)
        if receipt.get('schema')!='pi05_g1_mobile_report_receipt_v2' or receipt.get('benchmark_sha256')!=sha(benchmark) or receipt.get('audits')!={str(Path(p).resolve()):sha(p) for p in audits}:raise ValueError('report_v2蒙太奇不属于同一证据')
        for name,digest in receipt['artifacts_sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('report_v2产物已改变')
        montages=receipt.get('montages',[]);inputs[str(receipt_path)]=sha(receipt_path)
    output.mkdir(parents=True);report['plot']=plot(report,output);report['inputs_sha256']=inputs;report['tool_sha256']=sha(__file__);report['report_v2_montages']=montages
    for item in videos:item['link']=os.path.relpath(item['original_absolute_path'],output)
    report['raw_videos']=videos
    fields=['policy','case_id','cluster_id','full_success','carry_qualified','grasped','stable_placement','dropped','fell','timeout','carry_qualified_tick','first_carry_qualified_sim_seconds','first_carry_qualified_wall_seconds','minimum_actor_origin_xy_error_m','final_actor_origin_xy_error_m','full_task_sim_seconds_success_only','full_task_wall_seconds_success_only','warm_inference_median_seconds','warm_inference_p95_seconds','result_path']
    with (output/'episodes.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(report['episodes'])
    esc=html.escape;chunks=['<!doctype html><meta charset="utf-8"><title>G1 长程携物综合概览</title><style>body{font:16px sans-serif;margin:30px}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #bbb}video{max-width:95%}</style><h1>G1 长程携物综合概览</h1>']
    chunks.extend('<p>'+esc(report[k])+'</p>' for k in ['scope_zh','causal_limit_zh','timing_zh','xy_error_zh'])
    chunks.append('<table><tr><th>策略</th><th>完整成功 /20；16门槛</th><th>携物阶段 /20；Wilson95%区间</th><th>抓起代理 / 稳定放置 / 地面掉落 / 跌倒 / 剩余未完成(timeout)</th><th>最接近 / 最终 XY 误差中位数(m)</th><th>warm 推理中位数 / P95(ms)</th></tr>')
    for policy in report['policy_order']:
        t=report['policies'][policy];lat=t['warm_inference_seconds'];fmt=lambda x:'N/A' if x is None else f'{x:.3f}'
        chunks.append('<tr><td>'+esc(t['label'])+'</td><td>'+str(t['full_task_success'])+'/20；'+('达标' if t['full_task_16_of20_threshold_met'] else '未达标')+'</td><td>'+str(t['carry_qualified'])+'/20；['+f"{t['carry_qualified_wilson95ci'][0]*100:.1f}%, {t['carry_qualified_wilson95ci'][1]*100:.1f}%"+']</td><td>'+' / '.join(str(t['counts'][k]) for k in ['grasped','stable_placement','dropped','fell','timeout'])+'</td><td>'+fmt(t['minimum_actor_origin_xy_error_m']['median'])+' / '+fmt(t['final_actor_origin_xy_error_m']['median'])+'</td><td>'+fmt(None if lat['median'] is None else lat['median']*1000)+' / '+fmt(None if lat['p95'] is None else lat['p95']*1000)+'</td></tr>')
    chunks.append('</table><p>跌倒与地面掉落可同时出现；timeout 是“未完整成功、且未跌倒、未地面掉落”的剩余未完成标记，与后二者互斥，不单独证明时间上限已到。其他诊断原因可重叠，不能求和作为总失误数；零地面掉落不等于零失误。抓起代理仅表示曾在50Hz采样中左右手均接触且箱体原点较初始抬升超过3cm，不保证持续安全抓持。</p><p><a href="episodes.csv">逐回合CSV</a> · <a href="transport_report.json">完整统计JSON</a></p>')
    if report['plot']['status']=='rendered':chunks.append('<img width="95%" src="transport_overview.png"><p><a href="transport_overview.pdf">矢量PDF</a></p>')
    else:chunks.append('<p>绘图库不可用，未生成图：'+esc(report['plot']['reason'])+'</p>')
    chunks.append('<h2>同场配对速度与统计不确定性</h2><p>仅5个位置cluster，双侧sign-flip最小p=2/32=0.0625；不把20次种子重复视作20个独立场景。</p>')
    for pair in report['paired_comparisons'].values():
        chunks.append('<h3>'+esc(LABELS.get(pair['first'],pair['first'])+' vs '+LABELS.get(pair['second'],pair['second']))+'</h3><p>共同携物达标 '+str(pair['matched_carry_count'])+' 回合；共同完整成功 '+str(pair['matched_full_success_count'])+' 回合。'+('运输速度 N/A。' if not pair['transport_speed_available'] else '')+('完整任务速度 N/A。' if not pair['full_task_speed_available'] else '')+'</p><pre>'+esc(json.dumps(pair,ensure_ascii=False,indent=2))+'</pre>')
    chunks.append('<h2>代表回合：同一冻结case，禁止各挑最佳</h2><pre>'+esc(json.dumps(report['representatives'],ensure_ascii=False,indent=2))+'</pre>')
    for representative in report['representatives']:
        case=representative['case_id'];chunks.append('<h3>'+esc(case)+'</h3>')
        for video in videos:
            if video['case_id']==case:chunks.append('<a href="'+esc(video['link'],quote=True)+'">'+esc(LABELS.get(video['policy'],video['policy'])+' '+video['view'])+'</a> · ')
    if not montages:chunks.append('<p>未提供现有report_v2蒙太奇；上方与下方均链接原始双路全视频，不生成替代录像。</p>')
    for montage in montages:chunks.append('<p>report_v2同场蒙太奇 '+esc(montage['case_id'])+'</p><video controls src="'+esc(os.path.relpath(montage['path'],output),quote=True)+'"></video>')
    chunks.append('<h2>全部120路原始视频</h2>')
    for video in videos:chunks.append('<p><a href="'+esc(video['link'],quote=True)+'">'+esc(video['policy']+' / '+video['case_id']+' / '+video['view'])+'</a></p>')
    (output/'index.html').write_text('\n'.join(chunks))
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('概览生成期间输入改变')
    with (output/'transport_report.json').open('x') as f:f.write(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    receipt={'schema':'pi05_g1_mobile_transport_report_receipt_v1','status':'completed','summary_sha256':sha(output/'transport_report.json'),'inputs_sha256':inputs,'tool_sha256':sha(__file__),'artifacts_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file()},'gpu_started':False,'raw_evidence_modified':False}
    with (output/'receipt.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--audits',nargs='+',type=Path,required=True);p.add_argument('--transport',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--report-dir',type=Path);a=p.parse_args();r=generate(a.benchmark,a.audits,a.transport,a.output_dir,a.report_dir);print(json.dumps({'status':r['status'],'episodes':len(r['episodes'])}))

if __name__=='__main__':main()
