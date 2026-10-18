"""CPU-only completed mobile benchmark report; never evaluates policies.

python -m experiments.pi05_g1_mobile.report --benchmark SUMMARY.json \
  --audits AUDIT.json [AUDIT2.json ...] --output-dir NEW_DIR [--montage]
Requires matplotlib; montage additionally requires ffmpeg and Pillow.
Original videos remain untouched; HTML links are relative, with absolute-path/SHA
provenance in video_index.json for relinking an archived evidence tree.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess

FLAGS=('success','grasped','stable_placement','dropped','fell','timeout')
METRICS=('max_carried_box_displacement_m','contact_qualified_carried_path_m','carry_turn_degrees')
COLUMNS=('policy','case_id',*FLAGS,*METRICS,'task_sim_seconds','task_wall_seconds',
         'episode_sim_seconds','episode_wall_seconds','cold_process_to_frame0_seconds',
         'RPC_pause_total_seconds','model_load_seconds','result_path')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))


def audit_identity(audit):
    payload={k:v for k,v in audit.items() if k!='audit_sha256'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


def load_evidence(benchmark_path,audit_paths):
    """Require complete Cartesian evidence and independently hash-bound audit pass."""
    summary=read(benchmark_path)
    if (summary.get('schema')!='pi05_g1_mobile_benchmark_summary_v1' or
        summary.get('status')!='complete' or summary.get('complete') is not True or
        summary.get('evidence_qualified') is not True or summary.get('errors')):
        raise ValueError('Only completed evidence-qualified benchmark accepted')
    policies=summary['policy_order'];cases=summary['case_ids']
    if len(policies)!=3 or len(set(policies))!=3 or len(cases)!=20 or len(set(cases))!=20:
        raise ValueError('Exactly three policies and20 frozen cases required')
    audits={}
    for path in audit_paths:
        document=read(path)
        group=document['episodes'] if document.get('schema')=='pi05_g1_mobile_episode_audits_v1' else [document]
        if document.get('passed') is not True:raise ValueError('Independent audit report not passed')
        for audit in group:
            if (audit.get('schema')!='pi05_g1_mobile_episode_cpu_audit_v1' or
                audit.get('passed') is not True or audit.get('autonomous_policy_trial') is not True or
                audit.get('audit_sha256')!=audit_identity(audit)):
                raise ValueError('Missing/invalid independent episode audit identity')
            key=str(Path(audit['result_path']).resolve())
            if key in audits:raise ValueError('Duplicate independent episode audit')
            audits[key]=audit
    episodes={};videos=[]
    for episode in summary['episodes']:
        key=(episode['policy'],episode['case_id'])
        if key in episodes or key[0] not in policies or key[1] not in cases:raise ValueError('Duplicate/unplanned episode')
        result_path=Path(episode['result_path']).resolve();audit=audits.get(str(result_path))
        if audit is None:raise ValueError('Missing independent audit: '+str(result_path))
        digest=sha(result_path)
        if audit['inputs_sha256'].get(str(result_path))!=digest or summary['inputs_sha256'].get(str(result_path))!=digest:
            raise ValueError('Raw result not bound identically by benchmark and audit')
        result=read(result_path)
        if result.get('status')!='completed' or result.get('policy')!=key[0] or result.get('case_id')!=key[1]:
            raise ValueError('Raw result identity/status changed')
        if any(episode[k]!=audit['recomputed_acceptance'][k] for k in (*FLAGS,*METRICS)):
            raise ValueError('Summary metrics differ from independent recomputation')
        row={**episode,'RPC_pause_total_seconds':sum(episode['RPC_pause_wall_seconds']),
             '_result':result,'_audit':audit}
        for stream,field in [('head','head_video_path'),('observer','observer_video_path')]:
            evidence=audit['checks']['videos']['streams'][field];path=Path(evidence['path']).resolve()
            if evidence['fps']!=50 or evidence['frames']!=result['completed_control_ticks']+1:
                raise ValueError('Audited stream frame/fps mismatch')
            if not path.is_file() or sha(path)!=evidence['sha256']:
                raise ValueError('Raw video changed after independent audit')
            videos.append({'policy':key[0],'case_id':key[1],'view':stream,'original_absolute_path':str(path),
                'sha256':evidence['sha256'],'fps':50,'frames':evidence['frames'],
                'raw_result_path':str(result_path),'raw_result_sha256':digest,
                'audit_sha256':audit['audit_sha256']})
        episodes[key]=row
    if set(episodes)!={(p,c) for p in policies for c in cases}:raise ValueError('Incomplete60 episode evidence')
    for policy in policies:
        actual=sum(episodes[(policy,c)]['success'] for c in cases)
        if actual!=summary['policies'][policy]['formal_success_count']:raise ValueError('Aggregate score mismatch')
    return summary,episodes,videos


def representatives(summary,episodes):
    primary=summary['case_ids'][0]
    common=sorted(c for c in summary['case_ids'] if all(episodes[(p,c)]['success'] for p in summary['policy_order']))
    result=[{'kind':'first_frozen_case','case_id':primary}]
    if common and common[0]!=primary:result.append({'kind':'smallest_case_id_all_three_success','case_id':common[0]})
    return result


def chart(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    policies=summary['policy_order'];fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    successes=[summary['policies'][p]['formal_success_count'] for p in policies]
    ci=[summary['policies'][p]['success_wilson95ci'] for p in policies]
    # Wilson intervals need not contain the observed count at boundary counts.
    axes[0].bar(range(3),successes,color=['#4776aa','#d48b32','#6b9b6e'])
    for i,(low,high) in enumerate(ci):axes[0].plot([i,i],[low*20,high*20],color='black',marker='_')
    axes[0].set(title='Success count /20; Wilson95% interval',ylim=(0,21),ylabel='Cases')
    for j,flag in enumerate(('dropped','fell','timeout')):
        axes[1].bar([i+(j-1)*.24 for i in range(3)],
            [summary['policies'][p]['counts_observed'][flag] for p in policies],.24,label=flag)
    axes[1].set(title='Failure flags (overlap allowed)',ylim=(0,21),ylabel='Cases');axes[1].legend()
    for axis in axes:axis.set_xticks(range(3),policies,rotation=12,ha='right')
    fig.savefig(output,dpi=150);plt.close(fig)


def frame_caption(policy,view,row,tick,count,wall):
    """Retrospective outcome, clipped observed clocks, explicit held-last-frame."""
    local=min(tick,count-1);held=tick>=count
    caption=(policy+' | '+view+'\nFinal outcome: '+('success' if row['success'] else 'failure')+
        f"\nsim {local/50:.2f}s | actual obs-wall {wall[local]:.2f}s"+
        '\nSIM50fps playback; physics pauses omitted')
    qualified=row.get('qualified_stability_endpoint_sim_seconds');event=''
    if qualified is not None and local/50>=qualified:event='QUALIFIED at sim '+f'{qualified:.2f}s'
    elif row.get('first_grasp_sim_seconds') is not None and local/50>=row['first_grasp_sim_seconds']:event='Verified grasp observed'
    if held:event='END OF RECORDING / LAST FRAME HELD\nStopped sim '+f"{local/50:.2f}s; obs-wall {wall[local]:.2f}s"
    return caption,event


def montage(case,policies,episodes,output,width=480,height=360):
    """CPU ffmpeg decode/encode with Pillow labels; no drawtext dependency."""
    from PIL import Image,ImageDraw,ImageFont
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    frames=max(episodes[(p,case)]['_result']['completed_control_ticks']+1 for p in policies)
    sources=[];decoders=[];writer=None
    try:
        for view,field in [('observer','observer_video_path'),('head','head_video_path')]:
            for policy in policies:
                row=episodes[(policy,case)];raw=row['_result'];source=Path(raw[field]);count=raw['completed_control_ticks']+1
                trace_path=Path(row['result_path']).parent/'trajectory.jsonl'
                if trace_path.is_file():
                    # Trace already independently audited. Hash again before rendering.
                    if row.get('_audit') and row['_audit']['inputs_sha256'].get(str(trace_path))!=sha(trace_path):
                        raise ValueError('Representative trace changed after audit')
                    trace=[json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
                    wall=[(r['monotonic_ns']-trace[0]['monotonic_ns'])/1e9 for r in trace[:count]]
                    if len(wall)!=count:raise ValueError('Trace frame coverage differs')
                else:raise ValueError('Representative needs audited per-frame monotonic trace')
                proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(source),
                    '-vf',f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
                    '-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
                decoders.append({'proc':proc,'count':count,'last':None,'row':row,'policy':policy,'view':view,'wall':wall})
                sources.append({'path':str(source),'sha256':sha(source),'frames':count,'trace_path':str(trace_path),'trace_sha256':sha(trace_path)})
        writer=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{width*3}x{height*2}','-r','50','-i','pipe:0','-an','-c:v','libx264','-threads','2',
            '-preset','fast','-crf','23','-pix_fmt','yuv420p',str(output)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        font=ImageFont.truetype(str(font_path),14) if font_path.exists() else ImageFont.load_default()
        for tick in range(frames):
            canvas=Image.new('RGB',(width*3,height*2))
            for index,cell in enumerate(decoders):
                held=tick>=cell['count'];local=min(tick,cell['count']-1)
                if not held:
                    pixels=cell['proc'].stdout.read(width*height*3)
                    if len(pixels)!=width*height*3:raise ValueError('Raw video ended before audited frame count')
                    cell['last']=Image.frombytes('RGB',(width,height),pixels)
                image=cell['last'].copy();draw=ImageDraw.Draw(image);row=cell['row']
                draw.rectangle((0,0,width,95),fill='black')
                caption,event=frame_caption(cell['policy'],cell['view'],row,tick,cell['count'],cell['wall'])
                draw.multiline_text((7,4),caption,font=font,fill='white',spacing=2)
                if event:
                    draw.rectangle((0,height-52,width,height),fill='black');draw.multiline_text((7,height-48),event,font=font,fill='yellow')
                canvas.paste(image,((index%3)*width,(index//3)*height))
            writer.stdin.write(canvas.tobytes())
        writer.stdin.close();writer.stdin=None
        _,stderr=writer.communicate(timeout=120)
        if writer.returncode:raise RuntimeError('ffmpeg encoding failed: '+stderr.decode(errors='replace')[-4000:])
    finally:
        if writer is not None and writer.poll() is None:writer.kill();writer.wait()
        for cell in decoders:
            proc=cell['proc']
            if proc.stdout:proc.stdout.close()
            if proc.poll() is None:proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
    return {'case_id':case,'path':str(output),'sha256':sha(output),'fps':50,'frames':frames,'sources':sources,
        'timing':'Simulation-time50fps playback; inference/startup pauses are omitted, never wall-clock1x',
        'short_stream_semantics':'Last original frame held with explicit END OF RECORDING and fixed stopped sim/wall time',
        'selection':'Same frozen case for every policy; Final outcome label is retrospective, not a live done signal'}


def generate(benchmark_path,audit_paths,output,make_montage=False):
    summary,episodes,videos=load_evidence(benchmark_path,audit_paths)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    def rel(path):return os.path.relpath(Path(path).resolve(),output)
    for video in videos:video['relative_link']=rel(video['original_absolute_path'])
    index={'schema':'pi05_g1_mobile_report_video_index_v1','benchmark_path':str(Path(benchmark_path).resolve()),
           'benchmark_sha256':sha(benchmark_path),'audits':{str(Path(p).resolve()):sha(p) for p in audit_paths},
           'videos':videos,'originals_modified':False,'relink_note':'Relative links preserve local evidence-tree layout; original absolute paths retained for relocation.'}
    (output/'video_index.json').write_text(json.dumps(index,indent=2)+'\n')
    ordered=[episodes[(p,c)] for c in summary['case_ids'] for p in summary['policy_order']]
    with (output/'episodes.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS,extrasaction='ignore');writer.writeheader();writer.writerows(ordered)
    with (output/'video_index.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(videos[0]));writer.writeheader();writer.writerows(videos)
    chart(summary,output/'outcomes.png')
    selected=representatives(summary,episodes);montages=[]
    if make_montage:
        for i,item in enumerate(selected):montages.append({**montage(item['case_id'],summary['policy_order'],episodes,output/f'comparison_{i}.mp4'),'selection_rule':item['kind']})
    esc=html.escape
    chunks=['<!doctype html><html><head><meta charset="utf-8"><title>G1 mobile simulation benchmark</title><style>body{font-family:sans-serif;max-width:1500px;margin:2em auto}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #bbb;padding:5px}video{max-width:100%}</style></head><body>',
        '<h1>G1 mobile simulation benchmark</h1><p>Completed benchmark plus independent episode audits. Simulation evidence only; no physical robot capability claim. Five physical positions × four seed repeats. Video playback is simulation50fps; physics pauses during inference. No wall-clock1× claim.</p>',
        '<img src="outcomes.png" alt="Success counts with Wilson intervals and overlapping failure flags">',
        '<p><a href="episodes.csv">Episode CSV</a> · <a href="video_index.csv">Full original-video CSV</a> · <a href="video_index.json">Paths and SHA provenance</a></p>',
        '<h2>Efficiency: benchmark matched-success comparisons only</h2>']
    for pair in summary.get('paired_comparisons',{}).values():
        chunks.append('<h3>'+esc(pair['first']+' vs '+pair['second'])+'</h3>')
        chunks.append('<pre>'+esc(json.dumps(pair,indent=2) if pair.get('matched_success_count',0)>0 else 'N/A: no both-success matched cases. Fast failures are not ranked as efficient.')+'</pre>')
    chunks.append('<h2>Representative cases (predefined selection)</h2><pre>'+esc(json.dumps(selected,indent=2))+'</pre>')
    for media in montages:chunks.append('<p>'+esc(media['selection_rule']+'; '+media['case_id'])+'</p><video controls src="'+esc(rel(media['path']),quote=True)+'"></video>')
    chunks.append('<h2>All60 episodes /120 original video streams</h2><p>Carry displacement and contact-qualified carried path are separate measured quantities. Failure flags may overlap. Final outcome labels are retrospective.</p><table><tr>'+''.join('<th>'+esc(k)+'</th>' for k in COLUMNS if k!='result_path')+'<th>Raw videos</th></tr>')
    byvideo={(v['policy'],v['case_id'],v['view']):v for v in videos}
    for row in ordered:
        chunks.append('<tr>'+''.join('<td>'+esc(str(row.get(k,'')))+'</td>' for k in COLUMNS if k!='result_path')+'<td>'+ ' · '.join('<a href="'+esc(byvideo[(row['policy'],row['case_id'],v)]['relative_link'],quote=True)+'">'+v+'</a>' for v in ('head','observer'))+'</td></tr>')
    chunks.append('</table><p>Physical interpretation and final scientific conclusions remain in the investigator report.</p></body></html>')
    (output/'index.html').write_text('\n'.join(chunks))
    receipt={'schema':'pi05_g1_mobile_report_receipt_v1','benchmark_sha256':sha(benchmark_path),'audits':index['audits'],'completed_cases':60,'raw_video_streams':120,'representatives':selected,'montages':montages,'tool_sha256':sha(__file__),
             'artifacts_sha256':{str(p.relative_to(output)):sha(p) for p in sorted(output.iterdir()) if p.is_file()},'no_raw_evidence_modified':True}
    (output/'report_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark',required=True,type=Path);p.add_argument('--audits',required=True,nargs='+',type=Path);p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--montage',action='store_true');a=p.parse_args()
    result=generate(a.benchmark,a.audits,a.output_dir,a.montage);print(json.dumps({'completed_cases':result['completed_cases'],'raw_video_streams':result['raw_video_streams']}))

if __name__=='__main__':main()
