"""CSV index from an already qualified final benchmark and bound video manifests.

Reads summary/manifest metadata only. Never reopens raw result/trace/videos,
selects outcomes, runs inference, or alters evidence. File SHAs in CSV are the
identities already validated by the completed benchmark aggregation.
"""
import argparse,csv,hashlib,json,re
from pathlib import Path
KEYS=('head_raw','observer_raw','head_wall_annotated','observer_wall_annotated')
WALL=('head_wall_clock','observer_wall_clock')
FIELDS=('policy','case_id','view','video_kind','playback_kind','video_path','video_sha256','fps','frames','raw_result_path','raw_result_sha256','video_manifest_path','video_manifest_sha256','benchmark_summary_path','benchmark_summary_sha256')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def hash64(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None

def make_index(summary_path,output):
 summary_path=Path(summary_path).resolve();summary=json.loads(summary_path.read_text());h=digest(summary_path);output=Path(output);sidecar=output.with_suffix('.provenance.json')
 if output.exists() or sidecar.exists():raise FileExistsError('CSV/index provenance output must be new')
 policies=summary.get('policies',{})
 if (summary.get('schema')!='pi05_g1_round02_multi_policy_benchmark_v2' or summary.get('complete') is not True
     or summary.get('evidence_qualified') is not True or summary.get('errors') or summary.get('infrastructure_incomplete')
     or len(policies) not in (3,4) or any(p.get('completed')!=20 for p in policies.values())):
  raise ValueError('Index requires a complete qualified3/4-policy final benchmark,20 trials each')
 evidence=summary.get('video_evidence',[]);seen=set();counts={p:0 for p in policies};rows=[];manifests=[]
 for case in evidence:
  policy,cid=case['policy'],case['case_id']
  if policy not in policies or (policy,cid) in seen or case.get('complete') is not True or case.get('errors'):
   raise ValueError('Duplicate, foreign or incomplete video case')
  seen.add((policy,cid));counts[policy]+=1;mp=Path(case['manifest_path']);mp=mp if mp.is_absolute() else summary_path.parent/mp
  if digest(mp)!=case['manifest_sha256']:raise ValueError('Manifest bytes differ from verified summary')
  m=json.loads(mp.read_text())
  if m.get('schema')!='pi05_g1_round02_video_manifest_v1' or m.get('policy')!=policy or m.get('case_id')!=cid or not hash64(m.get('raw_result_sha256')):
   raise ValueError('Manifest identity/result SHA missing')
  video_entries={**case.get('videos',{}),**case.get('actual_wall_clock_videos',{})}
  if not set(KEYS)<=set(video_entries):raise ValueError('Four mandatory physical-time videos absent')
  raw=Path(video_entries['head_raw']['path']);raw=raw if raw.is_absolute() else mp.parent/raw
  result_path=Path(m.get('source_case_dir',raw.parent))/'result.json'
  for key in (*KEYS,*WALL):
   if key not in video_entries:continue
   item=video_entries[key];vp=Path(item['path']);vp=vp if vp.is_absolute() else mp.parent/vp
   if not hash64(item.get('sha256')):raise ValueError('Verified video SHA missing')
   rows.append({'policy':policy,'case_id':cid,'view':'head' if key.startswith('head') else 'observer','video_kind':key,
    'playback_kind':'measured_wall_clock' if key in WALL else 'simulation_time',
    'video_path':str(vp),'video_sha256':item['sha256'],'fps':item['fps'],'frames':item['frames'],
    'raw_result_path':str(result_path),'raw_result_sha256':m['raw_result_sha256'],
    'video_manifest_path':str(mp),'video_manifest_sha256':case['manifest_sha256'],
    'benchmark_summary_path':str(summary_path),'benchmark_summary_sha256':h})
  manifests.append({'policy':policy,'case_id':cid,'manifest_path':str(mp),'manifest_sha256':case['manifest_sha256'],'raw_result_sha256':m['raw_result_sha256']})
 if any(v!=20 for v in counts.values()) or len(seen)!=20*len(policies):raise ValueError('Video index must cover every20 trial for every policy')
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();writer.writerows(sorted(rows,key=lambda x:(x['policy'],x['case_id'],x['video_kind'])))
 receipt={'schema':'pi05_g1_round02_full_video_csv_index_v1','summary_path':str(summary_path),'summary_sha256':h,'csv_path':str(output.resolve()),'csv_sha256':digest(output),'indexed_cases':len(seen),'indexed_video_rows':len(rows),'policy_case_counts':counts,'manifests':manifests,'scope':'No outcome-based selection. Includes every verified raw/annotated view and every available representative wall-clock view. Raw result/video bytes were not reopened; their identities inherit the completed qualified summary and hash-verified bound manifest.'}
 sidecar.write_text(json.dumps(receipt,indent=2)+'\n');return receipt
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=make_index(a.summary,a.output);print(json.dumps({k:r[k] for k in ['indexed_cases','indexed_video_rows','csv_path','csv_sha256']}))
