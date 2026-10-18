"""CPU evidence movie of one fixed demonstration under three limit conditions."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw
from .video_compare import Cursor
from .video_export import writer,emit,finish,digest
from experiments.pi05_g1.contract import UPPER_NAMES
ROOT=Path('/home/ethan/Projects/Sai_Lab/.scratch/pi05_g1/round02')

def main():
 paths=[ROOT/'replay_gates_c10_A/unbounded',ROOT/'replay_gates_c10_A/bounded',ROOT/'replay_gates_c10_B_observer_v2/bounded'];labels=['V1 / UNBOUNDED ORIGINAL DEMONSTRATION','V1 / A: QUANTILE LIMITS + HARDWARE','V2 / B: RAW-TRAIN EXTREMA + HARDWARE']
 results=[json.loads((p/'result.json').read_text()) for p in paths];traces=[[json.loads(x) for x in (p/'trajectory.jsonl').read_text().splitlines()] for p in paths];init=[]
 for r,rows in zip(results,traces):
  row=rows[0];state={'q':row['joint_positions'],'root':row['root_link_pose_w'],'objects':row['acceptance_truth_only']};h=hashlib.sha256(json.dumps(state,sort_keys=True,allow_nan=False).encode()).hexdigest();assert h==r['initial_physical_state_sha256'];init.append(h)
 assert len(set(init))==1
 assert len(set(r['case_id'] for r in results))==1 and results[0]['case']['seed']==42
 camera=[r['frozen_interface']['camera'] for r in results];assert camera[0]==camera[1]==camera[2]
 target=ROOT/'video_review';target.mkdir(exist_ok=True);dst=target/'limit_replay_comparison.mp4'
 if dst.exists():raise FileExistsError(dst)
 cursors=[Cursor(p/'head_50fps.mp4') for p in paths];count=max(len(t) for t in traces);out,stream=writer(dst,1920,580,50)
 try:
  for i in range(count):
   canvas=Image.new('RGB',(1920,580),'#111111');draw=ImageDraw.Draw(canvas)
   draw.text((10,5),'SAME EPISODE0 / SEED42 ORIGINAL DEMONSTRATION - NO LEARNED POLICY; 50Hz SIMULATION PLAYBACK',fill='white')
   for col,(r,rows,cursor,label) in enumerate(zip(results,traces,cursors,labels)):
    x=col*640;draw.text((x+8,25),label,fill='white');draw.text((x+8,44),f"{'SUCCESS' if r['success'] else 'FAILURE'} | tick={min(i,len(rows)-1)} sim={min(i,len(rows)-1)/50:.2f}s",fill='#8aff8a' if r['success'] else '#ff9999')
    if i>=len(rows):draw.text((x+8,63),'HOLD FINAL OBSERVED FRAME - NO NEW ACTION',fill='#ffdd77')
    canvas.paste(cursor.at(i/50),(x,88))
   emit(out,stream,canvas)
  finish(out,stream)
 finally:
  for c in cursors:c.input.close()
 summaries=[];reference=traces[0]
 for r,rows in zip(results[1:],traces[1:]):
  n=min(len(reference),len(rows));names=r['source_joint_names'];ids=[names.index(x) for x in UPPER_NAMES]
  q=np.array([x['joint_positions'] for x in rows[:n]])[:,ids];qref=np.array([x['joint_positions'] for x in reference[:n]])[:,ids];d=np.abs(q-qref)
  root=np.array([x['root_link_pose_w'] for x in rows[:n]]);refroot=np.array([x['root_link_pose_w'] for x in reference[:n]]);angle=2*np.arccos(np.clip(np.abs(np.sum(root[:,3:]*refroot[:,3:],axis=1)/np.linalg.norm(root[:,3:],axis=1)/np.linalg.norm(refroot[:,3:],axis=1)),0,1))
  raw=np.array([x['issued_action32_raw'] for x in rows[1:]]);applied=np.array([x['applied_action32'] for x in rows[1:]]);delta=np.abs(raw-applied);channels=[]
  for j in range(32):
   if np.any(delta[:,j]>0):channels.append({'channel':j,'name':UPPER_NAMES[j] if j<28 else ['height','vx','vy','yawrate'][j-28],'frames_clipped':int(np.count_nonzero(delta[:,j]>0)),'maximum_absolute_clip_in_physical_units':float(delta[:,j].max())})
  summaries.append({'case_condition':labels[results.index(r)],'compared_to':'unbounded original demo','same_tick_samples':n,'maximum_arm14_measured_q_deviation_rad':float(d[:,:14].max()),'maximum_upper28_measured_q_deviation_rad':float(d.max()),'maximum_joint_deviation_by_name_rad':dict(zip(UPPER_NAMES,d.max(axis=0).tolist())),'maximum_pelvis_origin_pair_distance_m':float(np.linalg.norm(root[:,:3]-refroot[:,:3],axis=1).max()),'maximum_pelvis_orientation_deviation_deg':float(np.degrees(angle).max()),'pelvis_origin_trace_xyz_m':root[:,:3].tolist(),'clipping':channels})
 manifest={'schema':'pi05_g1_round02_limit_replay_comparison_v1','case_id':results[0]['case_id'],'sampling':'Same original episode0 seed42; all three specified limit conditions, no outcome-based case selection','model':'original demonstration, no learned policy','movie':{'path':str(dst),'fps':50,'frames':count,'duration_seconds':count/50,'sha256':digest(dst),'playback':'simulation-time, not wall-clock; short case final frame held and labelled'},'initial_state_hashes_recomputed':init,'initial_state_equal':True,'head_camera_configuration_equal':True,'observer_note':'v1 whole-body camera rejected for shelf occlusion; v2 accepted for new final suite. This comparison uses only the unchanged original head camera in all3 columns. Observer version and bounds both differ between v1A and v2B, so compare unchanged-head/physics evidence; do not attribute result to one joint.','inputs':[{'condition':label,'path':str(p),'success':r['success'],'result_sha256':digest(p/'result.json'),'trace_sha256':digest(p/'trajectory.jsonl'),'head_video_sha256':digest(p/'head_50fps.mp4'),'first_grasp_control_tick':r['first_grasp_control_tick'],'source_success_candidate_tick':r['source_success_candidate_tick'],'frozen_interface_sha256':r['frozen_interface_sha256']} for p,r,label in zip(paths,results,labels)],'bounded_trajectory_summaries':summaries,'inference_limit':'Only the combined quantile limit intervention is implicated by unbounded-v1 versus boundedA-v1. No isolated elbow/wrist causal test was performed. This does not establish the cause of all learned-policy failures.'}
 (target/'limit_replay_comparison_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n');print(json.dumps({'movie':manifest['movie'],'same_initial_state':True,'bounded_summaries':[{'condition':s['case_condition'],'max_arm_q_deviation':s['maximum_arm14_measured_q_deviation_rad'],'pelvis_max_distance':s['maximum_pelvis_origin_pair_distance_m'],'clipping':s['clipping']} for s in summaries]}))
if __name__=='__main__':main()
