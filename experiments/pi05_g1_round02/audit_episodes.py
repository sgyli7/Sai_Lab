"""Independent CPU audit of retained round02 physics, feedback and action evidence.

Never imports simulator runtime, creates ONNX/CUDA sessions, writes raw cases,
or changes task predicates. Reports anomalies instead of trusting result flags.
"""
from __future__ import annotations
import argparse,hashlib,importlib,json,sys,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.pi05_g1.contract import encode_state,decode_to_source,UPPER_NAMES
from experiments.pi05_g1.placement import evaluate as placement_evaluate,acceptance_identity
APPLE='apple_01_objaverse_robolab'

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def audit_case(directory):
 p=Path(directory);issues=[];report={'case_directory':str(p.resolve()),'anomalies':issues,'checks':{},'input_sha256':{}}
 def check(ok,message):
  if not bool(ok):issues.append(message)
 def numeric(a,b,label,atol=1e-7):
  x,y=np.asarray(a,float),np.asarray(b,float)
  check(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all() and np.allclose(x,y,atol=atol,rtol=0),label)
 def section(name,fn):
  count=len(issues)
  try:report['checks'][name]=fn()
  except Exception as exc:issues.append(f'{name}: {type(exc).__name__}: {exc}');report['checks'][name]={'exception':repr(exc)}
  report.setdefault('section_passed',{})[name]=len(issues)==count
 try:
  r=json.loads((p/'result.json').read_text());cfg=json.loads((p/'config.json').read_text());rows=[json.loads(x) for x in (p/'trajectory.jsonl').read_text().splitlines()]
  report.update(case_id=r['case_id'],policy=r['policy'],recorded_success=r.get('success'))
  for name in ['result.json','config.json','trajectory.jsonl','legacy_speed_audit.json','placement_audit.json']:report['input_sha256'][name]=digest(p/name)
  check(r.get('status')=='completed','case not completed')
  names=r['source_joint_names'];meta=r['placement_contact_view'];n=len(rows);identity=acceptance_identity()
  for k in ['acceptance_version','acceptance_sha256']:check(r.get(k)==identity[k],f'Frozen {k} mismatch')
  def telemetry():
   for i,row in enumerate(rows):
    check(row['control_tick']==i and row['physics_steps_since_reset']==i*4,f'tick{i}: physics/sample coverage mismatch')
    check(row['source_joint_names']==names,f'tick{i}: joint order changed')
    for key,shape in [('joint_positions',(43,)),('root_link_pose_w',(7,))]:
     v=np.asarray(row[key]);check(v.shape==shape and np.isfinite(v).all(),f'tick{i}: invalid{key}')
    if r.get('initial_physical_state_schema')=='pi05_g1_initial_kinematic_state_v2':
     for key,shape in [('joint_velocities',(43,)),('root_com_velocity_w',(6,)),('root_link_velocity_w',(6,))]:
      v=np.asarray(row[key]);check(v.shape==shape and np.isfinite(v).all(),f'tick{i}: invalid{key}')
     check(row.get('velocity_telemetry_physics_counters_before_after_equal') is True,f'tick{i}: velocity read counter invariant absent')
    if i:check(row['monotonic_ns']>rows[i-1]['monotonic_ns'],f'tick{i}: nonmonotonic capture time')
   row=rows[0];v1={'q':row['joint_positions'],'root':row['root_link_pose_w'],'objects':row['acceptance_truth_only']};h1=canonical_hash(v1)
   if r.get('initial_physical_state_schema')=='pi05_g1_initial_kinematic_state_v2':
    initial={'schema':r['initial_physical_state_schema'],'joint_names':names,'q':row['joint_positions'],'qdot':row['joint_velocities'],'root_link_pose_w':row['root_link_pose_w'],'root_com_velocity_w':row['root_com_velocity_w'],'root_link_velocity_w':row['root_link_velocity_w'],'objects':row['acceptance_truth_only']};h=canonical_hash(initial);check(h1==r['initial_pose_state_sha256_v1'],'v1 compatibility hash mismatch')
   else:h=h1
   check(h==r['initial_physical_state_sha256'],'independent initial state hash mismatch')
   return {'initial_schema':r.get('initial_physical_state_schema','pose_only_v1'),'recomputed_initial_hash':h,'recomputed_v1_hash':h1,'samples':n,'physical_controls':rows[-1]['control_tick']}
  section('telemetry_and_initial_hash',telemetry)
  physical={}
  def physical_checks():
   hand=[i+1 for i,name in enumerate(meta['robot_body_names']) if 'hand' in name.lower() or 'finger' in name.lower()];check(bool(hand),'No hand identities');z0=rows[0]['acceptance_truth_only'][APPLE]['pose'][2];grasp=None;candidate=None;fell=dropped=False
   for row in rows:
    tick=row['control_tick'];apple=row['acceptance_truth_only'][APPLE];matrix=np.asarray(row['acceptance_contacts_only']['apple_filter_force_matrix_world_n']);touch=np.count_nonzero(np.linalg.norm(matrix[hand],axis=1)>1e-6)
    if grasp is None and apple['pose'][2]-z0>.03 and touch>=2:grasp=tick
    fell |= row['root_link_pose_w'][2]+.795<.35 or row['upright']<.5
    low=apple['pose'][2]<-.2;dropped |= low;check(low==bool(row['original_terms'].get('object_dropped',False)),f'tick{tick}: source drop versus actual height disagreement')
    if row['original_success_before_audit_suppression']['value'] and tick<=300 and candidate is None:candidate=tick
   for key,value in [('first_grasp_control_tick',grasp),('grasped',grasp is not None),('source_success_candidate_tick',candidate),('fell',bool(fell)),('dropped',bool(dropped))]:check(r.get(key)==value,f'{key} mismatch from raw physics')
   scripts=Path(cfg['source_harness_dir']);sys.path.insert(0,str(scripts));module=importlib.import_module('unitree_g1_t1_source_placement');check(digest(Path(module.__file__))==r['frozen_interface']['legacy_geometry_audit_sha256'],'Original geometry auditor source changed')
   definition=json.loads(Path(cfg['geometry_definition']).read_text());robot=json.loads(Path(cfg['robot_geometry']).read_text())
   check(digest(cfg['geometry_definition'])==r['frozen_interface']['geometry_sha256'],'Object geometry identity mismatch');check(digest(cfg['robot_geometry'])==r['frozen_interface']['robot_geometry_sha256'],'Robot geometry identity mismatch')
   receipt={'placement_contact_view':meta,'success_auto_reset_suppressed_for_audit_only':r['source_terminal_predicates_unchanged'],'original_drop_and_timeout_terms_unchanged':True,'source_physics_hz':200,'source_control_hz':50,'completed_control_ticks':rows[-1]['control_tick'],'measured_physics_steps_since_reset':rows[-1]['physics_steps_since_reset'],'policy_calls':len(r['inference_calls'])}
   legacy=module.evaluate(definition,robot,receipt,rows);stored=json.loads((p/'legacy_speed_audit.json').read_text());maxdiff=0.
   check(len(legacy['samples'])==len(stored['samples']),'legacy sample counts differ')
   for new,old in zip(legacy['samples'],stored['samples']):
    for key,value in new.items():
     if isinstance(value,bool):check(value==old[key],f'legacy tick{new["tick"]}:{key} mismatch')
     elif isinstance(value,(int,float)):maxdiff=max(maxdiff,abs(value-old[key]))
   check(maxdiff<1e-10,'Recomputed legacy numeric sample difference exceeds tolerance')
   v2=placement_evaluate(legacy,rows,first_grasp_tick=grasp,source_success_candidate_tick=candidate,fell=fell,dropped=dropped);oldv2=json.loads((p/'placement_audit.json').read_text());qualified=v2['stable_placement_passed'];success=bool(grasp is not None and candidate is not None and grasp<candidate and qualified and not fell and not dropped)
   check(v2['qualified_window']==oldv2['qualified_window'],'v2 qualified window mismatch');check(v2['qualified_window']==r['qualified_stability_window'],'result qualified window mismatch');check(qualified==r['stable_placement'],'stable placement flag mismatch');check(success==r['success'],'success mismatch');check((not success and not fell and not dropped)==r['timeout'],'timeout mismatch')
   physical.update(success=success,first_grasp_tick=grasp,source_candidate_tick=candidate,qualified_window=v2['qualified_window'],fell=bool(fell),dropped=bool(dropped),legacy_speed_passed=legacy['source_diagnostic_placement_passed'],legacy_numeric_max_difference=maxdiff,physical_candidate_rule='Frozen original callback flag, not replaced or tuned')
   return physical
  section('raw_physics_geometry_and_acceptance',physical_checks)
  def actions_feedback():
   replay=cfg['mode']=='replay';bounds=bool(r['replay_action_limits_applied']);cadence=r['replan_interval_ticks'];clips=commands=0;query_ticks=[];chunks={};pngs=0;last=None
   if bounds:
    lower=np.asarray(r['action_bounds']['lower'],np.float32);upper=np.asarray(r['action_bounds']['upper'],np.float32);actual=r['frozen_interface']['actual_common_action_bounds'];numeric(lower,actual['lower'],'interface lower bounds mismatch');numeric(upper,actual['upper'],'interface upper bounds mismatch');check(digest(cfg.get('action_limits_file',cfg.get('action_limits')))==r['action_limits_sha256'],'limit file identity mismatch')
   expert=None
   if replay:
    path=Path(cfg['expert_sequence']);check(digest(path)==cfg['expert_sha256'],'expert action sequence changed');expert=json.loads(path.read_text());report['input_sha256']['expert_sequence']=digest(path)
   for tick in range(n-1):
    current,nxt=rows[tick],rows[tick+1]
    if replay:
     f=expert['frames'][min(tick,len(expert['frames'])-1)];raw=np.asarray(f['left_arm']+f['right_arm']+f['left_hand']+f['right_hand']+[f['base_height_m']]+f['navigate_mps_rps'],np.float32)
    elif tick>=300:raw=last.copy()
    else:
     qtick=tick//cadence*cadence
     if qtick not in chunks:
      query_ticks.append(qtick);chunkfile=p/'policy_observations'/f'actions_{qtick:04}.npy';chunk=np.load(chunkfile,allow_pickle=False);check(chunk.shape==(cfg['native_action_horizon'],32) and np.isfinite(chunk).all(),f'query{qtick}: native horizon/action invalid');check(cadence<=len(chunk),f'query{qtick}: cadence exceeds native horizon');chunks[qtick]=chunk
      obsfile=p/'policy_observations'/f'observation_{qtick:04}.json';obs=json.loads(obsfile.read_text());check(obs['control_tick']==qtick and obs['episode_id']==r['case_id'],f'query{qtick}: observation identity mismatch');check(obs['prompt']==cfg['prompt'],f'query{qtick}: prompt mismatch');numeric(obs['state'],encode_state(current['joint_positions'],names),f'query{qtick}: state32 feedback mismatch',atol=0)
      imagefile=p/'policy_observations'/obs['rgb_path'];pixels=np.array(Image.open(imagefile));pngs+=1;check(pixels.shape==(480,640,3) and pixels.dtype==np.uint8,f'query{qtick}: head image contract mismatch');h=hashlib.sha256(pixels.tobytes()).hexdigest();check(h==obs['rgb_sha256']==current['head_rgb_sha256'],f'query{qtick}: input RGB does not match actual observed frame')
      for fpath in [chunkfile,obsfile,imagefile]:report['input_sha256'][str(fpath.relative_to(p))]=digest(fpath)
     raw=np.asarray(chunks[qtick][tick-qtick],np.float32)
    numeric(raw,nxt['issued_action32_raw'],f'tick{tick}: raw native-prefix/replay/tail action mismatch')
    actual=np.clip(raw,lower,upper) if bounds else raw
    if bounds:clips+=int(np.count_nonzero((raw<lower)|(raw>upper)));commands+=32
    numeric(actual,nxt['applied_action32'],f'tick{tick}: actual common-bound clip mismatch')
    source=decode_to_source(actual,current['joint_positions'],names);numeric(source,nxt['applied_source_action50'],f'tick{tick}:32→50 source action mapping mismatch')
    # AGILE owns15leg/waist slots; independent upper target identity still must hold.
    ids=[names.index(x) for x in UPPER_NAMES];numeric(actual[:28],np.asarray(nxt['processed_joint_targets'])[ids],f'tick{tick}: upper28 WBC target changed')
    last=actual.copy()
   if bounds:check(clips==r['clipped_channels'] and commands==r['commanded_channels'],'clip accounting mismatch')
   calls=[x['observation_control_tick'] for x in r['inference_calls']];check(calls==query_ticks,'query count/order inconsistent with native-prefix replanning')
   obsfiles=list((p/'policy_observations').glob('observation_*.json'));check(len(obsfiles)==len(query_ticks),'unexpected/missing query observation files')
   if cfg.get('save_teacher_frames'):
    for i,row in enumerate(rows):
     path=p/row['teacher_head_rgb_path'];pixels=np.array(Image.open(path));check(pixels.shape==(480,640,3) and pixels.dtype==np.uint8 and hashlib.sha256(pixels.tobytes()).hexdigest()==row['head_rgb_sha256'],f'tick{i}: teacher lossless head PNG mismatch');report['input_sha256'][str(path.relative_to(p))]=digest(path);pngs+=1
    check(len(list((p/'teacher_head_rgb').glob('*.png')))==n,'teacher RGB coverage incomplete/extraneous')
   return {'controls_checked':n-1,'query_ticks':query_ticks,'native_horizon':cfg.get('native_action_horizon'),'executed_prefix_max':cadence,'clipped_channels':clips,'commanded_channels':commands,'pngs_verified':pngs,'scope':'Saved model-query state32 and RGB bytes checked. Actual q43 JSONL wire bytes are not separately captured, so q43 transmission relies on frozen service/request source identity plus physical q43 trace.'}
  section('action_execution_and_observation_feedback',actions_feedback)
 except Exception as exc:issues.append('fatal: '+repr(exc));report['exception_traceback']=traceback.format_exc()
 report['passed']=not issues;report['audit_tool_sha256']=digest(__file__);report['audit_sha256']=canonical_hash(report);return report

def discover(inputs):
 found=set()
 for value in inputs:
  path=Path(value)
  if path.is_file() and path.name=='result.json':found.add(path.parent.resolve())
  elif (path/'result.json').is_file():found.add(path.resolve())
  elif path.is_dir():found.update(p.parent.resolve() for p in path.rglob('result.json'))
  else:raise ValueError('No case path: '+str(path))
 return sorted(found)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('cases',nargs='+');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();paths=discover(args.cases)
 if not paths:raise ValueError('No case results found')
 if any(args.output.resolve().is_relative_to(p) for p in paths):raise ValueError('Audit output must be outside raw case directories')
 report={'schema':'pi05_g1_round02_independent_episode_audit_v1','cpu_only':True,'cases':[audit_case(p) for p in paths],'scope':'Full geometry/contact/v2 recomputation; no learned-policy effectiveness inference. Hashes are generalized kinematic-state receipts, not complete PhysX solver snapshots.'};report['passed']=all(x['passed'] for x in report['cases']);report['audit_sha256']=canonical_hash(report);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':report['passed'],'cases':[{k:c.get(k) for k in ['case_id','policy','passed','anomalies']} for c in report['cases']],'output':str(args.output),'audit_sha256':report['audit_sha256']}))
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
