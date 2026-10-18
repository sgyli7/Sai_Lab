"""CPU extract only actions proven by old post-step SHA; never extend terminal tail."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .contract import native_outputs_to_actions,GROUPS,WAIST_NAMES,STATE_NAMES,encode_state

ROOT=Path(__file__).resolve().parents[2]
ROUND=ROOT/'.scratch/pi05_g1/round03_mobile'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):return json.loads(Path(p).read_text())
def frame_sha(frame):return hashlib.sha256(json.dumps(frame,sort_keys=True,allow_nan=False).encode()).hexdigest()

def main():
    inventory_path=ROUND/'simulator_inventory.json';inventory=read(inventory_path)
    history=next(r for r in inventory['historical_records'] if 't2_source_release_mobile_20261002_0112' in r.get('path',''))
    source=Path(history['path']);folder=source.parent;record=read(source);trace_path=folder/'source.jsonl';audit_path=Path(history['audit_path']);audit=read(audit_path)
    if sha(trace_path)!=audit['trace_sha256']:raise ValueError('Historical trace differs from recorded independent audit')
    if record['schema']!='g1_t2_original_source_task_v1' or record['model_revision']!='dfe74af855007f26093f362cd2d7a2f404b64b93':raise ValueError('Historical source identity differs')
    if (record['policy_calls'],record['action_horizon'],record['source_control_hz'],record['source_physics_hz'],record['action_period_ns'])!=(19,50,50,200,20000000):raise ValueError('Historical cadence/count changed')
    traces=[json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    if [r['control_tick'] for r in traces]!=list(range(942)):raise ValueError('Require measured states0..941 only')
    if any(r['physics_steps_since_reset']!=r['control_tick']*4 for r in traces):raise ValueError('Trace physics counter mismatch')
    inputs={str(p):sha(p) for p in [inventory_path,source,trace_path,audit_path,folder/'unitree_g1_t2_source_task.py',folder/'executed_driver.py']}
    actions=[];raw=[];waist=[];chunks=[];frame_receipts=[]
    names=record['joint_names'];expected_groups={**{k:list(v) for k,v in GROUPS.items()},'waist':list(WAIST_NAMES)}
    if record['policy_joint_groups']!=expected_groups:raise ValueError('Historical group order differs from mobile contract')
    replies=sorted((folder/'source.captures').glob('reply_*.json'))
    if len(replies)!=19:raise ValueError('Exactly19 recorded reply chunks required')
    for chunk_index,p in enumerate(replies):
        tick=chunk_index*50
        if p.name!=f'reply_{tick:04d}.json':raise ValueError('Reply prefix/cadence mismatch')
        reply=read(p);request_path=p.with_name(f'request_{tick:04d}.json');request=read(request_path)
        inputs[str(p)]=sha(p);inputs[str(request_path)]=sha(request_path)
        if reply['profile']!='mobile_box' or reply['model_revision']!=record['model_revision'] or reply['action_period_ns']!=20000000 or reply['sequence_id']!=chunk_index+1 or len(reply['frames'])!=50:raise ValueError('Invalid original native50 reply')
        obs=request['observation']
        if request['schema']!='mobile_observation_v1' or request['profile']!='mobile_box' or request['sequence_id']!=chunk_index+1 or obs['episode_id']!=record['episode_id'] or obs['frame_id']!=tick or obs['sim_time_ns']!=tick*20000000 or reply['observation']!=obs:raise ValueError('Request/reply observation identity mismatch')
        for group,joints in expected_groups.items():
            measured=[traces[tick]['joint_positions'][names.index(name)] for name in joints]
            if request['state_groups'][group]!=measured:raise ValueError('Query proprioception differs from measured trace at tick'+str(tick))
        frames=reply['frames'];groups={k:np.asarray([[f[k] for f in frames]],dtype=np.float32) for k in ['left_arm','right_arm','left_hand','right_hand','waist']}
        groups['base_height_command']=np.asarray([[[f['base_height_m']] for f in frames]],dtype=np.float32)
        groups['navigate_command']=np.asarray([[f['navigate_mps_rps'] for f in frames]],dtype=np.float32)
        projected,decoded_waist=native_outputs_to_actions(groups)
        raw35=np.concatenate([groups[k][0] for k in ['left_arm','right_arm','left_hand','right_hand','waist','base_height_command','navigate_command']],axis=1)
        used=min(50,941-tick)
        for i in range(used):
            frame=frames[i];digest=frame_sha(frame);post=traces[tick+i+1]
            if post['policy_sequence']!=chunk_index+1 or post['last_action_frame']!=i or post['last_action_frame_sha256']!=digest:raise ValueError('Post-step original action SHA mismatch at action'+str(tick+i))
            frame_receipts.append({'action_index':tick+i,'pre_state_tick':tick+i,'post_state_tick':tick+i+1,'sequence_id':chunk_index+1,'chunk_frame':i,'native_frame_sha256':digest})
        actions.append(projected[:used]);raw.append(raw35[:used]);waist.append(decoded_waist[:used]);chunks.append({'reply_path':str(p),'reply_sha256':inputs[str(p)],'request_sha256':inputs[str(request_path)],'prefix_control_tick':tick,'sequence_id':chunk_index+1,'native_frames':50,'used_frames':used,'unused_tail_frames':50-used,'request_state_exactly_matches_trace':True})
    arrays={'actions32_941.npy':np.concatenate(actions),'raw35_941.npy':np.concatenate(raw),'decoded_waist3_941.npy':np.concatenate(waist),'joint_positions43_942.npy':np.asarray([r['joint_positions'] for r in traces],np.float32),'state32_942.npy':np.stack([encode_state(r['joint_positions'],names) for r in traces]),'root_link_pose_w_942.npy':np.asarray([r['root_link_pose_w'] for r in traces],np.float32),'brown_box_pose_w_942.npy':np.asarray([r['acceptance_truth_only']['brown_box']['pose'] for r in traces],np.float32),'blue_bin_pose_w_942.npy':np.asarray([r['acceptance_truth_only']['blue_sorting_bin']['pose'] for r in traces],np.float32)}
    if arrays['actions32_941.npy'].shape!=(941,32) or arrays['raw35_941.npy'].shape!=(941,35):raise ValueError('Wrong extracted action count')
    if any(not np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite reference array')
    output=ROUND/'replay_reference'
    if output.exists():raise FileExistsError('Preserve existing replay reference')
    output.mkdir();outputs={}
    for name,array in arrays.items():
        with (output/name).open('xb') as f:np.save(f,array,allow_pickle=False)
        outputs[name]={'path':str(output/name),'sha256':sha(output/name),'shape':list(array.shape),'dtype':str(array.dtype)}
    receipt={'schema':'pi05_g1_mobile_recorded_replay_reference_v1','status':'complete','source_record_schema':record['schema'],'model_revision':record['model_revision'],'seed':record['seed'],'episode_id':record['episode_id'],'native_action_horizon':50,'source_control_hz':50,'source_physics_hz':200,'chunks':chunks,'recorded_native_frames':950,'verified_actions':941,'measured_state_ticks_inclusive':[0,941],'excluded_reason':'Action index941/control942 caused terminal autoreset with no post-step pre-reset snapshot; no942 action or synthetic/held tail included. Last chunk9 unusedframes are preserved only by original source reply input hash.','effective_action32_order':'left_arm7,right_arm7,left_hand7,right_hand7,base_height1,navigate_vx_vy_yaw3','raw35_order':'left_arm7,right_arm7,left_hand7,right_hand7,waist3,base_height1,navigate_vx_vy_yaw3','waist_decoder_not_executed_joint_authority':'HomieV2.enable_waist=False; sourceupper28 passed toWBC; raw decodedwaist retained as evidence','initial_state':traces[0],'initial_joint_names':names,'state32_names':list(STATE_NAMES),'source_initial_measured_vs_request_exact':True,'initial_state_hash_scope':'Historical record lacks root/joint velocities; no fullkinematic-state identity invented','frame_action_sha_verification':frame_receipts,'inputs_sha256':inputs,'outputs':outputs,'source_success_status':{'source_proximity_trigger':history['source_proximity_success'],'formal_released_stable_verified':False,'terminal_autoreset':True},'script_sha256':sha(Path(__file__))}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'complete','output':str(output),'verified_actions':941,'measured_states':942,'chunks':19}))

if __name__=='__main__':main()
