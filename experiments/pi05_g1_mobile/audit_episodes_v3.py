"""Independent v3 entrypoint for native schema-v2 CPU audit; bounded MP4 container origin, unchanged physical checks.

Task failure can have valid evidence. Incomplete producer status remains nonformal
including --allow-nonformal diagnostics, even when available trace checks pass.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from . import acceptance,contract
from .benchmark import sha,read,checked_json,decoded_evidence,INITIAL_KEYS


def array(value,shape,name):
    a=np.asarray(value,dtype=np.float32)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Missing/nonfinite/wrong shape '+name)
    return a


def equal(a,b,name):
    if not np.array_equal(a,b):raise ValueError('Array mismatch '+name)


def compare(actual,expected,path='acceptance'):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('Field set mismatch '+path)
        for k in expected:compare(actual[k],expected[k],path+'.'+k)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError('List shape mismatch '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,path+'.'+str(i))
    elif type(expected) is float:
        if type(actual) not in (int,float) or not np.isfinite(actual) or abs(actual-expected)>1e-6:raise ValueError('Numeric mismatch '+path)
    elif type(actual)!=type(expected) or actual!=expected:raise ValueError('Value mismatch '+path)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def video_probe(path,expected_frames):
    import av
    count=0;first=last=None;previous=None;origin=None;relative=[]
    with av.open(str(path)) as container:
        stream=container.streams.video[0];fps=float(stream.average_rate)
        stream.codec_context.thread_count=2
        if not np.isfinite(fps) or abs(fps-50)>1e-6:raise ValueError('Video fps is not50: '+str(path))
        for frame in container.decode(stream):
            if frame.pts is None:raise ValueError('Video PTS missing')
            timestamp=float(frame.pts*stream.time_base)
            if not np.isfinite(timestamp):raise ValueError('Video PTS must be finite')
            if count==0:
                origin=timestamp
                if not 0<=origin<=.1:raise ValueError('Video container first PTS outside admitted0..0.1s')
                first=hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).hexdigest()
            physical_time=timestamp-origin
            if abs(physical_time-count/50)>1e-6 or (previous is not None and timestamp<=previous):
                raise ValueError('Video relative PTS discontinuity/reorder/speed mismatch')
            relative.append(physical_time);last_frame=frame;previous=timestamp;count+=1
        if count:last=hashlib.sha256(last_frame.to_ndarray(format='rgb24').tobytes()).hexdigest()
    if count!=expected_frames:raise ValueError('Video frame count differs from controls+1: '+str(path))
    return {'frames':count,'fps':fps,'first_pts_seconds':origin,'last_pts_seconds':previous,
            'relative_pts_seconds':relative,'max_relative_pts_error_seconds':max((abs(t-i/50) for i,t in enumerate(relative)),default=0.),
            'origin_rule':'Bounded MP4 container packaging origin0..0.1s recorded, then PTS-firstPTS; simulation timestamps and all physical task predicates unchanged.',
            'decoded_first_rgb_sha256':first,'decoded_last_rgb_sha256':last,'lossy_rgb_not_compared_to_raw_png':True}


def check_video_pair(videos):
    if len(videos)!=2:raise ValueError('Both head and observer streams required')
    head,observer=videos.values()
    if head['frames']!=observer['frames'] or head['fps']!=observer['fps'] or not np.allclose(head['relative_pts_seconds'],observer['relative_pts_seconds'],rtol=0,atol=1e-6):
        raise ValueError('Head/observer frame count or relative timeline differs')
    return {'passed':True,'compared_all_relative_frame_times':True,'tolerance_seconds':1e-6}


def audit_case(value,*,allow_nonformal=False,decoded_checker=None):
    path=Path(value).resolve()
    supplied_config=None
    if path.is_file():
        candidate=read(path)
        if 'mode' in candidate and 'output_dir' in candidate:supplied_config=path;path=Path(candidate['output_dir']).resolve()
    if path.is_dir():result_path=path/'result.json'
    elif path.name=='config.json':result_path=path.with_name('result.json')
    else:result_path=path
    folder=result_path.parent;inputs={};checks={};anomalies=[];recomputed=None;config=None;formal=False
    def bind(path):
        path=Path(path).resolve();inputs[str(path)]=sha(path);return path
    try:
        result=read(bind(result_path));config=read(bind(folder/'config.json'));trace=bind(folder/'trajectory.jsonl')
        if supplied_config is not None:
            from .simulator import validate_config
            if validate_config(read(bind(supplied_config)))!=config:raise ValueError('Supplied frozen config differs from raw executed config')
        if result.get('case_id')!=config['case']['case_id'] or result.get('policy')!=config.get('policy_id',config['mode']):raise ValueError('Result/config policy-case identity mismatch')
        if result.get('schema')!='pi05_g1_mobile_case_result_v1':raise ValueError('Unsupported result schema')
        formal=result.get('status')=='completed' and result.get('source_interface_passed') is True
        if not formal and not allow_nonformal:raise ValueError('Result status/interface incomplete; diagnostic option does not make a formal pass')
        if result.get('trace_sha256')!=inputs[str(trace)]:raise ValueError('Trace SHA missing/mismatch')
        rows=[checked_json(json.loads(line)) for line in trace.read_text().splitlines() if line.strip()]
        if not rows:raise ValueError('No physical observations')
        ticks=len(rows)-1
        for i,row in enumerate(rows):
            if row['control_tick']!=i or row['physics_steps_since_reset']!=i*4 or abs(row['sim_time_seconds']-i/50)>1e-9:raise ValueError('Trace4PhysX/50Hz time discontinuity/reset')
            if i and row['monotonic_ns']<=rows[i-1]['monotonic_ns']:raise ValueError('Nonmonotonic actual wall time')
            array(row['joint_positions'],(43,),'measuredq');array(row['joint_velocities'],(43,),'measuredqdot')
            contract.encode_state(row['joint_positions'],row['source_joint_names'])
            if row['source_joint_names']!=result['source_joint_names']:raise ValueError('Result/trace source joint names differ')
        if result.get('completed_control_ticks')!=ticks or result.get('actual_sdk_integrations')!=4*ticks:raise ValueError('Actual control/physics counter mismatch')
        if result.get('control_hz')!=50 or result.get('physics_hz')!=200:raise ValueError('Producer timing schema mismatch')
        checks['physical_counter_timing']={'passed':True,'controls':ticks,'observations':len(rows),'physics_integrations':ticks*4}
        initial={k:rows[0][k] for k in INITIAL_KEYS};digest=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
        if result.get('initial_physical_state_schema')!='pi05_g1_mobile_initial_kinematic_v1' or result.get('initial_physical_state_sha256')!=digest:raise ValueError('Measured full initial state SHA mismatch')
        checks['initial_kinematic_sha']={'passed':True,'sha256':digest}
        frozen=config.get('frozen_files',{})
        if not frozen:raise ValueError('Recorded frozen source/dependency roster required')
        for name,expected in frozen.items():
            if inputs[str(bind(name))]!=expected:raise ValueError('Frozen execution input changed: '+name)
        runtime_path=bind(result.get('runtime_identity_path',folder/'runtime_identity.json'));runtime=read(runtime_path)
        if inputs[str(runtime_path)]!=result.get('runtime_identity_sha256'):raise ValueError('Runtime identity SHA mismatch')
        if runtime.get('schema')!='pi05_g1_mobile_runtime_identity_v1' or runtime.get('frozen_files')!=frozen:raise ValueError('Runtime frozen source identity differs')
        from .simulator import SOURCE_BUILD
        if runtime.get('sdk_build')!=SOURCE_BUILD:raise ValueError('Actual runtime SDK differs')
        for key in ['source_joint_names','body_names','contact_view','wbc_config','offline_unused_assets','video_encoding','frozen_interface_sha256']:
            if runtime.get(key)!=result.get(key):raise ValueError('Runtime identity result mismatch: '+key)
        interface=read(bind(folder/'interface.json'))
        if interface!=runtime.get('interface') or hashlib.sha256(json.dumps(interface,sort_keys=True).encode()).hexdigest()!=result.get('frozen_interface_sha256'):raise ValueError('Actual interface content SHA mismatch')
        checks['actual_runtime_identity']={'passed':True,'runtime_identity_sha256':inputs[str(runtime_path)],'sdk_build':runtime['sdk_build'],'frozen_dependency_count':len(frozen),'full_roster_verified':True}
        mode=config['mode'];bounds=result.get('actual_action_bounds');limits=None
        if bounds:
            lo=array(bounds['lower'],(32,),'actual lower');hi=array(bounds['upper'],(32,),'actual upper')
            if np.any(lo>hi):raise ValueError('Actual bound intersection empty')
            limits=(lo,hi);limit_path=bind(config['action_limits_file'])
            if result.get('action_limits_sha256')!=inputs[str(limit_path)]:raise ValueError('Common action limits SHA mismatch')
            base=read(limit_path);base_lo=array(base['lower'],(32,),'common lower');base_hi=array(base['upper'],(32,),'common upper')
            if np.any(lo<base_lo) or np.any(hi>base_hi):raise ValueError('Actual bounds exceed common physical limits')
        elif mode=='policy':raise ValueError('Policy actual action bounds unavailable')
        raw_actions=[];clipped_count=0
        for tick in range(ticks):
            pre,post=rows[tick],rows[tick+1];names=pre['source_joint_names'];q=array(pre['joint_positions'],(43,),'preq')
            if post['source_joint_names']!=names:raise ValueError('Joint ordering changed')
            raw=array(post['issued_action32_raw'],(32,),'issued32');applied=np.clip(raw,*limits) if limits is not None else raw
            equal(applied,array(post['applied_action32'],(32,),'applied32'),'raw32 commonclip nextrow')
            source=contract.decode_to_source(applied,q,names)
            equal(source,array(post['applied_source_action50'],(50,),'source50'),'32-to50 named upper/height/nav nextrow')
            if mode=='diagnostic':equal(raw,np.r_[array(rows[0]['joint_positions'],(43,),'initialq')[[rows[0]['source_joint_names'].index(n) for n in contract.UPPER_NAMES]],.75,0.,0.,0.].astype(np.float32),'diagnostic fixed reset hold')
            raw_actions.append(raw);clipped_count+=int(np.count_nonzero(applied!=raw))
        if result.get('commanded_channels')!=ticks*32 or result.get('clipped_channels')!=clipped_count:raise ValueError('Clipping/commanded channel counters differ')
        checks['actions']={'passed':True,'executed_actions':ticks,'source_width':50,'effective_width':32,'clipped_channels':clipped_count,'next_row_applied_semantics':True}
        if mode=='replay':
            rp=bind(config['replay_actions_npy']);expected=config.get('replay_actions_sha256',config.get('replay_actions_npy_sha256'))
            if inputs[str(rp)]!=expected:raise ValueError('Replay input SHA mismatch')
            replay=np.load(rp,allow_pickle=False);equal(np.asarray(raw_actions),replay[:ticks].astype(np.float32),'replay exact recorded prefix')
        query_checks=[]
        if mode=='policy':
            if config['replan_interval_ticks']!=50 or config.get('native_action_horizon',50)!=50:raise ValueError('Audit requires frozen native50 execution cadence')
            ep,events=(decoded_evidence if decoded_checker is None else decoded_checker)(config,result);bind(ep);decoded={e['control_tick']:e for e in events[1:]}
            expected_ticks=list(range(0,ticks,50));calls=result['inference_calls']
            if [c['observation_control_tick'] for c in calls]!=expected_ticks:raise ValueError('Query cadence/prefix coverage not every50')
            captures=folder/'policy_observations'
            if sorted(p.name for p in captures.glob('actions_*.npy'))!=[f'actions_{tick:05}.npy' for tick in expected_ticks]:raise ValueError('Missing/extra query action chunks')
            if sorted(p.name for p in captures.glob('observation_*.json'))!=[f'observation_{tick:05}.json' for tick in expected_ticks] or sorted(p.name for p in captures.glob('head_*.png'))!=[f'head_{tick:05}.png' for tick in expected_ticks]:raise ValueError('Missing/extra query observation/PNG files')
            from PIL import Image
            for tick,call in zip(expected_ticks,calls):
                row=rows[tick];npyp=bind(captures/f'actions_{tick:05}.npy');chunk=array(np.load(npyp,allow_pickle=False),(50,32),'query chunk50x32')
                equal(chunk,array(decoded[tick]['actions32'],(50,32),'service decoded50x32'),'service-toquery chunk')
                count=min(50,ticks-tick);equal(np.asarray(raw_actions[tick:tick+count]),chunk[:count],'query executed prefix')
                observation=read(bind(captures/f'observation_{tick:05}.json'))
                if set(observation)!={'control_tick','state','measured_joint_positions43','measured_joint_names43','prompt','rgb_sha256'}:raise ValueError('Unexpected query observation fields; truth inputs prohibited')
                if observation['control_tick']!=tick or observation['prompt']!=contract.PROMPT or config['prompt']!=contract.PROMPT:raise ValueError('Query time/prompt identity differs')
                if observation['measured_joint_names43']!=row['source_joint_names']:raise ValueError('Query names not measured source order')
                equal(array(observation['measured_joint_positions43'],(43,),'query measuredq'),array(row['joint_positions'],(43,),'rowq'),'query measuredq')
                equal(array(observation['state'],(32,),'querystate32'),contract.encode_state(row['joint_positions'],row['source_joint_names']),'query no-truth state32')
                pp=bind(captures/f'head_{tick:05}.png')
                with Image.open(pp) as image:
                    if image.mode!='RGB' or image.size!=(640,480):raise ValueError('Query lossless PNG shape/type differs')
                    rgb=np.asarray(image,np.uint8);rgbsha=hashlib.sha256(rgb.tobytes()).hexdigest()
                if rgbsha!=row['head_rgb_sha256'] or observation['rgb_sha256']!=rgbsha:raise ValueError('Raw query RGB byteSHA mismatch')
                start,end=call['request_started_monotonic_ns'],call['reply_received_monotonic_ns']
                if not row['monotonic_ns']-1000000<=start<=end<=rows[tick+1]['monotonic_ns']+1000000:raise ValueError('RPC monotonic/observed frame time mismatch')
                query_checks.append({'tick':tick,'executed_prefix':count,'rgb_sha256':rgbsha,'npy_shape':[50,32]})
        checks['policy_feedback']={'passed':True,'mode':mode,'queries':query_checks,'raw_rgb_sha_scope':'Lossless query PNG only; lossy MP4 not compared to raw pixel SHA'}
        videos={}
        for key,name in [('head_video_path','head_50fps.mp4'),('observer_video_path','observer_50fps.mp4')]:
            vp=bind(result.get(key,folder/name));videos[key]={'path':str(vp),'sha256':inputs[str(vp)],**video_probe(vp,ticks+1)}
        checks['videos']={'passed':True,'streams':videos,'pair_consistency':check_video_pair(videos)}
        if config.get('evaluate_acceptance'):
            context={**config,'audit_body_names':result['body_names'],'audit_contact_metadata':result['contact_view']}
            # Keep nested SDK filter_paths unmodified; acceptance validates its real shape.
            for key in ['geometry_definition','robot_geometry']:bind(config[key])
            recomputed=acceptance.evaluate_episode(rows,context)
            reported=result.get('acceptance')
            if reported is None:
                if formal:raise ValueError('Completed task lacks recorded acceptance')
                checks['acceptance']={'passed':False,'diagnostic_recomputation_only':True,'recorded_acceptance_missing':True}
            else:
                compare(reported,recomputed)
                if result.get('success') is not recomputed['success']:raise ValueError('Result success disagrees with independently recomputed acceptance')
                checks['acceptance']={'passed':True,'success':recomputed['success'],'task_wall_seconds_tolerance':1e-6,'compared_all_return_fields':True}
        else:
            checks['acceptance']={'passed':mode in ['diagnostic','replay'],'task_acceptance_requested':False,'task_success_claimed':False}
        if not formal:anomalies.append('Producer status/interface incomplete; available checks are diagnostics, never a formal episode pass')
        # Detect raw files changing during independent CPU audit.
        for path,digest in inputs.items():
            if sha(path)!=digest:raise ValueError('Evidence changed during audit: '+path)
    except Exception as error:anomalies.append(type(error).__name__+': '+str(error))
    import inspect
    from . import benchmark
    for module_path in [acceptance.__file__,contract.__file__,benchmark.__file__,inspect.getsourcefile(contract.decode_to_source)]:inputs[str(Path(module_path).resolve())]=sha(module_path)
    report={'schema':'pi05_g1_mobile_episode_cpu_audit_v2','result_path':str(result_path),'case_id':config.get('case',{}).get('case_id') if config else None,'seed':config.get('case',{}).get('seed') if config else None,'passed':not anomalies and all(c.get('passed') is True for c in checks.values()),'anomalies':anomalies,'checks':checks,'recomputed_acceptance':{k:v for k,v in recomputed.items() if k!='samples'} if recomputed else None,'inputs_sha256':inputs,'tool_sha256':sha(__file__),'autonomous_policy_trial':bool(formal and config and config.get('mode')=='policy'),'task_failure_is_not_evidence_failure':True,'no_raw_files_modified':True}
    report['audit_sha256']=hashlib.sha256(json.dumps(report,sort_keys=True,allow_nan=False).encode()).hexdigest();return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episodes',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-nonformal',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous audit')
    results=[audit_case(value,allow_nonformal=a.allow_nonformal) for value in a.episodes]
    report={'schema':'pi05_g1_mobile_episode_audits_v2','passed':all(r['passed'] for r in results),'episodes':results,'tool_sha256':sha(__file__)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':report['passed'],'episodes':len(results)}))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
