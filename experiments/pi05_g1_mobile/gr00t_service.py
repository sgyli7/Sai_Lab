"""JSONL bridge to the original pinned GR00T N1.6 mobile-box policy.

RGB and named joint feedback only. It retains all native waist predictions as
inactive decoder evidence; the frozen Homie lower controller owns those joints.
"""
from __future__ import annotations
import argparse,base64,contextlib,json,sys,time,os
from pathlib import Path
import numpy as np
from .contract import PROMPT,observation_groups,native_outputs_to_actions
SCRIPTS=Path('/home/ethan/Projects/TempWorktree/Sai_Lab/unitree_g1/Bevy_Sim2Sim/crates/dev_tools/python/scripts')
MODEL=Path('/home/ethan/models/unitree_g1/mobile_box/dfe74af855007f26093f362cd2d7a2f404b64b93')
SOURCE=Path('/home/ethan/Projects/Sai_Lab/upstream/unitree_g1/isaac_gr00t_n16')
ENV=Path('/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13')
RECEIPT=Path('/home/ethan/Projects/Sai_Lab/.scratch/unitree_g1/policy/mobile_box_files.json')


def observation(req):
    allowed={'schema','episode_id','control_tick','rgb_shape','rgb_uint8_b64',
             'state','prompt','seed','measured_joint_positions43','measured_joint_names43'}
    if set(req)!=allowed or req['schema']!='pi05_g1_observation_v1':
        raise ValueError('Exact RGB/measured-joint request required; extra truth keys rejected')
    if req['prompt']!=PROMPT or req['rgb_shape']!=[480,640,3]:
        raise ValueError('Pinned task prompt and original640x480 head RGB required')
    rgb=base64.b64decode(req['rgb_uint8_b64'],validate=True)
    if len(rgb)!=480*640*3:raise ValueError('RGB byte count differs')
    state=observation_groups(req['measured_joint_positions43'],req['measured_joint_names43'])
    return {'video':{'ego_view':np.frombuffer(rgb,np.uint8).reshape(1,1,480,640,3)},
            'state':state,'language':{'annotation.human.task_description':[[PROMPT]]}}


def load_memory_diagnostic(stage, error=None):
    """Best-effort stderr only; never initializes CUDA for a diagnostic."""
    row={'event':'native_service_load_memory','stage':stage,'pid':os.getpid(),
         'monotonic_ns':time.monotonic_ns(),'wall_time_seconds':time.time(),
         'host_bytes':{},'cuda':{'initialized':False,'query_skipped':'not_yet_initialized'}}
    if error is not None:row['original_error']=repr(error)
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key,value=line.split(':',1)
            if key in ('MemFree','MemAvailable','Cached','Dirty'):
                row['host_bytes'][key]=int(value.split()[0])*1024
    except BaseException as diagnostic_error:row['host_query_error']=repr(diagnostic_error)
    try:
        # Loading policy imports torch anyway. Import alone does not create a
        # CUDA context, and is_initialized gates every CUDA memory query.
        with contextlib.redirect_stdout(sys.stderr):
            import torch
        initialized=torch.cuda.is_initialized()
        row['cuda']['initialized']=initialized
        if initialized:
            row['cuda'].pop('query_skipped',None)
            for name,query in [('mem_get_info',torch.cuda.mem_get_info),
                               ('allocated_bytes',torch.cuda.memory_allocated),
                               ('reserved_bytes',torch.cuda.memory_reserved),
                               ('max_allocated_bytes',torch.cuda.max_memory_allocated),
                               ('max_reserved_bytes',torch.cuda.max_memory_reserved)]:
                try:row['cuda'][name]=query()
                except BaseException as diagnostic_error:row['cuda'][name+'_error']=repr(diagnostic_error)
    except BaseException as diagnostic_error:row['cuda']['query_error']=repr(diagnostic_error)
    try:print(json.dumps(row,allow_nan=False),file=sys.stderr,flush=True)
    except BaseException:pass
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-scripts',type=Path,default=SCRIPTS)
    p.add_argument('--model-root',type=Path,default=MODEL)
    p.add_argument('--gr00t-source',type=Path,default=SOURCE)
    p.add_argument('--runtime-env',type=Path,default=ENV)
    p.add_argument('--receipt',type=Path,default=RECEIPT)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--decoded-evidence',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.source_scripts));started=time.monotonic()
    # Container UID differs from the owner of the read-only source mount. Trust
    # only this explicit checkout for this process; source revision/clean checks stay.
    git_count=int(os.environ.get('GIT_CONFIG_COUNT','0'))
    os.environ['GIT_CONFIG_COUNT']=str(git_count+1)
    os.environ[f'GIT_CONFIG_KEY_{git_count}']='safe.directory'
    os.environ[f'GIT_CONFIG_VALUE_{git_count}']=str(a.gr00t_source.resolve())
    load_memory_diagnostic('before_load')
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from unitree_g1_mobile_server import MobileBoxPolicy
            policy=MobileBoxPolicy(a.gr00t_source,a.model_root,a.receipt,a.seed,a.runtime_env)
    except BaseException as original_error:
        load_memory_diagnostic('load_failure',original_error)
        raise
    load_memory_diagnostic('after_load')
    load=time.monotonic()-started
    a.decoded_evidence.parent.mkdir(parents=True,exist_ok=True)
    with a.decoded_evidence.open('x',buffering=1) as evidence:
        evidence.write(json.dumps({'event':'model_loaded','load_seconds':load,'identity':policy.metadata},allow_nan=False)+'\n')
        try:
            for line in sys.stdin:
                req={}
                try:
                    if len(line)>4_000_000:raise ValueError('Request exceeds size bound')
                    req=json.loads(line);obs=observation(req);start=time.monotonic()
                    with contextlib.redirect_stdout(sys.stderr):outputs=policy.infer(obs)
                    elapsed=time.monotonic()-start
                    actions,waist=native_outputs_to_actions(outputs)
                    evidence.write(json.dumps({'event':'native_decoded','episode_id':req['episode_id'],
                        'control_tick':req['control_tick'],'actions32':actions.tolist(),
                        'decoded_waist3_inactive':waist.tolist(),'inference_seconds':elapsed},allow_nan=False)+'\n')
                    reply={'episode_id':req['episode_id'],'control_tick':req['control_tick'],
                           'actions':actions.tolist(),'inference_seconds':elapsed,'load_seconds':load,
                           'action_horizon':50,'native_decoder_width':35,'effective_action_width':32}
                except Exception as error:
                    reply={'episode_id':req.get('episode_id'),'control_tick':req.get('control_tick'),'error':repr(error)}
                print(json.dumps(reply,allow_nan=False),flush=True)
        finally:policy.close()
if __name__=='__main__':main()
