"""JSONL adapter to the pinned full N1.7 five-graph static-apple policy.

No simulator imports or truth input. RGB letterboxing and normalization remain
inside the original preprocess_video graph. GPU execution only on explicit use.
"""
import argparse,base64,contextlib,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from experiments.pi05_g1.contract import GROUPS
MODEL=Path('/home/ethan/models/unitree_g1/static_apple/7f78bebf1a90131e7304beacfcd47eb27bad16ab')
SCRIPTS=Path('/home/ethan/Projects/TempWorktree/Sai_Lab/unitree_g1/Bevy_Sim2Sim/crates/dev_tools/python/scripts')
PROMPT='move the apple to the plate'

def observation(req):
    if req['prompt']!=PROMPT:raise ValueError('GR00T export has fixed prompt: '+PROMPT)
    if req['rgb_shape']!=[480,640,3]:raise ValueError('Original head RGB must remain480x640x3')
    rgb=base64.b64decode(req['rgb_uint8_b64'],validate=True)
    if len(rgb)!=480*640*3:raise ValueError('RGB bytes mismatch')
    q=np.asarray(req['measured_joint_positions43'],np.float32);names=req['measured_joint_names43']
    if q.shape!=(43,) or len(names)!=43 or len(set(names))!=43 or not np.isfinite(q).all():raise ValueError('43 named measured joints required')
    out={'ego_view':np.frombuffer(rgb,np.uint8).reshape(1,480,640,3).astype(np.float32)}
    for group,groupnames in {**GROUPS,'waist':('waist_yaw_joint','waist_roll_joint','waist_pitch_joint')}.items():out[group]=q[[names.index(n) for n in groupnames]].reshape(1,-1)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--model-root',type=Path,default=MODEL);p.add_argument('--source-scripts',type=Path,default=SCRIPTS);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text())
    for item in manifest['files']:
        path=Path(item['path'])
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=item['sha256']:raise ValueError('Pinned source/model bytes changed: '+str(path))
    sys.path.insert(0,str(a.source_scripts));started=time.monotonic()
    with contextlib.redirect_stdout(sys.stderr):
        from unitree_g1_static_onnx import StaticAppleOnnx
        from unitree_g1_static_server import require_full_policy
        policy=StaticAppleOnnx(a.model_root,a.device,small_only=False);require_full_policy(policy)
    load=time.monotonic()-started
    print(json.dumps({'event':'gr00t_loaded','load_seconds':load,'provider':policy.provider}),file=sys.stderr,flush=True)
    for line in sys.stdin:
        req={}
        try:
            if len(line)>4_000_000:raise ValueError('Request too large')
            req=json.loads(line);obs=observation(req);start=time.monotonic()
            with contextlib.redirect_stdout(sys.stderr):outputs,stages=policy.infer(obs,seed=int(req['seed']))
            elapsed=time.monotonic()-start
            actions=np.concatenate([outputs[k][0] for k in ['left_arm','right_arm','left_hand','right_hand','base_height_command','navigate_command']],axis=1)
            if actions.shape!=(40,32) or not np.isfinite(actions).all():raise ValueError('Fixed GR00T40x32 output violated')
            reply={'episode_id':req['episode_id'],'control_tick':req['control_tick'],'actions':actions.tolist(),'inference_seconds':elapsed,'load_seconds':load,'stage_seconds':stages,'action_horizon':40,'decoded_waist':outputs['waist'][0].tolist(),'provider':policy.provider}
        except Exception as e:reply={'episode_id':req.get('episode_id'),'control_tick':req.get('control_tick'),'error':repr(e)}
        print(json.dumps(reply,allow_nan=False),flush=True)
if __name__=='__main__':main()
