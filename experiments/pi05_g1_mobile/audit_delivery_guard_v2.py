"""Shared native-v2 audit receipt guard for report and permanent retention."""
import hashlib,json
from pathlib import Path
import numpy as np
from .audit_episodes_v2 import check_video_pair

REQUIRED_CHECKS={'physical_counter_timing','initial_kinematic_sha','actual_runtime_identity','actions','policy_feedback','videos','acceptance'}


def validate(audit):
    if audit.get('schema')!='pi05_g1_mobile_episode_cpu_audit_v2' or audit.get('passed') is not True or audit.get('anomalies') or audit.get('autonomous_policy_trial') is not True:raise ValueError('Native v2 independent episode pass required')
    unsigned={k:v for k,v in audit.items() if k!='audit_sha256'}
    if hashlib.sha256(json.dumps(unsigned,sort_keys=True,allow_nan=False).encode()).hexdigest()!=audit.get('audit_sha256'):raise ValueError('V2 audit self SHA mismatch')
    checks=audit.get('checks',{})
    if not REQUIRED_CHECKS<=set(checks) or not all(isinstance(c,dict) and c.get('passed') is True for c in checks.values()):raise ValueError('Missing or failed v2 independent check')
    runtime=checks['actual_runtime_identity']
    if runtime.get('full_roster_verified') is not True or type(runtime.get('frozen_dependency_count')) is not int or runtime['frozen_dependency_count']<=0:raise ValueError('V2 frozen runtime/dependency verification missing')
    result=Path(audit['result_path']).resolve();bound=audit.get('inputs_sha256',{})
    for name in ['result.json','config.json','trajectory.jsonl','runtime_identity.json']:
        if str(result if name=='result.json' else result.parent/name) not in bound:raise ValueError('V2 actual evidence input missing: '+name)
    if any(not Path(path).is_absolute() for path in bound):raise ValueError('V2 evidence bindings must be absolute')
    videos=checks['videos'];streams=videos.get('streams',{})
    if set(streams)!={'head_video_path','observer_video_path'} or videos.get('pair_consistency',{}).get('passed') is not True:raise ValueError('V2 two-stream independent consistency proof missing')
    for stream in streams.values():
        n=stream.get('frames');times=np.asarray(stream.get('relative_pts_seconds',[]),float);origin=stream.get('first_pts_seconds');fps=stream.get('fps')
        if type(n) is not int or n<=0 or type(origin) not in (int,float) or not np.isfinite(origin) or not 0<=origin<=.1 or type(fps) not in (int,float) or not np.isfinite(fps) or fps!=50:raise ValueError('V2 actual video fps/origin invalid')
        if times.shape!=(n,) or not np.isfinite(times).all() or not np.allclose(times,np.arange(n)/50,rtol=0,atol=1e-6) or (n>1 and not np.all(np.diff(times)>0)):raise ValueError('V2 actual video relative PTS discontinuity')
        last=stream.get('last_pts_seconds')
        if type(last) not in (int,float) or not np.isfinite(last) or abs(last-origin-(n-1)/50)>1e-6:raise ValueError('V2 actual last PTS differs from full frame timeline')
        if not isinstance(stream.get('origin_rule'),str) or not stream['origin_rule']:raise ValueError('Video container-origin interpretation missing')
        if bound.get(str(Path(stream['path']).resolve()))!=stream.get('sha256'):raise ValueError('Video not bound in independent inputs')
    check_video_pair(streams)
    return audit
