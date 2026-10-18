"""Freeze disjoint cases, sparse-translation normalization, and explicit bounds candidates."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from experiments.pi05_g1.contract import ACTION_NAMES, encode_row


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    path=Path(path);data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise FileExistsError(f'Frozen output differs: {path}')
    else:
        with path.open('xb') as f:f.write(data)
    return sha(path)


def case(prefix, i, xy, seed):
    return {'case_id':f'{prefix}{i}_seed{seed}','seed':seed,'apple_offset_xy_m':list(xy)}


def make_cases():
    final=[case('p',i,xy,seed) for i,xy in enumerate(((0,0),(.02,0),(-.02,0),(0,.02),(0,-.02))) for seed in range(40,44)]
    dev=[case('dev',i,xy,100+i) for i,xy in enumerate(((0,0),(.01,0),(-.01,0),(0,-.01)))]
    teacher=[case('teacher',i,xy,seed) for i,xy in enumerate(((0,0),(.01,0),(0,-.01))) for seed in range(200,204)]
    return {'final_cases.json':{'schema':'pi05_g1_eval_cases_v1','role':'final_only_never_tune','cases':final},
            'development_cases.json':{'schema':'pi05_g1_eval_cases_v1','role':'development_only','cases':dev},
            'teacher_cases.json':{'schema':'pi05_g1_eval_cases_v1','role':'teacher_only_no_final_data', 'cases':teacher,
              'train_case_ids':[c['case_id'] for c in teacher if c['seed']!=203],
              'validation_case_ids':[c['case_id'] for c in teacher if c['seed']==203]}}


def make_source_fallback():
    train_xy=[(-.01,-.01),(-.01,0),(-.01,.01),(0,-.01),(0,0),(0,.01),(.01,-.01),(.01,0),(.01,.01)]
    val_xy=[(-.005,.005),(.005,-.005),(.005,.005)]
    train=[case('source_teacher',i,xy,200+i) for i,xy in enumerate(train_xy)]
    validation=[case('source_teacher_val',i,xy,213+i) for i,xy in enumerate(val_xy)]
    return {'schema':'pi05_g1_eval_cases_v1','role':'source_expert_fallback_only_route_before_collection',
            'cases':train+validation,'train_case_ids':[c['case_id'] for c in train],
            'validation_case_ids':[c['case_id'] for c in validation],
            'routing_rule':'Use only if pre-final development routing rejects the GR00T teacher family; never collect both families and choose by final performance.'}


def sparse_norm_v2(original, train_actions):
    result=copy.deepcopy(original);changes=[]
    for j in (29,30):
        lo=float(original['actions']['q01'][j]);hi=float(original['actions']['q99'][j]);values=train_actions[:,j]
        if hi-lo>1e-6 or values.min()==values.max():raise ValueError('Only collapsed nonconstant vx/vy may change')
        if lo!=0 or hi!=0:raise ValueError('This experiment requires zero-centered collapsed translation statistics')
        radius=float(np.max(np.abs(values)))
        result['actions']['q01'][j]=-radius;result['actions']['q99'][j]=radius
        changes.append({'index':j,'channel':ACTION_NAMES[j],'center':0.,'radius':radius,
                        'raw_train_min':float(values.min()),'raw_train_max':float(values.max()),
                        'nonzero_train_frames':int(np.count_nonzero(values))})
    return result,changes


def prepare(source_root, output):
    source_root=Path(source_root);output=Path(output)
    original_path=source_root/'prepared/train_norm_stats.json';manifest_path=source_root/'prepared/split_manifest.json'
    original=json.loads(original_path.read_text());manifest=json.loads(manifest_path.read_text());dataset=Path(manifest['dataset_root'])
    files={int(r['episode_id']):r for r in manifest['files']};actions=[]
    for eid in manifest['train']:
        rows=pq.read_table(dataset/files[eid]['parquet']).to_pylist()
        actions.extend(encode_row(row)['actions'] for row in rows)
    actions=np.asarray(actions,np.float64)
    if actions.shape[1]!=32 or not np.isfinite(actions).all():raise ValueError('Training actions must be finite N x 32')
    v2,changes=sparse_norm_v2(original,actions)
    if original['state']!=v2['state']:raise AssertionError('State normalization changed')
    case_outputs=make_cases();hashes={name:write_once(output/name,doc) for name,doc in case_outputs.items()}
    hashes['norm_stats_v2.json']=write_once(output/'norm_stats_v2.json',v2)
    lower=np.asarray(original['actions']['q01']).copy();upper=np.asarray(original['actions']['q99']).copy()
    minimum=actions.min(0);maximum=actions.max(0);lower[29:31]=minimum[29:31];upper[29:31]=maximum[29:31]
    common={'schema':'pi05_g1_round02_physical_bounds_candidate_v1','action_names':list(ACTION_NAMES),
            'source_manifest_sha256':sha(manifest_path),'original_norm_sha256':sha(original_path),
            'source_split':'train_only_166','train_episodes':manifest['train'],
            'upper28_hardware_intersection_required':True,'final_selection_requires_development_physical_evidence':True}
    a={**common,'candidate':'A_quantile_plus_sparse_translation','lower':lower.tolist(),'upper':upper.tolist(),
       'semantics':'Round01 q01/q99 bounds except vx/vy use original training raw extrema; asymmetric physical limits retained.'}
    b={**common,'candidate':'B_all32_train_raw_extrema_NOT_SELECTED','lower':minimum.tolist(),'upper':maximum.tolist(),
       'selected':False,'semantics':'All32 raw training extrema candidate; not the default and not a declaration of hardware feasibility.'}
    for name,doc in (('physical_bounds_A_quantile_plus_sparse_translation.json',a),('physical_bounds_B_train_raw_extrema_candidate.json',b)):
        hashes[name]=write_once(output/name,doc)
    from experiments.pi05_g1.contract import normalize,denormalize
    old_restored=denormalize(normalize(actions.astype(np.float32),original['actions']),original['actions'])
    new_restored=denormalize(normalize(actions.astype(np.float32),v2['actions']),v2['actions'])
    regression=[]
    for j in (29,30):
        rare=actions[:,j]!=0
        old_error=float(np.max(np.abs(old_restored[rare,j]-actions[rare,j])))
        new_error=float(np.max(np.abs(new_restored[rare,j]-actions[rare,j])))
        if old_error<=1e-6 or new_error>2e-8:raise AssertionError('Rare target regression did not detect old loss or new roundtrip')
        if normalize(np.zeros(32,np.float32),v2['actions'])[j]!=0:raise AssertionError('Zero translation must remain normalized zero')
        regression.append({'channel':ACTION_NAMES[j],'actual_nonzero_targets':int(rare.sum()),'old_roundtrip_max_error':old_error,'v2_roundtrip_max_error':new_error,'old_version_fails_this_regression':True})
    provenance={'schema':'pi05_g1_round02_sparse_norm_v2_provenance','source_norm_sha256':sha(original_path),
       'source_manifest_sha256':sha(manifest_path),'source_revision':manifest['revision'],'train_only_episodes':manifest['train'],
       'train_frames':len(actions),'changed_quantile_channels':changes,'other_statistics_unchanged':True,
       'actual_sparse_target_roundtrip_regression':regression,'bounds_policy':'Normalization range is independent of physical limits; A and B are separately recorded candidates. B is not selected.'}
    hashes['norm_v2_provenance.json']=write_once(output/'norm_v2_provenance.json',provenance)
    write_once(output/'inputs_freeze.json',{'schema':'pi05_g1_round02_inputs_freeze_v1','files_sha256':hashes,
       'case_roles_disjoint':True,'teacher_validation_seed':203,'final_cases_forbidden_for_tuning':True,
       'selected_physical_bounds':None,'round01_unchanged':True})
    return provenance


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.source_root,args.output),indent=2))
