"""Prepare fixed V4 reset/hold evidence, CPU only; never launches simulation.

python -m experiments.pi05_g1_mobile.prepare_v4_preflight --preparation-dir NEW
Both output case directories must not exist. These are zero-VLA diagnostics,
not teacher demonstrations, autonomous trials or task scores.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


REPO=Path(__file__).resolve().parents[2]
ROUND=REPO/'.scratch/pi05_g1/round03_mobile'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(preparation_dir):
    import numpy as np
    from .contract import UPPER_NAMES
    from .simulator import validate_config
    output=Path(preparation_dir).resolve()
    if output.exists():raise FileExistsError('Preparation directory must be new')
    hold=ROUND/'preflight/source_hold10_seed42_attempt02'
    observer_case=ROUND/'development/source_native_seed200_attempt02'
    base=json.loads((hold/'config.json').read_text())
    if json.loads((hold/'result.json').read_text()).get('status')!='completed':
        raise ValueError('Original hold10 must be completed before using its reset')
    with (hold/'trajectory.jsonl').open() as stream:reset=json.loads(stream.readline())
    if reset['control_tick']!=0 or reset['physics_steps_since_reset']!=0:
        raise ValueError('Expected actual original reset row')
    q=np.asarray(reset['joint_positions'],np.float32);names=reset['source_joint_names']
    if q.shape!=(43,) or len(set(names))!=43 or not np.isfinite(q).all():
        raise ValueError('Invalid reset joint evidence')
    action=np.r_[q[[names.index(n) for n in UPPER_NAMES]],.75,0.,0.,0.].astype(np.float32)
    matrix=np.repeat(action[None],100,axis=0)
    observer=json.loads((observer_case/'config.json').read_text())['observer_camera']
    layer=ROUND/'assets/long5m_open_aisle_v4.usda'
    composition=json.loads((ROUND/'assets/long5m_open_aisle_v4_composition.json').read_text())
    if digest(layer)!=composition['overlay_sha256'] or composition['unexpected_differences']:
        raise ValueError('V4 overlay identity/composition changed')
    cases={'diagnostic10':ROUND/'preflight/v4_diagnostic10_seed42_capture',
           'replay100':ROUND/'preflight/v4_replay100_seed42_capture'}
    if any(path.exists() for path in cases.values()):raise FileExistsError('Physical case outputs must be new')
    output.mkdir(parents=True)
    actions_path=output/'reset_hold100_actions.npy'
    with actions_path.open('xb') as stream:np.save(stream,matrix,allow_pickle=False)
    common=copy.deepcopy(base)
    common.update(observer_camera=observer,capture_preflight_scene_state=True,
        save_full_rgb=True,evaluate_acceptance=False,stop_on_verified_success=False,
        diagnostic_scope='Fixed original-reset upper28 targets + height0.75m + zero navigation; zero VLA, no autonomy/task score')
    common['assets']['background']=str(layer)
    common['scene']={'bin_pose':{'position_xyz':[-.245,-4.9,-.2641],'rotation_xyzw':[0.,0.,1.,0.]}}
    # Hash producer sources and the complete local background dependency tree.
    # This hashes physics/render assets, never GR00T/Pi checkpoint payloads.
    paths={Path(p).resolve() for p in base.get('frozen_files',{})}
    paths.update(Path(p).resolve() for p in common['assets'].values())
    paths.update(Path(p).resolve() for p in common.get('mesh_url_paths',{}).values())
    for package in ('pi05_g1_mobile','pi05_g1_round02','pi05_g1'):
        paths.update((REPO/'experiments'/package).glob('*.py'))
    paths.update([ROUND/'assets/long5m_background.usda',layer,Path(__file__).resolve()])
    original_background=Path(base['assets']['background'])
    paths.add(original_background)
    paths.update(p for p in original_background.parent.rglob('*') if p.is_file())
    # A source config's hashes intentionally refer to historical code. New
    # diagnostic configurations bind current bytes and retain historical IDs
    # separately rather than attempting to overwrite historical receipts.
    frozen={str(p):digest(p) for p in sorted(paths)}
    common['frozen_files']=frozen
    config_paths={}
    for name,path in cases.items():
        config=copy.deepcopy(common)
        config.update(mode='diagnostic_hold' if name=='diagnostic10' else 'replay',
            ticks=10 if name=='diagnostic10' else 100,
            case={'case_id':path.name,'seed':42},output_dir=str(path),
            policy_id='diagnostic_fixed_reset_hold_no_vla')
        if name=='replay100':
            config.update(replay_actions_npy=str(actions_path),replay_actions_sha256=digest(actions_path))
            config['frozen_files'][str(actions_path)]=digest(actions_path)
        validate_config(config)
        cp=output/(name+'_config.json')
        with cp.open('x') as stream:stream.write(json.dumps(config,indent=2,allow_nan=False)+'\n')
        config_paths[name]={'path':str(cp),'sha256':digest(cp),'physical_output_dir':str(path),'ticks':config['ticks']}
    if any(digest(path)!=expected for path,expected in frozen.items()):
        raise ValueError('A frozen dependency changed while preparing diagnostics')
    manifest={'schema':'pi05_g1_mobile_v4_preflight_preparation_v1',
        'status':'cpu_prepared_not_executed','configurations':config_paths,
        'replay':{'path':str(actions_path),'shape':list(matrix.shape),'dtype':str(matrix.dtype),
                  'sha256':digest(actions_path),'all100_rows_identical':True,'action32':action.tolist()},
        'original_reset_evidence':{'config':str(hold/'config.json'),'config_sha256':digest(hold/'config.json'),
            'trace':str(hold/'trajectory.jsonl'),'trace_sha256':digest(hold/'trajectory.jsonl'),
            'control_tick':0,'upper_names':list(UPPER_NAMES)},
        'observer_source':{'config':str(observer_case/'config.json'),'sha256':digest(observer_case/'config.json')},
        'frozen_dependency_count':len(frozen),'frozen_dependencies':frozen,
        'scope':'V4 physics/camera diagnostic only; zero VLA; no teacher/formal-case freeze, autonomous score or physics qualification; GPU execution remains root-owned'}
    with (output/'preparation_manifest.json').open('x') as stream:stream.write(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preparation-dir',type=Path,required=True)
    args=parser.parse_args();manifest=prepare(args.preparation_dir)
    print(json.dumps({'status':manifest['status'],'configurations':manifest['configurations'],
                      'frozen_dependency_count':manifest['frozen_dependency_count']}))


if __name__=='__main__':main()
