"""CPU-only frozen mobile60 factory; validates supplied val-only best, never selects.

No training, inference, or final/development outcome reads. Large artifacts are
stream-hashed once per inode at construction and by explicit CPU phase verify.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import re
import subprocess
from .benchmark import sha,read
from .contract import PROMPT,SCHEMA
from .simulator import validate_config
from . import offline_assets

ROOT=Path(__file__).resolve().parents[2];SCRATCH=ROOT/'.scratch/pi05_g1';ROUND=SCRATCH/'round03_mobile'
SCENARIOS=ROUND/'scenarios_frozen_v3.json'
SCENARIOS_SHA256='6c7885cedb68826ae431319c0d65f91e29d6749d350853732f2c7a559c050953'
TEMPLATE=ROUND/'prepared_long5m_v4_native_seed201_dev03_offline_unused_assets.json'
NORM=ROUND/'prepared/train_norm_stats.json';MANIFEST=ROUND/'prepared/split_manifest.json';OFFICIAL_BASE=SCRATCH/'pi05_base_pytorch'
OFFICIAL_BASE_SHA256='c1af1d96ce7d40f1f66fafd9adc658c39c88581a017ea5283b1eaa8dffac565d'
TOKENIZER=SCRATCH/'venv/openpi-cache/big_vision/paligemma_tokenizer.model'
TOKENIZER_SHA256='8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6'
NATIVE_RECEIPT=ROOT/'.scratch/unitree_g1/policy/mobile_box_files.json'
NATIVE_SOURCE=ROOT/'upstream/unitree_g1/isaac_gr00t_n16'
OFFLINE_HELPER=ROOT/'experiments/pi05_g1_mobile/offline_assets.py'
OBJECT_LIBRARY=ROOT/'upstream/unitree_g1/isaaclab_arena_static/isaaclab_arena/assets/object_library.py'
RUNTIME_INVENTORY=ROUND/'simulator_inventory.json';PRECISION_RULE=ROUND/'runtime_precision_rule.json'
PI_PYTHON=SCRATCH/'venv/bin/python'
POLICIES=('gr00t_n16','pi05_mobile','official_pi05_base')
SOURCE_ROOTS=(ROOT/'experiments/pi05_g1',ROOT/'experiments/pi05_g1_mobile',ROOT/'experiments/pi05_g1_round02',SCRATCH/'openpi/src',SCRATCH/'openpi/packages/openpi-client/src',NATIVE_SOURCE,ROOT/'upstream/unitree_g1/isaaclab_arena_static',ROOT/'upstream/unitree_g1/isaaclab_static/source')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def artifact(path,cache,expected=None):
    path=Path(path).resolve();st=path.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    if key not in cache:cache[key]=sha(path)
    if expected is not None and cache[key]!=expected:raise ValueError('Pinned artifact SHA mismatch: '+str(path))
    return {'path':str(path),'sha256':cache[key],'bytes':st.st_size,'device':st.st_dev,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns}


def git_identity(directory,expected):
    cmd=['git','-c','safe.directory='+str(directory),'-C',str(directory)]
    commit=subprocess.check_output(cmd+['rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(cmd+['status','--porcelain','--untracked-files=no'],text=True).strip()
    if commit!=expected or dirty:raise ValueError('Native checkout must match clean pinned revision')
    return {'path':str(directory),'commit':commit,'tracked_dirty':False}


def docker_identity(inventory):
    expected=inventory['runtime'];image=expected['owned_docker_image']
    actual=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',image],text=True).strip()
    if actual!=expected['image_id']:raise ValueError('Frozen source Docker image identity changed')
    return {'image':image,'image_id':actual,'source_build':expected['actual_sdk_VERSION'],'arena_commit':expected['arena_head'],'lab_commit':expected['lab_head']}


def verify_training(selected,result_path,frozen,cache):
    selected=Path(selected).resolve();result_path=Path(result_path).resolve();run=result_path.parent;result=read(result_path)
    if result.get('schema')!='pi05_g1_mobile_training_result_v1' or result.get('status')!='completed' or result.get('test_loaded') is not False:raise ValueError('Completed mobile training with test unloaded required')
    stopped=result.get('training_stopped',{})
    if stopped.get('complete_optimizer_update_boundary') is not True or stopped.get('step')!=result.get('step'):raise ValueError('Real completed optimizer-boundary training_stopped required')
    if selected.parent!=run or Path(result['selected_checkpoint']).resolve()!=selected:raise ValueError('Supplied selected must be recorded direct checkpoint child')
    pointer_path=run/'best_checkpoint.json';pointer=read(pointer_path);pointer_target=Path(pointer['path']);pointer_target=pointer_target.resolve() if pointer_target.is_absolute() else (run/pointer_target).resolve()
    if pointer_target!=selected:raise ValueError('Selected checkpoint differs from existing best pointer')
    identity=result['identity']
    if identity['manifest_sha256']!=sha(MANIFEST) or identity['norm_sha256']!=sha(NORM) or identity['prompt']!=PROMPT or identity['contract_schema']!=SCHEMA:raise ValueError('Training data/norm/prompt/contract identity differs')
    admission_path=run/'input_manifest.json';admission=read(admission_path)
    if identity['input_manifest_sha256']!=sha(admission_path):raise ValueError('Training input manifest changed')
    manifest=read(MANIFEST);val_ids=manifest['validation'];val_records=[r for r in manifest['files'] if r['episode_id'] in val_ids]
    expected_chunks=sum(math.ceil(r['length']/50) for r in val_records);expected_frames=sum(r['length'] for r in val_records)
    if len(val_ids)!=10 or len(val_records)!=10 or admission['validation_expected_chunks']!=expected_chunks:raise ValueError('Complete validation coverage identity changed')
    baseline=admission['official_base']
    if Path(baseline['path']).resolve()!=OFFICIAL_BASE/'model.safetensors':raise ValueError('Training was not initialized from official base')
    if baseline['sha256']!=OFFICIAL_BASE_SHA256:raise ValueError('Official base differs from previously verified converted payload SHA')
    base_artifact=artifact(baseline['path'],cache,OFFICIAL_BASE_SHA256)
    candidates=[]
    for directory in sorted(run.iterdir()):
        if not directory.is_dir() or not re.fullmatch(r'(?:step|final)_\d{6}',directory.name):continue
        mp=directory/'metadata.json';wp=directory/'model.safetensors'
        if not mp.is_file() or not wp.is_file():raise ValueError('Completed run contains partial checkpoint/payload: '+str(directory))
        meta=read(mp);frozen[str(mp)]=sha(mp)
        if meta.get('step')!=int(directory.name.split('_')[-1]) or meta['step']>result['step']:raise ValueError('Checkpoint local step differs from completed training')
        side=directory/'metadata_sha256.json'
        if not side.is_file() or read(side).get('metadata_sha256')!=sha(mp):raise ValueError('Checkpoint metadata SHA receipt missing/mismatch')
        frozen[str(side)]=sha(side)
        if meta.get('schema')!='pi05_g1_mobile_training_checkpoint_v1' or meta.get('identity')!=identity:raise ValueError('Checkpoint metadata identity differs within completed training')
        if meta.get('validation_complete') is not True:continue
        metric=meta.get('validation',{});value=meta.get('validation_flow_loss')
        if metric.get('complete') is not True or metric.get('episodes')!=10 or metric.get('chunks')!=admission['validation_expected_chunks'] or metric.get('frames')!=expected_frames or type(value) not in (int,float) or not math.isfinite(value) or metric.get('flow_loss')!=value:raise ValueError('Incomplete/nonfinite validation cannot certify best')
        candidates.append({'path':str(directory),'step':meta['step'],'validation_flow_loss':value,'metadata_sha256':sha(mp),'weights_sha256':meta.get('weights_sha256')})
    if not candidates:raise ValueError('No complete validation candidates')
    found=next((c for c in candidates if Path(c['path'])==selected),None);minimum=min(c['validation_flow_loss'] for c in candidates)
    earliest=min(candidates,key=lambda c:(c['validation_flow_loss'],c['step'],0 if Path(c['path']).name.startswith('step_') else 1))
    if not found or found['path']!=earliest['path']:raise ValueError('Best pointer violates existing strict-improvement/earlier-checkpoint tie rule')
    if not found or found['validation_flow_loss']!=minimum or pointer.get('validation_flow_loss')!=minimum or result.get('selected_validation_flow_loss')!=minimum or pointer.get('metadata_sha256')!=found['metadata_sha256']:raise ValueError('Existing best pointer is not consistent with complete validation minimum')
    if not found['weights_sha256']:raise ValueError('Selected checkpoint lacks recorded payload SHA')
    fine_artifact=artifact(selected/'model.safetensors',cache,found['weights_sha256'])
    for p in [result_path,pointer_path,admission_path,NORM,MANIFEST]:frozen[str(p.resolve())]=sha(p)
    return {'recorded_selected':found,'all_complete_validation_metadata':candidates,'training_result':str(result_path),'training_result_sha256':sha(result_path),'identity':identity,'selection_scope':'Verify supplied training best against complete validation only. No re-selection, final outcomes, test observations, or development scores read.'},base_artifact,fine_artifact


def verify_offline_assets(template,frozen):
    # Pin this narrow import filter to the admitted dev03 source, including all
    # retained official AST nodes. Never import or execute the object library.
    if template.get('offline_unused_assets') is not True:
        raise ValueError('All formal policies require offline_unused_assets=True')
    admitted=template.get('frozen_files',{})
    for path in (OFFLINE_HELPER,OBJECT_LIBRARY):
        path=path.resolve();expected=admitted.get(str(path))
        if not expected or sha(path)!=expected:
            raise ValueError('Pinned offline asset helper/source SHA mismatch: '+str(path))
        frozen[str(path)]=expected
    _,receipt=offline_assets.filtered_source(OBJECT_LIBRARY.read_bytes(),offline_assets.SOURCE_PATH)
    return {'enabled':True,'helper_path':str(OFFLINE_HELPER.resolve()),
            'helper_sha256':frozen[str(OFFLINE_HELPER.resolve())],
            'host_official_source_path':str(OBJECT_LIBRARY.resolve()),
            'filter_receipt':receipt}


def collect_dependencies(template,scenarios,frozen,cache):
    offline=verify_offline_assets(template,frozen)
    # Keep every admitted non-code asset hash. CPU tools added since development
    # are frozen at execution versions below rather than mistaken for asset drift.
    for document in [template,scenarios]:
        for name,digest in document.get('frozen_files',{}).items():
            p=Path(name).resolve()
            if p.suffix=='.py':continue
            if sha(p)!=digest:raise ValueError('Admitted asset/dependency changed: '+str(p))
            frozen[str(p)]=digest
    for p in [*template['assets'].values(),*template.get('mesh_url_paths',{}).values(),template['geometry_definition'],template['robot_geometry'],scenarios['action_limits_file'],PRECISION_RULE,RUNTIME_INVENTORY]:
        path=Path(p).resolve();frozen[str(path)]=sha(path)
    # Explicit ASCII overlay references; full binary USD texture/base closure is
    # supplied by the root-admitted template manifest and preserved above.
    queue=[Path(template['assets']['background']).resolve()];seen=set()
    while queue:
        p=queue.pop()
        if p in seen:continue
        seen.add(p);frozen[str(p)]=sha(p)
        if p.suffix=='.usda':
            for name in re.findall(r'@([^@]+)@',p.read_text()):
                if '://' in name:raise ValueError('Unresolved network dependency in frozen overlay')
                child=Path(name);child=child.resolve() if child.is_absolute() else (p.parent/child).resolve()
                frozen[str(child)]=sha(child)
                if child.suffix=='.usda':queue.append(child)
    roots=[*SOURCE_ROOTS,Path(template['source_harness_dir'])]
    for directory in roots:
        files=sorted(Path(directory).rglob('*.py'))
        if not files:raise ValueError('Execution source root missing/empty: '+str(directory))
        for p in files:frozen[str(p.resolve())]=sha(p)
    if sha(TOKENIZER)!=TOKENIZER_SHA256:raise ValueError('Runtime tokenizer differs from fixed SHA')
    frozen[str(TOKENIZER.resolve())]=TOKENIZER_SHA256
    receipt=read(NATIVE_RECEIPT);frozen[str(NATIVE_RECEIPT.resolve())]=sha(NATIVE_RECEIPT)
    if receipt['profile']!='mobile_box' or receipt['revision']!=template['HFmodel']['revision']:raise ValueError('Native mobile model receipt differs')
    native_root=Path(receipt['model_root']).resolve();large=[]
    for item in receipt['files']:
        p=native_root/item['path']
        if p.stat().st_size!=item['bytes']:raise ValueError('Native artifact size mismatch')
        if p.suffix=='.safetensors':large.append(artifact(p,cache,item['sha256']))
        else:
            if sha(p)!=item['sha256']:raise ValueError('Native metadata artifact SHA mismatch')
            frozen[str(p)]=item['sha256']
    processor_root=NATIVE_SOURCE/'gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2'
    for required in ['vocab.json','merges.txt','tokenizer_config.json','preprocessor_config.json','processor_config.json','chat_template.json']:
        if not (processor_root/required).is_file():raise ValueError('Vendored native processor/tokenizer dependency missing: '+required)
    for p in sorted(processor_root.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            if p.suffix in ['.safetensors','.bin'] or p.stat().st_size>16*1024**2:raise ValueError('Unexpected weight/large payload in native processor resources')
            frozen[str(p.resolve())]=sha(p)
    source=git_identity(NATIVE_SOURCE,receipt['mobile_gr00t_source']['revision'])
    for name,digest in receipt['mobile_gr00t_source']['files'].items():
        p=NATIVE_SOURCE/name
        if sha(p)!=digest:raise ValueError('Native source receipt mismatch')
        frozen[str(p.resolve())]=digest
    for p in OFFICIAL_BASE.glob('*.json'):frozen[str(p.resolve())]=sha(p)
    return {'offline_unused_assets':offline,'native_checkout':source,'docker':docker_identity(read(RUNTIME_INVENTORY)),'native_artifacts':large,'overlay_files':sorted(map(str,seen)),'native_processor_root':str(processor_root)}


def make_plan(selected_checkpoint,fine_training_result,output,episode_root):
    selected=Path(selected_checkpoint);result=Path(fine_training_result);output=Path(output).resolve();episode_root=Path(episode_root).resolve()
    if not selected.is_absolute() or not result.is_absolute():raise ValueError('Explicit absolute selected checkpoint and training result required')
    if output.exists() or episode_root.exists():raise FileExistsError('Both plan and raw episode roots must be fresh; never overwrite partial evidence')
    if output.is_relative_to(episode_root) or episode_root.is_relative_to(output):raise ValueError('Plan root and raw episode root must be separate trees')
    if sha(SCENARIOS)!=SCENARIOS_SHA256:raise ValueError('Frozen scenarios SHA changed')
    scenarios=read(SCENARIOS);template=read(TEMPLATE)
    if template.get('offline_unused_assets') is not True:raise ValueError('All formal policies require offline_unused_assets=True')
    if scenarios['schema']!='pi05_g1_mobile_frozen_scenarios_v1' or scenarios['primary_policies']!=list(POLICIES):raise ValueError('Scenario policy identity changed')
    cases=scenarios['cases']
    if len(cases)!=20 or len({c['case_id'] for c in cases})!=20 or {c['seed'] for c in cases}!={1040,1041,1042,1043} or len({c['cluster_id'] for c in cases})!=5:raise ValueError('Frozen20/5positions/4seeds required')
    if (scenarios['episode_seconds'],scenarios['ticks'],scenarios['native_action_horizon'],scenarios['replan_interval_ticks'],scenarios['control_hz'],scenarios['physics_hz'])!=(60.,3000,50,50,50,200):raise ValueError('Frozen duration/cadence/horizon changed')
    frozen={str(SCENARIOS.resolve()):SCENARIOS_SHA256,str(TEMPLATE.resolve()):sha(TEMPLATE)};cache={}
    selection,base,fine=verify_training(selected,result,frozen,cache)
    dependencies=collect_dependencies(template,scenarios,frozen,cache)
    artifacts={'official_pi05_base':base,'pi05_mobile':fine,'gr00t_n16':dependencies['native_artifacts']}
    selection.update(schema='pi05_g1_mobile_execution_selection_v1',model_artifacts=artifacts,dependencies=dependencies,scenarios_sha256=SCENARIOS_SHA256,frozen_files=dict(frozen),head_only=False,pi_common_norm=str(NORM.resolve()),pi_common_runtime='experiments.pi05_g1_mobile.pi05_service',large_artifact_verification='Factory streamSHA once per physical inode; explicit CPU phase-start/end verify_artifacts rehashes every payload once. Paths/inodes are recorded identity, never substituted for a SHA.')
    # No evidence/result lookup occurs: roots are merely required not to exist.
    output.mkdir(parents=True);write(output/'selection_receipt.json',selection);frozen[str(output/'selection_receipt.json')]=sha(output/'selection_receipt.json')
    jobs=[]
    for case_index,case in enumerate(cases):
        order=POLICIES[case_index%3:]+POLICIES[:case_index%3]
        for policy in order:
            config=copy.deepcopy(template);case_dir=episode_root/policy/case['case_id'];evidence=case_dir/'decoded.jsonl'
            config.update(mode='policy',policy_id=policy,case={k:v for k,v in case.items() if k!='scene'},output_dir=str(case_dir),episode_seconds=scenarios['episode_seconds'],ticks=scenarios['ticks'],native_action_horizon=50,replan_interval_ticks=50,scene=copy.deepcopy(case['scene']),acceptance=copy.deepcopy(scenarios['acceptance']),observer_camera=copy.deepcopy(scenarios['observer_camera']),action_limits_file=scenarios['action_limits_file'],prompt=PROMPT,evaluate_acceptance=True,stop_on_verified_success=True,capture_preflight_scene_state=False,offline_unused_assets=True,max_run_wall_seconds=840,save_full_rgb=False)
            config['frozen_files']=dict(frozen)
            if policy=='gr00t_n16':
                command=list(template['policy_command'])
                for flag,value in [('--seed',str(case['seed'])),('--decoded-evidence',str(evidence))]:
                    if command.count(flag)!=1:raise ValueError('Native template argv missing explicit seed/evidence')
                    command[command.index(flag)+1]=value
                config['policy_command']=command
            else:
                checkpoint=selected if policy=='pi05_mobile' else OFFICIAL_BASE
                config['policy_command']=[str(PI_PYTHON),'-m','experiments.pi05_g1_mobile.pi05_service','--checkpoint',str(checkpoint),'--norm-stats',str(NORM.resolve()),'--decoded-evidence',str(evidence)]
                config['norm_stats']=str(NORM.resolve());config['HFmodel']={'family':'pi05','initialization':'official pi05_base','selected_by_validation':policy=='pi05_mobile'}
                config['policy_artifacts']=artifacts[policy]
            config=validate_config(config);cp=output/'configs'/str(policy+'_'+case['case_id']+'.json');write(cp,config)
            jobs.append({'name':policy+'_'+case['case_id'],'policy':policy,'case_id':case['case_id'],'config_path':str(cp),'config_sha256':sha(cp),'estimated_wall_seconds':900,'result_path':str(case_dir/'result.json')})
    frozen.update({j['config_path']:j['config_sha256'] for j in jobs})
    plan={'schema':'pi05_g1_mobile_benchmark_plan_v1','primary_policies':list(POLICIES),'case_ids':[c['case_id'] for c in cases],'cases':cases,'jobs':jobs,'frozen_files':frozen,'model_artifacts':artifacts,'selection_receipt':str(output/'selection_receipt.json'),'raw_episode_root':str(episode_root),'order_rule':'Case-major with policy order rotated by caseindex mod3; no outcome-based ordering','large_artifact_phase_verification_required':True,'cpu_final_audit_and_benchmark_required':True,'scope':scenarios['task_scope'],'source_reset_randomization':scenarios['source_reset_randomization']}
    write(output/'benchmark_plan.json',plan);return plan


def verify_artifacts(plan_path,output):
    plan_path=Path(plan_path).resolve();output=Path(output)
    if output.exists():raise FileExistsError('Fresh CPU phase verification receipt required')
    plan=read(plan_path);cache={};observed={}
    for p,expected in plan['frozen_files'].items():
        if sha(p)!=expected:raise ValueError('Frozen execution input changed: '+p)
    selection=read(plan['selection_receipt'])
    if docker_identity(read(RUNTIME_INVENTORY))!=selection['dependencies']['docker']:raise ValueError('Execution Docker image identity changed since factory')
    native=selection['dependencies']['native_checkout']
    if 'commit' in native and git_identity(NATIVE_SOURCE,native['commit'])!=native:raise ValueError('Native checkout identity changed since factory')
    for policy,entries in plan['model_artifacts'].items():
        for entry in entries if isinstance(entries,list) else [entries]:observed[entry['path']]=artifact(entry['path'],cache,entry['sha256'])
    receipt={'schema':'pi05_g1_mobile_execution_phase_sha_v1','status':'verified','plan':str(plan_path),'plan_sha256':sha(plan_path),'frozen_files':plan['frozen_files'],'model_artifacts_observed':observed,'gpu_started':False};write(output,receipt);return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected-checkpoint',type=Path);p.add_argument('--fine-training-result',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--episode-root',type=Path);p.add_argument('--verify-plan',type=Path)
    a=p.parse_args()
    if a.verify_plan:result=verify_artifacts(a.verify_plan,a.output)
    else:
        if not all([a.selected_checkpoint,a.fine_training_result,a.episode_root]):p.error('Factory requires selected-checkpoint, fine-training-result, episode-root')
        result=make_plan(a.selected_checkpoint,a.fine_training_result,a.output,a.episode_root)
    print(json.dumps({'schema':result['schema'],'jobs':len(result.get('jobs',[]))}))

if __name__=='__main__':main()
