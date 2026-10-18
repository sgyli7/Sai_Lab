"""CPU-only lineage and complete mixed-validation candidates across declared runs.

Metadata hashes are always checked. Large model hashes are opt-in; the recovery
launcher must enable them once, while factory phase hashing remains separate.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

CHECKPOINT_SCHEMA='pi05_g1_round02_training_checkpoint_v1'
LINEAGE_SCHEMA='pi05_g1_round02_checkpoint_lineage_v1'
ROSTER_SCHEMA='pi05_g1_round02_candidate_training_dirs_v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))


def declared_training_dirs(value,current=None):
    """Read a roster path or explicit run-directory list; never discover siblings."""
    if isinstance(value,(str,Path)):
        doc=read(value)
        if doc.get('schema')!=ROSTER_SCHEMA:raise ValueError('Unknown candidate training roster')
        directories=doc['training_dirs']
        if current is not None and Path(doc['current_training_dir']).resolve()!=Path(current).resolve():raise ValueError('Roster current run mismatch')
    else:directories=value
    paths=[Path(path).resolve() for path in directories]
    if not paths or len(paths)!=len(set(paths)):raise ValueError('Explicit unique candidate training directories required')
    if current is not None and Path(current).resolve() not in paths:raise ValueError('Current run missing from candidate roster')
    return paths


def checked_candidate_path(path,training_dirs):
    candidate=Path(path).resolve()
    if candidate.parent not in declared_training_dirs(training_dirs) or candidate.name.endswith('.incomplete'):raise ValueError('Candidate must be a direct child of a declared training directory')
    return candidate


def resolve_lineage(checkpoint,verify_weights=False):
    """Infer legacy1200 ancestry or validate explicit recovery links recursively."""
    cache={};visiting=set()
    def digest(path):
        path=Path(path).resolve()
        if path not in cache:cache[path]=sha(path)
        return cache[path]
    def walk(path):
        path=Path(path).resolve()
        if path in visiting or len(visiting)>16:raise ValueError('Cyclic or excessive checkpoint lineage')
        if path.name.endswith('.incomplete'):raise ValueError('Incomplete checkpoint cannot be a parent')
        visiting.add(path)
        metadata_path=path/'metadata.json';meta=read(metadata_path);step=meta.get('step')
        if type(step) is not int or step<0:raise ValueError('Invalid local checkpoint step')
        if not (path/'model.safetensors').is_file():raise ValueError('Lineage model missing')
        if meta.get('schema')!=CHECKPOINT_SCHEMA:
            if step!=1200:raise ValueError('Genesis must be the original step1200 checkpoint')
            genesis={'path':str(path),'metadata_sha256':digest(metadata_path),'weights_sha256':digest(path/'model.safetensors') if verify_weights else None,'source_optimizer_steps':1200}
            result={'genesis_checkpoint':str(path),'genesis':genesis,'immediate_parent':None,'inherited_round_updates':0,'local_step':0,'cumulative_round_updates':0,'ancestor_chain':[{**genesis,'kind':'round01_genesis','step':1200}]}
            visiting.remove(path);return result
        explicit=meta.get('checkpoint_lineage');warm=meta.get('warmstart',{})
        parent=Path(explicit['immediate_parent']['path'] if explicit else warm['checkpoint']).resolve()
        inherited=walk(parent)
        parent_metadata=digest(parent/'metadata.json')
        expected_parent_weights=explicit['immediate_parent']['weights_sha256'] if explicit else warm.get('weights_sha256')
        if not isinstance(expected_parent_weights,str) or len(expected_parent_weights)!=64:raise ValueError('Parent weight SHA missing')
        if verify_weights and digest(parent/'model.safetensors')!=expected_parent_weights:raise ValueError('Parent weights SHA mismatch')
        if inherited['genesis']['weights_sha256'] is None and parent==Path(inherited['genesis_checkpoint']):inherited['genesis']['weights_sha256']=expected_parent_weights
        # The caller's actual inheritance binds its parent payload; the current
        # candidate itself may lack an own-payload SHA until phase verification.
        inherited['ancestor_chain'][-1]['weights_sha256']=expected_parent_weights
        parent_info={'path':str(parent),'metadata_sha256':parent_metadata,'weights_sha256':expected_parent_weights,'local_step':read(parent/'metadata.json')['step'],'cumulative_round_updates':inherited['cumulative_round_updates']}
        total=inherited['cumulative_round_updates']+step
        if explicit:
            if explicit.get('schema')!=LINEAGE_SCHEMA or explicit.get('optimizer_reset') is not True:raise ValueError('Unsupported recovery lineage or optimizer reset missing')
            if explicit['immediate_parent'].get('metadata_sha256')!=parent_metadata:raise ValueError('Parent metadata SHA mismatch')
            if warm.get('checkpoint')!=str(parent) or warm.get('weights_sha256')!=expected_parent_weights or warm.get('source_optimizer_payload_loaded') is not False or warm.get('optimizer_reset') is not True:raise ValueError('Warmstart and recovery lineage disagree')
            for key,value in [('inherited_round_updates',inherited['cumulative_round_updates']),('local_step',step),('cumulative_round_updates',total)]:
                if explicit.get(key)!=value or meta.get(key)!=value:raise ValueError('Recovery update accounting mismatch')
            genesis=explicit['genesis']
            if genesis.get('path')!=inherited['genesis_checkpoint'] or genesis.get('metadata_sha256')!=inherited['genesis']['metadata_sha256'] or genesis.get('weights_sha256')!=inherited['genesis']['weights_sha256'] or genesis.get('source_optimizer_steps')!=1200:raise ValueError('Genesis ancestry mismatch')
            receipt=explicit.get('source_failure_receipt')
            if receipt is not None:
                if sha(receipt['path'])!=receipt['sha256']:raise ValueError('Source failure receipt changed')
                failure=read(receipt['path'])
                if Path(failure['source_training_dir']).resolve()!=parent.parent or failure.get('source_step')!=read(parent/'metadata.json')['step'] or failure.get('status')!='host_memory_guard' or failure.get('exit_code')!=137 or failure.get('source_metadata_sha256')!=parent_metadata or failure.get('source_weights_sha256')!=expected_parent_weights:raise ValueError('Source failure receipt does not match immediate parent')
        elif inherited['cumulative_round_updates']!=0 or parent!=Path(inherited['genesis_checkpoint']):raise ValueError('Legacy warmstart may only directly reference original step1200')
        result={'genesis_checkpoint':inherited['genesis_checkpoint'],'genesis':inherited['genesis'],'immediate_parent':parent_info,'inherited_round_updates':inherited['cumulative_round_updates'],'local_step':step,'cumulative_round_updates':total,'ancestor_chain':inherited['ancestor_chain']+[{'path':str(path),'metadata_sha256':digest(metadata_path),'weights_sha256':digest(path/'model.safetensors') if verify_weights else meta.get('weights_sha256'),'kind':'round02_checkpoint','local_step':step,'cumulative_round_updates':total,'optimizer_reset_at_run_start':True}]}
        visiting.remove(path);return result
    return walk(checkpoint)


def collect_candidates(training_dirs,data_config,norm,loss_weights,verify_weights=False):
    """Return all completed validation candidates under one frozen data identity."""
    directories=declared_training_dirs(training_dirs);config=read(data_config);declared=config['sources'];names={s['name'] for s in declared}
    if len(declared)!=2 or len(names)!=2 or {s['kind'] for s in declared}!={'lerobot','npz_video'}:raise ValueError('Exactly original and fresh sources required')
    weights=read(loss_weights)
    if isinstance(weights,dict):weights=weights.get('actions',weights.get('weights',weights.get('loss_weights')))
    if not isinstance(weights,list) or len(weights)!=32:raise ValueError('Explicit32 weights required')
    expected={'data_config_sha256':sha(data_config),'norm_sha256':sha(norm),'loss_weights_sha256':sha(loss_weights)}
    candidates=[];metadata_hashes={};identity=None;genesis=None
    for directory in directories:
        if not directory.is_dir():raise ValueError('Declared candidate training directory missing')
        for metadata in sorted(directory.glob('*/metadata.json')):
            if metadata.parent.name.endswith('.incomplete'):continue
            path=checked_candidate_path(metadata.parent,directories);meta=read(metadata)
            if meta.get('schema')!=CHECKPOINT_SCHEMA:raise ValueError('Unknown checkpoint metadata schema')
            metadata_hashes[str(metadata.resolve())]=sha(metadata)
            current=meta.get('round02_data_identity',{})
            if any(current.get(key)!=value for key,value in expected.items()):raise ValueError('Checkpoint data/norm/loss identity mismatch')
            if identity is None:identity=current
            if current!=identity:raise ValueError('Checkpoint source identity changed across runs')
            sources=current.get('sources',[])
            if len(sources)!=2 or {s['name'] for s in sources}!=names:raise ValueError('Missing source receipts')
            for source in sources:
                declaration=next(s for s in declared if s['name']==source['name']);manifest=Path(declaration['manifest']).resolve()
                if source['kind']!=declaration['kind'] or Path(source['manifest']).resolve()!=manifest or source['manifest_sha256']!=sha(manifest):raise ValueError('Source manifest identity mismatch')
                for key in ('train_sampling_weight','validation_weight'):
                    if source[key]!=declaration[key]:raise ValueError('Source mixture weight changed')
                for key,value in declaration.get('time_sampling',{'kind':'uniform'}).items():
                    if source.get('time_sampling',{}).get(key)!=value:raise ValueError('Time sampling changed')
                if min(source['train_episodes'],source['validation_episodes'],source['validation_chunks'])<=0:raise ValueError('Empty validation coverage')
                if source['kind']=='lerobot' and source['validation_episodes']!=21:raise ValueError('Original validation21 required')
            if meta.get('loss_weights')!=weights:raise ValueError('Loss weights differ')
            lineage=resolve_lineage(path,verify_weights=verify_weights)
            for ancestor in lineage['ancestor_chain']:
                ancestor_metadata=Path(ancestor['path'])/'metadata.json'
                metadata_hashes[str(ancestor_metadata)]=ancestor['metadata_sha256']
                ancestor_doc=read(ancestor_metadata)
                failure=ancestor_doc.get('checkpoint_lineage',{}).get('source_failure_receipt')
                if failure:metadata_hashes[str(Path(failure['path']).resolve())]=failure['sha256']
            if genesis is None:genesis=lineage['genesis_checkpoint']
            if genesis!=lineage['genesis_checkpoint']:raise ValueError('Candidates have different genesis')
            if meta.get('validation_complete') is not True:continue
            metrics=meta.get('validation_sources')
            if not isinstance(metrics,dict) or set(metrics)!=names:raise ValueError('Complete validation requires both sources')
            numerator=denominator=0.
            for source in sources:
                metric=metrics[source['name']];weight=source['validation_weight'];loss=metric['flow_loss']
                if metric['episodes']!=source['validation_episodes'] or metric['chunks']!=source['validation_chunks'] or metric['frames']<=0:raise ValueError('Validation did not cover all declared chunks')
                if not isinstance(loss,(int,float)) or not math.isfinite(loss) or loss<0 or weight<=0 or metric['validation_weight']!=weight:raise ValueError('Invalid validation metric/weight')
                numerator+=loss*weight;denominator+=weight
            value=meta.get('validation_flow_loss')
            if not isinstance(value,(int,float)) or not math.isfinite(value) or not math.isclose(value,numerator/denominator,rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Combined validation aggregation mismatch')
            candidate={'path':str(path),'step':meta['step'],'local_step':meta['step'],'inherited_round_updates':lineage['inherited_round_updates'],'cumulative_round_updates':lineage['cumulative_round_updates'],'validation_flow_loss':value,'metadata_sha256':metadata_hashes[str(metadata.resolve())],'lineage':lineage}
            if verify_weights:candidate['weights_sha256']=sha(path/'model.safetensors')
            candidates.append(candidate)
    return {'candidates':candidates,'identity':identity,'metadata_hashes':metadata_hashes,'genesis_checkpoint':genesis}


def select_global_best(collection,pointer=None,current_training_dir=None):
    candidates=collection['candidates']
    if not candidates:raise ValueError('No complete mixed validation candidates')
    minimum=min(c['validation_flow_loss'] for c in candidates)
    if pointer is None:return next(c for c in candidates if c['validation_flow_loss']==minimum)
    path=Path(pointer['path'])
    if not path.is_absolute():
        if current_training_dir is None:raise ValueError('Relative best pointer needs current run')
        path=Path(current_training_dir)/path
    path=path.resolve();choice=next((c for c in candidates if c['path']==str(path)),None)
    if choice is None or choice['validation_flow_loss']!=minimum or pointer.get('validation_flow_loss')!=minimum:raise ValueError('Best pointer is not global complete validation minimum')
    return choice


def validate_failed_parent(parent,identity,budget_path,max_steps,genesis_checkpoint):
    """Recovery admission; hashes actual parent/genesis once. Never loads tensors."""
    parent=Path(parent).resolve();meta=read(parent/'metadata.json')
    if meta.get('schema')!=CHECKPOINT_SCHEMA or meta.get('step')!=200 or meta.get('validation_complete') is not True:raise ValueError('Recovery requires completed round02 step200')
    if meta.get('round02_data_identity')!=identity:raise ValueError('Recovery parent data identity differs')
    if max_steps>1656:raise ValueError('Recovery cap1656 accounts343 known updates plus one in-flight allowance')
    lineage=resolve_lineage(parent,verify_weights=True)
    if Path(lineage['genesis_checkpoint'])!=Path(genesis_checkpoint).resolve():raise ValueError('Unexpected original step1200 genesis')
    ledger=read(budget_path);matches=[row for row in ledger['runs'] if row.get('stage')=='training' and str(parent.parent) in row.get('command',[])]
    if len(matches)!=1:raise ValueError('Unique failed parent budget run required')
    failure=matches[0]
    if failure.get('status')!='host_memory_guard' or failure.get('exit_code')!=137 or failure.get('accounting_open') or failure.get('last_unit_properties',{}).get('ActiveState')!='inactive':raise ValueError('Parent failure must be closed host guard137')
    events=[]
    for line in (parent.parent/'progress.jsonl').read_text().splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:pass
    updates=[row['step'] for row in events if row.get('event')=='optimizer_step']
    if updates!=list(range(1,344)) or any(row.get('event')=='training_stopped' for row in events):raise ValueError('Failed source must retain343-step interrupted history')
    if list(parent.parent.glob('final_*/metadata.json')):raise ValueError('Failed source unexpectedly has final checkpoint')
    receipt={'schema':'pi05_g1_round02_weights_only_recovery_failure_v1','source_training_dir':str(parent.parent),'source_step':200,'status':failure['status'],'exit_code':failure['exit_code'],'source_budget_path':str(Path(budget_path).resolve()),'source_budget_sha256_at_admission':sha(budget_path),'budget_run':failure,'source_progress_sha256':sha(parent.parent/'progress.jsonl'),'completed_logged_updates':343,'retained_round_updates':200,'unpersisted_completed_updates':143,'inflight_update_allowance':1,'max_new_local_updates':1656,'source_metadata_sha256':sha(parent/'metadata.json'),'source_weights_sha256':sha(parent/'model.safetensors'),'genesis':lineage['genesis'],'optimizer_reset':True}
    return receipt,lineage
