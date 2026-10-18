"""Predeclared fresh-validation-only supplementary checkpoint plan (CPU only).

Never opens final raw results or test observations. Primary selection/configs
remain immutable; supplementary execution and budget admission are separate.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from .checkpoint_lineage import collect_candidates,declared_training_dirs
from .make_benchmark_plan import artifact

POLICY='pi05_fresh_validation'
ROOT=Path(__file__).resolve().parents[2]
CASES={f'p{p}_seed{s}' for p in range(5) for s in range(40,44)}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(),parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+v)))


def require(ok,message):
    if not ok:raise ValueError(message)


def write(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def identity_path(selection,key):
    expected=selection['data_identity'][key]
    paths=[Path(path).resolve() for path,digest in selection['frozen_files'].items() if digest==expected]
    require(len(paths)==1,'Ambiguous/missing frozen identity input: '+key)
    require(sha(paths[0])==expected,'Frozen identity input changed: '+key)
    return paths[0]


def make_plan(evaluation_root,training_dir,predeclaration):
    root=Path(evaluation_root).resolve();training_dir=Path(training_dir).resolve();predeclaration=Path(predeclaration).resolve()
    receipt_path=root/'fresh_validation_selection.json';plan_path=root/'supplementary_fresh_plan.json'
    require(root.is_dir(),'Primary evaluation factory must run first')
    if receipt_path.exists() or plan_path.exists() or any((root/'configs').glob(POLICY+'_*.json')):
        raise FileExistsError('Supplementary receipt/plan/configs must be new; preserve prior attempts')
    declaration=read(predeclaration)
    require(declaration.get('schema')=='pi05_g1_round02_supplementary_fresh_validation_predeclaration_v1'
            and declaration.get('final_outcomes_observed') is False and declaration.get('primary_models_and_mixed_selection_unchanged') is True
            and declaration.get('supplementary_policy_id')==POLICY,'Valid before-final-outcomes supplementary predeclaration required')
    require('tie by explicit run order then checkpoint path' in declaration.get('selection_rule',''),'Predeclared tie rule differs')
    # Existence-only guard: never open/read final outcome files for selection.
    require(not any((root/p/c/'result.json').exists() for p in ('gr00t_n17','pi05_round01','pi05_round02') for c in CASES),
            'Supplementary selection must be frozen before final outcomes exist')
    selection_path=root/'selection_receipt.json';main=read(selection_path);primary_path=root/'primary_plan.json';primary=read(primary_path)
    require(main.get('schema')=='pi05_g1_round02_final_selection_v1' and primary.get('schema')=='pi05_g1_round02_frozen_benchmark_plan_v1'
            and primary.get('phase')=='primary' and primary.get('policies')==['gr00t_n17','pi05_round01','pi05_round02'],'Immutable primary three-policy factory output required')
    require(primary['frozen_files'].get(str(selection_path))==sha(selection_path),'Primary selection receipt changed')
    for path,expected in primary['frozen_files'].items():require(sha(path)==expected,'Primary frozen input changed: '+path)
    roster_path=training_dir/'candidate_training_dirs.json';dirs=declared_training_dirs(roster_path,current=training_dir)
    require([str(p) for p in dirs]==main['selected']['candidate_training_dirs'],'Supplementary roster differs from primary candidate population')
    data=identity_path(main,'data_config_sha256');norm=identity_path(main,'norm_sha256');loss=identity_path(main,'loss_weights_sha256')
    collection=collect_candidates(dirs,data,norm,loss,verify_weights=False)
    require(collection['identity']==main['data_identity'],'Supplementary source identity differs from primary')
    candidates=collection['candidates'];require(bool(candidates),'No complete same-identity validation candidates')
    fingerprint=lambda rows:{(str(Path(c['path']).resolve()),c['metadata_sha256']) for c in rows}
    require(fingerprint(candidates)==fingerprint(main['all_complete_validation_candidates']),'Complete candidate population changed after primary selection')
    order={str(directory):i for i,directory in enumerate(dirs)};scores=[]
    for candidate in candidates:
        meta=read(Path(candidate['path'])/'metadata.json');metrics=meta['validation_sources'];fresh=metrics['fresh_teacher']
        scores.append({'path':candidate['path'],'run_order':order[str(Path(candidate['path']).parent)],
                       'fresh_validation_flow_loss':fresh['flow_loss'],'fresh_validation_episodes':fresh['episodes'],
                       'original_validation_flow_loss':metrics['original']['flow_loss'],
                       'combined_validation_flow_loss':candidate['validation_flow_loss'],'metadata_sha256':candidate['metadata_sha256'],
                       'local_step':candidate['local_step'],'cumulative_round_updates':candidate['cumulative_round_updates']})
    scores.sort(key=lambda r:(r['fresh_validation_flow_loss'],r['run_order'],r['path']))
    selected=next(c for c in candidates if c['path']==scores[0]['path']);lineage=selected['lineage']
    require(lineage['genesis']==main['selected_lineage']['genesis'],'Supplementary genesis path/metadata/payload differs from primary')
    known={Path(value['path']).resolve():value for value in [main['model_artifacts']['pi05_round01'],main['model_artifacts']['pi05_round02'],*main['ancestor_model_artifacts']]}
    cache={};ancestors=[];selected_path=Path(selected['path']).resolve()
    for node in lineage['ancestor_chain']:
        path=Path(node['path']).resolve();metadata_path=path/'metadata.json'
        require(collection['metadata_hashes'].get(str(metadata_path))==sha(metadata_path)==node['metadata_sha256'],'Supplementary ancestor metadata identity mismatch')
        actual=artifact(path/'model.safetensors',cache)
        if path!=selected_path:
            require(Path(actual['path']) in known,'Supplementary parent/ancestor is outside the primary verified lineage')
            expected=known[Path(actual['path'])]
            require(node['weights_sha256'] is not None and actual['sha256']==node['weights_sha256']==expected['sha256'],'Supplementary inherited ancestor weights changed')
            ancestors.append(actual)
        else:
            require(node['weights_sha256'] is None or node['weights_sha256']==actual['sha256'],'Supplementary selected payload differs from lineage declaration')
            payload=actual
    require(selected['local_step']!=0 or payload['sha256']==lineage['immediate_parent']['weights_sha256'],'Supplementary local0 payload differs from inherited parent')
    primary_payload=artifact(main['model_artifacts']['pi05_round02']['path'],cache)
    require(primary_payload['sha256']==main['model_artifacts']['pi05_round02']['sha256'],'Primary selected model payload changed')
    duplicate=payload['sha256']==main['model_artifacts']['pi05_round02']['sha256']
    frozen={**primary['frozen_files'],**collection['metadata_hashes'],str(predeclaration):sha(predeclaration),str(roster_path):sha(roster_path),
            str(primary_path):sha(primary_path),str(Path(__file__).resolve()):sha(__file__)}
    receipt={'schema':'pi05_g1_round02_supplementary_fresh_validation_selection_v1','status':'duplicate_primary_payload' if duplicate else 'unique_supplementary_payload',
       'policy_id':POLICY,'supplementary':True,'primary_selection_unchanged':True,'selection_criterion':'minimum complete fresh_teacher validation flow_loss',
       'tie_rule':'explicit run order, then checkpoint path','selected':selected,'selected_scores':scores[0],'candidate_scores':scores,
       'selected_round_optimizer_updates':selected['cumulative_round_updates'],'data_identity':collection['identity'],
       'predeclaration_path':str(predeclaration),'predeclaration_sha256':sha(predeclaration),'primary_selection_sha256':sha(selection_path),
       'model_artifact':payload,'primary_selected_payload_sha256':main['model_artifacts']['pi05_round02']['sha256'],
       'verified_shared_ancestor_artifacts':ancestors,'plan_created':not duplicate,
       'duplicate_fallback':'official_pi05_base only if duplicate; root admits any complete20 using timing/budget only; never both optional policies',
       'source_scope':'Validation metadata only. No final raw outcomes or test observations read. Raw result existence is checked solely to reject late selection.',
       'ancestor_phase_contract':'Every inherited ancestor is already among the primary verified model/ancestor artifacts. Primary phase-start/end reverify those shared ancestors; supplementary phase verifies its selected payload.',
       'large_payload_hash_contract':'CPU selection snapshots each distinct payload inode once; selected payload phase-start/end checks precede/follow execution. Large weights are not per-case frozen files.',
       'frozen_files':frozen}
    if duplicate:
        write(receipt_path,receipt);return receipt,None
    jobs=[];configs=[];seen=set()
    for job in primary['jobs']:
        cp=Path(job['config_path']);config=read(cp)
        if config.get('policy_id')!='pi05_round02':continue
        require(sha(cp)==job['config_sha256'],'Primary config SHA differs')
        cid=config['case']['case_id'];require(cid in CASES and cid not in seen,'Primary Pi final20 case coverage differs');seen.add(cid)
        new=copy.deepcopy(config);new.update(policy_id=POLICY,output_dir=str(root/POLICY/cid),video_output_dir=str(root/'video_exports'/POLICY/cid))
        command=new['policy_command'];require(command.count('--checkpoint')==1,'Primary Pi command checkpoint is ambiguous')
        index=command.index('--checkpoint')+1
        require(Path(command[index]).resolve()==Path(main['selected']['path']).resolve(),'Primary Pi config does not use the selected checkpoint')
        command[index]=str(selected_path)
        new['policy_artifacts'].update(checkpoint=str(selected_path),weights_sha256=payload['sha256'],supplementary=True,
            selection_criterion='predeclared_complete_fresh_validation_minimum',predeclaration_sha256=sha(predeclaration))
        output_config=root/'configs'/f'{POLICY}_{cid}.json'
        if output_config.exists() or Path(new['output_dir']).exists() or Path(new['video_output_dir']).exists():raise FileExistsError('Supplementary outputs/configs already exist')
        configs.append((output_config,new));jobs.append({'name':POLICY+'_'+cid,'config_path':str(output_config),'require_task_success':False})
    require(seen==CASES and len(jobs)==20,'Exactly the same complete final20 Pi cases required')
    write(receipt_path,receipt);frozen[str(receipt_path)]=sha(receipt_path)
    for (path,config),job in zip(configs,jobs):write(path,config);job['config_sha256']=sha(path);frozen[str(path)]=sha(path)
    plan={'schema':'pi05_g1_round02_frozen_benchmark_plan_v1','phase':'supplementary_fresh','policies':[POLICY],'jobs':jobs,
       'frozen_files':frozen,'selection_receipt':str(receipt_path),'model_artifacts':{POLICY:payload},'gr00t_artifacts':[],
       'ancestor_model_artifacts':ancestors,'supplementary':True,'primary_plan_unchanged':str(primary_path),
       'complete_case_requirement':20,'budget_admission':'Root timing-only predeclaration admission required; this factory never starts batch/runner.',
       'interpretation':'Supplementary fresh-validation choice; disclose small fresh validation population and original-data tradeoff. Primary mixed-choice report remains independent.'}
    write(plan_path,plan);return receipt,plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('evaluation-root','training-dir','predeclaration'):p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args();receipt,plan=make_plan(a.evaluation_root,a.training_dir,a.predeclaration)
    print(json.dumps({'status':receipt['status'],'plan_created':plan is not None,'selected':receipt['selected']['path']}))


if __name__=='__main__':main()
