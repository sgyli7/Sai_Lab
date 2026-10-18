"""CPU-only deterministic final plans; never inspect final outcomes or launch a model."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
SCRATCH=ROOT/'.scratch/pi05_g1'
POLICIES=('gr00t_n17','pi05_round01','pi05_round02')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+v)))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def checked(path,expected):
    path=Path(path).resolve()
    if sha(path)!=expected:raise ValueError('Immutable input SHA mismatch: '+str(path))
    return path


def select_checkpoint(training_dir,data_config,norm,loss_weights):
    """Select the global validation minimum across an explicit recovery roster."""
    from .checkpoint_lineage import collect_candidates
    training_dir=Path(training_dir).resolve();roster_path=training_dir/'candidate_training_dirs.json'
    if roster_path.is_file():
        roster=read(roster_path)
        if (roster.get('schema')!='pi05_g1_round02_candidate_training_dirs_v1'
                or Path(roster.get('current_training_dir','')).resolve()!=training_dir):
            raise ValueError('Candidate training directory roster has the wrong schema/current run')
        declared=roster.get('training_dirs',[])
        if not declared or any(not isinstance(p,str) or not Path(p).is_absolute() for p in declared):raise ValueError('Absolute candidate training directories required')
        training_dirs=[Path(p).resolve() for p in declared]
        if len(training_dirs)!=len(set(training_dirs)) or training_dir not in training_dirs:raise ValueError('Unique roster must include the current run')
    else:
        training_dirs=[training_dir]
    collected=collect_candidates(training_dirs,data_config,norm,loss_weights,verify_weights=False)
    candidates=collected['candidates'];pointer_path=training_dir/'best_checkpoint.json';pointer=read(pointer_path)
    pointer_value=Path(pointer['path']);selected=pointer_value.resolve() if pointer_value.is_absolute() else (training_dir/pointer_value).resolve()
    if selected.parent not in training_dirs:raise ValueError('Best pointer must refer to a direct checkpoint child of the declared roster')
    if not candidates:raise ValueError('No complete mixed-validation checkpoint')
    minimum=min(c['validation_flow_loss'] for c in candidates)
    candidate=next((c for c in candidates if Path(c['path']).resolve()==selected),None)
    if candidate is None or candidate['validation_flow_loss']!=minimum or pointer.get('validation_flow_loss')!=minimum:
        raise ValueError('Best pointer is not the global minimum of all complete roster candidates')
    if not roster_path.exists() and candidate.get('inherited_round_updates',0)>0:
        raise ValueError('Recovered training requires an explicit candidate_training_dirs roster')
    choice={**candidate,'candidate_training_dirs':[str(path) for path in training_dirs],
            'candidate_roster_path':str(roster_path) if roster_path.is_file() else None,
            'candidate_roster_sha256':sha(roster_path) if roster_path.is_file() else None,
            'genesis_checkpoint':str(Path(collected['genesis_checkpoint']).resolve())}
    metadata=dict(collected['metadata_hashes'])
    if roster_path.is_file():metadata[str(roster_path)]=sha(roster_path)
    return choice,candidates,collected['identity'],metadata,pointer_path


def artifact(path,cache):
    path=Path(path).resolve();st=path.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    if key not in cache:cache[key]=sha(path)
    return {'path':str(path),'sha256':cache[key],'bytes':st.st_size,'device':st.st_dev,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns}


def make_plan(training_dir,protocol_path,data_config,loss_weights,output,*,round01_checkpoint=None,official_base=None,gr00t_manifest=None,pi_python=None,gr00t_python=None):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('New evaluation root must not exist, including partial directories')
    protocol_path=Path(protocol_path).resolve();protocol=read(protocol_path);parent=protocol_path.parent
    if protocol.get('schema')!='pi05_g1_round02_frozen_final_protocol_v1' or protocol['primary_models']!=list(POLICIES):raise ValueError('Unknown final protocol')
    if protocol['replan_interval_ticks']!=40 or protocol['native_action_horizons']!={'pi05':50,'gr00t_n17':40}:raise ValueError('Final cadence/horizons changed')
    norm=checked(protocol['pi_common_norm_file'],protocol['pi_common_norm_sha256'])
    cases_path=checked(protocol['final_cases_file'],protocol['final_cases_sha256']);cases=read(cases_path)['cases']
    if len(cases)!=20 or len({c['case_id'] for c in cases})!=20 or {c['seed'] for c in cases}!={40,41,42,43}:raise ValueError('Expected frozen final20 cases')
    checked(data_config,protocol['data_config_sha256']);limits=checked(protocol['limits_file'],protocol['limits_file_sha256'])
    rule=checked(parent/'comparator_adapter_rule.json',protocol['pi_compare_rule_sha256']);rule_doc=read(rule)
    if rule_doc['pi05_common_norm_sha256']!=sha(norm):raise ValueError('Comparator normalization override changed')
    frozen={str(protocol_path):sha(protocol_path),str(Path(data_config).resolve()):sha(data_config),str(Path(loss_weights).resolve()):sha(loss_weights),str(norm):sha(norm),str(cases_path):sha(cases_path),str(limits):sha(limits),str(rule):sha(rule)}
    gates={}
    for name in ('bounded','unbounded'):
        gate=checked(protocol[name+'_replay_result'],protocol[name+'_replay_sha256']);g=read(gate)
        if (g.get('status')!='completed' or g.get('success') is not True or g.get('replay_passed') is not True
            or g.get('source_interface_passed') is not True or g.get('frozen_interface_sha256')!=protocol['common_interface_sha256']
            or g.get('acceptance_sha256')!=protocol['acceptance_sha256']
            or g.get('acceptance_version')!=protocol['acceptance_version']
            or g.get('initial_physical_state_schema')!=protocol['initial_state_schema']):raise ValueError('Frozen replay gate is incomplete/incompatible')
        gates[name]=gate;frozen[str(gate)]=sha(gate)
    template_path=gates['bounded'].parent/'config.json';template=read(template_path);frozen[str(template_path)]=sha(template_path)
    if template['observer_camera']!=protocol['camera_observer']:raise ValueError('Frozen observer configuration mismatch')
    choice,candidates,identity,metadata,pointer=select_checkpoint(training_dir,data_config,norm,loss_weights)
    frozen.update(metadata);frozen[str(pointer)]=sha(pointer)
    for source in identity['sources']:frozen[str(Path(source['manifest']).resolve())]=source['manifest_sha256']
    old=Path(round01_checkpoint or choice['genesis_checkpoint']).resolve();base=Path(official_base or SCRATCH/'pi05_base_pytorch').resolve()
    if read(old/'metadata.json').get('step')!=1200:raise ValueError('Round01 comparator must be frozen step1200')
    if Path(choice['genesis_checkpoint']).resolve()!=old:raise ValueError('Selected recursive lineage genesis differs from the original step1200 comparator')
    discovery=Path(gr00t_manifest or parent/'gr00t_discovery.json').resolve();d=read(discovery);frozen[str(discovery)]=sha(discovery)
    pi_python=str(Path(pi_python or SCRATCH/'venv/bin/python').absolute());gr00t_python=str(Path(gr00t_python or '/home/ethan/models/unitree_g1/envs/policy_onnx_cu13/bin/python').absolute())
    for directory in (ROOT/'experiments/pi05_g1',ROOT/'experiments/pi05_g1_round02'):
        for path in sorted(directory.glob('*.py')):frozen[str(path)]=sha(path)
    if frozen[str(ROOT/'experiments/pi05_g1_round02/simulator.py')]!=protocol['simulator_sha256']:raise ValueError('Simulator source changed since protocol freeze')
    cache={};artifacts={policy:artifact(checkpoint/'model.safetensors',cache) for policy,checkpoint in [('pi05_round01',old),('pi05_round02',Path(choice['path'])),('official_pi05_base',base)]}
    if artifacts['pi05_round01']['sha256']!=choice['lineage']['genesis']['weights_sha256']:raise ValueError('Genesis comparator weights changed')
    if choice['local_step']==0 and artifacts['pi05_round02']['sha256']!=choice['lineage']['immediate_parent']['weights_sha256']:
        raise ValueError('Local step0 payload differs from inherited parent weights')
    ancestor_artifacts=[]
    for ancestor in choice['lineage']['ancestor_chain']:
        entry=artifact(Path(ancestor['path'])/'model.safetensors',cache)
        expected=ancestor['weights_sha256']
        if expected is None and Path(ancestor['path']).resolve()!=Path(choice['path']).resolve():raise ValueError('Inherited ancestor weight SHA is absent: '+ancestor['path'])
        if expected is not None and entry['sha256']!=expected:raise ValueError('Selected ancestor weights changed: '+ancestor['path'])
        ancestor_artifacts.append(entry)
    groot_artifacts=[]
    for item in d['files']:
        entry=artifact(item['path'],cache)
        if entry['sha256']!=item['sha256'] or entry['bytes']!=item['bytes']:raise ValueError('Pinned GR00T model/source changed')
        groot_artifacts.append(entry)
    for checkpoint in (old,base):
        for path in checkpoint.glob('*.json'):frozen[str(path)]=sha(path)
    selection={'schema':'pi05_g1_round02_final_selection_v1','selected':choice,'selected_round_optimizer_updates':choice['cumulative_round_updates'],'selected_local_optimizer_updates':choice['local_step'],
       'selected_inherited_round_optimizer_updates':choice['inherited_round_updates'],
       'local_step_zero_is_not_necessarily_zero_round_updates':True,'selected_lineage':choice['lineage'],'ancestor_model_artifacts':ancestor_artifacts,'all_complete_validation_candidates':candidates,
       'data_identity':identity,'best_pointer_path':str(pointer),'best_pointer_sha256':sha(pointer),
       'model_artifacts':artifacts,'gr00t_artifacts':groot_artifacts,'comparator_adapter_rule':rule_doc,
       'selection_rule':'Global minimum complete mixed original21+fresh validation across the explicit recovery roster. Local step0 eligible with inherited updates disclosed. No final outcomes read.',
       'weight_verification_boundary':'Factory hashes each distinct payload inode once. Reverify payload SHA once at phase-start and phase-end with verify_plan_phase, never once per case. Paths/inodes recorded as artifact identities, not proof of immutability.',
       'frozen_files':frozen}
    output.mkdir(parents=True);write(output/'selection_receipt.json',selection);frozen[str(output/'selection_receipt.json')]=sha(output/'selection_receipt.json')
    plans={}
    for phase,policies in [('primary',POLICIES),('optional_base',('official_pi05_base',))]:
        jobs=[]
        for index,case in enumerate(cases):
            order=policies[index%len(policies):]+policies[:index%len(policies)]
            for policy in order:
                config=copy.deepcopy(template)
                for key in ('expert_sequence','expert_sha256','expected_interface_sha256'):config.pop(key,None)
                config.update(mode='policy',case=case,policy_id=policy,output_dir=str(output/policy/case['case_id']),
                    video_output_dir=str(output/'video_exports'/policy/case['case_id']),save_teacher_frames=False,
                    replan_interval_ticks=40,apply_action_limits=True,action_limits_file=str(limits),observer_camera=protocol['camera_observer'],
                    replay_result=str(gates['bounded']),unbounded_replay_result=str(gates['unbounded']),inference_timeout_seconds=240)
                if policy=='gr00t_n17':
                    config.update(prompt='move the apple to the plate',native_action_horizon=40,
                      policy_command=[gr00t_python,'-m','experiments.pi05_g1_round02.gr00t_service','--device','cuda','--manifest',str(discovery)],
                      policy_artifacts={'pinned_model_manifest_sha256':sha(discovery),'service_sha256':sha(ROOT/'experiments/pi05_g1_round02/gr00t_service.py')})
                else:
                    checkpoint=Path(artifacts[policy]['path']).parent
                    config.update(prompt='Pick up the apple and place it on the plate.',native_action_horizon=50,norm_stats=str(norm),
                      policy_command=['env','OPENPI_DATA_HOME='+str(SCRATCH/'venv/openpi-cache'),pi_python,'-m','experiments.pi05_g1_round02.pi05_service','--checkpoint',str(checkpoint),'--norm-stats',str(norm)],
                      policy_artifacts={'checkpoint':str(checkpoint),'weights_sha256':artifacts[policy]['sha256'],'norm_sha256':sha(norm),'service_sha256':sha(ROOT/'experiments/pi05_g1_round02/pi05_service.py'),
                        'normalization_override_disclosed':policy=='pi05_round01','comparator_adapter_rule_sha256':sha(rule)})
                config_path=output/'configs'/f"{policy}_{case['case_id']}.json";write(config_path,config)
                jobs.append({'name':policy+'_'+case['case_id'],'config_path':str(config_path),'config_sha256':sha(config_path),'require_task_success':False})
        phase_files={**frozen,**{job['config_path']:job['config_sha256'] for job in jobs}}
        plans[phase]={'schema':'pi05_g1_round02_frozen_benchmark_plan_v1','phase':phase,'policies':list(policies),'jobs':jobs,'frozen_files':phase_files,
          'selection_receipt':str(output/'selection_receipt.json'),'model_artifacts':artifacts,'gr00t_artifacts':groot_artifacts,'ancestor_model_artifacts':ancestor_artifacts,
          'order_rule':'Case-major; primary policy order rotates left by case index mod3 to balance machine thermal order.',
          'optional_rule':'Official base is a separate complete20 phase; run only if hard GPU budget permits the complete phase. Never report a partial phase as completed20.'}
        write(output/(phase+'_plan.json'),plans[phase])
    return plans


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-dir','protocol','data-config','loss-weights','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('round01-checkpoint','official-base','gr00t-manifest','pi-python','gr00t-python'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();plans=make_plan(a.training_dir,a.protocol,a.data_config,a.loss_weights,a.output,round01_checkpoint=a.round01_checkpoint,official_base=a.official_base,gr00t_manifest=a.gr00t_manifest,pi_python=a.pi_python,gr00t_python=a.gr00t_python)
    print(json.dumps({name:len(plan['jobs']) for name,plan in plans.items()}))


if __name__=='__main__':main()
