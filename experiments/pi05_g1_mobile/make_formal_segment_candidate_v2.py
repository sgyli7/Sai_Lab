"""Version2 stdlib preparation of ONE unapproved three-job segment; minimal snapshots.

No queue, model imports, model hashes, execution, or automatic approval. Real
batch/audit retain final authority. Prior evidence SHA/inventories are checked.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re

ROOT=Path('/home/ethan/Projects/Sai_Lab')
MOBILE=ROOT/'.scratch/pi05_g1/round03_mobile'
TEMPLATE=MOBILE/'formal_three_job_continuation_pose_speed_PENDING.json'
TEMPLATE_SHA256='26fb266c0c2a9d2238bf700f1599a5118e38d25e34004944e213cc14d70da7de'
FACTORY=ROOT/'experiments/pi05_g1_mobile/make_benchmark_plan_pose_speed.py'
SEGMENT_SCHEMA='pi05_g1_mobile_batch_segment_result_v1'
POLICIES=['gr00t_n16','pi05_mobile','official_pi05_base']
INITIAL_KEYS=('source_joint_names','joint_positions','joint_velocities','root_link_pose_w','root_link_velocity_w','root_com_velocity_w','acceptance_truth_only')


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def check(condition,message):
    if not condition:raise ValueError(message)


def absolute(path):
    path=Path(path)
    check(path.is_absolute(),'Absolute path required: '+str(path))
    check(not path.is_symlink(),'Symlink evidence refused: '+str(path))
    return path.resolve()


def verify_files(files,model_paths=()):
    check(isinstance(files,dict) and bool(files),'Nonempty frozen manifest required')
    for name,digest in files.items():
        path=absolute(name)
        check(str(path) not in model_paths,'Model payload must use phase identity, not frozen small manifest')
        check(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest),'Invalid SHA')
        check(path.is_file() and sha(path)==digest,'Missing/changed frozen evidence: '+str(path))


def model_entries(artifacts):
    return [entry for entries in artifacts.values() for entry in (entries if isinstance(entries,list) else [entries])]


def load_inputs(plan_path,phase_path):
    plan_path=absolute(plan_path);phase_path=absolute(phase_path);plan=read(plan_path);phase=read(phase_path)
    check(plan.get('schema')=='pi05_g1_mobile_benchmark_plan_v1','Wrong factory plan schema')
    check(plan.get('primary_policies')==POLICIES,'Three primary policy identities/order required')
    ids=plan.get('case_ids',[]);cases=plan.get('cases',[]);jobs=plan.get('jobs',[])
    check(len(ids)==20 and len(set(ids))==20 and len(jobs)==60,'Exactly20 cases/60 jobs required')
    check([c['case_id'] for c in cases]==ids and len({c['cluster_id'] for c in cases})==5 and {c['seed'] for c in cases}=={1040,1041,1042,1043},'Frozen5positions/fourseeds cases required')
    check(phase.get('schema')=='pi05_g1_mobile_execution_phase_sha_v1' and phase.get('status')=='verified' and phase.get('gpu_started') is False,'Verified CPU phase-start receipt required')
    digest=sha(plan_path)
    check(phase.get('plan_sha256')==digest and absolute(phase['plan'])==plan_path,'Stale phase/factory plan')
    frozen=plan.get('frozen_files',{});check(phase.get('frozen_files')==frozen,'Phase frozen map differs')
    selection_path=absolute(plan['selection_receipt']);check(frozen.get(str(selection_path))==sha(selection_path),'Selection SHA missing/mismatch')
    selection=read(selection_path)
    check(frozen.get(str(FACTORY.resolve()))==sha(FACTORY),'Actual factory source missing/changed in plan closure')
    check(isinstance(selection.get('frozen_files'),dict) and bool(selection['frozen_files']),'Selection frozen provenance missing')
    for name,expected_sha in selection['frozen_files'].items():check(frozen.get(name)==expected_sha,'Selection frozen inputs omitted/changed in plan')
    check(selection.get('factory_entry')=='experiments.pi05_g1_mobile.make_benchmark_plan_pose_speed','Actual pose-speed factory selection required')
    check(selection.get('model_artifacts')==plan.get('model_artifacts'),'Selection/model identities differ')
    measurement=plan.get('acceptance_measurement');check(isinstance(measurement,dict) and selection.get('acceptance_measurement')==measurement,'Selection/plan measurement differs')
    check(measurement.get('module')=='experiments.pi05_g1_mobile.acceptance_pose_speed' and plan.get('batch_entry')=='experiments.pi05_g1_mobile.batch_pose_speed','Pose-speed entries required')
    adoption=absolute(measurement['root_formal_adoption']);check(frozen.get(str(adoption))==measurement.get('root_formal_adoption_sha256')==sha(adoption),'Formal adoption missing from frozen closure')
    proof=read(adoption);check(proof.get('root_reviewed') is True and proof.get('formal_adoption') is True and type(proof.get('formal_trials_before_adoption')) is int and proof['formal_trials_before_adoption']==0,'Formal adoption identity invalid')
    for name,value in proof.get('pins',{}).items():check(frozen.get(name)==value,'Adoption pin omitted from plan')
    entries=model_entries(plan['model_artifacts']);observed=phase.get('model_artifacts_observed',{})
    check(set(observed)=={e['path'] for e in entries},'Phase model roster incomplete')
    for entry in entries:
        actual=observed[entry['path']];check(actual==entry,'Phase model artifact identity differs')
        st=absolute(entry['path']).stat()
        check(all(actual[k]==v for k,v in [('bytes',st.st_size),('device',st.st_dev),('inode',st.st_ino),('mtime_ns',st.st_mtime_ns)]),'Phase model stat identity stale; rerun CPU phase explicitly')
    verify_files(frozen,{e['path'] for e in entries})
    configs=[];outputs=[]
    for index,job in enumerate(jobs):
        ci=index//3;order=POLICIES[ci%3:]+POLICIES[:ci%3]
        check(job['case_id']==ids[ci] and job['policy']==order[index%3],'Factory rotation/order altered')
        cp=absolute(job['config_path']);check(job['config_sha256']==frozen.get(str(cp))==sha(cp),'Config SHA mismatch')
        config=read(cp);rp=absolute(job['result_path']);out=absolute(config['output_dir'])
        check(rp==out/'result.json' and config['case']['case_id']==job['case_id'] and config['policy_id']==job['policy'],'Job/config/result identity mismatch')
        check(config.get('mode')=='policy' and config.get('evaluate_acceptance') is True and config.get('acceptance_measurement')==measurement,'Policy/measurement mismatch')
        check(config.get('ticks')==3000 and config.get('episode_seconds')==60 and config.get('native_action_horizon')==50 and config.get('replan_interval_ticks')==50,'Frozen60s/50cadence differs')
        for name,value in proof['pins'].items():check(config['frozen_files'].get(name)==value,'Config omits adoption proof pin')
        check(config['frozen_files'].get(str(adoption))==sha(adoption),'Config omits formal adoption')
        outputs.append(str(out));configs.append((cp,config))
    check(len(set(outputs))==60,'Raw case directories not unique')
    return plan,configs,selection,digest


def inventory(folder):
    result={}
    for path in sorted(Path(folder).rglob('*')):
        check(not path.is_symlink(),'Symlink case evidence refused')
        if path.is_file():result[str(path.resolve())]=sha(path)
    return result


def verify_prefix(paths,plan,configs,plan_sha,segment_index):
    names=[str(absolute(p)) for p in paths];check(len(names)==len(set(names))==segment_index,'Explicit exactlyone distinct prior receipt per segment required')
    bindings={};records={}
    for number,name in enumerate(names):
        doc=read(name);start=3*number;stop=start+3
        check(doc.get('schema')==SEGMENT_SCHEMA and doc.get('status')=='completed' and doc.get('plan_sha256')==plan_sha,'Wrong/noncompleted/stale prior segment')
        for key,value in [('segment_start_index',start),('segment_stop_index_exclusive',stop),('expected_segment_jobs',3),('expected_total_jobs',60),('previous_completed_jobs',start),('completed_total_jobs',stop)]:check(type(doc.get(key)) is int and doc[key]==value,'Prior segment count/prefix mismatch: '+key)
        check(doc.get('resume_receipts_sha256')==bindings,'Prior receipt provenance chain differs')
        check(doc.get('benchmark_complete') is False and len(doc.get('jobs',[]))==3,'Prior receipt not exact3job nonfinal segment')
        for i,record in zip(range(start,stop),doc['jobs']):
            job=plan['jobs'][i];cp,config=configs[i];rp=absolute(job['result_path']);folder=rp.parent
            check(record.get('index')==i and record.get('policy')==job['policy'] and record.get('case_id')==job['case_id'] and record.get('status')=='completed','Prior record order/identity differs')
            check(record.get('result_path')==str(rp) and record.get('config_sha256')==sha(cp) and record.get('result_sha256')==sha(rp),'Prior config/result binding differs')
            result=read(rp)
            check(result.get('schema')=='pi05_g1_mobile_case_result_v1' and result.get('status')=='completed' and result.get('source_interface_passed') is True and not result.get('cleanup_errors'),'Prior producer incomplete')
            check(result.get('policy')==job['policy'] and result.get('case_id')==job['case_id'] and read(folder/'config.json')==config,'Prior executed identity differs')
            check(type(result.get('success')) is bool and result['acceptance']['success'] is result['success'] and record.get('success') is result['success'],'Prior task outcome missing/inconsistent')
            check(record.get('source_interface_passed') is True and sha(folder/'trajectory.jsonl')==result.get('trace_sha256'),'Prior trace/interface binding differs')
            check(inventory(folder)==record.get('case_files_sha256'),'Prior evidence inventory/content changed')
            lp=cp.with_suffix('.launch.json');log=cp.with_suffix('.launch.log');launch=read(lp)
            check(record.get('launch_path')==str(lp) and record.get('launch_log')==str(log) and record.get('launch_sha256')==sha(lp) and record.get('launch_log_sha256')==sha(log),'Prior launch hash/path differs')
            check(launch.get('schema')=='pi05_g1_mobile_case_launch_v1' and launch.get('exit_code')==0 and launch.get('config_sha256')==sha(cp),'Prior launch incomplete')
            records[i]=record
        bindings[name]=sha(name)
    check(set(records)==set(range(3*segment_index)),'Non-contiguous completed prefix')
    return bindings


# Exact benchmark_v2 INITIAL_KEYS + JSON serialization; this is only an early
# identity check. Independent v3 still rechecks trace counters/actions/physics.
PAIR_GUARD="""initial_hashes=[]
for planned in expected:
 rp=Path(planned['result_path']);r=batch.read(rp)
 with (rp.parent/'trajectory.jsonl').open() as stream:first=json.loads(next(line for line in stream if line.strip()))
 initial={k:first[k] for k in ('source_joint_names','joint_positions','joint_velocities','root_link_pose_w','root_link_velocity_w','root_com_velocity_w','acceptance_truth_only')}
 actual=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
 assert r.get('initial_physical_state_schema')=='pi05_g1_mobile_initial_kinematic_v1' and r.get('initial_physical_state_sha256')==actual
 initial_hashes.append(actual)
assert len(initial_hashes)==3 and len(set(initial_hashes))==1
"""


def substitute(value,mapping):
    if isinstance(value,str):
        for key,replacement in mapping.items():value=value.replace(key,replacement)
        return value
    if isinstance(value,list):return [substitute(x,mapping) for x in value]
    if isinstance(value,dict):return {k:substitute(v,mapping) for k,v in value.items()}
    return value


def generate(plan_path,phase_path,segment_index,prior_receipts,output):
    check(type(segment_index) is int and 0<=segment_index<20,'Segment index must be0..19')
    output=absolute(output);check(not output.exists(),'Fresh output directory required')
    check(sha(TEMPLATE)==TEMPLATE_SHA256,'Pinned pending template changed')
    template=read(TEMPLATE);plan,configs,selection,plan_sha=load_inputs(plan_path,phase_path)
    bindings=verify_prefix(prior_receipts,plan,configs,plan_sha,segment_index)
    check(output.is_relative_to(MOBILE.resolve()),'Driver result receipts must live below mobile root')
    start=segment_index*3;end=start+3;selected=plan['jobs'][start:end]
    for job,(cp,config) in zip(selected,configs[start:end]):
        check(not Path(config['output_dir']).exists() and not cp.with_suffix('.launch.json').exists() and not cp.with_suffix('.launch.log').exists(),'Upcoming case has existing partial/unclaimed artifacts')
    mapping={'PENDING_ABSOLUTE_FROZEN_BENCHMARK_PLAN_JSON':str(absolute(plan_path)),
        'PENDING_ABSOLUTE_NEW_SEGMENT_000_JSON':str(output/'segment.json'),
        'PENDING_ABSOLUTE_GLOBAL_BATCH_RESULT_JSON':str(absolute(plan_path).parent/'batch_result.json'),
        'PENDING_ABSOLUTE_NEW_RESUME_GUARD_JSON':str(output/'resume_guard.json'),
        'PENDING_ABSOLUTE_NEW_COMPLETION_GUARD_JSON':str(output/'completion_guard.json'),
        'PENDING_ABSOLUTE_NEW_NATIVE_V2_AUDIT_JSON':str(output/'independent_audit.json'),
        'PENDING_ABSOLUTE_NEW_SEGMENT_AUDIT_MEASUREMENT_IDENTITY_JSON':str(output/'audit_measurement_identity.json'),
        'PENDING_ABSOLUTE_NEW_SEGMENT_POINTER_JSON':str(output/'pointer.json')}
    for i,j in enumerate(selected):mapping['PENDING_ABSOLUTE_POLICY_CASE_DIR_'+str(i)]=str(Path(j['result_path']).parent)
    driver=substitute(copy.deepcopy(template['driver_plan_template']),mapping)
    driver['root_reviewed']=False;driver['review_status']='candidate only; Root must review actual finite segment and remaining allocated budget'
    stages=driver['stages'];gpu=stages[1]
    for i in [0,2]:
        argv=stages[i]['argv'];c=argv.index('-c');mode=argv[c+2];argv[c+4]=str(start);argv[c+5]=json.dumps([str(absolute(p)) for p in prior_receipts])
        if mode=='complete':
            code=argv[c+1];marker='output=Path(out);'
            check(marker in code,'Template completion code changed');code=code.replace(marker,PAIR_GUARD+marker,1)
            code=code.replace("'mode':mode,","'mode':mode,'actual_initial_state_match':True,'initial_physical_state_sha256':initial_hashes[0],",1)
            argv[c+1]=code
        for predicate in stages[i]['result']['predicates']:
            if predicate['path']==['verified_completed_prefix']:predicate['value']=start if i==0 else end
    stages[2]['result']['predicates'].append({'path':['actual_initial_state_match'],'op':'eq','value':True})
    a=gpu['argv'];a[a.index('--start-index')+1]=str(start)
    for path in prior_receipts:a+=['--resume-receipt',str(absolute(path))]
    gpu['id']='formal_segment_'+str(segment_index).zfill(3)+'_three_jobs'
    for predicate in gpu['result']['predicates']:
        key=predicate['path'][0]
        if key in {'segment_start_index','previous_completed_jobs'}:predicate['value']=start
        if key in {'segment_stop_index_exclusive','completed_total_jobs'}:predicate['value']=end
        if key=='benchmark_complete':predicate['value']=end==60
    pointer=stages[4]['argv'];ci=pointer.index('-c');code=pointer[ci+1]
    code=code.replace("assert guard['passed'] is True", "assert guard.get('actual_initial_state_match') is True\nassert guard['passed'] is True",1)
    # Prior producer prefix is independently reverified at preparation and both
    # runtime guards; no invented previous CPU pointer is claimed by this API.
    begin="if start:\n";finish="finished=end==60\n";left=code.index(begin);right=code.index(finish,left)
    code=code[:left]+code[right:];pointer[ci+1]=code
    for predicate in stages[4]['result']['predicates']:
        if predicate['path']==['next_start_index']:predicate['value']=end
        if predicate['path']==['producer60_completed']:predicate['value']=end==60
    frozen={}
    # v2: snapshot only direct execution inputs, not historical image/assets.
    # load_inputs already verifies the entire actual factory frozen manifest;
    # batch_pose_speed.load_plan repeats that verification before runtime jobs.
    # Keeping the complete plan + selection + phase JSON SHA preserves the
    # transitive roster without copying its payloads20times into driver journals.
    direct_paths=[absolute(plan_path),absolute(phase_path),absolute(plan['selection_receipt']),
        absolute(TEMPLATE),Path(__file__).resolve(),FACTORY.resolve(),
        absolute(plan['acceptance_measurement']['root_formal_adoption'])]
    direct_paths += [configs[i][0] for i in range(start,end)]
    direct_paths += [ROOT/'experiments/pi05_g1_mobile'/name for name in [
        '__init__.py','round03_driver_allocated.py','round03_driver.py','budget_allocated.py','budget.py',
        'batch_pose_speed.py','acceptance.py','acceptance_body_ids.py','acceptance_pose_speed.py',
        'sim_launcher_pose_speed.py','sim_launcher.py','simulator.py','benchmark.py','benchmark_v2.py',
        'contract.py','offline_assets.py','audit_episodes_v3.py','audit_episodes_v2.py','audit_delivery_guard_v2.py']]
    direct_paths += [ROOT/'experiments/pi05_g1/budget.py',ROOT/'experiments/vla_resources/gpu_queue.py']
    for path in direct_paths:
        path=path.resolve()
        check(path.is_file() and not path.is_symlink() and path.stat().st_size<=16*1024**2,'Direct driver input must be a small existing regular file')
        frozen[str(path)]=sha(path)
    frozen.update(bindings)
    check(all(Path(path).stat().st_size<=16*1024**2 for path in frozen),'Prior receipt exceeds driver small-file snapshot limit')
    driver['frozen_files']=frozen
    for stage in stages:
        if '-c' in stage['argv']:compile(stage['argv'][stage['argv'].index('-c')+1],'<generated CPU guard>','exec')
    check('PENDING_' not in json.dumps(driver),'Unfilled template placeholder')
    check(len(stages)==5 and gpu['max_seconds']==3000 and gpu['argv'][gpu['argv'].index('-m')+1]=='experiments.pi05_g1_mobile.batch_pose_speed','Five-stage/batch semantics changed')
    note={'schema':'pi05_g1_mobile_formal_segment_candidate_preparation_v2','status':'pending_root_review','root_reviewed':False,'segment_index':segment_index,'start_index':start,'stop_index_exclusive':end,'plan_sha256':plan_sha,'phase_receipt_sha256':sha(phase_path),'selection_receipt_sha256':sha(plan['selection_receipt']),'prior_receipts_sha256':bindings,'jobs':selected,'snapshot_scope':'Direct execution sources + plan/phase/selection/adoption + executing3configs + priorproducerreceipts only. Fulltransitive sources/assets/PNGs remain freshly hash-verified by load_inputs and each batch job, never duplicated in driver snapshots.', 'driver_snapshot_file_count':len(frozen),'driver_snapshot_source_bytes':sum(Path(path).stat().st_size for path in frozen),'actual_model_payloads_rehashed':False,'model_stat_identity_only_after_verified_phase':True,'early_actual_init_guard':'Stage3 reproduces exact benchmark_v2 INITIAL_KEYS/JSON hash for allthree trace firstrows; requires each result hash and allthree equality before independent pose-context v3 audit. No new success predicate.','prefix_guard_scope':'All previous completed producer receipts/inventory refreshed; earlier CPU audit pointers are not inputs/claimed. Each Root-approved segment must finish its own independent audit/pointer before continuing.','allocated_driver':'experiments.pi05_g1_mobile.round03_driver_allocated','allocated_supervisor':'experiments.pi05_g1_mobile.budget_allocated','GPU_started':False,'queue_requested':False,'approved':False,'required_root_checks':['Review actual selected training/model identity and fullphase receipt, even if selectedstep0','Review remaining total/stage allocated budget and exclusiveowned GPU quiescence before dryrun/request','Prior segments independent CPUaudit/pointer must have passed; producer-prefix verification alone is not physics approval','Inspect generated five stages/frozen roster/actualcase and approve only this finite segment; no60trial auto-loop']}
    output.mkdir(parents=True)
    for name,value in [('candidate_plan.json',driver),('preparation_receipt.json',note)]:
        with (output/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    return note


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--phase-start',type=Path,required=True)
    parser.add_argument('--segment-index',type=int,required=True);parser.add_argument('--resume-receipt',type=Path,action='append',default=[]);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();receipt=generate(args.plan,args.phase_start,args.segment_index,args.resume_receipt,args.output)
    print(json.dumps({'status':receipt['status'],'segment_index':receipt['segment_index'],'root_reviewed':False}))

if __name__=='__main__':main()
