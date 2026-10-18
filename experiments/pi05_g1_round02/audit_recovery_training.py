"""Bounded CPU-only acceptance of a finished recovery run; never load tensors.

Reads metadata/JSONL and the optimizer ZIP index plus <=16MiB pickle metadata.
Does not deserialize pickle, read tensor storages, hash model/optimizer payloads,
read test observations, start a model, or run a background monitor.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import json
import math
from pathlib import Path
import pickletools
import zipfile

from .checkpoint_lineage import collect_candidates,declared_training_dirs,read,select_global_best,sha


def optimizer_archive_receipt(path):
    """Structural evidence only; full storage hashes belong to final preservation."""
    path=Path(path)
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist();pickles=[entry for entry in entries if entry.filename.endswith('/data.pkl') or entry.filename=='data.pkl']
        if len(pickles)!=1 or pickles[0].file_size>16*1024**2:raise ValueError('Missing or excessive optimizer pickle metadata')
        payload=archive.read(pickles[0]);strings=set()
        for op,arg,_ in pickletools.genops(payload):
            if op.name in ('UNICODE','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8') and isinstance(arg,str):strings.add(arg)
        required={'optimizer','master_parameters','fp32_master_adamw_v1','exp_avg','exp_avg_sq','round02_data_identity'}
        if not required.issubset(strings):raise ValueError('Final archive lacks FP32 master/Adam/state identity keys')
        storage=[entry for entry in entries if '/data/' in entry.filename]
        if not storage:raise ValueError('Final optimizer archive has no tensor storages')
        return {'path':str(path.resolve()),'bytes':path.stat().st_size,'pickle_metadata_bytes':len(payload),'tensor_storage_entries':len(storage),'tensor_storage_declared_bytes':sum(entry.file_size for entry in storage),'fp32_master_adam_schema_keys_present':True,'tensor_payload_read':False,'full_tensor_crc_or_hash_verified':False,'interpretation':'ZIP directory and pickle opcode structure only; no unpickling or optimizer tensor/dtype load'}


def audit(training_dir,data_config,norm_stats,loss_weights,*,budget=None,command_receipt=None):
    run=Path(training_dir).resolve();report={'schema':'pi05_g1_round02_recovery_training_acceptance_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_dir':str(run),'gpu_started':False,'large_payloads_hashed':False,'checks':{},'errors':[]}
    checks=report['checks']
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition:report['errors'].append(name)
    try:
        dirs=declared_training_dirs(run/'candidate_training_dirs.json',run)
        collection=collect_candidates(dirs,data_config,norm_stats,loss_weights,verify_weights=False)
        selected=select_global_best(collection,read(run/'best_checkpoint.json'),run)
        report['selected']={key:selected[key] for key in ('path','local_step','inherited_round_updates','cumulative_round_updates','validation_flow_loss')}
        report['all_complete_candidates']=[{key:c[key] for key in ('path','local_step','inherited_round_updates','cumulative_round_updates','validation_flow_loss','metadata_sha256')} for c in collection['candidates']]
        report['genesis_checkpoint']=collection['genesis_checkpoint'];check('global_best_across_declared_runs',True)
        progress=run/'progress.jsonl';events=[];malformed=0
        for line in progress.read_text().splitlines():
            try:events.append(json.loads(line))
            except json.JSONDecodeError:malformed+=1
        check('no_partial_or_malformed_jsonl',malformed==0)
        start=next(row for row in events if row.get('event')=='training_started');admission=next(row for row in events if row.get('event')=='cpu_admission_complete')
        check('cpu_admission_identity',admission['round02_data_identity']==collection['identity'])
        check('batch1_acc8_lr1e5_threeview',start['config']['batch_size']==1 and start['config']['gradient_accumulation']==8 and start['config']['learning_rate']==1e-5 and start['config']['head_only'] is False)
        check('fresh_fp32_master_optimizer',start['optimizer_master_dtype']=='float32' and start['warmstart']['optimizer_reset'] is True and start['warmstart']['source_optimizer_payload_loaded'] is False)
        baseline=read(run/'step_000000/metadata.json');lineage=baseline['checkpoint_lineage'];parent=Path(lineage['immediate_parent']['path']);source=read(parent/'metadata.json');failure=read(lineage['source_failure_receipt']['path'])
        check('recovery_parent200_inherited200_localzero',source['step']==200 and baseline['step']==0 and baseline['inherited_round_updates']==baseline['cumulative_round_updates']==200)
        check('step0_and_parent_complete_validation',baseline['validation_complete'] and source['validation_complete'])
        differences={name:baseline['validation_sources'][name]['flow_loss']-metric['flow_loss'] for name,metric in source['validation_sources'].items()}
        # Equality of model bytes/coverage is independently auditable. BF16 GPU
        # validation numbers need not be bitwise identical across fresh runtime
        # state. Preserve differences; do not create a post-hoc tolerance gate.
        check('step0_parent_and_reload_losses_finite',all(math.isfinite(baseline['validation_sources'][name]['flow_loss']) and math.isfinite(source['validation_sources'][name]['flow_loss']) for name in differences))
        check('step0_parameter_dtype_counts_match_parent',isinstance(baseline.get('model_dtype_counts'),dict) and baseline['model_dtype_counts']==source.get('model_dtype_counts'))
        for name,metric in source['validation_sources'].items():check('step0_full_coverage_'+name,all(baseline['validation_sources'][name][key]==metric[key] for key in ('episodes','frames','chunks','validation_weight')))
        left=(run/'step_000000/model.safetensors').stat();right=(parent/'model.safetensors').stat()
        check('step0_exact_hardlink',left.st_dev==right.st_dev and left.st_ino==right.st_ino and left.st_size==right.st_size)
        check('step0_no_optimizer_payload',not (run/'step_000000/training_state.pt').exists() and baseline['optimizer_payload_saved'] is False)
        report['baseline']={'combined':baseline['validation_flow_loss'],'parent_combined':source['validation_flow_loss'],'combined_difference':baseline['validation_flow_loss']-source['validation_flow_loss'],'source_loss_differences':differences,'losses_bitwise_equal':baseline['validation_flow_loss']==source['validation_flow_loss'] and all(delta==0 for delta in differences.values()),'interpretation':'Weight storage identity, dtypes and complete validation coverage checked separately. Frozen runtime definitively applies a post-load FP32-to-BF16-to-FP32 conversion to stability layers, so identical file bytes do not guarantee identical loaded parameters relative to the parent in-memory state. Loss delta is diagnostic, with no new tolerance or selection-rule change; its magnitude is not attributed solely to GPU noise or quantified as a precision effect.','inherited_round_updates':200,'new_local_updates':0}
        report['known_recovery_interventions']={'optimizer':'Fresh FP32 master AdamW; prior moments and143 unsaved updates not resumed.','precision':'Frozen ModelRuntime invokes official selected dtype conversion after strict load. That function first casts all expert/VLM parameters toBF16, then promotes selected stability layers toFP32. CPU official-method toy demonstrates deterministic rounding while dtype counts and serialized bytes stay unchanged.','comparison_policy':'Root retained the same frozen loading path for all final pi policies; this is weights-only initialization, not exact numerical or optimizer resume.','evidence_artifact':'round02/runtime_postload_dtype_rounding_cpu_diagnostic.json','physical_effect':'Not inferred from the baseline loss delta; requires the frozen physical benchmark.'}
        steps=[row for row in events if row.get('event')=='optimizer_step'];local=max((row['step'] for row in steps),default=0)
        check('consecutive_completed_local_updates',[row['step'] for row in steps]==list(range(1,local+1)))
        check('finite_all_training_losses_gradients',all(math.isfinite(row[key]) for row in steps for key in ('flow_loss','gradient_norm','step_seconds')))
        check('step_lineage_counter_consistency',all(row['inherited_round_updates']==200 and row['cumulative_round_updates']==200+row['step'] for row in steps))
        probes={name:sum(row['parameter_update_probes'][name]['changed_elements']>0 for row in steps) for name in ('vision','VLM','expert','action_in_proj','action_out_proj')}
        check('all_five_sampled_probe_groups_updated',all(value>0 for value in probes.values()));report['probe_steps_with_sampled_updates']=probes
        counts={s['name']:{'prefix':0,'full':0} for s in collection['identity']['sources']};consistent=True
        for row in steps:
            for name in counts:
                draw=row['time_sampling_draw_counts'][name];consistent=consistent and sum(draw.values())==row['batch_sources'].count(name)
                for branch in counts[name]:counts[name][branch]+=draw[branch]
            consistent=consistent and len(row['batch_sources'])==8 and counts==row['cumulative_time_sampling_draw_counts']
        check('sampler_actual_draw_cumulative_consistency',consistent and sum(sum(value.values()) for value in counts.values())==local*8);report['actual_sampler_counts']=counts
        trims=[];interval=start['config']['checkpoint_every'];periodic=[]
        for path in sorted(run.glob('step_*/metadata.json')):
            if path.parent.name.endswith('.incomplete'):continue
            m=read(path)
            if m['step']==0:continue
            periodic.append(m['step']);check('periodic_'+str(m['step'])+'_model_only',m['step']%interval==0 and m['optimizer_payload_saved'] is False and not (path.parent/'training_state.pt').exists())
            cleanup=m.get('cpu_staging_cleanup',[]);check('periodic_'+str(m['step'])+'_trim_diagnostics_present',bool(cleanup))
            for entry in cleanup:
                check('periodic_'+str(m['step'])+'_'+entry['phase']+'_before_after',all(key in entry for key in ('before','after','trim_available','trim_return')))
                trims.append({'checkpoint':str(path.parent),'phase':entry['phase'],'trim_available':entry['trim_available'],'trim_return':entry['trim_return'],'diagnostic':entry.get('diagnostic'),'rss_reclaimed_bytes':entry['before'].get('VmRSS_bytes',0)-entry['after'].get('VmRSS_bytes',0),'anon_reclaimed_bytes':entry['before'].get('RssAnon_bytes',0)-entry['after'].get('RssAnon_bytes',0),'MemAvailable_delta_bytes':entry['after'].get('MemAvailable_bytes',0)-entry['before'].get('MemAvailable_bytes',0)})
        check('all_due_periodic_checkpoints_saved',periodic==list(range(interval,local+1,interval)))
        stops=[row for row in events if row.get('event')=='training_stopped'];check('exactly_one_true_training_stopped',len(stops)==1)
        if len(stops)==1:
            stop=stops[0];check('stopped_at_last_local_update',stop['step']==local and stop['cumulative_round_updates']==200+local)
            stop_selected=Path(stop['selected_checkpoint']);stop_selected=stop_selected if stop_selected.is_absolute() else run/stop_selected
            check('stopped_selection_matches_global_pointer',str(stop_selected.resolve())==selected['path'] and stop['best_validation_loss']==selected['validation_flow_loss'])
            final=(run/stop['final_checkpoint']).resolve();check('final_direct_child',final.parent==run and not final.name.endswith('.incomplete'))
            m=read(final/'metadata.json');check('final_complete_metadata_and_optimizer_flag',m['optimizer_payload_saved'] is True and m['step']==local and m['cumulative_round_updates']==200+local)
            check('last_pointer_matches_true_final',(run/read(run/'last_checkpoint.json')['path']).resolve()==final)
            check('final_only_optimizer_payload',[p.parent.resolve() for p in run.glob('*/training_state.pt') if not p.parent.name.endswith('.incomplete')]==[final])
            report['optimizer_archive']=optimizer_archive_receipt(final/'training_state.pt');check('final_optimizer_archive_structurally_complete',True)
            check('final_trim_diagnostics_present',{'after_model_serialize','after_optimizer_serialize'}<={entry['phase'] for entry in m.get('cpu_staging_cleanup',[])})
            report['true_stop']=stop
        report['trim_observations']=trims
        report['update_accounting']={'failed_attempt_completed_logged':failure['completed_logged_updates'],'failed_attempt_retained':failure['retained_round_updates'],'failed_attempt_unpersisted_completed':failure['unpersisted_completed_updates'],'unproven_inflight_allowance':failure['inflight_update_allowance'],'recovery_local_completed':local,'retained_lineage_cumulative':200+local,'all_known_executed_completed':failure['completed_logged_updates']+local,'cap_with_inflight_allowance':failure['completed_logged_updates']+failure['inflight_update_allowance']+local}
        check('lost143_distinct_from_retained200',failure['completed_logged_updates']==343 and failure['retained_round_updates']==200 and failure['unpersisted_completed_updates']==143)
        check('round_cap_with_inflight_allowance',report['update_accounting']['cap_with_inflight_allowance']<=2000)
        if budget:
            ledger=read(budget);matches=[row for row in ledger['runs'] if row.get('stage')=='training' and str(run) in row.get('command',[])]
            check('unique_owned_budget_run',len(matches)==1)
            if len(matches)==1:
                row=matches[0];report['budget_snapshot']=row
                check('recovery_budget_success_closed',row.get('status')=='completed' and row.get('exit_code')==0 and not row.get('accounting_open') and row.get('last_unit_properties',{}).get('ActiveState')=='inactive')
        if command_receipt:
            frozen=read(command_receipt);small={};large=[]
            for path,digest in frozen['frozen_files'].items():
                if Path(path).stat().st_size>4*1024**2:large.append(path);continue
                small[path]=sha(path)==digest
            check('frozen_small_command_inputs_unchanged',all(small.values()));report['frozen_non_payload_files_checked']=small;report['large_command_files_not_hashed']=large
        report['limits']=['Validation benefit is diagnostic; physical benefit requires frozen benchmark.','Known precision intervention: post-load FP32→BF16→FP32 rounding is established, not merely GPU noise. Byte-identical checkpoints and equal dtype counts do not prove unchanged loaded stable-layer values. The observed local0 loss delta is reported without causal or physical-effect attribution.','Fresh Adam reset and retained200 versus143 unpersisted updates are separate from precision conversion.','Probe updates sample128 elements/group, not proof all parameters changed.','Trim deltas are observed CPU RSS changes; MemAvailable also responds to other workloads and no fixed reclaim amount is assumed.','Final optimizer structural audit reads ZIP index/pickle opcodes only; full payload integrity/dtypes remain the producer/phase-preservation boundary.']
    except Exception as error:report['errors'].append(type(error).__name__+': '+str(error))
    report['status']='passed_cpu_acceptance' if not report['errors'] and all(checks.values()) else 'incomplete_or_failed'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('training-dir','data-config','norm-stats','loss-weights','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--budget');parser.add_argument('--command-receipt');args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError('Never overwrite an existing acceptance receipt')
    report=audit(args.training_dir,args.data_config,args.norm_stats,args.loss_weights,budget=args.budget,command_receipt=args.command_receipt)
    with output.open('x') as handle:handle.write(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'status':report['status'],'checks':len(report['checks']),'errors':report['errors'],'report':str(output.resolve())}))
    raise SystemExit(0 if report['status']=='passed_cpu_acceptance' else 2)


if __name__=='__main__':main()
