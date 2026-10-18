"""CPU-only command receipt; never launches its emitted GPU command."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'.scratch/pi05_g1/round03_mobile'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(*,output,shared_fifo_wrapper=None):
    output=Path(output).resolve()
    if output.parent!=DATA or output.exists():
        raise ValueError('New direct-child mobile output directory required')
    preflight_path=DATA/'training_cpu_preflight_no_optimizer_v2.json'
    prior=json.loads(preflight_path.read_text())
    manifest=DATA/'prepared/split_manifest.json';norm=DATA/'prepared/train_norm_stats.json'
    split=json.loads(manifest.read_text())
    chunks=sum((x['length']+49)//50 for x in split['files'] if x['episode_id'] in split['validation'])
    if [len(split[k]) for k in ('train','validation','test')]!=[80,10,10] or chunks!=190:
        raise ValueError('Frozen80/10/10 and190 validation chunks required')
    if digest(manifest)!=prior['manifest_sha256'] or digest(norm)!=prior['norm_sha256']:
        raise ValueError('Prepared mobile identity changed since CPU admission')
    argv=[str(ROOT/'.scratch/pi05_g1/venv/bin/python'),'-m','experiments.pi05_g1_mobile.training',
          '--checkpoint',str(ROOT/'.scratch/pi05_g1/pi05_base_pytorch'),
          '--dataset',str(DATA/'data/lerobot'),'--manifest',str(manifest),'--norm-stats',str(norm),
          '--output',str(output),'--hours','4.3','--max-steps','1600','--checkpoint-every','400',
          '--seed','42','--disk-reserve-gib','32','--save-reserve-seconds','450',
          '--validation-reserve-seconds','240']
    budget=[argv[0],'-m','experiments.pi05_g1_mobile.budget','--ledger',str(DATA/'gpu_budget.json'),
            '--stage','training','--max-seconds','16200','--host-reserve-gib','16',
            '--result-file',str(output/'training_result.json'),
            '--result-schema','pi05_g1_mobile_training_result_v1','--',*argv]
    wrapper=str(Path(shared_fifo_wrapper).resolve()) if shared_fifo_wrapper else '<ROOT_CONFIRMED_SHARED_FIFO_WRAPPER>'
    if shared_fifo_wrapper and not Path(wrapper).is_file():raise FileNotFoundError(wrapper)
    # The peer wrapper invocation contract is not yet known. Do not pretend a
    # shell string containing its path is an executable, validated launch recipe.
    sources={str(p):digest(p) for p in [Path(__file__),ROOT/'experiments/pi05_g1_mobile/training.py',ROOT/'experiments/pi05_g1_mobile/model_runtime.py',ROOT/'experiments/pi05_g1_mobile/contract.py',manifest,norm]}
    expected=prior['disk_estimate'];free=shutil.disk_usage(DATA).free
    return {'schema':'pi05_g1_mobile_training_command_preparation_v1','status':'prepared_not_launched',
            'launch_ready':False,'launch_blocker':'Root must provide shared FIFO wrapper invocation and approve pending teacher decision; fresh producer preflight remains mandatory',
            'shared_fifo_wrapper':wrapper,'shared_fifo_wrapper_argv':None,
            'environment':{'PYTHONPATH':str(ROOT/'.scratch/pi05_g1/openpi/src')+':'+str(ROOT),
                'JAX_PLATFORMS':'cpu','OPENPI_DATA_HOME':str(ROOT/'.scratch/pi05_g1/venv/openpi-cache'),
                'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},
            'environment_note':'CUDA visibility is root/supervisor owned; do not inherit CPU CUDA_VISIBLE_DEVICES empty into actual launch',
            'producer_argv':argv,'budget_argv':budget,'budget_command_for_review':shlex.join(budget),
            'cpu_preflight_argv':[*argv,'--preflight-only'],
            'completion_contract':{'path':str(output/'training_result.json'),'schema':'pi05_g1_mobile_training_result_v1','required_status':'completed','requires_training_stopped':True},
            'validation':{'episodes':10,'chunks':chunks,'step0_before_updates':True,'test_loaded':False,'selection':'Strict minimum complete validation, step0 included; ties preserve earlier checkpoint'},
            'optimizer':{'fresh_fp32_master_adam':True,'saved_by_default':False,'exact_resume_supported':False},
            'time':{'inner_seconds':15480,'updates_stop_seconds':14790,'final_save_reserve_seconds':450,'final_validation_reserve_seconds':240,'outer_seconds':16200,'outer_margin_seconds':720,
                'prediction':'1600 is a cap, not guaranteed. At8.5s/update=13600s leaves1190s before update deadline for startup/baseline/periodic IO and validation. At9.3s/update=14880s already exceeds update window before overhead.'},
            'disk':{'free_bytes_now':free,'prior_admission_estimate':expected,'currently_above_admission':free>=expected['required_free_bytes'],
                'maximum_new_payloads':'4 periodic +1 final; step0 same-filesystem official-base hardlink; estimate conservatively reserves6 full models',
                'model_bytes':expected['model_bytes'],'serialization_cpu_staging_upper_payload_bytes':expected['model_bytes'],
                'memory_warning':'Model-only saving still stages CPU bytes and may retain allocator memory; prior malloc_trim did not solve this. Keep external16GiB guard; no plateau guarantee for new model/run.'},
            'small_input_source_sha256':sources,'reference_admission_sha256':digest(preflight_path),
            'large_weights_hashed_or_loaded':False,'data_or_configuration_modified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output',default=str(DATA/'training_run01'))
    parser.add_argument('--shared-fifo-wrapper')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();result=build(output=args.training_output,shared_fifo_wrapper=args.shared_fifo_wrapper)
    target=Path(args.output)
    with target.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'receipt':str(target.resolve()),'launch_ready':False}))

if __name__=='__main__':main()
