"""Explicit sampled actor-position linear-speed acceptance measurement proposal.

Only the temporary evaluator copy's box linear velocity is replaced by adjacent
actor-origin displacement /0.02s. Original SDK6D velocities, all raw rows,
geometry, angular-speed gate, thresholds and simulator/policy remain unchanged.
First sample is undefined and cannot be eligible. Two seconds of all-pairs pose
stability, contacts, support, separation, standing and carry are still mandatory.

Default --config CFG runs the unchanged simulator under this context. CPU:
--module APPROVED_CPU_MODULE [--identity-output NEW.json] -- original CLI args.
Prebound old function aliases are deliberately not rewritten.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import runpy
import sys
import threading
import numpy as np
from . import acceptance as original
from . import acceptance_body_ids as body_ids
from . import simulator

SCHEMA='pi05_g1_mobile_pose_difference_speed_v1'
BODY_ADAPTER_SHA256='f73f6785c925645d1309613ef2b37afe789ce5b944c9ddfc44baa6002adb5459'
SIMULATOR_SHA256='ee39d23414170e602560a1f4c4dd364e03bee91623ba4fa5566d621577e4424f'
_RAW_EVALUATE=original.evaluate_episode
_RAW_PREFILTER=simulator.verified_success_check
_LOCK=threading.RLock()
FIRST_UNDEFINED_SENTINEL_M_S=1e6


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array(value,shape,label):
    result=np.asarray(value,dtype=float)
    if result.shape!=shape or not np.isfinite(result).all():raise ValueError('Invalid raw '+label)
    return result


def validate_raw_rows(rows):
    if not rows:raise ValueError('Nonempty raw trace required')
    previous=None
    for tick,row in enumerate(rows):
        if type(row['control_tick']) is not int or row['control_tick']!=tick or type(row['physics_steps_since_reset']) is not int or row['physics_steps_since_reset']!=4*tick:
            raise ValueError('Raw tick/physics counters not contiguous from reset')
        sim=row['sim_time_seconds']
        if type(sim) not in (int,float) or not math.isfinite(sim) or not math.isclose(sim,tick/50,rel_tol=0.,abs_tol=1e-12):raise ValueError('Raw simulation dt must be0.02s')
        for clock in ('monotonic_ns','wall_time_ns'):
            if type(row[clock]) is not int or (previous is not None and row[clock]<=previous[clock]):raise ValueError('Raw timestamps must strictly increase')
        for actor in ('brown_box','blue_sorting_bin'):
            truth=row['acceptance_truth_only'][actor]
            _array(truth['pose'],(7,),actor+' pose')
            _array(truth['velocity'],(6,),actor+' SDK velocity')
        _array(row['root_link_pose_w'],(7,),'root pose')
        _array(row['acceptance_robot_body_link_poses_w'],(53,7),'robot body poses')
        for field in ('root_link_velocity_w','root_com_velocity_w'):
            if field in row:_array(row[field],(6,),field)
        if previous is not None and not math.isclose(sim-previous['sim_time_seconds'],.02,rel_tol=0.,abs_tol=1e-12):raise ValueError('Adjacent simulation timestamps differ from0.02s')
        previous=row


def _sample_linear(previous,row):
    if previous is None:return np.array([FIRST_UNDEFINED_SENTINEL_M_S,0.,0.])
    current=_array(row['acceptance_truth_only']['brown_box']['pose'],(7,),'box pose')
    before=_array(previous['acceptance_truth_only']['brown_box']['pose'],(7,),'previous box pose')
    delta=(current[:3]-before[:3])/.02
    if not np.isfinite(delta).all():raise ValueError('Derived pose-difference speed is nonfinite')
    return delta


def _copy_row_with_linear(row,linear):
    truth=row['acceptance_truth_only'];box=truth['brown_box'];velocity=list(box['velocity'])
    velocity[:3]=linear.tolist()
    return {**row,'acceptance_truth_only':{**truth,'brown_box':{**box,'velocity':velocity}}}


def measured_rows(rows):
    """Validated temporary shallow row copies; no original nested data writes."""
    validate_raw_rows(rows)
    return [_copy_row_with_linear(row,_sample_linear(rows[i-1] if i else None,row)) for i,row in enumerate(rows)]


def measurement_identity():
    return {'schema':SCHEMA,'linear_speed_source':'Adjacent recorded box actor-origin world position difference divided by strict0.02s',
        'sampling_hz':50,'first_sample_undefined_and_ineligible':True,'first_sample_finite_sentinel_m_s':FIRST_UNDEFINED_SENTINEL_M_S,
        'raw_SDK_velocity6_retained_unchanged':True,'angular_speed_source':'Original raw SDK box velocity[3:6], unchanged angular<0.1rad/s gate',
        'placement_linear_speed_threshold_m_s':.02,'all_other_acceptance_math_and_thresholds_unchanged':True,
        'original_acceptance_sha256':body_ids.ORIGINAL_ACCEPTANCE_SHA256,'body_adapter_sha256':BODY_ADAPTER_SHA256,
        'measurement_module_sha256':sha(__file__),
        'scope':'Sampled pose motion, not continuous-time stillness or an explanation of SDK velocity-position discrepancy'}


def _evaluate_with_pose_speed(rows,config):
    # Hold the unpatched original function; never recurse through adapter.evaluate.
    copied=measured_rows(rows)
    judgment=_RAW_EVALUATE(copied,config)
    judgment['measurement_source']=measurement_identity()
    return judgment


def _prefilter_with_pose_speed(rows,config,evaluator):
    # Retain cheap original eligibility scheduling. Final evaluator always gets RAW.
    if (not rows or config.get('mode')!='policy' or config.get('stop_on_verified_success') is not True or
        config.get('evaluate_acceptance') is not True or rows[-1]['control_tick']<100 or rows[-1]['control_tick']%50):
        return _RAW_PREFILTER(rows,config,evaluator)
    validate_raw_rows(rows)
    proxy=[*rows[:-1],_copy_row_with_linear(rows[-1],_sample_linear(rows[-2],rows[-1]))]
    def raw_final_evaluator(ignored_proxy,context):return evaluator(rows,context)
    return _RAW_PREFILTER(proxy,config,raw_final_evaluator)


@contextmanager
def pose_speed_measurement():
    """Explicit module bindings for real simulator/approved CPU execution paths."""
    with _LOCK:
        if sha(body_ids.__file__)!=BODY_ADAPTER_SHA256 or sha(simulator.__file__)!=SIMULATOR_SHA256:
            raise ValueError('Frozen body adapter/simulator source changed')
        if original.evaluate_episode is not _RAW_EVALUATE or simulator.verified_success_check is not _RAW_PREFILTER:
            raise RuntimeError('Unexpected or nested measurement binding')
        with body_ids.body_name_geometry():
            original.evaluate_episode=_evaluate_with_pose_speed
            simulator.verified_success_check=_prefilter_with_pose_speed
            try:yield
            finally:
                original.evaluate_episode=_RAW_EVALUATE
                simulator.verified_success_check=_RAW_PREFILTER


def evaluate_episode(rows,config):
    with pose_speed_measurement():return original.evaluate_episode(rows,config)


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv);cpu='--module' in args
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--identity-output',type=Path)
    if cpu:
        parser.add_argument('--module',choices=sorted(body_ids.APPROVED_CPU_MODULES),required=True)
        if '--' not in args:parser.error('CPU module requires explicit -- separator')
        index=args.index('--');options=parser.parse_args(args[:index]);forwarded=args[index+1:]
        if not forwarded:parser.error('Explicit target module args required')
        target=options.module
    else:
        parser.add_argument('--config',required=True);options=parser.parse_args(args)
        target='experiments.pi05_g1_mobile.simulator';forwarded=['--config',options.config]
    spec=importlib.util.find_spec(target)
    if spec is None or not spec.origin:raise ValueError('Approved target source missing')
    identity={**measurement_identity(),'target_module':target,'target_source':spec.origin,'target_sha256':sha(spec.origin),'cpu_module_mode':cpu,
        'binding_scope':'Module attributes/future imports only; existing external function aliases are not rewritten'}
    if options.identity_output:
        with options.identity_output.open('x') as f:f.write(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(identity),file=sys.stderr,flush=True);previous=sys.argv
    try:
        sys.argv=[target,*forwarded]
        with pose_speed_measurement():
            if cpu:return runpy.run_module(target,run_name='__main__')
            return simulator.main()
    finally:sys.argv=previous

if __name__=='__main__':main()
