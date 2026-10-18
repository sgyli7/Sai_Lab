"""Bind the shared, prospectively qualified placement measurement explicitly."""
import hashlib
import json
from pathlib import Path

from experiments.pi05_g1_mobile.acceptance_pose_speed import measurement_identity, pose_speed_measurement
from experiments.pi05_g1_mobile.audit_episodes_v3 import audit_case as raw_audit_case
from .data import sha


def admit_qualification(path):
    path = Path(path).absolute()
    qualification = json.loads(path.read_text())
    if qualification.get('schema') != 'pi05_g1_mobile_placement_primitive_qualification_v1' or qualification.get('status') != 'completed' or qualification.get('passed') is not True:
        raise ValueError('Completed prospective physical placement qualification required')
    for flag in ['full_task_success', 'autonomous', 'teacher', 'native_policy_trial', 'full_task_trial', 'teacher_trial', 'formal_benchmark_trial']:
        if qualification.get(flag) is not False:
            raise ValueError('Primitive calibration must not be mislabeled as a policy/task trial')
    identity = measurement_identity()
    if qualification.get('measurement_source') != identity or qualification.get('controls') != 150 or qualification.get('physics_integrations') != 600 or qualification.get('qualified_stable_2s_windows', 0) < 1 or qualification.get('max_ready_span_seconds', 0) < 2:
        raise ValueError('Qualified measurement identity or observed two-second window differs')
    audit_path = Path(qualification['audit']).absolute()
    if sha(audit_path) != qualification['independent_audit_file_sha256']:
        raise ValueError('Independent qualification audit changed')
    group = json.loads(audit_path.read_text())
    if group.get('passed') is not True or len(group.get('episodes', [])) != 1:
        raise ValueError('Exactly one passed independent prospective audit required')
    audit = group['episodes'][0]
    canonical = {k: v for k, v in audit.items() if k != 'audit_sha256'}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if audit.get('passed') is not True or audit.get('audit_sha256') != digest or digest != qualification['independent_audit_sha256']:
        raise ValueError('Independent audit identity invalid')
    result_path = Path(qualification['episode']).absolute() / 'result.json'
    result = json.loads(result_path.read_text())
    if audit.get('result_path') != str(result_path) or audit.get('autonomous_policy_trial') is not False or result.get('success') is not False or result.get('status') != 'completed' or result.get('source_interface_passed') is not True or result.get('completed_control_ticks') != 150 or result.get('actual_sdk_integrations') != 600:
        raise ValueError('Prospective source simulator did not actually complete')
    if result.get('acceptance', {}).get('measurement_source') != identity or audit.get('recomputed_acceptance', {}).get('measurement_source') != identity:
        raise ValueError('Online and independent measurement definitions differ')
    frozen = dict(qualification['inputs_sha256'])
    for name, expected in audit['inputs_sha256'].items():
        if name in frozen and frozen[name] != expected:
            raise ValueError('Qualification/audit inputs differ')
        frozen[name] = expected
    frozen.update({str(path): sha(path), str(audit_path): sha(audit_path)})
    for name, expected in frozen.items():
        if sha(name) != expected:
            raise ValueError('Qualification evidence/source changed: ' + name)
    return {'path': str(path), 'sha256': sha(path), 'audit_path': str(audit_path), 'audit_sha256': sha(audit_path), 'measurement_source': identity, 'physical_controls': 150, 'physics_integrations': 600, 'qualified_two_second_windows': qualification['qualified_stable_2s_windows'], 'scope': 'Placement primitive calibration only; model selection and task outcomes unchanged'}, frozen


def audit_case(value, *, allow_nonformal=False, decoded_checker=None):
    with pose_speed_measurement():
        return raw_audit_case(value, allow_nonformal=allow_nonformal, decoded_checker=decoded_checker)
