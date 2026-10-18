"""Reproduce all recovery collector commands and independently audit physics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.pi05_g1_mobile.audit_episodes_v3 import audit_case as raw_audit
from experiments.lingbot_vla2_g1.data import sha
from .privileged_teacher_v4 import PlacementTeacher, validate_source
from .teacher_measurement import teacher_measurement


def decoded_evidence(config, result):
    declaration = config['privileged_TRAIN_teacher']
    validate_source(declaration)
    path = Path(config['output_dir']) / 'decoded.jsonl'
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if (events[0].get('event') != 'privileged_TRAIN_teacher_loaded'
            or events[0].get('counts_as_LingBot_success') is not False
            or events[0].get('autonomous_model_policy') is not False
            or events[0].get('declaration') != declaration
            or events[0]['teacher_sha256'] != sha(Path(__file__).with_name('privileged_teacher_v4.py'))):
        raise ValueError('Actual recovery collector identity differs')
    source = [json.loads(line) for line in Path(declaration['source_trace']).read_text().splitlines()]
    rows = [json.loads(line) for line in (Path(config['output_dir']) / 'trajectory.jsonl').read_text().splitlines()]
    teacher = PlacementTeacher(source, json.loads(Path(declaration['object_geometry']).read_text()),
                               json.loads(Path(declaration['robot_geometry']).read_text()),
                               result['body_names'], action_bounds=result['actual_action_bounds'])
    ticks = list(range(0, result['completed_control_ticks'], 50))
    if [event['control_tick'] for event in events[1:]] != ticks:
        raise ValueError('Collector command coverage differs')
    for tick, event in zip(ticks, events[1:]):
        expected, diagnostics = teacher.actions(rows[tick])
        if (not np.array_equal(expected, np.asarray(event['actions32'], np.float32))
                or any(event.get(key) != value for key, value in diagnostics.items())):
            raise ValueError('Recovery collector commands not reproducible at tick ' + str(tick))
    return path, events


def audit(config_path):
    with teacher_measurement():
        report = raw_audit(config_path, decoded_checker=decoded_evidence)
    report.pop('audit_sha256', None)
    report.update(schema='lingbot_privileged_TRAIN_teacher_CPU_audit_v1', collector_version=4,
                  autonomous_policy_trial=False, autonomous_model_policy=False,
                  formal_benchmark_trial=False, counts_as_LingBot_success=False,
                  privileged_scripted_TRAIN_collection=True,
                  attribution_source_sha256=sha(Path(__file__).with_name('teacher_measurement.py')),
                  collector_audit_source_sha256=sha(__file__))
    report['audit_sha256'] = hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.output.absolute().is_relative_to(Path(config['output_dir']).absolute()):
        raise ValueError('Audit belongs outside immutable raw evidence')
    report = audit(args.config)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'passed': report['passed'], 'anomalies': report['anomalies'],
                      'physical_teacher_success': (report.get('recomputed_acceptance') or {}).get('success'),
                      'LingBot_success_claim': False}))
    raise SystemExit(0 if report['passed'] else 2)


if __name__ == '__main__':
    main()
