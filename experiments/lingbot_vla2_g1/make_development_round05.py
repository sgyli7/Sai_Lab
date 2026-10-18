"""Freeze R5 actual-final development under the unchanged taskv4 protocol."""
import argparse
import copy
import json
from pathlib import Path

from .acceptance_context import measurement_identity
from .data import OWN, ROOT, sha
from .make_plan import experiment_sources, write
from .round05_selection import admit
from experiments.pi05_g1_mobile.simulator import validate_config


def prepare(training_result, output):
    output = Path(output).absolute()
    if output.exists() or not output.is_relative_to(OWN):
        raise FileExistsError('Fresh owned development plan required')
    gate_path = OWN / 'round05_candidate_roles_CPU_gate.json'
    gate = json.loads(gate_path.read_text())
    if gate.get('passed') is not True or gate.get('tests_run') != 4 or gate.get('GPU_initialized') is not False:
        raise ValueError('Actual R5 role/lineage CPU regressions required')
    for name, digest in gate['source_files_sha256'].items():
        if sha(name) != digest:
            raise ValueError('R5 candidate checker changed after its CPU gate')
    result, delta, frozen, payloads, provenance = admit(training_result)
    frozen.update(gate['source_files_sha256'])
    frozen[str(gate_path)] = sha(gate_path)
    cohort_path = OWN / 'round05_development_cohort.json'
    cohort = json.loads(cohort_path.read_text())
    if cohort.get('schema') != 'lingbot_round05_physical_development_cohort_v1' or len(cohort['cases']) != 2:
        raise ValueError('Predeclared two-case development cohort required')
    template = json.loads((OWN / 'resumed_turn01/configs/paired_pointmass.json').read_text())
    for name, digest in template['frozen_files'].items():
        if name in frozen and frozen[name] != digest:
            raise ValueError('Training and simulator source identities conflict: ' + name)
        frozen[name] = digest
    starts = [Path(__file__), ROOT / 'experiments/lingbot_vla2_g1/run_development_task_v4.py',
              ROOT / 'experiments/lingbot_vla2_g1/diagnostics/development_audit.py',
              ROOT / 'experiments/lingbot_vla2_g1/service_pointmass.py',
              ROOT / 'experiments/lingbot_vla2_g1/run_case_pose_speed.py',
              ROOT / 'experiments/lingbot_vla2_g1/budget_reallocated_v5.py',
              ROOT / 'experiments/lingbot_vla2_g1/job.py']
    for path in experiment_sources(starts):
        frozen[str(path)] = sha(path)
    frozen[str(cohort_path)] = sha(cohort_path)
    output.mkdir()
    write(output / 'training_provenance.json', provenance)
    frozen[str(output / 'training_provenance.json')] = sha(output / 'training_provenance.json')
    for name, digest in frozen.items():
        if sha(name) != digest:
            raise ValueError('Frozen development input changed: ' + name)
    jobs = []
    for index, case in enumerate(cohort['cases']):
        folder = output / 'episodes' / case['case_id']
        config = copy.deepcopy(template)
        config.update(case={k: v for k, v in case.items() if k in ('case_id', 'cluster_id', 'seed', 'nominal_xy_offset_m')},
                      scene=copy.deepcopy(case['scene']), ticks=case['ticks'], episode_seconds=case['episode_seconds'],
                      output_dir=str(folder), frozen_files=dict(frozen), save_full_rgb=False,
                      acceptance_measurement=measurement_identity(),
                      max_run_wall_seconds=400 if index == 0 else 840,
                      HFmodel={**template['HFmodel'], 'delta': str(delta), 'selected_by_validation': result['selected_step'] == result['physical_candidate_step']},
                      policy_command=[str(OWN / 'venv/bin/python'), '-m', 'experiments.lingbot_vla2_g1.service_pointmass',
                                      '--delta', str(delta), '--norm-stats', str(OWN / 'train_meanstd.json'),
                                      '--decoded-evidence', str(folder / 'decoded.jsonl')])
        config = validate_config(config)
        path = output / 'configs' / (case['case_id'] + '.json')
        write(path, config)
        jobs.append({'index': index, 'case_id': case['case_id'], 'config_path': str(path),
                     'config_sha256': sha(path), 'result_file': str(folder / 'result.json'),
                     'timeout_seconds': 450 if index == 0 else 900,
                     'scope': case['purpose']})
        frozen[str(path)] = sha(path)
    plan = {'schema': 'lingbot_task_v4_development_plan_v1', 'status': 'prepared',
            'frozen_files': frozen, 'immutable_payloads': payloads, 'jobs': jobs,
            'cohort': str(cohort_path), 'cohort_sha256': sha(cohort_path),
            'training_result': str(Path(training_result).absolute()),
            'selected_delta': str(delta), 'selected_new_step': result['physical_candidate_step'],
            'deployed_total_updates': 1400 + result['physical_candidate_step'],
            'formal20_benchmark': False, 'complete_task_thresholds_unchanged': True,
            'conditional_full_case': 'Run only after independently audited first-screen grasp with no fall/drop.',
            'max_GPU_seconds': 1800}
    write(output / 'plan.json', plan)
    print(json.dumps({'plan': str(output / 'plan.json'), 'plan_sha256': sha(output / 'plan.json'),
                      'selected_step': result['physical_candidate_step'], 'jobs': 2}), flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.training_result, args.output)


if __name__ == '__main__':
    main()
