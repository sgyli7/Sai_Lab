"""Freeze fresh independent development of the actual R8 final with coherent sampling."""
import argparse
import copy
import json
from pathlib import Path

from .acceptance_context import measurement_identity
from .data import OWN, ROOT, sha
from .make_plan import experiment_sources, write
from .round08_selection import admit
from experiments.pi05_g1_mobile.simulator import validate_config


def prepare(training_result, output):
    output = Path(output).absolute()
    if output.exists() or not output.is_relative_to(OWN):
        raise FileExistsError('Fresh owned development plan required')
    gate_path = OWN / 'round08_candidate_roles_CPU_gate.json'
    gate = json.loads(gate_path.read_text())
    if gate.get('passed') is not True or gate.get('tests_run') != 5 or gate.get('GPU_initialized') is not False:
        raise ValueError('Actual R8 role/lineage CPU regressions required')
    for name, digest in gate['source_files_sha256'].items():
        if sha(name) != digest:
            raise ValueError('R8 candidate checker changed after its CPU gate')
    sampling_gate_path = OWN / 'coherent_noise01_CPU_gate.json'
    sampling_gate = json.loads(sampling_gate_path.read_text())
    if sampling_gate.get('passed') is not True or sampling_gate.get('tests_run') != 3:
        raise ValueError('Actual coherent sampling and source-drop-veto CPU checks required')
    for name, digest in sampling_gate['source_files_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Sampling source changed after CPU regression gate')
    result, delta, frozen, payloads, provenance = admit(training_result)
    mount_gate_path = OWN / 'backup_mount_CPU_gate.json'
    mount_gate = json.loads(mount_gate_path.read_text())
    if (mount_gate.get('passed') is not True or mount_gate.get('GPU_context_created') is not False
            or mount_gate.get('actual_CPU_container_check', {}).get('backup_read_only') is not True):
        raise ValueError('Actual read-only permanent checkpoint visibility required')
    for name, digest in mount_gate['source_files_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Permanent checkpoint mount changed after CPU check')
        frozen[name] = digest
    frozen[str(mount_gate_path)] = sha(mount_gate_path)
    frozen.update(sampling_gate['source_files_sha256'])
    frozen[str(sampling_gate_path)] = sha(sampling_gate_path)
    frozen.update(gate['source_files_sha256'])
    frozen[str(gate_path)] = sha(gate_path)
    cohort_path = OWN / 'round08_development_cohort.json'
    cohort = json.loads(cohort_path.read_text())
    if cohort.get('schema') != 'lingbot_round08_physical_development_cohort_v1' or len(cohort['cases']) != 2:
        raise ValueError('Predeclared two-case development cohort required')
    if frozen.get(str(cohort_path)) != sha(cohort_path):
        raise ValueError('R8 development scenes must already be frozen in the training job')
    teachers = [json.loads((OWN / name / 'config.json').read_text())
                for name in ('placement_teacher02', 'placement_recovery_teacher04', 'placement_recovery_teacher08', 'placement_recovery_teacher09')]
    for case in cohort['cases']:
        if any(case['seed'] == teacher['case']['seed'] or case['scene'] == teacher['scene'] for teacher in teachers):
            raise ValueError('TRAIN recovery scenes cannot qualify independent R8 development')
    template = json.loads((OWN / 'resumed_turn01/configs/paired_pointmass.json').read_text())
    for name, digest in template['frozen_files'].items():
        if name in frozen and frozen[name] != digest:
            raise ValueError('Training and simulator source identities conflict: ' + name)
        frozen[name] = digest
    starts = [Path(__file__), ROOT / 'experiments/lingbot_vla2_g1/run_development_task_v6.py',
              ROOT / 'experiments/lingbot_vla2_g1/diagnostics/development_audit_v2.py',
              ROOT / 'experiments/lingbot_vla2_g1/service_coherent_noise.py',
              ROOT / 'experiments/lingbot_vla2_g1/run_case_with_backup.py',
              ROOT / 'experiments/lingbot_vla2_g1/budget_reallocated_v7.py',
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
                      policy_command=[str(OWN / 'venv/bin/python'), '-m', 'experiments.lingbot_vla2_g1.service_coherent_noise',
                                      '--delta', str(delta), '--norm-stats', str(OWN / 'train_meanstd.json'),
                                      '--decoded-evidence', str(folder / 'decoded.jsonl')])
        config['HFmodel']['sampling_adapter'] = 'coherent_antithetic_two_predictions_v1'
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
            'deployed_total_updates': 2040 + result['physical_candidate_step'],
            'sampling_adapter': 'coherent_antithetic_two_predictions_v1',
            'fresh_development_scenes_after_recovery_training': True,
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
