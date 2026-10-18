"""Freeze one diagnosis, a matched grasp ablation and one new TRAIN collection.

CPU preparation only. FIFO request and launch require the root's separate final
review. No formal-test trajectory is admitted for training.
"""
import copy
import datetime
import json
from pathlib import Path
import shutil

from experiments.lingbot_vla2_g1.acceptance_context import measurement_identity
from experiments.lingbot_vla2_g1.budget_continuation import admit_allocation
from experiments.lingbot_vla2_g1.data import OWN, ROOT, SHARED, sha
from experiments.lingbot_vla2_g1.make_plan import experiment_sources, write
from experiments.pi05_g1_mobile.simulator import validate_config


def prepare():
    destination = OWN / 'resumed_turn01'
    if destination.exists():
        raise FileExistsError('Preserve all previous preparations')
    allocation = admit_allocation(OWN / 'gpu_budget.json')
    old_plan = json.loads((OWN / 'evaluation_plan_round02/plan.json').read_text())
    native_path = SHARED / 'formal_plan_pose_speed/configs/gr00t_n16_long5m_center_s1040.json'
    native = json.loads(native_path.read_text())
    baseline = json.loads(Path(old_plan['jobs'][0]['config_path']).read_text())
    scenarios = json.loads((SHARED / 'scenarios_frozen_v3.json').read_text())
    frozen = {**baseline['frozen_files'], **native['frozen_files']}
    additions = [ROOT / 'experiments/lingbot_vla2_g1' / name for name in (
        'diagnostics/action_probe.py', 'diagnostics/resume_batch.py',
        'diagnostics/prepare_resume_turn.py', 'diagnostics/development_audit.py',
        'constant_inverse.py', 'service_pointmass.py', 'budget_continuation.py', 'job.py')]
    for path in experiment_sources(additions):
        frozen[str(path)] = sha(path)
    for path in [OWN / 'gpu_budget_allocation_user_resume_v4.json',
                 OWN / 'gpu_budget_before_user_resume_v4.json',
                 OWN / 'training_round02/selected/metadata.json']:
        frozen[str(path)] = sha(path)
    for name, expected in frozen.items():
        if sha(name) != expected:
            raise ValueError('Previously frozen input changed before preparation: ' + name)
    backup = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z')
    existing_bytes = sum(p.stat().st_size for folder in (OWN, backup) for p in folder.rglob('*') if p.is_file())
    free = shutil.disk_usage(OWN).free
    growth = 12 * 1024 ** 3
    if existing_bytes + growth > 48 * 1024 ** 3 or free - growth < 32 * 1024 ** 3:
        raise RuntimeError('Prospective owned48GiB combined cap and32GiB global floor required')
    cohort = {
        'schema': 'lingbot_resumed_development_cohort_v1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'formal20_unchanged_sha256': sha(SHARED / 'scenarios_frozen_v3.json'),
        'formal_or_original_test_demonstrations_used_for_training': False,
        'ablation': {'seed': 2026101701, 'xy_offset_m': [.01, .01],
                     'simulation_seconds': 20., 'purpose': 'Matched first-grasp diagnostic only; not the common60s benchmark.',
                     'train_admission_allowed': False},
        'teacher': {'seed': 2026101702, 'xy_offset_m': [-.01, -.01],
                    'simulation_seconds': 60., 'purpose': 'Fresh autonomous native TRAIN collection; full-success admission only after independent physics and every RGB/state/action alignment check.'},
        'held_fixed': 'Same GB10, G1/Homie, source scene assets, head/observer cameras,32actions,50Hz/200Hz and original full transport predicates.',
        'candidate_change': 'Only exactly zero-variance original80TRAIN action dimensions invert to their empirical mean; no object/goal state enters the policy.',
        'resources': {'allocation': allocation, 'current_combined_owned_bytes': existing_bytes,
                      'combined_owned_cap_bytes': 48 * 1024 ** 3, 'prospective_growth_bound_bytes': growth,
                      'free_disk_bytes': free, 'global_minimum_free_bytes': 32 * 1024 ** 3,
                      'host_minimum_available_bytes': 16 * 1024 ** 3,
                      'finite_FIFO_GPU_limit_seconds': 3000},
    }
    destination.mkdir()
    write(destination / 'cohort.json', cohort)
    frozen[str(destination / 'cohort.json')] = sha(destination / 'cohort.json')
    configs = []
    for label, original, seed, offset, seconds in (
        ('paired_baseline', baseline, 2026101701, [.01, .01], 20.),
        ('paired_pointmass', baseline, 2026101701, [.01, .01], 20.),
        ('native_teacher01', native, 2026101702, [-.01, -.01], 60.),
    ):
        config = copy.deepcopy(original)
        scene = copy.deepcopy(scenarios['cases'][0]['scene'])
        for axis in range(2):
            scene['box_pose']['position_xyz'][axis] += offset[axis]
        if any(scene == case['scene'] for case in scenarios['cases']):
            raise ValueError('Development physical position overlaps a formal position')
        case = {'case_id': 'resume01_' + label, 'cluster_id': 'resume01_development',
                'seed': seed, 'nominal_xy_offset_m': offset}
        folder = destination / 'episodes' / case['case_id']
        command = list(config['policy_command'])
        command[command.index('--decoded-evidence') + 1] = str(folder / 'decoded.jsonl')
        if label == 'native_teacher01':
            command[command.index('--seed') + 1] = str(seed)
        elif label == 'paired_pointmass':
            command[2] = 'experiments.lingbot_vla2_g1.service_pointmass'
        config.update(case=case, output_dir=str(folder), scene=scene,
                      ticks=round(seconds * 50), episode_seconds=seconds,
                      policy_command=command, frozen_files=dict(frozen),
                      save_full_rgb=(label == 'native_teacher01'),
                      max_run_wall_seconds=840 if seconds == 60 else 400,
                      acceptance_measurement=measurement_identity())
        if label != 'native_teacher01':
            config['policy_id'] = 'lingbot_vla2_selected'
        config = validate_config(config)
        path = destination / 'configs' / (label + '.json')
        write(path, config)
        configs.append((label, path, config))
    if configs[0][2]['scene'] != configs[1][2]['scene'] or configs[0][2]['case']['seed'] != configs[1][2]['case']['seed']:
        raise ValueError('Ablation must use the same actual reset request')
    py = str(OWN / 'venv/bin/python')
    output = destination / 'action_probe'
    jobs = [{'purpose': 'Offline real sampled-action and cache equivalence diagnosis; no updates',
             'argv': [py, '-m', 'experiments.lingbot_vla2_g1.diagnostics.action_probe',
                      '--delta', str(OWN / 'training_round02/selected'), '--prefix-manifest',
                      str(SHARED / 'prepared/carry_prefix_native201/manifest.json'), '--output', str(output)],
             'result_file': str(output / 'result.json'), 'timeout_seconds': 650}]
    for label, path, config in configs:
        frozen[str(path)] = sha(path)
        jobs.append({'purpose': label, 'argv': [py, '-m', 'experiments.lingbot_vla2_g1.run_case_pose_speed', '--config', str(path)],
                     'config_path': str(path), 'result_file': str(Path(config['output_dir']) / 'result.json'),
                     'timeout_seconds': 860 if label == 'native_teacher01' else 420})
    manifest = {'schema': 'lingbot_resumed_diagnostic_and_train_collection_v1', 'jobs': jobs,
                'formal_benchmark_trial': False, 'candidate_selected_on_original_validation': True,
                'weight_updates_in_this_turn': 0, 'expected_total_workload_limit_seconds': 2350,
                'finite_GPU_turn_seconds': 3000, 'independent_CPU_audits_after_GPU_release': True}
    manifest_path = destination / 'manifest.json'
    write(manifest_path, manifest)
    frozen[str(manifest_path)] = sha(manifest_path)
    payloads = dict(old_plan['immutable_payloads'])
    # Own code is copied once; immutable assets and all old experiment evidence
    # remain in place under their exact SHA roster and the existing archives.
    archive = destination / 'source_snapshot'
    copied = {}
    for name, digest in frozen.items():
        source = Path(name)
        if source.suffix == '.py' or source.is_relative_to(destination):
            target = archive / name.lstrip('/')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha(target) != digest:
                raise ValueError('Source snapshot changed during copy')
            copied[str(target)] = digest
    write(destination / 'preparation.json', {
        'schema': 'lingbot_resumed_finite_turn_CPU_preparation_v1', 'status': 'prepared_for_root_review',
        'frozen_files': frozen, 'immutable_payloads': payloads,
        'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
        'copied_source_files_sha256': copied,
        'actual_source_snapshot_bytes': sum(Path(n).stat().st_size for n in copied),
        'budget_allocation': allocation, 'resources': cohort['resources'],
        'GPU_initialized': False, 'FIFO_ticket_requested': False})
    print(json.dumps({'prepared': str(destination), 'files': len(frozen), 'source_snapshot_files': len(copied), 'jobs': len(jobs)}), flush=True)


if __name__ == '__main__':
    prepare()
