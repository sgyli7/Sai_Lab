"""Amend an unexecuted immutable plan to the jointly qualified measurement."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from experiments.pi05_g1_mobile.benchmark import read
from experiments.pi05_g1_mobile.simulator import validate_config
from .benchmark import load_plan as load_previous_plan
from .acceptance_context import admit_qualification, measurement_identity
from .data import OWN, ROOT, sha
from .make_plan import artifact, experiment_sources, write


def make_plan(previous_plan, qualification_path, output, episodes_root):
    previous_plan = Path(previous_plan).absolute()
    output, episodes_root = Path(output).absolute(), Path(episodes_root).absolute()
    if output.exists() or episodes_root.exists() or not output.is_relative_to(OWN) or not episodes_root.is_relative_to(OWN):
        raise ValueError('Fresh owned plan and raw episode roots required')
    previous = load_previous_plan(previous_plan)
    if Path(previous['episode_root']).exists() or any(Path(job['result_path']).exists() for job in previous['jobs']):
        raise ValueError('Measurement amendment requires zero previously executed formal trials')
    qualified, qualification_files = admit_qualification(qualification_path)
    identity = measurement_identity()
    for path, expected in previous['immutable_payloads'].items():
        if artifact(path, expected['sha256'])['bytes'] != expected['bytes']:
            raise ValueError('Selected model payload changed')
    prior_configs = {job['config_path'] for job in previous['jobs']}
    frozen = {name: expected for name, expected in previous['frozen_files'].items() if name not in prior_configs}
    frozen[str(previous_plan)] = sha(previous_plan)
    frozen.update(qualification_files)
    starts = [ROOT / 'experiments/lingbot_vla2_g1' / (name + '.py') for name in ['make_plan_pose_speed', 'evaluate_pose_speed', 'benchmark_pose_speed', 'run_case_pose_speed', 'sim_launcher_pose_speed', 'acceptance_context']]
    starts += [ROOT / 'experiments/pi05_g1_mobile' / (name + '.py') for name in ['acceptance_pose_speed', 'acceptance_body_ids', 'sim_launcher_pose_speed']]
    for path in experiment_sources(starts):
        frozen[str(path)] = sha(path)
    output.mkdir(parents=True)
    selection = read(previous['selection_receipt'])
    selection.update(acceptance_measurement=identity, shared_placement_qualification=qualified, previous_unexecuted_plan={'path': str(previous_plan), 'sha256': sha(previous_plan), 'formal_trials': 0}, model_selection_unchanged=True)
    selection_path = output / 'selection_receipt.json'
    write(selection_path, selection)
    frozen[str(selection_path)] = sha(selection_path)
    jobs = []
    for old_job in previous['jobs']:
        config = read(old_job['config_path'])
        folder = episodes_root / old_job['case_id']
        command = list(config['policy_command'])
        command[command.index('--decoded-evidence') + 1] = str(folder / 'decoded.jsonl')
        config.update(output_dir=str(folder), policy_command=command, acceptance_measurement=identity, frozen_files=dict(frozen))
        config = validate_config(config)
        path = output / 'configs' / (old_job['case_id'] + '.json')
        write(path, config)
        jobs.append({**old_job, 'config_path': str(path), 'config_sha256': sha(path), 'result_path': str(folder / 'result.json')})
    frozen.update({job['config_path']: job['config_sha256'] for job in jobs})
    plan = copy.deepcopy(previous)
    plan.update(schema='lingbot_vla2_g1_pose_speed_benchmark_plan_v1', jobs=jobs, frozen_files=frozen, selection_receipt=str(selection_path), episode_root=str(episodes_root), acceptance_measurement=identity, shared_placement_qualification=qualified, previous_unexecuted_plan={'path': str(previous_plan), 'sha256': sha(previous_plan), 'formal_trials': 0}, change_scope='Versioned evaluator observation semantics/body-name resolution only. Same selected weights, robot/controller/physics, scene/cameras, thresholds,20cases,50/200Hz and50tick chunks. No formal outcomes read.')
    write(output / 'plan.json', plan)
    for name, expected in frozen.items():
        path = Path(name)
        if path.suffix not in ['.py', '.json', '.usda', '.yaml'] or path.stat().st_size > 20 * 1024**2:
            continue
        target = output / 'source_snapshot' / name.lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target) != expected:
            raise ValueError('Source changed during qualified plan creation')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-plan', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes-root', type=Path, required=True)
    args = parser.parse_args()
    plan = make_plan(args.previous_plan, args.qualification, args.output, args.episodes_root)
    print(json.dumps({'schema': plan['schema'], 'cases': len(plan['jobs']), 'frozen_files': len(plan['frozen_files']), 'measurement': plan['acceptance_measurement']['schema']}))


if __name__ == '__main__':
    main()
