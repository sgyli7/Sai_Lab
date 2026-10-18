"""Finite development validation, with all thresholds from the common task.

Only an offline decision to launch a new independent episode uses acceptance
truth. Every action inside an episode still comes from RGB/q/text-only LingBot.
An unrun full case has no score. A completed screen never completes the goal.
"""
import argparse
import json
from pathlib import Path
import os
import shutil
import subprocess
import time
import numpy as np

from .data import OWN, ROOT, sha
from .diagnostics.development_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.absolute().is_relative_to(OWN):
        raise FileExistsError('Fresh owned development execution output required')
    plan = json.loads(args.plan.read_text())
    if plan.get('schema') != 'lingbot_task_v4_development_plan_v1' or len(plan['jobs']) != 2:
        raise ValueError('Exact predeclared screen/full-case pair required')
    for name, digest in plan['frozen_files'].items():
        if sha(name) != digest:
            raise ValueError('Frozen development input changed: ' + name)
    started = time.monotonic()
    args.output.mkdir(parents=True)
    report = {'schema': 'lingbot_task_v4_development_execution_v1', 'status': 'running',
              'plan_sha256': sha(args.plan), 'jobs': [], 'formal20_benchmark': False,
              'full_task_case_ran': False, 'full_task_case_success': None,
              'overall_robot_task_goal_completed': False}
    promote = False
    try:
        for index, job in enumerate(plan['jobs']):
            if index == 1 and not promote:
                report['jobs'].append({'index': index, 'case_id': job['case_id'],
                    'status': 'not_run_grasp_screen_failed', 'full_task_score': None})
                break
            if shutil.disk_usage(OWN).free < 25 * 1024 ** 3:
                raise RuntimeError('Keep24GiB global free plus1GiB next-case reserve')
            config_path = Path(job['config_path'])
            config = json.loads(config_path.read_text())
            if sha(config_path) != job['config_sha256'] or Path(config['output_dir']).exists():
                raise ValueError('New unchanged development case required')
            print(json.dumps({'event': 'development_case_started', 'index': index,
                              'case_id': job['case_id']}), flush=True)
            log = args.output / f'case_{index:02}_stdout.log'
            with log.open('x') as stream:
                code = subprocess.run([str(OWN / 'venv/bin/python'), '-m', 'experiments.lingbot_vla2_g1.run_case_pose_speed',
                                       '--config', str(config_path)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                      timeout=job['timeout_seconds'], check=False,
                                      env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')).returncode
            if code:
                raise RuntimeError('Development simulator process failed: ' + str(code))
            folder = Path(config['output_dir'])
            with (folder / 'trajectory.jsonl').open() as stream:
                initial = json.loads(next(stream))
            for actor, name in (('brown_box', 'box_pose'), ('blue_sorting_bin', 'bin_pose')):
                actual = np.asarray(initial['acceptance_truth_only'][actor]['pose'])
                expected = config['scene'][name]
                if not np.allclose(actual[:3], expected['position_xyz'], rtol=0, atol=1e-6):
                    raise ValueError('Actual development initial position differs: ' + actor)
                if abs(float(actual[3:] @ np.asarray(expected['rotation_xyzw']))) < 1 - 1e-6:
                    raise ValueError('Actual development initial orientation differs: ' + actor)
            checked = audit(config_path)
            audit_path = args.output / f'case_{index:02}_independent_audit.json'
            with audit_path.open('x') as stream:
                json.dump(checked, stream, indent=2, allow_nan=False)
            if checked.get('passed') is not True or checked.get('autonomous_policy_trial') is not True:
                raise RuntimeError('Independent development evidence audit failed: ' + repr(checked.get('anomalies')))
            value = checked['recomputed_acceptance']
            record = {'index': index, 'case_id': job['case_id'], 'status': 'completed',
                      'result_file': job['result_file'], 'result_sha256': sha(job['result_file']),
                      'audit': str(audit_path), 'audit_sha256': sha(audit_path),
                      'physical_outcome': value, 'actual_initial_scene_checked': True,
                      'raw_files_sha256': {str(p): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}}
            report['jobs'].append(record)
            if index == 0:
                promote = bool(value['grasped'] and not value['fell'] and not value['dropped'])
                report['grasp_screen_passed'] = promote
            else:
                report.update(full_task_case_ran=True, full_task_case_success=value['success'])
            print(json.dumps({'event': 'development_case_checked', 'index': index,
                              'grasped': value['grasped'], 'physical_full_success': value['success'],
                              'next_full_case_eligible': promote}), flush=True)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        with (args.output / 'result.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
