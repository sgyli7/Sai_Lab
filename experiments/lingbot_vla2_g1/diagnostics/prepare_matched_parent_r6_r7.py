"""Freeze a matched R6/R7 comparison after the actual R7 development failure.

Only the checkpoint and evidence destinations change. Both autonomous episodes
reuse the exact R7 reset requests, sensors, controller and acceptance settings.
This preparation neither trains a model nor claims a complete task success.
"""
import copy
import datetime
import json
from pathlib import Path

from experiments.lingbot_vla2_g1.data import OWN, ROOT, sha
from experiments.lingbot_vla2_g1.make_plan import experiment_sources, write
from experiments.lingbot_vla2_g1.round06_selection import admit
from experiments.vla_resources.gpu_queue import cleanup_confirmed
from experiments.pi05_g1_mobile.simulator import validate_config


def prepare():
    output = OWN / 'matched_parent_R6_against_R7_plan'
    if output.exists():
        raise FileExistsError('Preserve all prior comparison evidence')
    reference = OWN / 'round07_development_plan/plan.json'
    result_path = OWN / 'round07_development_preparation/execution/result.json'
    original, actual = [json.loads(p.read_text()) for p in (reference, result_path)]
    if (actual.get('synthetic') is True or actual.get('status') != 'completed'
            or actual.get('plan_sha256') != sha(reference)
            or actual.get('full_task_case_ran') is not True
            or actual.get('full_task_case_success') is not False
            or actual.get('grasp_screen_passed') is not True
            or len(actual.get('jobs', [])) != 2):
        raise ValueError('Actual audited R7 full-task failure required')
    frozen = dict(original['frozen_files'])
    for record in actual['jobs']:
        audit_path = Path(record['audit'])
        checked = json.loads(audit_path.read_text())
        if (sha(audit_path) != record['audit_sha256'] or checked.get('passed') is not True
                or checked.get('autonomous_policy_trial') is not True
                or checked.get('synthetic') is True or len(checked['checks']) != 7
                or not all(c.get('passed') is True for c in checked['checks'].values())
                or record.get('actual_initial_scene_checked') is not True):
            raise ValueError('Both actual R7 cases need all seven independent checks')
        frozen[str(audit_path)] = sha(audit_path)
        if sha(record['result_file']) != record['result_sha256']:
            raise ValueError('Actual R7 result changed')
        frozen[record['result_file']] = record['result_sha256']
    queue = json.loads((ROOT / '.scratch/vla_gpu_coordination/queue.json').read_text())
    ticket = next(t for t in queue['tickets'] if t['ticket'] == 't000094')
    row = json.loads((OWN / 'gpu_budget.json').read_text())['runs'][ticket['ledger_runs_before']]
    if ticket['status'] != 'completed' or not ticket.get('cleanup_confirmed') or not cleanup_confirmed(row):
        raise ValueError('Actual R7 development and owned cleanup must close first')
    training_path = OWN / 'training_task_round06/training_result.json'
    training, delta, parent_frozen, payloads, provenance = admit(training_path)
    identity = payloads[str(delta / 'trainable.safetensors')]
    if identity['sha256'] != 'f7403098fff697a034022bbbb00369f9f3cf065fd953bbcdc9fbf0e2dbd8b8c5':
        raise ValueError('Exact unchanged R6 parent checkpoint required')
    for name, digest in parent_frozen.items():
        if name in frozen and frozen[name] != digest:
            raise ValueError('Historical source identities conflict: ' + name)
        frozen[name] = digest
    for path in experiment_sources([Path(__file__), ROOT / 'experiments/lingbot_vla2_g1/run_development_task_v6.py',
                                    ROOT / 'experiments/lingbot_vla2_g1/prepare_development_round07.py']):
        frozen[str(path)] = sha(path)
    frozen.update({str(reference): sha(reference), str(result_path): sha(result_path)})
    for name, digest in frozen.items():
        if sha(name) != digest:
            raise ValueError('Frozen comparison input changed: ' + name)
    output.mkdir()
    write(output / 'training_provenance.json', provenance)
    frozen[str(output / 'training_provenance.json')] = sha(output / 'training_provenance.json')
    jobs, comparisons = [], []
    for old_job in original['jobs']:
        old_path = Path(old_job['config_path'])
        if sha(old_path) != old_job['config_sha256']:
            raise ValueError('Reference R7 case configuration changed')
        old = json.loads(old_path.read_text())
        config = copy.deepcopy(old)
        folder = output / 'episodes' / old_job['case_id']
        config['output_dir'] = str(folder)
        command = list(old['policy_command'])
        command[command.index('--delta') + 1] = str(delta)
        command[command.index('--decoded-evidence') + 1] = str(folder / 'decoded.jsonl')
        config['policy_command'] = command
        config['HFmodel']['delta'] = str(delta)
        config['HFmodel']['selected_by_validation'] = training['selected_step'] == training['physical_candidate_step']
        config['frozen_files'] = dict(frozen)
        config = validate_config(config)
        changed = {k for k in set(old) | set(config) if old.get(k) != config.get(k)}
        if changed != {'output_dir', 'policy_command', 'HFmodel', 'frozen_files'}:
            raise ValueError('A matched comparison may change only checkpoint and evidence destinations')
        expected_model = dict(old['HFmodel'], delta=str(delta))
        if config['HFmodel'] != expected_model:
            raise ValueError('Sampling or model interpretation changed')
        config_path = output / 'configs' / old_path.name
        write(config_path, config)
        jobs.append({**old_job, 'config_path': str(config_path), 'config_sha256': sha(config_path),
                     'result_file': str(folder / 'result.json')})
        comparisons.append({'case_id': old_job['case_id'], 'reference_config': str(old_path),
                            'reference_config_sha256': sha(old_path), 'parent_config': str(config_path),
                            'parent_config_sha256': sha(config_path), 'changed_top_level_keys': sorted(changed),
                            'all_other_configuration_values_exactly_equal': True,
                            'same_seed_scene_camera_actions_controller_and_acceptance': True})
        frozen[str(config_path)] = sha(config_path)
    pairing = {'schema': 'lingbot_actual_R6_R7_matched_development_CPU_pairing_v1',
               'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'reference_actual_R7_result': str(result_path), 'reference_actual_R7_result_sha256': sha(result_path),
               'reference_checkpoint_sha256': 'a0506cf616d2a05c7267a2794c2a408063dd96042c827023d19b4581b9b4d36d',
               'parent_checkpoint_sha256': identity['sha256'], 'comparisons': comparisons,
               'actual_configuration_pairing_passed': True, 'synthetic': False,
               'new_parent_robot_result_available': False, 'goal_completed': False}
    pairing_path = output / 'actual_CPU_pairing.json'
    write(pairing_path, pairing)
    frozen[str(pairing_path)] = sha(pairing_path)
    plan = copy.deepcopy(original)
    plan.update(frozen_files=frozen, immutable_payloads=payloads, jobs=jobs,
                training_result=str(training_path), selected_delta=str(delta),
                selected_new_step=training['physical_candidate_step'],
                deployed_total_updates=training['model_weight_lineage_optimizer_updates'],
                matched_parent_comparison=True, reference_actual_R7_result=str(result_path),
                pairing_receipt=str(pairing_path), formal20_benchmark=False)
    write(output / 'plan.json', plan)
    print(json.dumps({'plan': str(output / 'plan.json'), 'sha256': sha(output / 'plan.json'),
                      'actual_pairing_passed': True, 'checkpoint': identity['sha256']}), flush=True)


if __name__ == '__main__':
    prepare()
