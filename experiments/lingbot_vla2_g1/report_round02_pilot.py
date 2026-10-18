"""Describe the actual single repeated case without inventing a twenty-case rate."""
import json
from pathlib import Path

from .benchmark_pose_speed_v2 import aggregate
from .data import OWN, sha


def main():
    plan_path = OWN / 'evaluation_plan_round02/plan.json'
    receipt_path = OWN / 'evaluation_segments_round02_pilot01/receipt.json'
    output = OWN / 'round02_pilot_review.json'
    if output.exists():
        raise FileExistsError('Preserve the first completed pilot review')
    receipt = json.loads(receipt_path.read_text())
    training_path = OWN / 'training_round02/training_result.json'
    training = json.loads(training_path.read_text())
    plan = json.loads(plan_path.read_text())
    report = {'schema': 'lingbot_round02_single_case_pilot_review_v1',
              'completed_cases': 0, 'formal_20case_verdict_available': False,
              'full_20case_success_rate': None, 'frozen_cohort_cases': 20,
              'case_requested': plan['jobs'][0]['case_id'],
              'scope': 'One repeated case after seeing round01 outcomes; not a fresh blind cohort or a completed new twenty-case benchmark',
              'checkpoint_selection': 'Original complete validation only, unchanged by this pilot',
              'training_updates': training['optimizer_steps'],
              'selected_step': training['selected_step'],
              'selected_delta': training['selected_delta'],
              'selected_validation': training['selected_validation'],
              'training_source_weights': training['training_source_weights'],
              'acceptance_measurement': plan['acceptance_measurement'],
              'inputs_sha256': {str(p): sha(p) for p in [plan_path, receipt_path, training_path, Path(__file__).absolute()]}}
    if receipt.get('status') != 'completed':
        jobs = receipt.get('jobs', [])
        source_result = Path(plan['jobs'][0]['result_path'])
        if source_result.exists() and json.loads(source_result.read_text()).get('status') == 'completed':
            raise ValueError('A completed source result still needs its independent audit; do not finalize it as unrun')
        report.update(status='not_completed_within_bounded_pilot',
                      physical_task_outcome=None, infrastructure_or_queue_error=receipt.get('error'),
                      source_result_exists=source_result.exists(),
                      noncompleted_records=jobs, missing_cases_are_failures=False)
    else:
        jobs = receipt['jobs']
        if len(jobs) != 1 or jobs[0]['index'] != 0 or jobs[0]['status'] != 'completed':
            raise ValueError('Exactly the first frozen case must have completed')
        raw_path = OWN / 'benchmark_round02_one_case_raw_summary.json'
        raw = aggregate(plan_path, [receipt_path], raw_path)
        case_id = plan['jobs'][0]['case_id']
        unrun = {job['case_id'] for job in plan['jobs'][1:]}
        error_ids = {error.get('case_id') for error in raw['errors']}
        if (raw['completed_evidence_cases'] != 1 or len(raw['episodes']) != 1
                or raw['episodes'][0]['case_id'] != case_id or error_ids != unrun
                or len(raw['errors']) != 19 or not raw['interface_verification']['passed']):
            raise ValueError('Completed pilot evidence failed or errors extend beyond the nineteen unrun cases')
        baseline_path = OWN / 'benchmark_pose_speed_summary_v2.json'
        baseline = json.loads(baseline_path.read_text())
        previous = next(row for row in baseline['episodes'] if row['case_id'] == case_id)
        current = raw['episodes'][0]
        report.update(status='completed_single_case_pilot', completed_cases=1,
                      independent_audit_passed=True, physical_task_outcome=current['success'],
                      task_success_count=int(current['success']),
                      qualified_long_carry_count=int(current['long_carry_qualified']),
                      grasp_event_count=int(current['grasped']),
                      episode=current, round01_same_case=previous,
                      actual_initial_kinematics_equal_round01=current['initial_physical_state_sha256'] == previous['initial_physical_state_sha256'],
                      actual_full_interface_equal_round01=current['frozen_interface_sha256'] == previous['frozen_interface_sha256'],
                      actual_nonposition_interface_equal_round01=raw['interface_verification']['common_nonposition_interface_sha256'] == baseline['interface_verification']['common_nonposition_interface_sha256'],
                      warm_inference_seconds=raw['warm_inference_seconds'],
                      raw_summary=str(raw_path), nineteen_unrun_cases_not_failures=sorted(unrun))
        report['inputs_sha256'].update({str(raw_path): sha(raw_path), str(baseline_path): sha(baseline_path)})
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: report.get(key) for key in ['status', 'completed_cases',
          'task_success_count', 'qualified_long_carry_count', 'grasp_event_count']}))


if __name__ == '__main__':
    main()
