"""Aggregate the formal cohort through the existing benchmark, then apply the sixteen-success target."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .gates import apply_target, assert_result_destination, load_formal_plan


def pointmass_decoder(config, result):
    from experiments.lingbot_vla2_g1.diagnostics.development_audit_v2 import decoded_evidence
    return decoded_evidence(config, result)


def aggregate(plan_path, receipt_paths, output):
    plan_path, output = Path(plan_path).absolute(), Path(output).absolute()
    receipts = [Path(path).absolute() for path in receipt_paths]
    assert_result_destination(output)
    documents = [json.loads(path.read_text()) for path in receipts]
    import experiments.lingbot_vla2_g1.benchmark_pose_speed as benchmark
    saved = (benchmark.load_plan, benchmark.decoded_evidence)
    try:
        benchmark.load_plan = load_formal_plan
        benchmark.decoded_evidence = pointmass_decoder
        report = benchmark.aggregate(plan_path, receipts, output)
    finally:
        benchmark.load_plan, benchmark.decoded_evidence = saved
    report = apply_target(report, documents)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.incomplete')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.plan, args.receipt, args.output)
    print(json.dumps({'status': report['status'], 'target_success_count': report['target_success_count'],
                      'goal_performance_passed': report['goal_performance_passed'],
                      'actual_robot_success_claimed': report['actual_robot_success_claimed']}))


if __name__ == '__main__':
    main()
