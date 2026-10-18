"""Summarize paired physical evaluation without treating missing runs as failures."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
import numpy as np

METRICS = ('success', 'grasped', 'stable_placement', 'dropped', 'fell', 'timeout')
REQUIRED_ACCEPTANCE_VERSION = 'pi05_g1_placement_v2'


def current_acceptance_identity():
    # Loading the CPU-only sibling works both as a package and a direct CLI script.
    spec = importlib.util.spec_from_file_location('pi05_g1_frozen_placement', Path(__file__).with_name('placement.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.acceptance_identity()


def wilson(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    p = successes / count
    d = 1 + z*z/count
    center = (p + z*z/(2*count))/d
    half = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/d
    return [max(0., center-half), min(1., center+half)]


def paired_statistics(baseline, finetuned, seed=42, samples=20000):
    a, b = np.asarray(baseline, dtype=float), np.asarray(finetuned, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError('Nonempty paired binary vectors required')
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('Finite binary outcomes required')
    diff = b-a
    estimates = diff[np.random.default_rng(seed).integers(0, len(a), (samples, len(a)))].mean(1)
    gains = int(((a == 0) & (b == 1)).sum())
    losses = int(((a == 1) & (b == 0)).sum())
    discordant = gains+losses
    p = min(1., 2*sum(math.comb(discordant, i) for i in range(min(gains, losses)+1))/2**discordant) if discordant else 1.
    return {'paired_count': len(a), 'success_rate_difference': float(diff.mean()),
            'paired_bootstrap_95ci': np.quantile(estimates, [.025, .975]).tolist(),
            'bootstrap_seed': seed, 'bootstrap_samples': samples,
            'gains': gains, 'losses': losses, 'exact_mcnemar_p': p}


def clustered_statistics(baseline, finetuned, positions, seed=42, samples=20000):
    """Resample physical position clusters, retaining paired noise repeats inside each."""
    groups = {}
    for a, b, position in zip(baseline, finetuned, positions):
        groups.setdefault(tuple(position), []).append(float(b) - float(a))
    keys = sorted(groups)
    if not keys:
        return None
    # Positions have equal prespecified repetitions. Preserve observed sizes if partial.
    totals = np.asarray([sum(groups[key]) for key in keys])
    counts = np.asarray([len(groups[key]) for key in keys])
    selected = np.random.default_rng(seed).integers(0, len(keys), (samples, len(keys)))
    estimates = totals[selected].sum(1) / counts[selected].sum(1)
    return {'position_clusters': len(keys), 'paired_trials': int(counts.sum()),
            'trials_per_position': [{'apple_xy_offset_m': list(key), 'count': len(groups[key])} for key in keys],
            'success_rate_difference': float(totals.sum() / counts.sum()),
            'position_cluster_bootstrap_95ci': np.quantile(estimates, [.025, .975]).tolist(),
            'bootstrap_seed': seed, 'bootstrap_samples': samples,
            'interpretation': 'Five physical positions with repeated policy-noise trials; not 20 independent scenes.'}


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite result value')
    if isinstance(value, dict):
        for v in value.values():
            _finite(v)
    if isinstance(value, list):
        for v in value:
            _finite(v)


def distribution(values):
    return {'count': len(values), 'mean': float(np.mean(values)),
            'p50': float(np.quantile(values, .5)), 'p95': float(np.quantile(values, .95)),
            'maximum': float(max(values))} if values else None


def timing_summary(table):
    durations = {key: [] for key in ('source_candidate_sim_seconds', 'task_completion_sim_seconds_success_only',
                 'episode_sim_seconds', 'time_to_first_grasp_sim_seconds', 'launch_wall_seconds')}
    calls = [(row, call) for row in table.values() for call in row.get('inference_calls', [])]
    per_case = {}
    for case, row in table.items():
        hz = float(row.get('control_hz', 50))
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError('control_hz must be positive and finite')
        values = {}
        for key, tick_key in (('source_candidate_sim_seconds', 'source_success_candidate_tick'),
                              ('episode_sim_seconds', 'completed_control_ticks'),
                              ('time_to_first_grasp_sim_seconds', 'first_grasp_control_tick')):
            tick = row.get(tick_key)
            if tick is not None:
                if not isinstance(tick, (int, float)) or not math.isfinite(tick) or tick < 0:
                    raise ValueError(f'Invalid {tick_key}')
                values[key] = tick / hz
        if row['success']:
            window = row.get('qualified_stability_window')
            if not isinstance(window, dict) or window.get('passed') is not True or not isinstance(window.get('end_tick'), int) or window['end_tick'] < 0:
                raise ValueError('Successful trial requires its passed qualified stability-window end_tick')
            values['task_completion_sim_seconds_success_only'] = window['end_tick'] / hz
        if row.get('launch_wall_seconds') is not None:
            wall = row['launch_wall_seconds']
            if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0:
                raise ValueError('Invalid launch elapsed_seconds')
            values['launch_wall_seconds'] = wall
        per_case[case] = {**values, 'launch_wall_receipt': row.get('launch_wall_receipt'),
                          'result_source_path': row.get('result_source_path')}
        for key, value in values.items():
            durations[key].append(value)
    out = {key: distribution(value) for key, value in durations.items()}
    out['per_case_timing'] = per_case
    for field, output in (('inference_seconds', 'inference_seconds'),
                          ('simulation_paused_wall_seconds', 'simulation_pause_wall_seconds')):
        selected = [(call, float(call[field])) for _, call in calls if field in call]
        if any(not math.isfinite(value) or value < 0 for _, value in selected):
            raise ValueError(f'Invalid {field}')
        out[output] = distribution([value for _, value in selected])
        out[output + '_cold_tick0'] = distribution([value for call, value in selected if call.get('observation_control_tick') == 0])
        out[output + '_warm_ticks'] = distribution([value for call, value in selected if isinstance(call.get('observation_control_tick'), int) and call['observation_control_tick'] > 0])
    out['total_inference_pause_wall_seconds'] = sum(float(call['simulation_paused_wall_seconds']) for _, call in calls if 'simulation_paused_wall_seconds' in call)
    out['timing_definitions'] = {
        'task_completion_sim_seconds_success_only': 'Only successful trials: qualified_stability_window.end_tick / control_hz, including the required sampled stability hold.',
        'source_candidate_sim_seconds': 'Source candidate tick diagnostic for successful or unsuccessful trials; not task completion.',
        'episode_sim_seconds': 'Every observed completed trial: completed_control_ticks / control_hz.',
        'time_to_first_grasp_sim_seconds': 'First measured grasp proxy tick / control_hz; missing grasp has no invented time.',
        'launch_wall_seconds': 'Matched launch receipt elapsed_seconds: startup, physical stepping, model loading/inference and launcher cleanup; no task-only wall time is inferred.',
        'cold_warm': 'Cold = inference observation_control_tick 0; warm = observation_control_tick > 0. Calls with missing tick are included only in the all-call distribution.',
        'simulation_pause_wall_seconds': 'Measured synchronous simulation pause per inference call, including model IPC overhead.'}
    return out


def load_result_files(locations):
    """Discover only actual result.json files; attach exactly matched sibling receipts."""
    rows, seen = [], set()
    for location in map(Path, locations):
        files = sorted(location.rglob('result.json')) if location.is_dir() else [location]
        for file in files:
            file = file.resolve()
            if file in seen:
                continue
            seen.add(file)
            loaded = json.loads(file.read_text())
            records = loaded if isinstance(loaded, list) else loaded.get('results', [loaded])
            for record in records:
                row = dict(record)
                if row.get('video_path') and not Path(row['video_path']).is_absolute():
                    row['video_path'] = str((file.parent / row['video_path']).resolve())
                case = row.get('case_id')
                if isinstance(case, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', case) and file.name == 'result.json' and file.parent.name == case:
                    receipt_path = file.parent.parent / 'configs' / (case + '.launch.json')
                    if receipt_path.is_file():
                        receipt = json.loads(receipt_path.read_text())
                        if receipt.get('case_id') != case or receipt.get('policy') != row.get('policy'):
                            raise ValueError(f'Launch receipt case/policy mismatch: {receipt_path}')
                        _finite(receipt)
                        elapsed = receipt.get('elapsed_seconds')
                        if not isinstance(elapsed, (int, float)) or elapsed < 0:
                            raise ValueError(f'Launch receipt missing valid elapsed_seconds: {receipt_path}')
                        row['launch_wall_seconds'] = elapsed
                        row['launch_wall_receipt'] = str(receipt_path)
                row['result_source_path'] = str(file)
                rows.append(row)
    return rows


def summarize(cases, results, gates=None):
    ids = [str(case['case_id']) for case in cases]
    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError('Exactly 20 unique frozen case IDs required')
    indexed = {policy: {} for policy in ('baseline', 'finetuned')}
    errors, blocked = [], []
    for result in results:
        _finite(result)
        policy, case = result.get('policy'), str(result.get('case_id'))
        if policy == 'replay':
            continue
        if policy not in indexed or case not in ids:
            errors.append(f'unknown policy/case: {policy}/{case}')
            continue
        if case in indexed[policy]:
            raise ValueError(f'duplicate result: {policy}/{case}')
        if result.get('status') == 'blocked' or result.get('success') is None:
            blocked.append({'policy': policy, 'case_id': case, 'reason': result.get('reason', result.get('error'))})
            continue
        if any(type(result.get(key)) is not bool for key in METRICS):
            raise ValueError(f'{policy}/{case}: all outcome flags must be explicit booleans')
        if result['success'] and (not result['grasped'] or not result['stable_placement']
                                  or result['fell'] or result['dropped'] or result['timeout']):
            raise ValueError(f'{policy}/{case}: success contradicts physical outcome flags')
        expected_case = next(item for item in cases if str(item['case_id']) == case)
        actual_case = result.get('case', {})
        if 'seed' in expected_case and actual_case.get('seed') != expected_case['seed']:
            errors.append(f'frozen seed mismatch: {policy}/{case}')
        expected_offset = expected_case.get('apple_offset_xy_m', expected_case.get('apple_xy_offset_m'))
        actual_offset = actual_case.get('apple_offset_xy_m', actual_case.get('apple_xy_offset_m'))
        if expected_offset is not None and actual_offset != expected_offset:
            errors.append(f'frozen apple offset mismatch: {policy}/{case}')
        indexed[policy][case] = result
    summaries = {}
    evidence_missing = []
    for policy, table in indexed.items():
        counts = {key: sum(r[key] for r in table.values()) for key in METRICS}
        for case, row in table.items():
            video = row.get('video_path')
            if not video or not Path(video).is_file():
                evidence_missing.append({'policy': policy, 'case_id': case, 'video_path': video})
        summaries[policy] = {**timing_summary(table),
                             'real_time_1x_verified': bool(table) and all(row.get('real_time_1x_verified') is True for row in table.values()),
                             'completed': len(table), 'expected': 20,
                             'missing_cases': [case for case in ids if case not in table],
                             'counts': counts, 'success_wilson_95ci': wilson(counts['success'], len(table)),
                             'success_rate_observed': counts['success']/len(table) if table else None,
                             'meets_16_of_20': len(table) == 20 and counts['success'] >= 16}
    observed_pairs = [case for case in ids if case in indexed['baseline'] and case in indexed['finetuned']]
    paired, pair_eligibility = [], []
    for case in ids:
        baseline, finetuned = indexed['baseline'].get(case), indexed['finetuned'].get(case)
        pair = {'case_id': case, 'eligible': False}
        if baseline is None or finetuned is None:
            pair.update(status='missing_result', missing_policies=[policy for policy in indexed if case not in indexed[policy]])
        else:
            hashes = {policy: indexed[policy][case].get('initial_physical_state_sha256') for policy in indexed}
            invalid = [policy for policy, value in hashes.items() if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)]
            pair['initial_physical_state_sha256'] = hashes
            if invalid:
                pair.update(status='missing_or_invalid_initial_state_hash', missing_policies=invalid)
                errors.append(f'{case}: missing valid initial_physical_state_sha256 for ' + ', '.join(invalid))
            elif hashes['baseline'] != hashes['finetuned']:
                pair['status'] = 'initial_physical_state_mismatch'
                errors.append(f'{case}: baseline/finetuned initial physical state hashes differ')
            else:
                pair.update(status='eligible', eligible=True)
                paired.append(case)
        pair_eligibility.append(pair)
    stats = paired_statistics([indexed['baseline'][i]['success'] for i in paired],
                              [indexed['finetuned'][i]['success'] for i in paired]) if paired else None
    positions = [next(item for item in cases if str(item['case_id']) == case).get('apple_offset_xy_m',
                      next(item for item in cases if str(item['case_id']) == case).get('apple_xy_offset_m')) for case in paired]
    cluster_stats = clustered_statistics([indexed['baseline'][i]['success'] for i in paired],
                     [indexed['finetuned'][i]['success'] for i in paired], positions) if paired and all(p is not None for p in positions) else None
    physical_hashes = {row['initial_physical_state_sha256'] for table in indexed.values() for row in table.values()
                       if isinstance(row.get('initial_physical_state_sha256'), str)
                       and re.fullmatch(r'[0-9a-f]{64}', row['initial_physical_state_sha256'])}
    gates = gates or {}
    gate_status = {key: gates.get(key) is True for key in ('source_interface_passed', 'replay_passed')}
    # A failed per-run gate cannot be overridden by a global pass.
    for table in indexed.values():
        for row in table.values():
            for key in gate_status:
                if row.get(key) is not True:
                    gate_status[key] = False
    fingerprints = {row['frozen_interface_sha256'] for table in indexed.values()
                    for row in table.values() if row.get('frozen_interface_sha256')}
    if len(fingerprints) > 1:
        errors.append('baseline/finetuned frozen interface fingerprints differ')
    acceptance_bindings = set()
    expected_acceptance = current_acceptance_identity()
    if gates.get('acceptance_sha256') != expected_acceptance['acceptance_sha256']:
        errors.append('replay gate acceptance hash differs from the frozen local measurement module')
    for key in ('acceptance_sha256', 'frozen_interface_sha256'):
        if not isinstance(gates.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', gates[key]):
            errors.append(f'replay gate missing valid {key}')
    if gates.get('acceptance_version') != REQUIRED_ACCEPTANCE_VERSION:
        errors.append('replay gate acceptance version is not the frozen v2 standard')
    for policy, table in indexed.items():
        for case, row in table.items():
            acceptance_bindings.add((row.get('acceptance_version'), row.get('acceptance_sha256')))
            for key in ('acceptance_version', 'acceptance_sha256', 'frozen_interface_sha256'):
                if row.get(key) != gates.get(key) or row.get(key) is None:
                    errors.append(f'{policy}/{case}: {key} differs from verified replay gate')
    acceptance_identity = {'required_version': REQUIRED_ACCEPTANCE_VERSION,
                           'required_sha256': expected_acceptance['acceptance_sha256'],
                           'gate_version': gates.get('acceptance_version'),
                           'gate_sha256': gates.get('acceptance_sha256'),
                           'observed_bindings': [{'version': version, 'sha256': digest} for version, digest in sorted(acceptance_bindings, key=str)]}
    complete = len(paired) == 20 and not errors
    evidence_complete = complete and not evidence_missing
    eligible = evidence_complete and all(gate_status.values())
    improvement_ci = cluster_stats['position_cluster_bootstrap_95ci'] if cluster_stats else (stats['paired_bootstrap_95ci'] if stats else None)
    improved = bool(eligible and improvement_ci[0] > 0 and stats['exact_mcnemar_p'] < .05)
    return {'policies': summaries, 'observed_case_pair_count': len(observed_pairs),
            'eligible_physical_pair_count': len(paired), 'pair_eligibility': pair_eligibility, 'paired_statistics': stats, 'position_cluster_statistics': cluster_stats,
            'conditional_trial_statistics_notice': 'Paired 20-trial CI and McNemar describe the frozen repetitions; physical-position clustering limits scene generalization.',
            'distinct_initial_physical_state_hashes': len(physical_hashes) if physical_hashes else None,
            'initial_state_hashes_notice': 'Distinct measured byte hashes do not establish statistical independence; use the five position clusters for generalization uncertainty.',
            'complete': complete,
            'evidence_complete': evidence_complete, 'gates': gate_status,
            'frozen_interface_fingerprints': sorted(fingerprints), 'acceptance_identity': acceptance_identity,
            'missing_video_evidence': evidence_missing, 'blocked_runs': blocked, 'errors': errors,
            'training_benefit_proven': improved,
            'training_verdict': 'improvement_supported' if improved else 'not_established',
            'task_usable': eligible and summaries['finetuned']['meets_16_of_20'],
            'task_verdict': ('usable_in_simulation' if eligible and summaries['finetuned']['meets_16_of_20']
                             else 'incomplete_evidence' if not evidence_complete else 'not_qualified'),
            'scope': 'simulation only; real robot performance unverified'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--results', type=Path, nargs='+', required=True)
    parser.add_argument('--gates', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    if isinstance(cases, dict):
        cases = cases['cases']
    results = load_result_files(args.results)
    gates = json.loads(args.gates.read_text()) if args.gates else None
    report = summarize(cases, results, gates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ('complete', 'training_verdict', 'task_verdict')}))
