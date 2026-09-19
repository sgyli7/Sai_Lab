"""One final, fixed-seed evaluation after the candidate bundle is frozen."""
import argparse, hashlib, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from .tasks import TASKS, BASELINE
from .evaluate import run_suite
from .roller_evaluate import run_suite as roller_suite
from .play_sequence import run as play_sequence
from .keyboard_walk import run as keyboard_walk
from .bundle import bundle_paths


def suite_job(job):
    skill, label, source, backend, seeds, workers, dest = job
    task = TASKS[skill]
    if skill == 'roller':
        result = roller_suite(skill, source, backend, seeds, workers, dest,
                              reference_profile='source_play' if label == 'source_mujoco' else 'xml')
    else:
        scene = 'microduck_ball_stand_fix' if task.robot != 'microduck_roller' else None
        result = run_suite(skill, source, backend, seeds, workers, dest,
                           entry='both', scene_robot=scene,
                           reference_profile='source_play' if skill == 'roller_crouch' and label == 'source_mujoco' else 'xml')
    short = {k: v for k, v in result.items() if k != 'episodes'}
    short.update(label=label, episodes=len(result['episodes']))
    return short


def run(bundle, workers=6, seeds=range(1000, 1030), suite_processes=1, reference_bundle=None, out=None):
    bundle = Path(bundle)
    frozen = json.loads((bundle / 'bundle.json').read_text())
    if not frozen.get('verification_completed_unix'): raise ValueError('Bundle is not verified')
    bank = bundle_paths(bundle)
    reference_bank = None if reference_bundle is None else bundle_paths(reference_bundle)
    missing = [str(BASELINE / name) for task in TASKS.values()
               for name in (task.factory, task.previous) if not (BASELINE / name).is_file()]
    if missing:
        raise FileNotFoundError('Full A/B requires setup init --require-previous; missing: ' + ', '.join(missing))
    out = Path(out) if out is not None else bundle / 'holdout'
    out.mkdir(parents=True, exist_ok=False)
    seeds = list(seeds)
    plan = dict(seeds=seeds, started_unix=time.time(), bundle_frozen_unix=frozen['frozen_unix'],
                selection_changes_allowed=False, normal_scene='microduck_ball_stand_fix',
                labels=['candidate', 'source_mujoco', 'previous_godot'],
                source_roller_profile='source_play', godot_roller_profile='xml',
                suite_processes=suite_processes, workers_per_suite=workers,
                phase1_identical_actors={},
                reference_bundle=None if reference_bundle is None else str(reference_bundle))
    if reference_bank: plan['labels'].append('phase1_candidate')
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    summaries = []; jobs = []
    for skill, task in TASKS.items():
        cases = [('candidate', bank[skill], 'godot'), ('source_mujoco', task.source, 'mujoco'),
                 ('previous_godot', BASELINE / task.previous, 'godot')]
        if skill in ('roller', 'roller_crouch'): cases.append(('source_mujoco_xml', task.source, 'mujoco'))
        if reference_bank:
            current_hash=hashlib.sha256(Path(bank[skill]).read_bytes()).hexdigest()
            prior_hash=hashlib.sha256(Path(reference_bank[skill]).read_bytes()).hexdigest()
            if current_hash==prior_hash:
                # One physical evaluation suffices for a byte-identical actor
                # in this exact same scene/entry protocol. Do not call it a
                # second independent test or silently duplicate its counts.
                plan['phase1_identical_actors'][skill]=current_hash
            else:cases.append(('phase1_candidate', reference_bank[skill], 'godot'))
        for label, source, backend in cases:
            dest = out / skill / label
            jobs.append((skill, label, source, backend, seeds, workers, dest))
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    def retain(short):
        summaries.append(short)
        summaries.sort(key=lambda x: (list(TASKS).index(x['skill']), x['label']))
        (out / 'summary.json').write_text(json.dumps(summaries, indent=2))
        print(json.dumps(short), flush=True)
    if suite_processes == 1:
        for job in jobs: retain(suite_job(job))
    else:
        import multiprocessing
        with ProcessPoolExecutor(max_workers=suite_processes,
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(suite_job, job) for job in jobs]):
                retain(future.result())
    sequences = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(play_sequence, bank, out / 'continuous' / f'{seed}.json',
                               seed, 'microduck_ball_stand_fix'): seed for seed in seeds}
        for future in as_completed(futures):
            result = future.result()
            sequences.append(dict(seed=futures[future],
                skills={x['label']: x['metrics'] for x in result['segments']
                        if x['label'] in ('roulade', 'ground_pick', 'kick_left', 'kick_right')},
                ordinary_falls=[x['label'] for x in result['segments'] if x['label'] != 'roulade'
                    and x['metrics'].get('fell', x['metrics'].get('unintended_fall', False))]))
    (out / 'continuous_summary.json').write_text(json.dumps(sequences, indent=2))
    keyboard = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(keyboard_walk, source, out / 'keyboard' / label / f'{condition}_{seed}.json',
                               seed, condition, backend, 'microduck_ball_stand_fix'): label
                   for label, source, backend in [('candidate', bank['walking'], 'godot'),
                       ('source_mujoco', TASKS['walking'].source, 'mujoco')]
                   for condition in ('forward', 'turn', 'mixed') for seed in seeds}
        for future in as_completed(futures):
            row = future.result(); row['label'] = futures[future]; keyboard.append(row)
    (out / 'keyboard_summary.json').write_text(json.dumps(keyboard, indent=2))
    (out / 'completed.json').write_text(json.dumps(dict(completed_unix=time.time(),
        elapsed=time.time()-plan['started_unix'], scenarios=len(sequences), keyboard=len(keyboard)), indent=2))


def main():
    p = argparse.ArgumentParser(); p.add_argument('bundle', type=Path)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--suite-processes', type=int, default=1)
    p.add_argument('--seed-start', type=int, default=1000); p.add_argument('--seeds', type=int, default=30)
    p.add_argument('--reference-bundle', type=Path)
    p.add_argument('--out', type=Path, help='New output directory; never overwrite frozen holdout evidence')
    a = p.parse_args()
    if min(a.workers,a.suite_processes,a.seeds)<1:p.error('Worker and seed counts must be positive')
    run(a.bundle, a.workers, range(a.seed_start,a.seed_start+a.seeds),a.suite_processes,a.reference_bundle,a.out)


if __name__ == '__main__': main()
