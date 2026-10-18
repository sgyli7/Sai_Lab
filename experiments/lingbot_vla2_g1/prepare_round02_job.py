"""CPU preparation of one immutable, finite carry-augmentation training turn."""
import datetime
import json
from pathlib import Path
import shutil

from experiments.vla_resources import gpu_queue
from .budget_12h import admit_allocation
from .data import OWN, ROOT, SPLIT_SHA, sha
from .make_plan import artifact, experiment_sources, write

OWNER = '01a145c6-0b86-7623-a869-086b33292199'


def prepare():
    run = 'training_round02'
    output = OWN / run
    job_path = OWN / (run + '_job.json')
    snapshot = OWN / 'source_snapshots' / run
    if any(p.exists() for p in [output, job_path, snapshot]):
        raise FileExistsError('Preserve all previous attempts')
    allocation = admit_allocation(OWN / 'gpu_budget.json')
    proof_path = OWN / 'round02_prefix_cpu_admission.json'
    proof = json.loads(proof_path.read_text())
    if (proof.get('status') != 'passed' or proof.get('frames') != 1396
            or proof.get('source_split_sha256') != SPLIT_SHA
            or proof.get('prefix_train_only') is not True
            or proof.get('source_full_task_success') is not False):
        raise ValueError('Verified train-only carry evidence required')
    original = json.loads((OWN / 'training_round01_job.json').read_text())
    frozen = dict(original['frozen_files'])
    frozen.update(proof['source_admission']['extra_frozen_files'])
    starts = [Path(__file__), ROOT / 'experiments/lingbot_vla2_g1/training_carry_round02.py',
              ROOT / 'experiments/lingbot_vla2_g1/budget_12h.py']
    if not all(p.is_file() for p in starts):
        raise FileNotFoundError('Reviewed second-round source must be present')
    for p in experiment_sources(starts):
        frozen[str(p)] = sha(p)
    initial = OWN / 'training_round01/selected'
    for p in [proof_path, OWN / 'training_round01/training_result.json',
              initial / 'metadata.json', OWN / 'gpu_budget_allocation_12h_v3.json',
              OWN / 'gpu_budget_before_12h_v3.json', OWN / 'round02_cpu_gate.json',
              OWN / 'round02_training_source_derivation.json']:
        frozen[str(p)] = sha(p)
    for name, expected in frozen.items():
        if sha(name) != expected:
            raise ValueError('Frozen training input changed: ' + name)
    payloads = dict(original['immutable_payloads'])
    meta = json.loads((initial / 'metadata.json').read_text())
    payloads[str(initial / 'trainable.safetensors')] = artifact(
        initial / 'trainable.safetensors', meta['weights_sha256'])
    for name, expected in payloads.items():
        if artifact(name, expected['sha256'])['bytes'] != expected['bytes']:
            raise ValueError('Model payload identity changed')
    current = sum(p.stat().st_size for p in OWN.rglob('*') if p.is_file())
    source_bytes = sum(Path(p).stat().st_size for p in frozen)
    new_payload_bound = 2 * (meta['weights_bytes'] + 1024**2)
    own_upper = current + source_bytes + new_payload_bound + 2 * 1024**3
    free = shutil.disk_usage(OWN).free
    if own_upper > 36 * 1024**3 or free - source_bytes - new_payload_bound < 32 * 1024**3:
        raise RuntimeError('Preserve own36GiB cap and training32GiB free reserve')
    timing_path = OWN / 'original_goal_wall_deadline.json'
    timing = json.loads(timing_path.read_text())
    frozen[str(timing_path)] = sha(timing_path)
    ticket = gpu_queue.request(ROOT / '.scratch/vla_gpu_coordination', OWNER, 'training', 3600)
    try:
        py = str(OWN / 'venv/bin/python')
        argv = [py, '-m', 'experiments.lingbot_vla2_g1.training_carry_round02',
                '--output', str(output), '--initial-delta', str(initial),
                '--carry-prefix-manifest', proof['source_admission']['manifest_path'],
                '--norm-stats', str(OWN / 'train_meanstd.json'),
                '--hours', '0.9', '--max-steps', '400']
        result = output / 'training_result.json'
        schema = 'lingbot_vla2_g1_training_carry_round02_v1'
        supervisor = [py, '-m', 'experiments.lingbot_vla2_g1.budget_12h',
                      '--ledger', str(OWN / 'gpu_budget.json'), '--stage', 'training',
                      '--max-seconds', '3600', '--memory-high-gib', '80',
                      '--memory-max-gib', '90', '--host-reserve-gib', '16',
                      '--result-file', str(result), '--result-schema', schema,
                      '--', py, '-m', 'experiments.lingbot_vla2_g1.job', '--job', str(job_path)]
        job = {'schema': 'lingbot_vla2_verified_job_v1',
               'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'fifo_ticket': ticket['ticket'], 'stage': 'training', 'max_seconds': 3600,
               'result_file': str(result), 'result_schema': schema,
               'workload_argv': argv, 'supervisor_argv': supervisor,
               'frozen_files': frozen, 'immutable_payloads': payloads,
               'budget_allocation': allocation, 'original_wall_deadline': timing,
               'selection': {'split_sha256': SPLIT_SHA, 'validation_expected_frames': 9240,
                             'validation_expected_chunks': 190, 'original_train_episodes': 80,
                             'original_validation_episodes': 10, 'test_observations_opened': False,
                             'initial_round01_step': 800, 'optimizer_resume': False,
                             'sampling': {'original': .6, 'carry_prefix': .4},
                             'full_success_teacher': False},
               'disk_admission': {'free_bytes': free, 'current_own_bytes': current,
                                  'source_snapshot_bytes': source_bytes,
                                  'maximum_two_new_payload_bytes': new_payload_bound,
                                  'own_total_cap_bytes': 36 * 1024**3,
                                  'own_future_bound_bytes': own_upper,
                                  'global_training_reserve_bytes': 32 * 1024**3}}
        write(job_path, job)
        for name, expected in frozen.items():
            target = snapshot / 'files' / name.lstrip('/')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
            if sha(target) != expected:
                raise ValueError('Source changed during actual execution snapshot')
        write(snapshot / 'manifest.json', {'status': 'completed',
              'schema': 'lingbot_vla2_training_source_snapshot_v1',
              'job_sha256': sha(job_path), 'original_hashes': frozen,
              'actual_copied_bytes': source_bytes, 'model_payloads_copied': False})
        return job
    except BaseException:
        gpu_queue.cancel(ROOT / '.scratch/vla_gpu_coordination', ticket['ticket'], OWNER)
        raise


if __name__ == '__main__':
    job = prepare()
    print(json.dumps({'status': 'prepared', 'ticket': job['fifo_ticket'],
                      'frozen_files': len(job['frozen_files']),
                      'payloads': len(job['immutable_payloads']),
                      'disk': job['disk_admission']}))
