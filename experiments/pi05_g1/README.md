# π0.5 on the simulated G1

This experiment tests whether local full-parameter adaptation improves G1
apple pick-and-place in the frozen IsaacLab source environment. Loading a model,
lower training loss, recorded-action replay, and task success are distinct checks.
Physical task results, videos and paired uncertainty estimates are the outcome.

## Data and model identity

The pinned NVIDIA dataset revision is
`37ba80a99486a4c308477854f54b4e82e77777dc`. Its actual contents are 208
parquet/video pairs and 35,066 frames; the README's 200 and info.json's 251 are
inconsistent with its file manifest and 208-row episode table. Preparation checks
all video frames and timestamps and makes a seed-42 episode split of 166/21/21.
Normalization is fitted exclusively on the training split. Source files remain
unchanged; reconciliation is recorded in the preparation audit.

The base checkpoint is `gs://openpi-assets/checkpoints/pi05_base`:
29 objects, 12,441,749,581 bytes, verified against GCS MD5/CRC32C. The official
openpi source revision is `215abfb217dbac7d5f1273282331b9b1866c0479`.
The conversion report documents strict parameter coverage. One unused expert
language-logits head, with no checkpoint weights and no action-forward references,
is removed consistently from conversion and runtime; the actual pretrained
PaliGemma tied embeddings/head are retained.

`contract.py` owns named joint ordering: 28 absolute arm/hand targets in radians,
one absolute pelvis-height target in metres, and local planar velocity
`vx, vy, yaw_rate` in m/s, m/s and rad/s form 32 channels. They retain source AGILE
command semantics without hand scaling or height offsets. The original AGILE
controller produces 12 leg targets and zero-radian waist targets; its branch
ignores torsoRPY fields. State contains 28 arm/hand joints, three waist joints and
one padding value. The policy receives head RGB, self-state and the task prompt;
object pose and contact truth are restricted to outcome measurement.

## Reproduction

Run from the repository root. All large files and execution evidence are under
`.scratch/pi05_g1/`; `ENVIRONMENT.md` and `environment_probe.json` record the
isolated Python3.11 ARM64 installation and official Transformers patches.

```bash
export PYTHONPATH="$PWD/.scratch/pi05_g1/openpi/src:$PWD"
export OPENPI_DATA_HOME="$PWD/.scratch/pi05_g1/venv/openpi-cache"
export JAX_PLATFORMS=cpu

.scratch/pi05_g1/venv/bin/python experiments/pi05_g1/download_dataset.py
.scratch/pi05_g1/venv/bin/python experiments/pi05_g1/prepare_training_data.py \
  --dataset-root .scratch/pi05_g1/data --output-dir .scratch/pi05_g1/prepared

uv run --no-project --with google-crc32c --python 3.13 \
  experiments/pi05_g1/download_checkpoint.py --output .scratch/pi05_g1/pi05_base
CUDA_VISIBLE_DEVICES='' .scratch/pi05_g1/venv/bin/python \
  -m experiments.pi05_g1.convert_checkpoint \
  --checkpoint .scratch/pi05_g1/pi05_base --output .scratch/pi05_g1/pi05_base_pytorch

docker build -f experiments/pi05_g1/Dockerfile.pi05_g1 \
  -t sai-pi05-g1-source:6.0-frozen .
python3 -m experiments.pi05_g1.sim_launcher --docker --probe
```

GPU commands must run serially through `budget.py`. It records failed attempts
as well as successful ones against preflight's one-hour, training's four-hour,
evaluation's three-hour and the total eight-hour budgets. A user systemd service
enforces a wall-time deadline and memory limits; host free memory is watched
separately because GB10 CUDA allocations may escape cgroup accounting. Owned
Docker containers receive independent limits and are verified removed before
closing accounting. Kernel tasks stuck in uninterruptible allocation may outlive
SIGKILL: these remain open receipts and block new launches. The initial resume
OOM's actual 2,496 seconds remain charged, including its deadline overrun.
Use fresh evidence directories after any failed case.

```bash
python3 experiments/pi05_g1/budget.py \
  --ledger .scratch/pi05_g1/gpu_budget.json --stage preflight -- \
  python3 -m experiments.pi05_g1.sim_launcher --docker \
  --config .scratch/pi05_g1/sim_config_template.json

python3 experiments/pi05_g1/budget.py \
  --ledger .scratch/pi05_g1/gpu_budget.json --stage training -- \
  .scratch/pi05_g1/venv/bin/python -m experiments.pi05_g1.training \
  --dataset .scratch/pi05_g1/data/lerobot \
  --manifest .scratch/pi05_g1/prepared/split_manifest.json \
  --norm-stats .scratch/pi05_g1/prepared/train_norm_stats.json \
  --base-checkpoint .scratch/pi05_g1/pi05_base_pytorch \
  --output .scratch/pi05_g1/training
```

Training uses BF16 model computation with the official stable FP32 layers and
FP32 AdamW master weights/moments, batch size one, eight accumulated microbatches,
learning rate 1e-5, and at most 2,000 optimizer steps. The validation split selects
the checkpoint; test trajectories do not select hyperparameters. Old checkpoint
metadata and logs remain when local weight/optimizer payloads are pruned. An
independent archiver preserves every complete payload using immutable hard links
under `/home/ethan/ProjectBackups/2026-10-16/pi05_g1_first_round/checkpoints`.
The installed safetensors saver stages the complete 6.94GB model on CPU during
saving; that measured memory peak is included in resource supervision.

`run_evaluation.py --help` runs either policy over `eval_cases.json` and reuses
only completed cases with exactly matching configuration and checkpoint hash.
Successful original-demonstration replay, bound to the same source/camera/action
interface, is mandatory. The inference service uses a separate Python3.11 process
inside the source container. The simulator preserves 200Hz physics/50Hz control;
inference pauses are explicit. Captures, raw chunks, final commands, WBC targets,
physics truth, contacts, 50fps video and failure receipts are retained.

## Reading the result

The 20 trials are five fixed apple positions with four seeds each. They are
repeats at five positions, not 20 independent environments. Both policies use
the same inputs, limits, cases and noise seeds. A successful trial requires
hand-contact lift before placement, then two seconds of independently audited
released support on the plate, with no fall or drop. Acceptance v2 requires 101
consecutive 50Hz samples with plate support, the whole apple footprint inside the
plate, positive separation from every robot collider, zero robot contact and a
standing robot. The maximum pairwise apple position and orientation changes in
that window must be at most 5mm and 5 degrees. The source's immediate
success flag alone is insufficient. An explicit audit tail can extend the
original six-second episode by at most two seconds while holding the final action.

`summarize_evaluation.py` checks complete paired results and video evidence,
reports Wilson intervals, paired trial statistics and position-cluster bootstrap
uncertainty. Task usability requires at least 16/20 successes. Missing or
infrastructure-failed cases remain incomplete. Simulation success establishes
neither real-robot transfer nor continuous real-time operation.

The initial demonstration replay is retained even though its legacy audit failed:
instantaneous angular speed exceeded 0.1rad/s under contact jitter, while two-second
pose diameters were below 0.12mm and 0.5 degrees. Acceptance v2 was frozen before
any model evaluation; the original speed audit remains a diagnostic. A fresh
replay must qualify the changed measurement implementation before policy trials.

CPU checks: `.scratch/pi05_g1/venv/bin/python -m unittest discover -s experiments/pi05_g1/tests -v`.

Known scope limits are recorded with source evidence in experiment receipts.
The full dataset has 57 nonzero vx, 65 nonzero vy and 1,037 nonzero yaw-rate targets. Training quantiles collapse vx/vy to zero despite 57/65 nonzero training values, so the first-round adapter loses those rare commands; yaw is retained. This limitation is recorded in data_adapter_limitations.json. The frozen first-round normalization and scores remain unchanged.
The original `dropped` predicate means apple actor-origin world z < -0.2m;
it does not detect every slip or return to the tabletop. Dataset and replay
reset joints match exactly, but RGB frames differ. The published data does not
establish complete camera/root-pose capture provenance. This discrepancy is
retained without changing the frozen camera.
