# LingBot-VLA 2.0 on the same simulated G1 mobile task

Completed on2026-10-17:800real optimizer updates and20independently audited
physical simulation trials on the same GB10/G1/Homie task as the π0.5 peer.
The authoritative report is
`.scratch/lingbot_vla2_g1/benchmark_pose_speed_summary_v2.json` and its CSV.
Full task success and qualified4m/60deg transport are both0/20;7cases contain
a grasp event. The largest contact-qualified displacement is0.21664m.

This chat owns local LingBot-VLA 2.0 adaptation on the existing NVIDIA GB10
(ARM64, 128GB unified memory). The π0.5 chat is an equal peer. GPU phases use
`experiments.vla_resources.gpu_queue` and the same host flock. Each chat keeps its
own owned-process budget ledger; waiting uses no GPU budget. No foreign process
is stopped and no prior checkpoint/video is removed.

The task follows the reference chat's **latest mobile box transport objective**:
G1 with Homie balance/walking control, head640x480 RGB, measured joints and text,
50Hz control/200Hz physics, action chunks50. The common cohort is frozen at20cases
in `.scratch/pi05_g1/round03_mobile/scenarios_frozen_v3.json` (five physical
positions, four repeated seeds each). Acceptance requires grasp, at least4m
carried displacement, at least60deg turn, supported release stable2s in a roughly
5m source-target open aisle. This does not establish arbitrary navigation or
physical robot capability.

## Inputs and local model variant

Official source: `Robbyant/lingbot-vla-v2` at
`be969b8fd117fb70550c5d4bf4bc328211b5b1b6`.
Official checkpoint: `robbyant/lingbot-vla-v2-6b` at
`11c703bf6a5c1f45b3b69168482da11fdbba53d7`.
Qwen processor/config only:
`Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17`.
No second Qwen model is downloaded. Primary-source evidence is in
[UPSTREAM_RESEARCH.md](UPSTREAM_RESEARCH.md).

The official six F32 shards are verified and converted sequentially to BF16.
Source hashes, shapes and converted hashes are retained; only reproducible raw
staging downloads are removed after verification. Native depth/video query and
head structure stays present. An independent flag disables auxiliary teachers
and their losses. This is an explicitly documented action-only adaptation,
not a claim to reproduce the complete upstream multi-task training recipe.

Torch2.11+cu130 is read-only borrowed from the existing isolated π environment
through a lower-priority path. Transformers4.57.3, NumPy1.26.4 and small supporting
packages live in the separate LingBot Python3.11 environment. The raw upstream
checkout is pristine; an auditable overlay switches attention to SDPA and uses
PyTorch matrix operations for the original fused 3D MoE parameter storage. CPU
forward/gradient equivalence tests cover both backends.

`contract.py` explicitly embeds the unchanged effective32 API in model55:
arms0:14, hands39:53, action height30, state waist31:34, action base velocity36:39.
Unused dimensions are zero/masked. This is a **local G1 schema**, not an official
G1 mapping. All28 arm/hand targets remain absolute radians, height metres,
base velocity m/s,m/s,rad/s. State has31 observed dimensions and an external pad.
Object truth, route waypoints and actor poses never enter policy requests.

Data is the same pinned NVIDIA mobile100-trajectory set and exact80/10/10 split.
Mean/std is fitted exclusively on75918 training frames. The shared test split
has not been opened by data preparation. Source2m demonstrations and the authored
5m evaluation are different distributions; new qualified teacher data must be
separately frozen if later added.

## Current execution entry points

From `/home/ethan/Projects/Sai_Lab`, use
`.scratch/lingbot_vla2_g1/venv/bin/python` as `LINGBOT_PY` below.

```bash
# CPU preparation; never starts CUDA.
CUDA_VISIBLE_DEVICES='' "$LINGBOT_PY" -m experiments.lingbot_vla2_g1.prepare_assets
python3 -m experiments.lingbot_vla2_g1.patch_runtime
CUDA_VISIBLE_DEVICES='' "$LINGBOT_PY" -m experiments.lingbot_vla2_g1.data
CUDA_VISIBLE_DEVICES='' "$LINGBOT_PY" -m experiments.lingbot_vla2_g1.runtime

# CPU checks.
python3 -m unittest discover -s experiments/vla_resources/tests -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.scratch/lingbot_vla2_g1/runtime_source \
  "$LINGBOT_PY" -m unittest discover -s experiments/lingbot_vla2_g1/tests -v
```

GPU commands **must** use an owned supervisor inside a FIFO ticket. The first
compatibility phase is capped at900seconds and executes the real full checkpoint:
forward, backward, FP32-master AdamW update, frozen-VLM check and50x32 inference.
Only a fresh completed producer receipt plus proved cleanup counts as completed.

Training defaults are expert-only BF16, batch1, accumulation8, lr1e-5,
FP32-master AdamW, at most800updates/3.5hours; complete validation every100steps,
including the unchanged base as step0. Best candidate weights are held in bounded
CPU RAM and at most two immutable delta payloads are saved at termination. No
optimizer-resume payload is saved. The internal conservative supervisor ceiling
is8GPUhours across compatibility/training/evaluation, independent of π's budget;
it is not a commitment to consume the full allowance.

`training.py`, `service.py`, and `sim_launcher.py` provide the implementation;
GPU compatibility, actual training and physical outcomes must each be confirmed
by their own evidence. A successful import or lower loss alone is not a task
success. Final comparison uses the peer's frozen scene/controller/cases and
reports all infrastructure failures separately from valid physical failures.

## Verified local compatibility (2026-10-17)

`compatibility_attempt02/result.json` completed on the same GB10 after strict
loading of all1,708 tensors. Real training loss was0.426593, with883 gradient
tensors; sampled action-expert/state/output parameters changed and a sampled
frozen VLM parameter did not. The post-update ten-step action inference returned
finite50x32 actions in0.5335s. This single warmed measurement is not the physical
benchmark. Peak allocated CUDA memory was46,426,998,272B (43.24GiB).

Attempt01 failed on a local untyped YAML scientific-notation coefficient; its
84.58s remains in the independent ledger. Explicit numeric parsing and a CPU
regression check corrected it. Attempt02 used89.62s, and both owned units were
confirmed inactive with complete cleanup before releasing their FIFO turns.
Formal fine-tuning and physical outcomes remain separate stages.

## Completed first fine-tuning round

The actual source-only round completed as FIFO ticket `t000010`. Its immutable
work item is `.scratch/lingbot_vla2_g1/training_round01_job.json`, with292 frozen
small files, six rehashed pretrained shards, and an exact source snapshot.
The full original validation set covers9,240frames in190nonoverlapping chunks.
All800updates completed. Validation selected step800, improving flow loss from
0.6828147 to0.1764998 (74.15% lower). The190chunks cover9,240frames at every
validation point. This is an offline action-fit measurement; no formal
closed-loop outcome is implied.

| Update | Complete validation flow loss |
|---:|---:|
| 0 | 0.6828147 |
| 100 | 0.3523411 |
| 200 | 0.2805105 |
| 300 | 0.2452397 |
| 400 | 0.2209111 |
| 500 | 0.2015317 |
| 600 | 0.1909342 |
| 700 | 0.1810469 |
| 800 | 0.1764998 |

Actual owned GPU time for training was4,322.632seconds (72.04min), with no OOM
or budget overrun. Selected and final are the same immutable delta, so only one
3,876,326,766byte payload was written. `training_round01/training_result.json`
and `selected/metadata.json` bind the recipe, split, normalization and weights.

`restore_round01/result.json` independently completed after strict reload:
sampled action/state/expert CUDA values equal saved tensors, sampled frozen VLM
values equal the base, and same-seed actions repeat exactly. Two50x32 inference
calls took1.4306s cold and0.5221s warm. The separate FIFO turn was`t000015`.
The original8h internal total remains unchanged. A recorded reallocation
reserves1h preflight/2h training/5h evaluation after the measured training run;
all historical charges remain intact and the original supervisor source stays
unchanged. `budget_allocated.py` validates the allocation before later work.

The peer later qualified a1,396frame TRAIN-only native carry prefix. It is
recorded in `peer_carry_prefix_reference.json` and is **not used in this already
frozen source-only round**. Cross-model results must declare this recipe
difference. The prefix is not a complete released-stable placement success.

`make_plan.py` verifies the training result against every complete validation
candidate, validates selected-delta tensor headers on a meta model, and freezes
the same twenty scenarios. Its original unexecuted plan and all historical
source snapshots remain preserved.

## Shared qualified measurement and formal evaluation

Before either chat began formal trials, a common three-second physical placement
calibration and independent CPU audit qualified the explicit
`pi05_g1_mobile_pose_difference_speed_v1` measurement. Raw SDK box velocity and
recorded pose motion disagreed under contact. The versioned shared adapter uses
adjacent actor-origin world positions at50Hz for the existing0.02m/s linear
speed gate. The first undefined sample is ineligible. It retains raw SDK
velocities and the original angular-speed gate, all-pairs2s/5mm/5deg stability,
contact, support, separation, standing, carry and fall/drop requirements.
Geometry body IDs are explicitly resolved to names. Original simulator,
acceptance source, thresholds and failed diagnostic attempts are preserved.

The prospective calibration produced49overlapping two-second windows from
**one** three-second trial. It is a placement primitive qualification, not49
trials, an autonomous task success or training data. The common qualification is
`.scratch/pi05_g1/round03_mobile/placement_oracle_attempt03_pose_speed_qualification.json`.

`make_plan_pose_speed.py` bound that qualification before the first formal case,
without changing the step800 model selection. The original qualified plan is
`.scratch/lingbot_vla2_g1/evaluation_plan_pose_speed/plan.json`, SHA256
`5527af3bbfaaf4a5ba91510f1f13706ce872f39b0b1c100ac1f2bc37e8fa878c`.
It freezes2,861small dependencies and seven independently rehashed model
payloads. Both chats adopted the same measurement module SHA256
`1790afba45148e3a16506a9646575d52bf53b78f6b354e2b3744712849273815`.

The physical execution entry points are `evaluate_pose_speed.py`,
`run_case_pose_speed.py`, `sim_launcher_pose_speed.py`,
`acceptance_context.py` and `benchmark_pose_speed.py`. The earlier entry points
stay available as historical evidence. Each physical case gets its own owned
supervisor and FIFO ticket; a segment reserves at most three cases. CPU audits
check every action/query, all physical trace counters, acceptance recomputation,
raw-file identities and both complete50Hz videos. Immutable model payloads are
rehashed at both segment boundaries. A failed evidence audit stops the finite
driver and remains an unknown task outcome, never an ordinary policy failure.

The original finite20case driver is recorded in
`.scratch/lingbot_vla2_g1/preparation_scripts/evaluate_pose_speed_driver.py`;
its state and immutable segment receipts are under
`.scratch/lingbot_vla2_g1/evaluation_segments_pose_speed/`. It used frozen case
order and stopped at its first infrastructure failure. It never overwrote or
automatically retried a partial raw case. The original CPU aggregate at
`benchmark_pose_speed_summary.json` remains preserved with its reporting error;
the corrected authoritative v2 report is described below.

The first two frozen cases completed all3,000controls/12,000physics steps and
passed independent audits. Both lacked a verified grasp and timed out at60sim
seconds without a fall. A cohort success rate requires all20audited cases.
No formal outcome changes this round's selected model or task thresholds.

The third attempt, FIFO`t000020`, received an unexpected orchestration signal
and stopped after264.096charged GPU seconds. The parent driver recorded
`stopped`, its supervisor recorded`interrupted`, and the partial producer had
no completed result. Docker and owned-unit cleanup completed. Its original raw
directory and immutable failed segment receipt remain preserved. The historical
signal sender is unknown. That attempt is an unknown task outcome.

A CPU-only differential probe uses the original driver with only its GPU
subprocess replaced by a sleeping CPU child. An explicitly created same-group
SIGTERM reproduces the stopped driver; an independent systemd service survives
the same external group signal. Probe artifacts are under
`.scratch/lingbot_vla2_g1/diagnostics/formal_interruption_t20/`. This demonstrates
the isolation mechanism without assigning the historical signal to a sender.

`prepare_resume_pose_speed.py` independently verifies and rebinds the two
completed cases, preserves the failed raw inventory and charge, and changes
only dependency manifests and output paths for the remaining18cases. The signal
recovery plan is
`.scratch/lingbot_vla2_g1/evaluation_plan_pose_speed_resume01/plan.json`, SHA256
`97f9fb0cc43ad1e91a83ef716b326fd2603118b250650740e318a30ab3418d71`.
An integration gate confirms identical model payloads, selection, data, all20
cases, physics and measurement, exact first-two job records, and unchanged
failed raw bytes. Prefix gates reject a physical-policy retry, wrong case, or
noncontiguous prefix.

The isolated finite CPU coordinator was
`lingbot-eval-pose-resume01.service`, with exact command and source identities in
`.scratch/lingbot_vla2_g1/evaluation_resume01_service_launch.json`. Its state and
new immutable receipts are in
`.scratch/lingbot_vla2_g1/evaluation_segments_pose_speed_resume01/`.
It reserved at most3FIFO turns at a time and stopped on new infrastructure
failures. GPU workloads used the same owned budget supervisors. After the
peer's training turn closed, it continued through15audited cases.

The recovery coordinator subsequently completed15audited cases. Its original
internal33GiB prelaunch check then stopped before case16, with all completed
model-payload checks intact and all five remaining output directories absent.
This resource stop created no additional physical trial or GPU charge. The
earlier coordinator and partial segment receipt remain preserved.

A prospective CPU resource revision is recorded in
`.scratch/lingbot_vla2_g1/evaluation_plan_storage_v2/storage_admission.json`.
It uses an internal24GiB reserve plus1GiB admission buffer, informed by the
peer's existing24GiB archival resource plan. It explicitly supersedes this
chat's earlier internal32GiB reserve; it does not change a user-specified
numeric limit. The8h owned GPU total and25GiB own storage cap remain fixed.
The retained-output commitment for the last five cases is at most1GiB.
All historical models, videos, data and receipts remain present.

The final physical continuation plan is
`.scratch/lingbot_vla2_g1/evaluation_plan_storage_v2/plan.json`, SHA256
`3fbf1ee368c3335549b1c26239e57ca8f4ae0906998763407a59c9acfe4f77f1`.
Its complete twenty-job list, configuration hashes, result paths, selected
weights, camera/robot/controller/physics and scoring definitions are exactly
identical to the previous plan. The new `evaluate_storage_v2.py` changes only
the explicit CPU admission gate and its receipt. A CPU rejection check proves
that the older plan cannot silently use this resource revision.

`lingbot-eval-storage-v2-finish.service` completed two finite FIFO segments
to finish the remaining five cases. Exact launch identity and command are in
`.scratch/lingbot_vla2_g1/evaluation_storage_v2_service_launch.json`; state and
receipts are in`.scratch/lingbot_vla2_g1/evaluation_segments_storage_v2/`.
The new prefix receipt revalidates all fifteen original case records, audits,
raw inventories and closed GPU ledger rows. The original failed partial
attempt stays separately recorded as an unknown task outcome.

## Completed twenty-case result

The final physical segment completed all20cases. Its initial report correctly
retained all20case records but rejected the cohort because the reporting code
required a single full interface SHA across different scene positions. Full
interface identities include the box's initialX/Y coordinates, which are
intentionally different in the five frozen position groups.

`benchmark_pose_speed_v2.py` corrects only that CPU reporting check through
`interface_invariants.py`. Every original full interface hash is recomputed and
retained. Measured box positions must exactly equal each frozen case. Only the
four explicit boxX/Y leaves in the SDK scene and its scene configuration are
normalized for the common comparison. Every remaining identity field must be
exactly equal across all20cases, and full hashes must match within each four-seed
position group. CPU probes accept all20actual cases and reject an unplanned box
position, changed physics timestep, changed camera and incorrect full hash.
The original failed report and minimal two-case red reproduction remain intact.
No physical rerun or scoring change was needed for this reporting correction.

| Measurement | Observed result |
|---|---:|
| Evidence-qualified physical trials |20/20|
| Full task success |0/20|
| Contact-qualified4m/60deg transport |0/20|
| Grasp events |7/20|
| Released stable placements |0/20|
| Falls / ground-drop flags |0/20 /0/20|
| Timeouts |20/20|
| Maximum contact-qualified box displacement |0.21664m|
| Maximum contact-qualified turn |8.9956deg|
| Warm inference median / p95 |0.56785s /0.60239s|
| Warm inference samples |1,180|
| Cold model loading median |95.857s|

All trials ran60simulation seconds,3,000controls and12,000physics integrations.
Each retains both full50Hz videos, all control/state/physical records, query RGB
and decoder evidence. Grasp events are the shared contact/lift proxy, rather than
proof of sustained handling. Zero ground-drop flags does not establish error-free
handling. There are five physical position groups with four seed repeats each.

Simulation pauses during synchronous model RPC. The median520.661s actual task
wall time describes60s timeout rollouts, not successful task completion speed.
There are no successful episodes from which to estimate completion speed.

The owned ledger totals16,081.5477seconds(4h28min1.55s):415.2688preflight,
4,322.6321training and11,343.6468evaluation. This includes the interrupted
264.096s formal attempt and failed84.58s compatibility attempt. FIFO waiting
is excluded. All26owned rows are closed with resource cleanup confirmed;
the chat has no pending GPU reservation. Original models and all raw attempts
remain preserved. No cross-model ranking is asserted: this first round uses
only the original80training trajectories, while the peer later added a separate
1,396frame training-only carry prefix.

The first-round adaptation completes the local training/inference/evidence
pipeline but does not demonstrate the requested long-distance transport ability.
The final report, per-case CSV, training result, selected checkpoint metadata,
restore proof and exact source/provenance records remain available under
`.scratch/lingbot_vla2_g1/`.
