# Carry augmentation after the source-only baseline

The first experiment is complete and preserved: 800 optimizer updates, 20 audited
physical cases, 7 sampled grasp events, 0 qualified long carries and 0 complete
deliveries. Its report is
`.scratch/lingbot_vla2_g1/benchmark_pose_speed_summary_v2.json`.
That completed baseline does not complete the user's twelve-hour improvement goal.

The second recipe starts from the selected first-round parameters with a **new**
FP32-master AdamW optimizer. It samples original training frames with probability
0.6 and the independently audited native201 carry prefix with probability 0.4.
The prefix has 1,396 synchronized frames from one development family. The source
task failed release and stable placement; the prefix is not a successful-delivery
teacher. Its 14 finger-action channels are all zero, like the original data.

Original episode splits, the original 80-training-episode LingBot mean/std, the
10 complete validation episodes, and the unopened dataset test split are retained.
The prefix remains training-only. Its original audit binds the peer's quantile
normalizer, which is explicitly checked separately from LingBot's unchanged
normalizer. CPU admission has checked every raw supervised PNG, measured-joint
mapping, the existing next-row action alignment audit, and the full evidence
chain. The receipt is
`.scratch/lingbot_vla2_g1/round02_prefix_cpu_admission.json`.

The bounded execution allows 400 additional updates, microbatch 1, accumulation 8,
learning rate 1e-5, and complete original validation every 100 updates. Selection
uses the smallest complete validation flow loss, earliest tie, including the
initial checkpoint. No physical benchmark result selects a checkpoint within the
run. The hypothesis is improved carry coverage; benefit is unproved before the
second physical evaluation.

After peer ticket t51 was queued, a prospective resource decision at 04:37 UTC
set a graceful stop after complete validation at update 200. This preserves time
for one new physical pilot after t51 within the original wall window. The frozen
400-update ceiling and validation selection rule are unchanged; the actual update
count will be reported. The decision preceded every physical outcome of the new
policy and is recorded in `round02_resource_stop_rule.json`.

The hardware remains the same physical GB10, with the same G1/Homie simulation,
head camera, observation/action contract, physics, twenty scenarios, and success
criteria. This is a repeated evaluation of the same cohort after seeing the
first recipe's failure, not a new blind generalization test. Extra training data,
pretraining differences, and the narrow demonstration family must be disclosed in
any comparison with π0.5.

The original twelve-hour wall window runs from the goal's actual creation time
to 2026-10-17 05:33:07 UTC; queue waiting uses this wall window. Physical retesting
of the new checkpoint will be reported as partial unless all twenty new trials
are actually completed and audited within the window. The completed first-round
cohort remains separate.

GPU turns retain the shared FIFO and independent cgroup/host-memory safeguards.
The user-authorized ceiling is 43,200 owned GPU seconds; the first 26 charges,
totalling 16,081.547703910095 seconds, remain byte-for-byte in the preserved ledger
prefix. Queue waiting is not charged. The prospective allocation and storage
change are recorded in `gpu_budget_allocation_12h_v3.json`; the own output cap is
36 GiB, with the shared 24 GiB floor plus 1 GiB next-case buffer retained. Original
weights, failed attempts, raw traces, videos and receipts are retained.

Training completed at exactly 200 additional optimizer updates under t50. The
initial complete validation exactly reproduced the first-round selected flow
loss, 0.1764998310346366. Complete validation at 100/200 is
0.1691829534336215 / 0.16529748874083483 (each 190 chunks / 9,240 frames /
10 episodes). The selected and final checkpoint are the same step200 snapshot,
959 BF16 parameter tensors, 3,876,326,766 bytes, SHA256
`460e8473ca1bee7de91326d3610a6dc942612a717b1046f07d4cd0a43c02d0bb`.
The 1,601 attempted sample draws include one sample from an aborted next update;
only 1,600 microbatches contributed to the 200 applied updates. No optimizer
update201 occurred. All 36 persistent buffers remained unchanged.

Actual second-round owned GPU time was 1,233.1487176869996 seconds, including
preflight/loading and cleanup. Its FIFO ticket and owned unit closed successfully.
The t52 pilot completed all 3,000 controls / 12,000 physical integrations /
60 simulation seconds and passed the independent physical, action and video
audit. It failed the task: no qualified grasp, 0 m contact-qualified carry,
0 degrees qualified carry turn, and no stable released placement. No fall or
ground drop was recorded; the residual timeout flag is true. Actual initial
kinematics, full physical interface and normalized non-position interface all
match the same round01 case exactly. The pilot GPU turn was 540.0686877130065
seconds; task wall time was 507.356663042 seconds. Its 59 warm model calls had
median 0.5556705869967118 seconds. Failed-task elapsed time is not a task-speed
success measurement.

This is **one repeated case**, `long5m_center_s1040`; the other nineteen new-policy
cases were not run. The descriptive score is 0 task successes / 1 completed pilot,
not a new 0/20 result. The original source-only twenty-case report remains
unchanged: 0 complete tasks, 0 qualified long carries, 7 sampled grasp events.
Validation improved, but this pilot provides no measured task improvement.
See `.scratch/lingbot_vla2_g1/round02_pilot_review.json` and its referenced raw
summary/audit. Nine CPU tests and the actual queue waiting-timeout / cancellation
path passed before this retest.

Both new FIFO turns have closed and released their resources. Cumulative owned
GPU use, including every previous failure/interruption, is 17,854.7651093101
seconds (4 hours 57 minutes 34.8 seconds). Queue waits are separate GPU accounting
and count toward the original twelve-hour wall window.

Both selected trainable-parameter snapshots and reproduction metadata have a
separate readback-verified backup at
`/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z`.
The backup deliberately references the original immutable pretrained base and
dataset; it is not a self-contained complete model/runtime archive. Every original
workspace file remains in place. The combined owned output/backup future bound is
34,233,072,514 bytes, below the prospective 36 GiB cap.

The completed pilot's full raw episode, both original 50 fps videos, all query
observations, independent audit, segment receipt and final review were subsequently
appended and verified by readback. The final backup manifest is
`/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z/final_pilot_manifest.json`
(SHA256 `b8385a30f7edd5fa84b58b3d6207399747f4cdbf9be2c99d8677487262078cd6`).
The overall bounded-study receipt is
`.scratch/lingbot_vla2_g1/goal12h_completion_receipt.json`. It explicitly records
that local execution/fine-tuning and this evaluation work completed, while full
robot-task capability remained unproved and unsuccessful in the measured trials.
No additional GPU job is pending. All original failures, raw evidence and charges
remain preserved.
