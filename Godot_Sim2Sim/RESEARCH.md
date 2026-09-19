# Godot skill research — 2026-09-10

User-authorized objective: approach the original MuJoCo skill quality in real
Godot/Jolt, continuously iterate for at most eight hours, and extend to all nine
ONNX policies when the evidence supports it.

Start: 2026-09-09 17:39:04 UTC. Original wall deadline: 2026-09-10 01:39:04 UTC.
The user later clarified eight hours of actual work, excluding disconnections.
Two observed gaps total approximately 157 minutes; the recorded adjusted cutoff
is 2026-09-10 04:16:04 UTC. Implementation, diagnostics, training and evaluation
remain inside this working-time budget. See the supplementation section and
session.json for the original timestamps and approximate accounting.

## Fixed rules

- Preserve the baseline files and their hashes in `results/research_20260910`.
- Compare factory/MuJoCo, factory/Godot, previous finetune/Godot, and candidates.
- Complete task behavior is the objective; training reward is diagnostic only.
- Fix task semantics and physical telemetry before optimizing their rewards.
- Actor interface remains 61 observations / 14 joint-position action offsets.
- No supporting forces, playback poses, or hidden controller assists.
- Ground pick means the official mouth-down reach and return, not object grasping.
- Roller crouch is the phase-controlled crouch/glide/return maneuver.
- Freeze each evaluation protocol version before candidate comparisons. A protocol
  correction requires re-evaluating baselines, never silently moving the target.
- Separate development and final test seeds. Record failures and uncertainty.
- Candidate exports do not become default play policies without promotion.
- Do not overwrite user changes in README.md, SIM2SIM.md or HANDOFF.md.

## First comparison

Standing, walking, left kick: corrected PPO; teacher-regularized PPO; frozen
factory plus bounded residual PPO. Extend to the other tasks as time permits.
Reference anchoring uses deterministic means (factory ONNX has no exploration
distribution). Resume must preserve optimizer, normalizer, LR and RNG state.

The session JSON stores the deadline. Every experiment checks it before sampling
and updating. Results, traces, checkpoints and ONNX artifacts stay under the
session directory. Research source and tests are committed in stages.

## Confirmed reset/limit defect (18:19 UTC)

Native end-stop sweeps exposed two interacting errors in `_rebake_joints`:

1. `_joint_q` read the cached body bases from the previous episode, before the
   teleported pose refreshed them. Consequently the physical limits changed
   after every training reset, depending on the preceding failed trajectory.
2. Godot hinge angles are clockwise, opposite the MuJoCo coordinate convention.
   With the hinge rebaked at `q_reset`, its coordinate is `-(q-q_reset)`.
   XML bounds `[lo, hi]` must therefore become `[q_reset-hi, q_reset-lo]`.

The engine convention is documented in the [Godot Jolt hinge implementation](https://github.com/godotengine/godot/blob/master/modules/jolt_physics/joints/jolt_hinge_joint_3d.cpp).
The correction refreshes the body cache before rebaking and maps these bounds.
It changes neither the XML joint ranges nor the controller/torque parameters.
The converter now applies the same mapping at its generated reference pose.

Evidence: `results/research_20260910/joint_limit_probe.json` records the failed
alternatives; `joint_limit_fixed.json` records 36 successful native end-stop
checks across six joints, three initial poses and both bounds. Largest absolute
error: 0.000014025 radians. `tests/test_joint_limits.py` guards repeated resets
in a reused worker, rather than only checking serialized range properties.

Without any weight changes, the original ground-pick policy then achieved 3/3
complete Godot reaches/returns, with mouth-tip minimum near 0.0194 m versus the
MuJoCo reference near 0.0205 m. Native rendered playback reproduced the headless
physical measurements. Crouch tilt also dropped substantially. These are small
development samples, not final reliability claims.

The first residual runs `w01_residual` and `k01_residual` are invalidated for
promotion because their training workers changed effective bounds on reset.
They remain archived as negative evidence. Protocol v3 retains v2 task outcome
definitions, but re-evaluates all baselines under the corrected physical model.
Training resumes from source weights with immutable corrected limits.

Full-network increments use float64 internally for the subtraction and cast the
increment back to float32 before adding the unmodified native ONNX output.
Nonzero trained exports now pass the unchanged 1e-5 deployment tolerance; initial
exports match exactly. This is distinct from re-running the original ONNX in
float64, which would alter its finite-precision behavior.

Protocol v4 adds two adversarial outcome guards: a kick that falls and later
stands up is not a clean kick, and several revolutions are not a single roulade.
All v3 baseline traces were re-scored under v4, preserving their original files
and simulator fingerprints. Ongoing v3 pilot exports will also be re-scored
before selection. Candidate ranking prioritizes complete success rate, then
the continuous diagnostic score; partial progress never constitutes promotion.

## Entry-distribution correction (18:45 UTC)

Upstream `infer_policy.trigger_behavior` preserves `last_action` when switching
from standing to a kick/roulade. A cold reset with a zero action history is not
equivalent. Native tests verified the difference: source MuJoCo left-kick
completion rose from 1/3 cold starts to 3/3 actual standing-policy handoffs;
the first adapted kick finished 3/3 cold starts but 0/3 handoffs. It is therefore
not deployable despite its cold-start result.

The evaluator now explicitly labels `entry=reset` and `entry=standing`, and
supports a combined `both` suite. The latter performs one second of real source
standing control, preserves the action history, places the ball at the skill
trigger, then evaluates the candidate alone for the full task window. The
standing controller never assists within that evaluated window. All original
cold-start results remain archived as a separate stress test.

New training trials use a 50/50 mixture of those entry distributions. Independent
standing warmups are stepped together for efficiency. Finite one-shot maneuvers
end naturally at their task deadline; only continuous-control tasks bootstrap
artificial time-limit truncations. These changes and source checkpoint ancestry
are recorded per run; earlier and later trial scores must not be conflated.

The exact upstream bilateral transform is also being evaluated as a right-kick
initialization. It preserves the 61/14 interface and has a zero-error 10,000-input
ONNX check. It is a candidate policy transform, not evidence of physical symmetry
or a substitute for separate right-foot contact/direction tests.

## Runtime contracts and recovery curriculum (19:05 UTC)

The play state machine now gives phase zero on the first skill action, runs
ground pick for four seconds and kicks/roulade/crouch for five, and preserves
the preceding action history. Roller idle uses the roller policy with a zero
twist; the crouch policy is a separate five-second phase action on key 2/Y.
Normal MicroDuck play loads the free-ball scene and places the ball at each
kick trigger in the current heading frame, matching source play semantics.

Native capture exposed two rendering omissions in the new ball scene: OBJ
assets needed Godot import, and a colliding primitive sphere had no visual
mesh. The converter now emits a SphereMesh and play imports missing assets.
The current generated ball scene received only the visual nodes/resources;
its existing physical nodes/resources and physical spec are preserved. The
same kick trace before and after this display fix has identical measurements.
Generated JSON mapping descriptions change when regenerated by the corrected
converter, but their body/joint/geom/actuator data are unchanged.

The mixed-entry left-kick checkpoint `k07_mixed_resume/iteration_00163.pt`
passed 6/6 initial and 16/16 additional development episodes (both entries).
This establishes useful progress, not final held-out reliability. Subsequent
checkpoints reduce action variation but still turn more than source MuJoCo;
the categorical score saturates, so raw smoothness, tilt, direction and yaw
metrics remain necessary for final selection. An optional recorded heading
penalty and stronger existing smoothness/direction weights support polishing
trials without changing the default objective or evaluator.

`curriculum.py` constructs 184 physical starts from eight successful source
MuJoCo rolls using training seeds 40000–40007. Each retains full generalized
position/velocity and the actual last action. The source progress and pivot
history initialize the training critic/reward accumulator so replaying earlier
rotation does not earn new progress. No source actions run after that reset.
The first controlled comparison uses 50% such training starts and 50% ordinary
mixed entries. Evaluation always remains a full roll from normal cold and
standing entries. Tests verify pose/history transfer and clean accumulator
reset; a short end-to-end smoke run also completed with exact initial export.

## Reference clarification and walking hypotheses (19:18 UTC)

Upstream roller play sets passive-wheel frictionloss to 0.003 at runtime,
whereas its training XML leaves it zero. Both source MuJoCo references are now
retained explicitly: `source_play_reference/` applies the script override and
records it in the fingerprint; the original XML reference is unchanged.
Godot physics is unchanged. The override improves source glide/idle and still
leaves large source turn-rate errors. It also reduces crouch travel, so the
phase/crouch quality and actual glide distance must be reported separately.

The low-parameter command-conditioning search changes only an internal affine
layer on the three velocity-command features. Public observations and requested
velocities used for evaluation do not change. The exported ONNX includes the
adapter, source hash and coefficients, and its 10,000-input parity is exact.
The first probe's decimal filenames collided; it is explicitly invalidated and
re-run with unique filenames under `conditioning_probe_v2`. These scans are
development diagnostics, not evidence of general task completion.

An additional architecture hypothesis addresses the residual bound: on the old
walking policy's idle trajectories, 57.7% of per-joint standing-teacher action
differences exceed the 0.2-radian bound. `distill_walk.py` therefore teaches a
full-network increment from real standing/old-walking demonstrations generated
on separate training seeds 60000–60002. Expert switching is confined to data
collection. Physical evaluation executes the single exported student for the
whole trajectory, including idle and transitions. A held demonstration seed
checks supervised error; the existing physical development suite checks actual
behavior. Final evaluation seeds 1000+ remain unused.

Protocol v5 closes an idle loophole in locomotion evaluation: a small nonzero
velocity/yaw error can accumulate large drift over ten seconds. After allowing
0.5 seconds of braking in each idle interval, completion now also requires less
than 5 cm travel and 5 degrees yaw drift. Original v4 results remain intact;
`rescore.py` writes adjacent `summary.v5.json` files from their saved traces.
Kicks, roulade, standing and posture outcomes are unchanged. In-progress v4
experiments retain their recorded version and are rescored before comparison.

The 2,000-step distilled walking student achieves six clean idle episodes under
that stricter gate, with negligible travel and mean yaw drift about 0.3 degrees.
Its source gait still under-tracks translational commands. Unbounded 2x command
conditioning improved slow walking but caused falls at the fastest test; those
candidates are not suitable for promotion. Bounded conditioning on the distilled
student is evaluated separately across the complete development schedule.

New PPO logs separate physical reward from time-limit value bootstrapping and
include critic error, explained variance and increment size. Earlier continuous
task logs reported the bootstrapped training target as `reward`; physical
outcomes, rather than those reward curves, have always determined selection.

## Native roll review and alignment guard (19:39 UTC)

The reverse curriculum produced a cold-start roll that recovered to a stable
stance, but native video exposed a 155-degree change in final heading. The
corresponding source MuJoCo roll ends about 14 degrees from its initial heading.
It is recovery progress, not a satisfactory straight forward roll. Also, raw
unwrapped Euler yaw during inversion is misleading (the source trace reports
374 degrees while its actual final heading error is only 14 degrees).

Protocol v6 therefore adds an explicit wrapped final heading error and requires
it below 30 degrees for roulade completion. All six source development rolls
remain within 3–15 degrees. Other task criteria are unchanged from v5. Archived
traces are rescored into `summary.v6.json`; earlier outcome versions are retained.
The next roll trial adds a recorded aligned-landing reward, excess-rotation
penalty, leg-only final-pose term and the upstream bilateral consistency loss.
The final tucked head remains available as a physical cue distinguishing the
end of the maneuver from its initial standing pose.

A direct bilateral-mean ONNX ensemble of the factory and first recovery actor
was also tested; neither completed the roll. These are negative architectural
probes, not replacements for training. A three-iteration smoke run verified
the new regularized PPO path and unchanged export tolerance.

The distilled walker with bounded internal conditioning completes 42/72 strict
development episodes, has no falls, and retains six clean idle cases. It still
under-tracks fast travel and lateral motion. The first PPO continuation failed
before collecting data because nested ONNX metadata keys were duplicated;
the exporter now replaces current-stage metadata uniquely, with a nonzero
second-adaptation parity test. The continuation is retried as `wd02`.

The first alignment trial (`r04`) exposed another scale issue: an unbounded
squared excess-rotation penalty dominated the return after repeated rolls
(mean term around -157 versus ordinary positive terms around 1–3). It was
stopped after four minutes. `r05` restarts from the same `r03` parent using a
penalty normalized by pi and capped at one before its recorded multiplier.
Physical rewards and critic fitting returned to a useful range. A regression
check prevents this penalty from growing without bound.

## Interactive walking and bounded-roll result (20:04 UTC)

A small recorded command coupling (internal yaw command += 0.9 * forward
command) on `wd02` iteration 80 improves strict development completion to
45/72, score 0.5931, with zero falls. It corrects backwards heading and preserves
clean idle/turns. Fast travel, lateral travel and stop transitions remain weak.
`wd02` continued PPO eventually regressed to 17 falls in its iteration-154
development evaluation despite growing training reward. Later weights are not
promoted. This illustrates why checkpoints are selected by physical outcomes.

`wd03` starts from the verified coupled candidate. Training-only interactive
command tapes replace 25% of constant/standard-sequence episodes: seeded
0.5–2.5 s segments, bounded continuous twist commands, 0.1 s ramps and idle
segments. Both engines receive identical tapes; standing entry and action
history remain real. The existing fixed development/final tests are unchanged.
An explicitly recorded yaw reward variance is tightened from 0.18 to 0.04.
Contract checks cover reproducibility, bounds, ramps, both backends and reset
back to ordinary commands. A short PPO/export/evaluation smoke run validates
the complete path before the longer trial.

`r05` fixes numerical reward domination but still completes no normal-entry
rolls in three checks: trajectories repeat several revolutions. It is stopped
for the next architectural trial. `r06` uses full-network adaptation from
`r03`'s actual recovery parent, rather than inheriting `r05`'s degraded policy.
The 30-degree final heading guard is retained.

## COM velocity correction (20:07 UTC)

A fresh independent kinematic check found that `mj_objectVelocity(mjOBJ_BODY)`
already reports the body inertial COM velocity. The legacy state transfer
incorrectly treated it as the regular body-origin velocity and added omega ×
(xipos - xpos) a second time. At a 4 rad/s forward angular velocity this created
0.091 m/s trunk and 0.131 m/s jaw transfer errors. The MuJoCo COM Jacobian times
qvel matches the unmodified API velocity to numerical precision on every body.
The engine source explicitly distinguishes BODY (xipos) and XBODY (xpos):
https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_core_util.c

The backend transfer and reference COM metric now use the unmodified BODY
velocity; the ordinary backend state also stops using subtree-COM `cvel` as
trunk velocity. A random articulated-velocity regression checks every body's
linear/angular transfer against its COM Jacobian. Normal cold/handoff Godot
rollouts start from zero angular/joint velocities, so their physics and
outcomes are unchanged. Source mid-roll starts are materially affected.
`r06` is stopped and repeated as `r07_com_transfer`; previous curriculum runs
remain documented negatives under their original reset convention.

Protocol v7 keeps v6 outcome thresholds and records the corrected COM velocity
contract. Godot archives can be rescored unchanged; old MuJoCo traces require
fresh reference rollouts rather than silently relabeling the wrong velocities.
Both the source XML and source-play wheel-friction references are re-run.

## Explicit time input for a finite roll (20:15 UTC)

A separate architecture experiment (`r08_time_input`) gives the learned actor
a monotonic elapsed-time input in the otherwise zero first command slot:
cmd[0] = clip(seconds_since_trigger / 5, 0, 1). Input/output sizes stay 61/14.
This lets the policy distinguish standing before its roll from standing after
it. The ONNX declares `sim2sim_time_input_s=5`; training, independent evaluation,
native capture and interactive play read the same model contract. Factory and
previous policies without this metadata continue receiving their original
commands. The five-second physical task and completion thresholds do not change.

The frozen parent sees zero command slots inside the ONNX; only its trainable
increment receives elapsed time, with a unit normalization denominator for that
new feature. The critic uses the same scaling. Initial output equals the
zero-command parent, and a nonzero full-network update passes the unchanged
export tolerance. Training-only mid-roll states retain source elapsed time;
normal resets and real standing handoffs restart it at zero. Regression tests
verify that play and training count the first action at exactly time zero.
No pose sequence is played back: all 14 action offsets come from the exported
network under native Jolt dynamics. `r07` remains the no-time-input comparison.

Walking's coupled candidate also completed 119/192 additional development
episodes (seeds 110–117, both entries), with zero falls. Idle completed 15/16;
transition stopping still fails. These additional development samples do not
replace the final unseen-seed evaluation.

## State aggregation and motion feedback (20:23 UTC)

`distill_transitions.py` collects walking states under the student itself,
then labels zero-command states with factory standing and moving states with
the verified parent walker. It aggregates two successive student distributions
from training seeds 80000+, including random command tapes and real handoffs.
The exported single student is evaluated for every full trajectory. Balanced
idle/moving supervision is a training objective, never a runtime policy switch.
The first 1,000-step student completes 48/72 development cases with no falls.

A bounded roller yaw-gain search on separate development seed 300 tested
1, 1.5, 2, 2.5, 3, 4 and 5 times the old policy's command. Larger gains caused
falls; gain one remains best, so no conditioned roller is promoted. The full
old-roller comparison is retained under `roller_conditioning/best_full`.

For the next timed-roll trial, `roll_motion.py` records eight successful original
MuJoCo training-seed rolls, builds a median joint-pose / gravity-direction /
height / net-rotation reference, and supplies only reward errors to PPO.
The reference never sets native states or outputs actions. Its hash and source
physics are recorded. A regression and short PPO/export run check retained
curriculum time and nonzero updates. `r09_timed_motion` will compare this dense
feedback with the time-input-only `r08`. Evaluation still imports neither
training rewards nor motion-reference code.

The machine has 20 CPU threads. A third bounded lane runs `r08` with eight
worlds and 1,024 steps per update, preserving the same 8,192-sample update size
while the two existing lanes continue. CPU throughput is monitored; this is
independent simulator processing, not additional decision-making agents.

## Native landing demonstrations and explicit neural experts (20:43 UTC)

The on-policy walking distillation finishes at 54/72 basic development cases
and 142/192 additional cases, without falls. It solves all 16 extra idle cases
(max travel about 1 mm, max yaw 0.5 degrees), and 14/16 command-sequence cases.
Fast forward and lateral tracking remain weak; fast-forward yaw also needs
polish. This is a substantial improvement over the previous walker, not a
claim that every requested twist is achieved.

Native roll diagnostics show a first revolution followed by repeated launches.
For training demonstrations only, the partially adapted roll expert handed
control to the original balance expert after actual inversion and a supported
upright return. Nine of 24 training-seed demonstrations complete the strict
physical task. A time-aware monolithic student trained on these demonstrations
does not yet reproduce reliable recovery. A gated-increment ablation preserves
the parent's launch exactly for the first 1.8 seconds and also remains
unqualified. These teacher demonstrations are not counted as candidate results.

A further architecture probe (`roll_experts.py`) explicitly packages both
frozen neural experts and their time-input blend into one ONNX. It is a
**two-expert actor**, not a claim of monolithic distillation. The simulator
executes that single graph throughout every five-second evaluation, with no
external policy handoff or physical assistance. The ONNX records both expert
hashes and its blend interval; parity over 10,000 inputs is exact.

A 2.30–2.45 s blend achieves six stable single-revolution landings on the basic
development set, but only one also finishes within 30 degrees of the initial
heading (the others are about 31–78 degrees away). Thus it improves recovery
but is not reliable enough to promote. An alternative 2.10–2.25 s blend fails
the normal development set. `r10_expert_residual` trains a recorded, time-gated
residual on the stronger two-expert actor. `r08` and `r07` are stopped after
several consecutive unqualified checks; `r09` retains the separate source-motion
feedback hypothesis. All architectures retain the same physical outcome gates.

Time gating is part of the exported neural graph. Resume restores its recorded
configuration and rejects an incompatible change, while old checkpoints with
no gate remain compatible. Tests verify exact unchanged launch actions and
nonzero late adaptation in the actual exported model.

## Heading error is introduced before landing (20:47 UTC)

A diagnostic based on the trunk lateral axis, which remains meaningful through
a sagittal inversion, locates most yaw drift around 1.0–1.5 s. For cold seed
100 the error grows from -9 to -41 degrees before the late correction gate can
act. This motivates `r11_early_yaw_control`: a small 0.1-radian residual is
enabled from 0.2–0.5 s and a bounded world-vertical angular-velocity cost is
added. A regression confirms that this cost leaves pure forward rotation
unpenalized. The late-only `r10` remains a separate comparison.

`r09` is stopped after three normal-entry zero-completion checks; its dense
reference experiment is an unsuccessful bounded-budget trial, not a claim
that motion-reference learning cannot work. The current strongest roll
architecture has stable single-revolution recovery in all six basic cases;
only initial-heading recovery prevents most from qualifying.

## Direct heading observability and portability (21:06 UTC)

The heading objective previously depended on a trigger-relative angle that
neither actor nor critic received directly. Gyro/action history can help
implicitly, but identical balanced poses can still have different heading
returns. A separately declared variant supplies sin/cos of the trunk lateral
axis's angle relative to the trigger frame in cmd[1:3]. This axis stays
meaningful through a sagittal inversion; tests cover an entire forward turn
and reflected headings. The actor still has 61 inputs and 14 outputs, but
this variant explicitly changes two previously unused command slots.

`sim2sim_heading_input=lateral_axis_sin_cos` records that contract. A wrapper
masks the new slots for its frozen parent, and the trainable actor/critic use
unit scaling. Bilateral consistency negates heading sine and preserves cosine.
Play, generic rollout, native capture and independent evaluation derive the
same input from actual simulator orientation and the trigger frame. Consumer
code must honor the model metadata; the models do not infer elapsed time or
initial heading from a bare zero-filled observation. No physical control is
performed by the input adapter.

The original time-mixture parent repeats identically in two fresh full
development suites. Its heading-ready wrapper also gives identical trajectories
and outcomes (1/6): enabling the input itself does not improve the model.
A tiny trained heading-aware increment changes the contact trajectory enough
to achieve 4/6 basic and 8/16 extra cases. This sensitivity is why broader
unseen tests remain essential. `r12` is queued from that verified short-run
candidate. Meanwhile, early yaw-spin adaptation without direct heading input
(`r11` iteration 50) reaches 5/6 basic and 10/16 extra cases. Neither is yet
qualified at original reliability.

The walking candidate's fast-forward yaw bias is corrected by a recorded
input hinge: internal yaw += -3 * max(public_vx - 0.3, 0). Slower commands are
bitwise unchanged; 10,000-input parity is exact. The full suite remains 54/72
with improved continuous score 0.6146 and no falls. Absolute fast/lateral
velocity tracking remains below the requested target.

Measured CPU inference is comfortably within the 20 ms control interval:
the distilled walking graph is about 7.25 MB and 0.17 ms p99, and the two-expert
roll graph about 4.87 MB and 0.106 ms p99 in a 2,000-call one-thread sample.
These are inference-only timings under concurrent training, not full simulator
round-trip latency.

## Checkpoint continuity and worker recovery (21:22 UTC)

The actor/critic optimizers and global Python/NumPy/Torch RNGs were already
restored, but the vector environment's independent generator and episode
counter restarted. Checkpoints now also preserve that generator and counter.
Resumes intentionally begin fresh physical episodes; they do not pretend to
restore Godot's hidden solver/contact state. Legacy checkpoints use a disjoint
episode-seed range above their maximum possible prior episode count. Configs
record this distinction and hashes of observation and backend transfer code.

A native reset test verifies that a restored generator produces the same next
condition, RNG state and fresh observation. End-to-end save/resume smoke runs
advance iteration 2 to 5 and episode counter 2 to 4. A legacy timed-model resume
retains its 0.2–0.5 s gate and advances into a new seed range (counter 1030).

At 21:21, lanes A/B and their coordinators were absent without Python errors
or completion records. The original logs and checkpoints remain unchanged;
wh01 resumes at iteration 30 for 15 minutes, and r11 at iteration 70 for
25 minutes in new run directories. Lane C continued normally. The precise
external process-exit cause is unknown. All runs retain the original 8-hour
wall-clock cap; recovery does not extend it.

For r12, the stronger r11 iteration 50 parent replaces the earlier 4-step
heading-aware candidate: 5/6 basic and 10/16 extra cases versus 4/6 and 8/16.
Its wrapper masks the new heading slots for the parent while preserving time.
The new heading-aware residual uses the same small bound 0.1 and std 0.02 as
r11, making the heading-observability comparison more focused.

## Bilateral roll ensemble and real skill exits (21:31 UTC)

A renewed symmetry probe on the much stronger r11 iteration 50 actor succeeds
where the early factory/recovery probes failed. The single ONNX averages the
actor with its exact reflected counterpart; its architecture is explicitly
a bilateral ensemble containing four frozen neural expert branches plus the
learned corrections. Ten thousand input checks show zero export discrepancy
and zero bilateral discrepancy. Time remains an explicitly required input.

After 6/6 exploratory training-seed rolls, it passes 20/22 development rolls
(100–102 and 110–117, both entries), continuous score 0.9776. All 22 complete
one revolution and stand; two miss the 30-degree heading gate at 31.65 and
33.05 degrees. It is a stronger candidate, not yet proof of source reliability.
All 22 additional three-second handoffs to the factory standing actor remain
stable without resetting physics or action history. Maximum post-exit motion
is 3.8 mm and 2.08 degrees. Primary metrics match the earlier standalone
evaluations exactly; post-exit assistance cannot turn a primary failure into
a success. This separate check is implemented in research/handoff_check.py.

Native frame inspection and trace analysis reveal remaining quality gaps:
the ensemble settles around 2.83 s versus original MuJoCo 1.64 s, and its
rotation frontier is 7.18–7.90 rad versus 6.31–6.46 rad. It briefly leans
forward again before recovering. These continuous gaps remain visible even
when the binary task gate passes. Earlier internal balance blending was
therefore tested at 1.75, 1.9, 2.0, 2.1, 2.2 and 2.3 s on separate training
seeds. The first four degrade recovery; 2.2 passes 5/6 and 2.3 passes 6/6.
No timing change is selected. A separate factory-roll-plus-balance bilateral
ensemble at six earlier timings also passes 0/6 throughout; standing after
an incomplete maneuver does not count.

The mirror exporter now preserves phase sine for phase-conditioned maneuvers
and heading cosine for heading-aware rolls, matching the actor-training
reflection. Unit checks cover those input semantics and nested blend retiming
against independently rebuilt graphs. Existing right-kick transforms are
unchanged. The fast-walk yaw correction passes an additional 16 native tests
without falls (yaw RMSE 0.059 rad/s); speed remains 0.172 m/s and still fails
the strict 0.4 m/s target.

## Wider roll validation and source-quality kick objective (21:36 UTC)

On 60 additional development episodes (seeds 200–229, both entries), the
bilateral r11 iteration 95 actor achieves 57/60, with all 60 single and
standing. The early heading-aware r12 iteration 24 actor passes only 36/60
despite 6/6 basic success; its bilateral ensemble improves to 59/60, again
all 60 single and standing. Heading mean/p95/max are 14.40/25.45/34.12 degrees.
These seeds remain development data, not the reserved final holdout. The
large basic-to-wide gap reinforces why six-episode success is insufficient.
R13's pending parent is updated to that strongest heading-aware ensemble.

K07's kicks are reliable but not yet equivalent in motion quality: body yaw
is about 49 degrees versus 19.4 degrees for the four successful source left
kicks, and action variation is also larger. Source successful left ball-speed
mean is 0.689 m/s (right 0.531 m/s); K07 is about 0.85 m/s. K09 therefore tests
a small new residual with a recorded 0.7 m/s reward target, world-vertical spin
cost and stronger heading/action-smoothness penalties. The evaluator and its
full-duration no-fall gate do not change. K08 retains its earlier objective
as a separate comparison. Unit checks confirm the optional target caps the
speed incentive and the spin term preserves pure forward-roll rotation.

The actual interactive policy consumer was also measured, not only the
research runtime: walking p99 0.269 ms and bilateral rolling p99 0.297 ms
over 300 calls under training load. No inference-thread change is necessary.

## Continuous gameplay exposes deployment gaps (21:58 UTC)

A real PlayBrain sequence runs walking, braking, turning, ground pick, left
kick, roll, sit/rise and right kick in one native ball scene. Physics and
action history stay continuous, with zero resets after the initial start.
Two deployment bugs were exposed independently of the policy task gates:

- The unified walking student owns idle/braking, but its missing sidecar let
  play switch immediately to the factory standing actor on release. The first
  braking transition fell. The candidate now declares
  `sim2sim.use_stand_policy=false` in its manifest, and the sequence harness
  honors the same policy metadata as actual play. With only this change,
  walking, pick and left kick recover, but rising still falls.
- The sit toggle selected sitstand with a stand command, then locomotion
  selection immediately replaced it in the same tick. PlayBrain now retains
  the sitstand actor for a three-second rise before accepting locomotion or
  another maneuver. This is the policy's actual rise command, not a body-pose
  reset or external force. Controller tests cover the full duration and reset.

With both fixes, all three initial continuous sequences have no unintended
ordinary-action falls; all three ground picks and all six kicks succeed.
Rolling remains only 1/3 under these real histories, despite stronger isolated
results. A newer non-averaged r11 iteration 145 actor passes 60/60 standard
development cases (heading mean 10.23 degrees, original 9.70), but also only
1/3 continuous sequences. A simple deployed-walker idle warmup is insufficient
to reproduce the gap: r12's ensemble passes 30/30 such entry cases. The ball
scene itself changes contact trajectories slightly but all six isolated
cold/idle-entry probes succeed; it is not established as the failure cause.

R13 therefore uses real native prefixes from the declared deployment bank:
idle, walking, turning, pick, either kick, and the entire pre-roll game
sequence. The prefix actors control only the starting-state preparation; the
roll candidate exclusively controls its full five-second maneuver. Half of
training episodes remain cold starts. Training uses the actual ball scene,
and non-kick resets keep its ball away until a kick trigger. All actor and
walking-manifest hashes are recorded. Standard physical_tasks_v7 evaluation
keeps its original scene, cold starts and standing teacher; deployment entries
and continuous sequences are separate, explicitly identified checks.

The initially proposed parent was r11 iteration 145 with a parent-preserving
heading wrapper. Before R13 launched, the actual parent was changed to the
frozen r12 iteration 121 actor (`integration/r12_best_2159.onnx`): it passed
7/9 continuous development sequences versus r11 iteration 145's 5/9. R13's
config and source hash record this final choice.
The sequence check also records whole-transition drift and, separately, the
existing locomotion gate's 0.5-second braking allowance. Early transient yaw
(roughly 18–20 degrees after keyboard forward release) remains visible and is
not erased by the steady-state check.

## Roller command-contract correction (22:23 UTC)

Pinned upstream `microduck_velocity_rollers_env_cfg.py` describes command x
as push/coast/brake, not desired forward velocity, and the third command as
relative heading error. The implementation in
`RelativeHeadingVelocityCommand._update_command` computes target minus actual
yaw (positive counterclockwise), despite an inconsistent clockwise docstring.
The old generic velocity/yaw-rate roller protocol therefore cannot select
roller replacements. Its results, including wh01/wh02, remain available as
negative experiments under their original protocol. Other eight skills keep
physical_tasks_v7 without changes.

New `roller_throttle_heading_v1` evaluates push, coast from 0.3 m/s, brake
from 0.3 m/s, two heading changes and push/coast/brake sequences. It preserves
full-duration no-fall and final-standing checks. Coast permits passive drift;
braking must end below 0.05 m/s without sustained reverse motion. Heading
cases feed the actual wrapped target-minus-current error and require final
error below 0.15 radians. The reward explicitly penalizes reverse speed,
unlike the upstream brake reward's forward-velocity clamp. The original
checkpoint nevertheless runs backward under negative input in BOTH engines.
The pinned current training config and the checkpoint's exact historical
training revision are not proven identical; a reward loophole is a hypothesis,
not an established explanation of this artifact.

Fresh 18-episode results: original MuJoCo XML 9/18, original MuJoCo inference
profile 12/18, original Godot XML 12/18, previous Godot adaptation 6/18.
Original Godot and MuJoCo inference-profile scores are 0.5986 and 0.6021.
Their shared failures are braking and the combined sequence. The new native
PPO smoke reproduces the original 12/18 with exact initial export parity;
38 research tests pass. WH03 starts a distinct 35-minute, eight-environment
residual trial under the corrected contract, using unchanged Godot XML physics.

Upstream `infer_policy.py` separately applies 0.003 N m passive-wheel bearing
friction, absent from its training XML. An isolated Godot `source_play` scene
models the same dissipative constraint using a zero-speed bounded hinge
motor. Its torque-to-impulse conversion follows Godot's Jolt hinge source.
The standalone bearing check has inertia 0.0001 kg m²: angular speed falls
from 40 to 9.999975 rad/s in one second, matching the expected 30 rad/s²
deceleration. No support force, position target or default physics change is
introduced. This explicit profile also gives 12/18, score 0.5725; it corrects
coasting decay but does not solve active reverse motion. Both physics profiles
and their hashes remain separate. It is not selected as a default change.

## Missing ball-scene STAND changes the physical feet (22:36 UTC)

Keyboard checks exposed a systematic scene gap: the current walking actor's
six-seed forward yaw averages 4.38 degrees in the normal scene but 21.39 in
the actual ball scene. The robot specs have identical robot masses, inertias,
joints and MJCF geoms. Comparing the generated collision resources reveals
different left/right foot hulls. `scene_ball.xml` has no keyframes, whereas
`scene.xml` has STAND. The converter's support-patch approximation silently
used qpos0 when STAND was missing; some hull boundaries differ by almost
9 mm. Thus scene construction, not just entry history, caused distribution
shift between training and play.

The converter now falls back to the exact named Microduck STAND joint values
when all fourteen expected joints exist. It does not apply that pose to
unknown robots. A separate `microduck_ball_stand_fix` robot/scene preserves
the original scene for running trials and legacy comparisons. Both corrected
foot vertex arrays exactly equal the normal scene's arrays (max error zero).
Four conversion tests pass, including equivalent foot geometry with/without
ball keyframes. This restores consistency of the existing foot proxy; it
does not assert that the proxy equals MuJoCo's contact model.

Without changing the walking actor, the corrected ball scene reproduces the
normal scene's six-seed mean forward yaw of 4.38 degrees. Braking still needs
improvement. Corrected-scene kick checks: K07 5/6, K09 iteration 116 6/6,
factory left 0/6, factory right 2/6. K10 is therefore assigned the K09 parent
and this explicit corrected scene for both training and evaluation. All
earlier ball-scene results retain their original physics identity. Future
fingerprints additionally hash `robot.tscn`, since collision points are
stored there rather than fully represented in robot_spec.json.

Before the geometry finding, R13 iteration 25 passed 59/60 standard and 8/9
legacy-ball continuous sequences (the failure is 30.27-degree heading).
Iteration 37 passes only 7/9 despite a higher six-case score. Retiming the
blend from 2.3 to 2.2 seconds passes six training probes but degrades legacy
continuous sequences to 5/9; it is not selected. These findings remain valid
for their recorded scenes and do not establish corrected-scene performance.

Corrected-scene continuous checks then passed 9/9 for both r11 iteration 145
and r12 iteration 121, with no ordinary-action falls. R11's heading mean is
14.28 degrees (max 23.76), versus 17.91 for r12. R13 iteration 25 achieves
7/9 in the corrected scene, despite 8/9 in the legacy scene. R11 is retained
as the conservative corrected-scene roll parent.

At 00:11 UTC, the session has about 88 minutes remaining. K10 completed
30 minutes on canonical feet: iteration 81 keeps 6/6 physical successes and
reduces mean body yaw to 28.88 degrees and action variation to 0.097. Later
checkpoints regress in yaw despite the same binary success, so the earlier
checkpoint is retained. Its mirrored right actor is checked separately.
WH03's corrected roller contract ends at 12/18 and score 0.5933, below the
original actor's 0.5986; the original remains preferred. C01 keeps 6/6 crouch
success but increases tilt to roughly 9.5 degrees, so the original remains
the conservative choice there too.

The last short trials are R14 (22 minutes from r11 iteration 145, native
prefixes on canonical feet), K11 (22 minutes from K10 iteration 81, stronger
heading/smoothness objective), and WD05 (actual keyboard tapes added to broad
walking commands, canonical scene). WD05's smoke passes with exact export
parity. All final candidate selection must precede reserved holdout evaluation;
no holdout seed is used for optimization or repeated checkpoint selection.

## Candidate freeze (00:34 UTC)

Final development selection is the WHOLE controller bank: WD05 iteration 90,
R11 iteration 145, K10 iteration 81 left and its unscaled right reflection,
with original standing/pick/roller/crouch and the retained previous sitstand.
WD05's forward-key braking yaw after the existing 0.5-second allowance drops
from 9.59 to 4.90 degrees on six development seeds. Its broad task score is
slightly lower than the earlier student, but the selected bank passes all
four maneuvers in 12/12 complete continuous sequences, with zero ordinary
falls. The roll heading mean in those sequences is 10.34 degrees. R14
iteration 166 also passes those 12 sequences, but has lower wide standalone
reliability (59/60 versus R11's 60/60), so R11 remains selected.

Small kick action-offset contractions were tested INSIDE the ONNX graph with
unchanged PD/torque physics and exact 10,000-input functional parity. Gain
0.85 passes 59/60 left and 60/60 right and reduces speed/action variation;
gain 0.80 drops right success to 43/60. However, gain 0.85 right passes only
8/12 continuous sequences after sit/rise. Gain 0.90 with the original right
also introduces whole-bank failures. These attractive standalone results
are rejected in favor of the unscaled bank's verified transitions. K10 raw
left/right each pass 59/60 wide standalone development cases and both pass
all 12 selected-bank continuous sequences. The remaining isolated failure
risk is retained honestly for final holdout.

All 181 repository tests pass. WD05 stops at iteration 105 for final review;
R14 and K11 complete their 22-minute budgets. Candidate ONNX files and
sidecars are frozen into `results/research_20260910/delivery`, independently
of default policies. Trainable exports receive 10,000 random plus 10,000
realistic input parity checks, and the mirrored actor gets a separate
10,000-input reflection check. Final holdout is seeds 1000–1029, with both
cold and native standing entries, all original conditions, an entire
continuous play sequence and real keyboard tapes. No checkpoint is changed
after this holdout is opened. Results also include original MuJoCo and old
Godot actors in the same declared corrected scene, with both original XML
and inference-friction references retained for the roller robot.


## Export isolation audit (00:40 UTC)

The final original-model hash check detected a real test side effect: the
pre-existing `export_actor` implementation automatically copied EVERY export
into repository policies, including temporary `Walk_Godot.onnx` roundtrip tests.
The original walking actor was overwritten during the 181-test run. The
immutable research baseline and all selected candidates were unaffected.
The test export and sidecar are preserved in `test_export_incident/`; the actor
was restored from the exact baseline snapshot, and its sidecar from the paired
pre-existing `results/Walk_Godot.manifest.json` export (whose ONNX is byte-identical
to the baseline). The session did not record an initial sidecar hash, so we do
not claim byte-level proof for the pre-session sidecar. All 18 original ONNX
hashes now match the initial session record.

Exports now write only to the requested path. Explicit default destinations
still work; the separate publishing helper remains available for deliberate
promotion. The roundtrip test asserts no implicit publishing, and the runner
test explicitly exports into its temporary directory. Both affected suites
pass (four export tests and five runner tests). The audit is recorded in the
bundle without erasing its initial failed hash flag.

The nine frozen actors pass native play path resolution with no fallback,
61-to-14 shape checks, finite inference, walking idle ownership, and the roll's
explicit five-second time-input contract. All three learned checkpoint exports
pass 10,000 random plus 10,000 realistic inputs below 1e-5; maximum selected
export discrepancy is 1.91e-6. Right reflection is exact on 10,000 inputs.
Holdout seeds 1000–1029 opened only after the completed freeze; selection is
unchanged thereafter. Native Godot roll/kick and matching MuJoCo roll clips
are recorded in the delivery directory, using development seed 100 solely
for illustration.


## Phase-one holdout and user-authorized active-time extension

Phase-one holdout completed at 01:38:32 UTC, 31 seconds before the initial
wall-clock cutoff. Nine candidate task counts were: standing 60/60, walking
538/720, sitstand 240/240, ground pick 60/60, left kick 60/60, right kick 57/60,
roll 59/60, roller 115/180, crouch 60/60. Continuous sequences: pick/left/right
30/30, roll 28/30, zero ordinary falls. All three independent right-kick
failures were cold-start falls. The three roll heading failures across the
standalone and continuous checks still completed one revolution and stood up.
Successful roll recovery averaged 2.79 s versus source 1.62 s. Full figures,
source comparisons and their limitations are in RESEARCH_RESULT_20260910.md.
No phase-one selected actor was changed after opening those test seeds.

The user then clarified that eight hours means active work; disconnected hours
are excluded and should be made up. The session records two observed assistant
gaps, approximately 22:37–00:09 and 00:49–01:54 UTC (157 minutes total), and an
adjusted cutoff of 04:16:04 UTC. This is an approximate observed-disconnection
accounting, not a claim that background compute stopped throughout those gaps.
Original timestamps and completed background jobs are retained. Phase-one
artifacts are sealed by hash. Supplementary development seeds are 300–319;
new final seeds are 2000–2029, untouched until the next selection freeze.

WD06 gives a fresh bounded residual greater capacity and more high-speed and
lateral-command samples while preserving keyboard/idle training. KR03 adapts
the actual right-kick actor, with real controller prefixes including sit/rise;
it does not assume that the reflected left actor is physically symmetric.
R15 starts from a conservative 2.20–2.35 s expert blend and adds time/heading-
aware residual recovery learning with a weak original motion reference.
Earlier blend probes 2.10 and 2.15 pass only 2/6 and 4/6 new development cases;
2.20 passes 6/6. A launch-time gate preserves the first 0.8 s of the R15 parent.
All three new training runs have fixed 45-minute budgets. Initial exports
retain the parent exactly, and the right-prefix smoke passes physical 6/6.

The training runner now records configurable development seeds. Native prefix
training supports kicks as well as rolls, and the bank optionally includes
sit/rise. All 38 research regression tests pass. Final assessment can now run
independent suites in separate processes, retaining the exact same episode
function, conditions and gates; a two-process native/MuJoCo smoke passes.
This reduces evaluation overhead and avoids a one-hour serial suite queue.
Descriptive schema-2 packaging also records effective runtime contracts and
checks them unchanged before/after, with all ONNX hashes and inference intact.


## Walking-memory architecture comparison (02:18 UTC)

WD06's larger residual and higher velocity reward regress from the parent's
54/72 development cases to 50/72, 47/72 and 49/72. Sampling is stopped for
review; its checkpoints and finalization remain preserved. A separate internal
high-speed command-hinge scan also fails: gains 1/2/4 do not improve velocity,
and gain 8 falls in all six cases. Each exported adapter has exact functional
parity and preserves ordinary keyboard forward commands at or below 0.3 m/s.
It is a negative capacity/conditioning result, not a selected controller.

WD07 tests explicit memory of straight-motion yaw drift. The feature at obs[55]
is a bounded integral computed from existing gyro and projected gravity only;
commanded turns clear the reference. No true linear velocity, world position,
extra force, or external corrective action is used. The parent graph masks that
slot and remains exact; a new residual learns whether to react. ONNX metadata
explicitly declares the stateful contract. Training samples and deployment use
the same memory implementation, reset it per episode/policy switch, and avoid
advancing it on duplicate reads at one simulation time. The time step is the
existing 50 Hz control rate. All 185 repository tests pass, including three
memory contract/export/native-equivalence checks. The training smoke passes
all 12 idle/sequence cases with exact initial export parity. WD07 receives a
40-minute budget and will be judged on full physical and keyboard behavior,
not on whether the added memory is technically connected.


## Supplementary observability and optimizer comparison (02:29 UTC)

Wider new development: original right and KR03 iteration 26 both pass 39/40;
R11 passes 40/40, R15 iteration 15 passes 39/40. Merely advancing the entire
internal neural clock also fails (5/6, then 1/6 for the stronger warps), despite
exact 1,000-input functional checks. Physics time and full rollout duration
never change. No timing or high-speed adapter is selected from these probes.

The IMU-memory contract is extended to kicks. A yaw-invariant zero-command
actor sees angular rate but cannot directly distinguish equal stationary poses
at different accumulated headings. Explicit gyro/gravity history is therefore
a testable observability hypothesis; no claim is made that it has already
improved quality. KR04 tests this information with real kick prefixes and a
30-minute budget. A native smoke passes 6/6 and the state-consistency test now
covers both walking and right kick. Policy switches clear each prefix actor's
memory, including switches between two actors that both use it.

R15 actual KL remains about 0.0005 against the 0.015 target; the previous
scheduler only reduced learning rate. The locally installed rsl_rl adaptive
scheduler also increases it when KL is small. An explicit, optional upward
schedule is added with a much smaller 0.0003 cap, 1.2 growth, and the existing
rollback/overshoot guards. R16 branches from R15 iteration 60 and WD08 from
WD07 iteration 26, preserving actor, optimizer and episode-generator state;
each receives 30 minutes. Their parent trials stop and finalize separately.
This is a recorded learning-rate comparison, not an unlogged resume change.
The final assessment runner skips a redundant phase-one comparison only when
the actor is byte-identical, explicitly recording those identities instead of
counting them as independent tests.

## Supplementary deployment and wider checks (02:44 UTC)

The complete suite now passes 186 tests. A new reflection check verifies that
the declared yaw-memory feature changes sign consistently with a mirrored IMU
history. Intermediate walking exports now carry their own idle-ownership
sidecar; previously this was only attached at final packaging. Existing running
WD08 checkpoints receive the same explicit sidecar before keyboard evaluation.
Neither change alters the native physics or physical success thresholds.

Wider development remains necessary: R16 iteration 89 passes 6/6 initial cases
but only 37/40 across seeds 300–319, against R11's 40/40. Successful recovery is
2.782 s versus 2.778 s, with worse mean final heading (13.86 versus 8.20 degrees).
KR04 iteration 26 remains 39/40, equal to its parent, with body yaw 32.86 versus
30.85 degrees. Neither checkpoint currently qualifies as an improvement.
WD08 iteration 53 keeps 54/72 fixed-command cases but its six-seed keyboard
forward drift is 6.09 versus 4.15 degrees; backward tracking also regresses.
Lower last-second angular-rate error alone would conceal accumulated drift.

The read-only results/analyze_results.py artifact records quality diagnostics
alongside unchanged success flags. Roll recovery is the earliest post-inversion
time whose remaining 0.5-second windows all contain at least 90% standing
samples (tilt below 15 degrees, height above 0.08 m, at least one foot contact).
It reproduces the phase-one 2.78915-second candidate result. This diagnostic
does not replace the physical completion protocol or discard failed episodes.

## Supplementary selection freeze (03:00 UTC)

All supplementary training has ended. WD08 is stopped at iteration 121 after
its interval evaluation falls to 44/72. KR04 ends at 71 and R16 at 147 under
their recorded 30-minute caps. The final small-dev means remain unfavorable:
right yaw 31.76 degrees and roll final heading 19.34 degrees. There are now 64
completed run records, ten beyond phase one (seven training/resume runs and
three smokes), not 64 independent successful models.

R16 iteration 120 receives an additional actual-controller comparison over
seeds 300–311. Both banks pass all four maneuvers in all 12 continuous tapes,
with no ordinary falls. R16 improves continuous roll heading to 10.67 versus
17.31 degrees, but its standalone mean is 14.22 versus 8.20 and recovery is
2.786 versus 2.778 seconds. This context-dependent tradeoff does not demonstrate
an overall advance toward source quality; the simpler, already selected R11 is
retained. KR04 iteration 50 also falls to 38/40 versus the parent's 39/40.

The final nine-slot selection therefore retains every phase-one ONNX. A new
delivery_v2 bundle is frozen and independently export-verified before opening
seeds 2000–2029. Its improved descriptive manifests do not change inference or
physics. Phase-one artifacts remain sealed; supplementary experiments, failed
adapters, wider and keyboard comparisons, and the selection rationale are all
retained. A fresh independent assessment is still required even though the
final selected actor bytes are unchanged.

## Post-freeze kick recovery diagnostic (03:13 UTC)

The phase-one right-kick failures first contact the ball correctly at
0.10–0.12 s; tilt exceeds 20 degrees only at 2.02–2.44 s, and 70 degrees at
2.90–3.44 s. This motivates a bounded development-only neural-recovery probe
while delivery_v2's independent evaluation runs unchanged. Four explicit
kick/stand ONNX mixtures start blending at 0.4/0.8/1.2/1.6 s over 0.2 s. Each
executes a full native five-second episode, with elapsed time supplied by its
isolated probe harness. No shared runtime or physical evaluator is edited.

All four pass 6/6 development cases but worsen body yaw to 49.72–51.04 degrees
against the parent's 30.67. All are rejected; no final selection changes and
no test seeds are used. Functional checks are exact over 1,000 inputs. A scratch
filename collision in the probe initially reused 0.onnx/1.onnx; the four graphs
are reconstructed to distinct names and every SHA-256 is verified against all
recorded episode hashes. Original summaries remain intact alongside explicit
relocation records; no physical result is rerun or replaced. This is an
additional negative architecture probe, not another PPO training run.

## Last bounded recovery-learning hypothesis (03:20 UTC)

With working-time budget remaining, KR05 receives a separate 20-minute run to
test whether the rejected raw expert mixture can learn a better late recovery.
Its clock and relative-heading features are explicit. An isolated harness
subclass enables those inputs for kicks using the existing time_command; shared
World/play/evaluation code and delivery_v2 remain unchanged. The harness path
and SHA-256 are stored in every checkpoint configuration. Its residual is
time-gated from 1.2 to 1.6 s, so the original successful launch is preserved;
1,000 input checks before 1.2 s have exact zero error against the selected kick.

Promotion criteria are registered before the first development evaluation:
40/40 wider standalone cases, mean/p90 yaw no worse than the selected parent,
bounded successful action variation, and 12/12 actual continuous maneuvers with
no ordinary falls. Any qualified separate candidate would require verified
deployment and new independent seeds 3000–3029 before the adjusted cutoff.
Otherwise delivery_v2 remains the final bank. This late trial does not permit
changing the already-open 2000–2029 selection or using those results to tune it.


## Phase/heading right-kick experiment and deployable runtime (04:01 UTC)

KR05 completed 54 iterations / 220,288 samples in 1,212.68 seconds. Its late
stand-expert prior retained approximately 48.48 degrees of yaw on six development
cases; rejected. KR06 instead retained the selected right-kick anchor, declared
five-second time plus lateral-axis relative-heading inputs, and activated a
bounded residual from 0.12 to 0.30 seconds. Final iteration 43 completed after
172,832 samples / 909.12 seconds. Wide development: 40/40 versus parent 39/40,
mean yaw 30.4354 versus 30.8525 degrees, p90 33.2300 versus 33.6062. Earlier
checkpoint mean-yaw changes were partly failure-trajectory composition, not a
reliable quality gain. All 12 continuous development sequences passed every
maneuver with no ordinary falls.

The production world, actual play, continuous evaluation, and native training
prefix now support declared timed kicks while preserving existing actors. The
clock resets on maneuver entry and relative heading captures entry orientation.
The complete unittest suite passes 187 tests in 75.361 seconds. A separate final
selection is frozen before any 3000-series seed is opened. It must be compared
against V2 on the same fresh seeds and can be rejected without selecting another
checkpoint from those final data. V2 remains sealed. See finalize_late.py, the
preexisting late_kick_experiment_protocol.json, and late_selection_decision.json.


V3 final-harness setup correction: source_play was accidentally passed to three
Godot suites. All 180 attempts raised Unknown reference profile before reset or
any physics step; no traces were produced. Their raw summaries remain at the
original paths. correct_late_setup.py records the error before launching separate
*_xml directories with the proper fixed Godot profile. No checkpoint, seeds,
physical parameters, gates, or started physical rollouts were changed. Native
continuous and MuJoCo evaluations proceed independently.


## Final late holdout and delivery decision

{
  "recorded_unix": 1789013296.1755388,
  "seeds": [
    3000,
    3029
  ],
  "invalid_setup_attempts": 180,
  "valid_standalone_cases": 240,
  "standalone": {
    "candidate": {
      "successes": 60,
      "episodes": 60,
      "errors": 0,
      "failures": [],
      "all": {
        "yaw_drift_deg": 30.229024761726187,
        "ball_peak_forward_speed": 0.8717731446305345,
        "action_rate": 0.0961699320624272
      },
      "successful": {
        "yaw_drift_deg": 30.229024761726187,
        "ball_peak_forward_speed": 0.8717731446305345,
        "action_rate": 0.0961699320624272
      }
    },
    "prior_v2": {
      "successes": 58,
      "episodes": 60,
      "errors": 0,
      "failures": [
        {
          "seed": 3002,
          "entry": "reset",
          "error": null,
          "fell": true,
          "final_standing": false,
          "yaw_drift_deg": 60.77970187528246
        },
        {
          "seed": 3021,
          "entry": "reset",
          "error": null,
          "fell": true,
          "final_standing": false,
          "yaw_drift_deg": 62.31928745126492
        }
      ],
      "all": {
        "yaw_drift_deg": 31.27130561883356,
        "ball_peak_forward_speed": 0.8717742174171474,
        "action_rate": 0.19860459802051386
      },
      "successful": {
        "yaw_drift_deg": 30.227230134542523,
        "ball_peak_forward_speed": 0.8705756200948735,
        "action_rate": 0.09526982017118356
      }
    },
    "source_mujoco": {
      "successes": 45,
      "episodes": 60,
      "errors": 0,
      "failures": [
        {
          "seed": 3001,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 20.75531456929168
        },
        {
          "seed": 3004,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 23.705012676437974
        },
        {
          "seed": 3006,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 24.10160310019483
        },
        {
          "seed": 3008,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 29.43070252662635
        },
        {
          "seed": 3010,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 19.637241320760598
        },
        {
          "seed": 3013,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 21.096116942963572
        },
        {
          "seed": 3016,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 21.629330235808055
        },
        {
          "seed": 3017,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 19.26994237027226
        },
        {
          "seed": 3018,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 22.30364692133295
        },
        {
          "seed": 3020,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 9.244052151702524
        },
        {
          "seed": 3021,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 12.696003516700873
        },
        {
          "seed": 3022,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 14.485497812943938
        },
        {
          "seed": 3023,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 19.84671037637593
        },
        {
          "seed": 3024,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 21.33846384346928
        },
        {
          "seed": 3026,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 17.54370329475193
        }
      ],
      "all": {
        "yaw_drift_deg": 12.32526043070177,
        "ball_peak_forward_speed": 0.47187204969572183,
        "action_rate": 0.060460813778142136
      },
      "successful": {
        "yaw_drift_deg": 9.831828537388299,
        "ball_peak_forward_speed": 0.6291627329276281,
        "action_rate": 0.05532639647523562
      }
    },
    "previous_godot": {
      "successes": 40,
      "episodes": 60,
      "errors": 0,
      "failures": [
        {
          "seed": 3001,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 2.703086900101325
        },
        {
          "seed": 3003,
          "entry": "standing",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 5.084202578997195
        },
        {
          "seed": 3004,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 3.8075390420646484
        },
        {
          "seed": 3006,
          "entry": "reset",
          "error": null,
          "fell": true,
          "final_standing": false,
          "yaw_drift_deg": 36.41266599584503
        },
        {
          "seed": 3007,
          "entry": "standing",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 5.862580835818241
        },
        {
          "seed": 3011,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 2.2961791350024248
        },
        {
          "seed": 3013,
          "entry": "reset",
          "error": null,
          "fell": true,
          "final_standing": false,
          "yaw_drift_deg": 151.9848586222024
        },
        {
          "seed": 3014,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 8.43296671136678
        },
        {
          "seed": 3015,
          "entry": "reset",
          "error": null,
          "fell": true,
          "final_standing": false,
          "yaw_drift_deg": 116.11994935846069
        },
        {
          "seed": 3015,
          "entry": "standing",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 1.0219225306986568
        },
        {
          "seed": 3017,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 1.844927953703084
        },
        {
          "seed": 3017,
          "entry": "standing",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 5.023592800542412
        },
        {
          "seed": 3019,
          "entry": "reset",
          "error": null,
          "fell": true,
          "final_standing": false,
          "yaw_drift_deg": 410.66337189978753
        },
        {
          "seed": 3021,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 16.915339693702368
        },
        {
          "seed": 3022,
          "entry": "reset",
          "error": null,
          "fell": true,
          "final_standing": false,
          "yaw_drift_deg": 43.255382334048385
        },
        {
          "seed": 3023,
          "entry": "reset",
          "error": null,
          "fell": true,
          "final_standing": false,
          "yaw_drift_deg": 104.518138342821
        },
        {
          "seed": 3024,
          "entry": "standing",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 0.2358556211636598
        },
        {
          "seed": 3026,
          "entry": "reset",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 2.78203886401246
        },
        {
          "seed": 3028,
          "entry": "standing",
          "error": null,
          "fell": false,
          "final_standing": true,
          "yaw_drift_deg": 7.33807037688006
        },
        {
          "seed": 3029,
          "entry": "reset",
          "error": null,
          "fell": true,
          "final_standing": false,
          "yaw_drift_deg": 176.46088318466786
        }
      ],
      "all": {
        "yaw_drift_deg": 20.25663579675687,
        "ball_peak_forward_speed": 0.6863704371114333,
        "action_rate": 0.6943274325380723
      },
      "successful": {
        "yaw_drift_deg": 2.815864875588148,
        "ball_peak_forward_speed": 0.7935994652210686,
        "action_rate": 0.09893949180841446
      }
    }
  },
  "continuous": {
    "continuous_candidate": {
      "cases": 30,
      "errors": 0,
      "successes": {
        "ground_pick": 30,
        "kick_left": 30,
        "kick_right": 30,
        "roulade": 28
      },
      "ordinary_falls": [],
      "failures": [
        {
          "seed": 3005,
          "skill": "roulade",
          "metrics": {
            "seconds": 4.999999999999893,
            "fell": true,
            "final_standing": true,
            "final_z": 0.11590529058856122,
            "final_tilt": 0.7160107141418502,
            "max_tilt": 177.4850387633074,
            "vel_err_1s": 0.6102731487638862,
            "yaw_err_1s": 0.09621803378976918,
            "yaw_drift_deg": 329.03465379257307,
            "displacement": 0.6254124223090819,
            "mean_vx": -0.01616421641793735,
            "mean_wz": -0.05002763211254204,
            "action_rate": 0.23446710407733917,
            "head_contact_fraction": 0.28,
            "head_top_pivot": true,
            "inverted_trunk": true,
            "ordered_roll_events": true,
            "supported_fraction": 0.988,
            "supported_forward_rotation": 7.542554544806479,
            "net_rotation": 6.2436772338196125,
            "rotation_frontier": 7.670729665160178,
            "single_revolution": true,
            "final_heading_error_deg": 31.118003710204402,
            "success": false,
            "score": 0.9136628410906596
          }
        },
        {
          "seed": 3009,
          "skill": "roulade",
          "metrics": {
            "seconds": 4.999999999999893,
            "fell": true,
            "final_standing": true,
            "final_z": 0.11591265773175695,
            "final_tilt": 0.59322753022893,
            "max_tilt": 176.79441868386476,
            "vel_err_1s": 0.6120037555415968,
            "yaw_err_1s": 0.09329556497633502,
            "yaw_drift_deg": 329.4948657144122,
            "displacement": 0.6352173773347837,
            "mean_vx": -0.01704556428642551,
            "mean_wz": -0.04854299930892014,
            "action_rate": 0.27019327878952026,
            "head_contact_fraction": 0.284,
            "head_top_pivot": true,
            "inverted_trunk": true,
            "ordered_roll_events": true,
            "supported_fraction": 0.988,
            "supported_forward_rotation": 7.475481435954572,
            "net_rotation": 6.1967949089474885,
            "rotation_frontier": 7.673408195674421,
            "single_revolution": true,
            "final_heading_error_deg": 30.655895537691514,
            "success": false,
            "score": 0.9161470098706889
          }
        }
      ]
    },
    "continuous_prior_v2": {
      "cases": 30,
      "errors": 0,
      "successes": {
        "ground_pick": 30,
        "kick_left": 30,
        "kick_right": 30,
        "roulade": 28
      },
      "ordinary_falls": [],
      "failures": [
        {
          "seed": 3005,
          "skill": "roulade",
          "metrics": {
            "seconds": 4.999999999999893,
            "fell": true,
            "final_standing": true,
            "final_z": 0.11590529058856122,
            "final_tilt": 0.7160107141418502,
            "max_tilt": 177.4850387633074,
            "vel_err_1s": 0.6102731487638862,
            "yaw_err_1s": 0.09621803378976918,
            "yaw_drift_deg": 329.03465379257307,
            "displacement": 0.6254124223090819,
            "mean_vx": -0.01616421641793735,
            "mean_wz": -0.05002763211254204,
            "action_rate": 0.23446710407733917,
            "head_contact_fraction": 0.28,
            "head_top_pivot": true,
            "inverted_trunk": true,
            "ordered_roll_events": true,
            "supported_fraction": 0.988,
            "supported_forward_rotation": 7.542554544806479,
            "net_rotation": 6.2436772338196125,
            "rotation_frontier": 7.670729665160178,
            "single_revolution": true,
            "final_heading_error_deg": 31.118003710204402,
            "success": false,
            "score": 0.9136628410906596
          }
        },
        {
          "seed": 3009,
          "skill": "roulade",
          "metrics": {
            "seconds": 4.999999999999893,
            "fell": true,
            "final_standing": true,
            "final_z": 0.11591265773175695,
            "final_tilt": 0.59322753022893,
            "max_tilt": 176.79441868386476,
            "vel_err_1s": 0.6120037555415968,
            "yaw_err_1s": 0.09329556497633502,
            "yaw_drift_deg": 329.4948657144122,
            "displacement": 0.6352173773347837,
            "mean_vx": -0.01704556428642551,
            "mean_wz": -0.04854299930892014,
            "action_rate": 0.27019327878952026,
            "head_contact_fraction": 0.284,
            "head_top_pivot": true,
            "inverted_trunk": true,
            "ordered_roll_events": true,
            "supported_fraction": 0.988,
            "supported_forward_rotation": 7.475481435954572,
            "net_rotation": 6.1967949089474885,
            "rotation_frontier": 7.673408195674421,
            "single_revolution": true,
            "final_heading_error_deg": 30.655895537691514,
            "success": false,
            "score": 0.9161470098706889
          }
        }
      ]
    }
  },
  "accepted": true,
  "selection": "delivery_v3",
  "unchanged_evidence": "Other eight models byte-identical to V2; standalone and keyboard evidence inherited from V2 seeds 2000\u20132029. No claim that those tests were rerun.",
  "paired_successful_prior_v2": {
    "n": 58,
    "candidate": {
      "yaw_drift_deg": 30.43557258386273,
      "ball_peak_forward_speed": 0.8705745103156188,
      "action_rate": 0.09525739542883017
    },
    "reference": {
      "yaw_drift_deg": 30.227230134542523,
      "ball_peak_forward_speed": 0.8705756200948735,
      "action_rate": 0.09526982017118356
    }
  },
  "paired_successful_source_mujoco": {
    "n": 45,
    "candidate": {
      "yaw_drift_deg": 30.874402132057924,
      "ball_peak_forward_speed": 0.8586624879092459,
      "action_rate": 0.09056854794422785
    },
    "reference": {
      "yaw_drift_deg": 9.831828537388299,
      "ball_peak_forward_speed": 0.6291627329276281,
      "action_rate": 0.05532639647523562
    }
  }
}


Final integrity: all 272 phase-one sealed files, 24 phase-two selected files, and
24 phase-three selected files match their seals. All 18 original ONNX files match
initial hashes. All nine selected graphs/manifests match the bundle; full tests
pass 187/187. No owned training/evaluation job remains. The final two previously
failing cold right kicks fall at 3.64/3.60 seconds under V2; KR06 maximum tilt
is 6.67/6.61 degrees on the same already-counted seeds. This is a posthoc recovery
diagnostic, not an additional test.
