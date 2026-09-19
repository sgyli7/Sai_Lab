# Sai high-step screen — 2026-09-18

## Result

The first v6 high-step candidate is **rejected** and must not be deployed.  The
60-iteration screen was stopped at iteration 30 because it had zero successful
episodes.  Continuing the same optimization would not have tested a new
hypothesis.

The stopped run had zero falls, zero cargo losses and zero numerical failures.
It was safe but did not learn the task.  On the independent CPU MuJoCo 40 mm
course, both the warm start and iteration-30 actor stalled at the first edge:

| Candidate | Forward distance in 35 s | Cargo accel RMS | Physical airborne onsets | Commanded lift onsets |
| --- | ---: | ---: | ---: | ---: |
| warm start | 0.340 m | 0.653 m/s² | 250 | 0 after corrected 3 mm command threshold |
| iteration 30 | 0.273 m | 0.775 m/s² | 246 | 0 after corrected 3 mm command threshold |
| exact zero residual | 0.270 m | 0.639 m/s² | 215 | 0 |

The original report called every airborne onset a lift.  The exact-zero actor
proved that interpretation wrong: the right-front wheel repeatedly lost
contact while the learned skill intensity was exactly zero.  These were
wheel/edge contact rebounds from a stalled rolling controller, not repeated
learned lift commands.  The evaluator now reports commanded lift events and
uncommanded airborne/contact events separately.

Continuous physical-state replays are preserved at:

- `results/sai-stair-v6-high-screen-20260918/eval-warm-up40/up40-1601-candidate/continuous.mp4`
- `results/sai-stair-v6-high-screen-20260918/eval-cp30-up40/up40-1601-candidate/continuous.mp4`

## Foundational defects found by the screen

1. The high curriculum said “one height at a time”, but initialized with the
   first four eligible lanes: 40, 45, 50 and 55 mm.  It now starts at 40 mm and
   advances only after the measured success threshold.
2. The high candidate inherited a low-step actor and a hand-authored per-wheel
   teacher.  This supplied no successful high-step trajectory.
3. The task-space layer rate-limited the complete joint target instead of the
   learned residual.  A zero action therefore filtered the rolling prior.  It
   now rate-limits only the residual.
4. Enabling v6 replaced the accepted stance impedance with a simplified load
   allocator.  v6 now keeps the same accepted suspension and stance impedance
   online, as a residual architecture requires.
5. Python and native Godot were rechecked after these changes.  The fixed-state
   v6 contract passed with maximum target error below `4e-7`.

The exact-zero actor still stalls at 40 mm.  This is the correct baseline: the
ordinary rolling prior and suspension handle rough terrain, but do not contain
a high-step skill.  A new high-step learner needs successful exploration or a
privileged successful expert; the rejected per-wheel teacher will not be used
again.

## Research basis for the next design

The closest cross-platform stair-climbing work uses a position goal,
asymmetric actor-critic and an explicit stair-mode observation rather than a
velocity tracker alone.  Real wheeled-legged deployments also separate rough
terrain, where wheels stay in contact as active suspension, from stairs, where
the controller changes locomotion mode.  Parkour work addresses sparse skill
discovery by training privileged skill experts before distilling them into a
single deployable policy.

These findings support the current separation into rolling/suspension,
low-step and high-step skills.  They do not support reviving the old periodic
leg-lift target as the deployed controller.

Primary sources:

- https://arxiv.org/abs/2402.06143
- https://arxiv.org/abs/2405.01792
- https://proceedings.mlr.press/v229/zhuang23a.html
- https://proceedings.mlr.press/v270/chane-sane25a.html
