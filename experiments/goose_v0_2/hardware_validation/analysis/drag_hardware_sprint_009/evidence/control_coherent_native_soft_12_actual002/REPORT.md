# Candidate12 actual002 — bounded diagnostic FAIL

One actual physical run: 6732 fine steps / 6.732 s, 1000 Hz physics, 200 Hz original drive PD, 50 Hz Teacher. Original launch attempt001 had zero physical steps; dependency/numbering/read-only-log repair was independently audited. No second physical run or tuning followed.

Genuine cold-open capture proceeded in one free world into a complete 5000-fine-step loaded hold, then motion began at 5.800 s. No state/foot-material history reset occurred. Hold minima: bilateral jaw 26.566 N, foot 57.987 N; maximum torso tilt 1.9934°. The inherited prefix R/L material path was 3.518/3.340 mm.

Motion failed: first source clipping left ankle roll at 6.265 s, right hip yaw at 6.285 s, left hip roll at 6.290 s, right knee at 6.355 s. First irreversible 5 mm qualification failure was 6.433 s. The original 20 mm gross diagnostic guard stopped at 6.732 s. R/L reported cumulative material paths were 6.290/20.270 mm. No thresholds, caps or friction were relaxed.

Left N first reached zero at 6.644 s, but whole-sole clearance was only 0.0096 mm. Maximum left whole-sole minimum height during motion was 0.1052 mm; qualified air streak was zero. Thus actual air/backlanding steps=0, support exchange=0, continuous walking=FAIL. It is not a successful step merely because N was zero or a phase was requested.

|Endpoint|Goose backward|Cart backward|Cargo backward|
|---|---:|---:|---:|
|Last logged PRE,6.731 s|25.191 mm|13.140 mm|13.156 mm|
|True POST,6.732 s|25.250 mm|13.258 mm|13.275 mm|

The task's 0.5 m / >=0.1 m/s / command-to-stop<=5 s / 3 s stop-hold was not achieved. Mouth remained engaged (last PRE 27.66/23.11 N), and no robot-cart nonrod contacts or runtime robot-robot active contacts occurred. The observed failure is loaded support/traction tracking with source clipping and sliding; this one run does not establish a global hardware capacity limit or a mouth hardware defect.

Maximum contact penetration 0.7984 mm at 6.575 s and internal equality residual 0.2435 mm remain numerical limitations. Physical08 native mechanical/manufacturing and finite coverage exceptions remain; neither source legality nor proxy execution grants mechanical qualification.

The video is original-speed continuous neutral physics-proxy replay, showing goose, mouth, both feet, whole cart, visible 20 kg cargo and ground. It is not CAD visual or generated imagery. It changes display only and performs no integration.

## Evidence semantics

`summary.json` is preserved unchanged. Its final_* fields mean the last logged PRE, not true terminal. `true_terminal_poststate.json/.npz` is the true POST. `terminal_abort_prestate.json` records the unexecuted next-fine-step guard and its predictive/material accounting. Contacts store original native indices, geom/body identities, world frames/forces and matched PRE state. `actual_event_analysis.json` clipping counts are fine-sample requested-minus-applied flags, not a claim that each axis used its capacity for only that fraction.

`selected_actual_native_states.json` supplies PRE5.8/capture, PRE6.433/first5mm, PRE6.644/zero-N-not-air, PRE6.731/prestop; the true POST is separate. There are no hidden new forward/IK/LP/integration operations in this post-run event extraction.

All 80 physical08 source files are unchanged. Current frozen source/dependencies are included in the dedicated Lab control_source subtree; old candidate12 and failure attempt001 are untouched. NO JOBS.
