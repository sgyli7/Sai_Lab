# Sai physics-frequency audit — 2026-09-18

## Decision

The workshop default is reduced from **2000 Hz to 1000 Hz**. The native ONNX
controller remains **exactly 50 Hz**. A 500 Hz target remains desirable because
it matches the MuJoCo training timestep, but it is not yet safe as the product
default: Sai's hand-written cargo-belt constraint loses bilateral clamp contact
and introduces large deck acceleration at 500 Hz. Rates of 250 Hz and below
also destabilize the articulated body with the current actuator and constraint
parameters.

This is a conservative intermediate result, not a claim that 1000 Hz is a
normal game tick rate. Rendering remains independent at 30/60 FPS. Reaching
500 Hz requires replacing or recalibrating the cargo constraint, followed by
the same loaded rough-terrain and stair qualification.

## Timebase defect removed

The old runtime coupled four separate quantities:

- global Jolt physics at 2000 Hz;
- policy inference every 40 physics ticks;
- robot time as `tick * 0.0005`;
- test metrics differentiated with a fixed 0.5 ms timestep.

Changing only the engine tick rate would therefore have changed policy cadence,
timestamps, action timing and reported acceleration at once. Generated Sai
runtimes now derive simulation time and policy decimation from the active engine
rate. Supported sweep points all divide exactly into the 50 Hz policy:

| Physics | Physics steps per policy decision |
| ---: | ---: |
| 2000 Hz | 40 |
| 1000 Hz | 20 |
| 500 Hz | 10 |
| 250 Hz | 5 |
| 200 Hz | 4 |
| 100 Hz | 2 |

60 Hz is intentionally rejected because it cannot represent a strict 50 Hz
policy with integer fixed steps.

A separate native Hub smoke run at the new default produced 249 consecutive
in-process ONNX records with an exact 0.020000 s interval. The Hub reported
1000 Hz physics and the robot clock remained aligned with scene time, confirming
that the change did not silently alter controller cadence.

## Godot/Jolt qualification

All figures below come from native Godot 4.7.2/Jolt runs with the same 50 Hz
controller, 100 g physical payload, clamp command, terrain seed and controller
parameters. Metric derivatives use the actual physics delta.

### Rough terrain, three seeds

| Metric (mean) | 2000 Hz | 1000 Hz |
| --- | ---: | ---: |
| Completed / payload retained / bilateral clamp | 3/3 | 3/3 |
| Minimum upright dot | 0.9954 | 0.9967 |
| Forward speed | 0.4396 m/s | 0.4466 m/s |
| Cargo acceleration RMS | 0.654 m/s² | 0.826 m/s² |
| Cargo slip | 0.43 mm | 0.88 mm |
| Wall time for 8 s simulation | 5.50 s | 3.66 s |

1000 Hz reduced wall time by about **33%** while preserving completion, clamp,
payload retention and posture. Cargo acceleration increased by about 26%, so
1000 Hz is accepted as the immediate default but the cargo-smoothness metric
remains a required regression gate.

### Loaded stairs, one deterministic seed

| Course | Rate | Cleared at | Min upright | Cargo accel RMS | Slip | Bilateral clamp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 mm × 4 | 2000 | 11.134 s | 0.9235 | 2.534 m/s² | 0.14 mm | 100% |
| 20 mm × 4 | 1000 | 11.232 s | 0.9271 | 2.584 m/s² | 0.19 mm | 100% |
| 20 mm × 4 | 500 | 10.986 s | 0.9519 | 3.354 m/s² | 58.01 mm | 0% |
| 40 mm × 4 | 2000 | 17.846 s | 0.9063 | 2.017 m/s² | 0.24 mm | 100% |
| 40 mm × 4 | 1000 | 17.970 s | 0.9217 | 1.974 m/s² | 0.20 mm | 100% |
| 40 mm × 4 | 500 | 18.280 s | 0.9173 | 2.434 m/s² | 58.44 mm | 0% |

1000 Hz reproduced stair timing within 0.13 s, retained the payload and slightly
improved minimum upright posture. The 500 Hz robot still climbed, but the cargo
clamp stopped making bilateral contact and the payload moved about 58 mm. That
failure is why 500 Hz is not promoted.

The 500 Hz failure was tested against a timestep-scaled belt stiffness and
damping derived from the old 2000 Hz values. It still lost clamp contact, which
rules out a one-line gain scaling as a reliable fix. The next reduction requires
a proper constraint/actuator redesign and qualification rather than another
frequency-specific constant.

## MuJoCo corroboration after the v6 control audit

The articulated CPU MuJoCo harness now accepts an explicit physics rate while
holding the policy at 50 Hz. This catches a real stability boundary rather than
silently changing both clocks together. Three loaded rough-terrain seeds gave:

| Metric (mean) | 1000 Hz | 500 Hz |
| --- | ---: | ---: |
| Completed / payload retained / bilateral clamp | 3/3 | 3/3 |
| Minimum upright dot | 0.9932 | 0.9932 |
| Cargo acceleration RMS | 0.692 m/s² | 0.705 m/s² |
| Deck acceleration RMS | 2.113 m/s² | 2.492 m/s² |
| Cargo slip | 0.17 mm | 0.17 mm |

One 40 mm × 4 stair comparison also completed at both rates. Clearance time was
16.48 s at 1000 Hz and 16.40 s at 500 Hz; the 500 Hz run retained bilateral
clamp contact and had slightly lower cargo acceleration RMS. This legacy stair
behavior still violates the new 15-degree posture gate, so it is evidence about
the physics timebase only, not acceptance of the stair policy.

The next lower point showed a sharp numerical/control failure. At 250 Hz on the
same rough seed, bilateral clamp contact fell to zero, actuator saturation rose
to 99.45%, deck acceleration RMS rose from 2.12 to 49.81 m/s², and cargo
acceleration RMS rose from 0.71 to 6.82 m/s². The 100 Hz run eventually tipped.

These results make **500 Hz the justified engineering target**. It is also the
rate used by the v6 MuJoCo-Warp training model. The workshop stays at 1000 Hz
until the Godot/Jolt cargo constraint is redesigned and the 500 Hz native suite
passes; 250 Hz and below are rejected with the current actuator and contact
model.

The final v6 candidate repeated that cross-simulator split. At 1000 Hz, native
Godot/Jolt cleared the 40 mm × 4 course in 16.46 s, retained the payload, stayed
within 8.2 degrees, and passed the complete motion-safety gate. At 500 Hz the
same native actor did not clear the course in 45 s and did not retain the
payload. Therefore 1000 Hz remains the shipped/default rate. The 500 Hz result
is a concrete redesign target, not a supported runtime setting yet.

## Implementation and evidence

- Runtime timebase: `src/sim2sim/sai_timebase.py`
- Workshop configuration: `src/sim2sim/workshop.py`, `godot/hub/hub.gd`
- Dynamic robot time and policy scheduling are injected only into generated
  runtime copies; the upstream Sai source bundle remains unchanged.
- Corrected sweep: `results/sai-frequency-audit/rough-clamped-dt-corrected/summary.json`
- Three-seed and turn qualification:
  `results/sai-frequency-audit/qualification-1000-vs-2000/summary.json`
- Stair qualification: `results/sai-frequency-audit/stairs/summary.json`
- MuJoCo rate sweep and full traces:
  `results/sai-frequency-audit/mujoco-v6-timebase-20260918/`

Godot documents 60 Hz as the engine default, states that CPU usage scales with
physics tick rate, and recommends fixed rates selected for the actual physics.
It also warns that lowering a rate can change collisions and control behavior.
Those constraints are why this change uses a fixed qualified 1000 Hz rate rather
than dynamically adapting physics to render FPS.
