# Sai task-space stair candidate — 2026-09-18

## Status

`task-space-v6-candidate-20260918` is an experimental, simulation-only
candidate. It is not the default profile.

The controller keeps rolling, terrain suspension, stance impedance and heading
hold continuously active. The ONNX actor contributes a bounded task-space
residual only when a discrete edge is present in the wheel path. Final joint
targets are projected after suspension, so later controller layers cannot add a
valid skill back across the equipment margin.

## Accepted evidence

- MuJoCo-Warp 40 mm × 4: 64/64 safe completions, no timeout, joint violation or
  tilt violation; worst upright dot 0.9881 and mean completion 9.70 s.
- CPU MuJoCo 40 mm × 4 at 500 Hz: completed in 14.60 s, payload retained,
  minimum upright 0.9886.
- CPU MuJoCo 500 Hz terrain suite: rough, washboard, potholes, bumps, cross-slope
  and long slope all completed with payload retained. Minimum upright stayed
  above 0.993; cargo acceleration RMS was 0.47–0.75 m/s² except washboard at
  1.90 m/s².
- Native Godot/Jolt 20 mm × 4 at 1000 Hz: completed in 12.74 s, every wheel
  crossed each edge once, no redundant edge event, payload retained and full
  motion-safety gate passed.
- Native Godot/Jolt 40 mm × 4 at 1000 Hz: completed in 16.46 s, payload retained,
  cargo acceleration RMS 1.59 m/s², peak roll 2.65 degrees, peak pitch 8.34
  degrees, and full motion-safety gate passed.
- Python/Godot fixed-state contract: maximum target error 3.93e-7.

The recorded 40 mm run contains 585 native viewport frames and its complete
50 Hz state/action trace. The 4D analysis reports exactly four physical airborne
onsets per wheel, no interval above 10 degrees, and no interval with fewer than
two supports after initial spawn.

## Rejected evidence and limits

- Native Godot/Jolt at 500 Hz did not clear the 40 mm course in 45 s and did not
  retain the payload. The runtime default remains 1000 Hz.
- Native 60 mm × 4 did not complete. A focused 60 mm training branch produced
  no complete success through iteration 20 and had one MuJoCo-Warp numerical
  failure, so it was stopped and rejected.
- The terrain observation still uses simulator ray heights. This candidate is
  not a Sim2Real release until a sensor-backed estimator and actuator/contact
  calibration replace the oracle inputs.

## Evidence paths

- Warp acceptance: `results/sai-stair-v6-continuous-edge4-finalchain-20260918/eval-cp25-guard7-h700-64/report.json`
- CPU terrain suite: `results/sai-stair-v6-continuous-edge4-finalchain-20260918/cpu-mujoco-v6-terrain-suite-500/summary.json`
- Native 20/60 mm comparison: `results/sai-stair-v6-continuous-edge4-finalchain-20260918/godot-native-up20-up60-1000/summary.json`
- Native 40 mm acceptance: `results/sai-stair-v6-continuous-edge4-finalchain-20260918/godot-native-up40-1000-gated/summary.json`
- Video and 4D trace: `results/sai-stair-v6-continuous-edge4-finalchain-20260918/godot-native-up40-1000-video/`
