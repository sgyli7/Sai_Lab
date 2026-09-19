# MicroDuck MuJoCo <-> Tuanjie MVP

## Outcome

Produce a reproducible Windows x64 project that:

- installs and verifies the pinned MicroDuck and `microduck_rl` sources;
- runs every vendored 61-observation/14-action ONNX policy in MuJoCo;
- completes a 64-environment, 5-iteration PPO smoke train and the official ONNX export path;
- imports both legged and roller MicroDuck models into Tuanjie Engine 1.10.2;
- evaluates the same policies through Barracuda 3.0.1 behind `IPolicyRuntime`;
- supports keyboard commands and policy switching for locomotion and all shipped skills;
- exports and replays versioned rollout traces across both simulators;
- builds a Windows x64 player and can rerun the complete verification from one PowerShell command.

This is an internal, non-commercial MVP. MicroDuck code is Apache-2.0; its 3D assets remain
subject to the upstream CC BY-SA-NC terms.

## Locked inputs

The authoritative repositories, commits, and nine policy filenames live in
`upstream.lock.json`. Tests must reject floating branches and policy-set drift.

## Contracts

### Policy contract

- input: one `float32[1,61]` tensor;
- output: one finite `float32[1,14]` tensor;
- shared joint order: five left-leg servos, four neck/head servos, five right-leg servos;
- policy tick: 50 Hz; physics tick: 200 Hz;
- action target: `home_pose + action * action_scale`;
- inference backend in Tuanjie: Barracuda 3.0.1, hidden behind `IPolicyRuntime`;
- a converted Barracuda-compatible copy is permitted only when the original is retained and
  ONNX Runtime parity is at most `1e-5` on deterministic fixtures.

### Robot manifest contract

The compiled MuJoCo model is the physics source of truth. The generated manifest carries source
hashes, body hierarchy, mass/centre-of-mass/inertia, joints, limits, axes, servo indices, passive
wheels, collision shapes, visual meshes, sensors, and named start poses. Original STL files remain
the visual source.

MuJoCo vectors map to Tuanjie as `(-y, z, x)`. Rotations, joint axes, centre-of-mass values, and
inertia frames must use the corresponding basis transform rather than independent sign tweaks.
Joint axes and angular velocities are axial vectors, so the handedness-changing map includes the
determinant sign: `(x, y, z) -> (y, -z, -x)`. In particular, MuJoCo `+Z` hinge rotation becomes
Tuanjie `-Y`; treating the axis as an ordinary position vector silently reverses joint angles.

Every locked ONNX file declares an action scale of `1.0`, which is the Tuanjie runtime default.
Headless behavior scenarios may carry simulator-specific calibration separately: walking needs
`1.1` to cross the plain-XML contact-model gait threshold (versus `1.0` in the official inference
helper and `0.9` in `robotd`'s real-robot preset), while the roller scenario follows `robotd`'s
`0.8` preset. Scenario calibration must never be mistaken for model metadata.

### Rollout trace contract

JSONL output begins with versioned metadata and then deterministic frames containing time, active
policy, commands, root pose, servo positions/velocities, passive-wheel state, observations, raw
actions, applied targets, and contacts. Tuanjie imports the trace for ghost comparison; its own
recording can be audited with the same Python tooling.

## Vertical slices and gates

1. **Bootstrap**: a clean checkout resolves pinned upstreams and creates the Python environment.
2. **Policy audit**: all nine original ONNX files pass shape, dtype, finite-output, and repeatability
   tests in ONNX Runtime.
3. **MuJoCo behavior**: a headless scenario exists for every policy and produces finite traces plus
   role-specific behavior metrics.
4. **Training/export**: the official `microduck_rl` command runs 64 environments for five PPO
   iterations and its official exporter produces a passing ONNX file.
5. **Neutral import**: MJCF compilation produces manifests and Tuanjie prefabs for legged and roller
   variants with tested body/joint/mass/axis/limit parity.
6. **Tuanjie inference**: EditMode tests compare Barracuda with ONNX Runtime fixtures and PlayMode
   tests close the 61D -> policy -> 14D -> articulation loop for all nine policies.
7. **Interaction**: a keyboard map selects each skill and drives velocity/body/head/phase commands;
   an in-game overlay reports policy, commands, rates, and fault state.
8. **Round trip**: MuJoCo reference traces replay in Tuanjie and Tuanjie recordings pass the Python
   schema/parity audit.
9. **Delivery**: Codely makes one test-first useful overlay change; batch tests pass; Windows x64
   player builds; the one-command script succeeds after generated caches are removed.

## One-command entry

`scripts/run-mvp.ps1` is the final orchestration boundary. It must fail fast, preserve component
logs and JUnit/XML results under `artifacts/`, and offer focused stages for diagnostics without
weakening the default all-gates run.
