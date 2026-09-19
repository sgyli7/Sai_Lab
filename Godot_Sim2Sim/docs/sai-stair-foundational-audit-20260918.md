# Sai stair-control foundational audit (2026-09-18)

## Decision

The `sai-safe-residual-stairs-v5` run was stopped after 70 PPO iterations. Its
checkpoints are diagnostic artifacts only. The run made joint motion safer, but
its training and deployment control loops were different and its action was an
absolute joint target despite the residual name.

PPO, the position objective, asymmetric critic, native ONNX path, and the v5
safety envelope remain useful. The control hierarchy must change before more
large-scale training.

## Evidence in the previous design

- Training converted 16 policy values directly into absolute joint and wheel
  targets and used fixed 80/2 position gains.
- Deployment switched from the flat actor to the stair actor, then applied
  terrain suspension and stance impedance that were absent during training.
- The policy therefore never trained on the real transition state, action
  history, target history, or downstream low-level dynamics used at evaluation.
- A scripted edge teacher retained a 0.70 reward weight for risers at or below
  30 mm. That explicitly reinforced lifting on small obstacles.
- Actor initialization opened every riser and all four stair edges immediately;
  curriculum difficulty did not advance from per-class success.
- The 242-value observation used exact simulator ray heights. It is a Godot and
  MuJoCo observation, not a real-robot sensor contract.
- The source robot model documents missing link-to-link self collision and
  uncalibrated actuator, friction, armature, inertia, and contact parameters.

The independent 40-iteration checkpoint probe confirmed the distinction:
joint limits and target slew were safe, but the robot repeatedly lifted near the
first 40 mm edge, tilted beyond the acceptance envelope, and stalled.

## v6 control hierarchy

The rolling controller, active suspension, and stance impedance stay active at
all times. A learned policy produces only a bounded task-space skill residual:

1. four wheel-end fore/aft residuals;
2. four wheel-end vertical residuals;
3. four wheel-speed residuals;
4. chassis height, roll, and pitch residuals;
5. one continuous learned skill intensity.

HAA is no longer an unrestricted learned absolute target. Hip and knee targets
come from deterministic wheel-end inverse kinematics. The v5 joint margin,
target slew, torque, and collision projection remain the final safety layer.

Training and deployment must use the same order:

```text
flat rolling prior
-> suspension/contact state
-> learned task-space residual
-> inverse kinematics and safety projection
-> stance impedance/feedforward
-> actuator model
-> physics
```

The final implementation may still contain one native ONNX. A shared encoder,
roll/step/jump heads, and a continuous selector can be distilled behind this
single action contract.

## Training order

1. Qualify rough-terrain suspension with learned residual fixed at zero.
2. Train the low-step skill for one lift per wheel and edge.
3. Train a separate high-step skill for coordinated compression, flight, and
   landing.
4. Train transitions and the continuous selector on mixed courses.
5. Jointly fine-tune only after every skill retains its own acceptance rate.

Difficulty advances from success and safety rates for each terrain class.
Every batch retains rough and lower-step replay to prevent forgetting.

## Required acceptance

- Rough terrain: near-zero skill intensity, no airborne event, at least three
  valid contacts, and bounded chassis/payload translation and rotation jerk.
- Low stairs: at most one lift per wheel per edge and no premature lift.
- High stairs: one coherent flight is allowed, with bounded flight duration,
  landing impulse, jerk, joint load, torque, and power.
- Every case: zero fall, zero structural collision, adequate self-clearance,
  and robustness across yaw, tread, friction, load, disturbance, and seeds.
- The same candidate must pass MuJoCo CPU, MuJoCo-Warp, and Godot/Jolt, followed
  by continuous video review.

## Scope boundary

The next candidate is explicitly `godot-oracle-terrain` and simulation-only.
Sim2Real acceptance is blocked until a real terrain estimator exists and the
minimum actuator, wheel/ground, collision, mass, and latency ranges are measured.

## Primary references

- Simon Chamorro et al., *Reinforcement Learning for Blind Stair Climbing with
  Legged and Wheeled-Legged Robots*, ICRA 2024. The method uses a position task,
  asymmetric actor-critic training, and an explicit stair-mode input.
- Haojie Shi et al., *Terrain-Aware Quadrupedal Locomotion via Reinforcement
  Learning*, 2023. The learned policy adjusts parameters of a trajectory
  generator rather than directly owning every joint target.
- Yuxiang Yang et al., *Agile Continuous Jumping in Discontinuous Terrains*,
  ICRA 2025. The system separates a learned centroidal motion policy from a
  model-based low-level controller and models hardware dynamics for transfer.
