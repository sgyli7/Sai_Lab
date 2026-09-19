# Sai stair locomotion: training design

The production target is one terrain-conditioned actor that rolls compliantly on irregular ground, steps over low stairs, and discovers a coordinated dynamic maneuver for high stairs. A clocked gait or a controller-side table keyed by stair height is outside the target contract.

## Evidence behind the design

- Chamorro et al., [Reinforcement Learning for Blind Stair Climbing with Legged and Wheeled-Legged Robots](https://arxiv.org/abs/2402.06143), report that a position-based task is critical for stair climbing. Their velocity-task ablation suppresses the nonlinear approach, deceleration, and step-up behavior needed at an edge. They also use an asymmetric actor-critic, a terrain mode observation, difficulty curriculum, friction and delay randomization, and test on wheeled-legged robots.
- Rudin et al., [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://proceedings.mlr.press/v164/rudin22a.html), establish large parallel simulation and terrain curriculum as a practical route to robust locomotion.
- Agarwal et al., [Legged Locomotion in Challenging Terrains using Egocentric Vision](https://proceedings.mlr.press/v205/agarwal23a.html), train a privileged terrain teacher and then distill to the deployable perceptive policy. The memory requirement for terrain passing under the body is directly relevant to rear-wheel placement.
- Wang et al., [Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://arxiv.org/abs/2405.10830), show that teacher/student training under RL improves uneven-terrain tracking over a separate supervised transfer stage.
- Zhang et al., [Motion Priors Reimagined](https://proceedings.mlr.press/v305/zhang25j.html), preserve a smooth flat-terrain motion prior while a learned residual adapts it to complex terrain. This supports regularizing against Sai's smooth low-step actor instead of scripting lift phases.

## Repository diagnosis

The first stair PPO used the same 242-value observation for actor and critic, rewarded tracking a fixed 0.16 m/s speed, and simulated a free 100 g payload while deployment closed the cargo clamp. It covered discrete box stairs but no rough terrain bank. Those choices encourage repeated lifting to recover commanded speed and optimize dynamics that differ from Godot/Jolt.

## Implemented prototype

`scripts/train_sai_stair_ppo.py` now supports:

1. a goal-progress objective, leaving approach speed and the step-up velocity profile to the policy;
2. a privileged critic with exact rise, tread, goal distance, wheel clearance, and payload pose, while the exported actor remains on the 242-value deployable terrain/proprioception contract;
3. learned-expert compression to initialize one continuous terrain-conditioned actor, followed by PPO rather than a controller-side expert threshold;
4. a closed cargo clamp initialized at the same 67 mm target used by Godot/Jolt;
5. explicit low/high teacher weights, action-rate, lift-event, cargo-acceleration, and motion-prior regularization controls.

The prototype must beat the event-gated diagnostic baseline in CPU MuJoCo before native Godot/Jolt deployment. Passing requires physical course completion with the 100 g payload supported, acceptable upright margin and cargo acceleration, plus inspection of repeated wheel lift events. Export success alone is not acceptance.

## Auto-research outcome

The experiments reject a controller-side threshold as the final architecture, but retain it as an opt-in diagnostic baseline. With a physically clamped 100 g payload in Godot/Jolt, that baseline clears 20/40/60 mm with cargo RMS 1.00/1.57/3.96 m/s² and 4/9/13 repeated lift events. It completes every course, but medium/high lift quality misses acceptance.

A single MLP distilled from the low/high experts failed all three CPU MuJoCo courses. A learned continuous gate over the two frozen priors completed all three courses, establishing feasibility without deployment-time height thresholds. Comfort-constrained training improved 20/40/60 mm cargo RMS from 2.92/3.97/6.90 to 2.58/3.03/6.61 m/s², but retained too many lift events. Correcting the clamped-payload orientation and retraining reduced the 60 mm result to 4.85 m/s² and 20 repeated events, while Godot/Jolt still failed 40 mm.

Delay and actuator-strength randomization improved the deterministic 60 mm CPU result to 3.32 m/s² and 12 repeated events, but degraded 20/40 mm to 35.0/31.1 s with 52/26 repeated events. This candidate is rejected. The evidence supports jointly training the motion priors and gate under the position-task, asymmetric-critic, clamped-payload and randomized dynamics objective. Freezing old experts leaves the gate unable to create a smooth medium-height skill.
