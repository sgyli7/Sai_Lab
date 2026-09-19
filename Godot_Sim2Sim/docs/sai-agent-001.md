# Sai_Agent_001 profile

```sh
uv sync --extra sai
uv run --extra sai sim2sim-play --robot Sai_Agent_001
```

The optional profile pins the Sai source/model/policies to commit
`5d2ab070dcdbae61a8d750a3f812e77ad2568f26`. It supplies its own 26-body,
23-hinge/two-slider Jolt scene, 82D observation and 16D mixed position/velocity
controller. MicroDuck remains the default and retains its existing scene/policy
contracts. The launcher starts the policy service automatically; no separate
training server is needed.

W/S forward/reverse, A/D yaw, hold Shift to crouch and release to stand, R restart, Esc quit.
Godot 4.7.2 must be installed and on PATH (`--godot-bin` selects another location).

```sh
uv run --extra sai sim2sim-play --robot Sai_Agent_001 --stairs .02
uv run --extra sai sim2sim-play --robot Sai_Agent_001 --headless --case W --output results/sai-W.json
```

The pinned package passed eight flat keyboard cases and four-riser 20/40 mm
ascent/descent in Godot/Jolt. It uses simulated ground raycasts; camera/VLA terrain
reconstruction is not implemented. The default remains the 20/40 mm actor. Other MicroDuck
`--scene` presets do not automatically convert into the Sai profile; this first
adapter launches the package's own scene. The SO101 and cargo joints remain real
articulations, not fixed display parts.

## Physical pickup and loaded transport

```sh
uv run --extra sai sim2sim-play --robot Sai_Agent_001 --task cargo
```

The new pinned alpha includes the separate 100 g box pickup, opposing-pad clamp
and loaded crawl task. The public package passed 18/25 mm obstacles in MuJoCo
and Godot with no lost/unclamped physics steps. `--no-clamp` intentionally stops
before transport and exits 3; no object is welded or moved by pose assignment.
The pickup uses scripted model-state IK, not VLA; loaded crawl preserves its
frozen 57D actor and does not replace the 82D keyboard controller.

The launcher recreates both physics and controller state on R. Visible playback
also disables VSync and caps rendering to 60 FPS to avoid the observed 1 FPS
throttling on the development Linux display. Physics rates and contacts remain
unchanged. This is not a universal hardware-performance guarantee.


## Bundled 60 mm experiments (alpha.3)

```sh
uv run --extra sai sim2sim-play --robot Sai_Agent_001 --stairs .06 --stair-skill ascent60
uv run --extra sai sim2sim-play --robot Sai_Agent_001 --stairs .06 --descending --stair-skill descent60
```

These opt-in profiles are paired with their frozen control settings. They pass
nominal four-riser flights under a declared 45-second budget. The descent actor
also passes fifteen development parameter cases in both full MuJoCo and Godot;
these are not untouched-terrain or payload guarantees. It uses half crouch and
known-route steering at y=0, with A/D overriding steering while held.

The pinned package was fetched from GitHub and the existing `sim2sim-play`
entry point completed the 60 mm descent with real W input in 16.48 seconds.
All 38 original input tests still pass. The fresh core wheel separately passed
both 60 mm directions. See [integration evidence](sai-a3-profile-evidence.json).
