# Native MuJoCo architecture

## Decision

The playable MicroDuck MVP uses the official `org.mujoco` Unity plug-in as the
physics authority inside Tuanjie. This keeps the trained policy connected to
the same MJCF dynamics, contacts, sensors, actuator order, and passive roller
wheel joints used by the upstream MuJoCo environment.

Tuanjie still owns the application shell: scene lifecycle, input, rendering,
test execution, and the Windows player. Barracuda 3.0.1 owns ONNX inference.
Sentis is intentionally absent. Codely is an editor automation harness and does
not ship as part of the policy loop.

The PhysX/ArticulationBody importer is retained as a calibration surface for
coordinate, inertia, collider, material, and drive experiments. PhysX results
must not be used as the acceptance oracle for official MicroDuck behavior.

## Runtime ownership

```text
MujocoKeyboardPolicyInput
          |
          v
PolicyCommandState -------------------------------+
                                                   |
official MuJoCo mjData (joints, gyro, root pose)   |
          |                                        |
          +--------------+-------------------------+
                         v
             61-value observation
                         |
                         v
          Barracuda 3.0.1 CSharpBurst worker
                         |
                         v
               14-value raw action
                         |
                         v
       home pose + role scale + target filter
                         |
                         v
             mjData.ctrl[0..13]
                         |
                         v
          official native MuJoCo simulation
```

`Time.fixedDeltaTime` and the MJCF timestep are `0.005 s`, so physics runs at
200 Hz. The controller evaluates a policy every four physics steps, or 50 Hz,
and reapplies the last 14 targets on intervening steps.

The observation contract is fixed at 61 floats:

- 3 local angular-velocity values;
- 3 projected-gravity values;
- 14 joint positions relative to the home pose;
- 14 joint velocities;
- 14 previous-action values;
- 13 command values.

Actuator transmission IDs from the compiled MuJoCo model determine the 14
controlled joint addresses. The roller model additionally keeps its four
passive wheel joints under MuJoCo. Same-model policy hot swaps preserve the
previous action and target-filter state instead of resetting the robot. A
legged/roller change rebuilds and rebinds the native `MjScene` to the selected
imported model.

## Deterministic MJCF import

`OfficialMujocoPrefabImporter` processes three locked upstream scenes:

| Source | Generated prefab |
| --- | --- |
| `scene.xml` | `Generated/MuJoCo/Legged/MicroDuck-MuJoCo-Legged.prefab` |
| `scene_rollers.xml` | `Generated/MuJoCo/Roller/MicroDuck-MuJoCo-Roller.prefab` |
| `scene_ball.xml` | `Generated/MuJoCo/Ball/MicroDuck-MuJoCo-Ball.prefab` |

For each source, native MuJoCo first compiles and saves an expanded MJCF file,
removing `include` indirection and resolving compiler defaults. The official
plug-in then creates the full MuJoCo component hierarchy and imports referenced
meshes/materials. The bridge copies those resources into fixed asset folders,
rebinds the prefab, and deletes temporary staging assets.

The importer test requires complete bodies, joints, geoms, all 14 actuators,
six sensors, and seven sites for each variant. A second import must preserve
the generated asset set and GUIDs and must not duplicate components. The ball
variant additionally verifies its free joint, radius, mass, inertia, and
friction values.

`DemoSceneBuilder.CreateAllSceneAssets` generates both scenes:

- `MicroDuckNativeMvp.unity`: authoritative, playable MuJoCo/Barracuda scene;
- `MicroDuckMvp.unity`: non-authoritative PhysX calibration scene.

Only the native scene is passed to the Windows x64 player build.

## ONNX boundary

The nine upstream policy files and their SHA-256 hashes are preserved under
`Generated/Policies/Original`. The project's tested Barracuda 3.0.1 import path
uses restricted compatibility copies under `Generated/Policies/Barracuda`.

The conversion changes only the default opset declaration to 9 after checking
that the graph uses the allowed operator set. Zero, ramp, and seeded fixtures
must produce matching outputs within `1e-5`; the generated compatibility report
records the original and converted hashes, fixture count, and maximum error for
every policy. Runtime contracts reject incorrect tensor widths, non-finite
values, and implausibly large actions before targets reach MuJoCo.

## Verification gates

`scripts/run-mvp.ps1` is the acceptance entry point. Its gates cover:

1. pinned upstream origins/commits and Python/MuJoCo/Torch availability;
2. Python tests with at least 80% coverage plus Ruff;
3. the exact nine-policy ONNX bundle and its `61D -> 14D` contract;
4. nine deterministic headless MuJoCo behavior rollouts and JSONL traces;
5. the exact 64-environment, 5-iteration PPO smoke run and `model_4.pt`;
6. official ONNX export metadata, shapes, and 100 finite randomized runs;
7. deterministic generated assets and both Tuanjie scenes;
8. Tuanjie EditMode and PlayMode suites, including native behavior contracts;
9. a real Tuanjie/Barracuda trace checked against ONNX Runtime at `1e-5`;
10. independently captured Codely RED/GREEN evidence;
11. the native-scene Windows x64 build and its SHA-256 report.

The final build gate deletes the previous bundle, builds again, launches the
new player headlessly, and waits for a scene-owned smoke probe. The probe must
observe MuJoCo native version `3012000` (`3.12.0`), the exact Barracuda backend,
at least one completed policy tick, and finite `61/14/14` observation/action/
target buffers before the player exits successfully. A manifest hashes every
file in the rebuilt distribution, including `mujoco.dll` and the managed
MuJoCo bridge assemblies.

The native Windows DLL is supply-chain bound to the official MuJoCo 3.12.0
PyPI wheel: `uv.lock` pins the wheel URL/hash and `upstream.lock.json` pins its
member path plus the extracted DLL hash. Bootstrap compares both the Python
environment copy and the Tuanjie plug-in copy; the player gate compares the
freshly packaged DLL against that same immutable value.

The primary machine-readable verdict is `artifacts/mvp/mvp-report.json`.
Behavior and import gates also write focused reports so a failure can be traced
to the first policy, asset, metric, or runtime boundary that violated its
contract.

## Deliberate non-goals

- The PhysX scene is not a replacement for native MuJoCo behavior parity.
- Trace parity validates policy observations/actions, not arbitrary scene or
  physics-state round-trip export.
- NWH is not part of this robot MVP. It remains a candidate for a later,
  conventional Rigidbody vehicle adapter.
- Generated MicroDuck 3D assets are not cleared for commercial distribution;
  review the copied license and attribution records before publishing them.
