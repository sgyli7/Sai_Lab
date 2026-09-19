# NWH assessment for the MicroDuck MVP

## Local evidence

- Inspected project: `E:\Work\DemoProjects\TestNWH`
- Project editor: Unity `2022.3.49f1`
- Installed NWH Vehicle Physics 2 asset version: `10.20f`
- Source availability: 202 C# files below `Assets/3rdParty/NWH`, including 131 C# files in
  `Vehicle Physics 2`; the runtime and editor/setup code needed for integration are present.
- The project already contains a runnable NWH scene, but it has not been copied into this
  repository because the asset is separately licensed and is not needed by the first vertical
  slice.

## Fit decision

NWH is a good later-stage dependency for conventional Rigidbody vehicles with wheel,
powertrain, transmission, and surface simulation. It is not the correct physics owner for the
first MicroDuck slice:

- MicroDuck has one free articulation root and fourteen policy-controlled revolute servos.
- Its roller variant adds four passive wheel joints; the learned policy still controls the same
  fourteen servos.
- NWH's vehicle/powertrain abstraction would therefore duplicate or fight the imported
  ArticulationBody hierarchy instead of helping reproduce the MuJoCo policy contract.

The MVP consequently keeps NWH out of the MicroDuck prefab. Its acceptance path uses the
official MuJoCo 3.12 native runtime inside Tuanjie as the physics authority; the separate
PhysX/ArticulationBody import is retained only for calibration and comparison. A later
conventional-vehicle slice can import the locally owned NWH asset behind a separate
vehicle-control adapter. Tuanjie compatibility must be tested in that slice; Unity 2022
lineage is encouraging but is not treated as proof.
