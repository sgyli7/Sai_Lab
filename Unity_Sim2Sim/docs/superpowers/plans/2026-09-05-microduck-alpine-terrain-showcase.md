# MicroDuck Alpine Terrain Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a bright alpine Tuanjie showcase where the native MuJoCo MicroDuck can switch among seven real collision terrain modules and be inspected with a hybrid follow/free camera.

**Architecture:** A deterministic terrain catalog is the single source of truth for both native `MjGeom` collision and Tuanjie rendering. All modules live in one `MjScene`; a terrain navigator resets the active robot to catalogued spawn poses while camera and robot input are explicitly routed by mode.

**Tech Stack:** Tuanjie 1.10.2 / Unity 2022.3 API, official MuJoCo Unity plugin 3.12, Barracuda 3.0.1 CPU, NUnit EditMode/PlayMode tests, PowerShell/Python acceptance harnesses.

---

## File map

- `TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoTerrainCatalog.cs`: immutable module definitions and deterministic primitive generation.
- `TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoTerrainNavigator.cs`: active-module selection and safe MuJoCo reset.
- `TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoCameraRig.cs`: follow orbit, free flight, presets and focus.
- `TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoKeyboardPolicyInput.cs`: pause robot commands while free camera owns navigation keys.
- `TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoPolicyStatusOverlay.cs`: terrain/camera state and context-sensitive controls.
- `TuanjieProject/Assets/MicroDuck/Editor/AlpineEnvironmentBuilder.cs`: scene materials, sky, lights, scenery and `MjGeom` creation from the catalog.
- `TuanjieProject/Assets/MicroDuck/Editor/DemoSceneBuilder.cs`: compose the environment and runtime components.
- `TuanjieProject/Assets/MicroDuck/Tests/EditMode/MujocoTerrainCatalogTests.cs`: catalog and deterministic geometry tests.
- `TuanjieProject/Assets/MicroDuck/Tests/EditMode/MujocoCameraRigTests.cs`: pure camera state and input-ownership tests.
- `TuanjieProject/Assets/MicroDuck/Tests/EditMode/DemoSceneBuilderTests.cs`: generated-scene structure and visual configuration tests.
- `TuanjieProject/Assets/MicroDuck/Tests/PlayMode/MujocoTerrainPlayModeTests.cs`: native geom, reset, variant-switch and rollout tests.
- `scripts/run-environment-acceptance.ps1`: real Player terrain/camera capture.
- `tests/test_environment_acceptance.py`: acceptance harness contract tests.
- `scripts/run-mvp.ps1`: include environment evidence in the full gate.

### Task 1: Lock the terrain data contract

**Files:**
- Create: `TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoTerrainCatalog.cs`
- Create: `TuanjieProject/Assets/MicroDuck/Tests/EditMode/MujocoTerrainCatalogTests.cs`

- [ ] **Step 1: Write the failing catalog tests**

Assert the exact stable IDs, upstream dimensions, deterministic random-grid output, seven distinct spawn poses, and that every rendered primitive is also marked for MuJoCo collision.

```csharp
[Test]
public void CatalogContainsTheSevenApprovedModules()
{
    Assert.That(MujocoTerrainCatalog.Modules.Select(x => x.Id), Is.EqualTo(new[] {
        "flat_plaza", "upstream_pyramid_stairs", "upstream_random_grid",
        "upstream_pyramid_slope", "upstream_roller_slope", "rock_steps",
        "stairs_bridge"
    }));
}

[Test]
public void RandomGridIsDeterministicAndWithinUpstreamHeightLimit()
{
    TerrainModule a = MujocoTerrainCatalog.Find("upstream_random_grid");
    TerrainModule b = MujocoTerrainCatalog.Find("upstream_random_grid");
    Assert.That(a.Primitives, Is.EqualTo(b.Primitives));
    Assert.That(a.Primitives.Max(x => x.Size.y), Is.LessThanOrEqualTo(0.01f));
}
```

- [ ] **Step 2: Run EditMode tests and observe RED**

Run: `powershell -ExecutionPolicy Bypass -File scripts/run-mvp.ps1 -Stage tuanjie-editmode`

Expected: compile failure because `MujocoTerrainCatalog` does not exist.

- [ ] **Step 3: Implement the immutable catalog**

Define `TerrainPrimitive` (`Id`, `Center`, `Size`, `Euler`, `ColorKey`, `CollisionEnabled`) and `TerrainModule` (`Id`, `DisplayName`, `SpawnPosition`, `SpawnYaw`, `RecommendedSlots`, `Difficulty`, `Primitives`). Generate the 1.5 cm pyramid stairs, seeded 1 cm random grid, 1.7°–5.7° slope family, 2°–20° roller slope, rock steps and stairs/bridge without runtime randomness.

- [ ] **Step 4: Run EditMode tests and observe GREEN**

Run the same command. Expected: all existing tests plus catalog tests pass.

- [ ] **Step 5: Commit the slice**

```powershell
git add -- TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoTerrainCatalog.cs TuanjieProject/Assets/MicroDuck/Tests/EditMode/MujocoTerrainCatalogTests.cs
git commit -m "feat: define deterministic MicroDuck terrain catalog"
```

### Task 2: Build one native MuJoCo world with alpine visuals

**Files:**
- Create: `TuanjieProject/Assets/MicroDuck/Editor/AlpineEnvironmentBuilder.cs`
- Modify: `TuanjieProject/Assets/MicroDuck/Editor/DemoSceneBuilder.cs`
- Modify: `TuanjieProject/Assets/MicroDuck/Tests/EditMode/DemoSceneBuilderTests.cs`

- [ ] **Step 1: Write failing scene-structure tests**

Assert a `MicroDuck Alpine Environment` root, seven named module roots, an `MjGeom` for every collision primitive, no PhysX collider on scenery, one procedural skybox material, warm directional light, ambient/fog settings, and distinct terrain materials.

```csharp
[Test]
public void NativeSceneContainsApprovedAlpineWorld()
{
    Assert.That(GameObject.Find("MicroDuck Alpine Environment"), Is.Not.Null);
    foreach (TerrainModule module in MujocoTerrainCatalog.Modules)
        Assert.That(GameObject.Find($"Terrain/{module.Id}"), Is.Not.Null, module.Id);
    Assert.That(RenderSettings.skybox, Is.Not.Null);
    Assert.That(RenderSettings.skybox.shader.name, Is.EqualTo("Skybox/Procedural"));
}
```

- [ ] **Step 2: Run the focused EditMode suite and observe RED**

Run: `powershell -ExecutionPolicy Bypass -File scripts/run-mvp.ps1 -Stage tuanjie-editmode`

Expected: missing environment root and skybox assertions fail.

- [ ] **Step 3: Implement `AlpineEnvironmentBuilder.Build()`**

For every `TerrainPrimitive`, create one object with matching transform, an `MjGeom` box/sphere/capsule shape and a renderer that uses the catalog color key. Build procedural sky, sun, fill light, fog, mountains, conifers and clouds from built-in meshes/materials; remove/disable the imported infinite ground plane so it cannot mask module collision.

- [ ] **Step 4: Compose it from `CreateNativeSceneAsset()`**

Create the environment after the runtime `MjScene` root and before model instances so all active world geoms compile into both legged and roller models.

- [ ] **Step 5: Run EditMode tests and observe GREEN**

Expected: scene generation and all visual structure assertions pass.

- [ ] **Step 6: Commit the slice**

```powershell
git add -- TuanjieProject/Assets/MicroDuck/Editor/AlpineEnvironmentBuilder.cs TuanjieProject/Assets/MicroDuck/Editor/DemoSceneBuilder.cs TuanjieProject/Assets/MicroDuck/Tests/EditMode/DemoSceneBuilderTests.cs
git commit -m "feat: build native alpine terrain world"
```

### Task 3: Add safe terrain navigation and reset

**Files:**
- Create: `TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoTerrainNavigator.cs`
- Modify: `TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoDemoController.cs`
- Create: `TuanjieProject/Assets/MicroDuck/Tests/PlayMode/MujocoTerrainPlayModeTests.cs`

- [ ] **Step 1: Write failing PlayMode tests**

Load the generated native scene, cycle all seven module IDs, assert native geoms exist by name, reset root qpos to each spawn, assert all qvel values are zero, then switch legged → roller → legged and assert the same environment geom set remains compiled.

```csharp
[UnityTest]
public IEnumerator EveryTerrainResetUsesItsNativeSpawn()
{
    foreach (TerrainModule module in MujocoTerrainCatalog.Modules) {
        Assert.That(navigator.Select(module.Id), Is.True);
        yield return new WaitForFixedUpdate();
        Assert.That(ReadRootPosition(scene), Is.EqualTo(module.SpawnPosition).Within(1e-4));
        Assert.That(ReadMaximumAbsoluteVelocity(scene), Is.LessThan(1e-8));
    }
}
```

- [ ] **Step 2: Run PlayMode and observe RED**

Run: `powershell -ExecutionPolicy Bypass -File scripts/run-mvp.ps1 -Stage tuanjie-playmode`

Expected: `MujocoTerrainNavigator` is missing.

- [ ] **Step 3: Implement navigator and controller reset API**

`Select`, `Next`, and `Previous` call a controller reset method that writes the free-joint pose directly into current `MjData`, clears all qvel/ctrl/warm-start state, resets policy command history and calls `mj_forward` before resuming policy ticks.

- [ ] **Step 4: Run PlayMode and observe GREEN**

Expected: all modules and both variants retain real native collision geometry.

- [ ] **Step 5: Commit the slice**

```powershell
git add -- TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoTerrainNavigator.cs TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoDemoController.cs TuanjieProject/Assets/MicroDuck/Tests/PlayMode/MujocoTerrainPlayModeTests.cs
git commit -m "feat: navigate native MicroDuck terrain modules"
```

### Task 4: Replace the fixed camera with a hybrid director rig

**Files:**
- Create: `TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoCameraRig.cs`
- Delete: `TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoFollowCamera.cs`
- Modify: `TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoKeyboardPolicyInput.cs`
- Create: `TuanjieProject/Assets/MicroDuck/Tests/EditMode/MujocoCameraRigTests.cs`
- Modify: `TuanjieProject/Assets/MicroDuck/Tests/PlayMode/NativeMujocoScenePlayModeTests.cs`

- [ ] **Step 1: Write failing camera tests**

Test follow/free transitions, orbit clamps, zoom clamps, side/rear/top/showcase presets, `F` focus, active-variant target refresh and `OwnsNavigationInput == true` only in free-flight mode.

- [ ] **Step 2: Run EditMode and observe RED**

Expected: `MujocoCameraRig` does not exist.

- [ ] **Step 3: Implement the camera rig**

Use a serializable `CameraMode` and pure state update methods for testability. Follow mode consumes drag/wheel; free mode consumes RMB mouse and `WASD/QE`; `C` cycles presets and `F` focuses. Preserve a clear 35–55° field of view and near clip suitable for the 20 cm robot.

- [ ] **Step 4: Route keyboard ownership**

Bind `MujocoKeyboardPolicyInput` to the camera rig. When `OwnsNavigationInput` is true, send zero twist instead of reading movement keys; policy-number, reset and skill keys remain available.

- [ ] **Step 5: Run EditMode and PlayMode and observe GREEN**

Expected: camera state tests pass and the active Duck stays frameable in follow mode through model switches.

- [ ] **Step 6: Commit the slice**

```powershell
git add -- TuanjieProject/Assets/MicroDuck/MujocoRuntime TuanjieProject/Assets/MicroDuck/Tests
git commit -m "feat: add hybrid MicroDuck director camera"
```

### Task 5: Integrate terrain controls and operator HUD

**Files:**
- Modify: `TuanjieProject/Assets/MicroDuck/Editor/DemoSceneBuilder.cs`
- Modify: `TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoKeyboardPolicyInput.cs`
- Modify: `TuanjieProject/Assets/MicroDuck/MujocoRuntime/MujocoPolicyStatusOverlay.cs`
- Modify: `TuanjieProject/Assets/MicroDuck/Tests/EditMode/DemoSceneBuilderTests.cs`

- [ ] **Step 1: Write failing integration tests**

Assert scene bindings among controller, navigator, camera, input and HUD. Assert the HUD text contains current terrain, difficulty, compatibility label, camera mode, position and upright score.

- [ ] **Step 2: Run EditMode and observe RED**

- [ ] **Step 3: Add `T`/`Shift+T` terrain cycling and context-aware HUD**

Show `训练匹配` only for catalogued policy/terrain pairs and `训练扩展` for showcase-only obstacles. Display the correct keys for Follow versus Free Fly.

- [ ] **Step 4: Rebuild the native scene and run both Tuanjie suites**

Run EditMode and PlayMode stages. Expected: all integration and existing policy tests pass.

- [ ] **Step 5: Commit the slice**

```powershell
git add -- TuanjieProject/Assets/MicroDuck
git commit -m "feat: integrate terrain controls and operator HUD"
```

### Task 6: Prove policy-aware terrain behavior

**Files:**
- Modify: `TuanjieProject/Assets/MicroDuck/Tests/PlayMode/MujocoTerrainPlayModeTests.cs`
- Modify: `TuanjieProject/Assets/MicroDuck/Tests/PlayMode/OfficialPolicyPlayModeTests.cs`

- [ ] **Step 1: Add failing behavior assertions**

Retain all-nine-policy finite tensor coverage. Add a six-second flat and gentle-rough walking rollout with upright > 0.85 and positive progress; add roller flat/slope rollouts with finite state and no floor penetration. Extended obstacles require native contact/reset correctness, not unsupported-policy success.

- [ ] **Step 2: Run PlayMode and observe any behavior failures**

- [ ] **Step 3: Correct spawn poses/contact parameters only from evidence**

Adjust catalog geometry/spawn/contact values without changing policy outputs or weakening assertions. Record failure metrics in assertion messages.

- [ ] **Step 4: Run PlayMode until GREEN**

- [ ] **Step 5: Commit the slice**

```powershell
git add -- TuanjieProject/Assets/MicroDuck
git commit -m "test: verify policy-aware terrain behavior"
```

### Task 7: Automate visual and Player acceptance

**Files:**
- Create: `scripts/run-environment-acceptance.ps1`
- Create: `tests/test_environment_acceptance.py`
- Modify: `scripts/run-mvp.ps1`

- [ ] **Step 1: Write failing Python contract tests**

Require a fresh report containing build hash, Player PID/path, seven terrain captures, all camera modes, native policy health, and `requiresVisualReview=true`. Reject stale timestamps, missing screenshots, or a report that claims visual success without review.

- [ ] **Step 2: Run Python tests and observe RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_environment_acceptance.py -q`

- [ ] **Step 3: Implement the acceptance harness**

Launch the exact Windows build, drive `T`, `Shift+T`, `Tab`, `C`, `F`, mouse orbit and wheel zoom, capture named PNGs plus MP4, collect Player log errors and write structured JSON. Do not auto-assert aesthetic quality.

- [ ] **Step 4: Add the environment gate to `run-mvp.ps1`**

The normal full gate validates the environment report schema after build; interactive capture remains an explicit final acceptance step.

- [ ] **Step 5: Run Python tests and observe GREEN**

- [ ] **Step 6: Commit the slice**

```powershell
git add -- scripts/run-environment-acceptance.ps1 scripts/run-mvp.ps1 tests/test_environment_acceptance.py
git commit -m "test: automate MicroDuck environment acceptance"
```

### Task 8: Clean-room verification and delivery

**Files:**
- Modify only if evidence exposes a defect.

- [ ] **Step 1: Run the full clean MVP pipeline**

Run: `powershell -ExecutionPolicy Bypass -File scripts/run-mvp.ps1 -Stage all -CleanGenerated`

Expected: Python, MuJoCo, all-nine ONNX, EditMode, PlayMode, trace, Codely and Windows build gates pass with a fresh `artifacts/mvp/mvp-report.json`.

- [ ] **Step 2: Run interactive environment acceptance**

Run: `powershell -ExecutionPolicy Bypass -File scripts/run-environment-acceptance.ps1`

Expected: seven terrain frames, follow/free/preset frames, MP4, no Player errors and a fresh structured report.

- [ ] **Step 3: Inspect every produced image/video**

Reject and fix grey voids, missing sky, scale problems, camera clipping, unreadable Duck silhouette, duplicated/absent terrain, visual/collision mismatch or unstable robot behavior.

- [ ] **Step 4: Launch the verified Player for user acceptance**

Leave `Builds/Windows64/AgenticRobotGame.exe` running in follow mode on `flat_plaza`, with standing policy healthy and the full alpine world visible.

- [ ] **Step 5: Commit final evidence-driven fixes**

```powershell
git add -- TuanjieProject/Assets/MicroDuck scripts tests docs
git commit -m "feat: deliver verified MicroDuck alpine showcase"
```
