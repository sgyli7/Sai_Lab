# Deterministic Collision Hulls Implementation Plan

> **For agentic workers:** Execute this plan test-first, one vertical slice at a time.

**Goal:** Generate deterministic, PhysX-compatible collision meshes from MuJoCo's compiled
convex hulls without changing the visual meshes or the source-of-truth robot dynamics.

**Architecture:** The asset builder compiles a second in-memory `MjSpec` with every mesh capped at
128 hull vertices, exports only each non-tire collision hull to `Generated/CollisionMeshes`, and
augments applicable mesh entries in the Tuanjie import manifest with the generated asset path and
hash. Collision reports use schema v2 and describe convex-mesh or cylinder proxies; roller tires
remain cylinders.

**Tech Stack:** Python 3.12, MuJoCo `MjSpec`, pytest, Ruff.

---

### Task 1: Lock the observable bundle contract

**Files:**
- Modify: `tests/test_tuanjie_assets.py`

- [x] Add an integration test that generates the bundle and asserts report schema v2, a 128-vertex
  cap, four roller cylinders, non-tire convex-mesh assets, manifest linkage, hashes, and no collision
  OBJ above 252 triangles.
- [x] Run the focused test and confirm it fails because the current report is schema v1 and emits
  box/sphere metadata only.

### Task 2: Generate capped MuJoCo hull geometry

**Files:**
- Modify: `src/agenticrobot_bridge/tuanjie_assets.py`
- Test: `tests/test_tuanjie_assets.py`

- [x] Compile each source model through `MjSpec` after setting `mesh.maxhullvert = 128`.
- [x] Parse `mesh_graph` using MuJoCo's documented packed layout, compact referenced vertices, and
  export deterministic handedness-correct OBJ files for non-tire collidable meshes.
- [x] Cross-check shared mesh hulls across robot variants and reject missing/invalid graph data.
- [x] Run the focused test and confirm the generated topology passes.

### Task 3: Publish collision schema v2 and manifest linkage

**Files:**
- Modify: `src/agenticrobot_bridge/tuanjie_assets.py`
- Modify: `tests/test_tuanjie_assets.py`

- [x] Emit schema-v2 collision reports with `maxHullVertices`, `convexMeshCount`, and
  `cylinderCount`.
- [x] Add `collisionAssetFile` and `collisionAssetSha256` only to manifest meshes used by a
  non-tire collidable geom.
- [x] Keep tire records as cylinders with radius and half-width metadata.
- [x] Update legacy bundle assertions, rerun the focused file, then run the full Python suite and
  Ruff.
