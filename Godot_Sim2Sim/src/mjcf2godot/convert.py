#!/usr/bin/env python3
"""Compile an MJCF model with MuJoCo and emit a Godot 4 + Jolt scene.

The compiled MjModel / mj_forward(qpos0) state is the single source of truth.
XML is never re-parsed. Unmappable fields are listed in robot_spec.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from sim2sim.coords import basis_from_z, fmt_f, m2g_mat, m2g_vec, mat_to_quat_wxyz

GEOM_NAMES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
    int(mujoco.mjtGeom.mjGEOM_SDF): "sdf",
}

JOINT_NAMES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}


def _name(model: mujoco.MjModel, ntype, i: int) -> str:
    raw = mujoco.mj_id2name(model, ntype, i)
    return raw if raw else f"unnamed_{i}"


def _mat(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).reshape(3, 3)


def _fmt_transform(basis_cols: tuple[np.ndarray, np.ndarray, np.ndarray], origin: np.ndarray) -> str:
    """Godot tscn Transform3D 9 floats are basis *rows*, not concatenated axes.

    Basis(x_axis, y_axis, z_axis) stores
        rows[0] = (x.x, y.x, z.x)
    so the packed form is x.x, y.x, z.x, x.y, y.y, z.y, x.z, y.z, z.z.
    Writing x.x,x.y,x.z,... (axis-major) transposes the rotation. Body poses
    are overwritten on reset, but CollisionShape3D local transforms are not.
    """
    x, y, z = basis_cols
    nums = [
        x[0], y[0], z[0],
        x[1], y[1], z[1],
        x[2], y[2], z[2],
        origin[0], origin[1], origin[2],
    ]
    return "Transform3D(" + ", ".join(fmt_f(v) for v in nums) + ")"


def _identity_transform() -> str:
    return "Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)"


MAX_HULL_VERTS = 128
# STAND 2 mm support as one coplanar 16-gon (run vx not halved). Capsule
# pallet (FOOT_COLLISION="capsules") can form a t=1.26 toe+jaw tripod the
# 16-gon never does, but C3 z_min stays ~0.032; stagger/bumper/extend kill
# gait. One sagittal capsule is a 2-point heel–toe line: μ=1 still
# glue-pitches to t=0.92 and local_ppo falls. WorldBoundary floor is still
# a 4-point face. "fillet" (16 mm flat + raised toe) is kept as a branch:
# quarter-circle late-tipped C3 and killed run; 2 mm dual chamfer stood
# C3 on the heel (z_min~0.11); toe-only 2 mm slammed backward onto the
# jaw (z_min coincidentally 0.046, jaw x<0). 5 lateral capsules with a
# 0.6 mm heel-down stagger still compresses to all 5 lines by t=0.40;
# t=0.92 peels to the toe line but foot pitch is ~0.7° and hips hit.
# 6 capsules as a 0.8 mm binary step (3 heel + 3 toe): t=0.12 heel-only
# (right direction) but t=0.80 already both heights, t=0.92 toe line with
# ankle still ~0.012, then feet leave and hips hit; local_ppo run falls.
# Plant stays the full pad.
# C3 first floor hit is top_head_shell on jaw_soft (not the chin): a
# ~75 mm lateral knife edge. MJ solref lets it sink ~1.7 mm; the 512-hull
# first-hit is ~0.2 mm and trips the feet. Replace that one hull with a
# 2 mm lateral capsule on the min-z keel. jaw + bottom_head_shell stay hulls.
# Off (16-gon and migrating 3-spheres): 2 mm keel still launches the
# feet ~20 ms after first jaw contact. 4 mm radius does not change the
# first-hit height (bottom stays at mesh zmin). Plant is jaw hulls.
JAW_TOP_SHELL_CAPSULE = False
JAW_TOP_SHELL_RADIUS = 0.002
JAW_TOP_SHELL_BAND = 0.002
JAW_CHIN_HULLS = True
# 2 mm STAND band as a true 3D hull (no planar flatten). MJ C3 t=0.11
# first-hit is the two lowest hull verts on the outer lateral ridge
# (x≈0.022 and −0.009, y outer), dist already −4 mm. Flattening that
# band to a 16-gon puts the whole pad on one plane and glue-pitches.
# Off: 2 mm band as a true 3D hull (no planar flatten). MJ C3 t=0.11
# is the two lowest outer-ridge hull verts (dist already −4.2 mm).
# Godot still made a 3-point 35 mm sagittal manifold (camber 0.1 mm
# < slop 0.2 mm), peeled to the toe only at t=1.00, jaw launched
# the feet at t=1.32, z_min 0.031. Plant stays the full-pad 16-gon.
# Same 2 mm-band verts as band_hull, but one sphere per hull vert so
# Jolt cannot clip a face. Bottoms sit on the mesh (origin = p + r n̂).
# Off: same verts as spheres (point contacts). Default slop: n=64 full
# pad (report cap). slop=0.05 mm: t=0.40 heel-only like MJ, then body
# weight flattens the 0.1 mm camber by t=0.80; t=1.44 jaw still launches;
# z_min 0.031. Plant stays the full-pad 16-gon.
# Off: r=4 mm migrating 3-spheres (map MJ foot dist=−4.2 mm). t=1.32
# tripod still lost at t=1.34 — ankle jumped 5 mm, bottoms-on-mesh
# spheres cannot out-reach a hard jaw launch. speculative=2 mm made
# the t=1.34 jaw impulse 0.082 (worse). Plant stays 16-gon.
# Off: two sprung lateral capsules (heel+toe rails, host n_host=0,
# 6DOF at the capsule not the ankle COM, k_y=250, angular_z free).
# t=0.12 both rails (x≈−0.016/+0.030). Free pitch is a trapdoor:
# ankle COM goes through the floor (z≈−0.02) while the rail stays
# in contact. Peel is BACKWARD (toe unloads at t=1.14, jaw x<0 by
# t=0.80). t=0.92 tilt 7.5°. z_min 0.0454 at t=2.28 is occiput
# (jaw x −0.24). Plant stays 16-gon.
FOOT_COLLISION = "prism"
FOOT_SPHERE_RADIUS = 0.001
FOOT_SPHERE_N = 3
FOOT_SPHERE_BOTTOM = 0.0005
FOOT_BAND_HULL_CAP = 2048
FOOT_BAND_RIDGE = 0.0002
# Off: two sprung spheres at MJ C3 first-hit verts (outer ridge heel+toe).
# t=0.12 matches MJ (n=1 each, x≈−0.014/+0.028). Sphere-floor is still a
# hard constraint: t=0.40 vz>+0.7 bounce, t=0.80 hips+jaw, z_min 0.030.
# k=2e3/5e3 and mass 0.002/0.015 do not turn that into solref. Plant stays 16-gon.
# Off: ~16 mm heel sibling + 3 mm raised fore, Generic6DOF linear springs
# (k=1e5, c=200 and c=2000). t=0.40 heel-only (stagger held after impact).
# t=0.92 tilt 58° (full pad 31°, MJ 81°), t=1.00 77° / vz≈−0.34, t=1.14
# jaw n=6 launches feet, hips down, z_min 0.0315. Same short-pad conflict:
# peel is fast enough to hit the head with high vn. Plant stays 16-gon.
# Off: coplanar heel+toe LEAVES (host n_host=0, 6DOF k_y=900, k_pitch=1,
# mass=0.02, split x=0.008). Springs sag ~3 mm at t=0.12, but both leaves
# stay a coplanar 4+4 face through t=0.80 (tilt 22°, ankle z=0.010). Heel
# unloads at t=0.92 (tilt still 31°); remaining toe face ~21 mm then
# launches at jaw t=1.26. z_min 0.0311 hips. Two hard coplanar faces + Y
# springs do not unlock pitch — kinematically a rigid pad. Plant stays 16-gon.
FOOT_SPLIT_LEAVES = False
FOOT_SPLIT_HEEL = False
# Off: 16 mm heel host + far-toe sibling capsule (mass=0.03) gets
# t=1.22 tilt 81° / ankle z 0.026 / n_toe=2 — then jaw n=1 at t=1.24
# trips the μ=1 line (ank vz jump, feet leave, hips hit, z_min 0.031).
# Full jaw hull (cap 8192) still first-hit n=2. Full-pad edge capsule
# holds a 2 mm couple through the jaw frame then launches harder.
# Same-body second shape is eaten by Jolt's manifold. Hip hull is not fat.
# Off: short-heel sibling + 1.5 mm fore raise + Generic6DOF linear springs
# (k=3e3 and 3e4). Impact still flattens the stagger by t=0.12 (both
# islands n=4). k=3e4 reaches t=0.92 tilt 56° (full pad 31°, MJ 81°)
# but feet are gone before jaw, z_min 0.031. Plant stays the full pad.
FOOT_SPLIT_TOE = False
FOOT_TOE_KIND = "edge_capsule"
FOOT_SPLIT_X = -0.006
# Unused while host is the full 16-gon; kept for the convex-toe branch.
FOOT_HOST_X_MAX = 0.028
FOOT_TOE_X_MIN = 0.025
FOOT_TOE_Z_OFFSET = 0.0
FOOT_TOE_WINDOW = 0.003
FOOT_WORLD_X_MIN: float | None = None
FOOT_FORE_Z_OFFSET = 0.0
FOOT_HEEL_Z_OFFSET = 0.0
# Off: 6 sibling Y-sprung spheres (3×2 on the 2 mm envelope, host n_host=0,
# k_y=180, k_xz=2000, mass=0.01). Point contacts peel heel→mid→toe, and
# t=1.26 is a brief jaw+toe tripod, but t=0.92 tilt is only 27° (full pad
# 31°, MJ 81°) — 4 remaining points still a polygon. z_min 0.0318 hips.
# Springs sag <0.5 mm differentially; they do not buy MJ's 4.2 mm early peel.
FOOT_SIBLING_SPRING = False
# Off: 32 mm MJ first-hit window x∈[-0.009,0.023]. t=0.92 still 4-point /
# tilt 29° (full pad 31°); t=1.24 brief toe+jaw then launch, z_min 0.0312.
# Off: xmin=-0.006 2.5 s z_min 0.0464 is BACKWARD (jaw x −0.25, tilt 1.5°
# at t=0.92 then falls onto the occiput). Old 0.050 was not MJ C3.
# Off: body_pair_contact_cache=false does not peel (t=0.92 still 31°).
# Off: SoftBody rubber sole (scene pads, top-pinned, k=0.7): runtime
# SoftBody never enters Jolt space; scene pads punch through (hips at
# t=1.14, z_min 0.031). Off: scene SoftBody floor (grid-pin 12 cm plate
# sits; all-pin = no Rigid contacts; 45×20 mm foot plate punches or
# trampolines; BoxMesh mattress inflates). Off: 0.6 m sheet × 5 mm
# verts (14884 points): 8 cm lattice / 20 mm lattice / 40 mm pin-lines /
# total_mass=400 still punch a 45×20 mm plate. Jolt SoftBody cannot
# both resolve a foot and keep Rigid contacts. Off: HeightMap egg-carton floor
# (A=1.2 mm, λ=30 mm): spike box sits, C3 valley foot tunnels (ankle
# z≈−0.016), t=0.80 tilt 86°, z_min 0.0354 is a one-foot trapdoor.
# Off: same-body 16 mm heel + 22 mm gap + 10 mm toe at +1.5 mm: t=0.92
# tilt 48° (full pad 31°) jaw x +0.17, then feet leave at t=1.14, hips,
# z_min 0.031. Sibling small toe + per-frame weld sync overshoots
# (t=0.92 83° feet already gone). Short remaining face cannot hold
# 81° and the jaw together. Plant stays the full-pad 16-gon.
# Off: monolithic sprung floor (Generic6DOF springs ARE real: spike
# k=2000 holds 1 kg at 4.9 mm). Y-only k=2000 mass=0.15: sag 4 mm,
# t=0.92 still 32°/4-point, jaw-frame vz≈0 then hips, z_min 0.027.
# Pitch hinge k=0.25 + custom I_z=0.008: plate tips 34° by t=1.24 but
# is a trapdoor (robot tunnels, z_min −8). Shared-Y does not unlock
# robot-to-plate pitch; a world hinge cannot replace contact peel.
# Off: COM-aligned heavy tile GRID (6DOF at each tile COM, unlike the
# origin-offset joints that applied 0 force). k=2000/mass=1.5 rides
# tiles to z=−0.6; k=20000/mass=0.4 still sags 15.5 mm at t=0.12
# (impulse vs tile mass; spring force is 0 at x=0) then trampolines
# to z=2.5 m. First hit lands on 1–2 tiles, not one shared plate.
# Series mass-spring is not solref. Plant stays the static box.
# Off: 6 world-XY 5 mm slats toe-plant (jaw x<0, z_min 0.0386 occiput).
# Sole-plane 4 slats (n_host=4): t=0.12 heel slat, jaw x>0, but mid
# slats keep t=0.92 at 33° (empty-mid 2-island was 48°). Toe launches
# at jaw t=1.24, z_min 0.0317 hips. Plant stays the full-pad 16-gon.
FOOT_STRIPS = (
    {"world_x_min": None, "world_x_max": None, "z_offset": 0.0, "n_outline": 16},
)
# Off: 4.2 mm parabolic camber (MJ first-hit solref depth). Mid apex:
# t=0.12 midfoot, rolls BACK onto the heel, jaw x<0, z_min 0.0470 occiput.
# Heel apex: peel matches MJ through t=0.80 (49° vs 46°, toe-only) and
# t=0.92 is 75° vs 81° with a 0.4 mm toe edge; jaw still launches, z_min 0.0316.
# Anterior add-ons on that heel-camber (all n_host=2, second convex; same-hull
# lip flattened the rocker and planted at t=0.40):
# - 4 mm forward slab: t=0.80 already on the lip, t=0.92 70° / 0.5 mm line.
# - 2 mm STAND-vertical wall: body tilt 74° is not foot pitch (~25° from
#   ankle rise); wall still 65° to the floor, t=0.92 0.7 mm line.
# - 30° triangular wedge: t=0.80 mixed pad+wedge; t=0.92 already on the
#   front tip (cx 0.047); jaw impulse 0.114 (10× camber-only) launched harder.
# - R=12 mm 12-facet quarter-circle: t=0.80 span 14 mm; t=0.92 0.5 mm line
#   at cx 0.041; jaw impulse 0.116, z_min 0.0310.
# Extra anterior geometry lets the body pitch more so jaw hits harder.
# Jolt convex vs box still degenerates to a line at that foot pitch.
# Off: heel camber + monolithic Y-spring floor (k=2000, mass=0.15,
# coinciding-COM 6DOF). Camber still peels (t=0.80 55° toe-only) but the
# plate sags 4–8 mm so jaw meets it at t=0.92 (n=9, impulse 0.242) instead
# of t=1.00. Feet gone by t=1.00, hips on the sunken plate, z_min 0.0264
# (zrel 0.032). Early peel + a moving floor makes jaw earlier and harder.
# Off: heel camber + migrating jaw Y 6DOF (jaw-floor collision excluded,
# k=2000, lowest hull vert). Camber peel still matches through t=0.92
# (74.7°, 0.4 mm toe, jaw x +0.23, jpad still +24 mm). No tripod window:
# t=1.00 feet already n=0 / tilt 87° as the spring first compresses
# 8.8 mm (vn≈0.40); head then punches to jpad −47 mm, hips at t=1.14,
# z_min 0.0274. Camber never reaches MJ's 81°+feet+head overlap; a
# post-peel head spring cannot recapture a 0.4 mm edge the robot has
# already rolled past. Plant stays the flat 16-gon on the static box.
# Off: heel-tangent CIRCULAR rocker R=90 mm (n_outline=32). Chord 48 mm
# is φ≈32° / ~14 mm toe rise. Peels too fast: t=0.12 heel-only (span
# 1.5 mm), t=0.40 already 15.4° (parabola 10.8°), t=0.80 tilt 86° / jaw
# n=10 / feet already n=0 / impulse 0.040, z_min 0.0308 at t=0.96 hips.
# More roll range is a taller banana, not a longer tripod window. A
# single heel-tangent circle cannot be both 4.2 mm-gentle and 32° of
# arc. Plant stays the flat 16-gon on the static box.
FOOT_CAMBER_SAGITTA = 0.0
FOOT_CAMBER_APEX = "heel"
FOOT_CAMBER_RADIUS = 0.0
FOOT_TOE_LIP_LEN = 0.0
FOOT_TOE_LIP_HEIGHT = 0.0
FOOT_TOE_EXTEND_X = 0.0
FOOT_HEEL_MASS = 0.002
FOOT_HEEL_INERTIA = (1.0e-6, 1.0e-6, 1.0e-6)
FOOT_HEEL_LINEAR_DAMP = 0.0
FOOT_CAPSULE_RADIUS = 0.0015
FOOT_CAPSULE_N = 3
# 3 coplanar: C3 0.032, run vx 0.187. 6-cap 0.8 mm heel/toe step: C3
# still 0.032 (peel to toe at t=0.92, hips hit); local_ppo run falls.
FOOT_CAPSULE_Z_OFFSETS = (0.0, 0.0, 0.0)
FOOT_CAPSULE_TOE_EXTEND = 0.0
FOOT_CAPSULE_BUMPER_Z = 0.0
# Sagittal chamfer: 16 mm mid-foot flat (the strip that C3-tipped) +
# raised heel/toe so the hull is two ramps, not a C1 fillet.
# Quarter-circle (r≈16 mm) started at slope 0, so slop made a ~26 mm
# face; C3 t=0.92 stayed 17° and local_ppo vx collapsed to 0.03.
# Walk planted toe-off is only ~6°, so end height must be ≲2 mm
# (16 mm * tan 6°) or the toe never contacts. Revert to prism if
# the 2 mm catch stops C3 like a rocker or run still dies.
FOOT_FILLET_FLAT = 0.016
FOOT_CHAMFER_H = 0.002
FOOT_FILLET_N_ARC = 4
FOOT_FILLET_THICKNESS = 0.020


def _convex_points(verts: np.ndarray, max_verts: int = MAX_HULL_VERTS) -> np.ndarray:
    if len(verts) < 4:
        return verts
    try:
        hull = ConvexHull(verts)
        pts = verts[np.unique(hull.vertices)]
    except (QhullError, ValueError):
        pts = verts
    if len(pts) <= max_verts:
        return pts
    sampled = _farthest_point_sample(pts, max_verts)
    try:
        hull2 = ConvexHull(sampled)
        return sampled[np.unique(hull2.vertices)]
    except (QhullError, ValueError):
        return sampled


def _tire_cylinder(verts_i: np.ndarray) -> dict:
    """Wheel mesh convex hull is a pancake that sits on a face and will not roll.

    Cylinder along the thin PCA axis (the axle) matches a skate wheel.
    """
    pts = np.asarray(verts_i, dtype=np.float64)
    origin = pts.mean(axis=0)
    x = pts - origin
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    y_axis = np.asarray(vt[-1], dtype=np.float64)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-18)
    tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(y_axis[2])) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = np.cross(tmp, y_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-18)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-18)
    radial = x - np.outer(x @ y_axis, y_axis)
    radius = float(np.linalg.norm(radial, axis=1).max())
    height = 2.0 * float(np.abs(x @ y_axis).max())
    return {"radius": radius, "height": height, "origin": origin, "basis": (x_axis, y_axis, z_axis)}


def _jaw_top_shell_capsule(
    verts_i: np.ndarray,
    *,
    radius: float = JAW_TOP_SHELL_RADIUS,
    band: float = JAW_TOP_SHELL_BAND,
) -> dict:
    """Lateral capsule on top_head_shell's min-z keel (C3 first-hit edge)."""
    z = verts_i[:, 2]
    zmin = float(z.min())
    keel = verts_i[z <= (zmin + band)]
    if len(keel) < 4:
        keel = verts_i
    x = keel - keel.mean(axis=0)
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    y_axis = np.asarray(vt[0], dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(y_axis[2])) > 0.7:
        y_axis = np.asarray(vt[1], dtype=np.float64)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-18)
    x_axis = np.cross(y_axis, z_axis)
    nrm = float(np.linalg.norm(x_axis))
    if nrm < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / nrm
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-18)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-18)
    t = keel @ y_axis
    span = float(t.max() - t.min())
    y0 = 0.5 * (float(t.max()) + float(t.min()))
    x0 = float((keel @ x_axis).mean())
    origin = x0 * x_axis + y0 * y_axis + (zmin + radius) * z_axis
    height = max(span, 2.0 * radius + 1e-4)
    return {
        "kind": "capsule",
        "radius": radius,
        "height": height,
        "origin": origin,
        "basis": (x_axis, y_axis, z_axis),
        "span": span,
        "zmin": zmin,
    }


def _farthest_point_sample(pts: np.ndarray, k: int, start: int | None = None) -> np.ndarray:
    n = len(pts)
    if n <= k:
        return pts
    chosen = np.empty(k, dtype=np.int64)
    chosen[0] = int(start) if start is not None else int(np.argmax(np.sum(pts * pts, axis=1)))
    dist = np.full(n, np.inf)
    for i in range(1, k):
        d = pts - pts[chosen[i - 1]]
        dist = np.minimum(dist, np.einsum("ij,ij->i", d, d))
        chosen[i] = int(np.argmax(dist))
    return pts[chosen]


def _mesh_verts_faces(model: mujoco.MjModel, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
    adr = int(model.mesh_vertadr[mesh_id])
    n = int(model.mesh_vertnum[mesh_id])
    verts = np.asarray(model.mesh_vert[adr : adr + n], dtype=np.float64)
    fadr = int(model.mesh_faceadr[mesh_id])
    fn = int(model.mesh_facenum[mesh_id])
    faces = np.asarray(model.mesh_face[fadr : fadr + fn], dtype=np.int32)
    return verts, faces


def _apply_stand_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for k in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k)
        if name == "STAND":
            mujoco.mj_resetDataKeyframe(model, data, k)
            mujoco.mj_forward(model, data)
            return
    # scene_ball.xml includes the same robot but omits scene.xml's keyframes.
    # The support-patch approximation must use the same reference posture in
    # both scenes; using qpos0 silently shortens the ball scene's foot hulls.
    # Match the named Microduck joints, never assume generic actuator ordering.
    stand = {
        "left_hip_yaw": 0., "left_hip_roll": -math.pi / 36,
        "left_hip_pitch": -.457924, "left_knee": -.00494, "left_ankle": .452984,
        "neck_pitch": math.pi / 9, "head_pitch": math.pi / 9,
        "head_yaw": 0., "head_roll": 0., "right_hip_yaw": 0.,
        "right_hip_roll": math.pi / 36, "right_hip_pitch": .457924,
        "right_knee": .00494, "right_ankle": -.452984,
    }
    joints = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in stand}
    if all(j >= 0 for j in joints.values()):
        for name, joint in joints.items():
            data.qpos[int(model.jnt_qposadr[joint])] = stand[name]
        mujoco.mj_forward(model, data)


def _foot_mesh_inertial(model: mujoco.MjModel, data: mujoco.MjData, bid: int, gidx: int, verts: np.ndarray) -> np.ndarray:
    """All mesh verts in the foot inertial frame at STAND."""
    saved = np.array(data.qpos, copy=True)
    _apply_stand_keyframe(model, data)
    ximat_g = _mat(data.geom_xmat[gidx])
    xpos_g = np.asarray(data.geom_xpos[gidx], dtype=np.float64)
    world = (ximat_g @ verts.T).T + xpos_g
    ximat_i = _mat(data.ximat[bid])
    xipos_i = np.asarray(data.xipos[bid], dtype=np.float64)
    local = (ximat_i.T @ (world - xipos_i).T).T
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return local


def _foot_sole_inertial(model: mujoco.MjModel, data: mujoco.MjData, bid: int, gidx: int, verts: np.ndarray) -> np.ndarray:
    """Support-patch verts in the foot body's inertial frame, at STAND."""
    saved = np.array(data.qpos, copy=True)
    _apply_stand_keyframe(model, data)
    ximat_g = _mat(data.geom_xmat[gidx])
    xpos_g = np.asarray(data.geom_xpos[gidx], dtype=np.float64)
    world = (ximat_g @ verts.T).T + xpos_g
    zmin = float(world[:, 2].min())
    support = world[world[:, 2] <= zmin + 0.002]
    if len(support) < 4:
        support = world
    ximat_i = _mat(data.ximat[bid])
    xipos_i = np.asarray(data.xipos[bid], dtype=np.float64)
    local = (ximat_i.T @ (support - xipos_i).T).T
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return local


def _resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    """Even arc-length resample of a closed 3D polygon."""
    if len(pts) <= n:
        return pts
    d = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total < 1e-12:
        return pts[:n]
    s = s / total
    closed = np.vstack([pts, pts[0]])
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    out = np.empty((n, 3), dtype=np.float64)
    for i, ti in enumerate(t):
        k = int(np.searchsorted(s, ti, side="right") - 1)
        k = min(max(k, 0), len(pts) - 1)
        t0, t1 = s[k], s[k + 1]
        a = 0.0 if t1 <= t0 else (ti - t0) / (t1 - t0)
        out[i] = closed[k] * (1.0 - a) + closed[k + 1] * a
    return out


def _stand_inertial_up(model: mujoco.MjModel, data: mujoco.MjData, bid: int) -> np.ndarray:
    saved = np.array(data.qpos, copy=True)
    _apply_stand_keyframe(model, data)
    z_i = _mat(data.ximat[bid]).T @ np.array([0.0, 0.0, 1.0])
    z_i = z_i / (np.linalg.norm(z_i) + 1e-18)
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return z_i


def _stand_plane_axes(z_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z_i[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_i = np.cross(tmp, z_i)
    x_i /= np.linalg.norm(x_i)
    y_i = np.cross(z_i, x_i)
    return x_i, y_i


def _planar_prism(local: np.ndarray, z_i: np.ndarray, x_i: np.ndarray, y_i: np.ndarray, *, thickness: float, n_outline: int) -> np.ndarray:
    xy = np.column_stack((local @ x_i, local @ y_i))
    try:
        hull = ConvexHull(xy)
        outline = local[np.unique(hull.vertices)]
    except (QhullError, ValueError):
        outline = local
    xy_o = np.column_stack((outline @ x_i, outline @ y_i))
    try:
        h2 = ConvexHull(xy_o)
        outline = outline[h2.vertices]
    except (QhullError, ValueError):
        pass
    outline = _resample_closed(outline, n_outline)
    zmin = float((local @ z_i).min())
    bottom = outline - np.outer(outline @ z_i - zmin, z_i)
    top = bottom + thickness * z_i
    return np.vstack([bottom, top])


def _lat_range_at(s: np.ndarray, l: np.ndarray, s0: float, window: float = 0.005) -> tuple[float, float]:
    mask = np.abs(s - s0) <= window
    if int(mask.sum()) < 2:
        return float(l.min()), float(l.max())
    return float(l[mask].min()), float(l[mask].max())


def _foot_sole_fillet_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    gidx: int,
    verts: np.ndarray,
    *,
    flat: float = FOOT_FILLET_FLAT,
    chamfer_h: float = FOOT_CHAMFER_H,
    n_arc: int = FOOT_FILLET_N_ARC,
    thickness: float = FOOT_FILLET_THICKNESS,
) -> tuple[np.ndarray, dict]:
    """One convex: short zmin flat + linear sagittal chamfers, full width.

    Quarter-circles are tangent to the flat (slope 0), so Jolt slop turns
    them into a longer face. Endpoints raised by `chamfer_h` make an immediate
    ramp; hull does not fill back to a 49 mm plane.
    """
    local = _foot_sole_inertial(model, data, bid, gidx, verts)
    z_i = _stand_inertial_up(model, data, bid)
    fwd_i, lat_i = _foot_fwd_lat(model, data, bid, z_i)
    s = local @ fwd_i
    l = local @ lat_i
    h = local @ z_i
    zmin = float(h.min())
    smin, smax = float(s.min()), float(s.max())
    smid = 0.5 * (smin + smax)
    half = 0.5 * flat
    s_lo, s_hi = smid - half, smid + half

    def add_station(pts: list[np.ndarray], s0: float, zoff: float) -> None:
        l0, l1 = _lat_range_at(s, l, s0)
        for lv in np.linspace(l0, l1, 4):
            pts.append((s0 * fwd_i) + (float(lv) * lat_i) + ((zmin + zoff) * z_i))

    bottom: list[np.ndarray] = []
    mid = local[np.abs(s - smid) <= half]
    if len(mid) >= 4:
        xy = np.column_stack((mid @ fwd_i, mid @ lat_i))
        try:
            hull = ConvexHull(xy)
            outline = mid[hull.vertices]
        except (QhullError, ValueError):
            outline = mid
        for p in outline:
            bottom.append(p - (p @ z_i - zmin) * z_i)
    add_station(bottom, s_lo, 0.0)
    add_station(bottom, s_hi, 0.0)
    # Toe chamfer only. A heel ramp caught C3 backward (z_min~0.11).
    # MuJoCo C3 peels forward onto the toe; walk/run toe-off is 6–40°.
    for a in np.linspace(0.0, 1.0, n_arc + 1)[1:]:
        add_station(bottom, s_hi + a * (smax - s_hi), a * chamfer_h)
    bot = np.vstack(bottom)
    top = bot + thickness * z_i
    pts = np.vstack([bot, top])
    try:
        pts = pts[np.unique(ConvexHull(pts).vertices)]
    except (QhullError, ValueError):
        pass
    hh = pts @ z_i
    near = pts[hh <= zmin + 0.001]
    meta = {
        "n_prism": len(pts),
        "n_strips": 1,
        "n_heel": 0,
        "n_toe": 0,
        "world_x_min": smin,
        "heel_z": 0.0,
        "split_heel": False,
        "fillet_flat": flat,
        "fillet_r_heel": chamfer_h,
        "fillet_r_toe": chamfer_h,
        "chamfer_h": chamfer_h,
        "s_span": smax - smin,
        "flat_s_span": float((near @ fwd_i).max() - (near @ fwd_i).min()) if len(near) else 0.0,
        "h_span": float(hh.max() - hh.min()),
    }
    return pts, meta


def _foot_sole_prism_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    gidx: int,
    verts: np.ndarray,
    *,
    thickness: float = 0.008,
    n_outline: int = 16,
    sagittal_half: float | None = None,
    world_x_min: float | None = None,
    world_x_max: float | None = None,
    z_offset: float = 0.0,
) -> np.ndarray:
    """Planar prism of the STAND 2 mm support outline.

    world_x_min/max: clip in STAND world-X. z_offset raises the whole prism
    along STAND-up (a separate heel convex must be raised; one convex hull
    of mixed heights would flatten the heel back to zmin).
    """
    local = _foot_sole_inertial(model, data, bid, gidx, verts)
    z_i = _stand_inertial_up(model, data, bid)
    x_i, y_i = _stand_plane_axes(z_i)
    saved = np.array(data.qpos, copy=True)
    _apply_stand_keyframe(model, data)
    ximat_i = _mat(data.ximat[bid])
    xipos_i = np.asarray(data.xipos[bid], dtype=np.float64)
    if sagittal_half is not None:
        fwd_i = ximat_i.T @ np.array([1.0, 0.0, 0.0])
        fwd_i = fwd_i - z_i * float(fwd_i @ z_i)
        nrm = np.linalg.norm(fwd_i)
        fwd_i = fwd_i / nrm if nrm > 1e-12 else x_i
        coord = local @ fwd_i
        mid = 0.5 * (float(coord.min()) + float(coord.max()))
        clipped = local[np.abs(coord - mid) <= sagittal_half]
        if len(clipped) >= 4:
            local = clipped
    if world_x_min is not None or world_x_max is not None:
        world = (ximat_i @ local.T).T + xipos_i
        mask = np.ones(len(local), dtype=bool)
        if world_x_min is not None:
            mask &= world[:, 0] >= world_x_min
        if world_x_max is not None:
            mask &= world[:, 0] <= world_x_max
        if int(mask.sum()) >= 4:
            local = local[mask]
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    prism = _planar_prism(local, z_i, x_i, y_i, thickness=thickness, n_outline=n_outline)
    if z_offset:
        prism = prism + z_offset * z_i
    if FOOT_CAMBER_RADIUS > 0.0 or FOOT_CAMBER_SAGITTA > 0.0:
        fwd_i, _lat_i = _foot_fwd_lat(model, data, bid, z_i)
        s = prism @ fwd_i
        smin = float(s.min())
        smax = float(s.max())
        span = smax - smin
        if span > 1e-6:
            if FOOT_CAMBER_RADIUS > 0.0:
                # Heel-tangent circle: z = R - sqrt(R^2 - ds^2), ds from heel.
                ds = np.clip(s - smin, 0.0, FOOT_CAMBER_RADIUS - 1e-6)
                off = FOOT_CAMBER_RADIUS - np.sqrt(
                    np.maximum(FOOT_CAMBER_RADIUS**2 - ds**2, 0.0)
                )
            elif FOOT_CAMBER_APEX == "heel":
                # Lowest at world-X min (heel); toe raised by sagitta.
                off = FOOT_CAMBER_SAGITTA * ((s - smin) / span) ** 2
            elif FOOT_CAMBER_APEX == "toe":
                off = FOOT_CAMBER_SAGITTA * ((smax - s) / span) ** 2
            else:
                smid = 0.5 * (smin + smax)
                off = FOOT_CAMBER_SAGITTA * ((s - smid) / (0.5 * span)) ** 2
            prism = prism + np.outer(off, z_i)
    return prism


def _foot_sole_corners(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    prism: np.ndarray,
) -> list[list[float]]:
    """Four bottom-face corners in inertial local (heel/toe × lateral).

    2 outer-ridge points (MJ C3 first-hit sagittal pair) were tried as
    same-foot unilateral Y springs: t=0.12 sag 7.7 mm / 4 on; t=0.92 still
    4 on / tilt 32° (ankle z 0.000, feet stay flat — no Coulomb); heel
    unloads only at t=1.00; t=1.20 jaw+toe-springs then launch, z_min
    0.0309, jaw x +0.09. Sagittal 2-point Y springs resist pitch and
    cannot turn torso fold into foot peel. Unused while SOLE_SPRINGS is
    off.
    """
    z_i = _stand_inertial_up(model, data, bid)
    fwd_i, lat_i = _foot_fwd_lat(model, data, bid, z_i)
    s = prism @ fwd_i
    l = prism @ lat_i
    h = prism @ z_i
    h0 = float(h.min())
    bottom = h <= h0 + 0.0015
    if int(bottom.sum()) < 4:
        bottom = np.ones(len(prism), dtype=bool)
    sb, lb = s[bottom], l[bottom]
    s0, s1 = float(sb.min()), float(sb.max())
    l0, l1 = float(lb.min()), float(lb.max())
    out: list[list[float]] = []
    for ss in (s0, s1):
        for lv in (l0, l1):
            p = ss * fwd_i + float(lv) * lat_i + h0 * z_i
            out.append([round(float(c), 6) for c in p])
    return out


def _foot_toe_lip_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    prism: np.ndarray,
) -> np.ndarray:
    """Gated off (FOOT_TOE_LIP_LEN=0). Faceted quarter-circle / wedge / wall
    at the cambered toe as a second host convex. See FOOT_CAMBER comments.
    """
    z_i = _stand_inertial_up(model, data, bid)
    fwd_i, lat_i = _foot_fwd_lat(model, data, bid, z_i)
    s = prism @ fwd_i
    l = prism @ lat_i
    h = prism @ z_i
    smax = float(s.max())
    toe = np.abs(s - smax) <= 0.004
    if int(toe.sum()) < 2:
        l0, l1 = float(l.min()), float(l.max())
        h_toe = float(h.min()) + FOOT_CAMBER_SAGITTA
    else:
        l0, l1 = float(l[toe].min()), float(l[toe].max())
        h_toe = float(h[toe].min())
    radius = FOOT_TOE_LIP_LEN
    n_seg = 12
    corners = []
    for lv in (l0, l1):
        corners.append(smax * fwd_i + float(lv) * lat_i + (h_toe + radius) * z_i)
        for i in range(n_seg + 1):
            phi = 0.5 * np.pi * i / n_seg
            ss = smax + radius * np.sin(phi)
            hh = h_toe + radius * (1.0 - np.cos(phi))
            corners.append(ss * fwd_i + float(lv) * lat_i + hh * z_i)
    return np.asarray(corners, dtype=np.float64)


def _foot_fwd_lat(model: mujoco.MjModel, data: mujoco.MjData, bid: int, z_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    saved = np.array(data.qpos, copy=True)
    _apply_stand_keyframe(model, data)
    fwd_i = _mat(data.ximat[bid]).T @ np.array([1.0, 0.0, 0.0])
    fwd_i = fwd_i - z_i * float(fwd_i @ z_i)
    nrm = np.linalg.norm(fwd_i)
    fwd_i = fwd_i / nrm if nrm > 1e-12 else _stand_plane_axes(z_i)[0]
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    lat_i = np.cross(z_i, fwd_i)
    lat_i /= np.linalg.norm(lat_i) + 1e-18
    return fwd_i, lat_i


def _extend_toe_world_x(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    pts: np.ndarray,
    dx: float,
    x_gate: float = 0.020,
) -> np.ndarray:
    """Push toe-side prism verts along STAND world +X (single convex)."""
    if dx <= 0.0:
        return pts
    saved = np.array(data.qpos, copy=True)
    _apply_stand_keyframe(model, data)
    ximat_i = _mat(data.ximat[bid])
    xipos_i = np.asarray(data.xipos[bid], dtype=np.float64)
    world = (ximat_i @ pts.T).T + xipos_i
    ext_i = ximat_i.T @ np.array([1.0, 0.0, 0.0])
    out = np.array(pts, copy=True)
    out[world[:, 0] >= x_gate] += dx * ext_i
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return out


def _foot_sole_capsules(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    gidx: int,
    verts: np.ndarray,
    *,
    radius: float = FOOT_CAPSULE_RADIUS,
    n: int = FOOT_CAPSULE_N,
) -> tuple[list[dict], dict]:
    """Lateral capsules on the STAND 2 mm support (one RigidBody, no face).

    A coplanar 16-gon vs plane is a 4-point manifold: Godot ankles stay at
    z≈0.010 through C3 t=0.92 (31° fold) while MuJoCo feet pitch onto the
    toe. Capsule axes follow STAND lateral, sitting on zmin, so heel/toe
    lines can unload without a restoring face couple.
    """
    local = _foot_sole_inertial(model, data, bid, gidx, verts)
    z_i = _stand_inertial_up(model, data, bid)
    _fwd_i, lat_i = _foot_fwd_lat(model, data, bid, z_i)
    y_axis = lat_i / (np.linalg.norm(lat_i) + 1e-18)
    z_axis = z_i / (np.linalg.norm(z_i) + 1e-18)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-18
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis) + 1e-18
    hf = local @ x_axis
    hl = local @ y_axis
    hz = local @ z_axis
    fmin, fmax = float(hf.min()), float(hf.max())
    zmin = float(hz.min())
    stations = np.linspace(fmin + radius, fmax - radius + FOOT_CAPSULE_TOE_EXTEND, n)
    df = (fmax - fmin) / max(n, 1)
    z_off = FOOT_CAPSULE_Z_OFFSETS
    if len(z_off) != n:
        z_off = tuple(np.linspace(0.0, float(z_off[-1]) if z_off else 0.0, n))
    pieces: list[dict] = []
    spans: list[float] = []
    for i, f0 in enumerate(stations):
        mask = (hf >= f0 - 0.5 * df) & (hf <= f0 + 0.5 * df)
        if int(mask.sum()) < 2:
            mask = np.ones(len(local), dtype=bool)
        span = float(hl[mask].max() - hl[mask].min())
        y0 = 0.5 * (float(hl[mask].min()) + float(hl[mask].max()))
        height = max(span, 2.0 * radius + 1e-4)
        spans.append(span)
        origin = f0 * x_axis + y0 * y_axis + (zmin + radius + float(z_off[i])) * z_axis
        pieces.append(
            {
                "kind": "capsule",
                "radius": radius,
                "height": height,
                "origin": origin,
                "basis": (x_axis, y_axis, z_axis),
                "attach": "host",
            }
        )
    if FOOT_CAPSULE_BUMPER_Z > 0.0:
        toe_mask = hf >= (fmax - 0.5 * df)
        if int(toe_mask.sum()) < 2:
            toe_mask = np.ones(len(local), dtype=bool)
        span = float(hl[toe_mask].max() - hl[toe_mask].min())
        y0 = 0.5 * (float(hl[toe_mask].min()) + float(hl[toe_mask].max()))
        height = max(span, 2.0 * radius + 1e-4)
        origin = fmax * x_axis + y0 * y_axis + (zmin + FOOT_CAPSULE_BUMPER_Z) * z_axis
        pieces.append(
            {
                "kind": "capsule",
                "radius": radius,
                "height": height,
                "origin": origin,
                "basis": (x_axis, y_axis, z_axis),
                "attach": "toe",
            }
        )
    meta = {
        "n_prism": 0,
        "n_strips": len(pieces),
        "n_heel": 0,
        "n_toe": sum(1 for p in pieces if p.get("attach") == "toe"),
        "world_x_min": fmin,
        "heel_z": 0.0,
        "split_heel": False,
        "capsule_n": len(pieces),
        "capsule_r": radius,
        "capsule_span": float(np.mean(spans)) if spans else 0.0,
        "capsule_z": list(z_off),
    }
    return pieces, meta


def _foot_sole_sprung_rails(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    gidx: int,
    verts: np.ndarray,
) -> tuple[list[dict], dict]:
    """Heel+toe lateral capsules on independent Y-sprung siblings.

    Same-body 2 capsules still lock pitch (a rigid line-pair). Coplanar
    leaves lock because each island is a face and both 6DOFs sat at the
    ankle COM. One capsule per sibling, 6DOF at the capsule, lets the
    host pitch by compressing the two shocks differently.
    """
    pieces, meta = _foot_sole_capsules(model, data, bid, gidx, verts, n=2)
    for i, piece in enumerate(pieces):
        piece["attach"] = f"p{i}"
    meta["n_host"] = 0
    meta["n_heel"] = len(pieces)
    meta["capsule_n"] = len(pieces)
    return pieces, meta


def _foot_sole_sagittal_capsule(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    gidx: int,
    verts: np.ndarray,
    *,
    radius: float = FOOT_CAPSULE_RADIUS,
) -> tuple[list[dict], dict]:
    """One heel–toe capsule on zmin (line contact, not a 4-point face).

    Lateral capsules make a coplanar pallet that still glue-pitchs. A
    sagittal line is MuJoCo's first C3 contact (outer-edge heel+toe) and
    can unload an endpoint without rolling like spheres.
    """
    local = _foot_sole_inertial(model, data, bid, gidx, verts)
    z_i = _stand_inertial_up(model, data, bid)
    fwd_i, lat_i = _foot_fwd_lat(model, data, bid, z_i)
    y_axis = fwd_i / (np.linalg.norm(fwd_i) + 1e-18)
    z_axis = z_i / (np.linalg.norm(z_i) + 1e-18)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-18
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis) + 1e-18
    hf = local @ y_axis
    hl = local @ x_axis
    hz = local @ z_axis
    fmin, fmax = float(hf.min()), float(hf.max())
    zmin = float(hz.min())
    height = max(fmax - fmin, 2.0 * radius + 1e-4)
    origin = (
        (0.5 * (float(hl.min()) + float(hl.max()))) * x_axis
        + (0.5 * (fmin + fmax)) * y_axis
        + (zmin + radius) * z_axis
    )
    piece = {
        "kind": "capsule",
        "radius": radius,
        "height": height,
        "origin": origin,
        "basis": (x_axis, y_axis, z_axis),
        "attach": "host",
    }
    meta = {
        "n_prism": 0,
        "n_strips": 1,
        "n_heel": 0,
        "n_toe": 0,
        "world_x_min": fmin,
        "heel_z": 0.0,
        "split_heel": False,
        "capsule_n": 1,
        "capsule_r": radius,
        "capsule_span": height,
        "capsule_axis": "sagittal",
        "capsule_z": [0.0],
    }
    return [piece], meta


def _foot_mesh_toe_capsule(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    gidx: int,
    verts: np.ndarray,
    *,
    radius: float = FOOT_CAPSULE_RADIUS,
) -> dict:
    """Lateral capsule on the STAND mesh toe lip (not the flattened 2 mm pad)."""
    local = _foot_mesh_inertial(model, data, bid, gidx, verts)
    z_i = _stand_inertial_up(model, data, bid)
    _fwd_i, lat_i = _foot_fwd_lat(model, data, bid, z_i)
    y_axis = lat_i / (np.linalg.norm(lat_i) + 1e-18)
    z_axis = z_i / (np.linalg.norm(z_i) + 1e-18)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-18
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis) + 1e-18
    hf = local @ x_axis
    hl = local @ y_axis
    hz = local @ z_axis
    fmax = float(hf.max())
    tip = hf >= (fmax - FOOT_TOE_WINDOW)
    if int(tip.sum()) < 2:
        tip = np.ones(len(local), dtype=bool)
    z_tip = float(hz[tip].min())
    sole = tip & (hz <= (z_tip + 0.003))
    if int(sole.sum()) < 2:
        sole = tip
    span = float(hl[sole].max() - hl[sole].min())
    y0 = 0.5 * (float(hl[sole].min()) + float(hl[sole].max()))
    f0 = 0.5 * (float(hf[sole].min()) + float(hf[sole].max()))
    height = max(span, 2.0 * radius + 1e-4)
    origin = (
        f0 * x_axis
        + y0 * y_axis
        + (z_tip + radius + FOOT_TOE_Z_OFFSET) * z_axis
    )
    return {
        "kind": "capsule",
        "radius": radius,
        "height": height,
        "origin": origin,
        "basis": (x_axis, y_axis, z_axis),
        "attach": "toe",
        "z_tip": z_tip,
        "f0": f0,
    }


def _foot_pad_edge_capsule(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    gidx: int,
    verts: np.ndarray,
    *,
    radius: float = FOOT_CAPSULE_RADIUS,
) -> dict:
    """Lateral capsule on the STAND pad toe edge (coplanar, no extra lever)."""
    local = _foot_sole_inertial(model, data, bid, gidx, verts)
    z_i = _stand_inertial_up(model, data, bid)
    _fwd_i, lat_i = _foot_fwd_lat(model, data, bid, z_i)
    y_axis = lat_i / (np.linalg.norm(lat_i) + 1e-18)
    z_axis = z_i / (np.linalg.norm(z_i) + 1e-18)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-18
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis) + 1e-18
    hf = local @ x_axis
    hl = local @ y_axis
    hz = local @ z_axis
    fmax = float(hf.max())
    zmin = float(hz.min())
    tip = hf >= (fmax - 0.004)
    if int(tip.sum()) < 2:
        tip = np.ones(len(local), dtype=bool)
    span = float(hl[tip].max() - hl[tip].min())
    y0 = 0.5 * (float(hl[tip].min()) + float(hl[tip].max()))
    radius = 0.002
    height = max(span, 2.0 * radius + 1e-4)
    origin = fmax * x_axis + y0 * y_axis + (zmin + radius) * z_axis
    return {
        "kind": "capsule",
        "radius": radius,
        "height": height,
        "origin": origin,
        "basis": (x_axis, y_axis, z_axis),
        "attach": "toe",
    }


def _foot_mesh_toe_spheres(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    gidx: int,
    verts: np.ndarray,
    *,
    radius: float = FOOT_CAPSULE_RADIUS,
) -> list[dict]:
    """Two point contacts on the STAND mesh toe lip (MuJoCo C3 is 1–2 points)."""
    local = _foot_mesh_inertial(model, data, bid, gidx, verts)
    z_i = _stand_inertial_up(model, data, bid)
    _fwd_i, lat_i = _foot_fwd_lat(model, data, bid, z_i)
    y_axis = lat_i / (np.linalg.norm(lat_i) + 1e-18)
    z_axis = z_i / (np.linalg.norm(z_i) + 1e-18)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-18
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis) + 1e-18
    hf = local @ x_axis
    hl = local @ y_axis
    hz = local @ z_axis
    fmax = float(hf.max())
    tip = hf >= (fmax - FOOT_TOE_WINDOW)
    if int(tip.sum()) < 2:
        tip = np.ones(len(local), dtype=bool)
    z_tip = float(hz[tip].min())
    sole = tip & (hz <= (z_tip + 0.003))
    if int(sole.sum()) < 2:
        sole = tip
    y_lo, y_hi = float(hl[sole].min()), float(hl[sole].max())
    f0 = 0.5 * (float(hf[sole].min()) + float(hf[sole].max()))
    pieces = []
    for frac in (0.3, 0.7):
        y0 = y_lo + frac * (y_hi - y_lo)
        origin = f0 * x_axis + y0 * y_axis + (z_tip + radius + FOOT_TOE_Z_OFFSET) * z_axis
        pieces.append(
            {
                "kind": "sphere",
                "radius": radius,
                "origin": origin,
                "attach": "toe",
            }
        )
    return pieces


def _foot_sole_band_hull(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    gidx: int,
    verts: np.ndarray,
    *,
    cap: int = FOOT_BAND_HULL_CAP,
    ridge: float = FOOT_BAND_RIDGE,
) -> tuple[list[dict], dict]:
    """Convex hull of the STAND zmin+2 mm verts (not a planar prism).

    `_planar_prism` projects this cloud onto one plane. The real mesh's
    lowest 0.05–0.10 mm is a ~6 mm-wide outer lateral ridge — that is
    MuJoCo's first floor contact, not a 36 mm-wide rectangle.
    """
    local = _foot_sole_inertial(model, data, bid, gidx, verts)
    z_i = _stand_inertial_up(model, data, bid)
    h = local @ z_i
    zmin = float(h.min())
    ridge_pts = local[h <= (zmin + ridge)]
    try:
        hull = ConvexHull(local)
        pts = local[np.unique(hull.vertices)]
    except (QhullError, ValueError):
        pts = local
    if len(pts) > cap:
        # Never FPS the ridge: 128-pt full-foot hull stood on random verts.
        n_ridge = min(len(ridge_pts), cap // 2)
        core = ridge_pts if len(ridge_pts) <= n_ridge else _farthest_point_sample(ridge_pts, n_ridge)
        fill_n = cap - len(core)
        fill = _farthest_point_sample(pts, min(len(pts), fill_n + len(core))) if fill_n > 0 else core
        merged = np.vstack([core, fill])
        _, uid = np.unique(np.round(merged, 7), axis=0, return_index=True)
        pts = merged[np.sort(uid)][:cap]
        try:
            hull2 = ConvexHull(pts)
            pts = pts[np.unique(hull2.vertices)]
        except (QhullError, ValueError):
            pass
    meta = {
        "n_prism": len(pts),
        "n_ridge": int(len(ridge_pts)),
        "zmin": zmin,
        "dz": float(h.max() - zmin),
        "capsule_n": 0,
    }
    return [{"kind": "convex", "points": pts, "attach": "host"}], meta


def _foot_sole_band_spheres(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    gidx: int,
    verts: np.ndarray,
    *,
    radius: float = FOOT_SPHERE_RADIUS,
    n_keep: int = FOOT_SPHERE_N,
    bottom: float = FOOT_SPHERE_BOTTOM,
) -> tuple[list[dict], dict]:
    """MJ-sized contact set: 3 point contacts on the sole envelope.

    MuJoCo stand/walk keeps 1–3 foot-floor contacts (reduction), not a
    clipped face. 1060 spheres still filled the pad (n=64). Three
    farthest-in-plane verts of the lowest 0.5 mm envelope match that count
    and keep the 0.1 mm outer-ridge height difference.
    """
    local = _foot_sole_inertial(model, data, bid, gidx, verts)
    z_i = _stand_inertial_up(model, data, bid)
    z_i = z_i / (np.linalg.norm(z_i) + 1e-18)
    x_i, y_i = _stand_plane_axes(z_i)
    h = local @ z_i
    zmin = float(h.min())
    envelope = local[h <= (zmin + bottom)]
    if len(envelope) < 3:
        envelope = local
    xy = np.column_stack((envelope @ x_i, envelope @ y_i))
    k = min(int(n_keep), len(envelope))
    pts = _farthest_point_sample(xy, k)
    # Map XY samples back to 3D envelope verts (nearest).
    chosen = []
    for p in pts:
        d = np.sum((xy - p) ** 2, axis=1)
        chosen.append(envelope[int(np.argmin(d))])
    pts3 = np.asarray(chosen, dtype=np.float64)
    pieces = [
        {"kind": "sphere", "radius": radius, "origin": p + radius * z_i, "attach": "host"}
        for p in pts3
    ]
    meta = {
        "n_prism": len(pieces),
        "n_ridge": int((h <= (zmin + FOOT_BAND_RIDGE)).sum()),
        "zmin": zmin,
        "dz": float((pts3 @ z_i).max() - (pts3 @ z_i).min()) if len(pts3) else 0.0,
        "capsule_n": 0,
    }
    return pieces, meta


def _foot_sole_sprung_pogos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    gidx: int,
    verts: np.ndarray,
    *,
    radius: float = FOOT_SPHERE_RADIUS,
) -> tuple[list[dict], dict]:
    """6 independent sibling spheres on the STAND 2 mm envelope.

    Two coplanar 16-gon leaves stayed a 4+4 face (pitch locked). Each pogo
    is one point contact + a Y spring, so the ankle can pitch like a car
    on 6 shocks. 3 sagittal × 2 lateral, host has no floor collision.
    """
    local = _foot_sole_inertial(model, data, bid, gidx, verts)
    z_i = _stand_inertial_up(model, data, bid)
    z_i = z_i / (np.linalg.norm(z_i) + 1e-18)
    fwd_i, lat_i = _foot_fwd_lat(model, data, bid, z_i)
    h = local @ z_i
    zmin = float(h.min())
    env = local[h <= (zmin + 0.002)]
    if len(env) < 6:
        env = local
    s = env @ fwd_i
    l = env @ lat_i
    s_q = np.quantile(s, [0.12, 0.50, 0.88])
    l_q = np.quantile(l, [0.22, 0.78])
    pieces: list[dict] = []
    used: set[int] = set()
    i = 0
    for sq in s_q:
        for lq in l_q:
            d = (s - sq) ** 2 + (l - lq) ** 2
            for idx in np.argsort(d):
                j = int(idx)
                if j in used:
                    continue
                used.add(j)
                p = env[j]
                pieces.append(
                    {
                        "kind": "sphere",
                        "radius": radius,
                        "origin": p + radius * z_i,
                        "attach": f"p{i}",
                    }
                )
                i += 1
                break
    meta = {
        "n_prism": len(pieces),
        "n_strips": len(pieces),
        "n_heel": len(pieces),
        "n_toe": 0,
        "capsule_n": 0,
        "zmin": zmin,
        "split_heel": False,
    }
    return pieces, meta


def _foot_sole_cloud(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    gidx: int,
    verts: np.ndarray,
) -> np.ndarray:
    """Inertial-local verts Jolt spheres can snap to (band + pitched toe)."""
    band = _foot_sole_inertial(model, data, bid, gidx, verts)
    all_i = _foot_mesh_inertial(model, data, bid, gidx, verts)
    z_i = _stand_inertial_up(model, data, bid)
    fwd_i, _lat = _foot_fwd_lat(model, data, bid, z_i)
    hz = all_i @ z_i
    hf = all_i @ fwd_i
    zmin = float(hz.min())
    fmax = float(hf.max())
    toe = all_i[(hf >= (fmax - 0.012)) & (hz <= (zmin + 0.008))]
    cloud = np.vstack([band, toe]) if len(toe) else band
    _, uid = np.unique(np.round(cloud, 4), axis=0, return_index=True)
    return cloud[np.sort(uid)]


def _foot_peel_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    gidx: int,
    verts: np.ndarray,
) -> tuple[list[dict], dict]:
    """Fore pad on the foot body; optional heel/toe pad on a sibling RigidBody3D."""
    if FOOT_COLLISION == "band_hull":
        return _foot_sole_band_hull(model, data, bid, gidx, verts)
    if FOOT_COLLISION == "sprung_pogos":
        return _foot_sole_sprung_pogos(model, data, bid, gidx, verts)
    if FOOT_COLLISION == "sprung_rails":
        return _foot_sole_sprung_rails(model, data, bid, gidx, verts)
    if FOOT_COLLISION == "band_spheres":
        return _foot_sole_band_spheres(model, data, bid, gidx, verts)
    if FOOT_COLLISION == "capsules":
        return _foot_sole_capsules(model, data, bid, gidx, verts)
    if FOOT_COLLISION == "sagittal_capsule":
        return _foot_sole_sagittal_capsule(model, data, bid, gidx, verts)
    if FOOT_COLLISION == "fillet":
        pts, meta = _foot_sole_fillet_points(model, data, bid, gidx, verts)
        return [{"kind": "convex", "points": pts, "attach": "host"}], meta
    pieces: list[dict] = []
    n_pts = 0
    if FOOT_SPLIT_LEAVES:
        # Host collision off. Heel/toe siblings share the full pad, split
        # sagittally with 1 mm overlap so there is no gap line.
        split = FOOT_SPLIT_X
        overlap = 0.001
        specs = (
            {
                "world_x_min": None,
                "world_x_max": split + overlap,
                "z_offset": 0.0,
                "n_outline": 16,
                "attach": "heel",
                "extend": 0.0,
            },
            {
                "world_x_min": split - overlap,
                "world_x_max": None,
                "z_offset": 0.0,
                "n_outline": 16,
                "attach": "toe",
                "extend": 0.0,
            },
        )
    elif FOOT_SPLIT_HEEL:
        specs = (
            {"world_x_min": FOOT_SPLIT_X, "world_x_max": None, "z_offset": FOOT_FORE_Z_OFFSET, "n_outline": 16, "attach": "host", "extend": 0.0},
            {
                "world_x_min": None,
                "world_x_max": FOOT_SPLIT_X,
                "z_offset": FOOT_HEEL_Z_OFFSET,
                "n_outline": 16,
                "attach": "heel",
                "extend": 0.0,
            },
        )
    elif FOOT_SPLIT_TOE:
        host_xmax = -0.006 if FOOT_TOE_KIND == "edge_capsule" else None
        specs = (
            {"world_x_min": None, "world_x_max": host_xmax, "z_offset": 0.0, "n_outline": 16, "attach": "host", "extend": 0.0},
        )
        if FOOT_TOE_KIND == "convex":
            specs = specs + (
                {
                    "world_x_min": FOOT_TOE_X_MIN,
                    "world_x_max": None,
                    "z_offset": FOOT_TOE_Z_OFFSET,
                    "n_outline": 12,
                    "attach": "toe",
                    "extend": FOOT_TOE_EXTEND_X,
                },
            )
    else:
        specs = tuple(
            {**s, "attach": "host", "extend": s.get("extend", FOOT_TOE_EXTEND_X)} for s in FOOT_STRIPS
        )
    for spec in specs:
        pts = _foot_sole_prism_points(
            model,
            data,
            bid,
            gidx,
            verts,
            n_outline=int(spec["n_outline"]),
            thickness=0.008,
            world_x_min=spec["world_x_min"],
            world_x_max=spec["world_x_max"],
            z_offset=float(spec["z_offset"]),
        )
        pts = _extend_toe_world_x(model, data, bid, pts, float(spec.get("extend", 0.0)))
        if len(pts) < 8:
            continue
        pieces.append({"kind": "convex", "points": pts, "attach": spec["attach"]})
        n_pts += len(pts)
    if FOOT_TOE_LIP_LEN > 0.0:
        host_pts = next(
            (p["points"] for p in pieces if p.get("attach") == "host" and p["kind"] == "convex"),
            None,
        )
        if host_pts is not None:
            lip = _foot_toe_lip_points(model, data, bid, host_pts)
            pieces.append({"kind": "convex", "points": lip, "attach": "host"})
            n_pts += len(lip)
    if FOOT_SPLIT_TOE and FOOT_TOE_KIND == "capsule":
        pieces.append(_foot_mesh_toe_capsule(model, data, bid, gidx, verts))
    elif FOOT_SPLIT_TOE and FOOT_TOE_KIND == "edge_capsule":
        pieces.append(_foot_pad_edge_capsule(model, data, bid, gidx, verts))
    elif FOOT_SPLIT_TOE and FOOT_TOE_KIND == "sphere":
        pieces.extend(_foot_mesh_toe_spheres(model, data, bid, gidx, verts))
    if not pieces:
        fore = _foot_sole_prism_points(
            model, data, bid, gidx, verts, sagittal_half=None, world_x_min=FOOT_WORLD_X_MIN
        )
        pieces = [{"kind": "convex", "points": fore, "attach": "host"}]
        n_pts = len(fore)
    meta = {
        "n_prism": n_pts,
        "n_strips": len(pieces),
        "n_heel": sum(1 for p in pieces if p.get("attach") == "heel"),
        "n_toe": sum(1 for p in pieces if p.get("attach") == "toe"),
        "world_x_min": FOOT_SPLIT_X if FOOT_SPLIT_HEEL else FOOT_WORLD_X_MIN,
        "heel_z": FOOT_HEEL_Z_OFFSET,
        "split_heel": FOOT_SPLIT_HEEL,
    }
    return pieces, meta


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# generated by mjcf2godot\n")
        for v in verts:
            f.write(f"v {v[0]:.8g} {v[1]:.8g} {v[2]:.8g}\n")
        if faces is not None and len(faces) > 0:
            for tri in faces:
                f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")
        else:
            # point cloud fallback — Godot still imports verts
            pass


def compile_model(mjcf: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def hinge_rest_at_q0(model: mujoco.MjModel, data: mujoco.MjData) -> dict[int, dict]:
    """Body-frame hinge axes and parent←child rest rotation at qpos=0.

    MuJoCo `q` is the twist about the hinge from this rest, not from qpos0
    (which for Microduck is the standing/home pose baked into the XML).
    """
    saved = np.array(data.qpos, copy=True)
    hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == hinge:
            data.qpos[int(model.jnt_qposadr[j])] = 0.0
    mujoco.mj_forward(model, data)
    out: dict[int, dict] = {}
    for j in range(model.njnt):
        body = int(model.jnt_bodyid[j])
        parent = int(model.body_parentid[body])
        axis_world = np.asarray(data.xaxis[j], dtype=np.float64)
        if parent == 0:
            r_parent = np.eye(3)
        else:
            r_parent = np.asarray(data.xmat[parent], dtype=np.float64).reshape(3, 3)
        r_child = np.asarray(data.xmat[body], dtype=np.float64).reshape(3, 3)
        r_rel = r_parent.T @ r_child
        out[j] = {
            "axis_world": axis_world,
            "axis_parent_body": r_parent.T @ axis_world,
            "axis_child_body": r_child.T @ axis_world,
            "rest_rel": r_rel,
            "rest_rel_q0_wxyz": mat_to_quat_wxyz(r_rel),
        }
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return out


def build_spec(model: mujoco.MjModel, data: mujoco.MjData, mjcf: Path) -> dict:
    unmapped: list[dict] = []
    mapping: list[dict] = []

    bodies = []
    for b in range(model.nbody):
        name = _name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        parent = int(model.body_parentid[b])
        bodies.append(
            {
                "id": b,
                "name": name,
                "parent_id": parent,
                "parent": _name(model, mujoco.mjtObj.mjOBJ_BODY, parent) if b > 0 else None,
                "mass": float(model.body_mass[b]),
                "inertia": [float(x) for x in model.body_inertia[b]],
                "ipos": [float(x) for x in model.body_ipos[b]],
                "iquat_wxyz": [float(x) for x in model.body_iquat[b]],
                "xipos": [float(x) for x in data.xipos[b]],
                "ximat": [float(x) for x in data.ximat[b]],
            }
        )
        if b > 0 and model.body_mass[b] <= 0:
            unmapped.append({"item": f"body:{name}", "reason": "mass<=0"})

    rest0 = hinge_rest_at_q0(model, data)
    joints = []
    for j in range(model.njnt):
        jtype = int(model.jnt_type[j])
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        body = int(model.jnt_bodyid[j])
        parent = int(model.body_parentid[body])
        dofadr = int(model.jnt_dofadr[j])
        qposadr = int(model.jnt_qposadr[j])
        limited = bool(model.jnt_limited[j])
        rng = [float(x) for x in model.jnt_range[j]]
        rest = rest0[j]
        axis_world = rest["axis_world"]
        axis_parent_body = rest["axis_parent_body"]
        axis_child_body = rest["axis_child_body"]
        entry = {
            "id": j,
            "name": name,
            "type": JOINT_NAMES.get(jtype, str(jtype)),
            "type_id": jtype,
            "body_id": body,
            "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body),
            "parent_id": parent,
            "parent": _name(model, mujoco.mjtObj.mjOBJ_BODY, parent) if parent >= 0 else None,
            "qposadr": qposadr,
            "dofadr": dofadr,
            "limited": limited,
            "range": rng,
            "axis_world": [float(x) for x in axis_world],
            "axis_parent_body": [float(x) for x in axis_parent_body],
            "axis_child_body": [float(x) for x in axis_child_body],
            "rest_rel_q0_wxyz": [float(x) for x in rest["rest_rel_q0_wxyz"]],
            "anchor_world": [float(x) for x in data.xanchor[j]],
            "damping": float(model.dof_damping[dofadr]) if jtype != int(mujoco.mjtJoint.mjJNT_FREE) else 0.0,
            "armature": float(model.dof_armature[dofadr]) if jtype != int(mujoco.mjtJoint.mjJNT_FREE) else 0.0,
            "frictionloss": float(model.dof_frictionloss[dofadr])
            if jtype != int(mujoco.mjtJoint.mjJNT_FREE)
            else 0.0,
        }
        # XML passive_joint has frictionloss=0 (nonzero in MJCF breaks training).
        # infer_policy patches 0.003 into MuJoCo at runtime. Do NOT copy that
        # into Godot: τ = 0.003·tanh(qd/0.05) on a 4 g wheel (I≈5e-7) is
        # ~0.003 N·m, larger than Jolt's tire-floor contact torque, so the
        # wheels cannot spin. Free hinge (no extra PD) matches compiled XML.
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            # freejoint: 3 translations + 4 quat in qpos, 6 dofs
            pass
        elif jtype == int(mujoco.mjtJoint.mjJNT_HINGE):
            pass
        else:
            unmapped.append(
                {
                    "item": f"joint:{name}",
                    "reason": f"unsupported joint type {entry['type']}",
                }
            )
        if jtype != int(mujoco.mjtJoint.mjJNT_FREE):
            if entry["armature"] != 0.0:
                mapping.append(
                    {
                        "joint": name,
                        "action": (
                            f"armature={entry['armature']} added as diag(A nn^T) on child "
                            "inertia, principals floored to max/10 (Jolt has no rotor slot)"
                        ),
                    }
                )
            if entry["frictionloss"] != 0.0:
                mapping.append(
                    {
                        "joint": name,
                        "action": (
                            f"frictionloss={entry['frictionloss']} applied as smooth Coulomb "
                            "τ -= μ·tanh(qd/0.05) in the Python-driven PD (Jolt hinge friction not bound)"
                        ),
                    }
                )
        joints.append(entry)

    actuators = []
    for a in range(model.nu):
        name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        jnt = int(model.actuator_trnid[a, 0])
        gain = [float(x) for x in model.actuator_gainprm[a]]
        bias = [float(x) for x in model.actuator_biasprm[a]]
        kp = float(gain[0])
        # position actuator: biasprm[1] = -kp, biasprm[2] = -kv
        kv = -float(bias[2]) if abs(bias[2]) > 0 else 0.0
        fr = [float(x) for x in model.actuator_forcerange[a]]
        cr = [float(x) for x in model.actuator_ctrlrange[a]]
        actuators.append(
            {
                "id": a,
                "name": name,
                "joint_id": jnt,
                "joint": _name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt),
                "gaintype": int(model.actuator_gaintype[a]),
                "biastype": int(model.actuator_biastype[a]),
                "kp": kp,
                "kv": kv,
                "forcerange": fr,
                "ctrlrange": cr,
                "gainprm0": gain[0],
                "biasprm": bias[:3],
            }
        )

    geoms = []
    for g in range(model.ngeom):
        gtype = int(model.geom_type[g])
        name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        body = int(model.geom_bodyid[g])
        contype = int(model.geom_contype[g])
        conaffinity = int(model.geom_conaffinity[g])
        friction = [float(x) for x in model.geom_friction[g]]
        geoms.append(
            {
                "id": g,
                "name": name,
                "type": GEOM_NAMES.get(gtype, str(gtype)),
                "type_id": gtype,
                "body_id": body,
                "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body),
                "contype": contype,
                "conaffinity": conaffinity,
                "group": int(model.geom_group[g]),
                "friction": friction,
                "size": [float(x) for x in model.geom_size[g]],
                "dataid": int(model.geom_dataid[g]),
                "rgba": [float(x) for x in model.geom_rgba[g]],
            }
        )
        if gtype in (
            int(mujoco.mjtGeom.mjGEOM_HFIELD),
            int(mujoco.mjtGeom.mjGEOM_SDF),
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        ):
            unmapped.append({"item": f"geom:{name}", "reason": f"geom type {GEOM_NAMES.get(gtype)} not mapped 1:1"})

    if model.neq > 0:
        unmapped.append({"item": "equality", "reason": f"{model.neq} equality constraints are not mapped"})

    keyframes = []
    for k in range(model.nkey):
        name = _name(model, mujoco.mjtObj.mjOBJ_KEY, k)
        qpos = [float(x) for x in model.key_qpos[k]]
        ctrl = [float(x) for x in model.key_ctrl[k]] if model.nu else []
        keyframes.append({"id": k, "name": name, "qpos": qpos, "ctrl": ctrl})

    spec = {
        "mjcf": str(mjcf.resolve()),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "nu": int(model.nu),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "dt_xml": float(model.opt.timestep),
        "gravity_mujoco": [float(x) for x in model.opt.gravity],
        "integrator": int(model.opt.integrator),
        "cone": int(model.opt.cone),
        "solver": int(model.opt.solver),
        "iterations": int(model.opt.iterations),
        "bodies": bodies,
        "joints": joints,
        "actuators": actuators,
        "geoms": geoms,
        "keyframes": keyframes,
        "unmapped": unmapped,
        "mapping": mapping,
        "collision_layer_semantics": (
            "Godot requires (A.layer & B.mask) AND (B.layer & A.mask). "
            "MuJoCo collides if (contype1 & conaffinity2) OR (contype2 & conaffinity1). "
            "Mapping layer=contype, mask=conaffinity is equivalent when each geom has "
            "contype==conaffinity (true for Microduck collision/self_collision/floor)."
        ),
    }
    return spec


def _inertial_local(data: mujoco.MjData, body_id: int, world_p: np.ndarray) -> np.ndarray:
    xipos = np.asarray(data.xipos[body_id], dtype=np.float64)
    ximat = _mat(data.ximat[body_id])
    return ximat.T @ (np.asarray(world_p, dtype=np.float64) - xipos)


def _inertial_local_vec(data: mujoco.MjData, body_id: int, world_v: np.ndarray) -> np.ndarray:
    ximat = _mat(data.ximat[body_id])
    return ximat.T @ np.asarray(world_v, dtype=np.float64)


def _godot_world_transform_from_inertial(data: mujoco.MjData, body_id: int) -> tuple[str, list[float], list[float]]:
    xipos = np.asarray(data.xipos[body_id], dtype=np.float64)
    ximat = _mat(data.ximat[body_id])
    origin = m2g_vec(xipos)
    rg = m2g_mat(ximat)
    cols = (rg[:, 0], rg[:, 1], rg[:, 2])
    return _fmt_transform(cols, origin), origin.tolist(), cols


def godot_res_path(out_dir: Path, rel: str) -> str:
    """res:// path for a file written under --out. Must not hardcode walking `microduck/`."""
    out = Path(out_dir).resolve()
    parts = list(out.parts)
    if "generated" in parts:
        i = parts.index("generated")
        return "res://" + (Path(*parts[i:]) / rel).as_posix()
    return "res://" + (Path(out.name) / rel).as_posix()


def emit_robot_tscn(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: dict,
    out_dir: Path,
    *,
    no_visual: bool,
) -> None:
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    subresources: list[str] = []
    ext_resources: list[str] = []
    nodes: list[str] = []
    mapping = spec["mapping"]

    ext_id = 1
    sub_id = 1
    mesh_ext: dict[str, str] = {}

    def add_ext(rel: str, typ: str = "ArrayMesh") -> str:
        nonlocal ext_id
        if rel in mesh_ext:
            return mesh_ext[rel]
        eid = f"mesh_{ext_id}"
        ext_id += 1
        ext_resources.append(f'[ext_resource type="{typ}" path="{godot_res_path(out_dir, rel)}" id="{eid}"]')
        mesh_ext[rel] = eid
        return eid

    def add_sub(block: str) -> str:
        nonlocal sub_id
        sid = f"sub_{sub_id}"
        sub_id += 1
        subresources.append(block.replace("__ID__", sid))
        return sid

    # Per-body collision material (max sliding friction of colliding geoms)
    body_friction: dict[int, float] = defaultdict(lambda: 1.0)
    body_layer: dict[int, int] = defaultdict(int)
    body_mask: dict[int, int] = defaultdict(int)
    for g in spec["geoms"]:
        if g["contype"] == 0 and g["conaffinity"] == 0:
            continue
        if g["type"] == "plane":
            continue
        bid = g["body_id"]
        body_friction[bid] = max(body_friction[bid], g["friction"][0])
        body_layer[bid] |= g["contype"]
        body_mask[bid] |= g["conaffinity"]

    mat_ids: dict[int, str] = {}
    for bid, mu in body_friction.items():
        if bid == 0:
            continue
        sid = add_sub(
            "\n".join(
                [
                    '[sub_resource type="PhysicsMaterial" id="__ID__"]',
                    f"friction = {fmt_f(mu)}",
                    "bounce = 0.0",
                ]
            )
        )
        mat_ids[bid] = sid

    shape_blocks: list[tuple[int, str, str, str]] = []  # body, node_name, sid, transform
    named_shape_blocks: list[tuple[str, str, str, str]] = []  # parent_node, node_name, sid, transform
    heel_bodies: dict[str, dict] = {}

    for gidx, g in enumerate(spec["geoms"]):
        bid = g["body_id"]
        gtype = g["type_id"]
        colliding = not (g["contype"] == 0 and g["conaffinity"] == 0)
        visual = (g["group"] == 2) or (g["contype"] == 0)
        if g["type"] == "plane":
            mapping.append({"geom": g["name"], "action": "skip_plane_in_robot_scene (floor is in main.tscn)"})
            continue
        if bid == 0:
            mapping.append({"geom": g["name"], "action": "skip_world_geom"})
            continue

        ximat_g = _mat(data.geom_xmat[gidx])
        xpos_g = np.asarray(data.geom_xpos[gidx], dtype=np.float64)
        # geom frame in parent inertial local
        ximat_i = _mat(data.ximat[bid])
        xipos_i = np.asarray(data.xipos[bid], dtype=np.float64)
        rel_p = ximat_i.T @ (xpos_g - xipos_i)
        rel_r = ximat_i.T @ ximat_g  # geom local → inertial local

        if colliding and gtype != int(mujoco.mjtGeom.mjGEOM_PLANE):
            sid = None
            local_tf = _fmt_transform((rel_r[:, 0], rel_r[:, 1], rel_r[:, 2]), rel_p)
            if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                r = g["size"][0]
                sid = add_sub(f'[sub_resource type="SphereShape3D" id="__ID__"]\nradius = {fmt_f(r)}')
                mapping.append({"geom": g["name"], "action": f"SphereShape3D r={r}"})
            elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                # MuJoCo half-extents → Godot full size
                size = [2 * g["size"][0], 2 * g["size"][1], 2 * g["size"][2]]
                sid = add_sub(
                    f'[sub_resource type="BoxShape3D" id="__ID__"]\nsize = Vector3({fmt_f(size[0])}, {fmt_f(size[1])}, {fmt_f(size[2])})'
                )
                mapping.append({"geom": g["name"], "action": f"BoxShape3D size={size}"})
            elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                radius = g["size"][0]
                half = g["size"][1]
                height = 2 * half + 2 * radius
                sid = add_sub(
                    f'[sub_resource type="CapsuleShape3D" id="__ID__"]\nradius = {fmt_f(radius)}\nheight = {fmt_f(height)}'
                )
                mapping.append({"geom": g["name"], "action": f"CapsuleShape3D r={radius} h={height}"})
            elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                radius = g["size"][0]
                height = 2 * g["size"][1]
                sid = add_sub(
                    f'[sub_resource type="CylinderShape3D" id="__ID__"]\nradius = {fmt_f(radius)}\nheight = {fmt_f(height)}'
                )
                mapping.append({"geom": g["name"], "action": f"CylinderShape3D r={radius} h={height}"})
            elif gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh_id = g["dataid"]
                verts, faces = _mesh_verts_faces(model, mesh_id)
                # verts in mesh/geom frame → inertial local
                verts_i = (rel_r @ verts.T).T + rel_p
                is_foot = "foot_collision" in str(g["name"])
                if is_foot and FOOT_COLLISION == "concave":
                    tris = verts_i[faces]
                    packed = ", ".join(fmt_f(c) for v in tris.reshape(-1, 3) for c in v)
                    sid_k = add_sub(
                        "\n".join(
                            [
                                '[sub_resource type="ConcavePolygonShape3D" id="__ID__"]',
                                f"data = PackedVector3Array({packed})",
                                "backface_collision = true",
                            ]
                        )
                    )
                    shape_blocks.append((bid, f"col_{g['name']}_{gidx}", sid_k, _identity_transform()))
                    mapping.append(
                        {
                            "geom": g["name"],
                            "action": (
                                f"ConcavePolygonShape3D nvert={len(verts)} ntri={len(faces)} "
                                "(MuJoCo foot mesh, triangle contacts)"
                            ),
                        }
                    )
                    sid = None
                    local_tf = _identity_transform()
                elif is_foot:
                    pieces, meta = _foot_peel_contacts(model, data, bid, gidx, verts)
                    if FOOT_COLLISION == "band_spheres":
                        cloud = _foot_sole_cloud(model, data, bid, gidx, verts)
                        spec["bodies"][bid]["sole_verts"] = [
                            [round(float(c), 5) for c in p] for p in cloud
                        ]
                        spec["bodies"][bid]["sole_radius"] = FOOT_SPHERE_RADIUS
                    eye = (
                        np.array([1.0, 0.0, 0.0]),
                        np.array([0.0, 1.0, 0.0]),
                        np.array([0.0, 0.0, 1.0]),
                    )
                    host_name = spec["bodies"][bid]["name"]
                    n_host = 0
                    n_heel = 0
                    for k, piece in enumerate(pieces):
                        if piece["kind"] == "convex":
                            packed = ", ".join(fmt_f(c) for v in piece["points"] for c in v)
                            sid_k = add_sub(
                                "\n".join(
                                    [
                                        '[sub_resource type="ConvexPolygonShape3D" id="__ID__"]',
                                        "margin = 0.0",
                                        f"points = PackedVector3Array({packed})",
                                    ]
                                )
                            )
                            tf = _identity_transform()
                        elif piece["kind"] == "capsule":
                            sid_k = add_sub(
                                "\n".join(
                                    [
                                        '[sub_resource type="CapsuleShape3D" id="__ID__"]',
                                        f"radius = {fmt_f(piece['radius'])}",
                                        f"height = {fmt_f(piece['height'])}",
                                    ]
                                )
                            )
                            tf = _fmt_transform(piece["basis"], piece["origin"])
                        else:
                            sid_k = add_sub(
                                "\n".join(
                                    [
                                        '[sub_resource type="SphereShape3D" id="__ID__"]',
                                        "margin = 0.0",
                                        f"radius = {fmt_f(piece['radius'])}",
                                    ]
                                )
                            )
                            tf = _fmt_transform(eye, piece["origin"])
                        attach = piece.get("attach", "host")
                        if attach != "host":
                            extra_name = f"{host_name}__{attach}"
                            named_shape_blocks.append(
                                (extra_name, f"col_{g['name']}_{gidx}_{k}", sid_k, tf)
                            )
                            n_heel += 1
                            if extra_name not in heel_bodies:
                                heel_bodies[extra_name] = {
                                    "host_bid": bid,
                                    "host_name": host_name,
                                    "layer": body_layer.get(bid, 1) or 1,
                                    "mask": body_mask.get(bid, 1) or 1,
                                    "mat": mat_ids.get(bid),
                                }
                        else:
                            shape_blocks.append((bid, f"col_{g['name']}_{gidx}_{k}", sid_k, tf))
                            n_host += 1
                    host_pts = next(
                        (p["points"] for p in pieces if p.get("attach") == "host" and p["kind"] == "convex"),
                        None,
                    )
                    if host_pts is not None:
                        spec["bodies"][bid]["sole_corners"] = _foot_sole_corners(
                            model, data, bid, host_pts
                        )
                    mapping.append(
                        {
                            "geom": g["name"],
                            "action": (
                                f"foot {FOOT_COLLISION} nvert={len(verts)} npts={meta.get('n_prism', 0)} "
                                f"n_strips={meta.get('n_strips', 0)} "
                                f"n_host={n_host} n_extra={n_heel} capsules={meta.get('capsule_n', 0)} "
                                f"split_heel={FOOT_SPLIT_HEEL} split_toe={FOOT_SPLIT_TOE} "
                                f"toe_kind={FOOT_TOE_KIND} "
                                f"flat={meta.get('fillet_flat')} r_h={meta.get('fillet_r_heel')} "
                                f"r_t={meta.get('fillet_r_toe')} flat_s={meta.get('flat_s_span')} "
                                f"s_span={meta.get('s_span')}"
                            ),
                        }
                    )
                    sid = None
                    local_tf = _identity_transform()
                else:
                    bname = str(g.get("body", ""))
                    mesh_name = _name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
                    if (
                        JAW_TOP_SHELL_CAPSULE
                        and bname == "jaw_soft"
                        and mesh_name == "top_head_shell"
                    ):
                        piece = _jaw_top_shell_capsule(verts_i)
                        sid = add_sub(
                            "\n".join(
                                [
                                    '[sub_resource type="CapsuleShape3D" id="__ID__"]',
                                    f"radius = {fmt_f(piece['radius'])}",
                                    f"height = {fmt_f(piece['height'])}",
                                ]
                            )
                        )
                        local_tf = _fmt_transform(piece["basis"], piece["origin"])
                        mapping.append(
                            {
                                "geom": g["name"],
                                "action": (
                                    f"jaw top_head_shell CapsuleShape3D r={piece['radius']} "
                                    f"h={piece['height']:.4f} zmin={piece['zmin']:.4f} "
                                    f"(C3 first-hit keel, nvert={len(verts)})"
                                ),
                            }
                        )
                    elif (
                        bname == "jaw_soft"
                        and mesh_name in ("jaw", "bottom_head_shell")
                        and not JAW_CHIN_HULLS
                    ):
                        sid = None
                        mapping.append(
                            {
                                "geom": g["name"],
                                "action": f"skip {mesh_name} hull (JAW_CHIN_HULLS=False diagnostic)",
                            }
                        )
                    elif mesh_name == "tire":
                        piece = _tire_cylinder(verts_i)
                        # Visual stays the tire mesh. Collision cannot be the
                        # mesh hull (pancake) or a CylinderShape3D vs the floor
                        # box: Jolt puts two rim points and the wheel scrubs
                        # (4 s xy≈0.43). A sphere at the tread radius rolls.
                        sid = add_sub(
                            "\n".join(
                                [
                                    '[sub_resource type="SphereShape3D" id="__ID__"]',
                                    f"radius = {fmt_f(piece['radius'])}",
                                ]
                            )
                        )
                        local_tf = _fmt_transform(
                            piece["basis"],
                            piece["origin"],
                        )
                        mapping.append(
                            {
                                "geom": g["name"],
                                "action": (
                                    f"tire SphereShape3D r={piece['radius']:.4f} "
                                    f"(Jolt cylinder-vs-box scrubs; visual is tire mesh)"
                                ),
                            }
                        )
                    else:
                        cap = 512 if bname == "jaw_soft" else MAX_HULL_VERTS
                        pts = _convex_points(verts_i, max_verts=cap)
                        packed = ", ".join(fmt_f(c) for v in pts for c in v)
                        sid = add_sub(
                            f'[sub_resource type="ConvexPolygonShape3D" id="__ID__"]\npoints = PackedVector3Array({packed})'
                        )
                        local_tf = _identity_transform()  # already baked into points
                        mapping.append(
                            {
                                "geom": g["name"],
                                "action": (
                                    f"ConvexPolygonShape3D nvert={len(verts)} "
                                    f"nhull={len(pts)} mesh={mesh_name} (cap {cap})"
                                ),
                            }
                        )
            else:
                mapping.append({"geom": g["name"], "action": f"UNMAPPED collision type {g['type']}"})
            if sid is not None:
                shape_blocks.append((bid, f"col_{g['name']}_{gidx}", sid, local_tf))

        if visual and not no_visual and gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = g["dataid"]
            verts, faces = _mesh_verts_faces(model, mesh_id)
            verts_i = (rel_r @ verts.T).T + rel_p
            rel = f"meshes/{g['name']}_{gidx}.obj"
            _write_obj(out_dir / rel, verts_i, faces)
            eid = add_ext(rel)
            mapping.append({"geom": g["name"], "action": f"visual OBJ {rel}"})
            # stash for node emit
            g["_visual_ext"] = eid
        elif not no_visual and (visual or g["group"] == 0) and gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            # A primitive can be both visible and colliding (the kick ball).
            # CollisionShape3D alone has no rendered surface in play mode.
            radius = g["size"][0]
            g["_visual_sub"] = add_sub(
                f'[sub_resource type="SphereMesh" id="__ID__"]\nradius = {fmt_f(radius)}\nheight = {fmt_f(2 * radius)}'
            )
            g["_visual_tf"] = _fmt_transform((rel_r[:, 0], rel_r[:, 1], rel_r[:, 2]), rel_p)
            mapping.append({"geom": g["name"], "action": f"visual SphereMesh r={radius}"})

    nodes.append('[node name="Robot" type="Node3D"]')

    # Bodies (skip world)
    for b in range(1, model.nbody):
        body = spec["bodies"][b]
        name = body["name"]
        tf, _origin, _cols = _godot_world_transform_from_inertial(data, b)
        mass = body["mass"] if body["mass"] > 1e-9 else 1e-6
        ixx, iyy, izz = body["inertia"]
        layer = body_layer.get(b, 1) or 1
        mask = body_mask.get(b, 1) or 1
        mat_line = ""
        if b in mat_ids:
            mat_line = f'physics_material_override = SubResource("{mat_ids[b]}")\n'
        nodes.append(
            "\n".join(
                [
                    f'[node name="{name}" type="RigidBody3D" parent="."]',
                    f"transform = {tf}",
                    f"mass = {fmt_f(mass)}",
                    "center_of_mass_mode = 1",
                    "center_of_mass = Vector3(0, 0, 0)",
                    f"inertia = Vector3({fmt_f(ixx)}, {fmt_f(iyy)}, {fmt_f(izz)})",
                    "can_sleep = false",
                    "linear_damp_mode = 1",
                    "linear_damp = 0.0",
                    "angular_damp_mode = 1",
                    "angular_damp = 0.0",
                    f"collision_layer = {layer}",
                    f"collision_mask = {mask}",
                    mat_line.rstrip(),
                ]
            ).rstrip()
        )
        mapping.append(
            {
                "body": name,
                "action": f"RigidBody3D mass={mass:.6g} inertia=({ixx:.6g},{iyy:.6g},{izz:.6g}) layer={layer} mask={mask}",
            }
        )

    ixx_h, iyy_h, izz_h = FOOT_HEEL_INERTIA
    for heel_name, info in heel_bodies.items():
        host_bid = int(info["host_bid"])
        tf, _origin, _cols = _godot_world_transform_from_inertial(data, host_bid)
        mat_line = ""
        if info.get("mat"):
            mat_line = f'physics_material_override = SubResource("{info["mat"]}")\n'
        nodes.append(
            "\n".join(
                [
                    f'[node name="{heel_name}" type="RigidBody3D" parent="."]',
                    f"transform = {tf}",
                    f"mass = {fmt_f(FOOT_HEEL_MASS)}",
                    "center_of_mass_mode = 1",
                    "center_of_mass = Vector3(0, 0, 0)",
                    f"inertia = Vector3({fmt_f(ixx_h)}, {fmt_f(iyy_h)}, {fmt_f(izz_h)})",
                    "can_sleep = false",
                    "linear_damp_mode = 1",
                    f"linear_damp = {fmt_f(FOOT_HEEL_LINEAR_DAMP)}",
                    "angular_damp_mode = 1",
                    "angular_damp = 0.0",
                    f"collision_layer = {info['layer']}",
                    f"collision_mask = {info['mask']}",
                    mat_line.rstrip(),
                ]
            ).rstrip()
        )
        mapping.append(
            {
                "body": heel_name,
                "action": (
                    f"RigidBody3D heel sibling of {info['host_name']} "
                    f"mass={FOOT_HEEL_MASS} "
                    + (
                        "(6DOF linear springs, physics_server.gd)"
                        if FOOT_SIBLING_SPRING
                        else "(coplanar split, FixedConstraint weld)"
                    )
                ),
            }
        )

    # Collision shapes
    for bid, node_name, sid, tf in shape_blocks:
        bname = spec["bodies"][bid]["name"]
        safe = node_name.replace("/", "_")
        nodes.append(
            "\n".join(
                [
                    f'[node name="{safe}" type="CollisionShape3D" parent="{bname}"]',
                    f"transform = {tf}",
                    f'shape = SubResource("{sid}")',
                ]
            )
        )
    for parent_name, node_name, sid, tf in named_shape_blocks:
        safe = node_name.replace("/", "_")
        nodes.append(
            "\n".join(
                [
                    f'[node name="{safe}" type="CollisionShape3D" parent="{parent_name}"]',
                    f"transform = {tf}",
                    f'shape = SubResource("{sid}")',
                ]
            )
        )

    # Visual meshes
    if not no_visual:
        for gidx, g in enumerate(spec["geoms"]):
            eid = g.get("_visual_ext")
            sid = g.get("_visual_sub")
            if not eid and not sid:
                continue
            bname = g["body"]
            rgba = g["rgba"]
            # StandardMaterial3D
            mat = add_sub(
                "\n".join(
                    [
                        '[sub_resource type="StandardMaterial3D" id="__ID__"]',
                        f"albedo_color = Color({fmt_f(rgba[0])}, {fmt_f(rgba[1])}, {fmt_f(rgba[2])}, {fmt_f(rgba[3])})",
                    ]
                )
            )
            nodes.append(
                "\n".join(
                    [
                        f'[node name="vis_{g["name"]}_{gidx}" type="MeshInstance3D" parent="{bname}"]',
                        f'transform = {g.get("_visual_tf", _identity_transform())}',
                        f'mesh = ExtResource("{eid}")' if eid else f'mesh = SubResource("{sid}")',
                        f'surface_material_override/0 = SubResource("{mat}")',
                    ]
                )
            )

    # Hinge joints as children of parent body
    for j in spec["joints"]:
        if j["type"] != "hinge":
            mapping.append({"joint": j["name"], "action": f"skip type={j['type']}"})
            continue
        parent_id = j["parent_id"]
        child_id = j["body_id"]
        child = j["body"]
        if parent_id == 0:
            parent_name = "WorldAnchor"
            # Create a static world anchor once
            mapping.append({"joint": j["name"], "action": "hinge to WorldAnchor (parent=world)"})
        else:
            parent_name = j["parent"]

        anchor_i = _inertial_local(data, parent_id if parent_id != 0 else child_id, j["anchor_world"])
        axis_i = _inertial_local_vec(data, parent_id if parent_id != 0 else child_id, j["axis_world"])
        if parent_id == 0:
            # express in world inertial = world (identity) then Godot world... 
            # WorldAnchor will sit at identity; use Godot world coords for the joint transform.
            origin = m2g_vec(np.asarray(j["anchor_world"]))
            axis_g = m2g_vec(np.asarray(j["axis_world"]))
            x, y, z = basis_from_z(axis_g)
            tf = _fmt_transform((x, y, z), origin)
            parent_path = "WorldAnchor"
            # node paths from joint child of Robot
            node_a = "WorldAnchor"
            node_b = child
            joint_parent = "."
        else:
            x, y, z = basis_from_z(axis_i)
            tf = _fmt_transform((x, y, z), anchor_i)
            node_a = parent_name
            node_b = child
            joint_parent = parent_name

        lo, hi = j["range"] if j["limited"] else (-3.14159265, 3.14159265)
        # The generated body transforms are at data.qpos, not necessarily q=0.
        # Godot's clockwise hinge angle is the negative of MuJoCo's joint q.
        q_reference = float(data.qpos[int(j["qposadr"])])
        lo, hi = q_reference - hi, q_reference - lo
        limit_on = "true" if j["limited"] else "false"
        jname = f"joint_{j['name']}"
        if joint_parent == ".":
            nodes.append(
                "\n".join(
                    [
                        f'[node name="{jname}" type="HingeJoint3D" parent="."]',
                        f"transform = {tf}",
                        f'node_a = NodePath("../{node_a}")',
                        f'node_b = NodePath("../{node_b}")',
                        f"angular_limit/enable = {limit_on}",
                        f"angular_limit/lower = {fmt_f(lo)}",
                        f"angular_limit/upper = {fmt_f(hi)}",
                    ]
                )
            )
        else:
            na = ".."
            nb = f"../../{node_b}"
            nodes.append(
                "\n".join(
                    [
                        f'[node name="{jname}" type="HingeJoint3D" parent="{joint_parent}"]',
                        f"transform = {tf}",
                        f'node_a = NodePath("{na}")',
                        f'node_b = NodePath("{nb}")',
                        f"angular_limit/enable = {limit_on}",
                        f"angular_limit/lower = {fmt_f(lo)}",
                        f"angular_limit/upper = {fmt_f(hi)}",
                    ]
                )
            )
        mapping.append(
            {
                "joint": j["name"],
                "action": f"HingeJoint3D {node_a}->{node_b} limit=[{lo:.4g},{hi:.4g}] damping={j['damping']}",
            }
        )

    for heel_name, info in heel_bodies.items():
        host_name = info["host_name"]
        if FOOT_SIBLING_SPRING:
            mapping.append(
                {
                    "joint": f"spring_{heel_name}",
                    "action": (
                        f"Generic6DOFJoint3D linear springs {host_name}->{heel_name} "
                        "(physics_server.gd; k=1e5 to keep 3 mm stagger through impact)"
                    ),
                }
            )
            continue
        host_bid = int(info["host_bid"])
        tf, _origin, _cols = _godot_world_transform_from_inertial(data, host_bid)
        jname = f"weld_{heel_name}"
        nodes.append(
            "\n".join(
                [
                    f'[node name="{jname}" type="HingeJoint3D" parent="."]',
                    f"transform = {tf}",
                    f'node_a = NodePath("../{host_name}")',
                    f'node_b = NodePath("../{heel_name}")',
                    "angular_limit/enable = true",
                    "angular_limit/lower = 0.0",
                    "angular_limit/upper = 0.0",
                ]
            )
        )
        mapping.append(
            {
                "joint": jname,
                "action": f"HingeJoint3D 0-limit weld {host_name}->{heel_name} (Jolt FixedConstraint)",
            }
        )

    # WorldAnchor if needed
    if any(j["parent_id"] == 0 and j["type"] == "hinge" for j in spec["joints"]):
        nodes.insert(
            1,
            "\n".join(
                [
                    '[node name="WorldAnchor" type="StaticBody3D" parent="."]',
                    "collision_layer = 0",
                    "collision_mask = 0",
                ]
            ),
        )

    n_ext = len(ext_resources)
    n_sub = len(subresources)
    load_steps = n_ext + n_sub + 1
    parts = [f"[gd_scene load_steps={load_steps} format=3]\n"]
    if ext_resources:
        parts.append("\n".join(ext_resources) + "\n")
    if subresources:
        parts.append("\n\n".join(subresources) + "\n")
    parts.append("\n\n".join(nodes) + "\n")
    (out_dir / "robot.tscn").write_text("\n".join(parts), encoding="utf-8")


def NodePath_for(joint_parent: str, target: str) -> str:
    if joint_parent == ".":
        return target
    return target


def write_report(spec: dict, path: Path) -> None:
    lines = ["# MJCF → Godot mapping report", ""]
    lines.append(f"- MJCF: `{spec['mjcf']}`")
    lines.append(f"- bodies: {spec['nbody']}  joints: {spec['njnt']}  actuators: {spec['nu']}")
    lines.append(f"- gravity (MuJoCo): {spec['gravity_mujoco']}")
    lines.append(f"- XML timestep: {spec['dt_xml']}")
    lines.append("")
    lines.append("## Mapping")
    for m in spec["mapping"]:
        lines.append(f"- {m}")
    lines.append("")
    lines.append("## Unmapped (recorded, not faked)")
    if not spec["unmapped"]:
        lines.append("- (none)")
    else:
        # unique by item
        seen = set()
        for u in spec["unmapped"]:
            key = u["item"]
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- `{u['item']}`: {u['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert(mjcf: Path, out_dir: Path, *, no_visual: bool = False) -> dict:
    mjcf = mjcf.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model, data = compile_model(mjcf)
    spec = build_spec(model, data, mjcf)
    emit_robot_tscn(model, data, spec, out_dir, no_visual=no_visual)
    for g in spec["geoms"]:
        g.pop("_visual_ext", None)
        g.pop("_visual_sub", None)
        g.pop("_visual_tf", None)
    spec_path = out_dir / "robot_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    write_report(spec, out_dir / "MAPPING_REPORT.md")
    return spec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MJCF → Godot 4 Jolt scene")
    p.add_argument("--mjcf", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-visual", action="store_true")
    args = p.parse_args(argv)
    spec = convert(args.mjcf, args.out, no_visual=args.no_visual)
    n_unmap = len({u["item"] for u in spec["unmapped"]})
    print(f"wrote {args.out / 'robot.tscn'}")
    print(f"wrote {args.out / 'robot_spec.json'}")
    print(f"bodies={spec['nbody']} joints={spec['njnt']} actuators={spec['nu']} unmapped_items={n_unmap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
