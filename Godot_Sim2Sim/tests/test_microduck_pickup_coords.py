"""Cross-engine beak geometry contract from the compiled MuJoCo model.

Godot's RigidBody3D uses MuJoCo inertial *local* coordinates. Only world
vectors rotate from Z-up into Y-up; converting the local hinge axis again
would turn mouth pitch into a roll.
"""
from pathlib import Path
import re

import numpy as np

from sim2sim.coords import R_M2G, g2m_vec, m2g_vec
from sim2sim.pickup_env import PickupEnv
from sim2sim.pickup_scene import generate


def _godot_vector(source: str, name: str) -> np.ndarray:
    match = re.search(rf"const {name} := Vector3\(([^)]*)\)", source)
    assert match, name
    return np.array([float(x.strip()) for x in match.group(1).split(",")])


def test_compiled_beak_matches_godot_inertial_frame(tmp_path: Path) -> None:
    output = tmp_path / "pickup"
    generate(output, "bottle_full", "adhesion")
    env = PickupEnv(output / "scene_pickable.xml")
    env.reset(7, x=.07, lateral=0.)
    model, data = env.model, env.data
    head = model.body("jaw_soft").id
    hinge = model.joint("beak_pitch").id
    inertial_world = data.ximat[head].reshape(3, 3)
    pivot_local = inertial_world.T @ (data.xanchor[hinge]-data.xipos[head])
    axis_local = inertial_world.T @ data.xaxis[hinge]
    tip_local = inertial_world.T @ (
        data.site_xpos[model.site("mouth_tip").id]-data.xipos[head])
    source = Path("godot/hub/microduck_beak.gd").read_text()
    np.testing.assert_allclose(pivot_local, _godot_vector(source, "PIVOT_LOCAL"), atol=2e-6)
    np.testing.assert_allclose(tip_local, _godot_vector(source, "TIP_LOCAL"), atol=2e-6)
    np.testing.assert_allclose(axis_local, [0., 1., 0.], atol=2e-6)
    assert "pivot.rotation.y" in source
    # A proper rotation maps world up but does not change inertial local axes.
    np.testing.assert_allclose(R_M2G.T @ R_M2G, np.eye(3), atol=1e-12)
    assert np.linalg.det(R_M2G) > .999999
    np.testing.assert_allclose(m2g_vec([0., 0., 1.]), [0., 1., 0.])
    np.testing.assert_allclose(g2m_vec(m2g_vec(data.xaxis[hinge])), data.xaxis[hinge])
