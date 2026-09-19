import math
import json
from pathlib import Path

import numpy as np

from agenticrobot_bridge.robot_manifest import (
    compile_robot_manifest,
    mujoco_to_tuanjie_axial_vector,
    mujoco_to_tuanjie_inertia_diagonal,
    mujoco_to_tuanjie_quaternion,
    mujoco_to_tuanjie_vector,
    write_robot_manifest,
)


ROOT = Path(__file__).parents[1]
MODEL_ROOT = (
    ROOT
    / ".cache"
    / "upstream"
    / "microduck_rl"
    / "src"
    / "mjlab_microduck"
    / "robot"
    / "microduck"
)


def test_mujoco_vector_basis_maps_x_y_z_to_negative_y_z_x() -> None:
    assert mujoco_to_tuanjie_vector((1.0, 2.0, 3.0)) == (-2.0, 3.0, 1.0)


def test_axial_joint_axis_preserves_positive_rotation_across_reflected_basis() -> None:
    basis = np.asarray(((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
    mujoco_positive_quarter_turn_about_z = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    conjugated_rotation = basis @ mujoco_positive_quarter_turn_about_z @ basis.T

    mapped_axis = np.asarray(mujoco_to_tuanjie_axial_vector((0.0, 0.0, 1.0)))
    x, y, z = mapped_axis
    cross_matrix = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    mapped_positive_quarter_turn = (
        np.eye(3) + cross_matrix + cross_matrix @ cross_matrix
    )

    assert mapped_axis.tolist() == [0.0, -1.0, -0.0]
    np.testing.assert_allclose(mapped_positive_quarter_turn, conjugated_rotation, atol=1e-12)


def test_quaternion_mapping_uses_the_same_axial_rotation_convention() -> None:
    half_angle = math.pi / 4.0

    mapped = mujoco_to_tuanjie_quaternion(
        (math.cos(half_angle), 0.0, 0.0, math.sin(half_angle))
    )

    np.testing.assert_allclose(
        mapped,
        (math.cos(half_angle), 0.0, -math.sin(half_angle), -0.0),
        atol=1e-12,
    )


def test_inertia_diagonal_is_permuted_with_the_reflected_coordinate_basis() -> None:
    # The orientation quaternion is conjugated by B, so the diagonal tensor's
    # coordinate labels must be conjugated by the same signed permutation too.
    # Leaving (Ix, Iy, Iz) untouched rotates the principal moments onto the
    # wrong physical axes (most visibly on the roller wheels).
    assert mujoco_to_tuanjie_inertia_diagonal((1.0, 2.0, 3.0)) == (2.0, 3.0, 1.0)


def test_roller_wheel_keeps_its_large_axial_moment_on_the_mapped_hinge_axis() -> None:
    manifest = compile_robot_manifest(MODEL_ROOT / "robot_allcollisions_rollers.xml")
    tire = next(body for body in manifest.bodies if body.name == "tire")

    np.testing.assert_allclose(
        tire.inertia_diagonal,
        (2.8556e-7, 2.8556e-7, 5.3687e-7),
        rtol=1e-6,
    )

def test_legged_manifest_uses_compiled_model_values() -> None:
    manifest = compile_robot_manifest(MODEL_ROOT / "robot_allcollisions.xml")

    assert manifest.source_xml.name == "robot_allcollisions.xml"
    assert manifest.body_count == 15
    assert manifest.joint_count == 15
    assert manifest.servo_count == 14
    assert manifest.passive_joint_names == ()
    assert all(body.mass > 0 and math.isfinite(body.mass) for body in manifest.bodies)
    assert all(
        all(inertia > 0 and math.isfinite(inertia) for inertia in body.inertia_diagonal)
        for body in manifest.bodies
    )


def test_roller_manifest_preserves_four_passive_wheels() -> None:
    manifest = compile_robot_manifest(MODEL_ROOT / "robot_allcollisions_rollers.xml")

    assert manifest.variant == "roller"
    assert manifest.body_count == 19
    assert manifest.joint_count == 19
    assert manifest.servo_count == 14
    assert manifest.passive_joint_names == (
        "passive_LF_wheel",
        "passive_LR_wheel",
        "passive_RF_wheel",
        "passive_RR_wheel",
    )
    assert all(body.mass > 0 and math.isfinite(body.mass) for body in manifest.bodies)
    assert all(
        all(inertia > 0 and math.isfinite(inertia) for inertia in body.inertia_diagonal)
        for body in manifest.bodies
    )


def test_manifest_exports_hierarchy_transforms_and_joint_contract() -> None:
    upstream_root = ROOT / ".cache" / "upstream"
    manifest = compile_robot_manifest(
        MODEL_ROOT / "robot_allcollisions.xml", source_root=upstream_root
    )

    assert manifest.schema_version == 1
    assert manifest.source_relative_path == (
        "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
    )
    assert manifest.source_model_name == "microduck"
    assert len(manifest.source_sha256) == 64
    assert math.isclose(manifest.timestep, 0.002)
    assert manifest.gravity == (-0.0, -9.81, 0.0)

    bodies = {body.name: body for body in manifest.bodies}
    assert bodies["trunk_base"].parent_name is None
    assert bodies["trunk_base"].local_position == (-0.0, 0.12, 0.0)
    assert bodies["yaw2roll"].parent_name == "trunk_base"
    assert bodies["yaw2roll"].local_position == (-0.0175, -0.005, 0.006)
    assert len(bodies["trunk_base"].local_rotation_wxyz) == 4
    assert len(bodies["trunk_base"].center_of_mass) == 3
    assert len(bodies["trunk_base"].inertia_rotation_wxyz) == 4

    joints = {joint.name: joint for joint in manifest.joints}
    root = joints["trunk_base_freejoint"]
    assert root.body_name == "trunk_base"
    assert root.type == "free"
    assert root.qpos_address == 0
    assert root.dof_address == 0
    hip = joints["left_hip_yaw"]
    assert hip.body_name == "yaw2roll"
    assert hip.type == "hinge"
    assert hip.axis == (0.0, -1.0, -0.0)
    assert hip.limited is True
    assert hip.range_rad == (-0.4363323129985824, 0.5235987755982988)
    assert hip.passive is False

    payload = manifest.to_dict()
    assert payload["schemaVersion"] == 1
    assert payload["source"]["relativePath"] == manifest.source_relative_path
    assert json.loads(json.dumps(payload))["joints"][1]["name"] == "left_hip_yaw"


def test_manifest_exposes_mesh_collision_and_sensor_sources() -> None:
    upstream_root = ROOT / ".cache" / "upstream"
    legged = compile_robot_manifest(
        MODEL_ROOT / "robot_allcollisions.xml", source_root=upstream_root
    )
    roller = compile_robot_manifest(
        MODEL_ROOT / "robot_allcollisions_rollers.xml", source_root=upstream_root
    )

    assert len(legged.geoms) == 81
    assert len(legged.meshes) == 38
    assert len(legged.sensors) == 6
    assert sum(geom.collidable for geom in legged.geoms) == 11
    assert len({geom.name for geom in legged.geoms}) == len(legged.geoms)
    assert all(geom.type == "mesh" and geom.mesh_name for geom in legged.geoms)
    assert all(mesh.source_file.endswith(".stl") for mesh in legged.meshes)

    assert len(roller.geoms) == 89
    assert len(roller.meshes) == 37
    assert sum(geom.collidable for geom in roller.geoms) == 13
    assert roller.keyframes == ()

    payload = legged.to_dict()
    assert len(payload["geoms"]) == 81
    assert len(payload["meshes"]) == 38
    assert len(payload["sensors"]) == 6


def test_manifest_preserves_mujoco_geom_friction_vectors() -> None:
    manifest = compile_robot_manifest(
        MODEL_ROOT / "scene_ball.xml",
        source_root=ROOT / ".cache" / "upstream",
    )

    geoms = {geom.name: geom for geom in manifest.geoms}
    assert geoms["floor"].friction == (1.0, 0.005, 0.0001)
    assert geoms["ball_geom"].friction == (0.5, 0.005, 0.0001)
    payload = manifest.to_dict()
    payload_geoms = {geom["name"]: geom for geom in payload["geoms"]}
    assert payload_geoms["floor"]["friction"] == [1.0, 0.005, 0.0001]
    assert payload_geoms["ball_geom"]["friction"] == [0.5, 0.005, 0.0001]


def test_manifest_json_is_deterministic(tmp_path: Path) -> None:
    manifest = compile_robot_manifest(
        MODEL_ROOT / "robot_allcollisions_rollers.xml",
        source_root=ROOT / ".cache" / "upstream",
    )

    first = write_robot_manifest(manifest, tmp_path / "first.json")
    second = write_robot_manifest(manifest, tmp_path / "second.json")

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    assert json.loads(first.read_text(encoding="utf-8")) == manifest.to_dict()
