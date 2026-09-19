"""Neutral robot manifest generated from a compiled MuJoCo model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import mujoco


@dataclass(frozen=True, slots=True)
class BodyManifest:
    """One non-world body from a compiled MuJoCo model."""

    name: str
    parent_name: str | None
    local_position: tuple[float, float, float]
    local_rotation_wxyz: tuple[float, float, float, float]
    mass: float
    center_of_mass: tuple[float, float, float]
    inertia_diagonal: tuple[float, float, float]
    inertia_rotation_wxyz: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parentName": self.parent_name,
            "localPosition": list(self.local_position),
            "localRotationWxyz": list(self.local_rotation_wxyz),
            "mass": self.mass,
            "centerOfMass": list(self.center_of_mass),
            "inertiaTensor": list(self.inertia_diagonal),
            "inertiaRotationWxyz": list(self.inertia_rotation_wxyz),
        }


@dataclass(frozen=True, slots=True)
class JointManifest:
    """One robot joint, including its unactuated root joint."""

    name: str
    body_name: str
    type: str
    local_anchor: tuple[float, float, float]
    axis: tuple[float, float, float]
    limited: bool
    range_rad: tuple[float, float]
    damping: tuple[float, ...]
    friction_loss: tuple[float, ...]
    armature: tuple[float, ...]
    qpos_address: int
    dof_address: int
    passive: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bodyName": self.body_name,
            "type": self.type,
            "localAnchor": list(self.local_anchor),
            "axis": list(self.axis),
            "limited": self.limited,
            "rangeRad": list(self.range_rad),
            "damping": list(self.damping),
            "frictionLoss": list(self.friction_loss),
            "armature": list(self.armature),
            "qposAddress": self.qpos_address,
            "dofAddress": self.dof_address,
            "passive": self.passive,
        }


@dataclass(frozen=True, slots=True)
class ServoManifest:
    """One MuJoCo actuator and the joint it drives."""

    name: str
    joint_name: str
    index: int
    ctrl_limited: bool
    ctrl_range: tuple[float, float]
    force_limited: bool
    force_range: tuple[float, float]
    gain_prm: tuple[float, ...]
    bias_prm: tuple[float, ...]
    gear: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "jointName": self.joint_name,
            "index": self.index,
            "ctrlLimited": self.ctrl_limited,
            "ctrlRange": list(self.ctrl_range),
            "forceLimited": self.force_limited,
            "forceRange": list(self.force_range),
            "gainPrm": list(self.gain_prm),
            "biasPrm": list(self.bias_prm),
            "gear": list(self.gear),
        }


@dataclass(frozen=True, slots=True)
class GeomManifest:
    """One visual or collision geometry attached to a robot body."""

    name: str
    body_name: str
    type: str
    local_position: tuple[float, float, float]
    local_rotation_wxyz: tuple[float, float, float, float]
    size: tuple[float, float, float]
    mesh_name: str | None
    rgba: tuple[float, float, float, float]
    friction: tuple[float, float, float]
    contype: int
    conaffinity: int
    group: int
    collidable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bodyName": self.body_name,
            "type": self.type,
            "localPosition": list(self.local_position),
            "localRotationWxyz": list(self.local_rotation_wxyz),
            "size": list(self.size),
            "meshName": self.mesh_name,
            "rgba": list(self.rgba),
            "friction": list(self.friction),
            "contype": self.contype,
            "conaffinity": self.conaffinity,
            "group": self.group,
            "collidable": self.collidable,
        }


@dataclass(frozen=True, slots=True)
class MeshManifest:
    """One compiled mesh asset and its source file."""

    name: str
    source_file: str
    scale: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sourceFile": self.source_file,
            "scale": list(self.scale),
        }


@dataclass(frozen=True, slots=True)
class SensorManifest:
    """One named sensor from the compiled model."""

    name: str
    type: str
    dimension: int
    object_type: str
    object_name: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "dim": self.dimension,
            "objectType": self.object_type,
            "objectName": self.object_name,
        }


@dataclass(frozen=True, slots=True)
class RobotManifest:
    """Engine-neutral facts extracted from a compiled MuJoCo model."""

    source_xml: Path
    source_relative_path: str
    source_sha256: str
    source_model_name: str
    variant: str
    timestep: float
    gravity: tuple[float, float, float]
    bodies: tuple[BodyManifest, ...]
    joints: tuple[JointManifest, ...]
    servos: tuple[ServoManifest, ...]
    passive_joint_names: tuple[str, ...]
    geoms: tuple[GeomManifest, ...]
    meshes: tuple[MeshManifest, ...]
    sensors: tuple[SensorManifest, ...]
    keyframes: tuple[dict[str, Any], ...]

    @property
    def body_count(self) -> int:
        return len(self.bodies)

    @property
    def joint_count(self) -> int:
        return len(self.joints)

    @property
    def servo_count(self) -> int:
        return len(self.servos)

    @property
    def schema_version(self) -> int:
        return 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "source": {
                "relativePath": self.source_relative_path,
                "sha256": self.source_sha256,
                "modelName": self.source_model_name,
            },
            "variant": self.variant,
            "timestep": self.timestep,
            "gravity": list(self.gravity),
            "bodies": [body.to_dict() for body in self.bodies],
            "joints": [joint.to_dict() for joint in self.joints],
            "servos": [servo.to_dict() for servo in self.servos],
            "passiveJointNames": list(self.passive_joint_names),
            "geoms": [geom.to_dict() for geom in self.geoms],
            "meshes": [mesh.to_dict() for mesh in self.meshes],
            "sensors": [sensor.to_dict() for sensor in self.sensors],
            "keyframes": list(self.keyframes),
        }


def mujoco_to_tuanjie_vector(vector: Sequence[float]) -> tuple[float, float, float]:
    """Map a MuJoCo xyz vector into the Tuanjie/Unity coordinate basis."""

    if len(vector) != 3:
        raise ValueError(f"Expected a three-component vector, got {len(vector)}")
    x, y, z = vector
    return (-float(y), float(z), float(x))


def mujoco_to_tuanjie_axial_vector(
    vector: Sequence[float],
) -> tuple[float, float, float]:
    """Map an axial vector while preserving positive rotation under reflection.

    The MuJoCo-to-Tuanjie basis has determinant -1. Axial vectors therefore
    transform as ``det(B) * B * v``, unlike positions and linear directions.
    """

    if len(vector) != 3:
        raise ValueError(f"Expected a three-component vector, got {len(vector)}")
    x, y, z = vector
    return (float(y), -float(z), -float(x))


def mujoco_to_tuanjie_inertia_diagonal(
    diagonal: Sequence[float],
) -> tuple[float, float, float]:
    """Map principal-moment labels through the Tuanjie coordinate basis.

    The inertia-frame quaternion is conjugated by the signed permutation
    ``B: (x, y, z) -> (-y, z, x)``.  Its diagonal matrix must be conjugated by
    the same basis: ``B diag(Ix, Iy, Iz) B.T = diag(Iy, Iz, Ix)``.  Signs drop
    out because inertia is a second-order tensor.
    """

    if len(diagonal) != 3:
        raise ValueError(f"Expected three principal moments, got {len(diagonal)}")
    x, y, z = diagonal
    return (float(y), float(z), float(x))


def mujoco_to_tuanjie_quaternion(
    quaternion_wxyz: Sequence[float],
) -> tuple[float, float, float, float]:
    """Map a MuJoCo wxyz rotation through the handedness-changing basis."""

    if len(quaternion_wxyz) != 4:
        raise ValueError(f"Expected a four-component quaternion, got {len(quaternion_wxyz)}")
    w, x, y, z = quaternion_wxyz
    return (float(w), float(y), -float(z), -float(x))


def compile_robot_manifest(
    xml_path: str | Path,
    *,
    source_root: str | Path | None = None,
) -> RobotManifest:
    """Compile an MJCF file and export its resolved robot structure."""

    source_xml = Path(xml_path).resolve()
    model = mujoco.MjModel.from_xml_path(str(source_xml))
    source_root_path = Path(source_root).resolve() if source_root is not None else None

    if source_root_path is None:
        source_relative_path = source_xml.name
    else:
        source_relative_path = source_xml.relative_to(source_root_path).as_posix()

    servos = tuple(
        ServoManifest(
            name=_object_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
            joint_name=_object_name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(model.actuator_trnid[actuator_id, 0]),
            ),
            index=actuator_id,
            ctrl_limited=bool(model.actuator_ctrllimited[actuator_id]),
            ctrl_range=tuple(float(value) for value in model.actuator_ctrlrange[actuator_id]),
            force_limited=bool(model.actuator_forcelimited[actuator_id]),
            force_range=tuple(float(value) for value in model.actuator_forcerange[actuator_id]),
            gain_prm=tuple(float(value) for value in model.actuator_gainprm[actuator_id]),
            bias_prm=tuple(float(value) for value in model.actuator_biasprm[actuator_id]),
            gear=tuple(float(value) for value in model.actuator_gear[actuator_id]),
        )
        for actuator_id in range(model.nu)
    )
    servo_joint_names = {servo.joint_name for servo in servos}

    bodies = tuple(
        BodyManifest(
            name=_object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
            parent_name=(
                None
                if int(model.body_parentid[body_id]) == 0
                else _object_name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[body_id])
                )
            ),
            local_position=mujoco_to_tuanjie_vector(model.body_pos[body_id]),
            local_rotation_wxyz=mujoco_to_tuanjie_quaternion(model.body_quat[body_id]),
            mass=float(model.body_mass[body_id]),
            center_of_mass=mujoco_to_tuanjie_vector(model.body_ipos[body_id]),
            inertia_diagonal=mujoco_to_tuanjie_inertia_diagonal(
                model.body_inertia[body_id]
            ),
            inertia_rotation_wxyz=mujoco_to_tuanjie_quaternion(model.body_iquat[body_id]),
        )
        for body_id in range(1, model.nbody)
    )
    joints = tuple(
        _joint_manifest(model, joint_id, servo_joint_names)
        for joint_id in range(model.njnt)
    )
    passive_joint_names = tuple(
        joint.name for joint in joints if joint.passive
    )
    meshes = tuple(
        _mesh_manifest(model, mesh_id, source_xml, source_root_path)
        for mesh_id in range(model.nmesh)
    )
    geoms = tuple(_geom_manifest(model, geom_id) for geom_id in range(model.ngeom))
    sensors = tuple(_sensor_manifest(model, sensor_id) for sensor_id in range(model.nsensor))

    return RobotManifest(
        source_xml=source_xml,
        source_relative_path=source_relative_path,
        source_sha256=hashlib.sha256(source_xml.read_bytes()).hexdigest(),
        source_model_name=ElementTree.parse(source_xml).getroot().attrib.get(
            "model", source_xml.stem
        ),
        variant="roller" if passive_joint_names else "legged",
        timestep=float(model.opt.timestep),
        gravity=mujoco_to_tuanjie_vector(model.opt.gravity),
        bodies=bodies,
        joints=joints,
        servos=servos,
        passive_joint_names=passive_joint_names,
        geoms=geoms,
        meshes=meshes,
        sensors=sensors,
        keyframes=(),
    )


def write_robot_manifest(manifest: RobotManifest, destination: str | Path) -> Path:
    """Write a deterministic UTF-8 JSON representation of a manifest."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(
        manifest.to_dict(), sort_keys=True, indent=2, ensure_ascii=False
    ) + "\n"
    destination_path.write_text(document, encoding="utf-8", newline="\n")
    return destination_path


def _mesh_manifest(
    model: mujoco.MjModel,
    mesh_id: int,
    source_xml: Path,
    source_root: Path | None,
) -> MeshManifest:
    source_name = _path_string(model.paths, int(model.mesh_pathadr[mesh_id]))
    candidates = (
        source_xml.parent / "assets" / source_name,
        source_xml.parent / source_name,
    )
    source_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if source_root is not None:
        try:
            source_file = source_path.relative_to(source_root).as_posix()
        except ValueError:
            source_file = source_path.as_posix()
    else:
        source_file = source_path.relative_to(source_xml.parent).as_posix()
    return MeshManifest(
        name=_object_name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id),
        source_file=source_file,
        scale=mujoco_to_tuanjie_extent(model.mesh_scale[mesh_id]),
    )


def _geom_manifest(model: mujoco.MjModel, geom_id: int) -> GeomManifest:
    body_id = int(model.geom_bodyid[geom_id])
    body_name = _object_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    explicit_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    data_id = int(model.geom_dataid[geom_id])
    mesh_name = (
        _object_name(model, mujoco.mjtObj.mjOBJ_MESH, data_id)
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH and data_id >= 0
        else None
    )
    contype = int(model.geom_contype[geom_id])
    conaffinity = int(model.geom_conaffinity[geom_id])
    geom_type = mujoco.mjtGeom(int(model.geom_type[geom_id]))
    return GeomManifest(
        name=explicit_name or f"__geom_{geom_id:03d}_{body_name}",
        body_name=body_name,
        type=geom_type.name.removeprefix("mjGEOM_").lower(),
        local_position=mujoco_to_tuanjie_vector(model.geom_pos[geom_id]),
        local_rotation_wxyz=mujoco_to_tuanjie_quaternion(model.geom_quat[geom_id]),
        size=mujoco_to_tuanjie_extent(model.geom_size[geom_id]),
        mesh_name=mesh_name,
        rgba=tuple(float(value) for value in model.geom_rgba[geom_id]),
        friction=tuple(float(value) for value in model.geom_friction[geom_id]),
        contype=contype,
        conaffinity=conaffinity,
        group=int(model.geom_group[geom_id]),
        collidable=bool(contype or conaffinity),
    )


def _sensor_manifest(model: mujoco.MjModel, sensor_id: int) -> SensorManifest:
    object_type_value = int(model.sensor_objtype[sensor_id])
    object_id = int(model.sensor_objid[sensor_id])
    object_type = mujoco.mjtObj(object_type_value)
    return SensorManifest(
        name=_object_name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id),
        type=mujoco.mjtSensor(int(model.sensor_type[sensor_id])).name.removeprefix(
            "mjSENS_"
        ).lower(),
        dimension=int(model.sensor_dim[sensor_id]),
        object_type=object_type.name.removeprefix("mjOBJ_").lower(),
        object_name=(
            _object_name(model, object_type, object_id) if object_id >= 0 else None
        ),
    )


def mujoco_to_tuanjie_extent(extent: Sequence[float]) -> tuple[float, float, float]:
    """Reorder a non-negative xyz extent without applying reflection signs."""

    if len(extent) != 3:
        raise ValueError(f"Expected a three-component extent, got {len(extent)}")
    x, y, z = extent
    return (abs(float(y)), abs(float(z)), abs(float(x)))


def _path_string(paths: bytes, address: int) -> str:
    return paths[address:].split(b"\0", 1)[0].decode("utf-8")


def _joint_manifest(
    model: mujoco.MjModel,
    joint_id: int,
    servo_joint_names: set[str],
) -> JointManifest:
    name = _object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    joint_type = mujoco.mjtJoint(int(model.jnt_type[joint_id]))
    type_name = joint_type.name.removeprefix("mjJNT_").lower()
    dof_address = int(model.jnt_dofadr[joint_id])
    dof_count = {
        mujoco.mjtJoint.mjJNT_FREE: 6,
        mujoco.mjtJoint.mjJNT_BALL: 3,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }[joint_type]
    dof_slice = slice(dof_address, dof_address + dof_count)
    return JointManifest(
        name=name,
        body_name=_object_name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[joint_id])
        ),
        type=type_name,
        local_anchor=mujoco_to_tuanjie_vector(model.jnt_pos[joint_id]),
        axis=mujoco_to_tuanjie_axial_vector(model.jnt_axis[joint_id]),
        limited=bool(model.jnt_limited[joint_id]),
        range_rad=tuple(float(value) for value in model.jnt_range[joint_id]),
        damping=tuple(float(value) for value in model.dof_damping[dof_slice]),
        friction_loss=tuple(float(value) for value in model.dof_frictionloss[dof_slice]),
        armature=tuple(float(value) for value in model.dof_armature[dof_slice]),
        qpos_address=int(model.jnt_qposadr[joint_id]),
        dof_address=dof_address,
        passive=joint_type != mujoco.mjtJoint.mjJNT_FREE and name not in servo_joint_names,
    )


def _object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    name = mujoco.mj_id2name(model, object_type, object_id)
    if name is None:
        raise ValueError(f"Unnamed {object_type.name} object at index {object_id}")
    return name
