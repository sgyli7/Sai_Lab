import hashlib
import json
import struct
from pathlib import Path

import mujoco
import numpy as np

from agenticrobot_bridge.tuanjie_assets import (
    convert_stl_to_obj,
    main,
    prepare_tuanjie_assets,
)
from agenticrobot_bridge.upstream import load_upstream_lock


ROOT = Path(__file__).parents[1]
UPSTREAM_ROOT = ROOT / ".cache" / "upstream"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generated_hashes(generated: Path) -> dict[str, str]:
    return {
        path.relative_to(generated).as_posix(): _sha256(path)
        for path in sorted(generated.rglob("*"))
        if path.is_file()
    }


def _obj_vertex_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [float(value) for value in line.split()[1:4]]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("v ")
        ],
        dtype=np.float64,
    )
    assert vertices.shape[1:] == (3,)
    return vertices.min(axis=0), vertices.max(axis=0)


def _obj_topology_counts(path: Path) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    vertex_count = sum(line.startswith("v ") for line in lines)
    faces = [line for line in lines if line.startswith("f ")]
    assert vertex_count > 3
    assert faces
    assert all(
        len(face.split()) == 4
        and all(1 <= int(index) <= vertex_count for index in face.split()[1:])
        for face in faces
    )
    return vertex_count, len(faces)


def test_binary_stl_conversion_maps_basis_and_reverses_winding(tmp_path: Path) -> None:
    source = tmp_path / "triangle.stl"
    header = b"binary triangle".ljust(80, b"\0")
    triangle = struct.pack(
        "<12fH",
        0.0,
        0.0,
        1.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        0,
    )
    source.write_bytes(header + struct.pack("<I", 1) + triangle)

    result = convert_stl_to_obj(
        source,
        tmp_path / "triangle.obj",
        source_relative_path="upstream/assets/triangle.stl",
        source_repository="https://example.invalid/microduck_rl.git",
        source_commit="a" * 40,
    )

    lines = result.destination.read_text(encoding="utf-8").splitlines()
    assert "# Source SHA-256: " + _sha256(source) in lines
    assert "# Modified: converted MuJoCo coordinates to Tuanjie coordinates." in lines
    assert "v -2 3 1" in lines
    assert "v -5 6 4" in lines
    assert "v -8 9 7" in lines
    assert "f 1 3 2" in lines
    assert result.vertex_count == 3
    assert result.triangle_count == 1


def test_ascii_stl_conversion_is_supported_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "ascii.stl"
    source.write_text(
        """solid sample
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 0 1 0
 endloop
endfacet
endsolid sample
""",
        encoding="ascii",
    )

    first = convert_stl_to_obj(
        source,
        tmp_path / "first.obj",
        source_relative_path="sample.stl",
        source_repository="https://example.invalid/source.git",
        source_commit="b" * 40,
    )
    second = convert_stl_to_obj(
        source,
        tmp_path / "second.obj",
        source_relative_path="sample.stl",
        source_repository="https://example.invalid/source.git",
        source_commit="b" * 40,
    )

    assert first.vertex_count == 3
    assert first.triangle_count == 1
    assert first.destination.read_bytes() == second.destination.read_bytes()


def test_prepare_tuanjie_assets_exports_physx_sized_mujoco_collision_hulls(
    tmp_path: Path,
) -> None:
    tuanjie_project = tmp_path / "TuanjieProject"

    report = prepare_tuanjie_assets(ROOT, tuanjie_project)
    payload = report.to_dict()

    assert payload["schemaVersion"] == 2
    assert payload["checks"]["allCollisionMeshesPresent"] is True

    collision_meshes = {item["name"]: item for item in payload["collisionMeshes"]}
    assert set(collision_meshes) == {
        "bottom_head_shell",
        "hip_l",
        "jaw",
        "leg",
        "np_f970",
        "power_support",
        "sole_left",
        "sole_right",
        "top_head_shell",
    }
    for collision_mesh in collision_meshes.values():
        collision_asset = tuanjie_project / collision_mesh["objAsset"]
        vertex_count, triangle_count = _obj_topology_counts(collision_asset)
        assert collision_mesh["maxHullVertices"] == 128
        assert collision_mesh["vertexCount"] == vertex_count <= 128
        assert collision_mesh["triangleCount"] == triangle_count <= 252
        assert collision_mesh["objSha256"] == _sha256(collision_asset)
        assert collision_mesh["meshFrame"] == "mujoco-compiled-capped-convex-hull"

    manifest_payloads = {
        manifest["variant"]: json.loads(
            (tuanjie_project / manifest["asset"]).read_text(encoding="utf-8")
        )
        for manifest in payload["manifests"]
    }
    for manifest_payload in manifest_payloads.values():
        meshes = {mesh["name"]: mesh for mesh in manifest_payload["meshes"]}
        applicable_names = {
            geom["meshName"]
            for geom in manifest_payload["geoms"]
            if geom["collidable"]
            and geom["meshName"] is not None
            and geom["meshName"].lower() != "tire"
        }
        assert {
            mesh_name for mesh_name, mesh in meshes.items() if "collisionAssetFile" in mesh
        } == applicable_names
        for mesh_name in applicable_names:
            generated_mesh = collision_meshes[mesh_name]
            assert meshes[mesh_name]["collisionAssetFile"] == generated_mesh["objAsset"]
            assert meshes[mesh_name]["collisionAssetSha256"] == generated_mesh["objSha256"]

    collisions = {item["variant"]: item for item in payload["collisionProxies"]}
    assert collisions["legged"] == {
        "variant": "legged",
        "asset": "Assets/MicroDuck/Generated/Collision/legged.json",
        "proxyCount": 11,
        "convexMeshCount": 11,
        "cylinderCount": 0,
        "maxHullVertices": 128,
    }
    assert collisions["roller"] == {
        "variant": "roller",
        "asset": "Assets/MicroDuck/Generated/Collision/roller.json",
        "proxyCount": 13,
        "convexMeshCount": 9,
        "cylinderCount": 4,
        "maxHullVertices": 128,
    }
    for variant, collision_report in collisions.items():
        collision_payload = json.loads(
            (tuanjie_project / collision_report["asset"]).read_text(encoding="utf-8")
        )
        assert collision_payload["schemaVersion"] == 2
        assert collision_payload["variant"] == variant
        assert collision_payload["maxHullVertices"] == 128
        assert sum(
            proxy["proxyType"] == "convexMesh" for proxy in collision_payload["proxies"]
        ) == collision_report["convexMeshCount"]
        assert sum(
            proxy["proxyType"] == "cylinder" for proxy in collision_payload["proxies"]
        ) == collision_report["cylinderCount"]
        assert all(
            proxy["collisionAssetFile"] == collision_meshes[proxy["meshName"]]["objAsset"]
            and proxy["collisionAssetSha256"]
            == collision_meshes[proxy["meshName"]]["objSha256"]
            for proxy in collision_payload["proxies"]
            if proxy["proxyType"] == "convexMesh"
        )


def test_prepare_tuanjie_assets_builds_complete_deterministic_bundle(
    tmp_path: Path,
    capsys,
) -> None:
    tuanjie_project = tmp_path / "TuanjieProject"
    runtime = tuanjie_project / "Assets" / "MicroDuck" / "Runtime"
    runtime.mkdir(parents=True)
    (runtime / "keep.cs").write_text("// keep\n", encoding="utf-8")
    stale = tuanjie_project / "Assets" / "MicroDuck" / "Generated" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale", encoding="utf-8")

    report = prepare_tuanjie_assets(ROOT, tuanjie_project)
    generated = tuanjie_project / "Assets" / "MicroDuck" / "Generated"
    payload = json.loads((generated / "asset-report.json").read_text(encoding="utf-8"))
    lock = load_upstream_lock(ROOT / "upstream.lock.json")

    assert report.passed is True
    assert payload == report.to_dict()
    assert payload["schemaVersion"] == 2
    assert payload["generatedRoot"] == "Assets/MicroDuck/Generated"
    assert payload["checks"] == {
        "allCollisionMeshesPresent": True,
        "allReferencedObjPresent": True,
        "exactPolicySet": True,
        "manifestCountsMatch": True,
        "sourceHashesRecorded": True,
    }
    assert not stale.exists()
    assert (runtime / "keep.cs").read_text(encoding="utf-8") == "// keep\n"

    policies = payload["policies"]
    assert [policy["name"] for policy in policies] == list(lock.policies)
    assert len(list((generated / "Policies" / "Original").glob("*.onnx"))) == 9
    assert len(list((generated / "Policies" / "Barracuda").glob("*.onnx"))) == 9
    for policy in policies:
        original = tuanjie_project / policy["originalAsset"]
        converted = tuanjie_project / policy["barracudaAsset"]
        upstream = UPSTREAM_ROOT / "microduck" / "policies" / policy["name"]
        assert original.read_bytes() == upstream.read_bytes()
        assert policy["sourceSha256"] == _sha256(upstream)
        assert policy["barracudaSha256"] == _sha256(converted)

    fixtures = json.loads(
        (generated / "Policies" / "Barracuda" / "parity-fixtures.json").read_text()
    )
    assert fixtures["inputShape"] == [1, 61]
    assert fixtures["outputShape"] == [1, 14]
    assert len(fixtures["cases"]) == 27
    assert [
        (case["policyName"], case["fixtureName"]) for case in fixtures["cases"]
    ] == [
        (policy_name, fixture_name)
        for policy_name in lock.policies
        for fixture_name in ("zeros", "ramp", "seeded")
    ]
    assert all(
        len(case["input"]) == 61
        and len(case["expectedOutput"]) == 14
        and np.isfinite(case["expectedOutput"]).all()
        for case in fixtures["cases"]
    )

    manifests = {manifest["variant"]: manifest for manifest in payload["manifests"]}
    assert {
        variant: (
            item["bodyCount"],
            item["jointCount"],
            item["servoCount"],
            item["passiveJointCount"],
        )
        for variant, item in manifests.items()
    } == {"legged": (15, 15, 14, 0), "roller": (19, 19, 14, 4)}
    assert manifests["legged"]["sourceFile"].endswith("/scene.xml")
    assert manifests["roller"]["sourceFile"].endswith("/scene_rollers.xml")
    for manifest_report in manifests.values():
        manifest_payload = json.loads(
            (tuanjie_project / manifest_report["asset"]).read_text(encoding="utf-8")
        )
        assert all(
            (tuanjie_project / mesh["assetFile"]).is_file()
            and len(mesh["sourceSha256"]) == 64
            for mesh in manifest_payload["meshes"]
        )

    assert len(payload["meshes"]) == 43
    for mesh in payload["meshes"]:
        obj = tuanjie_project / mesh["objAsset"]
        assert obj.is_file()
        assert mesh["sourceSha256"]
        assert mesh["objSha256"] == _sha256(obj)
        assert mesh["vertexCount"] > 0
        assert mesh["triangleCount"] > 0

    # MuJoCo recenters and reorients imported meshes during compilation, then
    # compensates that change in each compiled geom transform.  The OBJ paired
    # with those transforms must therefore contain the compiled mesh vertices,
    # not the raw STL frame.  A raw sole STL is offset by roughly five
    # centimetres and starts one foot deeply inside Tuanjie's floor.
    model = mujoco.MjModel.from_xml_path(str(UPSTREAM_ROOT / "microduck_rl" / "src"
        / "mjlab_microduck" / "robot" / "microduck" / "scene.xml"))
    sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "sole_right")
    vertex_address = int(model.mesh_vertadr[sole_id])
    vertex_count = int(model.mesh_vertnum[sole_id])
    compiled_vertices = np.asarray(
        model.mesh_vert[vertex_address : vertex_address + vertex_count],
        dtype=np.float64,
    )
    expected_vertices = compiled_vertices[:, [1, 2, 0]].copy()
    expected_vertices[:, 0] *= -1.0
    actual_minimum, actual_maximum = _obj_vertex_bounds(
        generated / "Meshes" / "sole_right.obj"
    )
    np.testing.assert_allclose(actual_minimum, expected_vertices.min(axis=0), atol=1e-8)
    np.testing.assert_allclose(actual_maximum, expected_vertices.max(axis=0), atol=1e-8)

    collisions = {item["variant"]: item for item in payload["collisionProxies"]}
    assert collisions["legged"]["proxyCount"] == 11
    assert collisions["legged"]["convexMeshCount"] == 11
    assert collisions["legged"]["cylinderCount"] == 0
    assert collisions["roller"]["proxyCount"] == 13
    assert collisions["roller"]["convexMeshCount"] == 9
    assert collisions["roller"]["cylinderCount"] == 4
    roller_collision = json.loads(
        (tuanjie_project / collisions["roller"]["asset"]).read_text(encoding="utf-8")
    )
    assert roller_collision["schemaVersion"] == 2
    assert sum(
        proxy["proxyType"] == "cylinder" for proxy in roller_collision["proxies"]
    ) == 4
    assert all(
        len(proxy["halfExtents"]) == 3 and min(proxy["halfExtents"]) > 0
        for proxy in roller_collision["proxies"]
    )

    assert (generated / "Licenses" / "microduck-LICENSE.txt").read_bytes() == (
        UPSTREAM_ROOT / "microduck" / "LICENSE"
    ).read_bytes()
    assert (generated / "Licenses" / "microduck_rl-LICENSE.txt").read_bytes() == (
        UPSTREAM_ROOT / "microduck_rl" / "LICENSE"
    ).read_bytes()
    attribution = json.loads(
        (generated / "Licenses" / "ATTRIBUTION.json").read_text(encoding="utf-8")
    )
    assert attribution["meshLicense"] == "Creative Commons BY-SA-NC"
    assert attribution["repositories"]["microduck_rl"]["commit"] == (
        lock.repositories["microduck_rl"].commit
    )

    first_hashes = _generated_hashes(generated)
    assert main(
        [
            "--project-root",
            str(ROOT),
            "--tuanjie-project",
            str(tuanjie_project),
        ]
    ) == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["passed"] is True
    assert _generated_hashes(generated) == first_hashes
