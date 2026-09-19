"""mjcf2godot emit paths must follow --out, not a hardcoded walking robot folder."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from mjcf2godot.convert import convert, godot_res_path
from sim2sim.paths import microduck_rl

ROLLERS = microduck_rl() / "src/mjlab_microduck/robot/microduck/scene_rollers.xml"
WALK = ROLLERS.with_name('scene.xml')
BALL = ROLLERS.with_name('scene_ball.xml')


class TestGodotResPath(unittest.TestCase):
    def test_keeps_generated_robot_folder(self) -> None:
        self.assertEqual(
            godot_res_path(Path("/x/godot/generated/microduck_roller"), "meshes/a.obj"),
            "res://generated/microduck_roller/meshes/a.obj",
        )
        self.assertEqual(
            godot_res_path(Path("/x/godot/generated/microduck"), "meshes/a.obj"),
            "res://generated/microduck/meshes/a.obj",
        )


class TestRollerTscnMeshPaths(unittest.TestCase):
    @unittest.skipUnless(WALK.is_file() and BALL.is_file(), 'requires Microduck source assets')
    def test_ball_scene_preserves_the_same_robot_foot_collisions(self) -> None:
        def foot_shapes(scene):
            blocks = re.split(r'(?=^\[)', scene, flags=re.M)
            resources = {re.search(r'id="([^"]+)"', b).group(1): b
                         for b in blocks if b.startswith('[sub_resource')}
            result = {}
            for block in blocks:
                if not block.startswith('[node name="col_') or 'foot_collision' not in block.splitlines()[0]:
                    continue
                parent = re.search(r'parent="([^"]+)"', block).group(1)
                resource = re.search(r'shape = SubResource\("([^"]+)"\)', block).group(1)
                result[parent] = resources[resource].splitlines()[1:]
            return result
        with tempfile.TemporaryDirectory() as td:
            scenes = []
            for name, xml in [('walk', WALK), ('ball', BALL)]:
                out = Path(td) / 'godot/generated' / name
                convert(xml, out, no_visual=True)
                scenes.append(foot_shapes((out / 'robot.tscn').read_text()))
            self.assertEqual(set(scenes[0]), {'ankle_left', 'ankle_right'})
            self.assertEqual(scenes[0], scenes[1])

    def test_colliding_ball_has_a_visible_surface(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            xml = Path(td) / "ball.xml"
            xml.write_text('<mujoco><worldbody><body name="ball" pos="0 0 .035"><freejoint/>'
                           '<geom name="ball_geom" type="sphere" size=".035" mass=".015" rgba="1 .55 0 1"/>'
                           '</body></worldbody></mujoco>')
            out = Path(td) / "godot/generated/ball"
            spec = convert(xml, out)
            scene = (out / "robot.tscn").read_text()
            self.assertIn('type="SphereShape3D"', scene)
            self.assertIn('type="SphereMesh"', scene)
            self.assertIn('type="MeshInstance3D" parent="ball"', scene)
            self.assertAlmostEqual(spec["bodies"][1]["mass"], .015)
            self.assertFalse(any(k.startswith("_visual") for g in spec["geoms"] for k in g))

    @unittest.skipUnless(ROLLERS.is_file(), f"missing {ROLLERS}")
    def test_roller_convert_does_not_point_at_walking_meshes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "generated" / "microduck_roller"
            convert(ROLLERS, out)
            tscn = (out / "robot.tscn").read_text(encoding="utf-8")
            ext = [ln for ln in tscn.splitlines() if ln.startswith("[ext_resource")]
            sample = ext[0] if ext else "no ext_resource"
            self.assertTrue(
                any("res://generated/microduck_roller/meshes/" in ln for ln in ext),
                sample,
            )
            self.assertFalse(
                any('path="res://generated/microduck/meshes/' in ln for ln in ext),
                sample,
            )
            self.assertTrue((out / "meshes" / "unnamed_31_31.obj").is_file())
            self.assertIn('parent="tire"', tscn)
            self.assertIn('parent="tire_4"', tscn)
            self.assertIn("SphereShape3D", tscn)
            self.assertGreaterEqual(tscn.count('[sub_resource type="SphereShape3D"'), 4)
            spec = json.loads((out / "robot_spec.json").read_text())
            wheels = [j for j in spec["joints"] if str(j["name"]).startswith("passive_")]
            self.assertEqual(len(wheels), 4)
            for j in wheels:
                self.assertFalse(j["limited"], j["name"])
                self.assertEqual(j["frictionloss"], 0.0, j["name"])


if __name__ == "__main__":
    unittest.main()
