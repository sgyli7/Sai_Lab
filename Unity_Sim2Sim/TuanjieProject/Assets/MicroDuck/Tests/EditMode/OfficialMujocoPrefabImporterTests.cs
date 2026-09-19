using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AgenticRobot.MicroDuck.Editor;
using Mujoco;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class OfficialMujocoPrefabImporterTests
    {
        [Test]
        public void ImportsAllExpandedOfficialScenesAsCompleteDeterministicPrefabs()
        {
            IReadOnlyList<string> imported = OfficialMujocoPrefabImporter.ImportAll();

            Assert.That(imported, Is.EqualTo(new[]
            {
                OfficialMujocoPrefabImporter.LeggedPrefabAssetPath,
                OfficialMujocoPrefabImporter.RollerPrefabAssetPath,
                OfficialMujocoPrefabImporter.BallPrefabAssetPath,
            }));

            AssertCompletePrefab(
                OfficialMujocoPrefabImporter.LeggedPrefabAssetPath,
                OfficialMujocoPrefabImporter.LeggedResourcesAssetPath,
                expectedRootName: "MicroDuck-MuJoCo-Legged",
                expectedBodies: 15,
                expectedJoints: 15,
                expectedGeoms: 82,
                expectedMeshes: 38);
            AssertCompletePrefab(
                OfficialMujocoPrefabImporter.RollerPrefabAssetPath,
                OfficialMujocoPrefabImporter.RollerResourcesAssetPath,
                expectedRootName: "MicroDuck-MuJoCo-Roller",
                expectedBodies: 19,
                expectedJoints: 19,
                expectedGeoms: 90,
                expectedMeshes: 37);
            AssertCompletePrefab(
                OfficialMujocoPrefabImporter.BallPrefabAssetPath,
                OfficialMujocoPrefabImporter.BallResourcesAssetPath,
                expectedRootName: "MicroDuck-MuJoCo-Ball",
                expectedBodies: 16,
                expectedJoints: 16,
                expectedGeoms: 83,
                expectedMeshes: 38);

            GameObject ballPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                OfficialMujocoPrefabImporter.BallPrefabAssetPath);
            MjBody ballBody = ballPrefab.GetComponentsInChildren<MjBody>(true)
                .Single(body => body.name == "ball");
            Assert.That(ballBody.GetComponentInChildren<MjFreeJoint>(true).name, Is.EqualTo("ball_free"));
            MjGeom ballGeom = ballBody.GetComponentsInChildren<MjGeom>(true)
                .Single(geom => geom.name == "ball_geom");
            Assert.That(ballGeom.ShapeType, Is.EqualTo(MjShapeComponent.ShapeTypes.Sphere));
            Assert.That(ballGeom.Sphere.Radius, Is.EqualTo(0.035f).Within(1e-7f));
            MjInertial ballInertial = ballBody.GetComponentInChildren<MjInertial>(true);
            Assert.That(ballInertial.Mass, Is.EqualTo(0.015f).Within(1e-7f));
            Assert.That(ballInertial.DiagInertia.x, Is.EqualTo(1.225e-5f).Within(1e-9f));
            Assert.That(ballGeom.Settings.Friction.Sliding, Is.EqualTo(0.5f).Within(1e-7f));
            Assert.That(ballGeom.Settings.Friction.Torsional, Is.EqualTo(0.005f).Within(1e-7f));
            Assert.That(ballGeom.Settings.Friction.Rolling, Is.EqualTo(0.0001f).Within(1e-8f));

            AssertExpandedScene(
                OfficialMujocoPrefabImporter.LeggedExpandedXmlAssetPath,
                expectedRobotBody: "trunk_base",
                expectKeyframes: true);
            AssertExpandedScene(
                OfficialMujocoPrefabImporter.RollerExpandedXmlAssetPath,
                expectedRobotBody: "passive_LF_wheel",
                expectKeyframes: true);
            AssertExpandedScene(
                OfficialMujocoPrefabImporter.BallExpandedXmlAssetPath,
                expectedRobotBody: "ball",
                expectKeyframes: false);

            Assert.That(
                AssetDatabase.IsValidFolder(OfficialMujocoPrefabImporter.LeggedStagingAssetPath),
                Is.False,
                "The official importer staging folder must not leak into the project.");
            Assert.That(
                AssetDatabase.IsValidFolder(OfficialMujocoPrefabImporter.BallStagingAssetPath),
                Is.False,
                "The official importer staging folder must not leak into the project.");
            Assert.That(
                AssetDatabase.IsValidFolder(OfficialMujocoPrefabImporter.RollerStagingAssetPath),
                Is.False,
                "The official importer staging folder must not leak into the project.");
        }

        [Test]
        public void ReimportPreservesEveryGeneratedAssetGuidAndDoesNotDuplicateComponents()
        {
            OfficialMujocoPrefabImporter.ImportAll();
            string[] firstPaths = FindGeneratedAssetPaths();
            string[] firstGuids = firstPaths.Select(AssetDatabase.AssetPathToGUID).ToArray();

            OfficialMujocoPrefabImporter.ImportAll();
            string[] secondPaths = FindGeneratedAssetPaths();
            string[] secondGuids = secondPaths.Select(AssetDatabase.AssetPathToGUID).ToArray();

            Assert.That(secondPaths, Is.EqualTo(firstPaths));
            Assert.That(secondGuids, Is.EqualTo(firstGuids));

            GameObject legged = AssetDatabase.LoadAssetAtPath<GameObject>(
                OfficialMujocoPrefabImporter.LeggedPrefabAssetPath);
            GameObject roller = AssetDatabase.LoadAssetAtPath<GameObject>(
                OfficialMujocoPrefabImporter.RollerPrefabAssetPath);
            GameObject ball = AssetDatabase.LoadAssetAtPath<GameObject>(
                OfficialMujocoPrefabImporter.BallPrefabAssetPath);
            Assert.That(legged.GetComponentsInChildren<MjActuator>(true), Has.Length.EqualTo(14));
            Assert.That(roller.GetComponentsInChildren<MjActuator>(true), Has.Length.EqualTo(14));
            Assert.That(ball.GetComponentsInChildren<MjActuator>(true), Has.Length.EqualTo(14));
        }

        private static void AssertCompletePrefab(
            string prefabPath,
            string resourcesPath,
            string expectedRootName,
            int expectedBodies,
            int expectedJoints,
            int expectedGeoms,
            int expectedMeshes)
        {
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
            Assert.That(prefab, Is.Not.Null, prefabPath);
            Assert.That(prefab.name, Is.EqualTo(expectedRootName));
            Assert.That(prefab.GetComponentsInChildren<MjBody>(true), Has.Length.EqualTo(expectedBodies));
            Assert.That(prefab.GetComponentsInChildren<MjBaseJoint>(true), Has.Length.EqualTo(expectedJoints));
            Assert.That(prefab.GetComponentsInChildren<MjGeom>(true), Has.Length.EqualTo(expectedGeoms));
            Assert.That(prefab.GetComponentsInChildren<MjActuator>(true), Has.Length.EqualTo(14));
            Assert.That(prefab.GetComponentsInChildren<MjBaseSensor>(true), Has.Length.EqualTo(6));
            Assert.That(prefab.GetComponentsInChildren<MjSite>(true), Has.Length.EqualTo(7));
            Assert.That(
                prefab.GetComponentsInChildren<MjGlobalSettings>(true),
                Is.Empty,
                "Variant prefabs must defer global MuJoCo settings to the playable scene.");

            string[] meshAssetGuids = AssetDatabase.FindAssets("t:Mesh", new[] { resourcesPath });
            Assert.That(meshAssetGuids, Has.Length.EqualTo(expectedMeshes));

            foreach (MjGeom geom in prefab.GetComponentsInChildren<MjGeom>(true))
            {
                if (geom.ShapeType != MjShapeComponent.ShapeTypes.Mesh)
                {
                    continue;
                }

                Assert.That(geom.Mesh.Mesh, Is.Not.Null, $"Mesh is missing on geom '{geom.name}'.");
                string meshPath = AssetDatabase.GetAssetPath(geom.Mesh.Mesh).Replace('\\', '/');
                Assert.That(
                    meshPath,
                    Does.StartWith(resourcesPath + "/"),
                    $"Geom '{geom.name}' references a mesh outside its deterministic resource folder.");
            }

            foreach (MjHingeJoint joint in prefab.GetComponentsInChildren<MjHingeJoint>(true))
            {
                Vector3 axisInParentBody =
                    MjEngineTool.LocalTransformInParentBody(joint).Rotation * Vector3.right;
                Assert.That(
                    Vector3.Distance(axisInParentBody.normalized, Vector3.up),
                    Is.LessThan(1e-5f),
                    $"Joint '{joint.name}' must preserve MicroDuck's MJCF +Z axis. "
                    + $"Imported parent-body axis was {axisInParentBody}.");
            }

            foreach (MeshRenderer renderer in prefab.GetComponentsInChildren<MeshRenderer>(true))
            {
                Assert.That(renderer.sharedMaterial, Is.Not.Null, renderer.name);
                string materialPath = AssetDatabase.GetAssetPath(renderer.sharedMaterial).Replace('\\', '/');
                Assert.That(
                    materialPath,
                    Does.StartWith(resourcesPath + "/"),
                    $"Renderer '{renderer.name}' references a material outside its deterministic resource folder.");
            }
        }

        private static void AssertExpandedScene(
            string assetPath,
            string expectedRobotBody,
            bool expectKeyframes)
        {
            TextAsset expanded = AssetDatabase.LoadAssetAtPath<TextAsset>(assetPath);
            Assert.That(expanded, Is.Not.Null, assetPath);
            Assert.That(expanded.text, Does.Not.Contain("<include"));
            Assert.That(expanded.text, Does.Contain($"name=\"{expectedRobotBody}\""));
            Assert.That(expanded.text, Does.Contain("<actuator>"));
            Assert.That(expanded.text, Does.Contain("<sensor>"));
            Assert.That(expanded.text, Does.Contain("axis=\"0 0 1\""),
                "The saved import input must make MuJoCo's implicit +Z joint axis explicit.");
            Assert.That(
                expanded.text.Contains("<keyframe>"),
                Is.EqualTo(expectKeyframes));
        }

        private static string[] FindGeneratedAssetPaths()
        {
            return AssetDatabase.FindAssets(
                    string.Empty,
                    new[] { OfficialMujocoPrefabImporter.GeneratedRootAssetPath })
                .Select(AssetDatabase.GUIDToAssetPath)
                .Where(path => !string.IsNullOrEmpty(path))
                .OrderBy(path => path, StringComparer.Ordinal)
                .ToArray();
        }
    }
}
