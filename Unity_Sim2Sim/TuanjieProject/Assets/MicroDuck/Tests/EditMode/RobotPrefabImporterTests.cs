using System;
using System.Collections.Generic;
using System.Linq;
using AgenticRobot.MicroDuck.Editor;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class RobotPrefabImporterTests
    {
        private const string RobotAndFloorMaterialPath =
            "Assets/MicroDuck/Generated/PhysicsMaterials/MuJoCo-RobotAndFloor.physicMaterial";

        [Test]
        public void BuildsManifestBodyHierarchyWithPhysicalProperties()
        {
            RobotManifestData manifest = CreateFixtureManifest();
            GameObject prefabRoot = null;

            try
            {
                prefabRoot = RobotPrefabImporter.BuildInMemory(manifest);

                ArticulationBody[] bodies = prefabRoot.GetComponentsInChildren<ArticulationBody>();
                Assert.That(bodies.Length, Is.EqualTo(2));

                ArticulationBody root = bodies.Single(body => body.name == "root");
                ArticulationBody child = bodies.Single(body => body.name == "child");
                Assert.That(root.transform.parent, Is.EqualTo(prefabRoot.transform));
                Assert.That(child.transform.parent, Is.EqualTo(root.transform));
                Assert.That(root.transform.localPosition, Is.EqualTo(new Vector3(1f, 2f, 3f)));
                Assert.That(child.transform.localPosition, Is.EqualTo(new Vector3(0.1f, 0.2f, 0.3f)));
                Assert.That(root.mass, Is.EqualTo(0.42f).Within(1e-6f));
                Assert.That(root.centerOfMass, Is.EqualTo(new Vector3(0.01f, 0.02f, 0.03f)));
                Assert.That(root.inertiaTensor, Is.EqualTo(new Vector3(0.1f, 0.2f, 0.3f)));
                Assert.That(root.automaticCenterOfMass, Is.False);
                Assert.That(root.automaticInertiaTensor, Is.False);
                Assert.That(root.immovable, Is.False);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(prefabRoot);
            }
        }

        [Test]
        public void ConvertsHingeAxisLimitsAndForceDriveWithoutMisapplyingGeneralizedArmature()
        {
            RobotManifestData manifest = CreateFixtureManifest();
            GameObject prefabRoot = null;

            try
            {
                prefabRoot = RobotPrefabImporter.BuildInMemory(manifest);

                ArticulationBody child = prefabRoot
                    .GetComponentsInChildren<ArticulationBody>()
                    .Single(body => body.name == "child");
                Assert.That(child.jointType, Is.EqualTo(ArticulationJointType.RevoluteJoint));
                Assert.That(child.twistLock, Is.EqualTo(ArticulationDofLock.LimitedMotion));
                Assert.That(child.anchorPosition, Is.EqualTo(new Vector3(0f, 0f, 0f)));
                Assert.That(
                    Vector3.Distance(child.anchorRotation * Vector3.right, new Vector3(0f, -1f, 0f)),
                    Is.LessThan(1e-5f));

                ArticulationDrive drive = child.xDrive;
                Assert.That(drive.lowerLimit, Is.EqualTo(-Mathf.Rad2Deg).Within(1e-4f));
                Assert.That(drive.upperLimit, Is.EqualTo(Mathf.Rad2Deg).Within(1e-4f));
                Assert.That(drive.driveType, Is.EqualTo(ArticulationDriveType.Force));
                Assert.That(drive.stiffness, Is.EqualTo(0.55f).Within(1e-6f));
                Assert.That(drive.damping, Is.EqualTo(0.053f).Within(1e-6f));
                Assert.That(drive.forceLimit, Is.EqualTo(0.96f).Within(1e-6f));
                Assert.That(child.inertiaTensor.x, Is.EqualTo(0.1f).Within(1e-6f));
                Assert.That(child.inertiaTensor.y, Is.EqualTo(0.2f).Within(1e-6f));
                Assert.That(child.inertiaTensor.z, Is.EqualTo(0.3f).Within(1e-6f));
                Assert.That(Quaternion.Angle(child.inertiaTensorRotation, Quaternion.identity),
                    Is.LessThan(1e-4f));
                Assert.That(child.jointFriction, Is.EqualTo(0.004f).Within(1e-6f));
                Assert.That(child.linearDamping, Is.EqualTo(0f));
                Assert.That(child.angularDamping, Is.EqualTo(0f));
                Assert.That(child.solverIterations, Is.EqualTo(12));
                Assert.That(child.solverVelocityIterations, Is.EqualTo(4));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(prefabRoot);
            }
        }

        [Test]
        public void PassiveWheelUsesTheOfficialDeploymentBearingFriction()
        {
            RobotManifestData manifest = CreateFixtureManifest();
            manifest.joints[1].passive = true;
            manifest.joints[1].frictionLoss = new[] { 0f };
            manifest.passiveJointNames = new[] { manifest.joints[1].name };
            manifest.servos = Array.Empty<ManifestServoData>();
            GameObject prefabRoot = null;

            try
            {
                prefabRoot = RobotPrefabImporter.BuildInMemory(manifest);
                ArticulationBody wheel = prefabRoot
                    .GetComponentsInChildren<ArticulationBody>()
                    .Single(body => body.name == "child");

                Assert.That(wheel.jointFriction, Is.EqualTo(0.003f).Within(1e-6f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(prefabRoot);
            }
        }

        [Test]
        public void PassiveWheelArmatureCalibrationProxyAddsOnlyToTheAlignedPrincipalAxis()
        {
            RobotManifestData manifest = CreateFixtureManifest();
            manifest.variant = "roller";
            manifest.joints[1].passive = true;
            manifest.joints[1].axis = new[] { 1f, 0f, 0f };
            manifest.joints[1].armature = new[] { 0.04f };
            manifest.passiveJointNames = new[] { manifest.joints[1].name };
            manifest.servos = Array.Empty<ManifestServoData>();
            GameObject prefabRoot = null;

            try
            {
                prefabRoot = RobotPrefabImporter.BuildInMemory(manifest);
                ArticulationBody wheel = prefabRoot
                    .GetComponentsInChildren<ArticulationBody>()
                    .Single(body => body.name == "child");

                Assert.That(wheel.inertiaTensor.x, Is.EqualTo(0.14f).Within(1e-6f));
                Assert.That(wheel.inertiaTensor.y, Is.EqualTo(0.2f).Within(1e-6f));
                Assert.That(wheel.inertiaTensor.z, Is.EqualTo(0.3f).Within(1e-6f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(prefabRoot);
            }
        }

        [Test]
        public void RotatedRightWheelArmatureCalibrationUsesTheBodyLocalAxisExactlyOnce()
        {
            RobotManifestData manifest = CreateFixtureManifest();
            manifest.variant = "roller";
            ManifestBodyData wheelBody = manifest.bodies[1];
            wheelBody.localRotationWxyz = new[] { 0f, 1f, 0f, 0f };
            wheelBody.inertiaTensor = new[] { 0.1f, 0.2f, 0.21f };
            wheelBody.inertiaRotationWxyz = new[] { 0.5f, -0.5f, -0.5f, -0.5f };
            ManifestJointData wheelJoint = manifest.joints[1];
            wheelJoint.passive = true;
            wheelJoint.axis = new[] { 0f, -1f, 0f };
            wheelJoint.armature = new[] { 0.04f };
            manifest.passiveJointNames = new[] { wheelJoint.name };
            manifest.servos = Array.Empty<ManifestServoData>();
            GameObject prefabRoot = null;

            try
            {
                prefabRoot = RobotPrefabImporter.BuildInMemory(manifest);
                ArticulationBody wheel = prefabRoot
                    .GetComponentsInChildren<ArticulationBody>()
                    .Single(body => body.name == "child");

                Assert.That(wheel.inertiaTensor.x, Is.EqualTo(0.1f).Within(1e-6f));
                Assert.That(wheel.inertiaTensor.y, Is.EqualTo(0.2f).Within(1e-6f));
                Assert.That(wheel.inertiaTensor.z, Is.EqualTo(0.25f).Within(1e-6f));
                Assert.That(
                    Quaternion.Angle(
                        wheel.inertiaTensorRotation,
                        new Quaternion(-0.5f, -0.5f, -0.5f, 0.5f)),
                    Is.LessThan(1e-4f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(prefabRoot);
            }
        }

        [Test]
        public void PassiveWheelArmatureCalibrationRejectsAnUnalignedPrincipalFrame()
        {
            RobotManifestData manifest = CreateFixtureManifest();
            manifest.variant = "roller";
            manifest.bodies[1].inertiaRotationWxyz = new[]
            {
                0.9238795f,
                0f,
                0f,
                0.3826834f,
            };
            manifest.joints[1].passive = true;
            manifest.joints[1].axis = new[] { 1f, 0f, 0f };
            manifest.joints[1].armature = new[] { 0.04f };
            manifest.passiveJointNames = new[] { manifest.joints[1].name };
            manifest.servos = Array.Empty<ManifestServoData>();

            Assert.That(
                () => RobotPrefabImporter.BuildInMemory(manifest),
                Throws.TypeOf<InvalidOperationException>()
                    .With.Message.Contains("principal inertia axis"));
        }

        [Test]
        public void ImportedMicroDuckWheelArmatureCalibrationUsesTheExactPhysicalDiskFallback()
        {
            const string tempFolder =
                "Assets/MicroDuck/Tests/EditMode/PassiveWheelArmatureImporterTemp";
            const string prefabPath = tempFolder + "/fixture.prefab";
            EnsureAssetFolder(tempFolder);
            GameObject importedInstance = null;

            try
            {
                TextAsset manifestAsset = AssetDatabase.LoadAssetAtPath<TextAsset>(
                    RobotPrefabImporter.RollerManifestAssetPath);
                Assert.That(manifestAsset, Is.Not.Null);
                RobotManifestData manifest = RobotManifestJson.Parse(manifestAsset.text);
                ManifestJointData wheelJoint = manifest.joints.Single(
                    joint => joint.name == "passive_LF_wheel");
                ManifestBodyData wheelBody = manifest.bodies.Single(
                    body => body.name == wheelJoint.bodyName);

                RobotPrefabImporter.ImportManifestAsset(
                    RobotPrefabImporter.RollerManifestAssetPath,
                    prefabPath);

                GameObject importedPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
                Assert.That(importedPrefab, Is.Not.Null);
                importedInstance = PrefabUtility.InstantiatePrefab(importedPrefab) as GameObject;
                Assert.That(importedInstance, Is.Not.Null);
                ArticulationBody wheel = importedInstance
                    .GetComponentsInChildren<ArticulationBody>(true)
                    .Single(body => body.name == wheelBody.name);
                float armature = wheelJoint.armature[0];
                Assert.That(
                    wheel.inertiaTensor.x,
                    Is.EqualTo(wheelBody.inertiaTensor[0] + (armature * 0.5f)).Within(1e-9f));
                Assert.That(
                    wheel.inertiaTensor.y,
                    Is.EqualTo(wheelBody.inertiaTensor[1] + (armature * 0.5f)).Within(1e-9f));
                Assert.That(
                    wheel.inertiaTensor.z,
                    Is.EqualTo(wheelBody.inertiaTensor[2] + armature).Within(1e-9f));
                Assert.That(
                    wheel.inertiaTensor.x + wheel.inertiaTensor.y,
                    Is.GreaterThanOrEqualTo(wheel.inertiaTensor.z));
                foreach (float moment in new[]
                {
                    wheel.inertiaTensor.x,
                    wheel.inertiaTensor.y,
                    wheel.inertiaTensor.z,
                })
                {
                    Assert.That(float.IsNaN(moment), Is.False);
                    Assert.That(float.IsInfinity(moment), Is.False);
                    Assert.That(moment, Is.GreaterThan(0f));
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(importedInstance);
                AssetDatabase.DeleteAsset(tempFolder);
            }
        }

        [Test]
        public void BuildsVisualChildrenAndDeterministicColliderProxies()
        {
            RobotManifestData manifest = CreateFixtureManifest();
            manifest.meshes = new[]
            {
                new ManifestMeshData
                {
                    name = "shell_mesh",
                    sourceFile = "assets/shell.obj",
                    scale = new[] { 2f, 3f, 4f },
                },
                new ManifestMeshData
                {
                    name = "tire_mesh",
                    sourceFile = "assets/tire.obj",
                    scale = new[] { 1f, 1f, 1f },
                },
            };
            manifest.geoms = new[]
            {
                Geom("shell_visual", "child", "shell_mesh", false, new[] { 0.1f, 0.2f, 0.3f }),
                Geom("shell_collision", "child", "shell_mesh", true, new[] { 0.1f, 0.2f, 0.3f }),
                Geom("tire_collision", "child", "tire_mesh", true, new[] { 0.4f, 0.2f, 0.1f }),
                Geom("self_only_collision", "child", "shell_mesh", true,
                    new[] { 0.1f, 0.2f, 0.3f }, contype: 2, conaffinity: 2),
                Geom("world_floor", "world", null, true, new[] { 10f, 0.1f, 10f }),
            };

            GameObject prefabRoot = null;
            var visualMesh = new Mesh
            {
                name = "fixture-visual-mesh",
                vertices = new[]
                {
                    Vector3.zero,
                    Vector3.right,
                    Vector3.up,
                    Vector3.forward,
                },
                triangles = new[]
                {
                    0, 2, 1,
                    0, 1, 3,
                    0, 3, 2,
                    1, 2, 3,
                },
            };
            var collisionMesh = UnityEngine.Object.Instantiate(visualMesh);
            collisionMesh.name = "fixture-collision-mesh";
            try
            {
                prefabRoot = RobotPrefabImporter.BuildInMemory(
                    manifest,
                    ignored => visualMesh,
                    ignored => collisionMesh);

                Transform child = prefabRoot.transform.Find("root/child");
                Transform visual = child.Find("shell_visual");
                Assert.That(visual, Is.Not.Null);
                Assert.That(visual.localPosition, Is.EqualTo(new Vector3(0.01f, 0.02f, 0.03f)));
                Assert.That(visual.localScale, Is.EqualTo(new Vector3(2f, 3f, 4f)));
                Assert.That(visual.GetComponent<MeshFilter>().sharedMesh, Is.SameAs(visualMesh));
                Assert.That(visual.GetComponent<MeshRenderer>(), Is.Not.Null);
                Assert.That(visual.GetComponent<Collider>(), Is.Null);

                MeshCollider shellCollider = child.Find("shell_collision").GetComponent<MeshCollider>();
                Assert.That(shellCollider, Is.Not.Null);
                Assert.That(shellCollider.convex, Is.True);
                Assert.That(shellCollider.sharedMesh, Is.SameAs(collisionMesh));
                Assert.That(shellCollider.contactOffset, Is.EqualTo(0.001f).Within(1e-7f));
                AssertPersistentMaterial(shellCollider.sharedMaterial, RobotAndFloorMaterialPath, 1f);

                Transform tireColliderTransform = child.Find("tire_collision/tire_collision__cylinder");
                Assert.That(tireColliderTransform, Is.Not.Null);
                MeshCollider tireCollider = tireColliderTransform.GetComponent<MeshCollider>();
                Assert.That(tireCollider, Is.Not.Null);
                Assert.That(tireCollider.convex, Is.True);
                Assert.That(tireCollider.sharedMesh, Is.Not.Null);
                Assert.That(tireCollider.contactOffset, Is.EqualTo(0.001f).Within(1e-7f));
                AssertPersistentMaterial(tireCollider.sharedMaterial, RobotAndFloorMaterialPath, 1f);
                Assert.That(
                    tireColliderTransform.localScale,
                    Is.EqualTo(new Vector3(0.8f, 0.1f, 0.8f)));
                Assert.That(
                    Vector3.Distance(
                        tireColliderTransform.localRotation * Vector3.up,
                        Vector3.forward),
                    Is.LessThan(1e-5f));
                Assert.That(child.Find("self_only_collision").GetComponent<Collider>(), Is.Null,
                    "MuJoCo contype=2/conaffinity=2 does not collide with the type-1 world floor.");
                Assert.That(prefabRoot.transform.Find("world_floor"), Is.Null);
                Assert.That(
                    prefabRoot.GetComponentsInChildren<Transform>().Any(item => item.name == "world_floor"),
                    Is.False);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(prefabRoot);
                UnityEngine.Object.DestroyImmediate(visualMesh);
                UnityEngine.Object.DestroyImmediate(collisionMesh);
            }
        }

        [Test]
        public void ImportsManifestTextAssetAsAReusablePrefabAsset()
        {
            const string tempFolder = "Assets/MicroDuck/Tests/EditMode/ImporterTemp";
            const string manifestPath = tempFolder + "/fixture-manifest.asset";
            const string prefabPath = tempFolder + "/fixture.prefab";
            EnsureAssetFolder(tempFolder);

            try
            {
                RobotManifestData manifest = CreateFixtureManifest();
                manifest.geoms = new[]
                {
                    Geom("fixture_collision", "child", null, true, new[] { 0.1f, 0.2f, 0.3f }),
                };
                var manifestAsset = new TextAsset(JsonUtility.ToJson(manifest));
                AssetDatabase.CreateAsset(manifestAsset, manifestPath);

                string importedPath = RobotPrefabImporter.ImportManifestAsset(
                    manifestPath,
                    prefabPath);

                Assert.That(importedPath, Is.EqualTo(prefabPath));
                GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
                Assert.That(prefab, Is.Not.Null);
                Assert.That(prefab.GetComponentsInChildren<ArticulationBody>().Length, Is.EqualTo(2));
                Collider collider = prefab.GetComponentInChildren<Collider>();
                Assert.That(collider, Is.Not.Null);
                AssertPersistentMaterial(collider.sharedMaterial, RobotAndFloorMaterialPath, 1f);
            }
            finally
            {
                AssetDatabase.DeleteAsset(tempFolder);
            }
        }

        [Test]
        public void ImportsBothGeneratedRobotVariantsWithAllBodiesJointsAndMeshes()
        {
            IReadOnlyList<string> importedPrefabs = RobotPrefabImporter.ImportAll();

            Assert.That(importedPrefabs, Is.EqualTo(new[]
            {
                RobotPrefabImporter.LeggedPrefabAssetPath,
                RobotPrefabImporter.RollerPrefabAssetPath,
            }));
            AssertImportedRobot(
                RobotPrefabImporter.LeggedManifestAssetPath,
                RobotPrefabImporter.LeggedPrefabAssetPath,
                expectedBodies: 15,
                expectedJoints: 15,
                expectedServos: 14,
                expectedPassiveJoints: 0);
            AssertImportedRobot(
                RobotPrefabImporter.RollerManifestAssetPath,
                RobotPrefabImporter.RollerPrefabAssetPath,
                expectedBodies: 19,
                expectedJoints: 19,
                expectedServos: 14,
                expectedPassiveJoints: 4);
        }

        private static RobotManifestData CreateFixtureManifest()
        {
            return new RobotManifestData
            {
                schemaVersion = 1,
                source = new ManifestSourceData
                {
                    relativePath = "fixture.xml",
                    sha256 = new string('a', 64),
                    modelName = "fixture",
                },
                variant = "legged",
                timestep = 0.005f,
                gravity = new[] { 0f, -9.81f, 0f },
                bodies = new[]
                {
                    Body("root", null, new[] { 1f, 2f, 3f }, 0.42f),
                    Body("child", "root", new[] { 0.1f, 0.2f, 0.3f }, 0.1f),
                },
                joints = new[]
                {
                    Joint("root_free", "root", "free", new[] { 1f, 0f, 0f }),
                    Joint("hinge", "child", "hinge", new[] { 0f, -1f, 0f }),
                },
                servos = new[]
                {
                    new ManifestServoData
                    {
                        name = "hinge",
                        jointName = "hinge",
                        index = 0,
                        ctrlLimited = true,
                        ctrlRange = new[] { -1f, 1f },
                        forceLimited = true,
                        forceRange = new[] { -0.96f, 0.96f },
                        gainPrm = new[] { 0.55f },
                        biasPrm = new[] { 0f, -0.55f },
                        gear = new[] { 1f },
                    },
                },
                passiveJointNames = Array.Empty<string>(),
                geoms = Array.Empty<ManifestGeomData>(),
                meshes = Array.Empty<ManifestMeshData>(),
                sensors = Array.Empty<ManifestSensorData>(),
                keyframes = Array.Empty<ManifestKeyframeData>(),
            };
        }

        private static ManifestBodyData Body(
            string name,
            string parentName,
            float[] localPosition,
            float mass)
        {
            return new ManifestBodyData
            {
                name = name,
                parentName = parentName,
                localPosition = localPosition,
                localRotationWxyz = new[] { 1f, 0f, 0f, 0f },
                mass = mass,
                centerOfMass = new[] { 0.01f, 0.02f, 0.03f },
                inertiaTensor = new[] { 0.1f, 0.2f, 0.3f },
                inertiaRotationWxyz = new[] { 1f, 0f, 0f, 0f },
            };
        }

        private static ManifestJointData Joint(
            string name,
            string bodyName,
            string type,
            float[] axis)
        {
            return new ManifestJointData
            {
                name = name,
                bodyName = bodyName,
                type = type,
                localAnchor = new[] { 0f, 0f, 0f },
                axis = axis,
                limited = type == "hinge",
                rangeRad = new[] { -1f, 1f },
                damping = type == "free" ? new float[6] : new[] { 0.053f },
                frictionLoss = type == "free" ? new float[6] : new[] { 0.004f },
                armature = type == "free" ? new float[6] : new[] { 0.001f },
                qposAddress = type == "free" ? 0 : 7,
                dofAddress = type == "free" ? 0 : 6,
                passive = false,
            };
        }

        private static ManifestGeomData Geom(
            string name,
            string bodyName,
            string meshName,
            bool collidable,
            float[] size,
            int contype = 1,
            int conaffinity = 1)
        {
            var geom = new ManifestGeomData
            {
                name = name,
                bodyName = bodyName,
                type = meshName == null ? "plane" : "mesh",
                localPosition = new[] { 0.01f, 0.02f, 0.03f },
                localRotationWxyz = new[] { 1f, 0f, 0f, 0f },
                size = size,
                meshName = meshName,
                rgba = new[] { 0.2f, 0.3f, 0.4f, 1f },
                friction = new[] { 1f, 0.005f, 0.0001f },
                contype = collidable ? contype : 0,
                conaffinity = collidable ? conaffinity : 0,
                group = collidable ? 3 : 2,
                collidable = collidable,
            };
            return geom;
        }

        private static void EnsureAssetFolder(string path)
        {
            string[] parts = path.Split('/');
            string current = parts[0];
            for (int index = 1; index < parts.Length; index++)
            {
                string next = current + "/" + parts[index];
                if (!AssetDatabase.IsValidFolder(next))
                {
                    AssetDatabase.CreateFolder(current, parts[index]);
                }

                current = next;
            }
        }

        private static void AssertImportedRobot(
            string manifestPath,
            string prefabPath,
            int expectedBodies,
            int expectedJoints,
            int expectedServos,
            int expectedPassiveJoints)
        {
            TextAsset manifestAsset = AssetDatabase.LoadAssetAtPath<TextAsset>(manifestPath);
            Assert.That(manifestAsset, Is.Not.Null, $"Missing generated manifest {manifestPath}");
            RobotManifestData manifest = RobotManifestJson.Parse(manifestAsset.text);
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
            Assert.That(prefab, Is.Not.Null, $"Missing generated prefab {prefabPath}");

            ArticulationBody[] bodies = prefab.GetComponentsInChildren<ArticulationBody>(true);
            Assert.That(manifest.bodies.Length, Is.EqualTo(expectedBodies));
            Assert.That(manifest.joints.Length, Is.EqualTo(expectedJoints));
            Assert.That(manifest.servos.Length, Is.EqualTo(expectedServos));
            Assert.That(manifest.passiveJointNames.Length, Is.EqualTo(expectedPassiveJoints));
            Assert.That(bodies.Length, Is.EqualTo(expectedBodies));

            MicroDuckRig rig = prefab.GetComponent<MicroDuckRig>();
            Assert.That(rig, Is.Not.Null, "Generated prefab must be ready for the runtime controller.");
            Assert.That(
                rig.Variant,
                Is.EqualTo(manifest.variant == "roller" ? RobotVariant.Roller : RobotVariant.Legged));
            Assert.That(
                rig.RootBody.name,
                Is.EqualTo(manifest.bodies.Single(body => string.IsNullOrEmpty(body.parentName)).name));
            Assert.That(rig.ServoCount, Is.EqualTo(expectedServos));

            var joints = manifest.joints.ToDictionary(joint => joint.name, StringComparer.Ordinal);
            var bodiesByName = bodies.ToDictionary(body => body.name, StringComparer.Ordinal);
            Assert.That(
                manifest.servos.Count(servo => bodiesByName[joints[servo.jointName].bodyName].xDrive.stiffness > 0f),
                Is.EqualTo(expectedServos));
            Assert.That(
                manifest.joints.Count(joint => joint.passive
                    && bodiesByName[joint.bodyName].jointType == ArticulationJointType.RevoluteJoint
                    && bodiesByName[joint.bodyName].xDrive.stiffness == 0f),
                Is.EqualTo(expectedPassiveJoints));

            var targetRadians = Enumerable.Range(0, expectedServos)
                .Select(index => (index - 7) * 0.01f)
                .ToArray();
            rig.ApplyTargets(targetRadians);
            for (int index = 0; index < manifest.servos.Length; index++)
            {
                ManifestServoData servo = manifest.servos[index];
                ArticulationBody servoBody = bodiesByName[joints[servo.jointName].bodyName];
                Assert.That(
                    servoBody.xDrive.target,
                    Is.EqualTo(targetRadians[index] * Mathf.Rad2Deg).Within(1e-5f),
                    $"Servo index {index} ({servo.name}) is out of manifest order.");
            }

            var bodyNames = manifest.bodies.Select(body => body.name).ToHashSet(StringComparer.Ordinal);
            int expectedVisuals = manifest.geoms.Count(
                geom => bodyNames.Contains(geom.bodyName) && !string.IsNullOrEmpty(geom.meshName));
            int expectedColliders = manifest.geoms.Count(
                geom => bodyNames.Contains(geom.bodyName)
                    && geom.collidable
                    && ((geom.contype & 1) != 0 || (geom.conaffinity & 1) != 0));
            Assert.That(prefab.GetComponentsInChildren<MeshFilter>(true).Length, Is.EqualTo(expectedVisuals));
            Collider[] colliders = prefab.GetComponentsInChildren<Collider>(true);
            Assert.That(colliders.Length, Is.EqualTo(expectedColliders));
            Assert.That(colliders.Length, Is.GreaterThan(0));
            foreach (Collider collider in colliders)
            {
                AssertPersistentMaterial(collider.sharedMaterial, RobotAndFloorMaterialPath, 1f);
            }
            Assert.That(
                prefab.GetComponentsInChildren<Transform>(true).Any(item => item.name == "floor"),
                Is.False);
        }

        private static void AssertPersistentMaterial(
            PhysicMaterial material,
            string expectedPath,
            float expectedFriction)
        {
            Assert.That(material, Is.Not.Null);
            Assert.That(AssetDatabase.Contains(material), Is.True);
            Assert.That(AssetDatabase.GetAssetPath(material), Is.EqualTo(expectedPath));
            Assert.That(material.staticFriction, Is.EqualTo(expectedFriction).Within(1e-7f));
            Assert.That(material.dynamicFriction, Is.EqualTo(expectedFriction).Within(1e-7f));
            Assert.That(material.bounciness, Is.EqualTo(0f).Within(1e-7f));
            Assert.That(material.frictionCombine, Is.EqualTo(PhysicMaterialCombine.Maximum));
            Assert.That(material.bounceCombine, Is.EqualTo(PhysicMaterialCombine.Minimum));
        }
    }
}
