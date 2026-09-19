using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Editor
{
    /// <summary>
    /// Builds a Tuanjie articulation hierarchy from the engine-neutral MuJoCo manifest.
    /// </summary>
    public static class RobotPrefabImporter
    {
        public const float ContactOffsetMeters = 0.001f;
        public const int SolverIterations = 12;
        public const int SolverVelocityIterations = 4;
        public const float PassiveWheelBearingFriction = 0.003f;
        public const float PassiveWheelArmaturePrincipalAxisMinimumAlignment = 0.999f;

        public const string LeggedManifestAssetPath =
            "Assets/MicroDuck/Generated/Manifests/legged.json";

        public const string RollerManifestAssetPath =
            "Assets/MicroDuck/Generated/Manifests/roller.json";

        public const string LeggedPrefabAssetPath =
            "Assets/MicroDuck/Generated/Prefabs/MicroDuck-Legged.prefab";

        public const string RollerPrefabAssetPath =
            "Assets/MicroDuck/Generated/Prefabs/MicroDuck-Roller.prefab";

        public static IReadOnlyList<string> ImportAll()
        {
            string legged = ImportManifestAsset(
                LeggedManifestAssetPath,
                LeggedPrefabAssetPath);
            string roller = ImportManifestAsset(
                RollerManifestAssetPath,
                RollerPrefabAssetPath);
            AssetDatabase.SaveAssets();
            return new[] { legged, roller };
        }

        /// <summary>
        /// Public no-argument entry point for Tuanjie's -executeMethod batch mode.
        /// </summary>
        public static void ImportAllBatch()
        {
            IReadOnlyList<string> imported = ImportAll();
            Debug.Log($"Imported {imported.Count} MicroDuck prefabs: {string.Join(", ", imported)}");
        }

        [MenuItem("MicroDuck/Import Generated Robot Prefabs")]
        private static void ImportAllFromMenu()
        {
            ImportAllBatch();
        }

        public static string ImportManifestAsset(
            string manifestAssetPath,
            string prefabAssetPath)
        {
            RequireAssetPath(manifestAssetPath, nameof(manifestAssetPath));
            RequireAssetPath(prefabAssetPath, nameof(prefabAssetPath));

            TextAsset manifestAsset = AssetDatabase.LoadAssetAtPath<TextAsset>(manifestAssetPath);
            if (manifestAsset == null)
            {
                throw new FileNotFoundException(
                    $"Robot manifest TextAsset was not found at '{manifestAssetPath}'.",
                    manifestAssetPath);
            }

            RobotManifestData manifest = RobotManifestJson.Parse(manifestAsset.text);
            EnsureAssetFolder(Path.GetDirectoryName(prefabAssetPath)?.Replace('\\', '/'));

            GameObject instanceRoot = null;
            try
            {
                instanceRoot = BuildInMemory(
                    manifest,
                    mesh => ResolveMesh(mesh, manifestAssetPath, collision: false),
                    mesh => ResolveMesh(mesh, manifestAssetPath, collision: true));
                GameObject prefab = PrefabUtility.SaveAsPrefabAsset(
                    instanceRoot,
                    prefabAssetPath,
                    out bool savedSuccessfully);
                if (!savedSuccessfully || prefab == null)
                {
                    throw new InvalidOperationException(
                        $"Failed to save robot prefab at '{prefabAssetPath}'.");
                }

                return prefabAssetPath;
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(instanceRoot);
            }
        }

        public static GameObject BuildInMemory(RobotManifestData manifest)
        {
            return BuildInMemory(manifest, null, null);
        }

        public static GameObject BuildInMemory(
            RobotManifestData manifest,
            Func<ManifestMeshData, Mesh> meshResolver)
        {
            return BuildInMemory(manifest, meshResolver, meshResolver);
        }

        public static GameObject BuildInMemory(
            RobotManifestData manifest,
            Func<ManifestMeshData, Mesh> meshResolver,
            Func<ManifestMeshData, Mesh> collisionMeshResolver)
        {
            RobotManifestJson.Validate(manifest);

            var prefabRoot = new GameObject($"MicroDuck-{ToTitleCase(manifest.variant)}");
            var bodyObjects = new Dictionary<string, GameObject>(StringComparer.Ordinal);

            try
            {
                foreach (ManifestBodyData body in manifest.bodies)
                {
                    bodyObjects.Add(body.name, new GameObject(body.name));
                }

                foreach (ManifestBodyData body in manifest.bodies)
                {
                    GameObject bodyObject = bodyObjects[body.name];
                    Transform parent = string.IsNullOrEmpty(body.parentName)
                        ? prefabRoot.transform
                        : bodyObjects[body.parentName].transform;
                    bodyObject.transform.SetParent(parent, false);
                    bodyObject.transform.localPosition = Vector3From(body.localPosition);
                    bodyObject.transform.localRotation = QuaternionFromWxyz(body.localRotationWxyz);
                }

                var jointsByBody = manifest.joints.ToDictionary(
                    joint => joint.bodyName,
                    StringComparer.Ordinal);
                var servosByJoint = manifest.servos.ToDictionary(
                    servo => servo.jointName,
                    StringComparer.Ordinal);

                foreach (ManifestBodyData body in manifest.bodies)
                {
                    ArticulationBody articulation = bodyObjects[body.name].AddComponent<ArticulationBody>();
                    ManifestJointData joint = jointsByBody[body.name];
                    articulation.mass = body.mass;
                    articulation.linearDamping = 0f;
                    articulation.angularDamping = 0f;
                    articulation.solverIterations = SolverIterations;
                    articulation.solverVelocityIterations = SolverVelocityIterations;
                    articulation.automaticCenterOfMass = false;
                    articulation.centerOfMass = Vector3From(body.centerOfMass);
                    articulation.automaticInertiaTensor = false;
                    articulation.inertiaTensor =
                        ApplyTuanjiePassiveWheelArmatureCalibrationProxy(body, joint);
                    articulation.inertiaTensorRotation = QuaternionFromWxyz(body.inertiaRotationWxyz);

                    if (joint.type == "free")
                    {
                        articulation.immovable = false;
                    }
                    else
                    {
                        ConfigureHinge(
                            articulation,
                            joint,
                            servosByJoint.TryGetValue(joint.name, out ManifestServoData servo)
                                ? servo
                                : null);
                    }
                }

                AddGeometries(manifest, bodyObjects, meshResolver, collisionMeshResolver);
                AddRuntimeRig(manifest, prefabRoot, bodyObjects, jointsByBody);

                return prefabRoot;
            }
            catch
            {
                UnityEngine.Object.DestroyImmediate(prefabRoot);
                throw;
            }
        }

        private static void AddRuntimeRig(
            RobotManifestData manifest,
            GameObject prefabRoot,
            IReadOnlyDictionary<string, GameObject> bodyObjects,
            IReadOnlyDictionary<string, ManifestJointData> jointsByBody)
        {
            // Miniature manifests remain useful for isolated importer tests. A production
            // policy rig is attached only when the complete 14-action contract is present.
            if (manifest.servos.Length != PolicyContract.ActionCount)
            {
                return;
            }

            ManifestBodyData rootData = manifest.bodies.Single(
                body => string.IsNullOrEmpty(body.parentName));
            ArticulationBody rootBody = bodyObjects[rootData.name].GetComponent<ArticulationBody>();
            var jointsByName = jointsByBody.Values.ToDictionary(
                joint => joint.name,
                StringComparer.Ordinal);
            ArticulationBody[] orderedServoBodies = manifest.servos
                .OrderBy(servo => servo.index)
                .Select(servo => bodyObjects[jointsByName[servo.jointName].bodyName]
                    .GetComponent<ArticulationBody>())
                .ToArray();

            var rig = prefabRoot.AddComponent<MicroDuckRig>();
            rig.Configure(
                manifest.variant == "roller" ? RobotVariant.Roller : RobotVariant.Legged,
                rootBody,
                orderedServoBodies);
        }

        private static void AddGeometries(
            RobotManifestData manifest,
            IReadOnlyDictionary<string, GameObject> bodyObjects,
            Func<ManifestMeshData, Mesh> meshResolver,
            Func<ManifestMeshData, Mesh> collisionMeshResolver)
        {
            var meshesByName = manifest.meshes.ToDictionary(
                mesh => mesh.name,
                StringComparer.Ordinal);
            var materialsBySlidingFriction = new Dictionary<float, PhysicMaterial>();

            foreach (ManifestGeomData geom in manifest.geoms)
            {
                // MuJoCo's world floor is useful in a scene, but it is not part of a robot prefab.
                if (!bodyObjects.TryGetValue(geom.bodyName, out GameObject bodyObject))
                {
                    continue;
                }

                var geomObject = new GameObject(geom.name);
                geomObject.transform.SetParent(bodyObject.transform, false);
                geomObject.transform.localPosition = Vector3From(geom.localPosition);
                geomObject.transform.localRotation = QuaternionFromWxyz(geom.localRotationWxyz);

                Mesh resolvedMesh = null;
                Mesh resolvedCollisionMesh = null;
                if (!string.IsNullOrEmpty(geom.meshName)
                    && meshesByName.TryGetValue(geom.meshName, out ManifestMeshData meshData))
                {
                    geomObject.transform.localScale = Vector3From(meshData.scale);
                    resolvedMesh = meshResolver?.Invoke(meshData);
                    if (geom.collidable && !IsTire(geom))
                    {
                        resolvedCollisionMesh = collisionMeshResolver?.Invoke(meshData)
                            ?? resolvedMesh;
                    }
                    if (resolvedMesh != null)
                    {
                        geomObject.AddComponent<MeshFilter>().sharedMesh = resolvedMesh;
                        geomObject.AddComponent<MeshRenderer>();
                    }
                }

                if (!geom.collidable || !CollidesWithWorld(geom))
                {
                    continue;
                }

                Vector3 halfExtents = Vector3From(geom.size);
                float slidingFriction = geom.friction[0];
                if (!materialsBySlidingFriction.TryGetValue(
                    slidingFriction,
                    out PhysicMaterial physicsMaterial))
                {
                    physicsMaterial = MuJoCoPhysicsMaterialAssets.GetOrCreateForGeom(geom);
                    materialsBySlidingFriction.Add(slidingFriction, physicsMaterial);
                }

                if (IsTire(geom))
                {
                    AddTireCylinderCollider(
                        geomObject,
                        geom.name,
                        halfExtents,
                        physicsMaterial);
                }
                else if (resolvedCollisionMesh != null)
                {
                    MeshCollider collider = geomObject.AddComponent<MeshCollider>();
                    collider.convex = true;
                    collider.sharedMesh = resolvedCollisionMesh;
                    collider.contactOffset = ContactOffsetMeters;
                    collider.sharedMaterial = physicsMaterial;
                }
                else
                {
                    BoxCollider collider = geomObject.AddComponent<BoxCollider>();
                    collider.size = halfExtents * 2f;
                    collider.contactOffset = ContactOffsetMeters;
                    collider.sharedMaterial = physicsMaterial;
                }
            }
        }

        private static void AddTireCylinderCollider(
            GameObject geomObject,
            string geomName,
            Vector3 halfExtents,
            PhysicMaterial material)
        {
            Mesh cylinder = Resources.GetBuiltinResource<Mesh>("New-Cylinder.fbx");
            if (cylinder == null)
            {
                throw new InvalidOperationException("Tuanjie's built-in cylinder mesh is unavailable.");
            }

            float radius = Mathf.Max(halfExtents.x, halfExtents.y);
            float halfWidth = halfExtents.z;
            var colliderObject = new GameObject(geomName + "__cylinder");
            colliderObject.transform.SetParent(geomObject.transform, false);
            // The built-in cylinder has radius 0.5, full height 2, and a Y axis.
            // MuJoCo's compiled tire frame has radial X/Y and axial Z dimensions.
            colliderObject.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
            colliderObject.transform.localScale = new Vector3(
                radius * 2f,
                halfWidth,
                radius * 2f);
            MeshCollider collider = colliderObject.AddComponent<MeshCollider>();
            collider.sharedMesh = cylinder;
            collider.convex = true;
            collider.contactOffset = ContactOffsetMeters;
            collider.sharedMaterial = material;
        }

        private static bool IsTire(ManifestGeomData geom)
        {
            return ContainsOrdinalIgnoreCase(geom.name, "tire")
                || ContainsOrdinalIgnoreCase(geom.bodyName, "tire")
                || ContainsOrdinalIgnoreCase(geom.meshName, "tire");
        }

        private static bool CollidesWithWorld(ManifestGeomData geom)
        {
            const int worldContype = 1;
            const int worldConaffinity = 1;
            return (geom.contype & worldConaffinity) != 0
                || (worldContype & geom.conaffinity) != 0;
        }

        private static bool ContainsOrdinalIgnoreCase(string value, string expected)
        {
            return value?.IndexOf(expected, StringComparison.OrdinalIgnoreCase) >= 0;
        }

        private static Mesh ResolveMesh(
            ManifestMeshData mesh,
            string manifestAssetPath,
            bool collision)
        {
            string sourceFile = mesh.sourceFile?.Replace('\\', '/');
            string generatedAssetFile = mesh.assetFile?.Replace('\\', '/');
            string collisionAssetFile = mesh.collisionAssetFile?.Replace('\\', '/');
            string manifestFolder = Path.GetDirectoryName(manifestAssetPath)?.Replace('\\', '/');
            string generatedMeshPath = $"Assets/MicroDuck/Generated/Meshes/{mesh.name}.obj";
            string adjacentPath = string.IsNullOrEmpty(sourceFile) || string.IsNullOrEmpty(manifestFolder)
                ? null
                : $"{manifestFolder}/{sourceFile}";
            string directAssetPath = sourceFile?.StartsWith("Assets/", StringComparison.Ordinal)
                == true
                ? sourceFile
                : null;

            foreach (string candidate in new[]
            {
                collision ? collisionAssetFile : null,
                generatedAssetFile,
                generatedMeshPath,
                directAssetPath,
                adjacentPath,
            })
            {
                if (string.IsNullOrEmpty(candidate))
                {
                    continue;
                }

                Mesh resolved = AssetDatabase.LoadAllAssetsAtPath(candidate)
                    .OfType<Mesh>()
                    .FirstOrDefault();
                if (resolved != null)
                {
                    return resolved;
                }
            }

            return null;
        }

        private static void RequireAssetPath(string path, string parameterName)
        {
            if (string.IsNullOrWhiteSpace(path)
                || !path.Replace('\\', '/').StartsWith("Assets/", StringComparison.Ordinal))
            {
                throw new ArgumentException(
                    "Path must be a project-relative path below Assets/.",
                    parameterName);
            }
        }

        private static void EnsureAssetFolder(string assetFolder)
        {
            if (string.IsNullOrEmpty(assetFolder) || AssetDatabase.IsValidFolder(assetFolder))
            {
                return;
            }

            string parent = Path.GetDirectoryName(assetFolder)?.Replace('\\', '/');
            EnsureAssetFolder(parent);
            AssetDatabase.CreateFolder(parent, Path.GetFileName(assetFolder));
        }

        private static void ConfigureHinge(
            ArticulationBody articulation,
            ManifestJointData joint,
            ManifestServoData servo)
        {
            articulation.jointType = ArticulationJointType.RevoluteJoint;
            articulation.anchorPosition = Vector3From(joint.localAnchor);
            articulation.anchorRotation = Quaternion.FromToRotation(
                Vector3.right,
                Vector3From(joint.axis).normalized);
            articulation.twistLock = joint.limited
                ? ArticulationDofLock.LimitedMotion
                : ArticulationDofLock.FreeMotion;
            articulation.swingYLock = ArticulationDofLock.LockedMotion;
            articulation.swingZLock = ArticulationDofLock.LockedMotion;
            articulation.jointFriction = joint.passive
                ? PassiveWheelBearingFriction
                : joint.frictionLoss.Length > 0
                    ? joint.frictionLoss[0]
                    : 0f;

            ArticulationDrive drive = articulation.xDrive;
            drive.lowerLimit = joint.rangeRad[0] * Mathf.Rad2Deg;
            drive.upperLimit = joint.rangeRad[1] * Mathf.Rad2Deg;
            float stiffness = servo?.gainPrm?[0] ?? 0f;
            float damping = joint.damping.Length > 0 ? joint.damping[0] : 0f;
            drive.driveType = ArticulationDriveType.Force;
            drive.stiffness = stiffness;
            drive.damping = damping;

            drive.forceLimit = servo != null && servo.forceLimited
                ? Mathf.Max(Mathf.Abs(servo.forceRange[0]), Mathf.Abs(servo.forceRange[1]))
                : float.MaxValue;
            articulation.xDrive = drive;
        }

        /// <summary>
        /// Tuanjie-only calibration proxy for passive wheel armature. This is not MuJoCo
        /// physical parity: MuJoCo adds armature in generalized joint space, while this
        /// proxy adds it to one child-link principal moment. Restricting the approximation
        /// to passive wheels whose joint axis is already aligned with a principal inertia
        /// axis avoids a broad tensor approximation and prevents accidental servo changes.
        /// If the axial-only result violates a rigid body's inertia triangle inequality,
        /// PhysX rejects it; the engine-required behavior proxy then represents the added
        /// inertia as a thin wheel disk by adding half the armature to both transverse axes.
        /// </summary>
        private static Vector3 ApplyTuanjiePassiveWheelArmatureCalibrationProxy(
            ManifestBodyData body,
            ManifestJointData joint)
        {
            Vector3 inertiaTensor = Vector3From(body.inertiaTensor);
            if (!joint.passive)
            {
                return inertiaTensor;
            }

            if (joint.armature == null || joint.armature.Length == 0)
            {
                throw new InvalidOperationException(
                    $"Passive wheel joint '{joint.name}' has no armature value for the Tuanjie calibration proxy.");
            }

            float armature = joint.armature[0];
            if (float.IsNaN(armature) || float.IsInfinity(armature) || armature < 0f)
            {
                throw new InvalidOperationException(
                    $"Passive wheel joint '{joint.name}' has an invalid armature value {armature}.");
            }

            if (armature == 0f)
            {
                return inertiaTensor;
            }

            Vector3 bodyLocalJointAxis = Vector3From(joint.axis);
            if (float.IsNaN(bodyLocalJointAxis.sqrMagnitude)
                || float.IsInfinity(bodyLocalJointAxis.sqrMagnitude)
                || bodyLocalJointAxis.sqrMagnitude <= Mathf.Epsilon)
            {
                throw new InvalidOperationException(
                    $"Passive wheel joint '{joint.name}' has an invalid body-local axis.");
            }

            bodyLocalJointAxis.Normalize();
            Quaternion principalToBody = QuaternionFromWxyz(body.inertiaRotationWxyz);
            Vector3 jointAxisInPrincipalFrame =
                Quaternion.Inverse(principalToBody) * bodyLocalJointAxis;
            float xAlignment = Mathf.Abs(jointAxisInPrincipalFrame.x);
            float yAlignment = Mathf.Abs(jointAxisInPrincipalFrame.y);
            float zAlignment = Mathf.Abs(jointAxisInPrincipalFrame.z);
            int alignedPrincipalAxis = xAlignment >= yAlignment && xAlignment >= zAlignment
                ? 0
                : yAlignment >= zAlignment
                    ? 1
                    : 2;
            float alignment = Mathf.Max(xAlignment, Mathf.Max(yAlignment, zAlignment));
            if (alignment < PassiveWheelArmaturePrincipalAxisMinimumAlignment)
            {
                throw new InvalidOperationException(
                    $"Passive wheel joint '{joint.name}' is only {alignment:F6} aligned with its closest "
                    + "principal inertia axis; the Tuanjie armature calibration proxy requires "
                    + $"at least {PassiveWheelArmaturePrincipalAxisMinimumAlignment:F6} alignment.");
            }

            Vector3 calibratedInertiaTensor = inertiaTensor;
            calibratedInertiaTensor[alignedPrincipalAxis] += armature;
            if (!SatisfiesPrincipalInertiaTriangleInequalities(calibratedInertiaTensor))
            {
                float transverseDiskMoment = armature * 0.5f;
                for (int axis = 0; axis < 3; axis++)
                {
                    if (axis != alignedPrincipalAxis)
                    {
                        calibratedInertiaTensor[axis] += transverseDiskMoment;
                    }
                }
            }

            return calibratedInertiaTensor;
        }

        private static bool SatisfiesPrincipalInertiaTriangleInequalities(Vector3 inertiaTensor)
        {
            return inertiaTensor.x <= inertiaTensor.y + inertiaTensor.z
                && inertiaTensor.y <= inertiaTensor.x + inertiaTensor.z
                && inertiaTensor.z <= inertiaTensor.x + inertiaTensor.y;
        }

        private static Vector3 Vector3From(float[] values)
        {
            return new Vector3(values[0], values[1], values[2]);
        }

        private static Quaternion QuaternionFromWxyz(float[] values)
        {
            return new Quaternion(values[1], values[2], values[3], values[0]).normalized;
        }

        private static string ToTitleCase(string value)
        {
            return char.ToUpperInvariant(value[0]) + value.Substring(1);
        }
    }
}
