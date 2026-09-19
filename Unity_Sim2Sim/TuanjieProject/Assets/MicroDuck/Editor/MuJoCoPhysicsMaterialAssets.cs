using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Editor
{
    /// <summary>
    /// Owns the persistent PhysX approximation of the two MuJoCo contact materials
    /// used by the MVP. The manifest retains all three MuJoCo coefficients; PhysX's
    /// PhysicMaterial can represent only the first (sliding-friction) coefficient.
    /// </summary>
    public static class MuJoCoPhysicsMaterialAssets
    {
        public const string RobotAndFloorAssetPath =
            "Assets/MicroDuck/Generated/PhysicsMaterials/MuJoCo-RobotAndFloor.physicMaterial";

        public const string BallAssetPath =
            "Assets/MicroDuck/Generated/PhysicsMaterials/MuJoCo-Ball.physicMaterial";

        private const float RobotAndFloorFriction = 1f;
        private const float BallFriction = 0.5f;
        private const float CoefficientTolerance = 1e-6f;

        public static PhysicMaterial GetOrCreateRobotAndFloor()
        {
            return GetOrCreate(RobotAndFloorAssetPath, RobotAndFloorFriction);
        }

        public static PhysicMaterial GetOrCreateBall()
        {
            return GetOrCreate(BallAssetPath, BallFriction);
        }

        public static PhysicMaterial GetOrCreateForGeom(ManifestGeomData geom)
        {
            if (geom == null || geom.friction == null || geom.friction.Length != 3)
            {
                throw new ArgumentException("A geom with a three-value friction vector is required.", nameof(geom));
            }

            float slidingFriction = geom.friction[0];
            if (Mathf.Abs(slidingFriction - RobotAndFloorFriction) <= CoefficientTolerance)
            {
                return GetOrCreateRobotAndFloor();
            }

            if (Mathf.Abs(slidingFriction - BallFriction) <= CoefficientTolerance)
            {
                return GetOrCreateBall();
            }

            throw new InvalidOperationException(
                $"Geom '{geom.name}' uses unsupported MuJoCo sliding friction {slidingFriction:R}; "
                + "add an explicit persistent PhysicMaterial mapping before importing it.");
        }

        private static PhysicMaterial GetOrCreate(string assetPath, float friction)
        {
            EnsureAssetFolder(Path.GetDirectoryName(assetPath)?.Replace('\\', '/'));

            UnityEngine.Object existingAsset = AssetDatabase.LoadMainAssetAtPath(assetPath);
            if (existingAsset != null && !(existingAsset is PhysicMaterial))
            {
                throw new InvalidOperationException(
                    $"Asset '{assetPath}' exists but is not a PhysicMaterial.");
            }

            var material = existingAsset as PhysicMaterial;
            if (material == null)
            {
                material = new PhysicMaterial(Path.GetFileNameWithoutExtension(assetPath));
                AssetDatabase.CreateAsset(material, assetPath);
            }

            material.staticFriction = friction;
            material.dynamicFriction = friction;
            material.bounciness = 0f;
            material.frictionCombine = PhysicMaterialCombine.Maximum;
            material.bounceCombine = PhysicMaterialCombine.Minimum;
            EditorUtility.SetDirty(material);
            AssetDatabase.SaveAssets();
            return material;
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
    }
}
