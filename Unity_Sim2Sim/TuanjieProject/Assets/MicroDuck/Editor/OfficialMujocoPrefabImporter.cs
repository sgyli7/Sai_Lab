using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Mujoco;
using UnityEditor;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Editor
{
    /// <summary>
    /// Imports the two upstream MicroDuck MJCF scenes through the official MuJoCo Unity importer.
    /// Native MuJoCo first expands every include and compiler default. The official importer then
    /// creates the component hierarchy and converts the referenced STL files into Unity assets.
    /// </summary>
    public static class OfficialMujocoPrefabImporter
    {
        public const string GeneratedRootAssetPath =
            "Assets/MicroDuck/Generated/MuJoCo";
        public const string LeggedResourcesAssetPath =
            GeneratedRootAssetPath + "/Legged/Resources";
        public const string RollerResourcesAssetPath =
            GeneratedRootAssetPath + "/Roller/Resources";
        public const string BallResourcesAssetPath =
            GeneratedRootAssetPath + "/Ball/Resources";
        public const string LeggedPrefabAssetPath =
            GeneratedRootAssetPath + "/Legged/MicroDuck-MuJoCo-Legged.prefab";
        public const string RollerPrefabAssetPath =
            GeneratedRootAssetPath + "/Roller/MicroDuck-MuJoCo-Roller.prefab";
        public const string BallPrefabAssetPath =
            GeneratedRootAssetPath + "/Ball/MicroDuck-MuJoCo-Ball.prefab";
        public const string LeggedExpandedXmlAssetPath =
            GeneratedRootAssetPath + "/Legged/scene.expanded.xml";
        public const string RollerExpandedXmlAssetPath =
            GeneratedRootAssetPath + "/Roller/scene_rollers.expanded.xml";
        public const string BallExpandedXmlAssetPath =
            GeneratedRootAssetPath + "/Ball/scene_ball.expanded.xml";

        private const string LeggedStagingName = "MicroDuckOfficialLeggedImport";
        private const string RollerStagingName = "MicroDuckOfficialRollerImport";
        private const string BallStagingName = "MicroDuckOfficialBallImport";
        public const string LeggedStagingAssetPath =
            "Assets/Local/MjImports/" + LeggedStagingName;
        public const string RollerStagingAssetPath =
            "Assets/Local/MjImports/" + RollerStagingName;
        public const string BallStagingAssetPath =
            "Assets/Local/MjImports/" + BallStagingName;

        private const string UpstreamRobotFolder =
            ".cache/upstream/microduck_rl/src/mjlab_microduck/robot/microduck";

        [MenuItem("MicroDuck/Import Official MuJoCo Prefabs")]
        public static IReadOnlyList<string> ImportAll()
        {
            string repositoryRoot = GetRepositoryRoot();
            string sourceFolder = Path.Combine(
                repositoryRoot,
                UpstreamRobotFolder.Replace('/', Path.DirectorySeparatorChar));

            string legged = ImportVariant(
                Path.Combine(sourceFolder, "scene.xml"),
                LeggedStagingName,
                LeggedStagingAssetPath,
                "MicroDuck-MuJoCo-Legged",
                LeggedResourcesAssetPath,
                LeggedPrefabAssetPath,
                LeggedExpandedXmlAssetPath);
            string roller = ImportVariant(
                Path.Combine(sourceFolder, "scene_rollers.xml"),
                RollerStagingName,
                RollerStagingAssetPath,
                "MicroDuck-MuJoCo-Roller",
                RollerResourcesAssetPath,
                RollerPrefabAssetPath,
                RollerExpandedXmlAssetPath);
            string ball = ImportVariant(
                Path.Combine(sourceFolder, "scene_ball.xml"),
                BallStagingName,
                BallStagingAssetPath,
                "MicroDuck-MuJoCo-Ball",
                BallResourcesAssetPath,
                BallPrefabAssetPath,
                BallExpandedXmlAssetPath);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            return new[] { legged, roller, ball };
        }

        private static unsafe string ImportVariant(
            string sourceScenePath,
            string stagingName,
            string stagingAssetPath,
            string rootName,
            string resourcesAssetPath,
            string prefabAssetPath,
            string expandedXmlAssetPath)
        {
            if (!File.Exists(sourceScenePath))
            {
                throw new FileNotFoundException(
                    $"Official MicroDuck MJCF scene was not found at '{sourceScenePath}'.",
                    sourceScenePath);
            }

            string expandedXml = ExpandMjcf(sourceScenePath, stagingName);
            expandedXml = MakeImplicitJointAxesExplicit(expandedXml);
            SaveExpandedXml(expandedXmlAssetPath, expandedXml);

            DeleteAssetIfPresent(stagingAssetPath);
            GameObject importedRoot = null;
            try
            {
                var importer = new MjImporterWithAssets();
                importedRoot = importer.ImportString(expandedXml, stagingName, sourceScenePath);
                if (importedRoot == null)
                {
                    throw new InvalidOperationException(
                        $"The official MuJoCo importer could not import '{sourceScenePath}'.");
                }

                importedRoot.name = rootName;
                RemovePerModelGlobalSettings(importedRoot);
                DisableImportedSensorCameras(importedRoot);
                string stagingResourcesPath = stagingAssetPath + "/Resources";
                SynchronizeResources(stagingResourcesPath, resourcesAssetPath);
                RebindPersistentResources(importedRoot, resourcesAssetPath);

                EnsureAssetFolder(GetAssetDirectory(prefabAssetPath));
                GameObject prefab = PrefabUtility.SaveAsPrefabAsset(
                    importedRoot,
                    prefabAssetPath,
                    out bool savedSuccessfully);
                if (!savedSuccessfully || prefab == null)
                {
                    throw new InvalidOperationException(
                        $"Could not save official MuJoCo prefab at '{prefabAssetPath}'.");
                }

                return prefabAssetPath;
            }
            finally
            {
                if (importedRoot != null)
                {
                    UnityEngine.Object.DestroyImmediate(importedRoot);
                }

                DeleteAssetIfPresent(stagingAssetPath);
                AssetDatabase.SaveAssets();
            }
        }

        private static unsafe string ExpandMjcf(string sourceScenePath, string variantName)
        {
            string temporaryFolder = Path.Combine(
                Application.temporaryCachePath,
                "MicroDuckOfficialImports");
            Directory.CreateDirectory(temporaryFolder);
            string expandedPath = Path.Combine(temporaryFolder, variantName + ".expanded.xml");

            MujocoLib.mjModel_* model = null;
            try
            {
                MjEngineTool.LoadPlugins();
                model = MjEngineTool.LoadModelFromFile(sourceScenePath);
                if (model == null)
                {
                    throw new InvalidOperationException(
                        $"Native MuJoCo could not compile '{sourceScenePath}'.");
                }

                MjEngineTool.SaveModelToFile(expandedPath, model);
                return File.ReadAllText(expandedPath);
            }
            finally
            {
                if (model != null)
                {
                    MujocoLib.mj_deleteModel(model);
                }
            }
        }

        private static string MakeImplicitJointAxesExplicit(string expandedXml)
        {
            // MuJoCo's MJCF default joint axis is +Z. mj_saveLastXML legitimately
            // omits axis="0 0 1" when it serializes a compiled model, but the
            // MuJoCo Unity importer treats an omitted axis as +X. Make the MJCF
            // default explicit before importing so the generated component
            // hierarchy round-trips to the same native model as the source.
            var document = new System.Xml.XmlDocument();
            document.LoadXml(expandedXml);
            System.Xml.XmlNodeList joints = document.SelectNodes("/mujoco/worldbody//joint");
            foreach (System.Xml.XmlElement joint in joints)
            {
                string type = joint.GetAttribute("type");
                bool hasDirectionalAxis = string.IsNullOrEmpty(type)
                    || type == "hinge"
                    || type == "slide";
                if (hasDirectionalAxis && !joint.HasAttribute("axis"))
                {
                    joint.SetAttribute("axis", "0 0 1");
                }
            }

            using (var writer = new StringWriter())
            {
                document.Save(writer);
                return writer.ToString();
            }
        }

        private static void SaveExpandedXml(string assetPath, string expandedXml)
        {
            EnsureAssetFolder(GetAssetDirectory(assetPath));
            File.WriteAllText(GetFullAssetPath(assetPath), expandedXml, new UTF8Encoding(false));
            AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate);
        }

        private static void SynchronizeResources(
            string stagingResourcesPath,
            string destinationResourcesPath)
        {
            if (!AssetDatabase.IsValidFolder(stagingResourcesPath))
            {
                throw new DirectoryNotFoundException(
                    $"Official MuJoCo importer did not create '{stagingResourcesPath}'.");
            }

            EnsureAssetFolder(destinationResourcesPath);
            string[] stagingAssets = FindFiles(stagingResourcesPath);
            var expectedNames = new HashSet<string>(
                stagingAssets.Select(Path.GetFileName),
                StringComparer.OrdinalIgnoreCase);

            foreach (string staleAsset in FindFiles(destinationResourcesPath))
            {
                if (!expectedNames.Contains(Path.GetFileName(staleAsset)))
                {
                    if (!AssetDatabase.DeleteAsset(staleAsset))
                    {
                        throw new IOException($"Could not remove stale generated asset '{staleAsset}'.");
                    }
                }
            }

            foreach (string stagingAsset in stagingAssets)
            {
                string destinationAsset = NormalizeAssetPath(
                    destinationResourcesPath + "/" + Path.GetFileName(stagingAsset));
                SynchronizeAsset(stagingAsset, destinationAsset);
            }

            AssetDatabase.SaveAssets();
        }

        private static void SynchronizeAsset(string sourceAsset, string destinationAsset)
        {
            UnityEngine.Object source = AssetDatabase.LoadMainAssetAtPath(sourceAsset);
            UnityEngine.Object destination = AssetDatabase.LoadMainAssetAtPath(destinationAsset);
            if (destination == null)
            {
                if (!AssetDatabase.CopyAsset(sourceAsset, destinationAsset))
                {
                    throw new IOException(
                        $"Could not copy generated asset '{sourceAsset}' to '{destinationAsset}'.");
                }

                return;
            }

            if (source is Material sourceMaterial && destination is Material destinationMaterial)
            {
                EditorUtility.CopySerialized(sourceMaterial, destinationMaterial);
                EditorUtility.SetDirty(destinationMaterial);
                return;
            }

            string sourceFullPath = GetFullAssetPath(sourceAsset);
            string destinationFullPath = GetFullAssetPath(destinationAsset);
            File.Copy(sourceFullPath, destinationFullPath, overwrite: true);
            AssetDatabase.ImportAsset(destinationAsset, ImportAssetOptions.ForceUpdate);

            ModelImporter importer = AssetImporter.GetAtPath(destinationAsset) as ModelImporter;
            if (importer != null && !importer.isReadable)
            {
                importer.isReadable = true;
                importer.SaveAndReimport();
            }
        }

        private static void RebindPersistentResources(
            GameObject importedRoot,
            string resourcesAssetPath)
        {
            foreach (MjGeom geom in importedRoot.GetComponentsInChildren<MjGeom>(true))
            {
                if (geom.ShapeType != MjShapeComponent.ShapeTypes.Mesh || geom.Mesh.Mesh == null)
                {
                    continue;
                }

                string sourcePath = AssetDatabase.GetAssetPath(geom.Mesh.Mesh);
                string destinationPath = NormalizeAssetPath(
                    resourcesAssetPath + "/" + Path.GetFileName(sourcePath));
                Mesh persistentMesh = AssetDatabase.LoadAssetAtPath<Mesh>(destinationPath);
                if (persistentMesh == null)
                {
                    throw new InvalidOperationException(
                        $"Persistent mesh for geom '{geom.name}' was not found at '{destinationPath}'.");
                }

                geom.Mesh.Mesh = persistentMesh;
                EditorUtility.SetDirty(geom);
            }

            foreach (MeshRenderer renderer in importedRoot.GetComponentsInChildren<MeshRenderer>(true))
            {
                Material sourceMaterial = renderer.sharedMaterial;
                if (sourceMaterial == null)
                {
                    throw new InvalidOperationException(
                        $"Official MuJoCo renderer '{renderer.name}' has no material.");
                }

                string sourcePath = AssetDatabase.GetAssetPath(sourceMaterial);
                string destinationPath = NormalizeAssetPath(
                    resourcesAssetPath + "/" + Path.GetFileName(sourcePath));
                Material persistentMaterial = AssetDatabase.LoadAssetAtPath<Material>(destinationPath);
                if (persistentMaterial == null)
                {
                    throw new InvalidOperationException(
                        $"Persistent material for renderer '{renderer.name}' was not found at "
                        + $"'{destinationPath}'.");
                }

                renderer.sharedMaterial = persistentMaterial;
                EditorUtility.SetDirty(renderer);
            }
        }

        private static void RemovePerModelGlobalSettings(GameObject importedRoot)
        {
            // A playable scene owns exactly one global settings component. Keeping
            // one in every variant prefab would create competing singletons when a
            // legged/roller switch enables a different imported hierarchy.
            foreach (MjGlobalSettings settings in
                importedRoot.GetComponentsInChildren<MjGlobalSettings>(true))
            {
                UnityEngine.Object.DestroyImmediate(settings.gameObject);
            }
        }

        private static void DisableImportedSensorCameras(GameObject importedRoot)
        {
            // MJCF cameras are observation sensors, not display cameras. The
            // official importer creates enabled Unity Cameras for them; if left
            // enabled they render after the playable camera and replace the
            // window with the view from inside MicroDuck's head.
            foreach (Camera camera in importedRoot.GetComponentsInChildren<Camera>(true))
            {
                camera.enabled = false;
                EditorUtility.SetDirty(camera);
            }
        }

        private static string[] FindFiles(string assetFolder)
        {
            return AssetDatabase.FindAssets(string.Empty, new[] { assetFolder })
                .Select(AssetDatabase.GUIDToAssetPath)
                .Where(path => !string.IsNullOrEmpty(path) && File.Exists(GetFullAssetPath(path)))
                .Select(NormalizeAssetPath)
                .OrderBy(path => path, StringComparer.Ordinal)
                .ToArray();
        }

        private static void DeleteAssetIfPresent(string assetPath)
        {
            if ((AssetDatabase.IsValidFolder(assetPath)
                    || AssetDatabase.LoadMainAssetAtPath(assetPath) != null)
                && !AssetDatabase.DeleteAsset(assetPath))
            {
                throw new IOException($"Could not clear generated staging asset '{assetPath}'.");
            }
        }

        private static string GetRepositoryRoot()
        {
            return Path.GetFullPath(Path.Combine(Application.dataPath, "..", ".."));
        }

        private static string GetFullAssetPath(string assetPath)
        {
            return Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", assetPath.Replace('/', Path.DirectorySeparatorChar)));
        }

        private static string GetAssetDirectory(string assetPath)
        {
            return NormalizeAssetPath(Path.GetDirectoryName(assetPath));
        }

        private static string NormalizeAssetPath(string assetPath)
        {
            return assetPath?.Replace('\\', '/');
        }

        private static void EnsureAssetFolder(string assetFolder)
        {
            if (string.IsNullOrEmpty(assetFolder) || AssetDatabase.IsValidFolder(assetFolder))
            {
                return;
            }

            string parent = GetAssetDirectory(assetFolder);
            EnsureAssetFolder(parent);
            AssetDatabase.CreateFolder(parent, Path.GetFileName(assetFolder));
        }
    }
}
