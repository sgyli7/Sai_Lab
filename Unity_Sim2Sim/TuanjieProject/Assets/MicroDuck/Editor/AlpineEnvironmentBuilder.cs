using System;
using System.Collections.Generic;
using AgenticRobot.MicroDuck.Mujoco;
using Mujoco;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace AgenticRobot.MicroDuck.Editor
{
    public static class AlpineEnvironmentBuilder
    {
        public const string EnvironmentRootName = "MicroDuck Alpine Environment";
        public const string MaterialFolder = "Assets/MicroDuck/Generated/Environment";

        private static readonly IReadOnlyDictionary<string, Color> Palette =
            new Dictionary<string, Color>
            {
                ["plaza"] = new Color(0.64f, 0.70f, 0.57f),
                ["rough_base"] = new Color(0.39f, 0.49f, 0.31f),
                ["stairs_light"] = new Color(0.62f, 0.60f, 0.47f),
                ["stairs_dark"] = new Color(0.49f, 0.49f, 0.39f),
                ["cobble_light"] = new Color(0.54f, 0.55f, 0.49f),
                ["cobble_dark"] = new Color(0.40f, 0.42f, 0.39f),
                ["slope"] = new Color(0.48f, 0.61f, 0.36f),
                ["slope_top"] = new Color(0.67f, 0.71f, 0.49f),
                ["slope_easy"] = new Color(0.45f, 0.67f, 0.43f),
                ["slope_medium"] = new Color(0.80f, 0.64f, 0.31f),
                ["slope_hard"] = new Color(0.77f, 0.36f, 0.22f),
                ["roller_entry"] = new Color(0.59f, 0.62f, 0.54f),
                ["roller_runout"] = new Color(0.35f, 0.50f, 0.35f),
                ["forest_floor"] = new Color(0.31f, 0.47f, 0.25f),
                ["stone"] = new Color(0.57f, 0.55f, 0.49f),
                ["boulder"] = new Color(0.39f, 0.38f, 0.35f),
                ["bridge_step"] = new Color(0.50f, 0.31f, 0.16f),
                ["bridge_deck"] = new Color(0.62f, 0.39f, 0.19f),
            };

        public static GameObject Build()
        {
            GameObject oldEnvironment = GameObject.Find(EnvironmentRootName);
            if (oldEnvironment != null)
            {
                UnityEngine.Object.DestroyImmediate(oldEnvironment);
            }

            EnsureFolder(MaterialFolder);
            ConfigureAtmosphere();

            var environment = new GameObject(EnvironmentRootName);
            var terrainRoot = new GameObject("Terrain");
            terrainRoot.transform.SetParent(environment.transform, false);
            foreach (TerrainModuleDefinition module in MujocoTerrainCatalog.Modules)
            {
                BuildTerrainModule(terrainRoot.transform, module);
            }

            var sceneryRoot = new GameObject("Scenery");
            sceneryRoot.transform.SetParent(environment.transform, false);
            BuildScenery(sceneryRoot.transform);
            CreateSun(environment.transform);
            return environment;
        }

        private static void BuildTerrainModule(
            Transform terrainRoot,
            TerrainModuleDefinition module)
        {
            var moduleObject = new GameObject(module.Id);
            moduleObject.transform.SetParent(terrainRoot, false);
            foreach (TerrainPrimitiveDefinition primitive in module.Primitives)
            {
                BuildPhysicalPrimitive(moduleObject.transform, module.Id, primitive);
            }
        }

        private static void BuildPhysicalPrimitive(
            Transform parent,
            string moduleId,
            TerrainPrimitiveDefinition primitive)
        {
            PrimitiveType visualType = primitive.Kind == TerrainPrimitiveKind.Box
                ? PrimitiveType.Cube
                : primitive.Kind == TerrainPrimitiveKind.Cylinder
                    ? PrimitiveType.Cylinder
                    : PrimitiveType.Sphere;
            GameObject item = GameObject.CreatePrimitive(visualType);
            item.name = $"terrain_{moduleId}_{primitive.Id}";
            item.transform.SetParent(parent, false);
            item.transform.position = primitive.Center;
            item.transform.rotation = Quaternion.Euler(primitive.EulerAngles);
            item.transform.localScale = primitive.Size;
            Collider collider = item.GetComponent<Collider>();
            if (collider != null)
            {
                UnityEngine.Object.DestroyImmediate(collider);
            }

            MeshRenderer renderer = item.GetComponent<MeshRenderer>();
            renderer.sharedMaterial = GetOrCreateMaterial(
                $"Terrain-{primitive.MaterialKey}",
                Palette[primitive.MaterialKey],
                smoothness: primitive.MaterialKey.StartsWith("bridge", StringComparison.Ordinal)
                    ? 0.24f
                    : 0.08f);
            renderer.shadowCastingMode = ShadowCastingMode.On;
            renderer.receiveShadows = true;

            if (!primitive.CollisionEnabled)
            {
                return;
            }

            MjGeom geom = item.AddComponent<MjGeom>();
            geom.Mass = 0f;
            geom.Density = 0f;
            switch (primitive.Kind)
            {
                case TerrainPrimitiveKind.Box:
                    geom.ShapeType = MjShapeComponent.ShapeTypes.Box;
                    geom.Box.Extents = Vector3.one * 0.5f;
                    break;
                case TerrainPrimitiveKind.Ellipsoid:
                    geom.ShapeType = MjShapeComponent.ShapeTypes.Ellipsoid;
                    geom.Ellipsoid.Radiuses = Vector3.one * 0.5f;
                    break;
                case TerrainPrimitiveKind.Cylinder:
                    geom.ShapeType = MjShapeComponent.ShapeTypes.Cylinder;
                    geom.Cylinder.Radius = 0.5f;
                    geom.Cylinder.HalfHeight = 0.5f;
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            MjGeomSettings settings = MjGeomSettings.Default;
            settings.Friction.Sliding = 1f;
            settings.Friction.Torsional = 0.005f;
            settings.Friction.Rolling = 0.0001f;
            if (moduleId == "upstream_random_grid")
            {
                settings.Solver.SolRef.TimeConst = 0.04f;
            }
            geom.Settings = settings;
        }

        private static void ConfigureAtmosphere()
        {
            Shader skyShader = Shader.Find("Skybox/Procedural");
            if (skyShader == null)
            {
                throw new InvalidOperationException("Built-in procedural skybox shader is unavailable.");
            }

            Material sky = GetOrCreateMaterial("Alpine-Skybox", Color.white, 0f, skyShader);
            sky.SetFloat("_SunSize", 0.045f);
            sky.SetFloat("_SunSizeConvergence", 7f);
            sky.SetFloat("_AtmosphereThickness", 0.68f);
            sky.SetColor("_SkyTint", new Color(0.12f, 0.38f, 1f));
            sky.SetColor("_GroundColor", new Color(0.37f, 0.43f, 0.31f));
            sky.SetFloat("_Exposure", 1.22f);
            RenderSettings.skybox = sky;
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.Linear;
            RenderSettings.fogColor = new Color(0.64f, 0.78f, 0.85f);
            RenderSettings.fogStartDistance = 22f;
            RenderSettings.fogEndDistance = 68f;
            RenderSettings.ambientMode = AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.62f, 0.76f, 0.91f);
            RenderSettings.ambientEquatorColor = new Color(0.55f, 0.61f, 0.50f);
            RenderSettings.ambientGroundColor = new Color(0.24f, 0.29f, 0.20f);
            RenderSettings.ambientIntensity = 0.82f;
            RenderSettings.reflectionIntensity = 0.55f;
        }

        private static void CreateSun(Transform parent)
        {
            var sunObject = new GameObject("Alpine Sun");
            sunObject.transform.SetParent(parent, false);
            sunObject.transform.rotation = Quaternion.Euler(38f, -32f, 0f);
            Light sun = sunObject.AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.color = new Color(1f, 0.91f, 0.75f);
            sun.intensity = 1.28f;
            sun.shadows = LightShadows.Soft;
            sun.shadowStrength = 0.72f;
            RenderSettings.sun = sun;
        }

        private static void BuildScenery(Transform root)
        {
            Material mountainNear = GetOrCreateMaterial(
                "Scenery-MountainNear",
                new Color(0.25f, 0.38f, 0.31f),
                0.03f);
            Material mountainFar = GetOrCreateMaterial(
                "Scenery-MountainFar",
                new Color(0.34f, 0.49f, 0.48f),
                0.02f);
            Vector3[] mountainPositions =
            {
                new Vector3(-25f, 5.5f, 24f),
                new Vector3(-10f, 7f, 30f),
                new Vector3(7f, 5.8f, 28f),
                new Vector3(24f, 6.6f, 23f),
                new Vector3(-31f, 4.8f, 4f),
                new Vector3(31f, 5.2f, 6f),
            };
            for (int index = 0; index < mountainPositions.Length; index++)
            {
                CreateMountain(
                    root,
                    $"Mountain {index + 1:00}",
                    mountainPositions[index],
                    7f + ((index % 3) * 2f),
                    10f + ((index % 2) * 4f),
                    index % 2 == 0 ? mountainNear : mountainFar);
            }

            Material trunk = GetOrCreateMaterial(
                "Scenery-Trunk",
                new Color(0.27f, 0.16f, 0.08f),
                0.05f);
            Material pine = GetOrCreateMaterial(
                "Scenery-Pine",
                new Color(0.13f, 0.31f, 0.19f),
                0.03f);
            Vector3[] treePositions =
            {
                new Vector3(-13f, 0f, 4f), new Vector3(-12.5f, 0f, 13f),
                new Vector3(-4.5f, 0f, 14f), new Vector3(5f, 0f, 14.5f),
                new Vector3(13f, 0f, 13f), new Vector3(13f, 0f, 4f),
                new Vector3(-12f, 0f, -5f), new Vector3(12f, 0f, -5f),
            };
            for (int index = 0; index < treePositions.Length; index++)
            {
                CreatePine(
                    root,
                    $"Pine {index + 1:00}",
                    treePositions[index],
                    1.3f + ((index % 3) * 0.24f),
                    trunk,
                    pine);
            }

            Material cloud = GetOrCreateMaterial(
                "Scenery-Cloud",
                new Color(0.93f, 0.96f, 1f),
                0.15f);
            CreateCloud(root, "Cloud A", new Vector3(-12f, 10f, 22f), cloud);
            CreateCloud(root, "Cloud B", new Vector3(15f, 12f, 27f), cloud);
        }

        private static void CreateMountain(
            Transform parent,
            string name,
            Vector3 position,
            float radius,
            float height,
            Material material)
        {
            var mountain = new GameObject(name);
            mountain.transform.SetParent(parent, false);
            mountain.transform.position = position;
            var filter = mountain.AddComponent<MeshFilter>();
            filter.sharedMesh = CreateConeMesh(name + " Mesh", 14, radius, height);
            mountain.AddComponent<MeshRenderer>().sharedMaterial = material;
        }

        private static void CreatePine(
            Transform parent,
            string name,
            Vector3 position,
            float height,
            Material trunkMaterial,
            Material pineMaterial)
        {
            var tree = new GameObject(name);
            tree.transform.SetParent(parent, false);
            tree.transform.position = position;
            CreateVisualPrimitive(
                tree.transform,
                "Trunk",
                PrimitiveType.Cylinder,
                new Vector3(0f, height * 0.22f, 0f),
                new Vector3(0.12f, height * 0.44f, 0.12f),
                trunkMaterial);
            GameObject foliage = new GameObject("Foliage");
            foliage.transform.SetParent(tree.transform, false);
            foliage.transform.localPosition = new Vector3(0f, height * 0.45f, 0f);
            MeshFilter filter = foliage.AddComponent<MeshFilter>();
            filter.sharedMesh = CreateConeMesh(name + " Foliage Mesh", 10, height * 0.32f, height * 0.82f);
            foliage.AddComponent<MeshRenderer>().sharedMaterial = pineMaterial;
        }

        private static void CreateCloud(
            Transform parent,
            string name,
            Vector3 position,
            Material material)
        {
            var cloud = new GameObject(name);
            cloud.transform.SetParent(parent, false);
            cloud.transform.position = position;
            CreateVisualPrimitive(cloud.transform, "Left", PrimitiveType.Sphere, new Vector3(-1.1f, 0f, 0f), new Vector3(2.5f, 1.2f, 1.3f), material);
            CreateVisualPrimitive(cloud.transform, "Center", PrimitiveType.Sphere, new Vector3(0f, 0.35f, 0f), new Vector3(3.1f, 1.7f, 1.6f), material);
            CreateVisualPrimitive(cloud.transform, "Right", PrimitiveType.Sphere, new Vector3(1.2f, 0f, 0f), new Vector3(2.2f, 1.1f, 1.2f), material);
        }

        private static GameObject CreateVisualPrimitive(
            Transform parent,
            string name,
            PrimitiveType type,
            Vector3 localPosition,
            Vector3 localScale,
            Material material)
        {
            GameObject item = GameObject.CreatePrimitive(type);
            item.name = name;
            item.transform.SetParent(parent, false);
            item.transform.localPosition = localPosition;
            item.transform.localScale = localScale;
            Collider collider = item.GetComponent<Collider>();
            if (collider != null)
            {
                UnityEngine.Object.DestroyImmediate(collider);
            }
            item.GetComponent<MeshRenderer>().sharedMaterial = material;
            return item;
        }

        private static Mesh CreateConeMesh(string name, int segments, float radius, float height)
        {
            var vertices = new Vector3[segments + 2];
            var triangles = new int[segments * 6];
            vertices[0] = Vector3.zero;
            vertices[1] = new Vector3(0f, height, 0f);
            for (int index = 0; index < segments; index++)
            {
                float angle = index * Mathf.PI * 2f / segments;
                vertices[index + 2] = new Vector3(
                    Mathf.Cos(angle) * radius,
                    0f,
                    Mathf.Sin(angle) * radius);
                int next = ((index + 1) % segments) + 2;
                int offset = index * 6;
                triangles[offset] = 0;
                triangles[offset + 1] = next;
                triangles[offset + 2] = index + 2;
                triangles[offset + 3] = 1;
                triangles[offset + 4] = index + 2;
                triangles[offset + 5] = next;
            }

            var mesh = new Mesh { name = name };
            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static Material GetOrCreateMaterial(
            string name,
            Color color,
            float smoothness,
            Shader shader = null)
        {
            string path = $"{MaterialFolder}/{name}.mat";
            Material material = AssetDatabase.LoadAssetAtPath<Material>(path);
            Shader selectedShader = shader ?? Shader.Find("Standard");
            if (selectedShader == null)
            {
                throw new InvalidOperationException($"Shader for material '{name}' is unavailable.");
            }

            if (material == null)
            {
                material = new Material(selectedShader) { name = name };
                AssetDatabase.CreateAsset(material, path);
            }
            else if (material.shader != selectedShader)
            {
                material.shader = selectedShader;
            }

            if (material.HasProperty("_Color"))
            {
                material.color = color;
            }
            if (material.HasProperty("_Glossiness"))
            {
                material.SetFloat("_Glossiness", smoothness);
            }
            EditorUtility.SetDirty(material);
            return material;
        }

        private static void EnsureFolder(string path)
        {
            if (AssetDatabase.IsValidFolder(path))
            {
                return;
            }

            string parent = System.IO.Path.GetDirectoryName(path)?.Replace('\\', '/');
            if (string.IsNullOrEmpty(parent))
            {
                throw new InvalidOperationException($"Cannot create asset folder '{path}'.");
            }
            EnsureFolder(parent);
            AssetDatabase.CreateFolder(parent, System.IO.Path.GetFileName(path));
        }
    }
}
