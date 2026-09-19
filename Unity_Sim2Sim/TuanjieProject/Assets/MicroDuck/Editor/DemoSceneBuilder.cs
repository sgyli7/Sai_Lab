using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using AgenticRobot.MicroDuck.Mujoco;
using Mujoco;
using Unity.Barracuda;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Editor
{
    public static class DemoSceneBuilder
    {
        public const string SceneAssetPath =
            "Assets/MicroDuck/Generated/Scenes/MicroDuckMvp.unity";
        public const string NativeSceneAssetPath =
            "Assets/MicroDuck/Generated/Scenes/MicroDuckNativeMvp.unity";

        public static string CreateAllSceneAssets()
        {
            CreateSceneAsset();
            return CreateNativeSceneAsset();
        }

        [MenuItem("MicroDuck/Create MVP Demo Scene")]
        public static string CreateSceneAsset()
        {
            ConfigureSimulationTiming();
            ConfigureContactOffset();
            RobotPrefabImporter.ImportAll();
            EnsureAssetFolder(Path.GetDirectoryName(SceneAssetPath)?.Replace('\\', '/'));

            var scene = EditorSceneManager.NewScene(
                NewSceneSetup.EmptyScene,
                NewSceneMode.Single);

            GameObject leggedObject = InstantiateRobot(
                RobotPrefabImporter.LeggedPrefabAssetPath,
                new Vector3(-0.45f, 0f, 0f),
                0.125f);
            GameObject rollerObject = InstantiateRobot(
                RobotPrefabImporter.RollerPrefabAssetPath,
                new Vector3(0.45f, 0f, 0f),
                0.1385f);

            MicroDuckRig leggedRig = RequireRig(leggedObject, RobotVariant.Legged);
            MicroDuckRig rollerRig = RequireRig(rollerObject, RobotVariant.Roller);
            Collider floorCollider = CreateFloor(
                MuJoCoPhysicsMaterialAssets.GetOrCreateRobotAndFloor());
            MicroDuckSkillBall skillBall = CreateSkillBall(
                floorCollider,
                MuJoCoPhysicsMaterialAssets.GetOrCreateBall());

            var runtimeObject = new GameObject("MicroDuck Runtime");
            MicroDuckDemoController controller = runtimeObject.AddComponent<MicroDuckDemoController>();
            controller.Configure(leggedRig, rollerRig, LoadPolicyBindings(), skillBall);
            runtimeObject.AddComponent<KeyboardPolicyInput>().Configure(controller);
            runtimeObject.AddComponent<PolicyStatusOverlay>().Configure(controller);

            CreateLight();
            CreateCamera();

            if (!EditorSceneManager.SaveScene(scene, SceneAssetPath))
            {
                throw new InvalidOperationException($"Could not save MVP scene at '{SceneAssetPath}'.");
            }

            AssetDatabase.SaveAssets();
            return SceneAssetPath;
        }

        [MenuItem("MicroDuck/Create Native MuJoCo MVP Scene")]
        public static string CreateNativeSceneAsset()
        {
            ConfigureSimulationTiming();
            OfficialMujocoPrefabImporter.ImportAll();
            EnsureAssetFolder(Path.GetDirectoryName(NativeSceneAssetPath)?.Replace('\\', '/'));

            var nativeScene = EditorSceneManager.NewScene(
                NewSceneSetup.EmptyScene,
                NewSceneMode.Single);

            // MjComponent.OnEnable resolves the official MjScene singleton. Keep
            // this root first in the serialized scene so its Awake runs before any
            // imported body is enabled when the scene is loaded in Play Mode.
            var runtimeObject = new GameObject("MicroDuck Native Runtime");
            MjScene mujocoScene = runtimeObject.AddComponent<MicroDuckMjScene>();
            MujocoDemoController controller = runtimeObject.AddComponent<MujocoDemoController>();
            MjGlobalSettings globalSettings = runtimeObject.AddComponent<MjGlobalSettings>();
            globalSettings.UseRawGameObjectNames = true;

            AlpineEnvironmentBuilder.Build();

            GameObject leggedWithBall = InstantiateMujocoModel(
                OfficialMujocoPrefabImporter.BallPrefabAssetPath,
                active: true);
            GameObject roller = InstantiateMujocoModel(
                OfficialMujocoPrefabImporter.RollerPrefabAssetPath,
                active: false);
            DisableImportedFloor(leggedWithBall);
            DisableImportedFloor(roller);

            controller.Configure(mujocoScene, leggedWithBall, roller, LoadPolicyBindings());
            MujocoTerrainNavigator navigator =
                runtimeObject.AddComponent<MujocoTerrainNavigator>();
            navigator.Configure(controller);
            MujocoPolicyStatusOverlay overlay =
                runtimeObject.AddComponent<MujocoPolicyStatusOverlay>();
            runtimeObject.AddComponent<MujocoPlayerSmokeProbe>().Configure(controller);
            MujocoPlayerAcceptanceTour tour =
                runtimeObject.AddComponent<MujocoPlayerAcceptanceTour>();

            Camera camera = CreateCamera();
            camera.fieldOfView = 35f;
            camera.nearClipPlane = 0.03f;
            MujocoCameraRig cameraRig = camera.gameObject.AddComponent<MujocoCameraRig>();
            cameraRig.Configure(controller);
            overlay.Configure(controller, navigator, cameraRig);
            runtimeObject.AddComponent<MujocoKeyboardPolicyInput>()
                .Configure(controller, navigator, cameraRig);
            tour.Configure(controller, navigator, cameraRig);

            if (!EditorSceneManager.SaveScene(nativeScene, NativeSceneAssetPath))
            {
                throw new InvalidOperationException(
                    $"Could not save native MuJoCo MVP scene at '{NativeSceneAssetPath}'.");
            }

            AssetDatabase.SaveAssets();
            return NativeSceneAssetPath;
        }

        private static void ConfigureSimulationTiming()
        {
            const float physicsStepSeconds = 0.005f;
            UnityEngine.Object[] settingsAssets = AssetDatabase.LoadAllAssetsAtPath(
                "ProjectSettings/TimeManager.asset");
            if (settingsAssets.Length != 1)
            {
                throw new InvalidOperationException(
                    "Could not load the project's TimeManager settings asset.");
            }

            var serializedSettings = new SerializedObject(settingsAssets[0]);
            SerializedProperty fixedTimestep = serializedSettings.FindProperty("Fixed Timestep");
            if (fixedTimestep == null)
            {
                throw new InvalidOperationException(
                    "TimeManager does not expose the Fixed Timestep setting.");
            }

            fixedTimestep.floatValue = physicsStepSeconds;
            serializedSettings.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(settingsAssets[0]);
            Time.fixedDeltaTime = physicsStepSeconds;
            AssetDatabase.SaveAssets();
        }

        private static void ConfigureContactOffset()
        {
            UnityEngine.Object[] settingsAssets = AssetDatabase.LoadAllAssetsAtPath(
                "ProjectSettings/DynamicsManager.asset");
            if (settingsAssets.Length != 1)
            {
                throw new InvalidOperationException(
                    "Could not load the project's 3D physics settings asset.");
            }

            var serializedSettings = new SerializedObject(settingsAssets[0]);
            SerializedProperty contactOffset = serializedSettings.FindProperty("m_DefaultContactOffset");
            if (contactOffset == null)
            {
                throw new InvalidOperationException(
                    "3D physics settings do not expose the default contact offset.");
            }

            contactOffset.floatValue = RobotPrefabImporter.ContactOffsetMeters;
            serializedSettings.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(settingsAssets[0]);
            Physics.defaultContactOffset = RobotPrefabImporter.ContactOffsetMeters;
            AssetDatabase.SaveAssets();
        }

        public static void ConfigureBuildSettings()
        {
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(NativeSceneAssetPath) == null)
            {
                throw new FileNotFoundException(
                    $"Native MuJoCo MVP scene was not found at '{NativeSceneAssetPath}'.",
                    NativeSceneAssetPath);
            }
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(SceneAssetPath) == null)
            {
                throw new FileNotFoundException(
                    $"PhysX calibration scene was not found at '{SceneAssetPath}'.",
                    SceneAssetPath);
            }

            EditorBuildSettings.scenes = new[]
            {
                new EditorBuildSettingsScene(NativeSceneAssetPath, true),
                new EditorBuildSettingsScene(SceneAssetPath, true),
            };
        }

        public static BuildPlayerOptions CreateWindows64BuildOptions()
        {
            string repositoryRoot = Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", ".."));
            string executablePath = Path.Combine(
                repositoryRoot,
                "Builds",
                "Windows64",
                "AgenticRobotGame.exe");

            return new BuildPlayerOptions
            {
                scenes = new[] { NativeSceneAssetPath },
                locationPathName = executablePath,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None,
            };
        }

        [MenuItem("MicroDuck/Build Windows x64 MVP")]
        public static void BuildWindows64()
        {
            CreateAllSceneAssets();
            ConfigureBuildSettings();
            BuildPlayerOptions options = CreateWindows64BuildOptions();
            string outputDirectory = Path.GetDirectoryName(options.locationPathName);
            if (string.IsNullOrEmpty(outputDirectory))
            {
                throw new InvalidOperationException("Windows build output directory is empty.");
            }

            Directory.CreateDirectory(outputDirectory);
            BuildReport report = BuildPipeline.BuildPlayer(options);
            if (report.summary.result != BuildResult.Succeeded)
            {
                throw new InvalidOperationException(
                    $"Windows x64 build failed with result {report.summary.result} "
                    + $"and {report.summary.totalErrors} errors.");
            }

            Debug.Log(
                $"Built MicroDuck MVP for Windows x64 at '{options.locationPathName}' "
                + $"({report.summary.totalSize} bytes)." );
        }

        public static BuildPlayerOptions CreateMacOSBuildOptions()
        {
            string repositoryRoot = Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", ".."));
            string appPath = Path.Combine(
                repositoryRoot,
                "Builds",
                "macOS",
                "AgenticRobotGame.app");

            return new BuildPlayerOptions
            {
                scenes = new[] { NativeSceneAssetPath },
                locationPathName = appPath,
                target = BuildTarget.StandaloneOSX,
                options = BuildOptions.None,
            };
        }

        [MenuItem("MicroDuck/Build macOS MVP")]
        public static void BuildMacOS()
        {
            CreateAllSceneAssets();
            ConfigureBuildSettings();
            string architecture = RuntimeInformation.ProcessArchitecture == Architecture.Arm64
                ? "ARM64"
                : "x64";
            EditorUserBuildSettings.SetPlatformSettings(
                "OSXUniversal",
                "Architecture",
                architecture);
            string previousProductName = PlayerSettings.productName;
            PlayerSettings.productName = "AgenticRobotGame";
            try
            {
                BuildPlayerOptions options = CreateMacOSBuildOptions();
                string outputDirectory = Path.GetDirectoryName(options.locationPathName);
                if (string.IsNullOrEmpty(outputDirectory))
                {
                    throw new InvalidOperationException("macOS build output directory is empty.");
                }

                Directory.CreateDirectory(outputDirectory);
                BuildReport report = BuildPipeline.BuildPlayer(options);
                if (report.summary.result != BuildResult.Succeeded)
                {
                    throw new InvalidOperationException(
                        $"macOS build failed with result {report.summary.result} "
                        + $"and {report.summary.totalErrors} errors.");
                }

                Debug.Log(
                    $"Built MicroDuck MVP for macOS at '{options.locationPathName}' "
                    + $"({report.summary.totalSize} bytes)." );
            }
            finally
            {
                PlayerSettings.productName = previousProductName;
            }
        }

        private static GameObject InstantiateRobot(
            string assetPath,
            Vector3 position,
            float rootHeightMeters)
        {
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(assetPath);
            if (prefab == null)
            {
                throw new FileNotFoundException($"Robot prefab was not found at '{assetPath}'.", assetPath);
            }

            var instance = PrefabUtility.InstantiatePrefab(prefab) as GameObject;
            if (instance == null)
            {
                throw new InvalidOperationException($"Could not instantiate robot prefab '{assetPath}'.");
            }

            instance.name = prefab.name;
            instance.transform.position = position;
            MicroDuckRig rig = instance.GetComponent<MicroDuckRig>();
            if (rig == null || rig.RootBody == null)
            {
                throw new InvalidOperationException(
                    $"Robot prefab '{assetPath}' does not contain a configured root body.");
            }

            float heightCorrection = rootHeightMeters - rig.RootBody.transform.position.y;
            instance.transform.position += Vector3.up * heightCorrection;
            return instance;
        }

        private static GameObject InstantiateMujocoModel(string assetPath, bool active)
        {
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(assetPath);
            if (prefab == null)
            {
                throw new FileNotFoundException(
                    $"Official MuJoCo prefab was not found at '{assetPath}'.",
                    assetPath);
            }

            var instance = PrefabUtility.InstantiatePrefab(prefab) as GameObject;
            if (instance == null)
            {
                throw new InvalidOperationException(
                    $"Could not instantiate official MuJoCo prefab '{assetPath}'.");
            }

            instance.name = prefab.name;
            instance.SetActive(active);
            return instance;
        }

        private static void DisableImportedFloor(GameObject modelRoot)
        {
            foreach (MjGeom geom in modelRoot.GetComponentsInChildren<MjGeom>(includeInactive: true))
            {
                if (geom.ShapeType == MjShapeComponent.ShapeTypes.Plane
                    || string.Equals(geom.name, "floor", StringComparison.OrdinalIgnoreCase))
                {
                    geom.gameObject.SetActive(false);
                }
            }
        }

        private static MicroDuckRig RequireRig(GameObject robot, RobotVariant expectedVariant)
        {
            MicroDuckRig rig = robot.GetComponent<MicroDuckRig>();
            if (rig == null || rig.Variant != expectedVariant)
            {
                throw new InvalidOperationException(
                    $"Prefab '{robot.name}' does not contain a configured {expectedVariant} MicroDuckRig.");
            }

            return rig;
        }

        private static PolicyModelBinding[] LoadPolicyBindings()
        {
            return PolicyCatalog.Entries.Select(entry =>
            {
                string path = $"Assets/MicroDuck/Generated/Policies/Barracuda/{entry.FileName}";
                NNModel model = AssetDatabase.LoadAssetAtPath<NNModel>(path);
                if (model == null)
                {
                    throw new FileNotFoundException($"Barracuda policy model was not found at '{path}'.", path);
                }

                return new PolicyModelBinding { slot = entry.Slot, model = model };
            }).ToArray();
        }

        private static Collider CreateFloor(PhysicMaterial material)
        {
            GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
            floor.name = "Floor";
            floor.transform.position = new Vector3(0f, -0.05f, 0f);
            floor.transform.localScale = new Vector3(8f, 0.1f, 8f);
            Collider collider = floor.GetComponent<Collider>();
            collider.contactOffset = RobotPrefabImporter.ContactOffsetMeters;
            collider.sharedMaterial = material;
            return collider;
        }

        private static MicroDuckSkillBall CreateSkillBall(
            Collider floorCollider,
            PhysicMaterial material)
        {
            var ballObject = new GameObject("MicroDuck Skill Ball");
            Rigidbody body = ballObject.AddComponent<Rigidbody>();
            SphereCollider sphereCollider = ballObject.AddComponent<SphereCollider>();
            sphereCollider.contactOffset = RobotPrefabImporter.ContactOffsetMeters;
            sphereCollider.sharedMaterial = material;
            MicroDuckSkillBall skillBall = ballObject.AddComponent<MicroDuckSkillBall>();
            skillBall.Configure(body, sphereCollider, floorCollider);

            GameObject visual = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            visual.name = "Visual";
            UnityEngine.Object.DestroyImmediate(visual.GetComponent<Collider>());
            visual.transform.SetParent(ballObject.transform, worldPositionStays: false);
            visual.transform.localPosition = Vector3.zero;
            visual.transform.localRotation = Quaternion.identity;
            visual.transform.localScale = Vector3.one * (MicroDuckSkillBall.RadiusMeters * 2f);

            skillBall.Hide();
            return skillBall;
        }

        private static void CreateLight()
        {
            var lightObject = new GameObject("Directional Light");
            Light light = lightObject.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.2f;
            lightObject.transform.rotation = Quaternion.Euler(45f, -30f, 0f);
        }

        private static Camera CreateCamera()
        {
            var cameraObject = new GameObject("Main Camera");
            cameraObject.tag = "MainCamera";
            Camera camera = cameraObject.AddComponent<Camera>();
            cameraObject.AddComponent<AudioListener>();
            cameraObject.transform.position = new Vector3(0f, 1.15f, -2.4f);
            cameraObject.transform.LookAt(new Vector3(0f, 0.35f, 0f));
            camera.clearFlags = CameraClearFlags.Skybox;
            return camera;
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
