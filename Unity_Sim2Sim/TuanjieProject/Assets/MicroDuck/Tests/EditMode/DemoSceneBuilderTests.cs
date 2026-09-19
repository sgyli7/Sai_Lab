using System.IO;
using System.Linq;
using System.Reflection;
using AgenticRobot.MicroDuck.Editor;
using AgenticRobot.MicroDuck.Mujoco;
using Mujoco;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class DemoSceneBuilderTests
    {
        private const string RobotAndFloorMaterialPath =
            "Assets/MicroDuck/Generated/PhysicsMaterials/MuJoCo-RobotAndFloor.physicMaterial";

        private const string BallMaterialPath =
            "Assets/MicroDuck/Generated/PhysicsMaterials/MuJoCo-Ball.physicMaterial";

        [Test]
        public void CreatesPlayableMvpSceneWithBothRobotsAndOperatorControls()
        {
            RobotPrefabImporter.ImportAll();

            string scenePath = DemoSceneBuilder.CreateSceneAsset();
            Scene scene = EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);

            Assert.That(scenePath, Is.EqualTo(DemoSceneBuilder.SceneAssetPath));
            Assert.That(AssetDatabase.LoadAssetAtPath<SceneAsset>(scenePath), Is.Not.Null);
            GameObject floor = scene.GetRootGameObjects().Single(item => item.name == "Floor");
            Collider floorCollider = floor.GetComponent<Collider>();
            Assert.That(floorCollider.contactOffset,
                Is.EqualTo(0.001f).Within(1e-7f));
            AssertPersistentMaterial(floorCollider.sharedMaterial, RobotAndFloorMaterialPath, 1f);
            Assert.That(Object.FindObjectsOfType<Camera>(true).Length, Is.EqualTo(1));
            Assert.That(Object.FindObjectsOfType<Light>(true).Length, Is.EqualTo(1));

            MicroDuckRig[] rigs = Object.FindObjectsOfType<MicroDuckRig>(true);
            Assert.That(rigs.Select(rig => rig.Variant), Is.EquivalentTo(new[]
            {
                RobotVariant.Legged,
                RobotVariant.Roller,
            }));

            MicroDuckDemoController controller = Object.FindObjectOfType<MicroDuckDemoController>();
            KeyboardPolicyInput keyboard = Object.FindObjectOfType<KeyboardPolicyInput>();
            PolicyStatusOverlay overlay = Object.FindObjectOfType<PolicyStatusOverlay>();
            MicroDuckSkillBall[] balls = Object.FindObjectsOfType<MicroDuckSkillBall>(true);
            Assert.That(controller, Is.Not.Null);
            Assert.That(keyboard, Is.Not.Null);
            Assert.That(overlay, Is.Not.Null);
            Assert.That(overlay.Controller, Is.SameAs(controller));
            Assert.That(balls, Has.Length.EqualTo(1));
            Assert.That(controller.SkillBall, Is.SameAs(balls[0]));
            Assert.That(balls[0].gameObject.activeSelf, Is.False);
            Assert.That(balls[0].Collider.radius,
                Is.EqualTo(MicroDuckSkillBall.RadiusMeters).Within(1e-7f));
            Assert.That(balls[0].Body.mass,
                Is.EqualTo(MicroDuckSkillBall.MassKilograms).Within(1e-7f));
            AssertPersistentMaterial(balls[0].Collider.sharedMaterial, BallMaterialPath, 0.5f);
        }

        [Test]
        public void UsesTheOfficialDeploymentSpawnHeightForEachRobotVariant()
        {
            DemoSceneBuilder.CreateSceneAsset();

            MicroDuckRig[] rigs = Object.FindObjectsOfType<MicroDuckRig>(true);
            MicroDuckRig legged = rigs.Single(rig => rig.Variant == RobotVariant.Legged);
            MicroDuckRig roller = rigs.Single(rig => rig.Variant == RobotVariant.Roller);

            Assert.That(legged.RootBody.transform.position.y, Is.EqualTo(0.125f).Within(1e-6f));
            Assert.That(roller.RootBody.transform.position.y, Is.EqualTo(0.1385f).Within(1e-6f));
        }

        [Test]
        public void ConfiguresNativeAndCalibrationScenesForPlayModeCoverage()
        {
            DemoSceneBuilder.CreateSceneAsset();
            DemoSceneBuilder.CreateNativeSceneAsset();

            DemoSceneBuilder.ConfigureBuildSettings();

            Assert.That(EditorBuildSettings.scenes.Length, Is.EqualTo(2));
            Assert.That(
                EditorBuildSettings.scenes[0].path,
                Is.EqualTo(DemoSceneBuilder.NativeSceneAssetPath));
            Assert.That(EditorBuildSettings.scenes[0].enabled, Is.True);
            Assert.That(
                EditorBuildSettings.scenes[1].path,
                Is.EqualTo(DemoSceneBuilder.SceneAssetPath));
            Assert.That(EditorBuildSettings.scenes[1].enabled, Is.True);
        }

        [Test]
        public void CreatesWindows64BuildOptionsAtTheRepositoryBuildPath()
        {
            DemoSceneBuilder.CreateSceneAsset();
            DemoSceneBuilder.CreateNativeSceneAsset();
            DemoSceneBuilder.ConfigureBuildSettings();

            BuildPlayerOptions options = DemoSceneBuilder.CreateWindows64BuildOptions();

            Assert.That(options.target, Is.EqualTo(BuildTarget.StandaloneWindows64));
            Assert.That(
                options.scenes,
                Is.EqualTo(new[] { DemoSceneBuilder.NativeSceneAssetPath }));
            Assert.That(Path.IsPathRooted(options.locationPathName), Is.True);
            Assert.That(
                options.locationPathName.Replace('\\', '/'),
                Does.EndWith("/Builds/Windows64/AgenticRobotGame.exe"));
        }

        [Test]
        public void CreatesMacOSBuildOptionsAtTheRepositoryBuildPath()
        {
            DemoSceneBuilder.CreateSceneAsset();
            DemoSceneBuilder.CreateNativeSceneAsset();
            DemoSceneBuilder.ConfigureBuildSettings();

            BuildPlayerOptions options = DemoSceneBuilder.CreateMacOSBuildOptions();

            Assert.That(options.target, Is.EqualTo(BuildTarget.StandaloneOSX));
            Assert.That(
                options.scenes,
                Is.EqualTo(new[] { DemoSceneBuilder.NativeSceneAssetPath }));
            Assert.That(Path.IsPathRooted(options.locationPathName), Is.True);
            Assert.That(
                options.locationPathName.Replace('\\', '/'),
                Does.EndWith("/Builds/macOS/AgenticRobotGame.app"));
        }

        [Test]
        public void PersistsTheTwoHundredHertzPhysicsStepWhenCreatingTheScene()
        {
            Time.fixedDeltaTime = 0.02f;

            DemoSceneBuilder.CreateSceneAsset();

            Assert.That(Time.fixedDeltaTime, Is.EqualTo(0.005f).Within(1e-7f));
            string timeManagerPath = Path.Combine(
                Path.GetDirectoryName(Application.dataPath),
                "ProjectSettings",
                "TimeManager.asset");
            Assert.That(
                File.ReadAllText(timeManagerPath),
                Does.Contain("Fixed Timestep: 0.005"));
        }

        [Test]
        public void LoadsThePersistedTwoHundredHertzStepInANewEditorProcess()
        {
            Assert.That(Time.fixedDeltaTime, Is.EqualTo(0.005f).Within(1e-7f));
        }

        [Test]
        public void CreatesPlayableNativeSceneWithOfficialMuJoCoAndBarracudaControls()
        {
            string scenePath = DemoSceneBuilder.CreateNativeSceneAsset();
            Scene scene = EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);

            Assert.That(scenePath, Is.EqualTo(DemoSceneBuilder.NativeSceneAssetPath));
            Assert.That(AssetDatabase.LoadAssetAtPath<SceneAsset>(scenePath), Is.Not.Null);
            Assert.That(Object.FindObjectsOfType<MjScene>(true), Has.Length.EqualTo(1));
            Assert.That(Object.FindObjectsOfType<MujocoDemoController>(true), Has.Length.EqualTo(1));
            Assert.That(Object.FindObjectsOfType<MujocoKeyboardPolicyInput>(true), Has.Length.EqualTo(1));
            Assert.That(Object.FindObjectsOfType<MujocoPolicyStatusOverlay>(true), Has.Length.EqualTo(1));
            Assert.That(Object.FindObjectsOfType<MujocoPlayerSmokeProbe>(true), Has.Length.EqualTo(1));
            Assert.That(Object.FindObjectsOfType<MujocoPlayerAcceptanceTour>(true), Has.Length.EqualTo(1));
            Assert.That(Object.FindObjectsOfType<MicroDuckDemoController>(true), Is.Empty);

            MujocoTerrainNavigator navigator = Object.FindObjectOfType<MujocoTerrainNavigator>(true);
            MujocoCameraRig cameraRig = Object.FindObjectOfType<MujocoCameraRig>(true);
            MujocoDemoController nativeController =
                Object.FindObjectOfType<MujocoDemoController>(true);
            MujocoPlayerAcceptanceTour tour =
                Object.FindObjectOfType<MujocoPlayerAcceptanceTour>(true);
            MujocoPolicyStatusOverlay nativeOverlay =
                Object.FindObjectOfType<MujocoPolicyStatusOverlay>(true);
            Assert.That(navigator, Is.Not.Null);
            Assert.That(cameraRig, Is.Not.Null);
            Assert.That(ReadPrivateField<MujocoDemoController>(tour, "controller"),
                Is.SameAs(nativeController));
            Assert.That(ReadPrivateField<MujocoTerrainNavigator>(tour, "terrainNavigator"),
                Is.SameAs(navigator));
            Assert.That(ReadPrivateField<MujocoCameraRig>(tour, "cameraRig"),
                Is.SameAs(cameraRig));
            Assert.That(nativeOverlay.Controller,
                Is.SameAs(Object.FindObjectOfType<MujocoDemoController>(true)));
            Assert.That(nativeOverlay.Navigator, Is.SameAs(navigator));
            Assert.That(nativeOverlay.CameraRig, Is.SameAs(cameraRig));
            Assert.That(nativeOverlay.BuildTerrainText(), Does.Contain("中央安全广场"));
            Assert.That(nativeOverlay.BuildControlsText(), Does.Contain("T / Shift+T"));
            Assert.That(nativeOverlay.BuildControlsText(), Does.Contain("Tab"));

            MjGlobalSettings[] globalSettings = Object.FindObjectsOfType<MjGlobalSettings>(true);
            Assert.That(globalSettings, Has.Length.EqualTo(1));
            Assert.That(globalSettings[0].UseRawGameObjectNames, Is.True,
                "The policy stepper resolves the stable names from the official MJCF.");

            GameObject legged = scene.GetRootGameObjects().Single(
                item => item.name == "MicroDuck-MuJoCo-Ball");
            GameObject roller = scene.GetRootGameObjects().Single(
                item => item.name == "MicroDuck-MuJoCo-Roller");
            Assert.That(legged.activeSelf, Is.True);
            Assert.That(roller.activeSelf, Is.False);
            Assert.That(legged.GetComponentsInChildren<MjActuator>(true), Has.Length.EqualTo(14));
            Assert.That(roller.GetComponentsInChildren<MjActuator>(true), Has.Length.EqualTo(14));
            Assert.That(
                legged.GetComponentsInChildren<MjFreeJoint>(true)
                    .Any(joint => joint.name == "ball_free"),
                Is.True);
        }

        [Test]
        public void CreatesApprovedAlpineWorldFromTheNativeTerrainCatalog()
        {
            Scene scene = EditorSceneManager.NewScene(
                NewSceneSetup.EmptyScene,
                NewSceneMode.Single);
            AlpineEnvironmentBuilder.Build();

            GameObject environment = scene.GetRootGameObjects().Single(
                item => item.name == "MicroDuck Alpine Environment");
            Transform terrainRoot = environment.transform.Find("Terrain");
            Transform sceneryRoot = environment.transform.Find("Scenery");
            Assert.That(terrainRoot, Is.Not.Null);
            Assert.That(sceneryRoot, Is.Not.Null);

            int expectedGeomCount = 0;
            foreach (TerrainModuleDefinition module in MujocoTerrainCatalog.Modules)
            {
                Transform moduleRoot = terrainRoot.Find(module.Id);
                Assert.That(moduleRoot, Is.Not.Null, module.Id);
                Assert.That(moduleRoot.childCount, Is.EqualTo(module.Primitives.Count), module.Id);
                expectedGeomCount += module.Primitives.Count;
            }

            Assert.That(
                terrainRoot.GetComponentsInChildren<MjGeom>(includeInactive: true),
                Has.Length.EqualTo(expectedGeomCount));
            Assert.That(
                terrainRoot.GetComponentsInChildren<Collider>(includeInactive: true),
                Is.Empty,
                "MuJoCo terrain must not accidentally create parallel PhysX collision.");
            Assert.That(
                sceneryRoot.GetComponentsInChildren<Collider>(includeInactive: true),
                Is.Empty,
                "Alpine scenery is visual only and must not disagree with MuJoCo physics.");

            Assert.That(RenderSettings.skybox, Is.Not.Null);
            Assert.That(RenderSettings.skybox.shader.name, Is.EqualTo("Skybox/Procedural"));
            Assert.That(RenderSettings.fog, Is.True);
            Assert.That(RenderSettings.ambientIntensity, Is.GreaterThan(0.5f));

            Light sun = Object.FindObjectsOfType<Light>(includeInactive: true)
                .Single(light => light.name == "Alpine Sun");
            Assert.That(sun.type, Is.EqualTo(LightType.Directional));
            Assert.That(sun.color.r, Is.GreaterThan(sun.color.b));
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

        private static T ReadPrivateField<T>(object target, string name)
        {
            FieldInfo field = target.GetType().GetField(
                name,
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(field, Is.Not.Null, name);
            return (T)field.GetValue(target);
        }
    }
}
