using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using AgenticRobot.MicroDuck.Mujoco;
using Mujoco;
using NUnit.Framework;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class NativeMujocoScenePlayModeTests
    {
        private const string NativeSceneAssetPath =
            "Assets/MicroDuck/Generated/Scenes/MicroDuckNativeMvp.unity";
        private const int InitializationTimeoutFrames = 240;

        [Test]
        public void KeyboardPolicySwitchesRunAfterMujocoSceneLateUpdate()
        {
            DefaultExecutionOrder order = typeof(MujocoKeyboardPolicyInput)
                .GetCustomAttribute<DefaultExecutionOrder>();
            MethodInfo update = typeof(MujocoKeyboardPolicyInput).GetMethod(
                "Update",
                BindingFlags.Instance | BindingFlags.NonPublic);
            MethodInfo lateUpdate = typeof(MujocoKeyboardPolicyInput).GetMethod(
                "LateUpdate",
                BindingFlags.Instance | BindingFlags.NonPublic);

            Assert.That(order, Is.Not.Null,
                "Keyboard input needs an explicit late execution order for safe model switching.");
            Assert.That(order.order, Is.GreaterThan(0),
                "Keyboard input must run after the default-order MjScene LateUpdate.");
            Assert.That(update, Is.Null,
                "Cross-variant root activation during Update can trigger multiple MuJoCo rebuilds.");
            Assert.That(lateUpdate, Is.Not.Null,
                "Keyboard commands must be applied from LateUpdate after MjScene has finished its frame.");
        }

        [UnityTest]
        [Timeout(90000)]
        public IEnumerator NativeStandPolicyKeepsTheVisibleDuckUprightForFourSeconds()
        {
            yield return EditorSceneManager.LoadSceneAsyncInPlayMode(
                NativeSceneAssetPath,
                new LoadSceneParameters(LoadSceneMode.Single));
            yield return null;

            MjScene scene = UnityEngine.Object.FindObjectOfType<MjScene>();
            MujocoDemoController controller =
                UnityEngine.Object.FindObjectOfType<MujocoDemoController>();
            yield return WaitForHealthyPolicy(scene, controller, minimumTicks: 1);

            string generatedMjcfPath = Path.GetFullPath(Path.Combine(
                Application.dataPath,
                "..",
                "..",
                "artifacts",
                "visual-acceptance",
                "tuanjie-runtime-generated.xml"));
            Directory.CreateDirectory(Path.GetDirectoryName(generatedMjcfPath));
            SaveCompiledModel(scene, generatedMjcfPath);

            Assert.That(controller.SwitchPolicy(2), Is.True, controller.Fault);
            controller.ResetActiveRobot();
            double endTime = ReadSimulationTime(scene) + 4.0;
            while (ReadSimulationTime(scene) < endTime)
            {
                yield return new WaitForFixedUpdate();
            }

            MjBody trunk = FindBody(controller.ActiveModelRoot, "trunk_base");
            MjBody leftAnkle = FindBody(controller.ActiveModelRoot, "ankle_left");
            MjBody rightAnkle = FindBody(controller.ActiveModelRoot, "ankle_right");
            double nativeUpright = ReadBodyUpright(scene, "trunk_base");
            double nativeRootHeight = ReadRootPosition(scene, axis: 2);
            float visibleAnkleHeight = Mathf.Max(
                leftAnkle.transform.position.y,
                rightAnkle.transform.position.y);

            Assert.That(nativeUpright, Is.GreaterThan(0.8),
                $"Stand policy physically fell: upright={nativeUpright:R}.");
            Assert.That(nativeRootHeight, Is.GreaterThan(0.08),
                $"Stand policy trunk is too low: z={nativeRootHeight:R}m.");
            Assert.That(trunk.transform.position.y - visibleAnkleHeight,
                Is.GreaterThan(0.05f),
                "The rendered trunk must remain visibly above both ankles. "
                + $"trunkY={trunk.transform.position.y:R}, "
                + $"leftAnkleY={leftAnkle.transform.position.y:R}, "
                + $"rightAnkleY={rightAnkle.transform.position.y:R}.");
        }

        [UnityTest]
        [Timeout(90000)]
        public IEnumerator NativeWalkingPolicyMovesForwardWhileRemainingUprightForSixSeconds()
        {
            yield return EditorSceneManager.LoadSceneAsyncInPlayMode(
                NativeSceneAssetPath,
                new LoadSceneParameters(LoadSceneMode.Single));
            yield return null;

            MjScene scene = UnityEngine.Object.FindObjectOfType<MjScene>();
            MujocoDemoController controller =
                UnityEngine.Object.FindObjectOfType<MujocoDemoController>();
            MujocoKeyboardPolicyInput keyboard =
                UnityEngine.Object.FindObjectOfType<MujocoKeyboardPolicyInput>();
            yield return WaitForHealthyPolicy(scene, controller, minimumTicks: 1);

            Assert.That(controller.SwitchPolicy(1), Is.True, controller.Fault);
            controller.ResetActiveRobot();
            keyboard.enabled = false;
            controller.SetTwist(0.3f, 0f, 0f);
            double startX = ReadRootPosition(scene, axis: 0);
            double minimumUpright = 1.0;
            double endTime = ReadSimulationTime(scene) + 6.0;
            while (ReadSimulationTime(scene) < endTime)
            {
                yield return new WaitForFixedUpdate();
                minimumUpright = Math.Min(
                    minimumUpright,
                    ReadBodyUpright(scene, "trunk_base"));
            }

            double forwardDistance = ReadRootPosition(scene, axis: 0) - startX;
            Assert.That(minimumUpright, Is.GreaterThan(0.9),
                $"Walking policy fell during the six-second rollout: "
                + $"minimum upright={minimumUpright:R}.");
            Assert.That(forwardDistance, Is.GreaterThan(0.1),
                $"Walking policy did not produce visible forward travel: "
                + $"distance={forwardDistance:R}m.");
            Assert.That(controller.IsHealthy, Is.True, controller.Fault);
        }

        [UnityTest]
        [Timeout(90000)]
        public IEnumerator NativeRobotRemainsInsideAUsableCameraFrame()
        {
            yield return EditorSceneManager.LoadSceneAsyncInPlayMode(
                NativeSceneAssetPath,
                new LoadSceneParameters(LoadSceneMode.Single));
            yield return null;

            MjScene scene = UnityEngine.Object.FindObjectOfType<MjScene>();
            MujocoDemoController controller =
                UnityEngine.Object.FindObjectOfType<MujocoDemoController>();
            Camera camera = Camera.main;

            Assert.That(scene, Is.Not.Null);
            Assert.That(controller, Is.Not.Null);
            Assert.That(camera, Is.Not.Null, "The playable scene must contain a main camera.");
            Camera[] enabledOutputCameras = UnityEngine.Object
                .FindObjectsOfType<Camera>(includeInactive: false)
                .Where(candidate => candidate.enabled)
                .ToArray();
            Assert.That(enabledOutputCameras, Has.Length.EqualTo(1),
                "Imported MJCF sensor cameras must not overwrite the playable main camera.");
            Assert.That(enabledOutputCameras[0], Is.SameAs(camera));
            yield return WaitForHealthyPolicy(scene, controller, minimumTicks: 1);

            int[] checkpoints = { 5, 100, 300, 600, 1200, 2400 };
            int elapsed = 0;
            foreach (int checkpoint in checkpoints)
            {
                while (elapsed < checkpoint)
                {
                    elapsed++;
                    yield return new WaitForFixedUpdate();
                }

                AssertActiveRobotIsFramed(camera, controller, checkpoint);
                if (checkpoint == checkpoints[checkpoints.Length - 1]
                    && SystemInfo.graphicsDeviceType
                        != UnityEngine.Rendering.GraphicsDeviceType.Null)
                {
                    CaptureAndAssertRenderedFrame(camera, checkpoint);
                }
            }
        }

        [UnityTest]
        [Timeout(90000)]
        public IEnumerator NativeSceneRunsBarracudaPolicyAndRecreatesOnlyForRobotVariantChanges()
        {
            yield return EditorSceneManager.LoadSceneAsyncInPlayMode(
                NativeSceneAssetPath,
                new LoadSceneParameters(LoadSceneMode.Single));
            yield return null;

            MjScene scene = UnityEngine.Object.FindObjectOfType<MjScene>();
            MujocoDemoController controller =
                UnityEngine.Object.FindObjectOfType<MujocoDemoController>();
            MujocoKeyboardPolicyInput keyboard =
                UnityEngine.Object.FindObjectOfType<MujocoKeyboardPolicyInput>();

            Assert.That(scene, Is.Not.Null, "Native MVP scene must contain its official MjScene.");
            Assert.That(scene, Is.TypeOf<MicroDuckMjScene>(),
                "The native scene must use the early-order MjScene adapter so the official "
                + "singleton is registered before imported MjComponents are enabled.");
            Assert.That(controller, Is.Not.Null, "Native MVP scene must contain its policy controller.");
            Assert.That(keyboard, Is.Not.Null, "Native MVP scene must contain keyboard controls.");
            Assert.That(ReadKeyboardController(keyboard), Is.SameAs(controller),
                "The scene's keyboard component must control the native policy controller.");
            MjGlobalSettings[] globalSettings =
                UnityEngine.Object.FindObjectsOfType<MjGlobalSettings>(includeInactive: true);
            Assert.That(globalSettings, Has.Length.EqualTo(1));
            Assert.That(globalSettings[0].UseRawGameObjectNames, Is.True);

            yield return WaitForHealthyPolicy(scene, controller, minimumTicks: 1);

            Assert.That(controller.ActivePolicySlot, Is.EqualTo(2));
            Assert.That(controller.BackendName,
                Is.EqualTo("MuJoCo 3.12 + Barracuda 3.0.1 CPU"));
            AssertPolicyTickState(controller, "initial legged policy");

            controller.ResetActiveRobot();
            yield return WaitForHealthyPolicy(scene, controller, minimumTicks: 1);

            NativeModelSnapshot initialModel = CaptureModel(scene);
            GameObject initialLeggedRoot = controller.ActiveModelRoot;
            double initialPolicyRootX = ReadRootPositionX(scene);
            float[] initialObservation = controller.LastObservation;
            float[] initialAction = controller.LastRawAction;
            float[] initialTargets = controller.LastTargets;
            Assert.That(initialModel.nu, Is.EqualTo((ulong)PolicyContract.ActionCount));
            Assert.That(initialModel.hasPassiveWheel, Is.False);
            AssertNativeObjectName(
                scene,
                MujocoLib.mjtObj.mjOBJ_JOINT,
                "trunk_base_freejoint");
            AssertNativeObjectName(scene, MujocoLib.mjtObj.mjOBJ_BODY, "trunk_base");

            int recreationCount = 0;
            EventHandler<MjStepArgs> countRecreation = (_, __) => recreationCount++;
            scene.postInitEvent += countRecreation;

            // Slot 1 and slot 2 share the legged model. This must replace only the
            // Barracuda policy, leaving the native scene and state allocation intact.
            int ticksBeforeHotSwap = controller.PolicyTicks;
            Assert.That(controller.SwitchPolicy(1), Is.True, controller.Fault);
            Assert.That(controller.IsHealthy, Is.True, controller.Fault);
            Assert.That(controller.ActivePolicySlot, Is.EqualTo(1));
            Assert.That(controller.PolicyTicks, Is.EqualTo(ticksBeforeHotSwap));
            Assert.That(CaptureModel(scene), Is.EqualTo(initialModel),
                "A same-variant policy switch must preserve the MuJoCo model and data.");
            Assert.That(recreationCount, Is.Zero,
                "A same-variant policy switch must not recreate MjScene.");

            MjBody trunk = controller.ActiveModelRoot
                .GetComponentsInChildren<MjBody>(includeInactive: true)
                .Single(body => body.name == "trunk_base");
            Vector3 initialVisualPosition = trunk.transform.position;
            double initialSimulationTime = ReadSimulationTime(scene);
            InjectRootLinearVelocity(scene, metresPerSecond: 0.35);

            bool visualMoved = false;
            for (int step = 0; step < 16; step++)
            {
                yield return new WaitForFixedUpdate();
                AssertBodyTransformSynchronized(scene, trunk);
                visualMoved |= Vector3.Distance(
                    trunk.transform.position,
                    initialVisualPosition) > 1e-4f;
            }

            Assert.That(ReadSimulationTime(scene), Is.GreaterThan(initialSimulationTime));
            Assert.That(visualMoved, Is.True,
                "The visible trunk must move while the native MuJoCo scene advances.");
            Assert.That(controller.PolicyTicks, Is.GreaterThan(ticksBeforeHotSwap));
            AssertPolicyTickState(controller, "legged policy after native stepping");
            Assert.That(recreationCount, Is.Zero,
                "Ordinary native stepping must not recreate MjScene.");

            // Slot 7 needs the roller MJCF. Enabling that imported hierarchy causes
            // MjScene to rebuild once; the controller binds a fresh policy stepper in
            // postInitEvent and must return to a healthy state.
            Assert.That(controller.SwitchPolicy(7), Is.True, controller.Fault);
            yield return WaitForModelRecreationAndHealthyPolicy(
                controller,
                () => recreationCount,
                minimumTicks: 1);

            NativeModelSnapshot rollerModel = CaptureModel(scene);
            Assert.That(scene, Is.SameAs(UnityEngine.Object.FindObjectOfType<MjScene>()));
            Assert.That(recreationCount, Is.EqualTo(1),
                "Changing robot variants must rebuild the native model exactly once.");
            Assert.That(rollerModel.hasPassiveWheel, Is.True);
            Assert.That(rollerModel.nu, Is.EqualTo((ulong)PolicyContract.ActionCount));
            Assert.That(rollerModel.nq, Is.Not.EqualTo(initialModel.nq),
                "The roller hierarchy must compile into a distinct MuJoCo model.");
            Assert.That(controller.ActivePolicySlot, Is.EqualTo(7));
            Assert.That(controller.ActiveModelRoot.name, Does.Contain("Roller"));
            Assert.That(controller.BackendName,
                Is.EqualTo("MuJoCo 3.12 + Barracuda 3.0.1 CPU"));
            AssertPolicyTickState(controller, "roller policy after model recreation");
            AssertNativeObjectName(
                scene,
                MujocoLib.mjtObj.mjOBJ_JOINT,
                "passive_LF_wheel");

            // Exercise the reverse transition too. The official MjScene caches
            // component state while rebuilding; previously-bound inactive legged
            // joints must not rehydrate roller state after postInitEvent.
            SetRootPositionX(scene, 3.0);
            double rootAfterFirstReturnControl = double.NaN;
            double timeAfterFirstReturnControl = double.NaN;
            double rootVelocityAfterFirstReturnControl = double.NaN;
            double rootAfterFirstReturnStep = double.NaN;
            float[] firstReturnObservation = null;
            float[] firstReturnAction = null;
            float[] firstReturnTargets = null;
            EventHandler<MjStepArgs> captureFirstReturnControl = (_, __) =>
            {
                if (recreationCount >= 2 && double.IsNaN(rootAfterFirstReturnControl))
                {
                    rootAfterFirstReturnControl = ReadRootPositionX(scene);
                    timeAfterFirstReturnControl = ReadSimulationTime(scene);
                    rootVelocityAfterFirstReturnControl = ReadRootLinearVelocityX(scene);
                    firstReturnObservation = controller.LastObservation;
                    firstReturnAction = controller.LastRawAction;
                    firstReturnTargets = controller.LastTargets;
                }
            };
            EventHandler<MjStepArgs> captureFirstReturnStep = (_, __) =>
            {
                if (recreationCount >= 2 && double.IsNaN(rootAfterFirstReturnStep))
                {
                    rootAfterFirstReturnStep = ReadRootPositionX(scene);
                }
            };
            scene.ctrlCallback += captureFirstReturnControl;
            scene.postUpdateEvent += captureFirstReturnStep;
            Assert.That(controller.SwitchPolicy(2), Is.True, controller.Fault);
            yield return WaitForModelRecreationAndHealthyPolicy(
                controller,
                () => recreationCount,
                minimumRecreationCount: 2,
                minimumTicks: 1);
            scene.postInitEvent -= countRecreation;
            scene.ctrlCallback -= captureFirstReturnControl;
            scene.postUpdateEvent -= captureFirstReturnStep;

            NativeModelSnapshot returnedLeggedModel = CaptureModel(scene);
            Assert.That(recreationCount, Is.EqualTo(2),
                "Switching roller back to legged must rebuild exactly once more.");
            Assert.That(returnedLeggedModel.hasPassiveWheel, Is.False);
            Assert.That(controller.ActivePolicySlot, Is.EqualTo(2));
            Assert.That(controller.ActiveModelRoot, Is.SameAs(initialLeggedRoot));
            Assert.That(rootAfterFirstReturnControl, Is.EqualTo(0.0).Within(1e-6),
                "The controller must reset native state after MjScene rehydration and before stepping.");
            Assert.That(firstReturnObservation,
                Is.EqualTo(initialObservation).Within(1e-5f),
                "Reverse switching must recreate the same first policy observation as an explicit reset. "
                + $"First-control time={timeAfterFirstReturnControl}, "
                + $"rootVelocityX={rootVelocityAfterFirstReturnControl}.");
            Assert.That(firstReturnAction,
                Is.EqualTo(initialAction).Within(1e-5f));
            Assert.That(firstReturnTargets,
                Is.EqualTo(initialTargets).Within(1e-5f));
            Assert.That(rootAfterFirstReturnStep, Is.EqualTo(initialPolicyRootX).Within(0.02),
                "A rebuilt legged model must match a fresh legged start, not cached roller state.");
            AssertPolicyTickState(controller, "legged policy after reverse model recreation");
        }

        private static IEnumerator WaitForHealthyPolicy(
            MjScene scene,
            MujocoDemoController controller,
            int minimumTicks)
        {
            for (int frame = 0; frame < InitializationTimeoutFrames; frame++)
            {
                if (controller.IsHealthy && controller.PolicyTicks >= minimumTicks)
                {
                    yield break;
                }

                yield return new WaitForFixedUpdate();
            }

            Assert.Fail(
                $"Native controller did not initialize in {InitializationTimeoutFrames} frames: "
                + controller.Fault
                + Environment.NewLine
                + DescribeNativeNames(scene));
        }

        private static IEnumerator WaitForModelRecreationAndHealthyPolicy(
            MujocoDemoController controller,
            Func<int> recreationCount,
            int minimumTicks,
            int minimumRecreationCount = 1)
        {
            for (int frame = 0; frame < InitializationTimeoutFrames; frame++)
            {
                if (recreationCount() >= minimumRecreationCount
                    && controller.IsHealthy
                    && controller.PolicyTicks >= minimumTicks)
                {
                    yield break;
                }

                yield return null;
            }

            Assert.Fail(
                $"Cross-variant switch did not recreate and rebind the MuJoCo model in "
                + $"{InitializationTimeoutFrames} frames: {controller.Fault}");
        }

        private static MujocoDemoController ReadKeyboardController(
            MujocoKeyboardPolicyInput keyboard)
        {
            FieldInfo field = typeof(MujocoKeyboardPolicyInput).GetField(
                "controller",
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(field, Is.Not.Null, "Keyboard controller binding field was renamed.");
            return (MujocoDemoController)field.GetValue(keyboard);
        }

        private static void AssertActiveRobotIsFramed(
            Camera camera,
            MujocoDemoController controller,
            int fixedFrame)
        {
            Renderer[] sceneRenderers = UnityEngine.Object
                .FindObjectsOfType<Renderer>(includeInactive: false)
                .Where(renderer => renderer.enabled)
                .ToArray();
            var nearest = sceneRenderers
                .Select(renderer => new
                {
                    Renderer = renderer,
                    Distance = Mathf.Sqrt(renderer.bounds.SqrDistance(camera.transform.position)),
                })
                .OrderBy(item => item.Distance)
                .ToArray();
            string nearestContext = string.Join(
                "; ",
                nearest.Take(12).Select(item =>
                    $"{HierarchyPath(item.Renderer.transform)}:distance={item.Distance:R},"
                    + $"center={item.Renderer.bounds.center},size={item.Renderer.bounds.size}"));
            Assert.That(nearest[0].Distance, Is.GreaterThan(camera.nearClipPlane),
                $"An enabled renderer intersects the camera near plane at fixedFrame={fixedFrame}: "
                + nearestContext);
            MjBody trunk = controller.ActiveModelRoot
                .GetComponentsInChildren<MjBody>(includeInactive: false)
                .Single(body => body.name == "trunk_base");
            Renderer[] renderers = trunk
                .GetComponentsInChildren<Renderer>(includeInactive: false)
                .Where(renderer => renderer.enabled)
                .ToArray();
            Assert.That(renderers, Is.Not.Empty,
                "The active robot must have enabled renderers below trunk_base.");

            Bounds bounds = renderers[0].bounds;
            foreach (Renderer renderer in renderers.Skip(1))
            {
                bounds.Encapsulate(renderer.bounds);
            }

            Vector3 viewport = camera.WorldToViewportPoint(bounds.center);
            Vector3 cameraSpace = camera.transform.InverseTransformPoint(bounds.center);
            Vector2 robotViewportSize = ProjectedViewportSize(camera, bounds);
            string largestRenderers = string.Join(
                "; ",
                renderers
                    .OrderByDescending(renderer => renderer.bounds.size.sqrMagnitude)
                    .Take(8)
                    .Select(renderer =>
                        $"{renderer.name}:center={renderer.bounds.center},size={renderer.bounds.size}"));
            string context =
                $"fixedFrame={fixedFrame}; cameraPosition={camera.transform.position}; "
                + $"cameraForward={camera.transform.forward}; trunkPosition={trunk.transform.position}; "
                + $"boundsCenter={bounds.center}; boundsSize={bounds.size}; "
                + $"cameraSpace={cameraSpace}; viewport={viewport}; "
                + $"robotViewportSize={robotViewportSize}; renderers=[{largestRenderers}]";

            Assert.That(viewport.z, Is.GreaterThan(camera.nearClipPlane), context);
            Assert.That(cameraSpace.z, Is.InRange(0.8f, 2f), context);
            Assert.That(viewport.x, Is.InRange(0.1f, 0.9f), context);
            Assert.That(viewport.y, Is.InRange(0.1f, 0.9f), context);
            Assert.That(bounds.size.magnitude, Is.InRange(0.05f, 1.5f), context);
            Assert.That(
                Mathf.Max(robotViewportSize.x, robotViewportSize.y),
                Is.InRange(0.25f, 0.75f),
                "The robot must be large enough to inspect and control visually. " + context);
        }

        private static string HierarchyPath(Transform transform)
        {
            string path = transform.name;
            while (transform.parent != null)
            {
                transform = transform.parent;
                path = transform.name + "/" + path;
            }

            return path;
        }

        private static Vector2 ProjectedViewportSize(Camera camera, Bounds bounds)
        {
            Vector3 center = bounds.center;
            Vector3 extents = bounds.extents;
            float minimumX = float.PositiveInfinity;
            float minimumY = float.PositiveInfinity;
            float maximumX = float.NegativeInfinity;
            float maximumY = float.NegativeInfinity;
            for (int x = -1; x <= 1; x += 2)
            {
                for (int y = -1; y <= 1; y += 2)
                {
                    for (int z = -1; z <= 1; z += 2)
                    {
                        Vector3 viewport = camera.WorldToViewportPoint(
                            center + Vector3.Scale(extents, new Vector3(x, y, z)));
                        if (viewport.z <= camera.nearClipPlane)
                        {
                            continue;
                        }

                        minimumX = Mathf.Min(minimumX, viewport.x);
                        minimumY = Mathf.Min(minimumY, viewport.y);
                        maximumX = Mathf.Max(maximumX, viewport.x);
                        maximumY = Mathf.Max(maximumY, viewport.y);
                    }
                }
            }

            if (float.IsPositiveInfinity(minimumX))
            {
                return Vector2.zero;
            }

            return new Vector2(maximumX - minimumX, maximumY - minimumY);
        }

        private static void CaptureAndAssertRenderedFrame(Camera camera, int fixedFrame)
        {
            const int width = 640;
            const int height = 360;
            RenderTexture target = RenderTexture.GetTemporary(width, height, 24);
            RenderTexture previousActive = RenderTexture.active;
            RenderTexture previousTarget = camera.targetTexture;
            var image = new Texture2D(width, height, TextureFormat.RGB24, mipChain: false);
            try
            {
                camera.targetTexture = target;
                camera.Render();
                RenderTexture.active = target;
                image.ReadPixels(new Rect(0, 0, width, height), 0, 0);
                image.Apply();

                string artifactDirectory = Path.GetFullPath(Path.Combine(
                    Application.dataPath,
                    "../../artifacts/visual-acceptance"));
                Directory.CreateDirectory(artifactDirectory);
                File.WriteAllBytes(
                    Path.Combine(artifactDirectory, "playmode-camera.png"),
                    image.EncodeToPNG());

                Color32[] pixels = image.GetPixels32();
                int brightPixels = 0;
                int blueSkyPixels = 0;
                int greenGroundPixels = 0;
                var colorBuckets = new System.Collections.Generic.HashSet<int>();
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int index = (y * width) + x;
                        Color32 pixel = pixels[index];
                        int luminance = pixel.r + pixel.g + pixel.b;
                        if (luminance > 150)
                        {
                            brightPixels++;
                        }
                        if (y >= height * 2 / 3
                            && pixel.b > pixel.r
                            && pixel.b > pixel.g)
                        {
                            blueSkyPixels++;
                        }
                        if (y < height * 2 / 3
                            && pixel.g > pixel.r
                            && pixel.g > pixel.b * 0.75f)
                        {
                            greenGroundPixels++;
                        }
                        colorBuckets.Add(
                            ((pixel.r >> 5) << 6)
                            | ((pixel.g >> 5) << 3)
                            | (pixel.b >> 5));
                    }
                }

                string context =
                    $"fixedFrame={fixedFrame}; brightPixels={brightPixels}; "
                    + $"blueSkyPixels={blueSkyPixels}; greenGroundPixels={greenGroundPixels}; "
                    + $"colorBuckets={colorBuckets.Count}";

                Assert.That(brightPixels, Is.GreaterThan(width * height / 2), context);
                Assert.That(blueSkyPixels, Is.GreaterThan(width * height / 30), context);
                Assert.That(greenGroundPixels, Is.GreaterThan(width * height / 50), context);
                Assert.That(colorBuckets.Count, Is.GreaterThan(18), context);
            }
            finally
            {
                camera.targetTexture = previousTarget;
                RenderTexture.active = previousActive;
                RenderTexture.ReleaseTemporary(target);
                UnityEngine.Object.DestroyImmediate(image);
            }
        }

        private static void AssertPolicyTickState(
            MujocoDemoController controller,
            string context)
        {
            Assert.That(controller.IsHealthy, Is.True, $"{context}: {controller.Fault}");
            AssertFinite(
                controller.LastObservation,
                PolicyContract.ObservationCount,
                context + " observation");
            AssertFinite(
                controller.LastRawAction,
                PolicyContract.ActionCount,
                context + " raw action");
            AssertFinite(
                controller.LastTargets,
                PolicyContract.ActionCount,
                context + " target");
        }

        private static void AssertFinite(float[] values, int expectedCount, string context)
        {
            Assert.That(values, Has.Length.EqualTo(expectedCount), context);
            for (int index = 0; index < values.Length; index++)
            {
                Assert.That(float.IsNaN(values[index]) || float.IsInfinity(values[index]),
                    Is.False,
                    $"{context}[{index}] must be finite; got {values[index]}.");
            }
        }

        private static unsafe NativeModelSnapshot CaptureModel(MjScene scene)
        {
            Assert.That((IntPtr)scene.Model, Is.Not.EqualTo(IntPtr.Zero));
            Assert.That((IntPtr)scene.Data, Is.Not.EqualTo(IntPtr.Zero));
            bool hasPassiveWheel = MujocoLib.mj_name2id(
                scene.Model,
                (int)MujocoLib.mjtObj.mjOBJ_JOINT,
                "passive_LF_wheel") >= 0;
            return new NativeModelSnapshot(
                (IntPtr)scene.Model,
                (IntPtr)scene.Data,
                scene.Model->nq,
                scene.Model->nv,
                scene.Model->nu,
                hasPassiveWheel);
        }

        private static unsafe double ReadSimulationTime(MjScene scene)
        {
            Assert.That((IntPtr)scene.Data, Is.Not.EqualTo(IntPtr.Zero));
            return scene.Data->time;
        }

        private static unsafe double ReadRootPositionX(MjScene scene)
        {
            int jointId = MujocoLib.mj_name2id(
                scene.Model,
                (int)MujocoLib.mjtObj.mjOBJ_JOINT,
                "trunk_base_freejoint");
            Assert.That(jointId, Is.GreaterThanOrEqualTo(0));
            return scene.Data->qpos[scene.Model->jnt_qposadr[jointId]];
        }

        private static MjBody FindBody(GameObject root, string bodyName)
        {
            return root.GetComponentsInChildren<MjBody>(includeInactive: true)
                .Single(body => body.name == bodyName);
        }

        private static unsafe void SaveCompiledModel(MjScene scene, string path)
        {
            MjEngineTool.SaveModelToFile(path, scene.Model);
        }

        private static unsafe double ReadRootPosition(MjScene scene, int axis)
        {
            int jointId = MujocoLib.mj_name2id(
                scene.Model,
                (int)MujocoLib.mjtObj.mjOBJ_JOINT,
                "trunk_base_freejoint");
            return scene.Data->qpos[scene.Model->jnt_qposadr[jointId] + axis];
        }

        private static unsafe double ReadBodyUpright(MjScene scene, string bodyName)
        {
            int bodyId = MujocoLib.mj_name2id(
                scene.Model,
                (int)MujocoLib.mjtObj.mjOBJ_BODY,
                bodyName);
            return scene.Data->xmat[(9 * bodyId) + 8];
        }

        private static unsafe double ReadRootLinearVelocityX(MjScene scene)
        {
            int jointId = MujocoLib.mj_name2id(
                scene.Model,
                (int)MujocoLib.mjtObj.mjOBJ_JOINT,
                "trunk_base_freejoint");
            Assert.That(jointId, Is.GreaterThanOrEqualTo(0));
            return scene.Data->qvel[scene.Model->jnt_dofadr[jointId]];
        }

        private static unsafe void SetRootPositionX(MjScene scene, double value)
        {
            int jointId = MujocoLib.mj_name2id(
                scene.Model,
                (int)MujocoLib.mjtObj.mjOBJ_JOINT,
                "trunk_base_freejoint");
            Assert.That(jointId, Is.GreaterThanOrEqualTo(0));
            scene.Data->qpos[scene.Model->jnt_qposadr[jointId]] = value;
            MujocoLib.mj_forward(scene.Model, scene.Data);
        }

        private static unsafe void InjectRootLinearVelocity(
            MjScene scene,
            double metresPerSecond)
        {
            int jointId = MujocoLib.mj_name2id(
                scene.Model,
                (int)MujocoLib.mjtObj.mjOBJ_JOINT,
                "trunk_base_freejoint");
            Assert.That(jointId, Is.GreaterThanOrEqualTo(0));
            int dofAddress = scene.Model->jnt_dofadr[jointId];
            scene.Data->qvel[dofAddress] = metresPerSecond;
        }

        private static unsafe void AssertBodyTransformSynchronized(
            MjScene scene,
            MjBody body)
        {
            Assert.That(body.MujocoId, Is.GreaterThanOrEqualTo(0));
            Vector3 nativePosition = MjEngineTool.UnityVector3(
                MjEngineTool.MjVector3AtEntry(scene.Data->xpos, body.MujocoId));
            Assert.That(
                Vector3.Distance(body.transform.position, nativePosition),
                Is.LessThan(1e-5f),
                "MjBody visual transform must match the current native xpos after every step.");
        }

        private static unsafe void AssertNativeObjectName(
            MjScene scene,
            MujocoLib.mjtObj objectType,
            string objectName)
        {
            Assert.That(
                MujocoLib.mj_name2id(scene.Model, (int)objectType, objectName),
                Is.GreaterThanOrEqualTo(0),
                $"Compiled MuJoCo model must preserve official {objectType} name '{objectName}'.");
        }

        private static unsafe string DescribeNativeNames(MjScene scene)
        {
            if (scene == null || (IntPtr)scene.Model == IntPtr.Zero)
            {
                return "MuJoCo model is null.";
            }

            string[] joints = Enumerable.Range(0, checked((int)scene.Model->njnt))
                .Select(index => Marshal.PtrToStringAnsi(MujocoLib.mj_id2name(
                    scene.Model,
                    (int)MujocoLib.mjtObj.mjOBJ_JOINT,
                    index)) ?? "<unnamed>")
                .ToArray();
            string[] bodies = Enumerable.Range(0, checked((int)scene.Model->nbody))
                .Select(index => Marshal.PtrToStringAnsi(MujocoLib.mj_id2name(
                    scene.Model,
                    (int)MujocoLib.mjtObj.mjOBJ_BODY,
                    index)) ?? "<unnamed>")
                .ToArray();
            return $"Compiled joints=[{string.Join(", ", joints)}]; "
                + $"bodies=[{string.Join(", ", bodies)}].";
        }

        private readonly struct NativeModelSnapshot : IEquatable<NativeModelSnapshot>
        {
            public NativeModelSnapshot(
                IntPtr modelAddress,
                IntPtr dataAddress,
                ulong nq,
                ulong nv,
                ulong nu,
                bool hasPassiveWheel)
            {
                this.modelAddress = modelAddress;
                this.dataAddress = dataAddress;
                this.nq = nq;
                this.nv = nv;
                this.nu = nu;
                this.hasPassiveWheel = hasPassiveWheel;
            }

            private readonly IntPtr modelAddress;
            private readonly IntPtr dataAddress;
            public readonly ulong nq;
            public readonly ulong nv;
            public readonly ulong nu;
            public readonly bool hasPassiveWheel;

            public bool Equals(NativeModelSnapshot other)
            {
                return modelAddress == other.modelAddress
                    && dataAddress == other.dataAddress
                    && nq == other.nq
                    && nv == other.nv
                    && nu == other.nu
                    && hasPassiveWheel == other.hasPassiveWheel;
            }

            public override bool Equals(object obj)
            {
                return obj is NativeModelSnapshot other && Equals(other);
            }

            public override int GetHashCode()
            {
                unchecked
                {
                    int hash = modelAddress.GetHashCode();
                    hash = (hash * 397) ^ dataAddress.GetHashCode();
                    hash = (hash * 397) ^ nq.GetHashCode();
                    hash = (hash * 397) ^ nv.GetHashCode();
                    hash = (hash * 397) ^ nu.GetHashCode();
                    hash = (hash * 397) ^ hasPassiveWheel.GetHashCode();
                    return hash;
                }
            }
        }
    }
}
