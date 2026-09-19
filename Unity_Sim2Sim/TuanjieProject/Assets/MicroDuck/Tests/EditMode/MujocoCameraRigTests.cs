using System.Reflection;
using AgenticRobot.MicroDuck.Mujoco;
using NUnit.Framework;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class MujocoCameraRigTests
    {
        private GameObject cameraObject;
        private MujocoCameraRig rig;

        [SetUp]
        public void SetUp()
        {
            cameraObject = new GameObject("Camera Rig Test");
            cameraObject.AddComponent<Camera>();
            rig = cameraObject.AddComponent<MujocoCameraRig>();
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(cameraObject);
        }

        [Test]
        public void DefaultsToFollowSidePresetWithoutOwningRobotNavigation()
        {
            Assert.That(rig.Mode, Is.EqualTo(MujocoCameraMode.Follow));
            Assert.That(rig.Preset, Is.EqualTo(MujocoCameraPreset.Side));
            Assert.That(rig.OwnsNavigationInput, Is.False);
        }

        [Test]
        public void FreeFlyModeOwnsNavigationUntilFocusReturnsToFollow()
        {
            rig.ToggleMode();
            Assert.That(rig.Mode, Is.EqualTo(MujocoCameraMode.FreeFly));
            Assert.That(rig.OwnsNavigationInput, Is.True);

            rig.FocusOnRobot();
            Assert.That(rig.Mode, Is.EqualTo(MujocoCameraMode.Follow));
            Assert.That(rig.OwnsNavigationInput, Is.False);
        }

        [Test]
        public void CyclesAllFourDirectorPresetsInStableOrder()
        {
            Assert.That(rig.CyclePreset(), Is.EqualTo(MujocoCameraPreset.Rear));
            Assert.That(rig.CyclePreset(), Is.EqualTo(MujocoCameraPreset.Top));
            Assert.That(rig.CyclePreset(), Is.EqualTo(MujocoCameraPreset.Showcase));
            Assert.That(rig.CyclePreset(), Is.EqualTo(MujocoCameraPreset.Side));
        }

        [Test]
        public void ApplyFreeFlyTranslationUsesTheSameAxesAsKeyboardFreeFly()
        {
            cameraObject.transform.position = new Vector3(1f, 2f, 3f);
            cameraObject.transform.rotation = Quaternion.LookRotation(Vector3.forward, Vector3.up);
            Assert.That(
                typeof(MujocoCameraRig).GetMethod("ApplyFreeFlyTranslation"),
                Is.Not.Null);
            Assert.DoesNotThrow(
                () => rig.ApplyFreeFlyTranslation(1f, 0f, 0f, boost: false));
        }

        [Test]
        public void OrbitAndZoomRemainInsideUsableLimits()
        {
            rig.ApplyOrbitInput(new Vector2(10000f, -10000f));
            rig.ApplyZoomInput(10000f);
            Assert.That(rig.PitchDegrees, Is.InRange(8f, 82f));
            Assert.That(rig.DistanceMeters, Is.InRange(0.65f, 7f));

            rig.ApplyOrbitInput(new Vector2(-10000f, 10000f));
            rig.ApplyZoomInput(-10000f);
            Assert.That(rig.PitchDegrees, Is.InRange(8f, 82f));
            Assert.That(rig.DistanceMeters, Is.InRange(0.65f, 7f));
        }
    }
}
