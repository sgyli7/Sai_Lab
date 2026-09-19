using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class KickBallPlayModeTests
    {
        [UnityTest]
        public IEnumerator GeneratedSceneShowsAndResetsTheBallOnlyForKickPolicies()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;

            MicroDuckDemoController controller = Object.FindObjectOfType<MicroDuckDemoController>();
            Assert.That(controller, Is.Not.Null);
            Assert.That(controller.SkillBall, Is.Not.Null);
            Assert.That(controller.SkillBall.gameObject.activeSelf, Is.False);

            Assert.That(controller.SelectPolicy(5), Is.True, controller.Fault);
            AssertCanonicalPlacement(controller, unityRightSign: -1f);
            controller.SkillBall.Body.velocity = Vector3.one;
            controller.SkillBall.Body.angularVelocity = Vector3.one;

            Assert.That(controller.SelectPolicy(6), Is.True, controller.Fault);
            AssertCanonicalPlacement(controller, unityRightSign: 1f);
            Assert.That(controller.SkillBall.Velocity, Is.EqualTo(Vector3.zero));
            Assert.That(controller.SkillBall.AngularVelocity, Is.EqualTo(Vector3.zero));

            Assert.That(controller.SelectPolicy(2), Is.True, controller.Fault);
            Assert.That(controller.SkillBall.gameObject.activeSelf, Is.False);
        }

        [UnityTest]
        public IEnumerator BothKickPoliciesStrikeTheBallAndRemainStanding()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;

            MicroDuckDemoController controller = Object.FindObjectOfType<MicroDuckDemoController>();
            Assert.That(controller, Is.Not.Null);
            controller.enabled = false;

            bool originalAutoSimulation = Physics.autoSimulation;
            Physics.autoSimulation = false;
            try
            {
                AssertKick(controller, slot: 5, expectedFoot: "left_foot_collision");
                AssertKick(controller, slot: 6, expectedFoot: "right_foot_collision");
            }
            finally
            {
                Physics.autoSimulation = originalAutoSimulation;
            }
        }

        private static void AssertKick(
            MicroDuckDemoController controller,
            int slot,
            string expectedFoot)
        {
            Assert.That(controller.SelectPolicy(slot), Is.True, controller.Fault);
            Physics.Simulate(1e-6f);
            Vector3 initialBall = controller.SkillBall.Position;
            Vector3 initialRoot = controller.ActiveRig.RootBody.transform.position;
            Vector3 initialForward = Vector3.ProjectOnPlane(
                controller.ActiveRig.RootBody.transform.forward,
                Vector3.up).normalized;

            const float timestep = 0.005f;
            const int physicsSteps = 600;
            for (int step = 0; step < physicsSteps; step++)
            {
                float now = step * timestep;
                if (step % 4 == 0)
                {
                    Assert.That(controller.TickOnce(now), Is.True, controller.Fault);
                }

                controller.SkillBall.SetObservationTime(now);
                Physics.Simulate(timestep);
            }

            Vector3 ballDelta = controller.SkillBall.Position - initialBall;
            Vector3 rootDelta = controller.ActiveRig.RootBody.transform.position - initialRoot;
            float ballForward = Vector3.Dot(ballDelta, initialForward);
            float rootDrift = Vector3.ProjectOnPlane(rootDelta, Vector3.up).magnitude;
            float finalUpright = Vector3.Dot(
                controller.ActiveRig.RootBody.transform.up,
                Vector3.up);
            string evidence = $"slot={slot}, contact={controller.SkillBall.FirstNonFloorContactColliderName}, "
                + $"contactTime={controller.SkillBall.FirstNonFloorContactTimeSeconds:R}, "
                + $"ballDistance={ballDelta.magnitude:R}, ballForward={ballForward:R}, "
                + $"rootDrift={rootDrift:R}, finalUpright={finalUpright:R}";
            TestContext.Progress.WriteLine(evidence);

            Assert.That(controller.SkillBall.HasNonFloorContact, Is.True, evidence);
            Assert.That(controller.SkillBall.FirstNonFloorContactColliderName,
                Is.EqualTo(expectedFoot), evidence);
            Assert.That(controller.SkillBall.FirstNonFloorContactTimeSeconds,
                Is.InRange(0.05f, 0.35f), evidence);
            Assert.That(ballForward, Is.GreaterThan(0.1f), evidence);
            Assert.That(ballDelta.magnitude, Is.GreaterThan(0.25f), evidence);
            Assert.That(rootDrift, Is.LessThan(0.12f), evidence);
            Assert.That(finalUpright, Is.GreaterThan(0.8f), evidence);
        }

        private static void AssertCanonicalPlacement(
            MicroDuckDemoController controller,
            float unityRightSign)
        {
            Transform root = controller.ActiveRig.RootBody.transform;
            Vector3 forward = Vector3.ProjectOnPlane(root.forward, Vector3.up).normalized;
            Vector3 right = Vector3.ProjectOnPlane(root.right, Vector3.up).normalized;
            Vector3 expected = root.position
                + forward * MicroDuckSkillBall.ForwardOffsetMeters
                + right * (unityRightSign * MicroDuckSkillBall.LateralOffsetMeters);
            expected.y = MicroDuckSkillBall.RadiusMeters;

            Assert.That(controller.SkillBall.gameObject.activeSelf, Is.True);
            Assert.That(
                Vector3.Distance(controller.SkillBall.Position, expected),
                Is.LessThan(1e-5f));
        }
    }
}
