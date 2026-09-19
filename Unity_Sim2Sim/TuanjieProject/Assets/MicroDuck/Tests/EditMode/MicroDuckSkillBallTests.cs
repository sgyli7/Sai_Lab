using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class MicroDuckSkillBallTests
    {
        private readonly List<GameObject> createdObjects = new List<GameObject>();

        [TearDown]
        public void TearDown()
        {
            foreach (GameObject createdObject in createdObjects)
            {
                if (createdObject != null)
                {
                    Object.DestroyImmediate(createdObject);
                }
            }

            createdObjects.Clear();
        }

        [TestCase(PolicyRole.KickLeft, -1f)]
        [TestCase(PolicyRole.KickRight, 1f)]
        public void ResetForKickUsesTheCanonicalMuJoCoPlacementAndOfficialBallProperties(
            PolicyRole role,
            float unityRightSign)
        {
            Transform root = CreateObject("Robot Root").transform;
            root.position = new Vector3(1.2f, 0.125f, -0.7f);
            root.rotation = Quaternion.Euler(0f, 37f, 0f);
            MicroDuckSkillBall ball = CreateBall();
            ball.Body.velocity = new Vector3(1f, 2f, 3f);
            ball.Body.angularVelocity = new Vector3(4f, 5f, 6f);

            ball.ResetForKick(root, role);

            Vector3 expected = root.position
                + root.forward * MicroDuckSkillBall.ForwardOffsetMeters
                + root.right * (unityRightSign * MicroDuckSkillBall.LateralOffsetMeters);
            expected.y = MicroDuckSkillBall.RadiusMeters;
            Assert.That(ball.gameObject.activeSelf, Is.True);
            Assert.That(Vector3.Distance(ball.Position, expected), Is.LessThan(1e-6f));
            Assert.That(ball.Velocity, Is.EqualTo(Vector3.zero));
            Assert.That(ball.AngularVelocity, Is.EqualTo(Vector3.zero));
            Assert.That(ball.Body.mass,
                Is.EqualTo(MicroDuckSkillBall.MassKilograms).Within(1e-7f));
            Assert.That(ball.Collider.radius,
                Is.EqualTo(MicroDuckSkillBall.RadiusMeters).Within(1e-7f));
            Assert.That(ball.Body.inertiaTensor.x,
                Is.EqualTo(MicroDuckSkillBall.MomentOfInertiaKilogramMetersSquared).Within(1e-9f));
            Assert.That(ball.Body.inertiaTensor.y,
                Is.EqualTo(MicroDuckSkillBall.MomentOfInertiaKilogramMetersSquared).Within(1e-9f));
            Assert.That(ball.Body.inertiaTensor.z,
                Is.EqualTo(MicroDuckSkillBall.MomentOfInertiaKilogramMetersSquared).Within(1e-9f));
            Assert.That(ball.HasNonFloorContact, Is.False);
            Assert.That(ball.FirstNonFloorContactColliderName, Is.Empty);
            Assert.That(ball.FirstNonFloorContactTimeSeconds, Is.EqualTo(-1f));
        }

        [Test]
        public void ContactProbeIgnoresTheConfiguredFloorAndKeepsTheFirstOtherContact()
        {
            Collider floor = CreateObject("Ground").AddComponent<BoxCollider>();
            Collider leftFoot = CreateObject("ankle_left_collision").AddComponent<BoxCollider>();
            Collider rightFoot = CreateObject("ankle_right_collision").AddComponent<BoxCollider>();
            MicroDuckSkillBall ball = CreateBall(floor);

            ball.ObserveContact(floor, new Vector3(0f, 0f, 0f), 0.1f);
            Assert.That(ball.HasNonFloorContact, Is.False);

            Vector3 firstPoint = new Vector3(0.01f, 0.04f, 0.08f);
            ball.ObserveContact(leftFoot, firstPoint, 0.24f);
            ball.ObserveContact(rightFoot, new Vector3(1f, 2f, 3f), 0.5f);

            Assert.That(ball.HasNonFloorContact, Is.True);
            Assert.That(ball.FirstNonFloorContactColliderName, Is.EqualTo(leftFoot.name));
            Assert.That(ball.FirstNonFloorContactTimeSeconds, Is.EqualTo(0.24f).Within(1e-7f));
            Assert.That(ball.FirstNonFloorContactPoint, Is.EqualTo(firstPoint));
        }

        [Test]
        public void HideDisablesTheBallAndClearsItsMotion()
        {
            MicroDuckSkillBall ball = CreateBall();
            ball.Body.velocity = Vector3.one;
            ball.Body.angularVelocity = Vector3.one;

            ball.Hide();

            Assert.That(ball.gameObject.activeSelf, Is.False);
            Assert.That(ball.Velocity, Is.EqualTo(Vector3.zero));
            Assert.That(ball.AngularVelocity, Is.EqualTo(Vector3.zero));
        }

        private MicroDuckSkillBall CreateBall(Collider floor = null)
        {
            GameObject ballObject = CreateObject("MicroDuck Skill Ball");
            Rigidbody body = ballObject.AddComponent<Rigidbody>();
            SphereCollider sphere = ballObject.AddComponent<SphereCollider>();
            MicroDuckSkillBall ball = ballObject.AddComponent<MicroDuckSkillBall>();
            ball.Configure(body, sphere, floor);
            return ball;
        }

        private GameObject CreateObject(string name)
        {
            var createdObject = new GameObject(name);
            createdObjects.Add(createdObject);
            return createdObject;
        }
    }
}
