using NUnit.Framework;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class CoordinateBasisTests
    {
        [Test]
        public void MuJoCoVectorMapsToNegativeYZX()
        {
            Vector3 actual = CoordinateBasis.MuJoCoToTuanjie(new Vector3(1f, 2f, 3f));

            Assert.That(actual, Is.EqualTo(new Vector3(-2f, 3f, 1f)));
        }

        [Test]
        public void IdentityRotationRemainsIdentity()
        {
            Quaternion actual = CoordinateBasis.MuJoCoToTuanjie(Quaternion.identity);

            Assert.That(Quaternion.Angle(actual, Quaternion.identity), Is.LessThan(1e-5f));
        }

        [Test]
        public void PositiveMuJoCoZRotationBecomesNegativeTuanjieYRotation()
        {
            Quaternion mujoco = Quaternion.AngleAxis(90f, Vector3.forward);
            Quaternion actual = CoordinateBasis.MuJoCoToTuanjie(mujoco);
            Quaternion expected = Quaternion.AngleAxis(-90f, Vector3.up);

            Assert.That(Quaternion.Angle(actual, expected), Is.LessThan(1e-4f));
        }

        [Test]
        public void QuaternionRoundTripPreservesRotation()
        {
            Quaternion mujoco = Quaternion.Euler(23f, -41f, 67f);

            Quaternion roundTrip = CoordinateBasis.TuanjieToMuJoCo(
                CoordinateBasis.MuJoCoToTuanjie(mujoco));

            Assert.That(Quaternion.Angle(roundTrip, mujoco), Is.LessThan(1e-4f));
        }

        [Test]
        public void AxialVectorPreservesPositiveRotationAcrossHandednessChange()
        {
            Vector3 actual = CoordinateBasis.MuJoCoToTuanjieAxial(Vector3.forward);

            Assert.That(actual, Is.EqualTo(Vector3.down));
            Assert.That(
                Quaternion.Angle(
                    Quaternion.AngleAxis(90f, actual),
                    CoordinateBasis.MuJoCoToTuanjie(
                        Quaternion.AngleAxis(90f, Vector3.forward))),
                Is.LessThan(1e-4f));
        }
    }
}
