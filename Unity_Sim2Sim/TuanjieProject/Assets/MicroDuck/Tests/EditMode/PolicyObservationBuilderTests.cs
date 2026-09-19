using NUnit.Framework;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class PolicyObservationBuilderTests
    {
        [Test]
        public void BuildsTheExactSixtyOneValueLayoutInMuJoCoCoordinates()
        {
            var positions = new float[14];
            var home = new float[14];
            var velocities = new float[14];
            var previousAction = new float[14];
            var command = new float[13];
            for (int index = 0; index < 14; index++)
            {
                home[index] = index;
                positions[index] = index + 0.25f;
                velocities[index] = 100f + index;
                previousAction[index] = 200f + index;
            }

            for (int index = 0; index < 13; index++)
            {
                command[index] = 300f + index;
            }

            var result = new float[61];
            PolicyObservationBuilder.Build(
                CoordinateBasis.MuJoCoToTuanjieAxial(new Vector3(1f, 2f, 3f)),
                CoordinateBasis.MuJoCoToTuanjie(new Vector3(4f, 5f, 6f)),
                positions,
                velocities,
                home,
                previousAction,
                command,
                result);

            Assert.That(result[0..3], Is.EqualTo(new[] { 1f, 2f, 3f }));
            Assert.That(result[3..6], Is.EqualTo(new[] { 4f, 5f, 6f }));
            Assert.That(result[6..20], Is.All.EqualTo(0.25f));
            Assert.That(result[20..34], Is.EqualTo(velocities));
            Assert.That(result[34..48], Is.EqualTo(previousAction));
            Assert.That(result[48..61], Is.EqualTo(command));
        }

        [Test]
        public void NativeMuJoCoPathAcceptsPolicyBasisVectorsWithoutASecondConversion()
        {
            var positions = new float[14];
            var home = new float[14];
            var velocities = new float[14];
            var previousAction = new float[14];
            var command = new float[13];
            positions[0] = 0.75f;
            home[0] = 0.25f;
            velocities[1] = -2f;
            previousAction[2] = 3f;
            command[12] = 4f;

            var result = new float[61];
            PolicyObservationBuilder.BuildMuJoCo(
                new Vector3(1f, 2f, 3f),
                new Vector3(4f, 5f, 6f),
                positions,
                velocities,
                home,
                previousAction,
                command,
                result);

            Assert.That(result[0..6], Is.EqualTo(new[] { 1f, 2f, 3f, 4f, 5f, 6f }));
            Assert.That(result[6], Is.EqualTo(0.5f));
            Assert.That(result[21], Is.EqualTo(-2f));
            Assert.That(result[36], Is.EqualTo(3f));
            Assert.That(result[60], Is.EqualTo(4f));
        }

        [Test]
        public void RejectsEveryIncorrectBufferWidth()
        {
            var fourteen = new float[14];
            var thirteen = new float[13];
            var sixtyOne = new float[61];

            Assert.Throws<System.ArgumentException>(() => PolicyObservationBuilder.Build(
                Vector3.zero, Vector3.down, new float[13], fourteen, fourteen,
                fourteen, thirteen, sixtyOne));
            Assert.Throws<System.ArgumentException>(() => PolicyObservationBuilder.Build(
                Vector3.zero, Vector3.down, fourteen, fourteen, fourteen,
                fourteen, thirteen, new float[60]));
        }
    }
}
