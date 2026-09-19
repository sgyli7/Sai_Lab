using AgenticRobot.MicroDuck.Mujoco;
using NUnit.Framework;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class MujocoPlayerSmokeProbeTests
    {
        [Test]
        public void ParsesOnlyAnExplicitNonEmptySmokeReportArgument()
        {
            Assert.That(
                MujocoPlayerSmokeProbe.TryGetReportPath(
                    new[] { "player.exe", "-microduckSmokeReport", "C:/temp/smoke.json" },
                    out string reportPath),
                Is.True);
            Assert.That(reportPath, Is.EqualTo("C:/temp/smoke.json"));

            Assert.That(
                MujocoPlayerSmokeProbe.TryGetReportPath(
                    new[] { "player.exe", "-microduckSmokeReport" },
                    out _),
                Is.False);
            Assert.That(
                MujocoPlayerSmokeProbe.TryGetReportPath(
                    new[] { "player.exe", "-unrelated", "value" },
                    out _),
                Is.False);
        }

        [Test]
        public void FiniteContractRejectsNanInfinityAndWrongWidths()
        {
            Assert.That(
                MujocoPlayerSmokeProbe.AllFinite(new float[61], 61),
                Is.True);
            Assert.That(
                MujocoPlayerSmokeProbe.AllFinite(new float[60], 61),
                Is.False);
            var invalid = new float[14];
            invalid[5] = float.NaN;
            Assert.That(MujocoPlayerSmokeProbe.AllFinite(invalid, 14), Is.False);
            invalid[5] = float.PositiveInfinity;
            Assert.That(MujocoPlayerSmokeProbe.AllFinite(invalid, 14), Is.False);
        }
    }
}
