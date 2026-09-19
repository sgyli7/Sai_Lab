using System.Linq;
using System.Reflection;
using AgenticRobot.MicroDuck.Mujoco;
using NUnit.Framework;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class MujocoPlayerAcceptanceTourTests
    {
        [Test]
        public void ParsesOnlyExplicitTourReportAndFrameDirectoryArguments()
        {
            Assert.That(
                MujocoPlayerAcceptanceTour.TryGetArgument(
                    new[] { "player", "-microduckTourReport", "/tmp/tour.json" },
                    MujocoPlayerAcceptanceTour.ReportArgument,
                    out string reportPath),
                Is.True);
            Assert.That(reportPath, Is.EqualTo("/tmp/tour.json"));

            Assert.That(
                MujocoPlayerAcceptanceTour.TryGetArgument(
                    new[] { "player", "-microduckTourFrames", "/tmp/frames" },
                    MujocoPlayerAcceptanceTour.FramesArgument,
                    out string framesPath),
                Is.True);
            Assert.That(framesPath, Is.EqualTo("/tmp/frames"));

            Assert.That(
                MujocoPlayerAcceptanceTour.TryGetArgument(
                    new[] { "player", "-microduckTourReport" },
                    MujocoPlayerAcceptanceTour.ReportArgument,
                    out _),
                Is.False);
            Assert.That(
                MujocoPlayerAcceptanceTour.TryGetArgument(
                    new[] { "player", "-unrelated", "value" },
                    MujocoPlayerAcceptanceTour.ReportArgument,
                    out _),
                Is.False);
        }

        [Test]
        public void WalkingThresholdsMatchAlphaWalkingScenarioContract()
        {
            Assert.That(MujocoPlayerAcceptanceTour.WalkingMinForwardMeters, Is.EqualTo(0.03f));
            Assert.That(MujocoPlayerAcceptanceTour.WalkingMinUprightDot, Is.EqualTo(0.25f));
            Assert.That(MujocoPlayerAcceptanceTour.WalkingCommandForward, Is.EqualTo(0.2f));
            Assert.That(MujocoPlayerAcceptanceTour.WalkingDurationSeconds, Is.EqualTo(6f));
            Assert.That(MujocoPlayerAcceptanceTour.ResetMaxHorizontalMeters, Is.EqualTo(0.05f));
        }

        [Test]
        public void EventTableCoversSevenTerrainsAndFiveCameraModesInWindowsOrder()
        {
            DefaultExecutionOrder order = typeof(MujocoPlayerAcceptanceTour)
                .GetCustomAttribute<DefaultExecutionOrder>();
            Assert.That(order, Is.Not.Null);
            Assert.That(order.order, Is.EqualTo(1001));

            string names = string.Join(" ", MujocoPlayerAcceptanceTour.Events.Select(item => item.Name));
            string[] terrains =
            {
                "flat_plaza",
                "upstream_pyramid_stairs",
                "upstream_random_grid",
                "upstream_pyramid_slope",
                "upstream_roller_slope",
                "rock_steps",
                "stairs_bridge",
            };
            string[] cameras = { "Side", "Rear", "Top", "Showcase", "FreeFly" };
            foreach (string terrain in terrains)
            {
                Assert.That(names, Does.Contain(terrain), terrain);
            }

            foreach (string camera in cameras)
            {
                Assert.That(names, Does.Contain(camera), camera);
            }

            Assert.That(MujocoPlayerAcceptanceTour.Events[0].At, Is.EqualTo(0.2f));
            Assert.That(MujocoPlayerAcceptanceTour.Events[0].Kind, Is.EqualTo("press"));
            Assert.That(
                MujocoPlayerAcceptanceTour.Events.Last().Name,
                Is.EqualTo("final upright reset on flat_plaza"));
            Assert.That(
                MujocoPlayerAcceptanceTour.Events.Count(item => item.Kind == "orbit"),
                Is.EqualTo(1));
            Assert.That(
                MujocoPlayerAcceptanceTour.Events.Count(item => item.Kind == "wheel"),
                Is.EqualTo(1));
        }
    }
}
