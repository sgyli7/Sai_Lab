using System;
using System.Linq;
using NUnit.Framework;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class RolloutTraceTests
    {
        [Test]
        public void ParsesVersionedHeaderAndDeterministicFrames()
        {
            string jsonl = Header() + "\n" + Frame(0, 0.005f) + "\n" + Frame(1, 0.01f) + "\n";

            RolloutTrace trace = RolloutTraceJson.Parse(jsonl);

            Assert.That(trace.Header.policyName, Is.EqualTo("alpha_stand.onnx"));
            Assert.That(trace.Header.observationSize, Is.EqualTo(61));
            Assert.That(trace.Frames, Has.Length.EqualTo(2));
            Assert.That(trace.Frames[1].physicsStep, Is.EqualTo(1));
            Assert.That(trace.Frames[1].timeSeconds, Is.EqualTo(0.01f).Within(1e-7f));
            Assert.That(trace.Frames[0].jointPositionRad, Has.Length.EqualTo(14));
        }

        [Test]
        public void AcceptsOptionalFinalResultFromMuJoCoTrace()
        {
            string result = "{\"recordType\":\"result\",\"passed\":true}";
            string jsonl = Header() + "\n" + Frame(0, 0.005f) + "\n" + result + "\n";

            RolloutTrace trace = RolloutTraceJson.Parse(jsonl);

            Assert.That(trace.Frames, Has.Length.EqualTo(1));
            Assert.That(trace.Frames[0].physicsStep, Is.EqualTo(0));
        }

        [Test]
        public void RejectsActionWidthDriftBeforeReplay()
        {
            string badFrame = Frame(0, 0.005f).Replace(
                "\"rawAction\":[" + Values(14) + "]",
                "\"rawAction\":[" + Values(13) + "]");

            TraceValidationException error = Assert.Throws<TraceValidationException>(
                () => RolloutTraceJson.Parse(Header() + "\n" + badFrame));

            StringAssert.Contains("rawAction", error.Message);
        }

        [Test]
        public void RejectsNonMonotonicPhysicsSteps()
        {
            string jsonl = Header() + "\n" + Frame(1, 0.005f) + "\n" + Frame(1, 0.01f);

            TraceValidationException error = Assert.Throws<TraceValidationException>(
                () => RolloutTraceJson.Parse(jsonl));

            StringAssert.Contains("physicsStep", error.Message);
        }

        private static string Header()
        {
            return "{\"recordType\":\"header\",\"schemaVersion\":1,\"policyName\":\"alpha_stand.onnx\"," +
                   "\"policySha256\":\"" + new string('a', 64) + "\",\"scene\":\"scene.xml\"," +
                   "\"sceneSha256\":\"" + new string('b', 64) + "\",\"role\":\"stand\"," +
                   "\"robotVariant\":\"legged\",\"physicsTimestepSeconds\":0.005," +
                   "\"policyDecimation\":4,\"observationSize\":61,\"actionSize\":14," +
                   "\"servoNames\":[" + QuotedValues(14) + "],\"passiveWheelNames\":[]}";
        }

        private static string Frame(int physicsStep, float time)
        {
            return "{\"recordType\":\"frame\",\"physicsStep\":" + physicsStep +
                   ",\"policyStep\":0,\"timeSeconds\":" + time.ToString("R", System.Globalization.CultureInfo.InvariantCulture) +
                   ",\"rootPosition\":[0,0,0.125],\"rootQuaternionWxyz\":[1,0,0,0]," +
                   "\"rootLinearVelocity\":[0,0,0],\"rootAngularVelocity\":[0,0,0]," +
                   "\"jointPositionRad\":[" + Values(14) + "],\"jointVelocityRadPerSecond\":[" + Values(14) + "]," +
                   "\"passiveWheelVelocityRadPerSecond\":[],\"observation\":[" + Values(61) + "]," +
                   "\"rawAction\":[" + Values(14) + "],\"targetPositionRad\":[" + Values(14) + "]," +
                   "\"command\":[" + Values(13) + "],\"contacts\":[]}";
        }

        private static string Values(int count)
        {
            return string.Join(",", Enumerable.Repeat("0", count));
        }

        private static string QuotedValues(int count)
        {
            return string.Join(",", Enumerable.Range(0, count).Select(index => "\"servo" + index + "\""));
        }
    }
}
