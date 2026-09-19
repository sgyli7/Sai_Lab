using System.IO;
using NUnit.Framework;

namespace AgenticRobot.MicroDuck.Editor.Tests
{
    public sealed class TraceBatchExporterTests
    {
        [Test]
        public void ExportsARealStandPolicyTickToTheMvpArtifactDirectory()
        {
            string outputPath = TraceBatchExporter.ExportOnePolicyTrace();

            Assert.That(File.Exists(outputPath), Is.True, outputPath);
            StringAssert.EndsWith(
                "artifacts/mvp/traces/tuanjie-alpha_stand.jsonl",
                outputPath.Replace('\\', '/'));
            RolloutTrace trace = RolloutTraceJson.Parse(File.ReadAllText(outputPath));
            Assert.That(trace.Header.producer, Is.EqualTo("tuanjie"));
            Assert.That(trace.Header.policyName, Is.EqualTo("alpha_stand.onnx"));
            Assert.That(trace.Header.policySha256, Has.Length.EqualTo(64));
            Assert.That(trace.Frames, Has.Length.EqualTo(1));
            Assert.That(trace.Frames[0].observation, Has.Length.EqualTo(61));
            Assert.That(trace.Frames[0].rawAction, Has.Length.EqualTo(14));
        }
    }
}
