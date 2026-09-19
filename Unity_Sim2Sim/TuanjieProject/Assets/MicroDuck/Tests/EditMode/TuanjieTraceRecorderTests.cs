using System.IO;
using System.Linq;
using NUnit.Framework;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class TuanjieTraceRecorderTests
    {
        [Test]
        public void WritesVersionOneTuanjieJsonlThatTheSharedParserAccepts()
        {
            var header = new RolloutTraceHeaderData
            {
                recordType = "header",
                schemaVersion = 1,
                producer = "tuanjie",
                policyName = "alpha_stand.onnx",
                policySha256 = new string('a', 64),
                scene = "MicroDuckMvp.unity",
                sceneSha256 = new string('b', 64),
                role = "stand",
                robotVariant = "legged",
                physicsTimestepSeconds = 0.005f,
                policyDecimation = 4,
                observationSize = 61,
                actionSize = 14,
                servoNames = Enumerable.Range(0, 14).Select(index => $"servo{index}").ToArray(),
                passiveWheelNames = new string[0],
            };
            var frame = new RolloutTraceFrameData
            {
                recordType = "frame",
                physicsStep = 0,
                policyStep = 0,
                timeSeconds = 0.005f,
                rootPosition = new float[3],
                rootQuaternionWxyz = new[] { 1f, 0f, 0f, 0f },
                rootLinearVelocity = new float[3],
                rootAngularVelocity = new float[3],
                jointPositionRad = new float[14],
                jointVelocityRadPerSecond = new float[14],
                passiveWheelVelocityRadPerSecond = new float[0],
                observation = new float[61],
                rawAction = new float[14],
                targetPositionRad = new float[14],
                command = new float[13],
                contacts = new TraceContactData[0],
            };

            var recorder = new TuanjieTraceRecorder(header);
            recorder.AppendFrame(frame);
            string jsonl = recorder.Complete(passed: true);
            RolloutTrace replay = RolloutTraceJson.Parse(jsonl);

            StringAssert.StartsWith("{\"recordType\":\"header\"", jsonl);
            StringAssert.Contains("\"producer\":\"tuanjie\"", jsonl);
            StringAssert.EndsWith("{\"recordType\":\"result\",\"passed\":true}\n", jsonl);
            Assert.That(replay.Header.producer, Is.EqualTo("tuanjie"));
            Assert.That(replay.Frames, Has.Length.EqualTo(1));
        }

        [Test]
        public void SharedTraceNamesAndRolesMatchBothGeneratedManifestContracts()
        {
            string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
            Assert.That(projectRoot, Is.Not.Null);
            RobotManifestData legged = RobotManifestJson.Parse(File.ReadAllText(Path.Combine(
                projectRoot,
                "Assets/MicroDuck/Generated/Manifests/legged.json")));
            RobotManifestData roller = RobotManifestJson.Parse(File.ReadAllText(Path.Combine(
                projectRoot,
                "Assets/MicroDuck/Generated/Manifests/roller.json")));

            string[] officialServoNames = PolicyContract.ServoNames;
            Assert.That(
                legged.servos.OrderBy(servo => servo.index).Select(servo => servo.name),
                Is.EqualTo(officialServoNames));
            Assert.That(
                roller.servos.OrderBy(servo => servo.index).Select(servo => servo.name),
                Is.EqualTo(officialServoNames));
            Assert.That(roller.passiveJointNames, Is.EqualTo(PolicyContract.RollerPassiveWheelNames));

            string[] expectedRoles =
            {
                "walk", "stand", "sitstand", "ground-pick", "kick-left",
                "kick-right", "roller", "roller-crouch", "roulade",
            };
            Assert.That(
                PolicyCatalog.Entries.Select(entry => PolicyCatalog.ToTraceRoleName(entry.Role)),
                Is.EqualTo(expectedRoles));
        }
    }
}
