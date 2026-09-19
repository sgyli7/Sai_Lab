using System;
using System.Collections.Generic;
using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    public sealed class TraceValidationException : Exception
    {
        public TraceValidationException(string message) : base(message)
        {
        }
    }

    public sealed class RolloutTrace
    {
        public RolloutTrace(RolloutTraceHeaderData header, RolloutTraceFrameData[] frames)
        {
            Header = header;
            Frames = frames;
        }

        public RolloutTraceHeaderData Header { get; }
        public RolloutTraceFrameData[] Frames { get; }
    }

    [Serializable]
    public sealed class RolloutTraceHeaderData
    {
        public string recordType;
        public int schemaVersion;
        public string producer;
        public string policyName;
        public string policySha256;
        public string scene;
        public string sceneSha256;
        public string role;
        public string robotVariant;
        public float physicsTimestepSeconds;
        public int policyDecimation;
        public int observationSize;
        public int actionSize;
        public string[] servoNames;
        public string[] passiveWheelNames;
    }

    [Serializable]
    public sealed class RolloutTraceFrameData
    {
        public string recordType;
        public int physicsStep;
        public int policyStep;
        public float timeSeconds;
        public float[] rootPosition;
        public float[] rootQuaternionWxyz;
        public float[] rootLinearVelocity;
        public float[] rootAngularVelocity;
        public float[] jointPositionRad;
        public float[] jointVelocityRadPerSecond;
        public float[] passiveWheelVelocityRadPerSecond;
        public float[] observation;
        public float[] rawAction;
        public float[] targetPositionRad;
        public float[] command;
        public TraceContactData[] contacts;
    }

    [Serializable]
    public sealed class TraceContactData
    {
        public string geom1;
        public string geom2;
        public float distance;
    }

    [Serializable]
    internal sealed class TraceRecordTypeData
    {
        public string recordType;
    }

    public static class RolloutTraceJson
    {
        public static RolloutTrace Parse(string jsonl)
        {
            if (string.IsNullOrWhiteSpace(jsonl))
            {
                throw new TraceValidationException("Trace JSONL is empty.");
            }

            string[] lines = jsonl.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            if (lines.Length < 2)
            {
                throw new TraceValidationException("Trace must contain a header and at least one frame.");
            }

            RolloutTraceHeaderData header = ParseHeader(lines[0]);
            var frames = new List<RolloutTraceFrameData>(lines.Length - 1);
            int previousPhysicsStep = -1;
            float previousTime = -1f;
            for (int lineIndex = 1; lineIndex < lines.Length; lineIndex++)
            {
                TraceRecordTypeData envelope = ParseJson<TraceRecordTypeData>(lines[lineIndex], lineIndex + 1);
                if (envelope.recordType == "result")
                {
                    if (lineIndex != lines.Length - 1)
                    {
                        throw new TraceValidationException(
                            $"Trace line {lineIndex + 1} result record must be the final line.");
                    }

                    continue;
                }

                RolloutTraceFrameData frame = ParseFrame(lines[lineIndex], lineIndex + 1);
                ValidateFrame(frame, header, lineIndex + 1);
                if (frame.physicsStep <= previousPhysicsStep)
                {
                    throw new TraceValidationException(
                        $"Frame line {lineIndex + 1} physicsStep must increase strictly.");
                }

                if (frame.timeSeconds <= previousTime)
                {
                    throw new TraceValidationException(
                        $"Frame line {lineIndex + 1} timeSeconds must increase strictly.");
                }

                previousPhysicsStep = frame.physicsStep;
                previousTime = frame.timeSeconds;
                frames.Add(frame);
            }

            if (frames.Count == 0)
            {
                throw new TraceValidationException("Trace must contain at least one frame.");
            }

            return new RolloutTrace(header, frames.ToArray());
        }

        private static RolloutTraceHeaderData ParseHeader(string line)
        {
            TraceRecordTypeData envelope = ParseJson<TraceRecordTypeData>(line, 1);
            if (envelope.recordType != "header")
            {
                throw new TraceValidationException("Trace line 1 must be a header record.");
            }

            RolloutTraceHeaderData header = ParseJson<RolloutTraceHeaderData>(line, 1);
            Require(header.schemaVersion == 1, "Unsupported trace schemaVersion.");
            Require(header.observationSize == PolicyContract.ObservationCount, "Header observationSize must be 61.");
            Require(header.actionSize == PolicyContract.ActionCount, "Header actionSize must be 14.");
            Require(header.physicsTimestepSeconds > 0f, "Header physics timestep must be positive.");
            Require(header.policyDecimation > 0, "Header policy decimation must be positive.");
            Require(!string.IsNullOrWhiteSpace(header.policyName), "Header policyName is missing.");
            Require(header.servoNames != null && header.servoNames.Length == PolicyContract.ActionCount,
                "Header servoNames must contain 14 values.");
            Require(header.passiveWheelNames != null, "Header passiveWheelNames is missing.");
            return header;
        }

        private static RolloutTraceFrameData ParseFrame(string line, int lineNumber)
        {
            TraceRecordTypeData envelope = ParseJson<TraceRecordTypeData>(line, lineNumber);
            if (envelope.recordType != "frame")
            {
                throw new TraceValidationException($"Trace line {lineNumber} must be a frame record.");
            }

            return ParseJson<RolloutTraceFrameData>(line, lineNumber);
        }

        private static void ValidateFrame(
            RolloutTraceFrameData frame,
            RolloutTraceHeaderData header,
            int lineNumber)
        {
            RequireFinite(frame.timeSeconds, $"line {lineNumber} timeSeconds");
            RequireArray(frame.rootPosition, 3, $"line {lineNumber} rootPosition");
            RequireArray(frame.rootQuaternionWxyz, 4, $"line {lineNumber} rootQuaternionWxyz");
            RequireArray(frame.rootLinearVelocity, 3, $"line {lineNumber} rootLinearVelocity");
            RequireArray(frame.rootAngularVelocity, 3, $"line {lineNumber} rootAngularVelocity");
            RequireArray(frame.jointPositionRad, PolicyContract.ActionCount, $"line {lineNumber} jointPositionRad");
            RequireArray(frame.jointVelocityRadPerSecond, PolicyContract.ActionCount, $"line {lineNumber} jointVelocityRadPerSecond");
            RequireArray(frame.passiveWheelVelocityRadPerSecond, header.passiveWheelNames.Length,
                $"line {lineNumber} passiveWheelVelocityRadPerSecond");
            RequireArray(frame.observation, PolicyContract.ObservationCount, $"line {lineNumber} observation");
            RequireArray(frame.rawAction, PolicyContract.ActionCount, $"line {lineNumber} rawAction");
            RequireArray(frame.targetPositionRad, PolicyContract.ActionCount, $"line {lineNumber} targetPositionRad");
            RequireArray(frame.command, PolicyObservationBuilder.CommandCount, $"line {lineNumber} command");
            Require(frame.contacts != null, $"line {lineNumber} contacts is missing.");
        }

        private static T ParseJson<T>(string json, int lineNumber)
        {
            try
            {
                T result = JsonUtility.FromJson<T>(json);
                if (result == null)
                {
                    throw new TraceValidationException($"Trace line {lineNumber} deserialized to null.");
                }

                return result;
            }
            catch (ArgumentException error)
            {
                throw new TraceValidationException($"Trace line {lineNumber} is malformed: {error.Message}");
            }
        }

        private static void RequireArray(float[] values, int expected, string label)
        {
            Require(values != null && values.Length == expected, $"{label} must contain {expected} values.");
            foreach (float value in values)
            {
                RequireFinite(value, label);
            }
        }

        private static void RequireFinite(float value, string label)
        {
            Require(!float.IsNaN(value) && !float.IsInfinity(value), $"{label} contains a non-finite value.");
        }

        private static void Require(bool condition, string message)
        {
            if (!condition)
            {
                throw new TraceValidationException(message);
            }
        }
    }
}
