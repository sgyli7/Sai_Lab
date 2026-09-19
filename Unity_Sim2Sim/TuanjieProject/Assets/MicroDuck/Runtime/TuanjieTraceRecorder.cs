using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    [Serializable]
    internal sealed class RolloutTraceResultData
    {
        public string recordType = "result";
        public bool passed;
    }

    /// <summary>
    /// Writes the shared, line-delimited rollout contract without depending on editor APIs.
    /// </summary>
    public sealed class TuanjieTraceRecorder
    {
        private readonly List<string> records = new List<string>();
        private bool completed;
        private int frameCount;

        public TuanjieTraceRecorder(RolloutTraceHeaderData header)
        {
            if (header == null)
            {
                throw new ArgumentNullException(nameof(header));
            }

            if (header.producer != "tuanjie")
            {
                throw new TraceValidationException("Tuanjie trace header producer must be 'tuanjie'.");
            }

            records.Add(JsonUtility.ToJson(header));
        }

        public static TuanjieTraceRecorder ForController(
            MicroDuckDemoController controller,
            string policySha256,
            string scene,
            string sceneSha256,
            float physicsTimestepSeconds,
            int policyDecimation)
        {
            if (controller == null)
            {
                throw new ArgumentNullException(nameof(controller));
            }

            if (!controller.IsHealthy || controller.ActiveRig == null)
            {
                throw new InvalidOperationException("Controller must have a healthy active policy before recording.");
            }

            PolicyEntry entry = PolicyCatalog.GetBySlot(controller.ActivePolicySlot);
            var header = new RolloutTraceHeaderData
            {
                recordType = "header",
                schemaVersion = 1,
                producer = "tuanjie",
                policyName = entry.FileName,
                policySha256 = policySha256 ?? string.Empty,
                scene = scene ?? string.Empty,
                sceneSha256 = sceneSha256 ?? string.Empty,
                role = PolicyCatalog.ToTraceRoleName(entry.Role),
                robotVariant = entry.RobotVariant == RobotVariant.Roller ? "roller" : "legged",
                physicsTimestepSeconds = physicsTimestepSeconds,
                policyDecimation = policyDecimation,
                observationSize = PolicyContract.ObservationCount,
                actionSize = PolicyContract.ActionCount,
                servoNames = controller.ActiveRig.ServoNames,
                passiveWheelNames = controller.ActiveRig.PassiveWheelNames,
            };
            return new TuanjieTraceRecorder(header);
        }

        public void AppendControllerFrame(
            MicroDuckDemoController controller,
            int physicsStep,
            int policyStep,
            float timeSeconds)
        {
            if (controller == null)
            {
                throw new ArgumentNullException(nameof(controller));
            }

            MicroDuckRig rig = controller.ActiveRig;
            if (!controller.IsHealthy || rig == null)
            {
                throw new InvalidOperationException("Controller must have a healthy active policy before recording a frame.");
            }

            ArticulationBody root = rig.RootBody;
            Vector3 rootPosition = CoordinateBasis.TuanjieToMuJoCo(root.transform.position);
            Quaternion rootRotation = CoordinateBasis.TuanjieToMuJoCo(root.transform.rotation);
            Vector3 rootVelocity = CoordinateBasis.TuanjieToMuJoCo(root.velocity);
            Vector3 rootAngularVelocity = CoordinateBasis.TuanjieToMuJoCoAxial(root.angularVelocity);
            AppendFrame(new RolloutTraceFrameData
            {
                recordType = "frame",
                physicsStep = physicsStep,
                policyStep = policyStep,
                timeSeconds = timeSeconds,
                rootPosition = Vector(rootPosition),
                rootQuaternionWxyz = new[] { rootRotation.w, rootRotation.x, rootRotation.y, rootRotation.z },
                rootLinearVelocity = Vector(rootVelocity),
                rootAngularVelocity = Vector(rootAngularVelocity),
                jointPositionRad = controller.LastJointPositionRad,
                jointVelocityRadPerSecond = controller.LastJointVelocityRadPerSecond,
                passiveWheelVelocityRadPerSecond = rig.ReadPassiveWheelVelocityRadPerSecond(),
                observation = controller.LastObservation,
                rawAction = controller.LastRawAction,
                targetPositionRad = controller.LastTargets,
                command = controller.LastCommand,
                contacts = Array.Empty<TraceContactData>(),
            });
        }

        public void AppendFrame(RolloutTraceFrameData frame)
        {
            if (completed)
            {
                throw new InvalidOperationException("Cannot append a frame after the trace is complete.");
            }

            if (frame == null)
            {
                throw new ArgumentNullException(nameof(frame));
            }

            records.Add(JsonUtility.ToJson(frame));
            frameCount++;
        }

        public string Complete(bool passed)
        {
            if (completed)
            {
                throw new InvalidOperationException("Trace is already complete.");
            }

            if (frameCount == 0)
            {
                throw new TraceValidationException("Trace must contain at least one frame.");
            }

            records.Add(JsonUtility.ToJson(new RolloutTraceResultData { passed = passed }));
            completed = true;
            var output = new StringBuilder();
            foreach (string record in records)
            {
                output.Append(record).Append('\n');
            }

            return output.ToString();
        }

        private static float[] Vector(Vector3 value)
        {
            return new[] { value.x, value.y, value.z };
        }

    }
}
