using System;
using System.Collections.Generic;
using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    public sealed class MicroDuckRig : MonoBehaviour
    {
        private static readonly string[] RollerPassiveWheelBodyNames =
        {
            "tire",
            "tire_2",
            "tire_3",
            "tire_4",
        };

        [SerializeField] private RobotVariant variant;
        [SerializeField] private ArticulationBody rootBody;
        [SerializeField] private ArticulationBody[] servoBodies = Array.Empty<ArticulationBody>();

        private Vector3 initialRootPosition;
        private Quaternion initialRootRotation;
        private bool initialPoseCaptured;

        public RobotVariant Variant => variant;
        public ArticulationBody RootBody => rootBody;
        public int ServoCount => servoBodies == null ? 0 : servoBodies.Length;
        public string[] ServoNames => PolicyContract.ServoNames;
        public string[] PassiveWheelNames => variant == RobotVariant.Roller
            ? PolicyContract.RollerPassiveWheelNames
            : Array.Empty<string>();

        public void Configure(
            RobotVariant robotVariant,
            ArticulationBody root,
            ArticulationBody[] orderedServoBodies)
        {
            if (root == null)
            {
                throw new ArgumentNullException(nameof(root));
            }

            if (orderedServoBodies == null || orderedServoBodies.Length != PolicyContract.ActionCount)
            {
                throw new ArgumentException(
                    $"Rig must contain exactly {PolicyContract.ActionCount} ordered servo bodies.",
                    nameof(orderedServoBodies));
            }

            for (int index = 0; index < orderedServoBodies.Length; index++)
            {
                if (orderedServoBodies[index] == null)
                {
                    throw new ArgumentException($"Servo body at index {index} is null.", nameof(orderedServoBodies));
                }
            }

            variant = robotVariant;
            rootBody = root;
            servoBodies = (ArticulationBody[])orderedServoBodies.Clone();
            CaptureInitialPose();
        }

        public void ReadPolicyState(
            float[] jointPositionRad,
            float[] jointVelocityRadPerSecond,
            out Vector3 localAngularVelocity,
            out Vector3 localProjectedGravity)
        {
            ValidateConfigured();
            PolicyContract.ValidateAction(jointPositionRad);
            PolicyContract.ValidateAction(jointVelocityRadPerSecond);
            localAngularVelocity = rootBody.transform.InverseTransformDirection(rootBody.angularVelocity);
            Vector3 gravity = Physics.gravity.sqrMagnitude > 0f ? Physics.gravity.normalized : Vector3.down;
            localProjectedGravity = rootBody.transform.InverseTransformDirection(gravity);

            for (int index = 0; index < servoBodies.Length; index++)
            {
                ArticulationReducedSpace position = servoBodies[index].jointPosition;
                ArticulationReducedSpace velocity = servoBodies[index].jointVelocity;
                if (position.dofCount != 1 || velocity.dofCount != 1)
                {
                    throw new InvalidOperationException(
                        $"Servo body '{servoBodies[index].name}' must have exactly one degree of freedom.");
                }

                jointPositionRad[index] = position[0];
                jointVelocityRadPerSecond[index] = velocity[0];
            }
        }

        public void ApplyTargets(float[] targetPositionRad)
        {
            ValidateConfigured();
            PolicyContract.ValidateAction(targetPositionRad);
            for (int index = 0; index < servoBodies.Length; index++)
            {
                ArticulationDrive drive = servoBodies[index].xDrive;
                drive.target = targetPositionRad[index] * Mathf.Rad2Deg;
                servoBodies[index].xDrive = drive;
            }
        }

        public float[] ReadPassiveWheelVelocityRadPerSecond()
        {
            ValidateConfigured();
            if (variant != RobotVariant.Roller)
            {
                return Array.Empty<float>();
            }

            ArticulationBody[] bodies = GetComponentsInChildren<ArticulationBody>(includeInactive: true);
            var velocities = new float[RollerPassiveWheelBodyNames.Length];
            for (int index = 0; index < RollerPassiveWheelBodyNames.Length; index++)
            {
                ArticulationBody wheel = null;
                foreach (ArticulationBody body in bodies)
                {
                    if (body.name == RollerPassiveWheelBodyNames[index])
                    {
                        wheel = body;
                        break;
                    }
                }

                if (wheel == null)
                {
                    throw new InvalidOperationException(
                        $"Roller passive wheel body '{RollerPassiveWheelBodyNames[index]}' is missing.");
                }

                ArticulationReducedSpace velocity = wheel.jointVelocity;
                if (velocity.dofCount != 1)
                {
                    throw new InvalidOperationException(
                        $"Roller passive wheel body '{wheel.name}' must have exactly one degree of freedom.");
                }

                velocities[index] = velocity[0];
            }

            return velocities;
        }

        public void ResetPose(float[] homePositionRad)
        {
            ValidateConfigured();
            PolicyContract.ValidateAction(homePositionRad);
            if (!initialPoseCaptured)
            {
                CaptureInitialPose();
            }

            rootBody.TeleportRoot(initialRootPosition, initialRootRotation);
            rootBody.velocity = Vector3.zero;
            rootBody.angularVelocity = Vector3.zero;

            var jointPositions = new List<float>();
            var jointVelocities = new List<float>();
            var jointForces = new List<float>();
            var dofStartIndices = new List<int>();
            rootBody.GetJointPositions(jointPositions);
            rootBody.GetJointVelocities(jointVelocities);
            rootBody.GetJointForces(jointForces);
            rootBody.GetDofStartIndices(dofStartIndices);
            for (int dof = 0; dof < jointPositions.Count; dof++)
            {
                jointPositions[dof] = 0f;
                jointVelocities[dof] = 0f;
                jointForces[dof] = 0f;
            }

            for (int index = 0; index < servoBodies.Length; index++)
            {
                int articulationIndex = servoBodies[index].index;
                if (articulationIndex < 0 || articulationIndex >= dofStartIndices.Count)
                {
                    throw new InvalidOperationException(
                        $"Servo body '{servoBodies[index].name}' has invalid articulation index {articulationIndex}.");
                }

                int dofStart = dofStartIndices[articulationIndex];
                if (dofStart < 0 || dofStart >= jointPositions.Count)
                {
                    throw new InvalidOperationException(
                        $"Servo body '{servoBodies[index].name}' has invalid degree-of-freedom index {dofStart}.");
                }

                jointPositions[dofStart] = homePositionRad[index];
            }

            rootBody.SetJointPositions(jointPositions);
            rootBody.SetJointVelocities(jointVelocities);
            rootBody.SetJointForces(jointForces);

            ApplyTargets(homePositionRad);
        }

        private void Awake()
        {
            if (rootBody != null)
            {
                CaptureInitialPose();
            }
        }

        private void CaptureInitialPose()
        {
            initialRootPosition = rootBody.transform.position;
            initialRootRotation = rootBody.transform.rotation;
            initialPoseCaptured = true;
        }

        private void ValidateConfigured()
        {
            if (rootBody == null || servoBodies == null || servoBodies.Length != PolicyContract.ActionCount)
            {
                throw new InvalidOperationException("MicroDuck rig is not configured with a root and 14 servos.");
            }
        }
    }
}
