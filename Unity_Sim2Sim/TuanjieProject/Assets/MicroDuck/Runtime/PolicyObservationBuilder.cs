using System;
using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    public static class PolicyObservationBuilder
    {
        public const int CommandCount = 13;

        public static void Build(
            Vector3 tuanjieLocalAngularVelocity,
            Vector3 tuanjieLocalProjectedGravity,
            float[] jointPositionRad,
            float[] jointVelocityRadPerSecond,
            float[] homePositionRad,
            float[] previousAction,
            float[] command,
            float[] destination)
        {
            BuildMuJoCo(
                CoordinateBasis.TuanjieToMuJoCoAxial(tuanjieLocalAngularVelocity),
                CoordinateBasis.TuanjieToMuJoCo(tuanjieLocalProjectedGravity),
                jointPositionRad,
                jointVelocityRadPerSecond,
                homePositionRad,
                previousAction,
                command,
                destination);
        }

        /// <summary>
        /// Builds an observation from vectors that are already expressed in MuJoCo's
        /// right-handed +Z-up policy basis. This is the entry point used when the
        /// official MuJoCo runtime owns the simulation inside Tuanjie.
        /// </summary>
        public static void BuildMuJoCo(
            Vector3 mujocoLocalAngularVelocity,
            Vector3 mujocoLocalProjectedGravity,
            float[] jointPositionRad,
            float[] jointVelocityRadPerSecond,
            float[] homePositionRad,
            float[] previousAction,
            float[] command,
            float[] destination)
        {
            Validate(jointPositionRad, PolicyContract.ActionCount, nameof(jointPositionRad));
            Validate(jointVelocityRadPerSecond, PolicyContract.ActionCount, nameof(jointVelocityRadPerSecond));
            Validate(homePositionRad, PolicyContract.ActionCount, nameof(homePositionRad));
            Validate(previousAction, PolicyContract.ActionCount, nameof(previousAction));
            Validate(command, CommandCount, nameof(command));
            Validate(destination, PolicyContract.ObservationCount, nameof(destination));

            destination[0] = mujocoLocalAngularVelocity.x;
            destination[1] = mujocoLocalAngularVelocity.y;
            destination[2] = mujocoLocalAngularVelocity.z;
            destination[3] = mujocoLocalProjectedGravity.x;
            destination[4] = mujocoLocalProjectedGravity.y;
            destination[5] = mujocoLocalProjectedGravity.z;

            for (int index = 0; index < PolicyContract.ActionCount; index++)
            {
                destination[6 + index] = jointPositionRad[index] - homePositionRad[index];
                destination[20 + index] = jointVelocityRadPerSecond[index];
                destination[34 + index] = previousAction[index];
            }

            Array.Copy(command, 0, destination, 48, CommandCount);
        }

        private static void Validate(float[] values, int expected, string parameterName)
        {
            if (values == null)
            {
                throw new ArgumentNullException(parameterName);
            }

            if (values.Length != expected)
            {
                throw new ArgumentException(
                    $"Expected {expected} values, but received {values.Length}.",
                    parameterName);
            }
        }
    }
}
