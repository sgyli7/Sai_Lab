using System;

namespace AgenticRobot.MicroDuck
{
    public static class PolicyContract
    {
        public const int ObservationCount = 61;
        public const int ActionCount = 14;

        private static readonly string[] OrderedServoNames =
        {
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
            "left_ankle",
            "neck_pitch",
            "head_pitch",
            "head_yaw",
            "head_roll",
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee",
            "right_ankle",
        };

        private static readonly string[] OrderedRollerPassiveWheelNames =
        {
            "passive_LF_wheel",
            "passive_LR_wheel",
            "passive_RF_wheel",
            "passive_RR_wheel",
        };

        public static string[] ServoNames => (string[])OrderedServoNames.Clone();
        public static string[] RollerPassiveWheelNames =>
            (string[])OrderedRollerPassiveWheelNames.Clone();

        public static void ValidateObservation(float[] observation)
        {
            ValidateWidth(observation, ObservationCount, nameof(observation));
        }

        public static void ValidateAction(float[] action)
        {
            ValidateWidth(action, ActionCount, nameof(action));
        }

        private static void ValidateWidth(float[] values, int expected, string parameterName)
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
