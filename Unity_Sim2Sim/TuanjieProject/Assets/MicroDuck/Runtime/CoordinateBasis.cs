using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    /// <summary>
    /// Converts MuJoCo's right-handed +Z-up basis to Tuanjie's left-handed +Y-up basis.
    /// </summary>
    public static class CoordinateBasis
    {
        public static Vector3 MuJoCoToTuanjie(Vector3 value)
        {
            return new Vector3(-value.y, value.z, value.x);
        }

        public static Vector3 MuJoCoToTuanjieAxial(Vector3 value)
        {
            // Angular quantities are pseudovectors. The extra determinant sign is
            // required because this basis conversion changes handedness.
            return new Vector3(value.y, -value.z, -value.x);
        }

        public static Vector3 TuanjieToMuJoCo(Vector3 value)
        {
            return new Vector3(value.z, -value.x, value.y);
        }

        public static Vector3 TuanjieToMuJoCoAxial(Vector3 value)
        {
            return new Vector3(-value.z, value.x, -value.y);
        }

        public static Quaternion MuJoCoToTuanjie(Quaternion value)
        {
            // For the improper basis transform B, the quaternion vector part maps as
            // det(B) * B * (x, y, z), while the scalar part remains unchanged.
            Quaternion converted = new Quaternion(value.y, -value.z, -value.x, value.w);
            return Quaternion.Normalize(converted);
        }

        public static Quaternion TuanjieToMuJoCo(Quaternion value)
        {
            Quaternion converted = new Quaternion(-value.z, value.x, -value.y, value.w);
            return Quaternion.Normalize(converted);
        }
    }
}
