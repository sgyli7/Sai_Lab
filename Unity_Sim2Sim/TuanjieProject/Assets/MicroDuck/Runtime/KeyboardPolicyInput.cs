using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    public sealed class KeyboardPolicyInput : MonoBehaviour
    {
        [SerializeField] private MicroDuckDemoController controller;
        [SerializeField] private float forwardSpeed = 0.3f;
        [SerializeField] private float lateralSpeed = 0.2f;
        [SerializeField] private float yawRate = 0.8f;
        [SerializeField] private float headRange = 0.5f;
        [SerializeField] private float bodyHeightRange = 0.03f;
        [SerializeField] private float bodyAngleRange = 0.2f;

        public void Configure(MicroDuckDemoController value)
        {
            controller = value;
        }

        private void Update()
        {
            if (controller == null)
            {
                return;
            }

            for (int slot = 1; slot <= 9; slot++)
            {
                if (Input.GetKeyDown(KeyCode.Alpha0 + slot))
                {
                    controller.SelectPolicy(slot);
                }
            }

            float forward = Axis(KeyCode.S, KeyCode.W) * forwardSpeed;
            float left = Axis(KeyCode.D, KeyCode.A) * lateralSpeed;
            float yaw = Axis(KeyCode.E, KeyCode.Q) * yawRate;
            controller.SetTwist(forward, left, yaw);

            float neckPitch = Axis(KeyCode.K, KeyCode.I) * headRange;
            float headPitch = Axis(KeyCode.DownArrow, KeyCode.UpArrow) * headRange;
            float headYaw = Axis(KeyCode.RightArrow, KeyCode.LeftArrow) * headRange;
            float headRoll = Axis(KeyCode.L, KeyCode.J) * headRange;
            controller.SetHead(neckPitch, headPitch, headYaw, headRoll);

            float height = Axis(KeyCode.PageDown, KeyCode.PageUp) * bodyHeightRange;
            float roll = Axis(KeyCode.X, KeyCode.Z) * bodyAngleRange;
            float pitch = Axis(KeyCode.V, KeyCode.C) * bodyAngleRange;
            controller.SetBody(height, roll, pitch);

            if (Input.GetKeyDown(KeyCode.Space))
            {
                controller.TriggerSkill();
            }

            if (Input.GetKeyDown(KeyCode.R))
            {
                controller.ResetActiveRig();
            }
        }

        private static float Axis(KeyCode negative, KeyCode positive)
        {
            return (Input.GetKey(positive) ? 1f : 0f) - (Input.GetKey(negative) ? 1f : 0f);
        }
    }
}
