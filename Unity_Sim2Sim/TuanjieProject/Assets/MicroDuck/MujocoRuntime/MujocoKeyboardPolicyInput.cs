using UnityEngine;

namespace AgenticRobot.MicroDuck.Mujoco
{
    // Cross-variant switches enable an entire imported MJCF hierarchy. Running
    // after the default-order MjScene LateUpdate ensures every newly enabled
    // MjComponent reaches the following Update together and requests one rebuild.
    [DefaultExecutionOrder(3000)]
    public sealed class MujocoKeyboardPolicyInput : MonoBehaviour
    {
        [SerializeField] private MujocoDemoController controller;
        [SerializeField] private MujocoTerrainNavigator terrainNavigator;
        [SerializeField] private MujocoCameraRig cameraRig;
        [SerializeField] private float forwardSpeed = 0.3f;
        [SerializeField] private float lateralSpeed = 0.2f;
        [SerializeField] private float yawRate = 0.8f;
        [SerializeField] private float headRange = 0.5f;
        [SerializeField] private float bodyHeightRange = 0.03f;
        [SerializeField] private float bodyAngleRange = 0.2f;

        public void Configure(MujocoDemoController value)
        {
            controller = value;
        }

        public void Configure(
            MujocoDemoController value,
            MujocoTerrainNavigator navigator,
            MujocoCameraRig camera)
        {
            controller = value;
            terrainNavigator = navigator;
            cameraRig = camera;
        }

        private void LateUpdate()
        {
            if (controller == null)
            {
                return;
            }

            for (int slot = 1; slot <= 9; slot++)
            {
                if (Input.GetKeyDown(KeyCode.Alpha0 + slot))
                {
                    controller.SwitchPolicy(slot);
                }
            }

            bool cameraOwnsNavigation = cameraRig != null && cameraRig.OwnsNavigationInput;
            controller.SetTwist(
                cameraOwnsNavigation ? 0f : Axis(KeyCode.S, KeyCode.W) * forwardSpeed,
                cameraOwnsNavigation ? 0f : Axis(KeyCode.D, KeyCode.A) * lateralSpeed,
                cameraOwnsNavigation ? 0f : Axis(KeyCode.E, KeyCode.Q) * yawRate);
            controller.SetHead(
                Axis(KeyCode.K, KeyCode.I) * headRange,
                Axis(KeyCode.DownArrow, KeyCode.UpArrow) * headRange,
                Axis(KeyCode.RightArrow, KeyCode.LeftArrow) * headRange,
                Axis(KeyCode.L, KeyCode.J) * headRange);
            controller.SetBody(
                Axis(KeyCode.PageDown, KeyCode.PageUp) * bodyHeightRange,
                Axis(KeyCode.X, KeyCode.Z) * bodyAngleRange,
                Axis(KeyCode.Comma, KeyCode.Period) * bodyAngleRange);

            if (Input.GetKeyDown(KeyCode.Space))
            {
                controller.TriggerSkill();
            }

            if (Input.GetKeyDown(KeyCode.R))
            {
                controller.ResetActiveRobot();
            }

            if (terrainNavigator != null && Input.GetKeyDown(KeyCode.T))
            {
                if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
                {
                    terrainNavigator.Previous();
                }
                else
                {
                    terrainNavigator.Next();
                }
            }
        }

        private static float Axis(KeyCode negative, KeyCode positive)
        {
            return (Input.GetKey(positive) ? 1f : 0f)
                - (Input.GetKey(negative) ? 1f : 0f);
        }
    }
}
