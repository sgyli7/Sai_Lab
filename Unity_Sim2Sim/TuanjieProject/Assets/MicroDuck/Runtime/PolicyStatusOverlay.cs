using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    [DisallowMultipleComponent]
    public sealed class PolicyStatusOverlay : MonoBehaviour
    {
        [SerializeField] private MicroDuckDemoController controller;

        public MicroDuckDemoController Controller => controller;

        public string StatusText
        {
            get
            {
                if (controller == null)
                {
                    return "MicroDuck MVP | Controller not configured";
                }

                string health = controller.IsHealthy ? "Ready" : controller.Fault;
                return $"MicroDuck MVP | [{controller.ActivePolicySlot}] {controller.ActivePolicyName}"
                    + $" | {controller.BackendName} | ticks {controller.PolicyTicks} | {health}";
            }
        }

        public void Configure(MicroDuckDemoController value)
        {
            controller = value;
        }

        private void OnGUI()
        {
            const string controls = "1-9 policy | WASD move | QE yaw | Space skill | R reset";
            GUI.Box(new Rect(12f, 12f, Mathf.Max(500f, Screen.width - 24f), 58f), string.Empty);
            GUI.Label(new Rect(24f, 20f, Screen.width - 48f, 20f), StatusText);
            GUI.Label(new Rect(24f, 42f, Screen.width - 48f, 20f), controls);
        }
    }
}
