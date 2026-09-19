using System.Linq;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Mujoco
{
    public sealed class MujocoPolicyStatusOverlay : MonoBehaviour
    {
        [SerializeField] private MujocoDemoController controller;
        [SerializeField] private MujocoTerrainNavigator navigator;
        [SerializeField] private MujocoCameraRig cameraRig;

        private GUIStyle headerStyle;
        private GUIStyle bodyStyle;

        public MujocoDemoController Controller => controller;
        public MujocoTerrainNavigator Navigator => navigator;
        public MujocoCameraRig CameraRig => cameraRig;

        public void Configure(MujocoDemoController value)
        {
            controller = value;
        }

        public void Configure(
            MujocoDemoController value,
            MujocoTerrainNavigator terrainNavigator,
            MujocoCameraRig camera)
        {
            controller = value;
            navigator = terrainNavigator;
            cameraRig = camera;
        }

        public string BuildTerrainText()
        {
            if (navigator == null)
            {
                return "Terrain | not configured";
            }

            TerrainModuleDefinition module = navigator.ActiveModule;
            string compatibility;
            if (!module.IsTrainingMatched)
            {
                compatibility = "训练扩展";
            }
            else if (controller == null
                || controller.ActivePolicySlot <= 0
                || module.RecommendedPolicySlots.Contains(controller.ActivePolicySlot))
            {
                compatibility = "训练匹配";
            }
            else
            {
                compatibility = "当前策略未匹配";
            }

            return $"Terrain {navigator.ActiveIndex + 1}/{MujocoTerrainCatalog.Modules.Count} "
                + $"| {module.DisplayName} | {module.Difficulty} | {compatibility}";
        }

        public string BuildControlsText()
        {
            string cameraControls = cameraRig != null
                && cameraRig.Mode == MujocoCameraMode.FreeFly
                ? "FREE FLY: RMB look · WASD/QE move · wheel speed"
                : "FOLLOW: LMB orbit · wheel zoom";
            return "1-9 policy | T / Shift+T terrain | Tab camera | C preset | F focus "
                + "| Space skill | R reset | "
                + cameraControls;
        }

        private void OnGUI()
        {
            if (controller == null)
            {
                return;
            }

            string health = controller.IsHealthy ? "healthy" : controller.Fault;
            EnsureStyles();
            float width = Mathf.Min(1120f, Screen.width - 16f);
            GUI.color = new Color(0.07f, 0.13f, 0.16f, 0.88f);
            GUI.Box(new Rect(8f, 8f, width, 126f), GUIContent.none);
            GUI.color = Color.white;
            GUI.Label(
                new Rect(18f, 14f, width - 20f, 27f),
                $"MicroDuck | [{controller.ActivePolicySlot}] {controller.ActivePolicyName} "
                + $"| {controller.BackendName} | ticks {controller.PolicyTicks} | {health}",
                headerStyle);
            GUI.Label(
                new Rect(18f, 42f, width - 20f, 25f),
                BuildTerrainText(),
                bodyStyle);
            GUI.Label(
                new Rect(18f, 67f, width - 20f, 25f),
                BuildControlsText(),
                bodyStyle);
            Vector3 position = controller.RootPositionMeters;
            float[] command = controller.LastCommand;
            float commandedForward = command.Length > 0 ? command[0] : 0f;
            GUI.Label(
                new Rect(18f, 94f, width - 20f, 25f),
                $"position x={position.x:F3} m y={position.y:F3} m z={position.z:F3} m "
                + $"| upright={controller.TrunkUpright:F3} | forward command={commandedForward:F2}",
                bodyStyle);
        }

        private void EnsureStyles()
        {
            if (headerStyle != null)
            {
                return;
            }

            headerStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 15,
                fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.92f, 0.98f, 1f) },
            };
            bodyStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 13,
                normal = { textColor = new Color(0.87f, 0.93f, 0.88f) },
            };
        }
    }
}
