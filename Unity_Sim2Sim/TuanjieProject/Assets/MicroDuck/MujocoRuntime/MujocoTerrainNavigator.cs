using System;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Mujoco
{
    [DefaultExecutionOrder(2500)]
    public sealed class MujocoTerrainNavigator : MonoBehaviour
    {
        [SerializeField] private MujocoDemoController controller;
        [SerializeField] private int activeIndex;

        public TerrainModuleDefinition ActiveModule =>
            MujocoTerrainCatalog.Modules[NormalizeIndex(activeIndex)];
        public int ActiveIndex => NormalizeIndex(activeIndex);

        public void Configure(MujocoDemoController value)
        {
            controller = value ?? throw new ArgumentNullException(nameof(value));
            activeIndex = IndexOf(MujocoTerrainCatalog.DefaultModuleId);
        }

        public bool Select(string moduleId)
        {
            int index = IndexOf(moduleId);
            if (index < 0 || controller == null)
            {
                return false;
            }

            activeIndex = index;
            TerrainModuleDefinition module = ActiveModule;
            return controller.ResetActiveRobotAt(
                module.SpawnPosition,
                module.SpawnYawDegrees,
                module.InitialVelocityMetersPerSecond);
        }

        public bool Next()
        {
            int next = NormalizeIndex(activeIndex + 1);
            return Select(MujocoTerrainCatalog.Modules[next].Id);
        }

        public bool Previous()
        {
            int previous = NormalizeIndex(activeIndex - 1);
            return Select(MujocoTerrainCatalog.Modules[previous].Id);
        }

        private void Start()
        {
            if (controller != null && controller.IsHealthy)
            {
                Select(ActiveModule.Id);
            }
        }

        private static int IndexOf(string moduleId)
        {
            for (int index = 0; index < MujocoTerrainCatalog.Modules.Count; index++)
            {
                if (MujocoTerrainCatalog.Modules[index].Id == moduleId)
                {
                    return index;
                }
            }
            return -1;
        }

        private static int NormalizeIndex(int index)
        {
            int count = MujocoTerrainCatalog.Modules.Count;
            return ((index % count) + count) % count;
        }
    }
}
