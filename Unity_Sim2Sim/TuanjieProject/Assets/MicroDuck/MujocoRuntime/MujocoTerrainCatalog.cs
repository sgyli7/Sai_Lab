using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Mujoco
{
    public enum TerrainPrimitiveKind
    {
        Box,
        Ellipsoid,
        Cylinder,
    }

    public readonly struct TerrainPrimitiveDefinition : IEquatable<TerrainPrimitiveDefinition>
    {
        public TerrainPrimitiveDefinition(
            string id,
            TerrainPrimitiveKind kind,
            Vector3 center,
            Vector3 size,
            Vector3 eulerAngles,
            string materialKey,
            bool collisionEnabled = true)
        {
            Id = id ?? throw new ArgumentNullException(nameof(id));
            Kind = kind;
            Center = center;
            Size = size;
            EulerAngles = eulerAngles;
            MaterialKey = materialKey ?? throw new ArgumentNullException(nameof(materialKey));
            CollisionEnabled = collisionEnabled;
        }

        public string Id { get; }
        public TerrainPrimitiveKind Kind { get; }
        public Vector3 Center { get; }
        public Vector3 Size { get; }
        public Vector3 EulerAngles { get; }
        public string MaterialKey { get; }
        public bool CollisionEnabled { get; }

        public bool Equals(TerrainPrimitiveDefinition other)
        {
            return Id == other.Id
                && Kind == other.Kind
                && Center == other.Center
                && Size == other.Size
                && EulerAngles == other.EulerAngles
                && MaterialKey == other.MaterialKey
                && CollisionEnabled == other.CollisionEnabled;
        }

        public override bool Equals(object obj)
        {
            return obj is TerrainPrimitiveDefinition other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = Id.GetHashCode();
                hash = (hash * 397) ^ (int)Kind;
                hash = (hash * 397) ^ Center.GetHashCode();
                hash = (hash * 397) ^ Size.GetHashCode();
                hash = (hash * 397) ^ EulerAngles.GetHashCode();
                hash = (hash * 397) ^ MaterialKey.GetHashCode();
                hash = (hash * 397) ^ CollisionEnabled.GetHashCode();
                return hash;
            }
        }
    }

    public sealed class TerrainModuleDefinition
    {
        public TerrainModuleDefinition(
            string id,
            string displayName,
            Vector3 spawnPosition,
            float spawnYawDegrees,
            bool isTrainingMatched,
            string difficulty,
            float maximumSurfaceVariationMeters,
            float minimumSlopeDegrees,
            float maximumSlopeDegrees,
            IEnumerable<int> recommendedPolicySlots,
            IEnumerable<TerrainPrimitiveDefinition> primitives)
        {
            Id = id ?? throw new ArgumentNullException(nameof(id));
            DisplayName = displayName ?? throw new ArgumentNullException(nameof(displayName));
            SpawnPosition = spawnPosition;
            SpawnYawDegrees = spawnYawDegrees;
            IsTrainingMatched = isTrainingMatched;
            Difficulty = difficulty ?? throw new ArgumentNullException(nameof(difficulty));
            MaximumSurfaceVariationMeters = maximumSurfaceVariationMeters;
            MinimumSlopeDegrees = minimumSlopeDegrees;
            MaximumSlopeDegrees = maximumSlopeDegrees;
            RecommendedPolicySlots = Array.AsReadOnly(recommendedPolicySlots.ToArray());
            Primitives = Array.AsReadOnly(primitives.ToArray());
        }

        public string Id { get; }
        public string DisplayName { get; }
        public Vector3 SpawnPosition { get; }
        public float SpawnYawDegrees { get; }
        public bool IsTrainingMatched { get; }
        public string Difficulty { get; }
        public float MaximumSurfaceVariationMeters { get; }
        public float MinimumSlopeDegrees { get; }
        public float MaximumSlopeDegrees { get; }
        public Vector3 InitialVelocityMetersPerSecond => Id == "upstream_roller_slope"
            ? new Vector3(0.35f, 0f, 0f)
            : Vector3.zero;
        public ReadOnlyCollection<int> RecommendedPolicySlots { get; }
        public ReadOnlyCollection<TerrainPrimitiveDefinition> Primitives { get; }
    }

    /// <summary>
    /// Deterministic, engine-independent description of every physical showcase surface.
    /// Unity renderers and native MuJoCo geoms are both created from these definitions.
    /// </summary>
    public static class MujocoTerrainCatalog
    {
        public const string DefaultModuleId = "flat_plaza";
        public const int RandomGridSeed = 20260905;

        private static readonly ReadOnlyCollection<TerrainModuleDefinition> modules =
            Array.AsReadOnly(new[]
            {
                CreateModule("flat_plaza"),
                CreateModule("upstream_pyramid_stairs"),
                CreateModule("upstream_random_grid"),
                CreateModule("upstream_pyramid_slope"),
                CreateModule("upstream_roller_slope"),
                CreateModule("rock_steps"),
                CreateModule("stairs_bridge"),
            });

        public static ReadOnlyCollection<TerrainModuleDefinition> Modules => modules;

        public static TerrainModuleDefinition Find(string id)
        {
            TerrainModuleDefinition module = modules.FirstOrDefault(candidate => candidate.Id == id);
            if (module == null)
            {
                throw new ArgumentOutOfRangeException(nameof(id), id, "Unknown MicroDuck terrain module.");
            }

            return module;
        }

        public static TerrainModuleDefinition CreateModule(string id)
        {
            switch (id)
            {
                case "flat_plaza":
                    return CreateFlatPlaza();
                case "upstream_pyramid_stairs":
                    return CreatePyramidStairs();
                case "upstream_random_grid":
                    return CreateRandomGrid();
                case "upstream_pyramid_slope":
                    return CreatePyramidSlope();
                case "upstream_roller_slope":
                    return CreateRollerSlope();
                case "rock_steps":
                    return CreateRockSteps();
                case "stairs_bridge":
                    return CreateStairsBridge();
                default:
                    throw new ArgumentOutOfRangeException(nameof(id), id, "Unknown MicroDuck terrain module.");
            }
        }

        private static TerrainModuleDefinition CreateFlatPlaza()
        {
            return Module(
                "flat_plaza",
                "中央安全广场",
                new Vector3(0f, 0.125f, 0f),
                true,
                "安全",
                0f,
                0f,
                0f,
                new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 },
                Box("plaza", new Vector3(0f, -0.05f, 0f), new Vector3(6f, 0.1f, 6f), "plaza"));
        }

        private static TerrainModuleDefinition CreatePyramidStairs()
        {
            const float stepWidth = 0.15f;
            const float stepHeight = 0.015f;
            const float platformWidth = 2f;
            var primitives = new List<TerrainPrimitiveDefinition>
            {
                Box("stairs_base", new Vector3(-8f, -0.05f, 0f), new Vector3(8f, 0.1f, 8f), "rough_base"),
            };

            int levelCount = Mathf.FloorToInt((8f - platformWidth) / (stepWidth * 2f));
            for (int level = 1; level <= levelCount; level++)
            {
                float width = 8f - (2f * level * stepWidth);
                float top = level * stepHeight;
                primitives.Add(Box(
                    $"stairs_level_{level:00}",
                    new Vector3(-8f, (-0.1f + top) * 0.5f, 0f),
                    new Vector3(width, 0.1f + top, width),
                    level % 2 == 0 ? "stairs_light" : "stairs_dark"));
            }

            float plateauTop = levelCount * stepHeight;
            return Module(
                "upstream_pyramid_stairs",
                "官方微型金字塔台阶",
                new Vector3(-8f, plateauTop + 0.125f, 0f),
                true,
                "训练基准 · 1.5 cm/级",
                stepHeight,
                0f,
                0f,
                new[] { 1, 2 },
                primitives.ToArray());
        }

        private static TerrainModuleDefinition CreateRandomGrid()
        {
            const int cellCount = 17;
            const float cellWidth = 0.45f;
            const float maximumHeight = 0.01f;
            var random = new System.Random(RandomGridSeed);
            var primitives = new List<TerrainPrimitiveDefinition>(1 + (cellCount * cellCount))
            {
                Box("grid_base", new Vector3(8f, -0.055f, 0f), new Vector3(8f, 0.11f, 8f), "rough_base"),
            };

            float span = cellCount * cellWidth;
            float first = -0.5f * (span - cellWidth);
            for (int row = 0; row < cellCount; row++)
            {
                for (int column = 0; column < cellCount; column++)
                {
                    float height = (float)random.NextDouble() * maximumHeight;
                    float thickness = 0.08f + height;
                    primitives.Add(Box(
                        $"grid_{row:00}_{column:00}",
                        new Vector3(
                            8f + first + (column * cellWidth),
                            (-0.08f + height) * 0.5f,
                            first + (row * cellWidth)),
                        new Vector3(cellWidth * 0.985f, thickness, cellWidth * 0.985f),
                        (row + column) % 2 == 0 ? "cobble_light" : "cobble_dark"));
                }
            }

            return Module(
                "upstream_random_grid",
                "官方随机格石路",
                new Vector3(8f, 0.135f, 0f),
                true,
                "训练基准 · 1 cm",
                maximumHeight,
                0f,
                0f,
                new[] { 1, 2 },
                primitives.ToArray());
        }

        private static TerrainModuleDefinition CreatePyramidSlope()
        {
            const float angle = 5.7f;
            const float run = 3f;
            const float plateauWidth = 2f;
            float rise = Mathf.Tan(angle * Mathf.Deg2Rad) * run;
            float rampLength = Mathf.Sqrt((run * run) + (rise * rise));
            const float thickness = 0.12f;
            var primitives = new List<TerrainPrimitiveDefinition>
            {
                Box("slope_base", new Vector3(0f, -0.06f, 9f), new Vector3(8f, 0.12f, 8f), "rough_base"),
                Box("slope_plateau", new Vector3(0f, rise * 0.5f, 9f), new Vector3(plateauWidth, rise, plateauWidth), "slope_top"),
                Box("slope_west", new Vector3(-2.5f, (rise - thickness) * 0.5f, 9f), new Vector3(rampLength, thickness, plateauWidth), "slope", new Vector3(0f, 0f, angle)),
                Box("slope_east", new Vector3(2.5f, (rise - thickness) * 0.5f, 9f), new Vector3(rampLength, thickness, plateauWidth), "slope", new Vector3(0f, 0f, -angle)),
                Box("slope_south", new Vector3(0f, (rise - thickness) * 0.5f, 6.5f), new Vector3(plateauWidth, thickness, rampLength), "slope", new Vector3(-angle, 0f, 0f)),
                Box("slope_north", new Vector3(0f, (rise - thickness) * 0.5f, 11.5f), new Vector3(plateauWidth, thickness, rampLength), "slope", new Vector3(angle, 0f, 0f)),
            };

            return Module(
                "upstream_pyramid_slope",
                "官方缓坡",
                new Vector3(0f, rise + 0.125f, 9f),
                true,
                "训练基准 · 1.7°–5.7°",
                rise,
                1.7f,
                angle,
                new[] { 1, 2 },
                primitives.ToArray());
        }

        private static TerrainModuleDefinition CreateRollerSlope()
        {
            const float startX = -7f;
            const float entryLength = 2f;
            const float rampLength = 6f;
            const float runoutLength = 4f;
            const float thickness = 0.12f;
            float[] angles = { 2f, 11f, 20f };
            float[] laneZ = { -10.5f, -9f, -7.5f };
            var primitives = new List<TerrainPrimitiveDefinition>();

            for (int lane = 0; lane < angles.Length; lane++)
            {
                float angle = angles[lane];
                float drop = Mathf.Tan(angle * Mathf.Deg2Rad) * rampLength;
                float slopedLength = Mathf.Sqrt((rampLength * rampLength) + (drop * drop));
                string suffix = $"{angle:00}deg";
                primitives.Add(Box(
                    $"roller_entry_{suffix}",
                    new Vector3(startX + (entryLength * 0.5f), -0.05f, laneZ[lane]),
                    new Vector3(entryLength, 0.1f, 1.2f),
                    "roller_entry"));
                primitives.Add(Box(
                    $"roller_ramp_{suffix}",
                    new Vector3(startX + entryLength + (rampLength * 0.5f), (-drop - thickness) * 0.5f, laneZ[lane]),
                    new Vector3(slopedLength, thickness, 1.2f),
                    lane == 0 ? "slope_easy" : lane == 1 ? "slope_medium" : "slope_hard",
                    new Vector3(0f, 0f, -angle)));
                primitives.Add(Box(
                    $"roller_runout_{suffix}",
                    new Vector3(startX + entryLength + rampLength + (runoutLength * 0.5f), -drop - 0.05f, laneZ[lane]),
                    new Vector3(runoutLength, 0.1f, 1.2f),
                    "roller_runout"));
            }

            float easySpawnX = startX + entryLength + 0.3f;
            float easySpawnSurface = -Mathf.Tan(2f * Mathf.Deg2Rad) * 0.3f;
            return Module(
                "upstream_roller_slope",
                "官方滚轮长坡",
                new Vector3(easySpawnX, easySpawnSurface + 0.125f, laneZ[0]),
                true,
                "训练基准 · 2° / 11° / 20°",
                0f,
                2f,
                20f,
                new[] { 7, 8 },
                primitives.ToArray());
        }

        private static TerrainModuleDefinition CreateRockSteps()
        {
            var primitives = new List<TerrainPrimitiveDefinition>
            {
                Box("rocks_base", new Vector3(-8f, -0.05f, 9f), new Vector3(6f, 0.1f, 6f), "forest_floor"),
            };

            for (int index = 0; index < 7; index++)
            {
                float x = -10.4f + (index * 0.78f);
                float z = 9f + ((index % 2 == 0 ? -1f : 1f) * 0.18f);
                float height = 0.03f + ((index % 3) * 0.012f);
                primitives.Add(Box(
                    $"stepping_stone_{index:00}",
                    new Vector3(x, height * 0.5f, z),
                    new Vector3(0.58f, height, 0.58f),
                    "stone"));
            }

            Vector3[] rocks =
            {
                new Vector3(-9.7f, 0.14f, 10.6f),
                new Vector3(-8.5f, 0.19f, 7.6f),
                new Vector3(-7.0f, 0.12f, 10.3f),
                new Vector3(-5.8f, 0.16f, 8.1f),
            };
            for (int index = 0; index < rocks.Length; index++)
            {
                primitives.Add(Ellipsoid(
                    $"rock_{index:00}",
                    rocks[index],
                    new Vector3(0.42f + (index * 0.04f), 0.28f, 0.36f),
                    "boulder"));
            }

            return Module(
                "rock_steps",
                "岩石与踏石",
                new Vector3(-10.4f, 0.165f, 9f),
                false,
                "训练扩展",
                0.054f,
                0f,
                0f,
                new[] { 1 },
                primitives.ToArray());
        }

        private static TerrainModuleDefinition CreateStairsBridge()
        {
            var primitives = new List<TerrainPrimitiveDefinition>
            {
                Box("bridge_base", new Vector3(8f, -0.05f, 9f), new Vector3(6f, 0.1f, 6f), "forest_floor"),
            };

            const int stairCount = 5;
            const float stepDepth = 0.32f;
            const float stepHeight = 0.04f;
            float firstX = 5.4f;
            for (int step = 0; step < stairCount; step++)
            {
                float top = (step + 1) * stepHeight;
                primitives.Add(Box(
                    $"bridge_up_{step:00}",
                    new Vector3(firstX + (step * stepDepth), top * 0.5f, 9f),
                    new Vector3(stepDepth, top, 1.1f),
                    "bridge_step"));
                primitives.Add(Box(
                    $"bridge_down_{step:00}",
                    new Vector3(10.6f - (step * stepDepth), top * 0.5f, 9f),
                    new Vector3(stepDepth, top, 1.1f),
                    "bridge_step"));
            }

            primitives.Add(Box(
                "bridge_deck",
                new Vector3(8f, (stairCount * stepHeight) - 0.035f, 9f),
                new Vector3(2.0f, 0.07f, 1.1f),
                "bridge_deck"));

            return Module(
                "stairs_bridge",
                "标准楼梯与小桥",
                new Vector3(firstX, stepHeight + 0.125f, 9f),
                false,
                "训练扩展",
                stepHeight,
                0f,
                0f,
                new[] { 1 },
                primitives.ToArray());
        }

        private static TerrainModuleDefinition Module(
            string id,
            string displayName,
            Vector3 spawnPosition,
            bool isTrainingMatched,
            string difficulty,
            float maximumSurfaceVariationMeters,
            float minimumSlopeDegrees,
            float maximumSlopeDegrees,
            int[] recommendedPolicySlots,
            params TerrainPrimitiveDefinition[] primitives)
        {
            return new TerrainModuleDefinition(
                id,
                displayName,
                spawnPosition,
                0f,
                isTrainingMatched,
                difficulty,
                maximumSurfaceVariationMeters,
                minimumSlopeDegrees,
                maximumSlopeDegrees,
                recommendedPolicySlots,
                primitives);
        }

        private static TerrainPrimitiveDefinition Box(
            string id,
            Vector3 center,
            Vector3 size,
            string materialKey,
            Vector3 eulerAngles = default)
        {
            return new TerrainPrimitiveDefinition(
                id,
                TerrainPrimitiveKind.Box,
                center,
                size,
                eulerAngles,
                materialKey);
        }

        private static TerrainPrimitiveDefinition Ellipsoid(
            string id,
            Vector3 center,
            Vector3 size,
            string materialKey)
        {
            return new TerrainPrimitiveDefinition(
                id,
                TerrainPrimitiveKind.Ellipsoid,
                center,
                size,
                Vector3.zero,
                materialKey);
        }
    }
}
