using System.Linq;
using AgenticRobot.MicroDuck.Mujoco;
using NUnit.Framework;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class MujocoTerrainCatalogTests
    {
        [Test]
        public void ContainsTheSevenApprovedModulesInStableOrder()
        {
            Assert.That(
                MujocoTerrainCatalog.Modules.Select(module => module.Id),
                Is.EqualTo(new[]
                {
                    "flat_plaza",
                    "upstream_pyramid_stairs",
                    "upstream_random_grid",
                    "upstream_pyramid_slope",
                    "upstream_roller_slope",
                    "rock_steps",
                    "stairs_bridge",
                }));
        }

        [Test]
        public void EveryModuleHasCollisionGeometryAndAUniqueSafeSpawn()
        {
            Assert.That(MujocoTerrainCatalog.Modules, Has.All.Matches<TerrainModuleDefinition>(
                module => module.Primitives.Count > 0));
            Assert.That(MujocoTerrainCatalog.Modules, Has.All.Matches<TerrainModuleDefinition>(
                module => module.Primitives.All(primitive => primitive.CollisionEnabled)));

            int uniqueSpawns = MujocoTerrainCatalog.Modules
                .Select(module => module.SpawnPosition)
                .Distinct()
                .Count();
            Assert.That(uniqueSpawns, Is.EqualTo(MujocoTerrainCatalog.Modules.Count));
        }

        [Test]
        public void UpstreamRoughGeometryRespectsMicroDuckScaleLimits()
        {
            TerrainModuleDefinition stairs =
                MujocoTerrainCatalog.Find("upstream_pyramid_stairs");
            TerrainModuleDefinition grid =
                MujocoTerrainCatalog.Find("upstream_random_grid");
            TerrainModuleDefinition slope =
                MujocoTerrainCatalog.Find("upstream_pyramid_slope");
            TerrainModuleDefinition roller =
                MujocoTerrainCatalog.Find("upstream_roller_slope");

            Assert.That(stairs.MaximumSurfaceVariationMeters, Is.EqualTo(0.015f).Within(1e-6f));
            Assert.That(grid.MaximumSurfaceVariationMeters, Is.EqualTo(0.010f).Within(1e-6f));
            Assert.That(slope.MinimumSlopeDegrees, Is.EqualTo(1.7f).Within(0.1f));
            Assert.That(slope.MaximumSlopeDegrees, Is.EqualTo(5.7f).Within(0.1f));
            Assert.That(roller.MinimumSlopeDegrees, Is.EqualTo(2f).Within(1e-6f));
            Assert.That(roller.MaximumSlopeDegrees, Is.EqualTo(20f).Within(1e-6f));
        }

        [Test]
        public void RandomGridGenerationIsDeterministic()
        {
            TerrainModuleDefinition first = MujocoTerrainCatalog.CreateModule(
                "upstream_random_grid");
            TerrainModuleDefinition second = MujocoTerrainCatalog.CreateModule(
                "upstream_random_grid");

            Assert.That(first.Primitives.Count, Is.EqualTo(290));
            Assert.That(second.Primitives.Count, Is.EqualTo(first.Primitives.Count));
            for (int index = 0; index < first.Primitives.Count; index++)
            {
                Assert.That(second.Primitives[index], Is.EqualTo(first.Primitives[index]));
            }
        }

        [Test]
        public void CompatibilityLabelsSeparateTrainingMatchesFromExtensions()
        {
            Assert.That(MujocoTerrainCatalog.Find("flat_plaza").IsTrainingMatched, Is.True);
            Assert.That(MujocoTerrainCatalog.Find("upstream_random_grid").IsTrainingMatched, Is.True);
            Assert.That(MujocoTerrainCatalog.Find("upstream_roller_slope").IsTrainingMatched, Is.True);
            Assert.That(MujocoTerrainCatalog.Find("rock_steps").IsTrainingMatched, Is.False);
            Assert.That(MujocoTerrainCatalog.Find("stairs_bridge").IsTrainingMatched, Is.False);
        }
    }
}
