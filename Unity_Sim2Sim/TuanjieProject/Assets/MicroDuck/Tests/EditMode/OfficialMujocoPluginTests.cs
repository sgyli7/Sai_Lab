using System.IO;
using Mujoco;
using NUnit.Framework;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class OfficialMujocoPluginTests
    {
        [Test]
        public unsafe void NativePluginLoadsAndStepsTheOfficialMicroDuckScene()
        {
            string repositoryRoot = Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", ".."));
            string scenePath = Path.Combine(
                repositoryRoot,
                ".cache",
                "upstream",
                "microduck_rl",
                "src",
                "mjlab_microduck",
                "robot",
                "microduck",
                "scene.xml");
            Assert.That(File.Exists(scenePath), Is.True, scenePath);

            MujocoLib.mjModel_* model = null;
            MujocoLib.mjData_* data = null;
            try
            {
                model = MjEngineTool.LoadModelFromFile(scenePath);
                Assert.That(model == null, Is.False);
                Assert.That(model->nq, Is.EqualTo(21));
                Assert.That(model->nv, Is.EqualTo(20));
                Assert.That(model->nu, Is.EqualTo(14));

                data = MujocoLib.mj_makeData(model);
                Assert.That(data == null, Is.False);
                MujocoLib.mj_resetDataKeyframe(model, data, 0);
                MujocoLib.mj_forward(model, data);
                double startTime = data->time;
                for (int step = 0; step < 20; step++)
                {
                    MujocoLib.mj_step(model, data);
                }

                Assert.That(data->time, Is.GreaterThan(startTime));
                for (int index = 0; index < (int)model->nq; index++)
                {
                    Assert.That(double.IsNaN(data->qpos[index]), Is.False, $"qpos[{index}]");
                    Assert.That(double.IsInfinity(data->qpos[index]), Is.False, $"qpos[{index}]");
                }
            }
            finally
            {
                if (data != null)
                {
                    MujocoLib.mj_deleteData(data);
                }

                if (model != null)
                {
                    MujocoLib.mj_deleteModel(model);
                }
            }
        }
    }
}
