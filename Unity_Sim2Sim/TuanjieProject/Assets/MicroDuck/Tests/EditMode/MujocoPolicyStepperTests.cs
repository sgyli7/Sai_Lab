using System;
using System.IO;
using AgenticRobot.MicroDuck.Mujoco;
using Mujoco;
using NUnit.Framework;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class MujocoPolicyStepperTests
    {
        [Test]
        public unsafe void BindsOfficialModelResetsHomeAndClosesOneBarracudaCompatibleTick()
        {
            string scenePath = OfficialScenePath("scene.xml");
            MujocoLib.mjModel_* model = null;
            MujocoLib.mjData_* data = null;
            try
            {
                model = MjEngineTool.LoadModelFromFile(scenePath);
                data = MujocoLib.mj_makeData(model);
                var runtime = new RecordingRuntime(0.25f);
                var commands = new PolicyCommandState();
                commands.SelectSlot(2);
                using (var stepper = new MujocoPolicyStepper(
                    model, data, runtime, commands, RobotVariant.Legged, 1f))
                {
                    stepper.ResetToHome();
                    Assert.That(stepper.ServoNames.Length, Is.EqualTo(14));
                    Assert.That(data->qpos[2], Is.EqualTo(0.125).Within(1e-9));

                    Assert.That(stepper.TickPolicy(0f, out string error), Is.True, error);
                    Assert.That(runtime.Calls, Is.EqualTo(1));
                    Assert.That(runtime.LastObservation[0..3], Is.All.EqualTo(0f).Within(1e-5f));
                    Assert.That(
                        runtime.LastObservation[3..6],
                        Is.EqualTo(new[] { 0f, 0f, -1f }).Within(1e-5f));
                    Assert.That(runtime.LastObservation[6..20], Is.All.EqualTo(0f).Within(1e-5f));
                    Assert.That(stepper.LastTargets[0], Is.EqualTo(0.25f).Within(1e-6f));
                    Assert.That(data->ctrl[0], Is.EqualTo(0.25).Within(1e-6));
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

        [Test]
        public unsafe void ResetClearsPolicyHistoryAndSkillPhaseBeforeTheNextNativeTick()
        {
            string scenePath = OfficialScenePath("scene.xml");
            MujocoLib.mjModel_* model = null;
            MujocoLib.mjData_* data = null;
            try
            {
                model = MjEngineTool.LoadModelFromFile(scenePath);
                data = MujocoLib.mj_makeData(model);
                var runtime = new MutableRecordingRuntime { Value = 0.25f };
                var commands = new PolicyCommandState();
                commands.SelectSlot(3);
                using (var stepper = new MujocoPolicyStepper(
                    model, data, runtime, commands, RobotVariant.Legged, 1f))
                {
                    stepper.ResetToHome();
                    Assert.That(stepper.TickPolicy(0f, out string baselineError),
                        Is.True, baselineError);
                    float[] baselineObservation = stepper.LastObservation;
                    float[] baselineTargets = stepper.LastTargets;

                    runtime.Value = -0.75f;
                    commands.TriggerSkill(0.02f);
                    Assert.That(stepper.TickPolicy(0.02f, out string dirtyError),
                        Is.True, dirtyError);

                    commands.Reset();
                    stepper.ResetToHome();
                    runtime.Value = 0.25f;
                    Assert.That(stepper.TickPolicy(0f, out string resetError),
                        Is.True, resetError);

                    Assert.That(stepper.LastObservation,
                        Is.EqualTo(baselineObservation).Within(1e-6f));
                    Assert.That(stepper.LastObservation[34..48], Is.All.Zero.Within(1e-6f));
                    Assert.That(stepper.LastCommand[0], Is.Zero.Within(1e-6f));
                    Assert.That(stepper.LastTargets,
                        Is.EqualTo(baselineTargets).Within(1e-6f));
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

        private static string OfficialScenePath(string fileName)
        {
            return Path.GetFullPath(Path.Combine(
                Application.dataPath,
                "..",
                "..",
                ".cache",
                "upstream",
                "microduck_rl",
                "src",
                "mjlab_microduck",
                "robot",
                "microduck",
                fileName));
        }

        private sealed class RecordingRuntime : IPolicyRuntime
        {
            private readonly float value;

            public RecordingRuntime(float value)
            {
                this.value = value;
            }

            public int Calls { get; private set; }
            public float[] LastObservation { get; } = new float[PolicyContract.ObservationCount];
            public string BackendName => "recording";

            public void Evaluate(float[] observation, float[] actionDestination)
            {
                Calls++;
                Array.Copy(observation, LastObservation, observation.Length);
                Array.Fill(actionDestination, value);
            }

            public void Dispose()
            {
            }
        }

        private sealed class MutableRecordingRuntime : IPolicyRuntime
        {
            public float Value { get; set; }
            public string BackendName => "mutable-recording";

            public void Evaluate(float[] observation, float[] actionDestination)
            {
                Array.Fill(actionDestination, Value);
            }

            public void Dispose()
            {
            }
        }
    }
}
