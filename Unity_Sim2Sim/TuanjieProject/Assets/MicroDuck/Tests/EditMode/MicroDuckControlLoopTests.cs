using System;
using NUnit.Framework;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class MicroDuckControlLoopTests
    {
        [Test]
        public void ClosesObservationInferenceFeedbackAndTargetLoopWithoutAllocatingOutputs()
        {
            var commands = new PolicyCommandState();
            commands.SelectSlot(1);
            commands.SetTwist(0.2f, 0f, 0f);
            var runtime = new RecordingRuntime();
            var loop = new MicroDuckControlLoop(runtime, commands, new float[14], 1f);
            var positions = new float[14];
            var velocities = new float[14];
            var targets = new float[14];

            bool first = loop.Step(
                CoordinateBasis.MuJoCoToTuanjieAxial(new Vector3(1f, 2f, 3f)),
                CoordinateBasis.MuJoCoToTuanjie(new Vector3(0f, 0f, -1f)),
                positions,
                velocities,
                0f,
                targets,
                out string firstError);
            bool second = loop.Step(
                Vector3.zero,
                CoordinateBasis.MuJoCoToTuanjie(new Vector3(0f, 0f, -1f)),
                positions,
                velocities,
                0.02f,
                targets,
                out string secondError);

            Assert.That(first, Is.True, firstError);
            Assert.That(second, Is.True, secondError);
            Assert.That(runtime.Calls, Is.EqualTo(2));
            Assert.That(runtime.ObservedPreviousActionsOnSecondCall, Is.All.EqualTo(0.5f));
            Assert.That(runtime.LastObservation[48], Is.EqualTo(0.2f));
            Assert.That(targets, Is.All.EqualTo(0.5f));
        }

        [Test]
        public void FailedInferenceReturnsHomePoseAsSafeTarget()
        {
            var home = new float[14];
            Array.Fill(home, 0.25f);
            var loop = new MicroDuckControlLoop(
                new NonFiniteRuntime(), new PolicyCommandState(), home, 1f);
            var targets = new float[14];

            bool success = loop.Step(
                Vector3.zero, Vector3.down, new float[14], new float[14],
                0f, targets, out string error);

            Assert.That(success, Is.False);
            Assert.That(targets, Is.All.EqualTo(0.25f));
            StringAssert.Contains("non-finite", error);
        }

        [Test]
        public void NativeMuJoCoStepKeepsGyroAndGravityInThePolicyBasis()
        {
            var runtime = new RecordingRuntime();
            var loop = new MicroDuckControlLoop(
                runtime, new PolicyCommandState(), new float[14], 1f);

            Assert.That(loop.StepMuJoCo(
                new Vector3(1f, 2f, 3f),
                new Vector3(4f, 5f, 6f),
                new float[14],
                new float[14],
                0f,
                new float[14],
                out string error), Is.True, error);

            Assert.That(
                runtime.LastObservation[0..6],
                Is.EqualTo(new[] { 1f, 2f, 3f, 4f, 5f, 6f }));
        }

        [Test]
        public void ContinuationCarriesPreviousActionAndTargetFilterAcrossPolicyRuntimes()
        {
            var commands = new PolicyCommandState();
            var firstRuntime = new ConstantRuntime(0.5f, "first");
            var firstLoop = new MicroDuckControlLoop(
                firstRuntime, commands, new float[PolicyContract.ActionCount], 1f);
            var positions = new float[PolicyContract.ActionCount];
            var velocities = new float[PolicyContract.ActionCount];
            var firstTargets = new float[PolicyContract.ActionCount];

            Assert.That(firstLoop.Step(
                Vector3.zero,
                Vector3.down,
                positions,
                velocities,
                0f,
                firstTargets,
                out string firstError), Is.True, firstError);

            MicroDuckControlLoopContinuation continuation = firstLoop.CaptureContinuation();
            var secondRuntime = new RecordingConstantRuntime(1f, "second");
            var secondLoop = new MicroDuckControlLoop(
                secondRuntime, commands, new float[PolicyContract.ActionCount], 0.8f, continuation);
            var secondTargets = new float[PolicyContract.ActionCount];

            Assert.That(secondLoop.LastRawAction, Is.EqualTo(firstLoop.LastRawAction));
            Assert.That(secondLoop.Step(
                Vector3.zero,
                Vector3.down,
                positions,
                velocities,
                0.02f,
                secondTargets,
                out string secondError), Is.True, secondError);

            Assert.That(
                secondRuntime.ObservedPreviousAction,
                Is.All.EqualTo(0.5f).Within(1e-6f),
                "The recurrent previous-action observation must survive a same-rig policy hot swap.");
            Assert.That(
                secondTargets[0],
                Is.EqualTo(0.7f * 0.8f + 0.3f * 0.5f).Within(1e-6f),
                "Leg targets must continue from the previous filter output.");
            Assert.That(
                secondTargets[5],
                Is.EqualTo(0.5f * 0.8f + 0.5f * 0.5f).Within(1e-6f),
                "Head targets must continue from the previous filter output.");
        }

        [Test]
        public void ResetMakesTheNextTickEquivalentToAFreshControlLoop()
        {
            var commands = new PolicyCommandState();
            commands.SelectSlot(1);
            var runtime = new MutableRecordingRuntime { Action = 0.75f };
            var loop = new MicroDuckControlLoop(
                runtime, commands, new float[PolicyContract.ActionCount], 1f);
            var targets = new float[PolicyContract.ActionCount];

            Assert.That(loop.StepMuJoCo(
                Vector3.zero, Vector3.down,
                new float[PolicyContract.ActionCount],
                new float[PolicyContract.ActionCount],
                1f, targets, out string warmupError), Is.True, warmupError);

            runtime.Action = -0.25f;
            loop.Reset();
            Assert.That(loop.StepMuJoCo(
                Vector3.zero, Vector3.down,
                new float[PolicyContract.ActionCount],
                new float[PolicyContract.ActionCount],
                0f, targets, out string resetError), Is.True, resetError);
            float[] resetObservation = (float[])runtime.LastObservation.Clone();
            float[] resetTargets = (float[])targets.Clone();

            var freshRuntime = new MutableRecordingRuntime { Action = -0.25f };
            var freshLoop = new MicroDuckControlLoop(
                freshRuntime, commands, new float[PolicyContract.ActionCount], 1f);
            var freshTargets = new float[PolicyContract.ActionCount];
            Assert.That(freshLoop.StepMuJoCo(
                Vector3.zero, Vector3.down,
                new float[PolicyContract.ActionCount],
                new float[PolicyContract.ActionCount],
                0f, freshTargets, out string freshError), Is.True, freshError);

            Assert.That(resetObservation, Is.EqualTo(freshRuntime.LastObservation).Within(1e-6f));
            Assert.That(resetObservation[34..48], Is.All.Zero.Within(1e-6f));
            Assert.That(resetTargets, Is.EqualTo(freshTargets).Within(1e-6f));
            Assert.That(resetTargets, Is.All.EqualTo(-0.25f).Within(1e-6f),
                "The first target after reset must not blend with the pre-reset EMA value.");
        }

        private sealed class RecordingRuntime : IPolicyRuntime
        {
            public int Calls { get; private set; }
            public float[] LastObservation { get; } = new float[61];
            public float[] ObservedPreviousActionsOnSecondCall { get; } = new float[14];
            public string BackendName => "recording";

            public void Evaluate(float[] observation, float[] actionDestination)
            {
                Calls++;
                Array.Copy(observation, LastObservation, observation.Length);
                if (Calls == 2)
                {
                    Array.Copy(observation, 34, ObservedPreviousActionsOnSecondCall, 0, 14);
                }

                Array.Fill(actionDestination, 0.5f);
            }

            public void Dispose()
            {
            }
        }

        private sealed class MutableRecordingRuntime : IPolicyRuntime
        {
            public float Action { get; set; }
            public float[] LastObservation { get; } =
                new float[PolicyContract.ObservationCount];
            public string BackendName => "mutable-recording";

            public void Evaluate(float[] observation, float[] actionDestination)
            {
                Array.Copy(observation, LastObservation, observation.Length);
                Array.Fill(actionDestination, Action);
            }

            public void Dispose()
            {
            }
        }

        private sealed class NonFiniteRuntime : IPolicyRuntime
        {
            public string BackendName => "non-finite";

            public void Evaluate(float[] observation, float[] actionDestination)
            {
                Array.Fill(actionDestination, float.PositiveInfinity);
            }

            public void Dispose()
            {
            }
        }

        private class ConstantRuntime : IPolicyRuntime
        {
            private readonly float action;

            public ConstantRuntime(float action, string backendName)
            {
                this.action = action;
                BackendName = backendName;
            }

            public string BackendName { get; }

            public virtual void Evaluate(float[] observation, float[] actionDestination)
            {
                Array.Fill(actionDestination, action);
            }

            public void Dispose()
            {
            }
        }

        private sealed class RecordingConstantRuntime : ConstantRuntime
        {
            public RecordingConstantRuntime(float action, string backendName)
                : base(action, backendName)
            {
            }

            public float[] ObservedPreviousAction { get; } =
                new float[PolicyContract.ActionCount];

            public override void Evaluate(float[] observation, float[] actionDestination)
            {
                Array.Copy(
                    observation,
                    34,
                    ObservedPreviousAction,
                    0,
                    ObservedPreviousAction.Length);
                base.Evaluate(observation, actionDestination);
            }
        }
    }
}
