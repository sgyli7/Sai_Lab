using System;
using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    public sealed class MicroDuckControlLoopContinuation
    {
        private readonly float[] previousAction;
        private readonly PolicyTargetFilterContinuation targetFilter;

        internal MicroDuckControlLoopContinuation(
            float[] previousAction,
            PolicyTargetFilterContinuation targetFilter)
        {
            PolicyContract.ValidateAction(previousAction);
            this.previousAction = (float[])previousAction.Clone();
            this.targetFilter = targetFilter ?? throw new ArgumentNullException(nameof(targetFilter));
        }

        internal void Restore(float[] actionDestination, PolicyTargetFilter filterDestination)
        {
            PolicyContract.ValidateAction(actionDestination);
            if (filterDestination == null)
            {
                throw new ArgumentNullException(nameof(filterDestination));
            }

            Array.Copy(previousAction, actionDestination, previousAction.Length);
            filterDestination.RestoreContinuation(targetFilter);
        }
    }

    public sealed class MicroDuckControlLoop : IDisposable
    {
        private readonly IPolicyRuntime runtime;
        private readonly PolicyCommandState commands;
        private readonly float[] homePositionRad;
        private readonly float actionScale;
        private readonly PolicyTargetFilter targetFilter = new PolicyTargetFilter();
        private readonly float[] observation = new float[PolicyContract.ObservationCount];
        private readonly float[] rawAction = new float[PolicyContract.ActionCount];
        private readonly float[] previousAction = new float[PolicyContract.ActionCount];
        private readonly float[] command = new float[PolicyObservationBuilder.CommandCount];
        private bool disposed;

        public MicroDuckControlLoop(
            IPolicyRuntime runtime,
            PolicyCommandState commands,
            float[] homePositionRad,
            float actionScale)
            : this(runtime, commands, homePositionRad, actionScale, null)
        {
        }

        public MicroDuckControlLoop(
            IPolicyRuntime runtime,
            PolicyCommandState commands,
            float[] homePositionRad,
            float actionScale,
            MicroDuckControlLoopContinuation continuation)
        {
            this.runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
            this.commands = commands ?? throw new ArgumentNullException(nameof(commands));
            PolicyContract.ValidateAction(homePositionRad);
            this.homePositionRad = (float[])homePositionRad.Clone();
            if (actionScale <= 0f || float.IsNaN(actionScale) || float.IsInfinity(actionScale))
            {
                throw new ArgumentOutOfRangeException(nameof(actionScale));
            }

            this.actionScale = actionScale;
            if (continuation != null)
            {
                continuation.Restore(previousAction, targetFilter);
                Array.Copy(previousAction, rawAction, rawAction.Length);
            }
        }

        public string BackendName => runtime.BackendName;
        public float[] LastObservation => (float[])observation.Clone();
        public float[] LastRawAction => (float[])rawAction.Clone();
        public float[] LastCommand => (float[])command.Clone();

        public MicroDuckControlLoopContinuation CaptureContinuation()
        {
            if (disposed)
            {
                throw new ObjectDisposedException(nameof(MicroDuckControlLoop));
            }

            return new MicroDuckControlLoopContinuation(
                previousAction,
                targetFilter.CaptureContinuation());
        }

        /// <summary>
        /// Clears recurrent observation/action history and target smoothing
        /// without recreating the loaded inference worker.
        /// </summary>
        public void Reset()
        {
            if (disposed)
            {
                throw new ObjectDisposedException(nameof(MicroDuckControlLoop));
            }

            Array.Clear(observation, 0, observation.Length);
            Array.Clear(rawAction, 0, rawAction.Length);
            Array.Clear(previousAction, 0, previousAction.Length);
            Array.Clear(command, 0, command.Length);
            targetFilter.Reset();
        }

        public bool Step(
            Vector3 tuanjieLocalAngularVelocity,
            Vector3 tuanjieLocalProjectedGravity,
            float[] jointPositionRad,
            float[] jointVelocityRadPerSecond,
            float nowSeconds,
            float[] targetPositionRad,
            out string error)
        {
            return StepInternal(
                tuanjieLocalAngularVelocity,
                tuanjieLocalProjectedGravity,
                jointPositionRad,
                jointVelocityRadPerSecond,
                nowSeconds,
                targetPositionRad,
                false,
                out error);
        }

        public bool StepMuJoCo(
            Vector3 mujocoLocalAngularVelocity,
            Vector3 mujocoLocalProjectedGravity,
            float[] jointPositionRad,
            float[] jointVelocityRadPerSecond,
            float nowSeconds,
            float[] targetPositionRad,
            out string error)
        {
            return StepInternal(
                mujocoLocalAngularVelocity,
                mujocoLocalProjectedGravity,
                jointPositionRad,
                jointVelocityRadPerSecond,
                nowSeconds,
                targetPositionRad,
                true,
                out error);
        }

        private bool StepInternal(
            Vector3 localAngularVelocity,
            Vector3 localProjectedGravity,
            float[] jointPositionRad,
            float[] jointVelocityRadPerSecond,
            float nowSeconds,
            float[] targetPositionRad,
            bool valuesAreAlreadyInMuJoCoBasis,
            out string error)
        {
            if (disposed)
            {
                throw new ObjectDisposedException(nameof(MicroDuckControlLoop));
            }

            PolicyContract.ValidateAction(targetPositionRad);
            commands.BuildCommand(nowSeconds, command);
            if (valuesAreAlreadyInMuJoCoBasis)
            {
                PolicyObservationBuilder.BuildMuJoCo(
                    localAngularVelocity,
                    localProjectedGravity,
                    jointPositionRad,
                    jointVelocityRadPerSecond,
                    homePositionRad,
                    previousAction,
                    command,
                    observation);
            }
            else
            {
                PolicyObservationBuilder.Build(
                    localAngularVelocity,
                    localProjectedGravity,
                    jointPositionRad,
                    jointVelocityRadPerSecond,
                    homePositionRad,
                    previousAction,
                    command,
                    observation);
            }

            if (!PolicyRuntimeContract.TryEvaluate(runtime, observation, rawAction, out error))
            {
                Array.Copy(homePositionRad, targetPositionRad, targetPositionRad.Length);
                return false;
            }

            targetFilter.Apply(homePositionRad, rawAction, actionScale, targetPositionRad);
            Array.Copy(rawAction, previousAction, rawAction.Length);
            return true;
        }

        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            runtime.Dispose();
            disposed = true;
        }
    }
}
