using System;

namespace AgenticRobot.MicroDuck
{
    public sealed class PolicyTargetFilter
    {
        private const float HeadAlpha = 0.5f;
        private const float LegAlpha = 0.7f;
        private readonly float[] previous = new float[PolicyContract.ActionCount];
        private bool initialized;

        public void Apply(
            float[] homePositionRad,
            float[] rawAction,
            float actionScale,
            float[] destination)
        {
            PolicyContract.ValidateAction(homePositionRad);
            PolicyContract.ValidateAction(rawAction);
            PolicyContract.ValidateAction(destination);
            if (float.IsNaN(actionScale) || float.IsInfinity(actionScale) || actionScale <= 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(actionScale), "Action scale must be finite and positive.");
            }

            for (int index = 0; index < PolicyContract.ActionCount; index++)
            {
                float target = homePositionRad[index] + rawAction[index] * actionScale;
                if (initialized)
                {
                    float alpha = index >= 5 && index <= 8 ? HeadAlpha : LegAlpha;
                    target = alpha * target + (1f - alpha) * previous[index];
                }

                destination[index] = target;
            }

            Array.Copy(destination, previous, destination.Length);
            initialized = true;
        }

        public void Reset()
        {
            Array.Clear(previous, 0, previous.Length);
            initialized = false;
        }

        internal PolicyTargetFilterContinuation CaptureContinuation()
        {
            return new PolicyTargetFilterContinuation(previous, initialized);
        }

        internal void RestoreContinuation(PolicyTargetFilterContinuation continuation)
        {
            if (continuation == null)
            {
                throw new ArgumentNullException(nameof(continuation));
            }

            continuation.CopyPreviousTo(previous);
            initialized = continuation.Initialized;
        }
    }

    internal sealed class PolicyTargetFilterContinuation
    {
        private readonly float[] previous;

        public PolicyTargetFilterContinuation(float[] previousTarget, bool initialized)
        {
            PolicyContract.ValidateAction(previousTarget);
            previous = (float[])previousTarget.Clone();
            Initialized = initialized;
        }

        public bool Initialized { get; }

        public void CopyPreviousTo(float[] destination)
        {
            PolicyContract.ValidateAction(destination);
            Array.Copy(previous, destination, previous.Length);
        }
    }
}
