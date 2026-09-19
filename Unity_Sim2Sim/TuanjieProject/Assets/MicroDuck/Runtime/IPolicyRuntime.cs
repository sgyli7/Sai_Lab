using System;

namespace AgenticRobot.MicroDuck
{
    public interface IPolicyRuntime : IDisposable
    {
        string BackendName { get; }

        void Evaluate(float[] observation, float[] actionDestination);
    }

    public static class PolicyRuntimeContract
    {
        private const float MaximumActionMagnitude = 5f;

        public static bool TryEvaluate(
            IPolicyRuntime runtime,
            float[] observation,
            float[] actionDestination,
            out string error)
        {
            if (runtime == null)
            {
                throw new ArgumentNullException(nameof(runtime));
            }

            PolicyContract.ValidateObservation(observation);
            PolicyContract.ValidateAction(actionDestination);
            try
            {
                runtime.Evaluate(observation, actionDestination);
                for (int index = 0; index < actionDestination.Length; index++)
                {
                    float action = actionDestination[index];
                    if (float.IsNaN(action) || float.IsInfinity(action))
                    {
                        Array.Clear(actionDestination, 0, actionDestination.Length);
                        error = $"Policy returned a non-finite action at index {index}.";
                        return false;
                    }

                    if (Math.Abs(action) > MaximumActionMagnitude)
                    {
                        Array.Clear(actionDestination, 0, actionDestination.Length);
                        error = $"Policy returned an out-of-range action at index {index}.";
                        return false;
                    }
                }

                error = string.Empty;
                return true;
            }
            catch (Exception exception)
            {
                Array.Clear(actionDestination, 0, actionDestination.Length);
                error = $"{runtime.BackendName} inference failed: {exception.Message}";
                return false;
            }
        }
    }
}
