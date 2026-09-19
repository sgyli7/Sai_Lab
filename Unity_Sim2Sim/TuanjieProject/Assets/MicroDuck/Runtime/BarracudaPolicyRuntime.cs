using System;
using Unity.Barracuda;

namespace AgenticRobot.MicroDuck
{
    public sealed class BarracudaPolicyRuntime : IPolicyRuntime
    {
        private readonly IWorker worker;
        private readonly string inputName;
        private readonly string outputName;
        private bool disposed;

        public BarracudaPolicyRuntime(
            NNModel modelAsset,
            WorkerFactory.Type workerType = WorkerFactory.Type.CSharpBurst)
        {
            if (modelAsset == null)
            {
                throw new ArgumentNullException(nameof(modelAsset));
            }

            Model model = ModelLoader.Load(modelAsset);
            if (model.inputs.Count != 1 || model.outputs.Count != 1)
            {
                throw new ArgumentException("Policy must expose exactly one input and one output.", nameof(modelAsset));
            }

            Model.Input input = model.inputs[0];
            if (StaticElementCount(input.shape) != PolicyContract.ObservationCount)
            {
                throw new ArgumentException(
                    $"Policy input '{input.name}' must contain {PolicyContract.ObservationCount} elements.",
                    nameof(modelAsset));
            }

            inputName = input.name;
            outputName = model.outputs[0];
            worker = WorkerFactory.CreateWorker(workerType, model);
        }

        public string BackendName => "Barracuda 3.0.1 CPU";

        public void Evaluate(float[] observation, float[] actionDestination)
        {
            if (disposed)
            {
                throw new ObjectDisposedException(nameof(BarracudaPolicyRuntime));
            }

            PolicyContract.ValidateObservation(observation);
            PolicyContract.ValidateAction(actionDestination);
            using (var input = new Tensor(1, PolicyContract.ObservationCount, observation, inputName))
            {
                worker.Execute(input);
                Tensor output = worker.PeekOutput(outputName);
                if (output.length != PolicyContract.ActionCount)
                {
                    throw new InvalidOperationException(
                        $"Policy output '{outputName}' contains {output.length} elements; expected {PolicyContract.ActionCount}.");
                }

                for (int index = 0; index < actionDestination.Length; index++)
                {
                    actionDestination[index] = output[index];
                }
            }
        }

        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            worker.Dispose();
            disposed = true;
        }

        private static int StaticElementCount(int[] shape)
        {
            if (shape == null || shape.Length == 0)
            {
                return 0;
            }

            int count = 1;
            foreach (int dimension in shape)
            {
                if (dimension <= 0)
                {
                    return 0;
                }

                count *= dimension;
            }

            return count;
        }
    }
}
