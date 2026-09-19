using System;
using NUnit.Framework;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class PolicyRuntimeContractTests
    {
        [Test]
        public void ExecutesAValidRuntimeIntoCallerOwnedStorage()
        {
            var runtime = new ConstantRuntime(0.25f);
            var output = new float[14];

            bool success = PolicyRuntimeContract.TryEvaluate(
                runtime, new float[61], output, out string error);

            Assert.That(success, Is.True, error);
            Assert.That(output, Is.All.EqualTo(0.25f));
            Assert.That(error, Is.Empty);
        }

        [Test]
        public void NonFiniteOutputFailsClosedAndClearsEveryAction()
        {
            var runtime = new ConstantRuntime(float.NaN);
            var output = new float[14];

            bool success = PolicyRuntimeContract.TryEvaluate(
                runtime, new float[61], output, out string error);

            Assert.That(success, Is.False);
            Assert.That(output, Is.All.Zero);
            StringAssert.Contains("non-finite", error);
        }

        [Test]
        public void OutOfRangeFiniteActionFailsClosedAndClearsEveryAction()
        {
            var positiveOverflow = new ConstantRuntime(5.5f);
            var positiveOutput = new float[14];
            var negativeOverflow = new ConstantRuntime(-5.5f);
            var negativeOutput = new float[14];

            bool positiveSuccess = PolicyRuntimeContract.TryEvaluate(
                positiveOverflow, new float[61], positiveOutput, out string positiveError);
            bool negativeSuccess = PolicyRuntimeContract.TryEvaluate(
                negativeOverflow, new float[61], negativeOutput, out string negativeError);

            Assert.That(positiveSuccess, Is.False);
            Assert.That(positiveOutput, Is.All.Zero);
            StringAssert.Contains("out-of-range", positiveError);
            Assert.That(negativeSuccess, Is.False);
            Assert.That(negativeOutput, Is.All.Zero);
            StringAssert.Contains("out-of-range", negativeError);
        }

        [Test]
        public void RuntimeExceptionFailsClosedAndReturnsDiagnostic()
        {
            var runtime = new ThrowingRuntime();
            var output = new float[14];

            bool success = PolicyRuntimeContract.TryEvaluate(
                runtime, new float[61], output, out string error);

            Assert.That(success, Is.False);
            Assert.That(output, Is.All.Zero);
            StringAssert.Contains("synthetic failure", error);
        }

        [Test]
        public void TargetFilterMatchesTheSharedHeadAndLegEmaContract()
        {
            var filter = new PolicyTargetFilter();
            var home = new float[14];
            var action = new float[14];
            var output = new float[14];
            Array.Fill(action, 1f);

            filter.Apply(home, action, 2f, output);
            Assert.That(output, Is.All.EqualTo(2f));

            Array.Fill(action, 0f);
            filter.Apply(home, action, 2f, output);
            for (int index = 0; index < output.Length; index++)
            {
                float expected = index >= 5 && index <= 8 ? 1f : 0.6f;
                Assert.That(output[index], Is.EqualTo(expected).Within(1e-6f));
            }
        }

        private sealed class ConstantRuntime : IPolicyRuntime
        {
            private readonly float value;

            public ConstantRuntime(float value)
            {
                this.value = value;
            }

            public string BackendName => "test";

            public void Evaluate(float[] observation, float[] actionDestination)
            {
                Array.Fill(actionDestination, value);
            }

            public void Dispose()
            {
            }
        }

        private sealed class ThrowingRuntime : IPolicyRuntime
        {
            public string BackendName => "test";

            public void Evaluate(float[] observation, float[] actionDestination)
            {
                throw new InvalidOperationException("synthetic failure");
            }

            public void Dispose()
            {
            }
        }
    }
}
