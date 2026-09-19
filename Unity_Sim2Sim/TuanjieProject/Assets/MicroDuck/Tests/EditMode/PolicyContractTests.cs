using System;
using NUnit.Framework;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class PolicyContractTests
    {
        [Test]
        public void ValidatesTheSharedPolicyShape()
        {
            Assert.That(PolicyContract.ObservationCount, Is.EqualTo(61));
            Assert.That(PolicyContract.ActionCount, Is.EqualTo(14));
            Assert.DoesNotThrow(() => PolicyContract.ValidateObservation(new float[61]));
            Assert.DoesNotThrow(() => PolicyContract.ValidateAction(new float[14]));
        }

        [Test]
        public void RejectsWrongPolicyWidthsBeforeInferenceOrActuation()
        {
            Assert.Throws<ArgumentException>(() => PolicyContract.ValidateObservation(new float[60]));
            Assert.Throws<ArgumentException>(() => PolicyContract.ValidateAction(new float[15]));
        }
    }
}
