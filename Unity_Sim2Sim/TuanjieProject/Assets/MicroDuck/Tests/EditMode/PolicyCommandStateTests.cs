using System.Linq;
using NUnit.Framework;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class PolicyCommandStateTests
    {
        [Test]
        public void CatalogPinsAllNineOfficialPoliciesAndRobotVariants()
        {
            Assert.That(PolicyCatalog.Entries, Has.Length.EqualTo(9));
            Assert.That(PolicyCatalog.Entries[0].FileName, Is.EqualTo("alpha_walking.onnx"));
            Assert.That(PolicyCatalog.Entries[6].FileName, Is.EqualTo("roller.onnx"));
            Assert.That(PolicyCatalog.Entries[8].FileName, Is.EqualTo("roulade.onnx"));
            Assert.That(PolicyCatalog.Entries[0].RobotVariant, Is.EqualTo(RobotVariant.Legged));
            Assert.That(PolicyCatalog.Entries[6].RobotVariant, Is.EqualTo(RobotVariant.Roller));
            Assert.That(PolicyCatalog.Entries[7].RobotVariant, Is.EqualTo(RobotVariant.Roller));
            Assert.That(PolicyCatalog.Entries.Select(entry => entry.ActionScale), Is.EqualTo(new[]
            {
                1.1f, 1f, 1f, 1f, 1f, 1f, 0.8f, 0.8f, 1f,
            }));
        }

        [Test]
        public void GroundPickCatalogEntryClampsAtPhasePointEight()
        {
            Assert.That(PolicyCatalog.GetBySlot(4).PhaseEnd, Is.EqualTo(0.8f));
        }

        [Test]
        public void RollerCrouchCatalogEntryUsesFiveSecondPeriod()
        {
            Assert.That(PolicyCatalog.GetBySlot(8).PhasePeriodSeconds, Is.EqualTo(5f));
        }

        [Test]
        public void LocomotionCommandUsesForwardLeftYawThenHeadAndBodyLayout()
        {
            var state = new PolicyCommandState();
            state.SelectSlot(1);
            state.SetTwist(0.3f, -0.2f, 0.4f);
            state.SetHead(0.1f, 0.2f, 0.3f, 0.4f);
            state.SetBody(0.05f, -0.1f, 0.15f);
            var command = new float[13];

            state.BuildCommand(10f, command);

            Assert.That(command, Is.EqualTo(new[]
            {
                0.3f, -0.2f, 0.4f,
                0.1f, 0.2f, 0.3f, 0.4f,
                0f, 0f, 0.05f, -0.1f, 0.15f, 0f,
            }));
        }

        [Test]
        public void GroundPickTriggerProducesClampedCosineSinePhase()
        {
            var state = new PolicyCommandState();
            state.SelectSlot(4);
            state.TriggerSkill(2f);
            var command = new float[13];

            state.BuildCommand(3f, command);

            Assert.That(command[0], Is.EqualTo(0f).Within(1e-5f));
            Assert.That(command[1], Is.EqualTo(1f).Within(1e-5f));
            state.BuildCommand(20f, command);
            Assert.That(command[0], Is.EqualTo(Mathf.Cos(Mathf.PI * 1.6f)).Within(1e-5f));
            Assert.That(command[1], Is.EqualTo(Mathf.Sin(Mathf.PI * 1.6f)).Within(1e-5f));
        }

        [Test]
        public void ParameterlessSkillTriggerUsesCurrentFixedTime()
        {
            var state = new PolicyCommandState();
            state.SelectSlot(4);
            var command = new float[13];

            state.TriggerSkill();
            state.BuildCommand(Time.fixedTime, command);

            Assert.That(command[0], Is.EqualTo(1f).Within(1e-5f));
            Assert.That(command[1], Is.Zero.Within(1e-5f));
        }

        [Test]
        public void RollerCrouchUsesFiveSecondPhasePeriod()
        {
            var state = new PolicyCommandState();
            state.SelectSlot(8);
            state.TriggerSkill(2f);
            var command = new float[13];

            state.BuildCommand(3.25f, command);

            Assert.That(command[0], Is.Zero.Within(1e-5f));
            Assert.That(command[1], Is.EqualTo(1f).Within(1e-5f));
        }

        [Test]
        public void SitStandTriggerTogglesThePostureFlag()
        {
            var state = new PolicyCommandState();
            state.SelectSlot(3);
            var command = new float[13];

            state.TriggerSkill(0f);
            state.BuildCommand(0f, command);
            Assert.That(command[0], Is.EqualTo(1f));
            state.TriggerSkill(1f);
            state.BuildCommand(1f, command);
            Assert.That(command[0], Is.Zero);
        }

        [Test]
        public void ResetClearsInputsAndSkillStateWithoutChangingSelectedPolicy()
        {
            var state = new PolicyCommandState();
            state.SelectSlot(3);
            state.SetTwist(0.3f, -0.2f, 0.4f);
            state.SetHead(0.1f, 0.2f, 0.3f, 0.4f);
            state.SetBody(0.05f, -0.1f, 0.15f);
            state.TriggerSkill(2f);

            state.Reset();
            var command = new float[13];
            state.BuildCommand(20f, command);

            Assert.That(state.Selected.Slot, Is.EqualTo(3));
            Assert.That(command, Is.All.Zero);

            state.SelectSlot(1);
            state.BuildCommand(20f, command);
            Assert.That(command, Is.All.Zero,
                "Reset must clear latent locomotion/head/body inputs as well as the active skill.");

            state.SelectSlot(4);
            state.TriggerSkill(2f);
            state.Reset();
            state.BuildCommand(20f, command);
            Assert.That(state.Selected.Slot, Is.EqualTo(4));
            Assert.That(command[0], Is.EqualTo(1f).Within(1e-6f));
            Assert.That(command[1], Is.Zero.Within(1e-6f),
                "A reset phase policy must match its never-triggered initial phase.");
        }

        [TestCase(0)]
        [TestCase(10)]
        public void RejectsPolicySlotsOutsideOneThroughNine(int slot)
        {
            var state = new PolicyCommandState();
            Assert.Throws<System.ArgumentOutOfRangeException>(() => state.SelectSlot(slot));
        }
    }
}
