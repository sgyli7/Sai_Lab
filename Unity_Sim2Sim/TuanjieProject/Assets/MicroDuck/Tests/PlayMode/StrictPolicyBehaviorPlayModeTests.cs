using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class StrictPolicyBehaviorPlayModeTests
    {
        private const float TimestepSeconds = 0.005f;
        private const int PhysicsStepsPerPolicyStep = 4;
        private const float RootLinearSpeedLimit = 3f;
        private const float RootAngularSpeedLimit = 20f;
        private const float MinimumGravityNorm = 0.98f;
        private const float MaximumGravityNorm = 1.02f;
        private const float HardActionLimit = 5f;

        private static readonly Vector3 MouthTipInJawMuJoCo =
            new Vector3(-0.00809334f, 0f, -0.0777383f);

        [UnityTest]
        [Explicit(
            "PhysX calibration diagnostic only; authoritative MicroDuck policy "
            + "acceptance runs against native MuJoCo.")]
        public IEnumerator OfficialPoliciesMeetMuJoCoDerivedSustainedAndCompoundBehaviorContracts()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;

            MicroDuckDemoController controller =
                UnityEngine.Object.FindObjectOfType<MicroDuckDemoController>();
            Assert.That(controller, Is.Not.Null, "Generated MVP scene must contain its controller.");
            controller.enabled = false;

            float originalFixedDeltaTime = Time.fixedDeltaTime;
            SimulationMode originalSimulationMode = Physics.simulationMode;
            var scenarios = new List<ScenarioMetric>();
            try
            {
                Time.fixedDeltaTime = TimestepSeconds;
                Physics.simulationMode = SimulationMode.Script;

                scenarios.Add(RunSafely("walk", controller, RunWalk));
                scenarios.Add(RunSafely("stand", controller, RunStand));
                scenarios.Add(RunSafely("sit-stand-cycle", controller, RunSitStand));
                scenarios.Add(RunSafely("ground-pick-cycle", controller, RunGroundPick));
                scenarios.Add(RunSafely("roller-drive", controller, RunRoller));
                scenarios.Add(RunSafely(
                    "roller-crouch-live-hot-swap",
                    controller,
                    RunRollerCrouchCompound));
                scenarios.Add(RunSafely(
                    "roulade-live-hot-swap",
                    controller,
                    RunRouladeCompound));
            }
            finally
            {
                Physics.simulationMode = originalSimulationMode;
                Time.fixedDeltaTime = originalFixedDeltaTime;
            }

            foreach (ScenarioMetric scenario in scenarios)
            {
                scenario.Seal();
            }

            WriteReport(scenarios);
            string[] failures = scenarios
                .SelectMany(scenario => scenario.checks
                    .Where(check => !check.passed)
                    .Select(check => $"{scenario.scenarioName}/{check.name}: "
                        + $"observed {check.observed:R} {check.unit}, expected "
                        + $"[{check.minimum:R}, {check.maximum:R}] {check.unit}"))
                .ToArray();

            Assert.That(
                failures,
                Is.Empty,
                "MuJoCo-derived behavior contract failures:\n" + string.Join("\n", failures));
        }

        private static ScenarioMetric RunWalk(MicroDuckDemoController controller)
        {
            BehaviorProbe probe = Prepare(controller, "walk", 1);
            controller.SetTwist(0.2f, 0f, 0f);
            probe.BeginStage("commanded-walk");
            probe.Simulate(6f);

            ScenarioMetric metric = probe.Finish();
            metric.AddRange("local forward displacement", metric.localForwardDisplacementMeters,
                0.35f, 0.78f, "m");
            metric.AddAbsoluteMaximum("local lateral displacement",
                metric.localLateralDisplacementMeters, 0.12f, "m");
            metric.AddMinimum("minimum upright", metric.minimumUprightDot, 0.90f, "dot");
            metric.AddMinimum("final upright", metric.finalUprightDot, 0.95f, "dot");
            return metric;
        }

        private static ScenarioMetric RunStand(MicroDuckDemoController controller)
        {
            BehaviorProbe probe = Prepare(controller, "stand", 2);
            probe.BeginStage("stationary-stand");
            probe.Simulate(4f);

            ScenarioMetric metric = probe.Finish();
            metric.AddMaximum("planar drift", metric.planarDisplacementMeters, 0.03f, "m");
            metric.AddMinimum("minimum upright", metric.minimumUprightDot, 0.95f, "dot");
            metric.AddMaximum("height excursion", metric.rootHeightExcursionMeters, 0.025f, "m");
            return metric;
        }

        private static ScenarioMetric RunSitStand(MicroDuckDemoController controller)
        {
            BehaviorProbe probe = Prepare(controller, "sit-stand-cycle", 3);
            float initialHeight = probe.RootPosition.y;

            probe.BeginStage("standing-before-sit");
            probe.Simulate(1f);
            controller.TriggerSkill(probe.CurrentTime);

            probe.BeginStage("sit-and-hold");
            probe.Simulate(3.5f);
            controller.TriggerSkill(probe.CurrentTime);

            probe.BeginStage("stand-recovery");
            probe.Simulate(3.5f);

            ScenarioMetric metric = probe.Finish();
            float heightDrop = initialHeight - metric.minimumRootHeightMeters;
            float lowDwell = probe.LongestDwellBelow(
                initialHeight - 0.04f,
                startSeconds: 1f,
                endSeconds: 4.5f);
            metric.lowPostureDwellSeconds = lowDwell;
            metric.triggerCount = 2;
            metric.AddRange("commanded sitting height drop", heightDrop, 0.045f, 0.085f, "m");
            metric.AddMinimum("continuous low-posture dwell", lowDwell, 1f, "s");
            metric.AddAbsoluteMaximum("recovered height error",
                metric.finalRootHeightMeters - initialHeight, 0.02f, "m");
            metric.AddMinimum("minimum upright through sit-stand cycle",
                metric.minimumUprightDot, 0.90f, "dot");
            metric.AddMinimum("final upright", metric.finalUprightDot, 0.90f, "dot");
            metric.AddMaximum("planar drift", metric.planarDisplacementMeters, 0.10f, "m");
            return metric;
        }

        private static ScenarioMetric RunGroundPick(MicroDuckDemoController controller)
        {
            BehaviorProbe probe = Prepare(controller, "ground-pick-cycle", 4);
            float initialHeight = probe.RootPosition.y;
            controller.TriggerSkill(probe.CurrentTime);
            probe.BeginStage("phase-zero-to-point-eight");
            probe.Simulate(4f);

            ScenarioMetric metric = probe.Finish();
            metric.triggerCount = 1;
            metric.AddRange("minimum mouth-tip height above ground",
                metric.minimumMouthTipHeightMeters, -0.01f, 0.04f, "m");
            metric.AddRange("joint excursion", metric.maxJointExcursionRad, 1.2f, 2.7f, "rad");
            metric.AddMaximum("final joint offset from entry pose",
                metric.finalMaxJointOffsetRad, 0.20f, "rad");
            metric.AddAbsoluteMaximum("recovered height error",
                metric.finalRootHeightMeters - initialHeight, 0.025f, "m");
            metric.AddMinimum("minimum upright through ground-pick cycle",
                metric.minimumUprightDot, 0.75f, "dot");
            metric.AddMinimum("final upright", metric.finalUprightDot, 0.90f, "dot");
            metric.AddMaximum("planar drift", metric.planarDisplacementMeters, 0.10f, "m");
            metric.AddRange("final clamped phase", PhaseFromCommand(controller.LastCommand),
                0.795f, 0.805f, "cycle");
            return metric;
        }

        private static ScenarioMetric RunRoller(MicroDuckDemoController controller)
        {
            BehaviorProbe probe = Prepare(controller, "roller-drive", 7);
            controller.SetTwist(0.3f, 0f, 0f);
            probe.BeginStage("commanded-roller-drive");
            probe.Simulate(6f);

            ScenarioMetric metric = probe.Finish();
            StageMetric drive = metric.FindStage("commanded-roller-drive");
            metric.AddRange("local forward displacement", metric.localForwardDisplacementMeters,
                0.50f, 1.55f, "m");
            metric.AddAbsoluteMaximum("local lateral displacement",
                metric.localLateralDisplacementMeters, 0.20f, "m");
            float directionRatio = metric.planarDisplacementMeters > 1e-6f
                ? metric.localForwardDisplacementMeters / metric.planarDisplacementMeters
                : 0f;
            metric.directionalProgressRatio = directionRatio;
            metric.AddMinimum("forward-to-planar progress ratio", directionRatio, 0.85f, "ratio");
            metric.AddMinimum("minimum upright", metric.minimumUprightDot, 0.90f, "dot");
            metric.AddMinimum("last-second mean forward speed",
                drive.meanForwardSpeedLastSecondMetersPerSecond, 0.10f, "m/s");
            for (int index = 0; index < PolicyContract.RollerPassiveWheelNames.Length; index++)
            {
                metric.AddRange(
                    $"{PolicyContract.RollerPassiveWheelNames[index]} maximum speed",
                    drive.maximumAbsolutePassiveWheelSpeedRadPerSecond[index],
                    5f,
                    40f,
                    "rad/s");
            }

            return metric;
        }

        private static ScenarioMetric RunRollerCrouchCompound(MicroDuckDemoController controller)
        {
            BehaviorProbe probe = Prepare(controller, "roller-crouch-live-hot-swap", 7);
            controller.SetTwist(0.3f, 0f, 0f);

            probe.BeginStage("roller-approach");
            probe.Simulate(3f);
            float approachForwardSpeed = Vector3.Dot(
                controller.ActiveRig.RootBody.velocity,
                probe.InitialForward);
            float[] wheelsBeforeCrouch =
                controller.ActiveRig.ReadPassiveWheelVelocityRadPerSecond();
            float preCrouchHeight = probe.RootPosition.y;

            HotSwapSnapshot toCrouch = probe.CaptureHotSwap();
            bool crouchSelected = controller.HotSwapPolicy(8);
            probe.RecordHotSwap(toCrouch, crouchSelected, "roller-to-crouch");
            if (crouchSelected)
            {
                controller.TriggerSkill(probe.CurrentTime);
                probe.BeginStage("roller-crouch-phase");
                probe.Simulate(3.5f);
                probe.TickNow();
            }

            float observedCrouchPhase = PhaseFromCommand(controller.LastCommand);
            HotSwapSnapshot toRoller = probe.CaptureHotSwap();
            bool rollerRestored = crouchSelected && controller.HotSwapPolicy(7);
            probe.RecordHotSwap(toRoller, rollerRestored, "crouch-to-roller");
            Vector3 recoveryStartPosition = probe.RootPosition;
            if (rollerRestored)
            {
                controller.SetTwist(0.3f, 0f, 0f);
                probe.BeginStage("roller-recovery");
                probe.Simulate(1.5f);
            }

            ScenarioMetric metric = probe.Finish();
            StageMetric crouch = metric.FindStageOrNull("roller-crouch-phase");
            StageMetric recovery = metric.FindStageOrNull("roller-recovery");
            metric.triggerCount = crouchSelected ? 1 : 0;
            metric.AddMinimum("approach forward speed", approachForwardSpeed, 0.10f, "m/s");
            for (int index = 0; index < wheelsBeforeCrouch.Length; index++)
            {
                metric.AddMinimum(
                    $"{PolicyContract.RollerPassiveWheelNames[index]} moving before crouch",
                    Mathf.Abs(wheelsBeforeCrouch[index]),
                    0.10f,
                    "rad/s");
            }

            metric.AddExact("live hot-swap count", metric.hotSwapCount, 2f, "count");
            metric.AddExact("all hot-swaps retained the same rig",
                metric.allHotSwapsUsedSameRig ? 1f : 0f, 1f, "bool");
            metric.AddExact("all hot-swaps preserved live state",
                metric.allHotSwapsPreservedState ? 1f : 0f, 1f, "bool");
            metric.AddRange("crouch phase at hand-back", observedCrouchPhase,
                0.695f, 0.705f, "cycle");

            if (crouch != null)
            {
                metric.AddRange("crouch height drop",
                    preCrouchHeight - crouch.minimumRootHeightMeters,
                    0.045f,
                    0.09f,
                    "m");
                metric.AddMinimum("crouch minimum upright", crouch.minimumUprightDot,
                    0.90f, "dot");
            }
            else
            {
                metric.AddExact("crouch stage completed", 0f, 1f, "bool");
            }

            if (recovery != null)
            {
                metric.AddAbsoluteMaximum("recovery height error",
                    metric.finalRootHeightMeters - preCrouchHeight, 0.03f, "m");
                metric.AddMinimum("recovery final upright", metric.finalUprightDot,
                    0.90f, "dot");
                metric.AddMinimum("recovery local forward displacement",
                    probe.ProjectForward(metric.finalRootPosition - recoveryStartPosition),
                    0.10f,
                    "m");
                metric.AddMinimum("recovery last-second mean forward speed",
                    recovery.meanForwardSpeedLastSecondMetersPerSecond,
                    0.10f,
                    "m/s");
            }
            else
            {
                metric.AddExact("roller recovery stage completed", 0f, 1f, "bool");
            }

            return metric;
        }

        private static ScenarioMetric RunRouladeCompound(MicroDuckDemoController controller)
        {
            BehaviorProbe probe = Prepare(controller, "roulade-live-hot-swap", 2);
            probe.BeginStage("standing-before-roulade");
            probe.Simulate(1f);

            Vector3 rouladeStartPosition = probe.RootPosition;
            HotSwapSnapshot toRoulade = probe.CaptureHotSwap();
            bool rouladeSelected = controller.HotSwapPolicy(9);
            probe.RecordHotSwap(toRoulade, rouladeSelected, "stand-to-roulade");
            if (rouladeSelected)
            {
                controller.TriggerSkill(probe.CurrentTime);
                probe.BeginStage("roulade");
                probe.Simulate(2f);
            }

            HotSwapSnapshot toStand = probe.CaptureHotSwap();
            bool standRestored = rouladeSelected && controller.HotSwapPolicy(2);
            probe.RecordHotSwap(toStand, standRestored, "roulade-to-stand");
            if (standRestored)
            {
                probe.BeginStage("standing-recovery");
                probe.Simulate(2f);
            }

            ScenarioMetric metric = probe.Finish();
            StageMetric roulade = metric.FindStageOrNull("roulade");
            metric.triggerCount = rouladeSelected ? 1 : 0;
            metric.AddExact("live hot-swap count", metric.hotSwapCount, 2f, "count");
            metric.AddExact("all hot-swaps retained the same rig",
                metric.allHotSwapsUsedSameRig ? 1f : 0f, 1f, "bool");
            metric.AddExact("all hot-swaps preserved live state",
                metric.allHotSwapsPreservedState ? 1f : 0f, 1f, "bool");
            if (roulade != null)
            {
                metric.AddRange("unwrapped local pitch rotation",
                    Mathf.Abs(roulade.cumulativeLocalPitchRad), 5.2f, 7.3f, "rad");
                metric.AddMaximum("inverted upright dot", roulade.minimumUprightDot,
                    -0.80f, "dot");
            }
            else
            {
                metric.AddExact("roulade stage completed", 0f, 1f, "bool");
            }

            metric.AddRange("local forward displacement through roll and recovery",
                probe.ProjectForward(metric.finalRootPosition - rouladeStartPosition),
                0.25f,
                0.90f,
                "m");
            metric.AddMinimum("final upright after standing recovery",
                metric.finalUprightDot, 0.90f, "dot");
            return metric;
        }

        private static BehaviorProbe Prepare(
            MicroDuckDemoController controller,
            string scenarioName,
            int slot)
        {
            controller.SetTwist(0f, 0f, 0f);
            controller.SetHead(0f, 0f, 0f, 0f);
            controller.SetBody(0f, 0f, 0f);
            if (!controller.SelectPolicy(slot))
            {
                throw new InvalidOperationException(
                    $"Could not select policy slot {slot}: {controller.Fault}");
            }

            Physics.SyncTransforms();
            Physics.Simulate(1e-6f);
            return new BehaviorProbe(controller, scenarioName);
        }

        private static ScenarioMetric RunSafely(
            string scenarioName,
            MicroDuckDemoController controller,
            Func<MicroDuckDemoController, ScenarioMetric> run)
        {
            try
            {
                return run(controller);
            }
            catch (Exception error)
            {
                var metric = new ScenarioMetric
                {
                    scenarioName = scenarioName,
                    finite = false,
                    allPolicyTicksHealthy = false,
                    firstFault = BehaviorFaultFrame.FromException(controller, error),
                };
                metric.AddExact("scenario completed without exception", 0f, 1f, "bool");
                return metric;
            }
        }

        private static float PhaseFromCommand(float[] command)
        {
            if (command == null || command.Length < 2 || !IsFinite(command[0]) || !IsFinite(command[1]))
            {
                return -1f;
            }

            float phase = Mathf.Atan2(command[1], command[0]) / (2f * Mathf.PI);
            return phase < 0f ? phase + 1f : phase;
        }

        private static float ActionLimitForSlot(int slot)
        {
            // Each envelope is 1.75 times the largest absolute action in the checked-in
            // official MuJoCo rollout, rounded upward.  The global fail-closed ceiling is 5.
            switch (slot)
            {
                case 1:
                    return 1.18f;
                case 2:
                    return 0.49f;
                case 3:
                    return 2.84f;
                case 4:
                    return 3.00f;
                case 5:
                    return 3.17f;
                case 6:
                    return 3.56f;
                case 7:
                    return 1.60f;
                case 8:
                    return 5.00f;
                case 9:
                    return 4.86f;
                default:
                    throw new ArgumentOutOfRangeException(nameof(slot));
            }
        }

        private static void WriteReport(List<ScenarioMetric> scenarios)
        {
            string repositoryRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "..", ".."));
            string outputPath = Path.Combine(
                repositoryRoot,
                "artifacts",
                "mvp",
                "tuanjie-policy-behavior.json");
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
            var report = new BehaviorReport
            {
                schemaVersion = 2,
                contractSource = "official MuJoCo rollout metrics and 1.75x action envelopes",
                timestepSeconds = TimestepSeconds,
                physicsStepsPerPolicyStep = PhysicsStepsPerPolicyStep,
                passed = scenarios.All(scenario => scenario.passed),
                scenarios = scenarios.ToArray(),
            };
            string json = JsonUtility.ToJson(report, prettyPrint: true);
            File.WriteAllText(outputPath, json + "\n");
            TestContext.Progress.WriteLine(json);
        }

        private static bool AllFinite(float[] values)
        {
            if (values == null)
            {
                return false;
            }

            foreach (float value in values)
            {
                if (!IsFinite(value))
                {
                    return false;
                }
            }

            return true;
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }

        private static float MaximumAbsolute(float[] values)
        {
            float maximum = 0f;
            if (values == null)
            {
                return maximum;
            }

            foreach (float value in values)
            {
                if (IsFinite(value))
                {
                    maximum = Mathf.Max(maximum, Mathf.Abs(value));
                }
            }

            return maximum;
        }

        private static float MaximumAbsoluteDifference(float[] left, float[] right)
        {
            if (left == null || right == null || left.Length != right.Length)
            {
                return float.PositiveInfinity;
            }

            float maximum = 0f;
            for (int index = 0; index < left.Length; index++)
            {
                maximum = Mathf.Max(maximum, Mathf.Abs(left[index] - right[index]));
            }

            return maximum;
        }

        private sealed class BehaviorProbe
        {
            private readonly MicroDuckDemoController controller;
            private readonly ScenarioMetric metric;
            private readonly ArticulationBody root;
            private readonly Vector3 initialPosition;
            private readonly float[] initialJointPosition;
            private readonly Transform mouthJaw;
            private readonly List<FrameSample> frames = new List<FrameSample>();
            private readonly List<StageMetric> stages = new List<StageMetric>();
            private readonly Dictionary<int, PolicyActionMetric> actionMetrics =
                new Dictionary<int, PolicyActionMetric>();
            private Quaternion previousRootRotation;
            private StageMetric activeStage;
            private int physicsStep;
            private float cumulativeLocalPitchRad;

            public BehaviorProbe(MicroDuckDemoController controller, string scenarioName)
            {
                this.controller = controller;
                root = controller.ActiveRig.RootBody;
                initialPosition = root.transform.position;
                previousRootRotation = root.transform.rotation;
                InitialForward = HorizontalUnit(root.transform.forward, Vector3.forward);
                InitialRight = HorizontalUnit(root.transform.right, Vector3.right);
                initialJointPosition = ReadJointPositions(controller.ActiveRig);
                mouthJaw = FindBodyTransform(controller.ActiveRig, "jaw_soft");
                metric = new ScenarioMetric
                {
                    scenarioName = scenarioName,
                    finite = true,
                    allPolicyTicksHealthy = true,
                    allHotSwapsUsedSameRig = true,
                    allHotSwapsPreservedState = true,
                    initialRootPosition = initialPosition,
                    initialRootHeightMeters = initialPosition.y,
                    minimumRootHeightMeters = initialPosition.y,
                    maximumRootHeightMeters = initialPosition.y,
                    minimumUprightDot = Vector3.Dot(root.transform.up, Vector3.up),
                    minimumGravityNorm = float.PositiveInfinity,
                    maximumGravityNorm = float.NegativeInfinity,
                    minimumMouthTipHeightMeters = MouthTipHeight(),
                    maximumAbsolutePassiveWheelSpeedRadPerSecond = new float[4],
                };
                AddPolicyToSequence();
                if (mouthJaw == null)
                {
                    CaptureFault("jaw_soft is missing, so the canonical mouth_tip cannot be measured");
                }

                ObservePhysicsFrame();
            }

            public float CurrentTime { get; private set; }
            public Vector3 InitialForward { get; }
            public Vector3 InitialRight { get; }
            public Vector3 RootPosition => root.transform.position;

            public void BeginStage(string name)
            {
                EndActiveStage();
                AddPolicyToSequence();
                activeStage = new StageMetric(
                    name,
                    controller.ActivePolicySlot,
                    controller.ActivePolicyName,
                    CurrentTime,
                    RootPosition,
                    root.transform.rotation,
                    cumulativeLocalPitchRad,
                    MouthTipHeight());
            }

            public void Simulate(float seconds)
            {
                int steps = Mathf.RoundToInt(seconds / TimestepSeconds);
                if (Mathf.Abs(steps * TimestepSeconds - seconds) > 1e-6f)
                {
                    throw new ArgumentException(
                        $"Duration {seconds:R} is not an integral number of physics steps.",
                        nameof(seconds));
                }

                for (int index = 0; index < steps; index++)
                {
                    if (physicsStep % PhysicsStepsPerPolicyStep == 0)
                    {
                        TickNow();
                    }

                    Physics.Simulate(TimestepSeconds);
                    physicsStep++;
                    CurrentTime = physicsStep * TimestepSeconds;
                    ObservePhysicsFrame();
                }
            }

            public void TickNow()
            {
                bool ticked = controller.TickOnce(CurrentTime);
                metric.policyTickCount++;
                metric.allPolicyTicksHealthy &= ticked && controller.IsHealthy;
                if (!ticked || !controller.IsHealthy)
                {
                    CaptureFault("policy tick failed: " + controller.Fault);
                }

                float[] observation = controller.LastObservation;
                float[] action = controller.LastRawAction;
                float[] targets = controller.LastTargets;
                float[] joints = controller.LastJointPositionRad;
                float[] jointVelocities = controller.LastJointVelocityRadPerSecond;
                bool tickFinite = AllFinite(observation)
                    && AllFinite(action)
                    && AllFinite(targets)
                    && AllFinite(joints)
                    && AllFinite(jointVelocities);
                metric.finite &= tickFinite;
                if (!tickFinite)
                {
                    CaptureFault("policy tick produced a non-finite tensor");
                }

                if (observation.Length == PolicyContract.ObservationCount)
                {
                    float gravityNorm = new Vector3(
                        observation[3], observation[4], observation[5]).magnitude;
                    metric.gravityObservationCount++;
                    if (IsFinite(gravityNorm))
                    {
                        metric.minimumGravityNorm = Mathf.Min(metric.minimumGravityNorm, gravityNorm);
                        metric.maximumGravityNorm = Mathf.Max(metric.maximumGravityNorm, gravityNorm);
                    }

                    if (!IsFinite(gravityNorm)
                        || gravityNorm < MinimumGravityNorm
                        || gravityNorm > MaximumGravityNorm)
                    {
                        CaptureFault($"projected-gravity norm {gravityNorm:R} left the unit envelope");
                    }
                }
                else
                {
                    CaptureFault($"observation length was {observation.Length}, expected 61");
                }

                float maxAction = MaximumAbsolute(action);
                metric.maximumAbsoluteRawAction = Mathf.Max(metric.maximumAbsoluteRawAction, maxAction);
                int slot = controller.ActivePolicySlot;
                float actionLimit = ActionLimitForSlot(slot);
                if (!actionMetrics.TryGetValue(slot, out PolicyActionMetric policyAction))
                {
                    policyAction = new PolicyActionMetric
                    {
                        slot = slot,
                        policyName = controller.ActivePolicyName,
                        maximumAllowedAbsoluteAction = actionLimit,
                    };
                    actionMetrics.Add(slot, policyAction);
                }

                policyAction.sampleCount++;
                policyAction.maximumObservedAbsoluteAction = Mathf.Max(
                    policyAction.maximumObservedAbsoluteAction,
                    maxAction);
                if (maxAction > actionLimit || maxAction > HardActionLimit)
                {
                    CaptureFault(
                        $"raw action {maxAction:R} exceeded slot {slot} envelope {actionLimit:R}");
                }

                for (int index = 0; index < joints.Length && index < initialJointPosition.Length; index++)
                {
                    metric.maxJointExcursionRad = Mathf.Max(
                        metric.maxJointExcursionRad,
                        Mathf.Abs(joints[index] - initialJointPosition[index]));
                    metric.maxTargetExcursionRad = Mathf.Max(
                        metric.maxTargetExcursionRad,
                        Mathf.Abs(targets[index] - initialJointPosition[index]));
                }

                activeStage?.ObservePolicyTick(maxAction);
            }

            public HotSwapSnapshot CaptureHotSwap()
            {
                return new HotSwapSnapshot
                {
                    rig = controller.ActiveRig,
                    rootPosition = root.transform.position,
                    rootRotation = root.transform.rotation,
                    rootVelocity = root.velocity,
                    rootAngularVelocity = root.angularVelocity,
                    reducedJointPosition = ReadReducedState(root, velocities: false),
                    reducedJointVelocity = ReadReducedState(root, velocities: true),
                    passiveWheelVelocity = controller.ActiveRig.ReadPassiveWheelVelocityRadPerSecond(),
                    rawAction = controller.LastRawAction,
                    targets = controller.LastTargets,
                    policyTicks = controller.PolicyTicks,
                };
            }

            public void RecordHotSwap(HotSwapSnapshot before, bool succeeded, string label)
            {
                metric.hotSwapCount++;
                if (!succeeded)
                {
                    metric.allHotSwapsUsedSameRig = false;
                    metric.allHotSwapsPreservedState = false;
                    CaptureFault(label + " failed: " + controller.Fault);
                    return;
                }

                bool sameRig = ReferenceEquals(before.rig, controller.ActiveRig);
                float positionJump = Vector3.Distance(before.rootPosition, root.transform.position);
                float rotationJump = Quaternion.Angle(before.rootRotation, root.transform.rotation);
                float velocityJump = Vector3.Distance(before.rootVelocity, root.velocity);
                float angularVelocityJump = Vector3.Distance(
                    before.rootAngularVelocity,
                    root.angularVelocity);
                float jointPositionJump = MaximumAbsoluteDifference(
                    before.reducedJointPosition,
                    ReadReducedState(root, velocities: false));
                float jointVelocityJump = MaximumAbsoluteDifference(
                    before.reducedJointVelocity,
                    ReadReducedState(root, velocities: true));
                float wheelVelocityJump = MaximumAbsoluteDifference(
                    before.passiveWheelVelocity,
                    controller.ActiveRig.ReadPassiveWheelVelocityRadPerSecond());
                float actionJump = MaximumAbsoluteDifference(before.rawAction, controller.LastRawAction);
                float targetJump = MaximumAbsoluteDifference(before.targets, controller.LastTargets);
                bool ticksPreserved = before.policyTicks == controller.PolicyTicks;

                metric.allHotSwapsUsedSameRig &= sameRig;
                metric.maximumHotSwapRootPositionJumpMeters = Mathf.Max(
                    metric.maximumHotSwapRootPositionJumpMeters,
                    positionJump);
                metric.maximumHotSwapRootRotationJumpDegrees = Mathf.Max(
                    metric.maximumHotSwapRootRotationJumpDegrees,
                    rotationJump);
                metric.maximumHotSwapRootVelocityJumpMetersPerSecond = Mathf.Max(
                    metric.maximumHotSwapRootVelocityJumpMetersPerSecond,
                    velocityJump);
                metric.maximumHotSwapRootAngularVelocityJumpRadPerSecond = Mathf.Max(
                    metric.maximumHotSwapRootAngularVelocityJumpRadPerSecond,
                    angularVelocityJump);
                metric.maximumHotSwapJointPositionJumpRad = Mathf.Max(
                    metric.maximumHotSwapJointPositionJumpRad,
                    jointPositionJump);
                metric.maximumHotSwapJointVelocityJumpRadPerSecond = Mathf.Max(
                    metric.maximumHotSwapJointVelocityJumpRadPerSecond,
                    jointVelocityJump);
                metric.maximumHotSwapWheelVelocityJumpRadPerSecond = Mathf.Max(
                    metric.maximumHotSwapWheelVelocityJumpRadPerSecond,
                    wheelVelocityJump);
                metric.maximumHotSwapActionJump = Mathf.Max(
                    metric.maximumHotSwapActionJump,
                    actionJump);
                metric.maximumHotSwapTargetJumpRad = Mathf.Max(
                    metric.maximumHotSwapTargetJumpRad,
                    targetJump);

                bool preserved = sameRig
                    && positionJump <= 1e-5f
                    && rotationJump <= 1e-4f
                    && velocityJump <= 1e-6f
                    && angularVelocityJump <= 1e-6f
                    && jointPositionJump <= 1e-6f
                    && jointVelocityJump <= 1e-6f
                    && wheelVelocityJump <= 1e-6f
                    && actionJump <= 1e-6f
                    && targetJump <= 1e-6f
                    && ticksPreserved;
                metric.allHotSwapsPreservedState &= preserved;
                if (!preserved)
                {
                    CaptureFault(label + " introduced a live-state discontinuity");
                }

                AddPolicyToSequence();
            }

            public float LongestDwellBelow(float height, float startSeconds, float endSeconds)
            {
                float longest = 0f;
                float current = 0f;
                foreach (FrameSample frame in frames)
                {
                    if (frame.timeSeconds < startSeconds || frame.timeSeconds > endSeconds)
                    {
                        continue;
                    }

                    if (frame.rootPosition.y <= height)
                    {
                        current += TimestepSeconds;
                        longest = Mathf.Max(longest, current);
                    }
                    else
                    {
                        current = 0f;
                    }
                }

                return longest;
            }

            public float ProjectForward(Vector3 displacement)
            {
                return Vector3.Dot(displacement, InitialForward);
            }

            public ScenarioMetric Finish()
            {
                EndActiveStage();
                metric.simulatedSeconds = CurrentTime;
                metric.physicsStepCount = physicsStep;
                metric.finalRootPosition = root.transform.position;
                metric.finalRootHeightMeters = root.transform.position.y;
                metric.finalUprightDot = Vector3.Dot(root.transform.up, Vector3.up);
                Vector3 displacement = root.transform.position - initialPosition;
                metric.localForwardDisplacementMeters = Vector3.Dot(displacement, InitialForward);
                metric.localLateralDisplacementMeters = Vector3.Dot(displacement, InitialRight);
                metric.planarDisplacementMeters = Vector3.ProjectOnPlane(
                    displacement,
                    Vector3.up).magnitude;
                metric.rootHeightExcursionMeters =
                    metric.maximumRootHeightMeters - metric.minimumRootHeightMeters;
                metric.cumulativeLocalPitchRad = cumulativeLocalPitchRad;
                metric.finalMaxJointOffsetRad = MaximumAbsoluteDifference(
                    initialJointPosition,
                    ReadJointPositions(controller.ActiveRig));
                metric.finalPassiveWheelSpeedRadPerSecond =
                    controller.ActiveRig.ReadPassiveWheelVelocityRadPerSecond();
                metric.policySequence = metric.policySequenceBuilder.ToArray();
                metric.policyActions = actionMetrics.Values.OrderBy(value => value.slot).ToArray();
                metric.stages = stages.ToArray();

                if (metric.gravityObservationCount == 0)
                {
                    metric.minimumGravityNorm = 0f;
                    metric.maximumGravityNorm = 0f;
                }

                metric.AddExact("all policy ticks healthy",
                    metric.allPolicyTicksHealthy ? 1f : 0f, 1f, "bool");
                metric.AddExact("all tensors and rigid-body state finite",
                    metric.finite ? 1f : 0f, 1f, "bool");
                metric.AddRange("projected-gravity norm",
                    metric.minimumGravityNorm,
                    MinimumGravityNorm,
                    MaximumGravityNorm,
                    "norm");
                metric.AddRange("projected-gravity maximum norm",
                    metric.maximumGravityNorm,
                    MinimumGravityNorm,
                    MaximumGravityNorm,
                    "norm");
                metric.AddMaximum("maximum root linear speed",
                    metric.maximumRootLinearSpeedMetersPerSecond,
                    RootLinearSpeedLimit,
                    "m/s");
                metric.AddMaximum("maximum root angular speed",
                    metric.maximumRootAngularSpeedRadPerSecond,
                    RootAngularSpeedLimit,
                    "rad/s");
                metric.AddMaximum("hard raw-action safety ceiling",
                    metric.maximumAbsoluteRawAction,
                    HardActionLimit,
                    "action");
                foreach (PolicyActionMetric action in metric.policyActions)
                {
                    metric.AddMaximum(
                        $"slot {action.slot} MuJoCo-derived raw-action envelope",
                        action.maximumObservedAbsoluteAction,
                        action.maximumAllowedAbsoluteAction,
                        "action");
                    metric.AddMinimum(
                        $"slot {action.slot} action sample count",
                        action.sampleCount,
                        1f,
                        "count");
                }

                return metric;
            }

            private void ObservePhysicsFrame()
            {
                Vector3 position = root.transform.position;
                Quaternion rotation = root.transform.rotation;
                Vector3 linearVelocity = root.velocity;
                Vector3 angularVelocity = root.angularVelocity;
                float upright = Vector3.Dot(root.transform.up, Vector3.up);
                float mouthTipHeight = MouthTipHeight();
                float[] wheels;
                try
                {
                    wheels = controller.ActiveRig.ReadPassiveWheelVelocityRadPerSecond();
                }
                catch (Exception error)
                {
                    wheels = Array.Empty<float>();
                    CaptureFault("could not read passive-wheel state: " + error.Message);
                }

                bool stateFinite = IsFinite(position.x)
                    && IsFinite(position.y)
                    && IsFinite(position.z)
                    && IsFinite(rotation.x)
                    && IsFinite(rotation.y)
                    && IsFinite(rotation.z)
                    && IsFinite(rotation.w)
                    && IsFinite(linearVelocity.x)
                    && IsFinite(linearVelocity.y)
                    && IsFinite(linearVelocity.z)
                    && IsFinite(angularVelocity.x)
                    && IsFinite(angularVelocity.y)
                    && IsFinite(angularVelocity.z)
                    && IsFinite(upright)
                    && IsFinite(mouthTipHeight)
                    && AllFinite(wheels);
                metric.finite &= stateFinite;
                if (!stateFinite)
                {
                    CaptureFault("physics frame contained a non-finite value");
                }

                float linearSpeed = linearVelocity.magnitude;
                float angularSpeed = angularVelocity.magnitude;
                if (IsFinite(linearSpeed))
                {
                    metric.maximumRootLinearSpeedMetersPerSecond = Mathf.Max(
                        metric.maximumRootLinearSpeedMetersPerSecond,
                        linearSpeed);
                    if (linearSpeed >= RootLinearSpeedLimit)
                    {
                        CaptureFault($"root linear speed reached {linearSpeed:R} m/s");
                    }
                }

                if (IsFinite(angularSpeed))
                {
                    metric.maximumRootAngularSpeedRadPerSecond = Mathf.Max(
                        metric.maximumRootAngularSpeedRadPerSecond,
                        angularSpeed);
                    if (angularSpeed >= RootAngularSpeedLimit)
                    {
                        CaptureFault($"root angular speed reached {angularSpeed:R} rad/s");
                    }
                }

                metric.minimumRootHeightMeters = Mathf.Min(metric.minimumRootHeightMeters, position.y);
                metric.maximumRootHeightMeters = Mathf.Max(metric.maximumRootHeightMeters, position.y);
                metric.minimumUprightDot = Mathf.Min(metric.minimumUprightDot, upright);
                metric.minimumMouthTipHeightMeters = Mathf.Min(
                    metric.minimumMouthTipHeightMeters,
                    mouthTipHeight);
                for (int index = 0;
                    index < wheels.Length
                    && index < metric.maximumAbsolutePassiveWheelSpeedRadPerSecond.Length;
                    index++)
                {
                    metric.maximumAbsolutePassiveWheelSpeedRadPerSecond[index] = Mathf.Max(
                        metric.maximumAbsolutePassiveWheelSpeedRadPerSecond[index],
                        Mathf.Abs(wheels[index]));
                }

                cumulativeLocalPitchRad += SignedLocalPitchIncrement(
                    previousRootRotation,
                    rotation);
                previousRootRotation = rotation;
                float forwardSpeed = Vector3.Dot(linearVelocity, InitialForward);
                var frame = new FrameSample(CurrentTime, position);
                frames.Add(frame);
                activeStage?.ObservePhysicsFrame(
                    CurrentTime,
                    position,
                    rotation,
                    upright,
                    linearSpeed,
                    angularSpeed,
                    forwardSpeed,
                    wheels,
                    mouthTipHeight,
                    cumulativeLocalPitchRad);
            }

            private void EndActiveStage()
            {
                if (activeStage == null)
                {
                    return;
                }

                activeStage.Complete(
                    CurrentTime,
                    root.transform.position,
                    Vector3.Dot(root.transform.up, Vector3.up),
                    cumulativeLocalPitchRad,
                    InitialForward,
                    InitialRight);
                stages.Add(activeStage);
                activeStage = null;
            }

            private void AddPolicyToSequence()
            {
                string value = $"{controller.ActivePolicySlot}:{controller.ActivePolicyName}";
                if (metric.policySequenceBuilder.Count == 0
                    || metric.policySequenceBuilder[metric.policySequenceBuilder.Count - 1] != value)
                {
                    metric.policySequenceBuilder.Add(value);
                }
            }

            private void CaptureFault(string reason)
            {
                if (metric.firstFault != null)
                {
                    return;
                }

                metric.firstFault = BehaviorFaultFrame.Capture(
                    controller,
                    activeStage?.name ?? "setup",
                    physicsStep,
                    CurrentTime,
                    reason);
            }

            private float MouthTipHeight()
            {
                if (mouthJaw == null)
                {
                    return root.transform.position.y;
                }

                Vector3 local = CoordinateBasis.MuJoCoToTuanjie(MouthTipInJawMuJoCo);
                return mouthJaw.TransformPoint(local).y;
            }

            private static Vector3 HorizontalUnit(Vector3 direction, Vector3 fallback)
            {
                Vector3 horizontal = Vector3.ProjectOnPlane(direction, Vector3.up);
                return horizontal.sqrMagnitude > 1e-8f ? horizontal.normalized : fallback;
            }

            private static float SignedLocalPitchIncrement(Quaternion previous, Quaternion current)
            {
                Quaternion delta = Quaternion.Inverse(previous) * current;
                if (delta.w < 0f)
                {
                    delta = new Quaternion(-delta.x, -delta.y, -delta.z, -delta.w);
                }

                Vector3 vector = new Vector3(delta.x, delta.y, delta.z);
                float magnitude = vector.magnitude;
                if (magnitude < 1e-8f)
                {
                    return 0f;
                }

                float angle = 2f * Mathf.Atan2(magnitude, Mathf.Clamp(delta.w, -1f, 1f));
                return angle * vector.x / magnitude;
            }

            private static float[] ReadJointPositions(MicroDuckRig rig)
            {
                var position = new float[PolicyContract.ActionCount];
                var velocity = new float[PolicyContract.ActionCount];
                rig.ReadPolicyState(position, velocity, out _, out _);
                return position;
            }

            private static float[] ReadReducedState(ArticulationBody root, bool velocities)
            {
                var values = new List<float>();
                if (velocities)
                {
                    root.GetJointVelocities(values);
                }
                else
                {
                    root.GetJointPositions(values);
                }

                return values.ToArray();
            }

            private static Transform FindBodyTransform(MicroDuckRig rig, string bodyName)
            {
                foreach (ArticulationBody body in
                    rig.GetComponentsInChildren<ArticulationBody>(includeInactive: true))
                {
                    if (body.name == bodyName)
                    {
                        return body.transform;
                    }
                }

                return null;
            }
        }

        private sealed class HotSwapSnapshot
        {
            public MicroDuckRig rig;
            public Vector3 rootPosition;
            public Quaternion rootRotation;
            public Vector3 rootVelocity;
            public Vector3 rootAngularVelocity;
            public float[] reducedJointPosition;
            public float[] reducedJointVelocity;
            public float[] passiveWheelVelocity;
            public float[] rawAction;
            public float[] targets;
            public int policyTicks;
        }

        private sealed class FrameSample
        {
            public FrameSample(float timeSeconds, Vector3 rootPosition)
            {
                this.timeSeconds = timeSeconds;
                this.rootPosition = rootPosition;
            }

            public readonly float timeSeconds;
            public readonly Vector3 rootPosition;
        }

        [Serializable]
        private sealed class BehaviorReport
        {
            public int schemaVersion;
            public string contractSource;
            public float timestepSeconds;
            public int physicsStepsPerPolicyStep;
            public bool passed;
            public ScenarioMetric[] scenarios;
        }

        [Serializable]
        private sealed class ScenarioMetric
        {
            public string scenarioName;
            public string[] policySequence = Array.Empty<string>();
            public float simulatedSeconds;
            public int physicsStepCount;
            public int policyTickCount;
            public int triggerCount;
            public bool passed;
            public bool allPolicyTicksHealthy;
            public bool finite;
            public Vector3 initialRootPosition;
            public Vector3 finalRootPosition;
            public float initialRootHeightMeters;
            public float finalRootHeightMeters;
            public float minimumRootHeightMeters;
            public float maximumRootHeightMeters;
            public float rootHeightExcursionMeters;
            public float localForwardDisplacementMeters;
            public float localLateralDisplacementMeters;
            public float planarDisplacementMeters;
            public float directionalProgressRatio;
            public float minimumUprightDot;
            public float finalUprightDot;
            public float cumulativeLocalPitchRad;
            public float lowPostureDwellSeconds;
            public float minimumMouthTipHeightMeters;
            public float maxTargetExcursionRad;
            public float maxJointExcursionRad;
            public float finalMaxJointOffsetRad;
            public float maximumRootLinearSpeedMetersPerSecond;
            public float maximumRootAngularSpeedRadPerSecond;
            public int gravityObservationCount;
            public float minimumGravityNorm;
            public float maximumGravityNorm;
            public float maximumAbsoluteRawAction;
            public float[] maximumAbsolutePassiveWheelSpeedRadPerSecond = Array.Empty<float>();
            public float[] finalPassiveWheelSpeedRadPerSecond = Array.Empty<float>();
            public int hotSwapCount;
            public bool allHotSwapsUsedSameRig;
            public bool allHotSwapsPreservedState;
            public float maximumHotSwapRootPositionJumpMeters;
            public float maximumHotSwapRootRotationJumpDegrees;
            public float maximumHotSwapRootVelocityJumpMetersPerSecond;
            public float maximumHotSwapRootAngularVelocityJumpRadPerSecond;
            public float maximumHotSwapJointPositionJumpRad;
            public float maximumHotSwapJointVelocityJumpRadPerSecond;
            public float maximumHotSwapWheelVelocityJumpRadPerSecond;
            public float maximumHotSwapActionJump;
            public float maximumHotSwapTargetJumpRad;
            public StageMetric[] stages = Array.Empty<StageMetric>();
            public PolicyActionMetric[] policyActions = Array.Empty<PolicyActionMetric>();
            public BehaviorCheck[] checks = Array.Empty<BehaviorCheck>();
            public BehaviorCheck firstFailedCheck;
            public BehaviorFaultFrame firstFault;

            [NonSerialized] public readonly List<string> policySequenceBuilder = new List<string>();
            [NonSerialized] private readonly List<BehaviorCheck> checkBuilder =
                new List<BehaviorCheck>();

            public void AddRange(string name, float observed, float minimum, float maximum, string unit)
            {
                checkBuilder.Add(new BehaviorCheck(name, observed, minimum, maximum, unit));
            }

            public void AddMinimum(string name, float observed, float minimum, string unit)
            {
                AddRange(name, observed, minimum, float.MaxValue, unit);
            }

            public void AddMaximum(string name, float observed, float maximum, string unit)
            {
                AddRange(name, observed, -float.MaxValue, maximum, unit);
            }

            public void AddAbsoluteMaximum(string name, float observed, float maximum, string unit)
            {
                AddRange(name, Mathf.Abs(observed), 0f, maximum, unit);
            }

            public void AddExact(string name, float observed, float expected, string unit)
            {
                AddRange(name, observed, expected, expected, unit);
            }

            public StageMetric FindStage(string name)
            {
                StageMetric found = FindStageOrNull(name);
                if (found == null)
                {
                    throw new InvalidOperationException($"Behavior stage '{name}' was not recorded.");
                }

                return found;
            }

            public StageMetric FindStageOrNull(string name)
            {
                return stages.FirstOrDefault(stage => stage.name == name);
            }

            public void Seal()
            {
                checks = checkBuilder.ToArray();
                firstFailedCheck = checks.FirstOrDefault(check => !check.passed);
                passed = firstFailedCheck == null;
            }
        }

        [Serializable]
        private sealed class StageMetric
        {
            public string name;
            public int policySlot;
            public string policyName;
            public float startSeconds;
            public float endSeconds;
            public float durationSeconds;
            public Vector3 initialRootPosition;
            public Vector3 finalRootPosition;
            public float initialRootHeightMeters;
            public float finalRootHeightMeters;
            public float minimumRootHeightMeters;
            public float maximumRootHeightMeters;
            public float heightExcursionMeters;
            public float heightDropFromStartMeters;
            public float localForwardDisplacementMeters;
            public float localLateralDisplacementMeters;
            public float planarDisplacementMeters;
            public float minimumUprightDot;
            public float finalUprightDot;
            public float maximumRootLinearSpeedMetersPerSecond;
            public float maximumRootAngularSpeedRadPerSecond;
            public float maximumAbsoluteRawAction;
            public float cumulativeLocalPitchRad;
            public float minimumMouthTipHeightMeters;
            public float meanForwardSpeedLastSecondMetersPerSecond;
            public float[] maximumAbsolutePassiveWheelSpeedRadPerSecond = new float[4];
            public float[] finalPassiveWheelSpeedRadPerSecond = Array.Empty<float>();

            [NonSerialized] private readonly float initialCumulativePitch;
            [NonSerialized] private readonly List<TimedSpeed> forwardSpeeds =
                new List<TimedSpeed>();
            [NonSerialized] private float[] lastPassiveWheelSpeedRadPerSecond =
                Array.Empty<float>();

            public StageMetric(
                string name,
                int policySlot,
                string policyName,
                float startSeconds,
                Vector3 initialRootPosition,
                Quaternion initialRootRotation,
                float initialCumulativePitch,
                float initialMouthTipHeight)
            {
                this.name = name;
                this.policySlot = policySlot;
                this.policyName = policyName;
                this.startSeconds = startSeconds;
                this.initialRootPosition = initialRootPosition;
                this.initialRootHeightMeters = initialRootPosition.y;
                this.initialCumulativePitch = initialCumulativePitch;
                minimumRootHeightMeters = initialRootPosition.y;
                maximumRootHeightMeters = initialRootPosition.y;
                minimumUprightDot = Vector3.Dot(initialRootRotation * Vector3.up, Vector3.up);
                minimumMouthTipHeightMeters = initialMouthTipHeight;
            }

            public void ObservePolicyTick(float maximumAction)
            {
                maximumAbsoluteRawAction = Mathf.Max(maximumAbsoluteRawAction, maximumAction);
            }

            public void ObservePhysicsFrame(
                float timeSeconds,
                Vector3 position,
                Quaternion rotation,
                float upright,
                float linearSpeed,
                float angularSpeed,
                float forwardSpeed,
                float[] wheels,
                float mouthTipHeight,
                float cumulativePitch)
            {
                minimumRootHeightMeters = Mathf.Min(minimumRootHeightMeters, position.y);
                maximumRootHeightMeters = Mathf.Max(maximumRootHeightMeters, position.y);
                minimumUprightDot = Mathf.Min(minimumUprightDot, upright);
                maximumRootLinearSpeedMetersPerSecond = Mathf.Max(
                    maximumRootLinearSpeedMetersPerSecond,
                    linearSpeed);
                maximumRootAngularSpeedRadPerSecond = Mathf.Max(
                    maximumRootAngularSpeedRadPerSecond,
                    angularSpeed);
                minimumMouthTipHeightMeters = Mathf.Min(
                    minimumMouthTipHeightMeters,
                    mouthTipHeight);
                for (int index = 0;
                    index < wheels.Length
                    && index < maximumAbsolutePassiveWheelSpeedRadPerSecond.Length;
                    index++)
                {
                    maximumAbsolutePassiveWheelSpeedRadPerSecond[index] = Mathf.Max(
                        maximumAbsolutePassiveWheelSpeedRadPerSecond[index],
                        Mathf.Abs(wheels[index]));
                }

                lastPassiveWheelSpeedRadPerSecond = (float[])wheels.Clone();
                forwardSpeeds.Add(new TimedSpeed(timeSeconds, forwardSpeed));
                cumulativeLocalPitchRad = cumulativePitch - initialCumulativePitch;
            }

            public void Complete(
                float endSeconds,
                Vector3 finalPosition,
                float finalUpright,
                float cumulativePitch,
                Vector3 forward,
                Vector3 right)
            {
                this.endSeconds = endSeconds;
                durationSeconds = endSeconds - startSeconds;
                finalRootPosition = finalPosition;
                finalRootHeightMeters = finalPosition.y;
                heightExcursionMeters = maximumRootHeightMeters - minimumRootHeightMeters;
                heightDropFromStartMeters = initialRootHeightMeters - minimumRootHeightMeters;
                Vector3 displacement = finalPosition - initialRootPosition;
                localForwardDisplacementMeters = Vector3.Dot(displacement, forward);
                localLateralDisplacementMeters = Vector3.Dot(displacement, right);
                planarDisplacementMeters = Vector3.ProjectOnPlane(displacement, Vector3.up).magnitude;
                finalUprightDot = finalUpright;
                cumulativeLocalPitchRad = cumulativePitch - initialCumulativePitch;
                finalPassiveWheelSpeedRadPerSecond =
                    (float[])lastPassiveWheelSpeedRadPerSecond.Clone();

                float lastSecondStart = Mathf.Max(startSeconds, endSeconds - 1f);
                float sum = 0f;
                int count = 0;
                foreach (TimedSpeed sample in forwardSpeeds)
                {
                    if (sample.timeSeconds >= lastSecondStart)
                    {
                        sum += sample.value;
                        count++;
                    }
                }

                meanForwardSpeedLastSecondMetersPerSecond = count > 0 ? sum / count : 0f;
            }
        }

        private readonly struct TimedSpeed
        {
            public TimedSpeed(float timeSeconds, float value)
            {
                this.timeSeconds = timeSeconds;
                this.value = value;
            }

            public readonly float timeSeconds;
            public readonly float value;
        }

        [Serializable]
        private sealed class PolicyActionMetric
        {
            public int slot;
            public string policyName;
            public int sampleCount;
            public float maximumObservedAbsoluteAction;
            public float maximumAllowedAbsoluteAction;
        }

        [Serializable]
        private sealed class BehaviorCheck
        {
            public string name;
            public float observed;
            public float minimum;
            public float maximum;
            public string unit;
            public bool passed;

            public BehaviorCheck(
                string name,
                float observed,
                float minimum,
                float maximum,
                string unit)
            {
                this.name = name;
                this.observed = observed;
                this.minimum = minimum;
                this.maximum = maximum;
                this.unit = unit;
                passed = IsFinite(observed) && observed >= minimum && observed <= maximum;
            }
        }

        [Serializable]
        private sealed class BehaviorFaultFrame
        {
            public string reason;
            public string controllerFault;
            public string stage;
            public int physicsStep;
            public float timeSeconds;
            public int policySlot;
            public string policyName;
            public Vector3 rootPosition;
            public Quaternion rootRotationXyzw;
            public Vector3 rootLinearVelocity;
            public Vector3 rootAngularVelocity;
            public float uprightDot;
            public float gravityNorm;
            public float maximumAbsoluteRawAction;
            public float[] rawAction = Array.Empty<float>();
            public float[] observation = Array.Empty<float>();
            public float[] passiveWheelVelocityRadPerSecond = Array.Empty<float>();

            public static BehaviorFaultFrame Capture(
                MicroDuckDemoController controller,
                string stage,
                int physicsStep,
                float timeSeconds,
                string reason)
            {
                ArticulationBody root = controller.ActiveRig?.RootBody;
                float[] observation = controller.LastObservation;
                float gravityNorm = observation.Length == PolicyContract.ObservationCount
                    ? new Vector3(observation[3], observation[4], observation[5]).magnitude
                    : 0f;
                float[] wheels = Array.Empty<float>();
                if (controller.ActiveRig != null)
                {
                    try
                    {
                        wheels = controller.ActiveRig.ReadPassiveWheelVelocityRadPerSecond();
                    }
                    catch
                    {
                        wheels = Array.Empty<float>();
                    }
                }

                return new BehaviorFaultFrame
                {
                    reason = reason,
                    controllerFault = controller.Fault,
                    stage = stage,
                    physicsStep = physicsStep,
                    timeSeconds = timeSeconds,
                    policySlot = controller.ActivePolicySlot,
                    policyName = controller.ActivePolicyName,
                    rootPosition = root != null ? root.transform.position : Vector3.zero,
                    rootRotationXyzw = root != null ? root.transform.rotation : Quaternion.identity,
                    rootLinearVelocity = root != null ? root.velocity : Vector3.zero,
                    rootAngularVelocity = root != null ? root.angularVelocity : Vector3.zero,
                    uprightDot = root != null
                        ? Vector3.Dot(root.transform.up, Vector3.up)
                        : 0f,
                    gravityNorm = gravityNorm,
                    maximumAbsoluteRawAction = MaximumAbsolute(controller.LastRawAction),
                    rawAction = controller.LastRawAction,
                    observation = observation,
                    passiveWheelVelocityRadPerSecond = wheels,
                };
            }

            public static BehaviorFaultFrame FromException(
                MicroDuckDemoController controller,
                Exception error)
            {
                if (controller == null)
                {
                    return new BehaviorFaultFrame
                    {
                        reason = error.ToString(),
                        stage = "setup",
                    };
                }

                return Capture(controller, "exception", -1, 0f, error.ToString());
            }
        }
    }
}
