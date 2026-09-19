using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace AgenticRobot.MicroDuck.Mujoco
{
    /// <summary>
    /// Opt-in player tour. Inert unless <c>-microduckTourReport</c> is passed,
    /// then drives the same event table as the Windows visual-acceptance script
    /// through public controller/camera/terrain APIs and writes numeric evidence.
    /// </summary>
    [DefaultExecutionOrder(1001)]
    public sealed class MujocoPlayerAcceptanceTour : MonoBehaviour
    {
        public const string ReportArgument = "-microduckTourReport";
        public const string FramesArgument = "-microduckTourFrames";
        public const float TourDurationSeconds = 24f;
        public const int FramesPerSecond = 8;

        // From config/policy-scenarios.json alpha_walking.acceptance; do not
        // change that JSON. The Windows PlayMode walking test uses a stricter
        // pair (0.1 m / 0.9); this tour uses the published scenario contract.
        public const float WalkingMinForwardMeters = 0.03f;
        public const float WalkingMinUprightDot = 0.25f;
        public const float ResetMaxHorizontalMeters = 0.05f;
        public const float WalkingCommandForward = 0.2f;
        public const float WalkingDurationSeconds = 6f;

        private const int MaximumWaitFrames = 600;
        private const int VkTab = 0x09;
        private const int VkDigit1 = 0x31;
        private const int VkDigit2 = 0x32;
        private const int VkDigit7 = 0x37;
        private const int VkC = 0x43;
        private const int VkF = 0x46;
        private const int VkR = 0x52;
        private const int VkT = 0x54;
        private const int VkW = 0x57;

        [SerializeField] private MujocoDemoController controller;
        [SerializeField] private MujocoTerrainNavigator terrainNavigator;
        [SerializeField] private MujocoCameraRig cameraRig;

        public static readonly TourEvent[] Events =
        {
            Press(0.2f, "select standing policy 2 on flat_plaza", VkDigit2),
            Press(0.5f, "reset upright on flat_plaza", VkR),
            Press(1.0f, "camera Side to Rear", VkC),
            Press(2.0f, "camera Rear to Top", VkC),
            Press(3.0f, "camera Top to Showcase", VkC),
            Press(4.0f, "camera Showcase to Side", VkC),
            Orbit(4.3f, "orbit follow camera with LMB", 90, -25),
            Wheel(4.7f, "zoom follow camera with mouse wheel", 120),
            Press(5.0f, "enter camera FreeFly", VkTab),
            Down(5.3f, "move FreeFly camera forward", VkW),
            Up(6.2f, "stop FreeFly camera", VkW),
            Press(6.4f, "leave camera FreeFly", VkTab),
            Press(6.7f, "focus camera on Duck", VkF),
            Press(7.5f, "terrain upstream_pyramid_stairs", VkT),
            Press(9.8f, "terrain upstream_random_grid", VkT),
            Press(10.1f, "select walking policy 1", VkDigit1),
            Down(10.5f, "walk across upstream_random_grid", VkW),
            Up(11.7f, "stop on upstream_random_grid", VkW),
            Press(12.4f, "terrain upstream_pyramid_slope", VkT),
            Press(14.8f, "terrain upstream_roller_slope", VkT),
            Press(15.1f, "select roller policy 7", VkDigit7),
            Press(17.5f, "terrain rock_steps", VkT),
            Press(17.8f, "return to legged walking policy 1", VkDigit1),
            Press(20.2f, "terrain stairs_bridge", VkT),
            Press(21.45f, "camera handoff Side to Rear", VkC),
            Press(21.65f, "camera handoff Rear to Top", VkC),
            Press(21.85f, "camera handoff Top to Showcase", VkC),
            Press(22.05f, "restore camera Side preset for handoff", VkC),
            Press(22.3f, "return to flat_plaza for handoff", VkT),
            Press(22.55f, "select standing policy 2 for handoff", VkDigit2),
            Press(22.8f, "final upright reset on flat_plaza", VkR),
        };

        public void Configure(
            MujocoDemoController value,
            MujocoTerrainNavigator navigator,
            MujocoCameraRig camera)
        {
            controller = value;
            terrainNavigator = navigator;
            cameraRig = camera;
        }

        public static bool TryGetArgument(
            string[] arguments,
            string flag,
            out string value)
        {
            value = string.Empty;
            if (arguments == null)
            {
                return false;
            }

            for (int index = 0; index < arguments.Length - 1; index++)
            {
                if (string.Equals(arguments[index], flag, StringComparison.Ordinal)
                    && !string.IsNullOrWhiteSpace(arguments[index + 1]))
                {
                    value = arguments[index + 1];
                    return true;
                }
            }

            return false;
        }

        private IEnumerator Start()
        {
            if (!TryGetArgument(
                Environment.GetCommandLineArgs(),
                ReportArgument,
                out string reportPath))
            {
                yield break;
            }

            TryGetArgument(
                Environment.GetCommandLineArgs(),
                FramesArgument,
                out string framesDirectory);

            var faults = new List<string>();
            var eventEvidence = new List<EventEvidence>();
            MujocoKeyboardPolicyInput keyboard =
                UnityEngine.Object.FindObjectOfType<MujocoKeyboardPolicyInput>();
            if (keyboard != null)
            {
                keyboard.enabled = false;
            }

            yield return WaitUntilHealthy(minimumTicks: 1);
            if (controller == null || !controller.IsHealthy)
            {
                faults.Add(controller == null
                    ? "MujocoDemoController was not found."
                    : $"Controller failed to initialize: {controller.Fault}");
                WriteAndQuit(reportPath, eventEvidence, default, faults, passed: false);
                yield break;
            }

            if (string.IsNullOrWhiteSpace(framesDirectory))
            {
                faults.Add("Missing -microduckTourFrames directory.");
            }
            else
            {
                Directory.CreateDirectory(Path.GetFullPath(framesDirectory));
            }

            bool holdingWalk = false;
            bool holdingFreeFly = false;
            int nextEvent = 0;
            int frameCount = Mathf.CeilToInt(TourDurationSeconds * FramesPerSecond);
            float startSeconds = Time.realtimeSinceStartup;

            for (int frameIndex = 0; frameIndex < frameCount; frameIndex++)
            {
                float targetTime = frameIndex / (float)FramesPerSecond;
                while ((Time.realtimeSinceStartup - startSeconds) < targetTime)
                {
                    ApplyHeldInputs(holdingWalk, holdingFreeFly);
                    yield return null;
                }

                float elapsed = Time.realtimeSinceStartup - startSeconds;
                while (nextEvent < Events.Length && Events[nextEvent].At <= elapsed)
                {
                    TourEvent tourEvent = Events[nextEvent];
                    Dispatch(tourEvent, ref holdingWalk, ref holdingFreeFly);
                    eventEvidence.Add(new EventEvidence
                    {
                        name = tourEvent.Name,
                        kind = tourEvent.Kind,
                        virtualKey = tourEvent.HasVirtualKey ? tourEvent.VirtualKey : (int?)null,
                        sentAtSeconds = Math.Round(elapsed, 3),
                    });
                    nextEvent++;
                    elapsed = Time.realtimeSinceStartup - startSeconds;
                }

                ApplyHeldInputs(holdingWalk, holdingFreeFly);
                if (!string.IsNullOrWhiteSpace(framesDirectory))
                {
                    yield return new WaitForEndOfFrame();
                    string framePath = Path.Combine(
                        Path.GetFullPath(framesDirectory),
                        $"frame-{(frameIndex + 1):D4}.png");
                    ScreenCapture.CaptureScreenshot(framePath);
                    yield return new WaitForEndOfFrame();
                }
                else
                {
                    yield return null;
                }
            }

            for (int flush = 0; flush < FramesPerSecond; flush++)
            {
                yield return new WaitForEndOfFrame();
            }

            holdingWalk = false;
            holdingFreeFly = false;
            if (controller != null)
            {
                controller.SetTwist(0f, 0f, 0f);
            }

            InteractionChecks checks = default;
            yield return RunNumericChecks(checksReceiver => checks = checksReceiver, faults);
            bool passed = checks.passed && faults.Count == 0;
            WriteAndQuit(reportPath, eventEvidence, checks, faults, passed);
        }

        private void Dispatch(
            TourEvent tourEvent,
            ref bool holdingWalk,
            ref bool holdingFreeFly)
        {
            switch (tourEvent.Kind)
            {
                case "press":
                    DispatchPress(tourEvent.VirtualKey);
                    break;
                case "down":
                    if (tourEvent.VirtualKey == VkW)
                    {
                        if (cameraRig != null && cameraRig.OwnsNavigationInput)
                        {
                            holdingFreeFly = true;
                            holdingWalk = false;
                        }
                        else
                        {
                            holdingWalk = true;
                            holdingFreeFly = false;
                            controller.SetTwist(WalkingCommandForward, 0f, 0f);
                        }
                    }

                    break;
                case "up":
                    if (tourEvent.VirtualKey == VkW)
                    {
                        holdingWalk = false;
                        holdingFreeFly = false;
                        controller.SetTwist(0f, 0f, 0f);
                    }

                    break;
                case "orbit":
                    cameraRig?.ApplyOrbitInput(new Vector2(tourEvent.DeltaX, tourEvent.DeltaY));
                    break;
                case "wheel":
                    // Windows mouse_event wheel ticks are 120 per notch; Unity
                    // mouseScrollDelta is 1 per notch.
                    cameraRig?.ApplyZoomInput(tourEvent.WheelDelta / 120f);
                    break;
            }
        }

        private void DispatchPress(int virtualKey)
        {
            if (virtualKey >= VkDigit1 && virtualKey <= 0x39)
            {
                controller.SwitchPolicy(virtualKey - 0x30);
                return;
            }

            switch (virtualKey)
            {
                case VkR:
                    controller.ResetActiveRobot();
                    break;
                case VkC:
                    cameraRig?.CyclePreset();
                    break;
                case VkTab:
                    cameraRig?.ToggleMode();
                    break;
                case VkF:
                    cameraRig?.FocusOnRobot();
                    break;
                case VkT:
                    terrainNavigator?.Next();
                    break;
            }
        }

        private void ApplyHeldInputs(bool holdingWalk, bool holdingFreeFly)
        {
            if (holdingWalk)
            {
                controller.SetTwist(WalkingCommandForward, 0f, 0f);
            }

            if (holdingFreeFly && cameraRig != null)
            {
                cameraRig.ApplyFreeFlyTranslation(1f, 0f, 0f, boost: false);
            }
        }

        private IEnumerator RunNumericChecks(
            Action<InteractionChecks> setChecks,
            List<string> faults)
        {
            var checks = new InteractionChecks();
            if (terrainNavigator != null && !terrainNavigator.Select("flat_plaza"))
            {
                faults.Add("Could not select flat_plaza for the numeric walking segment.");
            }

            if (!controller.SwitchPolicy(1))
            {
                faults.Add($"Could not select walking policy 1: {controller.Fault}");
            }

            controller.ResetActiveRobot();
            yield return WaitUntilHealthy(minimumTicks: 1);

            Vector3 start = Horizontal(controller.RootPositionMeters);
            Vector3 heading = ResolveHeading(controller);
            float startFixed = Time.fixedTime;
            float minUpright = controller.TrunkUpright;
            while (Time.fixedTime - startFixed < WalkingDurationSeconds)
            {
                controller.SetTwist(WalkingCommandForward, 0f, 0f);
                minUpright = Mathf.Min(minUpright, controller.TrunkUpright);
                yield return new WaitForFixedUpdate();
            }

            controller.SetTwist(0f, 0f, 0f);
            Vector3 end = Horizontal(controller.RootPositionMeters);
            checks.walkingForwardMeters = Vector3.Dot(end - start, heading);
            checks.walkingUprightDot = minUpright;
            if (checks.walkingForwardMeters < WalkingMinForwardMeters)
            {
                faults.Add(
                    $"Walking forward displacement {checks.walkingForwardMeters:R} m "
                    + $"is below {WalkingMinForwardMeters:R} m.");
            }

            if (checks.walkingUprightDot < WalkingMinUprightDot)
            {
                faults.Add(
                    $"Walking upright dot {checks.walkingUprightDot:R} "
                    + $"is below {WalkingMinUprightDot:R}.");
            }

            RecordTensorFaults(controller, "walking", faults);

            controller.ResetActiveRobot();
            yield return new WaitForFixedUpdate();
            yield return new WaitForFixedUpdate();
            Vector3 resetTrunk = Horizontal(controller.RootPositionMeters);
            Vector3 resetTarget = Horizontal(controller.ResetPositionMeters);
            checks.resetDistanceMeters = Vector3.Distance(resetTrunk, resetTarget);
            if (checks.resetDistanceMeters > ResetMaxHorizontalMeters || !controller.IsHealthy)
            {
                faults.Add(
                    $"Reset distance {checks.resetDistanceMeters:R} m "
                    + $"(limit {ResetMaxHorizontalMeters:R} m); healthy={controller.IsHealthy}; "
                    + controller.Fault);
            }

            RecordTensorFaults(controller, "reset", faults);

            GameObject leggedRoot = controller.ActiveModelRoot;
            int ticks = controller.PolicyTicks;
            bool ticksMonotonic = true;
            if (!controller.SwitchPolicy(2) || !string.IsNullOrEmpty(controller.Fault)
                || !ReferenceEquals(controller.ActiveModelRoot, leggedRoot))
            {
                ticksMonotonic = false;
                faults.Add($"Hot-swap 1→2 failed: {controller.Fault}");
            }

            yield return WaitUntilHealthy(minimumTicks: ticks + 1);
            if (controller.PolicyTicks < ticks)
            {
                ticksMonotonic = false;
                faults.Add("PolicyTicks decreased during 1→2 hot-swap.");
            }

            ticks = controller.PolicyTicks;
            if (!controller.SwitchPolicy(1) || !string.IsNullOrEmpty(controller.Fault)
                || !ReferenceEquals(controller.ActiveModelRoot, leggedRoot))
            {
                ticksMonotonic = false;
                faults.Add($"Hot-swap 2→1 failed: {controller.Fault}");
            }

            yield return WaitUntilHealthy(minimumTicks: ticks + 1);
            if (controller.PolicyTicks < ticks)
            {
                ticksMonotonic = false;
                faults.Add("PolicyTicks decreased during 2→1 hot-swap.");
            }

            checks.hotSwapTicksMonotonic = ticksMonotonic
                && ReferenceEquals(controller.ActiveModelRoot, leggedRoot)
                && string.IsNullOrEmpty(controller.Fault);
            RecordTensorFaults(controller, "hot-swap", faults);

            GameObject beforeRoller = controller.ActiveModelRoot;
            if (!controller.SwitchPolicy(7))
            {
                faults.Add($"Switch 1→7 failed: {controller.Fault}");
            }

            yield return WaitUntilHealthy(minimumTicks: 1);
            checks.rollerModelRebuilt = controller.ActiveModelRoot != null
                && !ReferenceEquals(controller.ActiveModelRoot, beforeRoller)
                && controller.ActiveModelRoot.name.IndexOf("Roller", StringComparison.Ordinal) >= 0
                && controller.IsHealthy;
            if (!checks.rollerModelRebuilt)
            {
                faults.Add(
                    "1→7 did not rebuild onto the Roller model root: "
                    + (controller.ActiveModelRoot == null
                        ? "<null>"
                        : controller.ActiveModelRoot.name));
            }

            RecordTensorFaults(controller, "roller", faults);
            GameObject rollerRoot = controller.ActiveModelRoot;
            if (!controller.SwitchPolicy(8))
            {
                faults.Add($"Switch 7→8 failed: {controller.Fault}");
            }

            yield return WaitUntilHealthy(minimumTicks: 1);
            if (!ReferenceEquals(controller.ActiveModelRoot, rollerRoot)
                || !string.IsNullOrEmpty(controller.Fault))
            {
                faults.Add($"7→8 changed ActiveModelRoot or faulted: {controller.Fault}");
            }

            RecordTensorFaults(controller, "roller-hot-swap", faults);
            if (!controller.SwitchPolicy(1))
            {
                faults.Add($"Switch 8→1 failed: {controller.Fault}");
            }

            yield return WaitUntilHealthy(minimumTicks: 1);
            checks.leggedRestored = controller.ActiveModelRoot != null
                && ReferenceEquals(controller.ActiveModelRoot, leggedRoot)
                && controller.ActiveModelRoot.name.IndexOf("Roller", StringComparison.Ordinal) < 0
                && controller.IsHealthy
                && string.IsNullOrEmpty(controller.Fault);
            if (!checks.leggedRestored)
            {
                faults.Add(
                    "8→1 did not restore the Legged model root: "
                    + (controller.ActiveModelRoot == null
                        ? "<null>"
                        : controller.ActiveModelRoot.name)
                    + "; " + controller.Fault);
            }

            RecordTensorFaults(controller, "legged-restore", faults);
            checks.tensorsFinite = faults.All(fault =>
                fault.IndexOf("tensor", StringComparison.OrdinalIgnoreCase) < 0);
            checks.faults = faults.ToArray();
            checks.passed = checks.walkingForwardMeters >= WalkingMinForwardMeters
                && checks.walkingUprightDot >= WalkingMinUprightDot
                && checks.resetDistanceMeters <= ResetMaxHorizontalMeters
                && checks.hotSwapTicksMonotonic
                && checks.rollerModelRebuilt
                && checks.leggedRestored
                && checks.tensorsFinite
                && faults.Count == 0
                && controller.IsHealthy;
            setChecks(checks);
        }

        private IEnumerator WaitUntilHealthy(int minimumTicks)
        {
            int waited = 0;
            while (waited < MaximumWaitFrames
                && (controller == null
                    || !controller.IsHealthy
                    || controller.PolicyTicks < minimumTicks))
            {
                waited++;
                yield return new WaitForFixedUpdate();
            }
        }

        private static void RecordTensorFaults(
            MujocoDemoController value,
            string context,
            List<string> faults)
        {
            bool observation = MujocoPlayerSmokeProbe.AllFinite(
                value.LastObservation,
                PolicyContract.ObservationCount);
            bool action = MujocoPlayerSmokeProbe.AllFinite(
                value.LastRawAction,
                PolicyContract.ActionCount);
            if (!observation || !action)
            {
                faults.Add(
                    $"{context} tensors are not finite 61/14: "
                    + $"obs={value.LastObservation?.Length ?? 0}, "
                    + $"act={value.LastRawAction?.Length ?? 0}.");
            }
        }

        private static Vector3 ResolveHeading(MujocoDemoController controller)
        {
            // Native walking measures qpos axis 0 (MuJoCo +X). The official
            // MjEngineTool.UnityVector3 used by RootPositionMeters keeps that
            // axis as Unity +X (see the overlay "position x=" readout).
            Vector3 heading = Quaternion.Euler(0f, controller.ResetYawDegrees, 0f)
                * Vector3.right;
            heading = Horizontal(heading);
            if (heading.sqrMagnitude < 1e-8f)
            {
                heading = Vector3.right;
            }

            return heading.normalized;
        }

        private static Vector3 Horizontal(Vector3 value)
        {
            value.y = 0f;
            return value;
        }

        private void WriteAndQuit(
            string reportPath,
            List<EventEvidence> eventEvidence,
            InteractionChecks checks,
            List<string> faults,
            bool passed)
        {
            if (checks.faults == null)
            {
                checks.faults = faults.ToArray();
            }

            string json = BuildReportJson(eventEvidence, checks, passed);
            string absolutePath = Path.GetFullPath(reportPath);
            string directory = Path.GetDirectoryName(absolutePath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.WriteAllText(absolutePath, json);
            if (passed)
            {
                Debug.Log("MICRODUCK_PLAYER_TOUR_PASS");
            }
            else
            {
                Debug.LogError("MICRODUCK_PLAYER_TOUR_FAIL " + string.Join("; ", faults));
            }

            Application.Quit(passed ? 0 : 1);
        }

        private static string BuildReportJson(
            List<EventEvidence> eventEvidence,
            InteractionChecks checks,
            bool passed)
        {
            var builder = new StringBuilder();
            builder.Append("{\n");
            builder.Append("  \"schemaVersion\": 2,\n");
            builder.Append("  \"captureCompleted\": true,\n");
            builder.Append("  \"requiresVisualReview\": true,\n");
            builder.Append("  \"capturedAt\": \"");
            builder.Append(DateTimeOffset.UtcNow.ToString("o", CultureInfo.InvariantCulture));
            builder.Append("\",\n");
            builder.Append("  \"scene\": \"");
            builder.Append(Escape(SceneManager.GetActiveScene().name));
            builder.Append("\",\n");
            builder.Append("  \"events\": [\n");
            for (int index = 0; index < eventEvidence.Count; index++)
            {
                EventEvidence item = eventEvidence[index];
                builder.Append("    {\n");
                builder.Append("      \"name\": \"");
                builder.Append(Escape(item.name));
                builder.Append("\",\n");
                builder.Append("      \"kind\": \"");
                builder.Append(Escape(item.kind));
                builder.Append("\",\n");
                builder.Append("      \"virtualKey\": ");
                builder.Append(item.virtualKey.HasValue
                    ? item.virtualKey.Value.ToString(CultureInfo.InvariantCulture)
                    : "null");
                builder.Append(",\n");
                builder.Append("      \"sentAtSeconds\": ");
                builder.Append(item.sentAtSeconds.ToString("0.###", CultureInfo.InvariantCulture));
                builder.Append("\n    }");
                builder.Append(index + 1 < eventEvidence.Count ? ",\n" : "\n");
            }

            builder.Append("  ],\n");
            builder.Append("  \"interactionChecks\": {\n");
            builder.Append("    \"walkingForwardMeters\": ");
            builder.Append(FormatFloat(checks.walkingForwardMeters));
            builder.Append(",\n");
            builder.Append("    \"walkingUprightDot\": ");
            builder.Append(FormatFloat(checks.walkingUprightDot));
            builder.Append(",\n");
            builder.Append("    \"resetDistanceMeters\": ");
            builder.Append(FormatFloat(checks.resetDistanceMeters));
            builder.Append(",\n");
            builder.Append("    \"hotSwapTicksMonotonic\": ");
            builder.Append(checks.hotSwapTicksMonotonic ? "true" : "false");
            builder.Append(",\n");
            builder.Append("    \"rollerModelRebuilt\": ");
            builder.Append(checks.rollerModelRebuilt ? "true" : "false");
            builder.Append(",\n");
            builder.Append("    \"leggedRestored\": ");
            builder.Append(checks.leggedRestored ? "true" : "false");
            builder.Append(",\n");
            builder.Append("    \"tensorsFinite\": ");
            builder.Append(checks.tensorsFinite ? "true" : "false");
            builder.Append(",\n");
            builder.Append("    \"faults\": [");
            string[] listedFaults = checks.faults ?? Array.Empty<string>();
            for (int index = 0; index < listedFaults.Length; index++)
            {
                if (index > 0)
                {
                    builder.Append(", ");
                }

                builder.Append("\"");
                builder.Append(Escape(listedFaults[index]));
                builder.Append("\"");
            }

            builder.Append("],\n");
            builder.Append("    \"passed\": ");
            builder.Append(passed ? "true" : "false");
            builder.Append("\n  }\n");
            builder.Append("}\n");
            return builder.ToString();
        }

        private static string FormatFloat(float value)
        {
            return value.ToString("R", CultureInfo.InvariantCulture);
        }

        private static string Escape(string value)
        {
            if (string.IsNullOrEmpty(value))
            {
                return string.Empty;
            }

            return value
                .Replace("\\", "\\\\")
                .Replace("\"", "\\\"")
                .Replace("\n", "\\n")
                .Replace("\r", "\\r");
        }

        private static TourEvent Press(float at, string name, int virtualKey)
        {
            return new TourEvent(at, name, "press", virtualKey, 0, 0, 0);
        }

        private static TourEvent Down(float at, string name, int virtualKey)
        {
            return new TourEvent(at, name, "down", virtualKey, 0, 0, 0);
        }

        private static TourEvent Up(float at, string name, int virtualKey)
        {
            return new TourEvent(at, name, "up", virtualKey, 0, 0, 0);
        }

        private static TourEvent Orbit(float at, string name, int deltaX, int deltaY)
        {
            return new TourEvent(at, name, "orbit", -1, deltaX, deltaY, 0);
        }

        private static TourEvent Wheel(float at, string name, int wheelDelta)
        {
            return new TourEvent(at, name, "wheel", -1, 0, 0, wheelDelta);
        }

        public readonly struct TourEvent
        {
            public TourEvent(
                float at,
                string name,
                string kind,
                int virtualKey,
                int deltaX,
                int deltaY,
                int wheelDelta)
            {
                At = at;
                Name = name;
                Kind = kind;
                VirtualKey = virtualKey;
                DeltaX = deltaX;
                DeltaY = deltaY;
                WheelDelta = wheelDelta;
            }

            public float At { get; }
            public string Name { get; }
            public string Kind { get; }
            public int VirtualKey { get; }
            public int DeltaX { get; }
            public int DeltaY { get; }
            public int WheelDelta { get; }
            public bool HasVirtualKey => VirtualKey >= 0;
        }

        private struct EventEvidence
        {
            public string name;
            public string kind;
            public int? virtualKey;
            public double sentAtSeconds;
        }

        private struct InteractionChecks
        {
            public float walkingForwardMeters;
            public float walkingUprightDot;
            public float resetDistanceMeters;
            public bool hotSwapTicksMonotonic;
            public bool rollerModelRebuilt;
            public bool leggedRestored;
            public bool tensorsFinite;
            public string[] faults;
            public bool passed;
        }
    }
}
