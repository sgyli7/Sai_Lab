using System;
using System.Collections;
using System.IO;
using System.Runtime.InteropServices;
using Mujoco;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace AgenticRobot.MicroDuck.Mujoco
{
    /// <summary>
    /// Opt-in player-only acceptance probe. It is inert unless the runner passes
    /// -microduckSmokeReport, then exits after proving a native policy tick.
    /// </summary>
    [DefaultExecutionOrder(1000)]
    public sealed unsafe class MujocoPlayerSmokeProbe : MonoBehaviour
    {
        private const string ReportArgument = "-microduckSmokeReport";
        private const int MaximumFixedFrames = 600;
        private const int MinimumPolicyTicks = 3;

        [SerializeField] private MujocoDemoController controller;

        public void Configure(MujocoDemoController value)
        {
            controller = value;
        }

        public static bool TryGetReportPath(string[] arguments, out string reportPath)
        {
            reportPath = string.Empty;
            if (arguments == null)
            {
                return false;
            }

            for (int index = 0; index < arguments.Length - 1; index++)
            {
                if (string.Equals(arguments[index], ReportArgument, StringComparison.Ordinal)
                    && !string.IsNullOrWhiteSpace(arguments[index + 1]))
                {
                    reportPath = arguments[index + 1];
                    return true;
                }
            }

            return false;
        }

        public static bool AllFinite(float[] values, int expectedCount)
        {
            if (values == null || values.Length != expectedCount)
            {
                return false;
            }

            foreach (float value in values)
            {
                if (float.IsNaN(value) || float.IsInfinity(value))
                {
                    return false;
                }
            }

            return true;
        }

        private IEnumerator Start()
        {
            if (!TryGetReportPath(Environment.GetCommandLineArgs(), out string reportPath))
            {
                yield break;
            }

            int waitedFrames = 0;
            while (waitedFrames < MaximumFixedFrames
                && (controller == null
                    || !controller.IsHealthy
                    || controller.PolicyTicks < MinimumPolicyTicks))
            {
                waitedFrames++;
                yield return new WaitForFixedUpdate();
            }

            PlayerSmokeReport report = Capture(waitedFrames);
            WriteReport(reportPath, report);
            if (report.passed)
            {
                Debug.Log($"MICRODUCK_PLAYER_SMOKE_PASS ticks={report.policyTicks}");
            }
            else
            {
                Debug.LogError($"MICRODUCK_PLAYER_SMOKE_FAIL {report.fault}");
            }

            Application.Quit(report.passed ? 0 : 1);
        }

        private PlayerSmokeReport Capture(int waitedFrames)
        {
            bool hasController = controller != null;
            float[] observation = hasController ? controller.LastObservation : Array.Empty<float>();
            float[] action = hasController ? controller.LastRawAction : Array.Empty<float>();
            float[] targets = hasController ? controller.LastTargets : Array.Empty<float>();
            bool finite = AllFinite(observation, PolicyContract.ObservationCount)
                && AllFinite(action, PolicyContract.ActionCount)
                && AllFinite(targets, PolicyContract.ActionCount);
            int nativeVersion = MujocoLib.mj_version();
            string nativeVersionString = Marshal.PtrToStringAnsi(MujocoLib.mj_versionString())
                ?? string.Empty;
            string fault = hasController ? controller.Fault : "MujocoDemoController was not found.";
            bool passed = hasController
                && controller.IsHealthy
                && controller.PolicyTicks >= MinimumPolicyTicks
                && controller.BackendName == "MuJoCo 3.12 + Barracuda 3.0.1 CPU"
                && nativeVersion == 3012000
                && nativeVersionString == "3.12.0"
                && finite;
            if (!passed && string.IsNullOrEmpty(fault))
            {
                fault = $"Native player did not satisfy the smoke contract in {waitedFrames} fixed frames.";
            }

            return new PlayerSmokeReport
            {
                schemaVersion = 1,
                passed = passed,
                scene = SceneManager.GetActiveScene().name,
                nativeVersion = nativeVersion,
                nativeVersionString = nativeVersionString,
                backend = hasController ? controller.BackendName : "not loaded",
                policyTicks = hasController ? controller.PolicyTicks : 0,
                observationCount = observation.Length,
                actionCount = action.Length,
                targetCount = targets.Length,
                allFinite = finite,
                waitedFixedFrames = waitedFrames,
                fault = fault,
            };
        }

        private static void WriteReport(string reportPath, PlayerSmokeReport report)
        {
            string absolutePath = Path.GetFullPath(reportPath);
            string directory = Path.GetDirectoryName(absolutePath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.WriteAllText(absolutePath, JsonUtility.ToJson(report, prettyPrint: true) + "\n");
        }

        [Serializable]
        private sealed class PlayerSmokeReport
        {
            public int schemaVersion;
            public bool passed;
            public string scene;
            public int nativeVersion;
            public string nativeVersionString;
            public string backend;
            public int policyTicks;
            public int observationCount;
            public int actionCount;
            public int targetCount;
            public bool allFinite;
            public int waitedFixedFrames;
            public string fault;
        }
    }
}
