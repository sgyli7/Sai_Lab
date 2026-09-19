using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using AgenticRobot.MicroDuck.Mujoco;
using Mujoco;
using NUnit.Framework;
using Unity.Barracuda;
using UnityEditor;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed unsafe class AuthoritativeMujocoPolicyBehaviorTests
    {
        private const double PhysicsTimestepSeconds = 0.005;
        private const int PolicyDecimation = 4;
        private const string PolicyAssetDirectory =
            "Assets/MicroDuck/Generated/Policies/Barracuda";

        private static readonly Scenario[] OfficialScenarios =
        {
            new Scenario(1, "scene.xml", 6.0, 0.2f, 0f, new[]
            {
                Check.Minimum("minForwardDistanceMeters", "forwardDistanceMeters", 0.03),
                Check.Minimum("minUprightDot", "minimumUprightDot", 0.25),
            }),
            new Scenario(2, "scene.xml", 4.0, 0f, 0f, new[]
            {
                Check.Minimum("minUprightDot", "minimumUprightDot", 0.5),
                Check.Maximum("maxRootDriftMeters", "rootDriftMeters", 0.2),
            }),
            new Scenario(3, "scene.xml", 8.0, 0f, 0f, new[]
            {
                Check.Minimum("minHeightExcursionMeters", "heightExcursionMeters", 0.015),
                Check.Minimum("minFinalUprightDot", "finalUprightDot", 0.35),
            }, 1.0, 4.5),
            new Scenario(4, "scene.xml", 4.0, 0f, 0f, new[]
            {
                Check.Minimum("minJointExcursionRad", "jointExcursionRad", 0.1),
                Check.Minimum("minFinalUprightDot", "finalUprightDot", 0.2),
            }, 0.0),
            new Scenario(5, "scene_ball.xml", 3.0, 0f, 0f, new[]
            {
                Check.Minimum("minBallDisplacementMeters", "ballDisplacementMeters", 0.01),
                Check.Minimum("minJointExcursionRad", "jointExcursionRad", 0.1),
            }),
            new Scenario(6, "scene_ball.xml", 3.0, 0f, 0f, new[]
            {
                Check.Minimum("minBallDisplacementMeters", "ballDisplacementMeters", 0.01),
                Check.Minimum("minJointExcursionRad", "jointExcursionRad", 0.1),
            }),
            new Scenario(7, "scene_rollers.xml", 6.0, 0.3f, 0f, new[]
            {
                Check.Minimum("minForwardDistanceMeters", "forwardDistanceMeters", 0.04),
                Check.Minimum("minUprightDot", "minimumUprightDot", 0.2),
            }),
            new Scenario(8, "scene_rollers.xml", 3.0, 0f, 0f, new[]
            {
                Check.Minimum("minHeightExcursionMeters", "heightExcursionMeters", 0.01),
                Check.Minimum("minJointExcursionRad", "jointExcursionRad", 0.1),
            }, 0.0),
            new Scenario(9, "scene.xml", 3.0, 0f, 0f, new[]
            {
                Check.Minimum("minPitchRotationRad", "pitchRotationRad", 3.0),
                Check.Minimum("minJointExcursionRad", "jointExcursionRad", 0.1),
            }),
        };

        [Test]
        public void AllOfficialPoliciesAndLiveSameModelHotSwapsMeetBehaviorContracts()
        {
            AssertCatalogMatchesLockedScenarios();
            var scenarioResults = new List<BehaviorResult>();
            var compoundResults = new List<BehaviorResult>();

            foreach (Scenario scenario in OfficialScenarios)
            {
                scenarioResults.Add(RunSafely(
                    PolicyCatalog.GetBySlot(scenario.Slot).FileName,
                    () => RunScenario(scenario)));
            }

            compoundResults.Add(RunSafely(
                "stand-roulade-stand-live-hot-swap",
                RunLeggedHotSwapSequence));
            compoundResults.Add(RunSafely(
                "roller-crouch-roller-live-hot-swap",
                RunRollerHotSwapSequence));

            var report = new BehaviorReport
            {
                schemaVersion = 1,
                generatedUtc = DateTime.UtcNow.ToString("O"),
                physicsAuthority = "official MuJoCo 3.12 native runtime",
                nativeVersion = MujocoLib.mj_version(),
                nativeVersionString = Marshal.PtrToStringAnsi(MujocoLib.mj_versionString()),
                policyRuntime = "Barracuda 3.0.1 CSharp",
                physicsTimestepSeconds = PhysicsTimestepSeconds,
                policyDecimation = PolicyDecimation,
                scenarioResults = scenarioResults.ToArray(),
                compoundResults = compoundResults.ToArray(),
            };
            report.passed = report.scenarioResults.All(result => result.passed)
                && report.compoundResults.All(result => result.passed);
            string artifactPath = WriteArtifact(report);

            string[] failures = report.scenarioResults
                .Concat(report.compoundResults)
                .Where(result => !result.passed)
                .Select(result => result.FailureSummary())
                .ToArray();
            TestContext.Progress.WriteLine(
                $"Authoritative MuJoCo behavior artifact: {artifactPath}");
            Assert.That(
                failures,
                Is.Empty,
                "Authoritative MuJoCo/Barracuda behavior failures:\n" +
                string.Join("\n", failures));
        }

        private static BehaviorResult RunScenario(Scenario scenario)
        {
            PolicyEntry entry = PolicyCatalog.GetBySlot(scenario.Slot);
            using (var harness = NativeHarness.Create(entry, scenario.SceneFile))
            {
                harness.Commands.SetTwist(scenario.ForwardCommand, scenario.LeftCommand, 0f);
                harness.Reset(entry.Role);
                foreach (double triggerTime in scenario.TriggerTimesSeconds)
                {
                    if (Math.Abs(triggerTime) <= 1e-12)
                    {
                        harness.Commands.TriggerSkill(0f);
                    }
                }

                StateMetrics metrics = harness.CaptureInitialMetrics();
                var triggered = new HashSet<double>(
                    scenario.TriggerTimesSeconds.Where(time => time <= 1e-12));
                int physicsSteps = RequiredStepCount(scenario.DurationSeconds);
                for (int physicsStep = 0; physicsStep < physicsSteps; physicsStep++)
                {
                    double now = harness.Time;
                    foreach (double triggerTime in scenario.TriggerTimesSeconds)
                    {
                        if (!triggered.Contains(triggerTime) && now + 1e-9 >= triggerTime)
                        {
                            harness.Commands.TriggerSkill((float)now);
                            triggered.Add(triggerTime);
                        }
                    }

                    harness.Step(physicsStep % PolicyDecimation == 0);
                    metrics.Observe(harness);
                }

                MetricValues values = metrics.Complete(harness);
                return BuildResult(
                    entry.FileName,
                    new[] { entry.FileName },
                    scenario.DurationSeconds,
                    physicsSteps,
                    harness.PolicyTicks,
                    values,
                    scenario.Checks,
                    harness.MaximumAbsoluteRawAction,
                    0.0,
                    string.Empty);
            }
        }

        private static BehaviorResult RunLeggedHotSwapSequence()
        {
            PolicyEntry stand = PolicyCatalog.GetBySlot(2);
            PolicyEntry roulade = PolicyCatalog.GetBySlot(9);
            using (var harness = NativeHarness.Create(stand, "scene.xml"))
            {
                harness.Reset(stand.Role);
                StateMetrics overall = harness.CaptureInitialMetrics();
                RunStage(harness, overall, 1.0);

                double maximumStateJump = harness.HotSwap(roulade);
                StateMetrics roll = harness.CaptureInitialMetrics();
                RunStage(harness, overall, roll, 3.0);
                MetricValues rollValues = roll.Complete(harness);

                maximumStateJump = Math.Max(maximumStateJump, harness.HotSwap(stand));
                RunStage(harness, overall, 2.0);
                MetricValues values = overall.Complete(harness);
                values.pitchRotationRad = rollValues.pitchRotationRad;
                var checks = new[]
                {
                    Check.Maximum("maxHotSwapStateJump", "hotSwapStateJump", 1e-12),
                    Check.Minimum("minRouladePitchRotationRad", "pitchRotationRad", 3.0),
                    Check.Minimum("minFinalUprightDot", "finalUprightDot", 0.35),
                };
                return BuildResult(
                    "stand-roulade-stand-live-hot-swap",
                    new[] { stand.FileName, roulade.FileName, stand.FileName },
                    6.0,
                    RequiredStepCount(6.0),
                    harness.PolicyTicks,
                    values,
                    checks,
                    harness.MaximumAbsoluteRawAction,
                    maximumStateJump,
                    string.Empty);
            }
        }

        private static BehaviorResult RunRollerHotSwapSequence()
        {
            PolicyEntry roller = PolicyCatalog.GetBySlot(7);
            PolicyEntry crouch = PolicyCatalog.GetBySlot(8);
            using (var harness = NativeHarness.Create(roller, "scene_rollers.xml"))
            {
                harness.Commands.SetTwist(0.3f, 0f, 0f);
                harness.Reset(roller.Role);
                StateMetrics overall = harness.CaptureInitialMetrics();
                RunStage(harness, overall, 1.0);

                double maximumStateJump = harness.HotSwap(crouch);
                harness.Commands.TriggerSkill((float)harness.Time);
                StateMetrics crouchMetrics = harness.CaptureInitialMetrics();
                RunStage(harness, overall, crouchMetrics, 3.5);
                MetricValues crouchValues = crouchMetrics.Complete(harness);

                maximumStateJump = Math.Max(maximumStateJump, harness.HotSwap(roller));
                harness.Commands.SetTwist(0.3f, 0f, 0f);
                StateMetrics recovery = harness.CaptureInitialMetrics();
                RunStage(harness, overall, recovery, 2.0);
                MetricValues recoveryValues = recovery.Complete(harness);

                MetricValues values = overall.Complete(harness);
                values.heightExcursionMeters = crouchValues.heightExcursionMeters;
                values.forwardDistanceMeters = recoveryValues.forwardDistanceMeters;
                var checks = new[]
                {
                    Check.Maximum("maxHotSwapStateJump", "hotSwapStateJump", 1e-12),
                    Check.Minimum("minCrouchHeightExcursionMeters", "heightExcursionMeters", 0.01),
                    Check.Minimum("minRecoveryForwardDistanceMeters", "forwardDistanceMeters", 0.02),
                    Check.Minimum("minFinalUprightDot", "finalUprightDot", 0.2),
                };
                return BuildResult(
                    "roller-crouch-roller-live-hot-swap",
                    new[] { roller.FileName, crouch.FileName, roller.FileName },
                    6.5,
                    RequiredStepCount(6.5),
                    harness.PolicyTicks,
                    values,
                    checks,
                    harness.MaximumAbsoluteRawAction,
                    maximumStateJump,
                    string.Empty);
            }
        }

        private static void RunStage(
            NativeHarness harness,
            StateMetrics overall,
            double durationSeconds)
        {
            RunStage(harness, overall, null, durationSeconds);
        }

        private static void RunStage(
            NativeHarness harness,
            StateMetrics overall,
            StateMetrics stage,
            double durationSeconds)
        {
            int physicsSteps = RequiredStepCount(durationSeconds);
            for (int physicsStep = 0; physicsStep < physicsSteps; physicsStep++)
            {
                harness.Step(physicsStep % PolicyDecimation == 0);
                overall.Observe(harness);
                stage?.Observe(harness);
            }
        }

        private static BehaviorResult RunSafely(
            string name,
            Func<BehaviorResult> run)
        {
            try
            {
                return run();
            }
            catch (Exception error)
            {
                return new BehaviorResult
                {
                    name = name,
                    passed = false,
                    finiteState = false,
                    failure = error.ToString(),
                };
            }
        }

        private static BehaviorResult BuildResult(
            string name,
            string[] policySequence,
            double durationSeconds,
            int physicsSteps,
            int policyTicks,
            MetricValues values,
            Check[] contracts,
            double maximumAbsoluteRawAction,
            double hotSwapStateJump,
            string failure)
        {
            values.maximumAbsoluteRawAction = maximumAbsoluteRawAction;
            values.hotSwapStateJump = hotSwapStateJump;
            var checks = new List<CheckResult>
            {
                Check.Minimum("finiteState", "finiteState", 1.0).Evaluate(values),
                Check.Maximum(
                    "maxAbsoluteRawAction",
                    "maximumAbsoluteRawAction",
                    5.0).Evaluate(values),
            };
            checks.AddRange(contracts.Select(contract => contract.Evaluate(values)));
            return new BehaviorResult
            {
                name = name,
                policySequence = policySequence,
                simulatedSeconds = durationSeconds,
                physicsStepCount = physicsSteps,
                policyTickCount = policyTicks,
                finiteState = values.finiteState >= 1.0,
                metrics = values,
                checks = checks.ToArray(),
                failure = failure,
                passed = string.IsNullOrEmpty(failure) && checks.All(check => check.passed),
            };
        }

        private static int RequiredStepCount(double durationSeconds)
        {
            double exact = durationSeconds / PhysicsTimestepSeconds;
            int rounded = (int)Math.Round(exact);
            if (Math.Abs(exact - rounded) > 1e-9)
            {
                throw new ArgumentException(
                    $"Duration {durationSeconds:R} is not divisible by timestep " +
                    $"{PhysicsTimestepSeconds:R}.");
            }

            return rounded;
        }

        private static void AssertCatalogMatchesLockedScenarios()
        {
            Assert.That(OfficialScenarios, Has.Length.EqualTo(9));
            Assert.That(PolicyCatalog.Entries, Has.Length.EqualTo(9));
            foreach (Scenario scenario in OfficialScenarios)
            {
                PolicyEntry entry = PolicyCatalog.GetBySlot(scenario.Slot);
                Assert.That(entry.Slot, Is.EqualTo(scenario.Slot));
                Assert.That(
                    File.Exists(OfficialScenePath(scenario.SceneFile)),
                    Is.True,
                    scenario.SceneFile);
                Assert.That(
                    AssetDatabase.LoadAssetAtPath<NNModel>(
                        $"{PolicyAssetDirectory}/{entry.FileName}"),
                    Is.Not.Null,
                    entry.FileName);
            }
        }

        private static string WriteArtifact(BehaviorReport report)
        {
            string repositoryRoot = RepositoryRoot();
            string outputDirectory = Path.Combine(repositoryRoot, "artifacts", "mvp");
            Directory.CreateDirectory(outputDirectory);
            string outputPath = Path.Combine(
                outputDirectory,
                "tuanjie-native-mujoco-policy-behavior.json");
            File.WriteAllText(
                outputPath,
                JsonUtility.ToJson(report, true) + Environment.NewLine);
            return outputPath;
        }

        private static string RepositoryRoot()
        {
            return Path.GetFullPath(Path.Combine(Application.dataPath, "..", ".."));
        }

        private static string OfficialScenePath(string sceneFile)
        {
            return Path.Combine(
                RepositoryRoot(),
                ".cache",
                "upstream",
                "microduck_rl",
                "src",
                "mjlab_microduck",
                "robot",
                "microduck",
                sceneFile);
        }

        private sealed class NativeHarness : IDisposable
        {
            private readonly MujocoLib.mjModel_* model;
            private readonly MujocoLib.mjData_* data;
            private readonly int rootQposAddress;
            private readonly int trunkBodyId;
            private readonly int ballQposAddress;
            private readonly int[] jointQposAddresses;
            private MujocoPolicyStepper stepper;
            private int ticks;
            private double maximumAbsoluteRawAction;

            private NativeHarness(
                MujocoLib.mjModel_* model,
                MujocoLib.mjData_* data,
                PolicyCommandState commands,
                MujocoPolicyStepper stepper)
            {
                this.model = model;
                this.data = data;
                Commands = commands;
                this.stepper = stepper;
                int rootJoint = RequiredId(
                    model,
                    MujocoLib.mjtObj.mjOBJ_JOINT,
                    "trunk_base_freejoint");
                rootQposAddress = model->jnt_qposadr[rootJoint];
                trunkBodyId = RequiredId(model, MujocoLib.mjtObj.mjOBJ_BODY, "trunk_base");
                int ballJoint = MujocoLib.mj_name2id(
                    model,
                    (int)MujocoLib.mjtObj.mjOBJ_JOINT,
                    "ball_free");
                ballQposAddress = ballJoint >= 0 ? model->jnt_qposadr[ballJoint] : -1;
                jointQposAddresses = new int[PolicyContract.ActionCount];
                for (int actuator = 0; actuator < jointQposAddresses.Length; actuator++)
                {
                    int joint = model->actuator_trnid[2 * actuator];
                    jointQposAddresses[actuator] = model->jnt_qposadr[joint];
                }
            }

            public PolicyCommandState Commands { get; }
            public double Time => data->time;
            public int PolicyTicks => ticks;
            public double MaximumAbsoluteRawAction => maximumAbsoluteRawAction;

            public static NativeHarness Create(PolicyEntry entry, string sceneFile)
            {
                MujocoLib.mjModel_* model = null;
                MujocoLib.mjData_* data = null;
                IPolicyRuntime runtime = null;
                try
                {
                    model = MjEngineTool.LoadModelFromFile(OfficialScenePath(sceneFile));
                    if (model == null)
                    {
                        throw new InvalidOperationException($"Could not load {sceneFile}.");
                    }

                    model->opt.timestep = PhysicsTimestepSeconds;
                    data = MujocoLib.mj_makeData(model);
                    if (data == null)
                    {
                        throw new InvalidOperationException($"Could not allocate data for {sceneFile}.");
                    }

                    runtime = CreateRuntime(entry);
                    var commands = new PolicyCommandState();
                    commands.SelectSlot(entry.Slot);
                    var stepper = new MujocoPolicyStepper(
                        model,
                        data,
                        runtime,
                        commands,
                        entry.RobotVariant,
                        entry.ActionScale);
                    runtime = null;
                    return new NativeHarness(model, data, commands, stepper);
                }
                catch
                {
                    runtime?.Dispose();
                    if (data != null)
                    {
                        MujocoLib.mj_deleteData(data);
                    }

                    if (model != null)
                    {
                        MujocoLib.mj_deleteModel(model);
                    }

                    throw;
                }
            }

            public void Reset(PolicyRole role)
            {
                stepper.ResetToHome();
                stepper.PlaceBallForRole(role);
            }

            public void Step(bool evaluatePolicy)
            {
                if (evaluatePolicy)
                {
                    if (!stepper.TickPolicy((float)data->time, out string error))
                    {
                        throw new InvalidOperationException(
                            $"Policy tick {ticks} failed at MuJoCo t={data->time:R}: {error}");
                    }

                    ticks++;
                    foreach (float action in stepper.LastRawAction)
                    {
                        maximumAbsoluteRawAction = Math.Max(
                            maximumAbsoluteRawAction,
                            Math.Abs(action));
                    }
                }

                stepper.ApplyLastTargets();
                MujocoLib.mj_step(model, data);
            }

            public double HotSwap(PolicyEntry replacement)
            {
                double[] qpos = Copy(data->qpos, checked((int)model->nq));
                double[] qvel = Copy(data->qvel, checked((int)model->nv));
                MicroDuckControlLoopContinuation continuation = stepper.CaptureContinuation();
                IPolicyRuntime runtime = CreateRuntime(replacement);
                MujocoPolicyStepper next = null;
                try
                {
                    next = new MujocoPolicyStepper(
                        model,
                        data,
                        runtime,
                        Commands,
                        replacement.RobotVariant,
                        replacement.ActionScale,
                        continuation);
                    runtime = null;
                }
                finally
                {
                    runtime?.Dispose();
                }

                stepper.Dispose();
                stepper = next;
                Commands.SelectSlot(replacement.Slot);
                return Math.Max(
                    MaximumAbsoluteDifference(qpos, data->qpos),
                    MaximumAbsoluteDifference(qvel, data->qvel));
            }

            public StateMetrics CaptureInitialMetrics()
            {
                return new StateMetrics(this);
            }

            public void Dispose()
            {
                stepper?.Dispose();
                stepper = null;
                if (data != null)
                {
                    MujocoLib.mj_deleteData(data);
                }

                if (model != null)
                {
                    MujocoLib.mj_deleteModel(model);
                }
            }

            public double RootPosition(int axis)
            {
                return data->qpos[rootQposAddress + axis];
            }

            public double RootQuaternion(int component)
            {
                return data->qpos[rootQposAddress + 3 + component];
            }

            public double UprightDot()
            {
                return data->xmat[(9 * trunkBodyId) + 8];
            }

            public double JointPosition(int index)
            {
                return data->qpos[jointQposAddresses[index]];
            }

            public double BallPosition(int axis)
            {
                return ballQposAddress >= 0 ? data->qpos[ballQposAddress + axis] : 0.0;
            }

            public bool HasBall => ballQposAddress >= 0;

            public bool IsFinite()
            {
                return AllFinite(data->qpos, checked((int)model->nq))
                    && AllFinite(data->qvel, checked((int)model->nv))
                    && AllFinite(data->ctrl, checked((int)model->nu))
                    && stepper.LastObservation.All(value =>
                        AuthoritativeMujocoPolicyBehaviorTests.IsFinite(value))
                    && stepper.LastRawAction.All(value =>
                        AuthoritativeMujocoPolicyBehaviorTests.IsFinite(value))
                    && stepper.LastTargets.All(value =>
                        AuthoritativeMujocoPolicyBehaviorTests.IsFinite(value));
            }

            private static IPolicyRuntime CreateRuntime(PolicyEntry entry)
            {
                string assetPath = $"{PolicyAssetDirectory}/{entry.FileName}";
                NNModel asset = AssetDatabase.LoadAssetAtPath<NNModel>(assetPath);
                if (asset == null)
                {
                    throw new FileNotFoundException(
                        $"Barracuda model asset was not imported: {assetPath}");
                }

                return new BarracudaPolicyRuntime(asset, WorkerFactory.Type.CSharp);
            }

            private static int RequiredId(
                MujocoLib.mjModel_* model,
                MujocoLib.mjtObj objectType,
                string objectName)
            {
                int id = MujocoLib.mj_name2id(model, (int)objectType, objectName);
                if (id < 0)
                {
                    throw new InvalidOperationException(
                        $"MuJoCo {objectType} '{objectName}' was not found.");
                }

                return id;
            }

            private static double[] Copy(double* source, int count)
            {
                var result = new double[count];
                for (int index = 0; index < count; index++)
                {
                    result[index] = source[index];
                }

                return result;
            }

            private static double MaximumAbsoluteDifference(double[] expected, double* actual)
            {
                double maximum = 0.0;
                for (int index = 0; index < expected.Length; index++)
                {
                    maximum = Math.Max(maximum, Math.Abs(expected[index] - actual[index]));
                }

                return maximum;
            }

            private static bool AllFinite(double* values, int count)
            {
                for (int index = 0; index < count; index++)
                {
                    if (!AuthoritativeMujocoPolicyBehaviorTests.IsFinite(values[index]))
                    {
                        return false;
                    }
                }

                return true;
            }
        }

        private sealed class StateMetrics
        {
            private readonly double[] initialRoot = new double[3];
            private readonly double[] initialBall = new double[3];
            private readonly double[] jointMinimum = new double[PolicyContract.ActionCount];
            private readonly double[] jointMaximum = new double[PolicyContract.ActionCount];
            private double minimumHeight;
            private double maximumHeight;
            private double minimumUpright;
            private double previousPitch;
            private double cumulativePitch;
            private bool finite = true;

            public StateMetrics(NativeHarness harness)
            {
                for (int axis = 0; axis < 3; axis++)
                {
                    initialRoot[axis] = harness.RootPosition(axis);
                    initialBall[axis] = harness.BallPosition(axis);
                }

                for (int index = 0; index < PolicyContract.ActionCount; index++)
                {
                    jointMinimum[index] = harness.JointPosition(index);
                    jointMaximum[index] = jointMinimum[index];
                }

                minimumHeight = maximumHeight = initialRoot[2];
                minimumUpright = harness.UprightDot();
                previousPitch = RootPitch(harness);
                finite = harness.IsFinite();
            }

            public void Observe(NativeHarness harness)
            {
                minimumHeight = Math.Min(minimumHeight, harness.RootPosition(2));
                maximumHeight = Math.Max(maximumHeight, harness.RootPosition(2));
                minimumUpright = Math.Min(minimumUpright, harness.UprightDot());
                for (int index = 0; index < PolicyContract.ActionCount; index++)
                {
                    double value = harness.JointPosition(index);
                    jointMinimum[index] = Math.Min(jointMinimum[index], value);
                    jointMaximum[index] = Math.Max(jointMaximum[index], value);
                }

                double pitch = RootPitch(harness);
                cumulativePitch += WrappedAngleDifference(pitch, previousPitch);
                previousPitch = pitch;
                finite &= harness.IsFinite();
            }

            public MetricValues Complete(NativeHarness harness)
            {
                double dx = harness.RootPosition(0) - initialRoot[0];
                double dy = harness.RootPosition(1) - initialRoot[1];
                double jointExcursion = 0.0;
                for (int index = 0; index < PolicyContract.ActionCount; index++)
                {
                    jointExcursion = Math.Max(
                        jointExcursion,
                        jointMaximum[index] - jointMinimum[index]);
                }

                double ballDisplacement = 0.0;
                if (harness.HasBall)
                {
                    double ballDx = harness.BallPosition(0) - initialBall[0];
                    double ballDy = harness.BallPosition(1) - initialBall[1];
                    double ballDz = harness.BallPosition(2) - initialBall[2];
                    ballDisplacement = Math.Sqrt(
                        (ballDx * ballDx) + (ballDy * ballDy) + (ballDz * ballDz));
                }

                return new MetricValues
                {
                    finiteState = finite ? 1.0 : 0.0,
                    forwardDistanceMeters = dx,
                    rootDriftMeters = Math.Sqrt((dx * dx) + (dy * dy)),
                    minimumUprightDot = minimumUpright,
                    finalUprightDot = harness.UprightDot(),
                    heightExcursionMeters = maximumHeight - minimumHeight,
                    jointExcursionRad = jointExcursion,
                    ballDisplacementMeters = ballDisplacement,
                    pitchRotationRad = Math.Abs(cumulativePitch),
                };
            }

            private static double RootPitch(NativeHarness harness)
            {
                double w = harness.RootQuaternion(0);
                double x = harness.RootQuaternion(1);
                double y = harness.RootQuaternion(2);
                double z = harness.RootQuaternion(3);
                double matrixXx = 1.0 - (2.0 * ((y * y) + (z * z)));
                double matrixZx = 2.0 * ((x * z) - (w * y));
                return Math.Atan2(-matrixZx, matrixXx);
            }

            private static double WrappedAngleDifference(double current, double previous)
            {
                double difference = current - previous;
                while (difference > Math.PI)
                {
                    difference -= 2.0 * Math.PI;
                }

                while (difference < -Math.PI)
                {
                    difference += 2.0 * Math.PI;
                }

                return difference;
            }
        }

        private sealed class Scenario
        {
            public Scenario(
                int slot,
                string sceneFile,
                double durationSeconds,
                float forwardCommand,
                float leftCommand,
                Check[] checks,
                params double[] triggerTimesSeconds)
            {
                Slot = slot;
                SceneFile = sceneFile;
                DurationSeconds = durationSeconds;
                ForwardCommand = forwardCommand;
                LeftCommand = leftCommand;
                Checks = checks;
                TriggerTimesSeconds = triggerTimesSeconds ?? Array.Empty<double>();
            }

            public int Slot { get; }
            public string SceneFile { get; }
            public double DurationSeconds { get; }
            public float ForwardCommand { get; }
            public float LeftCommand { get; }
            public Check[] Checks { get; }
            public double[] TriggerTimesSeconds { get; }
        }

        private sealed class Check
        {
            private Check(string name, string metric, string comparison, double threshold)
            {
                Name = name;
                Metric = metric;
                Comparison = comparison;
                Threshold = threshold;
            }

            public string Name { get; }
            public string Metric { get; }
            public string Comparison { get; }
            public double Threshold { get; }

            public static Check Minimum(string name, string metric, double threshold)
            {
                return new Check(name, metric, ">=", threshold);
            }

            public static Check Maximum(string name, string metric, double threshold)
            {
                return new Check(name, metric, "<=", threshold);
            }

            public CheckResult Evaluate(MetricValues metrics)
            {
                double observed = metrics.Read(Metric);
                bool passed = IsFinite(observed)
                    && (Comparison == ">=" ? observed >= Threshold : observed <= Threshold);
                return new CheckResult
                {
                    name = Name,
                    metric = Metric,
                    comparison = Comparison,
                    observed = observed,
                    threshold = Threshold,
                    passed = passed,
                };
            }
        }

        [Serializable]
        private sealed class BehaviorReport
        {
            public int schemaVersion;
            public string generatedUtc;
            public string physicsAuthority;
            public int nativeVersion;
            public string nativeVersionString;
            public string policyRuntime;
            public double physicsTimestepSeconds;
            public int policyDecimation;
            public bool passed;
            public BehaviorResult[] scenarioResults;
            public BehaviorResult[] compoundResults;
        }

        [Serializable]
        private sealed class BehaviorResult
        {
            public string name;
            public string[] policySequence = Array.Empty<string>();
            public double simulatedSeconds;
            public int physicsStepCount;
            public int policyTickCount;
            public bool finiteState;
            public bool passed;
            public MetricValues metrics;
            public CheckResult[] checks = Array.Empty<CheckResult>();
            public string failure;

            public string FailureSummary()
            {
                if (!string.IsNullOrEmpty(failure))
                {
                    return $"{name}: {failure}";
                }

                return $"{name}: " + string.Join(
                    ", ",
                    checks.Where(check => !check.passed).Select(check =>
                        $"{check.name} observed {check.observed:R} " +
                        $"{check.comparison} {check.threshold:R}"));
            }
        }

        [Serializable]
        private sealed class MetricValues
        {
            public double finiteState;
            public double forwardDistanceMeters;
            public double rootDriftMeters;
            public double minimumUprightDot;
            public double finalUprightDot;
            public double heightExcursionMeters;
            public double jointExcursionRad;
            public double ballDisplacementMeters;
            public double pitchRotationRad;
            public double maximumAbsoluteRawAction;
            public double hotSwapStateJump;

            public double Read(string metric)
            {
                switch (metric)
                {
                    case "finiteState":
                        return finiteState;
                    case "forwardDistanceMeters":
                        return forwardDistanceMeters;
                    case "rootDriftMeters":
                        return rootDriftMeters;
                    case "minimumUprightDot":
                        return minimumUprightDot;
                    case "finalUprightDot":
                        return finalUprightDot;
                    case "heightExcursionMeters":
                        return heightExcursionMeters;
                    case "jointExcursionRad":
                        return jointExcursionRad;
                    case "ballDisplacementMeters":
                        return ballDisplacementMeters;
                    case "pitchRotationRad":
                        return pitchRotationRad;
                    case "maximumAbsoluteRawAction":
                        return maximumAbsoluteRawAction;
                    case "hotSwapStateJump":
                        return hotSwapStateJump;
                    default:
                        throw new ArgumentOutOfRangeException(
                            nameof(metric),
                            metric,
                            "Unknown behavior metric.");
                }
            }
        }

        [Serializable]
        private sealed class CheckResult
        {
            public string name;
            public string metric;
            public string comparison;
            public double observed;
            public double threshold;
            public bool passed;
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }

        private static bool IsFinite(double value)
        {
            return !double.IsNaN(value) && !double.IsInfinity(value);
        }
    }
}
