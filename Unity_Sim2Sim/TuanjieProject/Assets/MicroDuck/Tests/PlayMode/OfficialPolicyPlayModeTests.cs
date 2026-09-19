using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class OfficialPolicyPlayModeTests
    {
        private static readonly float[] HomePositionRad =
        {
            0f, -0.0873f, -0.4579f, -0.0049f, 0.4530f,
            0.3491f, 0.3491f, 0f, 0f,
            0f, 0.0873f, 0.4579f, 0.0049f, -0.4530f,
        };

        [UnityTest]
        public IEnumerator GeneratedSceneTicksAllNinePoliciesThroughBarracudaAndArticulation()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;

            MicroDuckDemoController controller = UnityEngine.Object.FindObjectOfType<MicroDuckDemoController>();
            Assert.That(controller, Is.Not.Null, "Generated MVP scene must contain its controller.");

            foreach (PolicyEntry policy in PolicyCatalog.Entries)
            {
                Assert.That(controller.SelectPolicy(policy.Slot), Is.True, controller.Fault);
                controller.SetTwist(0.1f, 0f, 0f);
                bool ticked = controller.TickOnce(Time.fixedTime + policy.Slot * 0.02f);

                Assert.That(ticked, Is.True, $"{policy.FileName}: {controller.Fault}");
                Assert.That(controller.IsHealthy, Is.True, $"{policy.FileName}: {controller.Fault}");
                Assert.That(controller.ActivePolicyName, Is.EqualTo(policy.FileName));
                AssertFinite(controller.LastObservation, PolicyContract.ObservationCount, policy.FileName + " observation");
                AssertFinite(controller.LastRawAction, PolicyContract.ActionCount, policy.FileName + " action");
                AssertFinite(controller.LastTargets, PolicyContract.ActionCount, policy.FileName + " targets");
                yield return new WaitForFixedUpdate();
            }
        }

        [UnityTest]
        public IEnumerator RecordsOneRealPolicyTickUsingTheSharedRoundTripSchema()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;

            MicroDuckDemoController controller = UnityEngine.Object.FindObjectOfType<MicroDuckDemoController>();
            Assert.That(controller.SelectPolicy(2), Is.True, controller.Fault);
            var recorder = TuanjieTraceRecorder.ForController(
                controller,
                new string('a', 64),
                "MicroDuckMvp.unity",
                string.Empty,
                Time.fixedDeltaTime,
                4);

            Assert.That(controller.TickOnce(0.005f), Is.True, controller.Fault);
            recorder.AppendControllerFrame(controller, physicsStep: 0, policyStep: 0, timeSeconds: 0.005f);
            string jsonl = recorder.Complete(passed: true);
            RolloutTrace trace = RolloutTraceJson.Parse(jsonl);

            Assert.That(trace.Header.producer, Is.EqualTo("tuanjie"));
            Assert.That(trace.Header.policyName, Is.EqualTo("alpha_stand.onnx"));
            Assert.That(trace.Header.policySha256, Is.EqualTo(new string('a', 64)));
            Assert.That(trace.Header.role, Is.EqualTo("stand"));
            Assert.That(trace.Header.servoNames, Is.EqualTo(PolicyContract.ServoNames));
            Assert.That(trace.Frames, Has.Length.EqualTo(1));
            AssertFinite(trace.Frames[0].observation, 61, "recorded observation");
            AssertFinite(trace.Frames[0].rawAction, 14, "recorded action");
            AssertFinite(trace.Frames[0].targetPositionRad, 14, "recorded target");
            AssertFinite(trace.Frames[0].jointPositionRad, 14, "recorded joint positions");
            AssertFinite(trace.Frames[0].command, 13, "recorded command");
            Assert.That(trace.Frames[0].contacts, Is.Empty);
            Assert.That(trace.Frames[0].rawAction, Is.EqualTo(controller.LastRawAction));
        }

        [UnityTest]
        public IEnumerator RollerTraceRecordsAllFourPassiveWheelVelocities()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;

            MicroDuckDemoController controller = UnityEngine.Object.FindObjectOfType<MicroDuckDemoController>();
            Assert.That(controller.SelectPolicy(7), Is.True, controller.Fault);
            var recorder = TuanjieTraceRecorder.ForController(
                controller,
                new string('c', 64),
                "MicroDuckMvp.unity",
                string.Empty,
                Time.fixedDeltaTime,
                4);
            Assert.That(controller.TickOnce(0.005f), Is.True, controller.Fault);
            recorder.AppendControllerFrame(controller, 0, 0, 0.005f);

            RolloutTrace trace = RolloutTraceJson.Parse(recorder.Complete(passed: true));

            Assert.That(trace.Header.role, Is.EqualTo("roller"));
            Assert.That(trace.Header.passiveWheelNames, Is.EqualTo(PolicyContract.RollerPassiveWheelNames));
            AssertFinite(
                trace.Frames[0].passiveWheelVelocityRadPerSecond,
                4,
                "roller passive wheel velocities");
        }

        [UnityTest]
        public IEnumerator SelectingAPolicyResetsAllJointStateToTheMuJoCoHomeKeyframe()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;

            MicroDuckDemoController controller = UnityEngine.Object.FindObjectOfType<MicroDuckDemoController>();
            Assert.That(controller.SelectPolicy(2), Is.True, controller.Fault);
            var positions = new float[PolicyContract.ActionCount];
            var velocities = new float[PolicyContract.ActionCount];
            controller.ActiveRig.ReadPolicyState(
                positions,
                velocities,
                out Vector3 localAngularVelocity,
                out Vector3 localProjectedGravity);

            Assert.That(positions, Is.EqualTo(HomePositionRad).Within(1e-6f));
            Assert.That(velocities, Is.All.EqualTo(0f).Within(1e-7f));
            Assert.That(localAngularVelocity, Is.EqualTo(Vector3.zero));
            Assert.That(localProjectedGravity.sqrMagnitude, Is.GreaterThan(0.99f));
        }

        [UnityTest]
        public IEnumerator SwitchingRollerPoliciesClearsPassiveWheelMomentum()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;

            MicroDuckDemoController controller = UnityEngine.Object.FindObjectOfType<MicroDuckDemoController>();
            Assert.That(controller.SelectPolicy(7), Is.True, controller.Fault);
            foreach (string bodyName in new[] { "tire", "tire_2", "tire_3", "tire_4" })
            {
                FindBody(controller.ActiveRig, bodyName).jointVelocity =
                    new ArticulationReducedSpace(12f);
            }

            Assert.That(
                controller.ActiveRig.ReadPassiveWheelVelocityRadPerSecond(),
                Is.All.EqualTo(12f).Within(1e-6f));

            Assert.That(controller.SelectPolicy(8), Is.True, controller.Fault);
            Assert.That(
                controller.ActiveRig.ReadPassiveWheelVelocityRadPerSecond(),
                Is.All.EqualTo(0f).Within(1e-6f));
        }

        [UnityTest]
        public IEnumerator HotSwappingRollerPoliciesPreservesLiveRigAndControlState()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;

            MicroDuckDemoController controller = UnityEngine.Object.FindObjectOfType<MicroDuckDemoController>();
            Assert.That(controller.SelectPolicy(7), Is.True, controller.Fault);
            controller.enabled = false;
            Assert.That(controller.TickOnce(0f), Is.True, controller.Fault);

            MicroDuckRig rig = controller.ActiveRig;
            ArticulationBody root = rig.RootBody;
            root.TeleportRoot(
                root.transform.position + new Vector3(0.17f, 0.03f, -0.11f),
                Quaternion.Euler(3f, 11f, -4f) * root.transform.rotation);
            root.velocity = new Vector3(0.31f, -0.07f, 0.23f);
            root.angularVelocity = new Vector3(-0.4f, 0.6f, 0.2f);

            var jointPositions = new List<float>();
            var jointVelocities = new List<float>();
            root.GetJointPositions(jointPositions);
            root.GetJointVelocities(jointVelocities);
            for (int index = 0; index < jointPositions.Count; index++)
            {
                jointPositions[index] += 0.001f * (index + 1);
                jointVelocities[index] = 0.05f * (index + 1);
            }

            root.SetJointPositions(jointPositions);
            root.SetJointVelocities(jointVelocities);

            Vector3 expectedPosition = root.transform.position;
            Quaternion expectedRotation = root.transform.rotation;
            Vector3 expectedVelocity = root.velocity;
            Vector3 expectedAngularVelocity = root.angularVelocity;
            var expectedJointPositions = new List<float>();
            var expectedJointVelocities = new List<float>();
            root.GetJointPositions(expectedJointPositions);
            root.GetJointVelocities(expectedJointVelocities);
            float[] previousAction = controller.LastRawAction;
            float[] previousTargets = controller.LastTargets;
            int previousTicks = controller.PolicyTicks;

            Assert.That(controller.HotSwapPolicy(8), Is.True, controller.Fault);

            Assert.That(controller.ActiveRig, Is.SameAs(rig));
            Assert.That(Vector3.Distance(root.transform.position, expectedPosition), Is.LessThan(1e-6f));
            Assert.That(Quaternion.Angle(root.transform.rotation, expectedRotation), Is.LessThan(1e-5f));
            Assert.That(root.velocity, Is.EqualTo(expectedVelocity));
            Assert.That(root.angularVelocity, Is.EqualTo(expectedAngularVelocity));
            var actualJointPositions = new List<float>();
            var actualJointVelocities = new List<float>();
            root.GetJointPositions(actualJointPositions);
            root.GetJointVelocities(actualJointVelocities);
            Assert.That(actualJointPositions, Is.EqualTo(expectedJointPositions).Within(1e-7f));
            Assert.That(actualJointVelocities, Is.EqualTo(expectedJointVelocities).Within(1e-7f));
            Assert.That(controller.LastRawAction, Is.EqualTo(previousAction).Within(1e-7f));
            Assert.That(controller.LastTargets, Is.EqualTo(previousTargets).Within(1e-7f));
            Assert.That(controller.PolicyTicks, Is.EqualTo(previousTicks));

            Assert.That(controller.TickOnce(0.02f), Is.True, controller.Fault);
            Assert.That(
                controller.LastObservation.Skip(34).Take(PolicyContract.ActionCount),
                Is.EqualTo(previousAction).Within(1e-6f),
                "The new policy must observe the previous policy's last action.");
            float[] nextAction = controller.LastRawAction;
            float[] nextTargets = controller.LastTargets;
            for (int index = 0; index < PolicyContract.ActionCount; index++)
            {
                float unfiltered = HomePositionRad[index] + nextAction[index] * 0.8f;
                float alpha = index >= 5 && index <= 8 ? 0.5f : 0.7f;
                float expectedTarget = alpha * unfiltered + (1f - alpha) * previousTargets[index];
                Assert.That(nextTargets[index], Is.EqualTo(expectedTarget).Within(1e-5f),
                    $"Target filter continuation at action {index}");
            }
        }

        [UnityTest]
        public IEnumerator HotSwapRejectsCrossVariantPolicyWithoutChangingTheActiveRig()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;

            MicroDuckDemoController controller = UnityEngine.Object.FindObjectOfType<MicroDuckDemoController>();
            Assert.That(controller.SelectPolicy(7), Is.True, controller.Fault);
            controller.enabled = false;
            MicroDuckRig activeRig = controller.ActiveRig;
            activeRig.RootBody.velocity = new Vector3(0.2f, 0f, 0.4f);

            Assert.That(controller.HotSwapPolicy(9), Is.False);

            StringAssert.Contains("use SelectPolicy", controller.Fault);
            Assert.That(controller.ActivePolicySlot, Is.EqualTo(7));
            Assert.That(controller.ActiveRig, Is.SameAs(activeRig));
            Assert.That(activeRig.RootBody.velocity, Is.EqualTo(new Vector3(0.2f, 0f, 0.4f)));
        }

        [UnityTest]
        public IEnumerator HomeJointStateProducesTheSameKeyBodyPositionsAsMuJoCo()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;

            MicroDuckDemoController controller = UnityEngine.Object.FindObjectOfType<MicroDuckDemoController>();
            Assert.That(controller.SelectPolicy(2), Is.True, controller.Fault);
            controller.enabled = false;
            bool originalAutoSimulation = Physics.autoSimulation;
            Vector3 originalGravity = Physics.gravity;
            Physics.autoSimulation = false;
            Physics.gravity = Vector3.zero;
            try
            {
                // Articulation reduced-coordinate writes are applied to child Transforms
                // by the physics scene.  A near-zero, force-free step exposes the exact
                // imported forward kinematics without adding measurable dynamics.
                Physics.Simulate(1e-6f);
            }
            finally
            {
                Physics.gravity = originalGravity;
                Physics.autoSimulation = originalAutoSimulation;
            }

            Transform root = controller.ActiveRig.RootBody.transform;

            foreach (string bodyName in new[]
            {
                "yaw2roll", "hip_l", "upper_leg_left", "leg", "ankle_left",
                "bearing_roll", "hip_l_2", "upper_leg_right", "leg_2", "ankle_right",
                "neck", "neck_pitch", "yaw_roll_motion", "jaw_soft",
            })
            {
                ArticulationBody body = FindBody(controller.ActiveRig, bodyName);
                TestContext.Progress.WriteLine(
                    $"HOME_BODY {bodyName} offset={(body.transform.position - root.position).ToString("R")} "
                    + $"rotation={body.transform.rotation.ToString("R")}");
            }

            AssertBodyOffset(root, controller.ActiveRig, "ankle_left",
                new Vector3(-0.057928894f, -0.09463194f, 0.000015922f));
            AssertBodyOffset(root, controller.ActiveRig, "ankle_right",
                new Vector3(0.057928894f, -0.09463194f, 0.000015922f));
            AssertBodyOffset(root, controller.ActiveRig, "jaw_soft",
                new Vector3(0f, 0.112598647f, -0.009002612f));
        }

        [UnityTest]
        [Explicit(
            "PhysX armature calibration diagnostic only; authoritative dynamics run in native MuJoCo.")]
        public IEnumerator AnkleStepResponseMatchesTheMuJoCoArmatureDynamics()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;

            MicroDuckDemoController controller = UnityEngine.Object.FindObjectOfType<MicroDuckDemoController>();
            Assert.That(controller.SelectPolicy(2), Is.True, controller.Fault);
            controller.enabled = false;

            Collider[] colliders = controller.ActiveRig.GetComponentsInChildren<Collider>(true);
            bool[] colliderStates = colliders.Select(collider => collider.enabled).ToArray();
            bool originalAutoSimulation = Physics.autoSimulation;
            Vector3 originalGravity = Physics.gravity;
            var observed = new List<float>();
            try
            {
                Physics.autoSimulation = false;
                Physics.gravity = Vector3.zero;
                foreach (Collider collider in colliders)
                {
                    collider.enabled = false;
                }

                Physics.Simulate(1e-6f);
                float[] target = (float[])HomePositionRad.Clone();
                target[4] += 0.1f;
                controller.ActiveRig.ApplyTargets(target);
                var jointPosition = new float[PolicyContract.ActionCount];
                var jointVelocity = new float[PolicyContract.ActionCount];
                int[] sampleSteps = { 1, 4, 10, 20, 40, 80 };
                int nextSample = 0;
                for (int step = 1; step <= 80; step++)
                {
                    Physics.Simulate(0.005f);
                    if (step == sampleSteps[nextSample])
                    {
                        controller.ActiveRig.ReadPolicyState(
                            jointPosition,
                            jointVelocity,
                            out _,
                            out _);
                        observed.Add(jointPosition[4] - HomePositionRad[4]);
                        nextSample++;
                        if (nextSample == sampleSteps.Length)
                        {
                            break;
                        }
                    }
                }
            }
            finally
            {
                for (int index = 0; index < colliders.Length; index++)
                {
                    colliders[index].enabled = colliderStates[index];
                }

                Physics.gravity = originalGravity;
                Physics.autoSimulation = originalAutoSimulation;
            }

            float[] expected =
            {
                0.000603857f,
                0.005262624f,
                0.021940415f,
                0.052844197f,
                0.086390496f,
                0.092675738f,
            };
            Assert.That(observed, Has.Count.EqualTo(expected.Length));
            for (int index = 0; index < expected.Length; index++)
            {
                Assert.That(
                    observed[index],
                    Is.EqualTo(expected[index]).Within(0.008f),
                    $"MuJoCo ankle response sample {index}: expected {expected[index]:R}, "
                    + $"observed {observed[index]:R}; all=[{string.Join(",", observed)}]");
            }
        }

        [UnityTest]
        [Explicit(
            "PhysX passive-hold calibration diagnostic only; authoritative dynamics run in native MuJoCo.")]
        public IEnumerator DiagnosticHomeDrivesHoldTheRobotWithoutPolicyForTwoSeconds()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;

            MicroDuckDemoController controller = UnityEngine.Object.FindObjectOfType<MicroDuckDemoController>();
            Assert.That(controller.SelectPolicy(2), Is.True, controller.Fault);
            controller.enabled = false;
            ArticulationBody root = controller.ActiveRig.RootBody;
            Vector3 initialPosition = root.transform.position;
            Quaternion initialRotation = root.transform.rotation;
            float initialColliderBottom = float.PositiveInfinity;
            string lowestCollider = string.Empty;
            var belowGround = new List<string>();
            foreach (Collider collider in controller.ActiveRig.GetComponentsInChildren<Collider>())
            {
                float bottom = collider.bounds.min.y;
                if (bottom < initialColliderBottom)
                {
                    initialColliderBottom = bottom;
                    lowestCollider = collider.name;
                }

                if (bottom < -0.001f)
                {
                    belowGround.Add($"{collider.name}:{bottom:R}");
                }
            }

            bool originalAutoSimulation = Physics.autoSimulation;
            Physics.autoSimulation = false;
            try
            {
                for (int step = 0; step < 400; step++)
                {
                    Physics.Simulate(0.005f);
                }
            }
            finally
            {
                Physics.autoSimulation = originalAutoSimulation;
            }

            Vector3 displacement = root.transform.position - initialPosition;
            float planarDrift = new Vector2(displacement.x, displacement.z).magnitude;
            float rotation = Quaternion.Angle(initialRotation, root.transform.rotation);
            float upright = Vector3.Dot(root.transform.up, Vector3.up);
            string evidence = $"initialColliderBottom={initialColliderBottom:R} ({lowestCollider}), "
                + $"belowGround=[{string.Join(",", belowGround)}], "
                + $"finalHeight={root.transform.position.y:R}, planarDrift={planarDrift:R}, "
                + $"rotationDegrees={rotation:R}, uprightDot={upright:R}";
            TestContext.Progress.WriteLine(evidence);

            Assert.That(upright, Is.GreaterThan(0.5f), evidence);
            Assert.That(planarDrift, Is.LessThan(0.2f), evidence);
        }

        private static void AssertFinite(float[] values, int expectedLength, string label)
        {
            Assert.That(values, Has.Length.EqualTo(expectedLength), label);
            for (int index = 0; index < values.Length; index++)
            {
                Assert.That(float.IsNaN(values[index]) || float.IsInfinity(values[index]), Is.False,
                    $"{label}[{index}] must be finite.");
            }
        }

        private static void AssertBodyOffset(
            Transform root,
            MicroDuckRig rig,
            string bodyName,
            Vector3 expectedOffset)
        {
            ArticulationBody found = null;
            foreach (ArticulationBody body in rig.GetComponentsInChildren<ArticulationBody>(includeInactive: true))
            {
                if (body.name == bodyName)
                {
                    found = body;
                    break;
                }
            }

            Assert.That(found, Is.Not.Null, bodyName);
            Vector3 actualOffset = found.transform.position - root.position;
            Assert.That(Vector3.Distance(actualOffset, expectedOffset), Is.LessThan(1e-4f),
                $"{bodyName} expected offset "
                + $"({expectedOffset.x:R},{expectedOffset.y:R},{expectedOffset.z:R}), got "
                + $"({actualOffset.x:R},{actualOffset.y:R},{actualOffset.z:R}).");
        }

        private static ArticulationBody FindBody(MicroDuckRig rig, string bodyName)
        {
            foreach (ArticulationBody body in rig.GetComponentsInChildren<ArticulationBody>(includeInactive: true))
            {
                if (body.name == bodyName)
                {
                    return body;
                }
            }

            Assert.Fail($"Articulation body {bodyName} was not found.");
            return null;
        }
    }
}
