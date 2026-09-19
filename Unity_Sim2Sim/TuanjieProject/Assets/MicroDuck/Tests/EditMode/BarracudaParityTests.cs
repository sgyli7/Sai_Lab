using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using Unity.Barracuda;
using UnityEditor;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class BarracudaParityTests
    {
        private const string FixtureAssetPath =
            "Assets/MicroDuck/Generated/Policies/Barracuda/parity-fixtures.json";

        private const string ModelAssetDirectory =
            "Assets/MicroDuck/Generated/Policies/Barracuda";

        [Test]
        public void AllOfficialPoliciesMatchOnnxRuntimeReferenceFixtures()
        {
            ParityFixtureFile fixtures = LoadFixtures();
            Assert.That(fixtures.schemaVersion, Is.EqualTo(1));
            Assert.That(fixtures.inputShape, Is.EqualTo(new[] { 1, PolicyContract.ObservationCount }));
            Assert.That(fixtures.outputShape, Is.EqualTo(new[] { 1, PolicyContract.ActionCount }));
            Assert.That(fixtures.cases, Has.Length.EqualTo(PolicyCatalog.Entries.Length * 3));

            var casesByPolicy = new Dictionary<string, List<ParityCase>>(StringComparer.Ordinal);
            foreach (ParityCase parityCase in fixtures.cases)
            {
                Assert.That(parityCase.input, Has.Length.EqualTo(PolicyContract.ObservationCount));
                Assert.That(parityCase.expectedOutput, Has.Length.EqualTo(PolicyContract.ActionCount));
                if (!casesByPolicy.TryGetValue(parityCase.policyName, out List<ParityCase> policyCases))
                {
                    policyCases = new List<ParityCase>();
                    casesByPolicy.Add(parityCase.policyName, policyCases);
                }

                policyCases.Add(parityCase);
            }

            float maximumAbsoluteError = 0f;
            foreach (PolicyEntry entry in PolicyCatalog.Entries)
            {
                Assert.That(casesByPolicy.TryGetValue(entry.FileName, out List<ParityCase> policyCases),
                    Is.True, $"Missing parity fixtures for {entry.FileName}.");
                Assert.That(policyCases, Has.Count.EqualTo(3));

                string modelPath = $"{ModelAssetDirectory}/{entry.FileName}";
                NNModel model = AssetDatabase.LoadAssetAtPath<NNModel>(modelPath);
                Assert.That(model, Is.Not.Null, $"Barracuda did not import {modelPath} as an NNModel.");

                using (var runtime = new BarracudaPolicyRuntime(model, WorkerFactory.Type.CSharp))
                {
                    foreach (ParityCase parityCase in policyCases)
                    {
                        var actual = new float[PolicyContract.ActionCount];
                        runtime.Evaluate(parityCase.input, actual);
                        for (int index = 0; index < actual.Length; index++)
                        {
                            float error = Mathf.Abs(actual[index] - parityCase.expectedOutput[index]);
                            maximumAbsoluteError = Mathf.Max(maximumAbsoluteError, error);
                            Assert.That(error, Is.LessThanOrEqualTo(1e-5f),
                                $"{entry.FileName}/{parityCase.fixtureName}[{index}] " +
                                $"expected {parityCase.expectedOutput[index]:R}, got {actual[index]:R}.");
                        }
                    }
                }
            }

            TestContext.Progress.WriteLine(
                $"Barracuda/ONNX Runtime parity: 9 models, 27 cases, maxAbsError={maximumAbsoluteError:R}");
        }

        private static ParityFixtureFile LoadFixtures()
        {
            string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
            Assert.That(projectRoot, Is.Not.Null);
            string fixturePath = Path.Combine(projectRoot, FixtureAssetPath.Replace('/', Path.DirectorySeparatorChar));
            Assert.That(File.Exists(fixturePath), Is.True, $"Missing fixture file: {fixturePath}");
            ParityFixtureFile fixtures = JsonUtility.FromJson<ParityFixtureFile>(File.ReadAllText(fixturePath));
            Assert.That(fixtures, Is.Not.Null);
            return fixtures;
        }

        [Serializable]
        private sealed class ParityFixtureFile
        {
            public int schemaVersion;
            public int[] inputShape;
            public int[] outputShape;
            public ParityCase[] cases;
        }

        [Serializable]
        private sealed class ParityCase
        {
            public string fixtureName;
            public string policyName;
            public float[] input;
            public float[] expectedOutput;
        }
    }
}
