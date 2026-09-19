using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Editor
{
    public static class TraceBatchExporter
    {
        public const string StandPolicyAssetPath =
            "Assets/MicroDuck/Generated/Policies/Barracuda/alpha_stand.onnx";

        [MenuItem("MicroDuck/Export Tuanjie Round-Trip Trace")]
        public static void ExportOnePolicyBatch()
        {
            string outputPath = ExportOnePolicyTrace();
            Debug.Log($"Exported one real Tuanjie policy tick to '{outputPath}'.");
        }

        public static string ExportOnePolicyTrace()
        {
            string projectRoot = Directory.GetParent(Application.dataPath)?.FullName
                ?? throw new InvalidOperationException("Could not resolve the Tuanjie project root.");
            string repositoryRoot = Path.GetFullPath(Path.Combine(projectRoot, ".."));
            string scenePath = Path.Combine(
                projectRoot,
                DemoSceneBuilder.SceneAssetPath.Replace('/', Path.DirectorySeparatorChar));
            string policyPath = Path.Combine(
                projectRoot,
                StandPolicyAssetPath.Replace('/', Path.DirectorySeparatorChar));
            if (!File.Exists(scenePath))
            {
                throw new FileNotFoundException("Generated MVP scene is missing.", scenePath);
            }

            if (!File.Exists(policyPath))
            {
                throw new FileNotFoundException("Converted stand policy is missing.", policyPath);
            }

            EditorSceneManager.OpenScene(DemoSceneBuilder.SceneAssetPath, OpenSceneMode.Single);
            MicroDuckDemoController controller = UnityEngine.Object.FindObjectOfType<MicroDuckDemoController>();
            if (controller == null)
            {
                throw new InvalidOperationException("Generated MVP scene does not contain a MicroDuck controller.");
            }

            if (!controller.SelectPolicy(2))
            {
                throw new InvalidOperationException($"Could not select stand policy: {controller.Fault}");
            }

            float timestep = Time.fixedDeltaTime;
            var recorder = TuanjieTraceRecorder.ForController(
                controller,
                Sha256(policyPath),
                DemoSceneBuilder.SceneAssetPath,
                Sha256(scenePath),
                timestep,
                4);
            if (!controller.TickOnce(timestep))
            {
                throw new InvalidOperationException($"Stand policy tick failed: {controller.Fault}");
            }

            recorder.AppendControllerFrame(controller, 0, 0, timestep);
            string jsonl = recorder.Complete(passed: true);
            string outputPath = Path.Combine(
                repositoryRoot,
                "artifacts",
                "mvp",
                "traces",
                "tuanjie-alpha_stand.jsonl");
            string outputDirectory = Path.GetDirectoryName(outputPath);
            if (string.IsNullOrEmpty(outputDirectory))
            {
                throw new InvalidOperationException("Trace output directory is empty.");
            }

            Directory.CreateDirectory(outputDirectory);
            File.WriteAllText(outputPath, jsonl, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            return outputPath;
        }

        private static string Sha256(string path)
        {
            using (SHA256 algorithm = SHA256.Create())
            using (FileStream stream = File.OpenRead(path))
            {
                byte[] digest = algorithm.ComputeHash(stream);
                var output = new StringBuilder(digest.Length * 2);
                foreach (byte value in digest)
                {
                    output.Append(value.ToString("x2"));
                }

                return output.ToString();
            }
        }
    }
}
