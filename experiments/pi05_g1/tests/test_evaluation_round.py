"""CPU-only orchestration tests; no policy, simulator, or GPU is launched."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.pi05_g1.evaluation_round import run_round, select_checkpoint, sha256


class EvaluationRoundChecks(unittest.TestCase):
    def fixture(self, root):
        def write(name, data):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
            return path
        manifest = write("manifest.json", {"train": [0], "validation": [1], "test": list(range(2, 23))})
        norm = write("norm.json", {"training_only": True})
        training = root / "training_run02"
        selected = training / "step_000100"
        selected.mkdir(parents=True)
        (selected / "model.safetensors").write_bytes(b"selected immutable weights")
        write("training_run02/step_000100/metadata.json", {"step": 100, "validation_flow_loss": .1, "identity": {"manifest_sha256": sha256(manifest), "norm_sha256": sha256(norm)}})
        write("training_run02/best_checkpoint.json", {"path": "step_000100", "validation_flow_loss": .1})
        base = root / "base"
        base.mkdir()
        (base / "model.safetensors").write_bytes(b"official base placeholder for CPU orchestration")
        return argparse.Namespace(training_dir=training, template=write("template.json", {"frozen": True}), cases=write("cases.json", {"cases": [{"case_id": f"p{i}"} for i in range(20)]}), replay_result=write("replay.json", {"source_interface_passed": True, "replay_passed": True}), norm_stats=norm, dataset=root / "data", manifest=manifest, base_checkpoint=base, output=root / "evaluation")

    def test_serial_round_keeps_preselected_weights_and_accepts_task_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            commands = []
            def runner(command, **kwargs):
                commands.append(command)
                self.assertNotIn("stdout", kwargs)
                self.assertNotIn("stderr", kwargs)
                self.assertNotIn("start_new_session", kwargs)
                if command[2].endswith("evaluate_offline"):
                    Path(command[command.index("--output") + 1]).write_text(json.dumps({"status": "complete", "split": "test"}))
                if command[2].endswith("summarize_evaluation"):
                    Path(command[command.index("--output") + 1]).write_text(json.dumps({"complete": True, "evidence_complete": True, "gates": {"source_interface_passed": True, "replay_passed": True}, "training_verdict": "not_established", "task_verdict": "not_qualified"}))
                return SimpleNamespace(returncode=0)
            receipt = run_round(args, runner=runner)
            self.assertEqual(receipt["status"], "completed")
            self.assertFalse(receipt["test_checkpoint_selection"])
            self.assertEqual(receipt["selected"]["step"], 100)
            self.assertEqual([item[2].split(".")[-1] for item in commands], ["run_evaluation", "run_evaluation", "evaluate_offline", "evaluate_offline", "summarize_evaluation"])
            self.assertEqual(commands[1][commands[1].index("--checkpoint") + 1], receipt["selected"]["checkpoint"])
            self.assertEqual(commands[3][commands[3].index("--checkpoint") + 1], receipt["selected"]["checkpoint"])
            self.assertEqual(receipt["summary_verdicts"]["task_verdict"], "not_qualified")

    def test_nonzero_child_stops_without_inventing_remaining_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            calls = []
            def runner(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=7)
            with self.assertRaisesRegex(RuntimeError, "physical_baseline"):
                run_round(args, runner=runner)
            receipt = json.loads((args.output / "round_receipt.json").read_text())
            self.assertEqual(len(calls), 1)
            self.assertEqual(receipt["status"], "infrastructure_failed")
            self.assertEqual(receipt["children"][0]["exit_code"], 7)
            self.assertNotIn("summary_verdicts", receipt)

    def test_identity_or_pointer_disagreement_rejects_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            pointer = Path(args.training_dir) / "best_checkpoint.json"
            pointer.write_text(json.dumps({"path": "step_000100", "validation_flow_loss": .2}))
            with self.assertRaisesRegex(ValueError, "validation loss disagree"):
                select_checkpoint(args.training_dir, args.manifest, args.norm_stats)
            pointer.write_text(json.dumps({"path": "step_000100", "validation_flow_loss": .1}))
            Path(args.norm_stats).write_text("{}")
            with self.assertRaisesRegex(ValueError, "normalization identity"):
                select_checkpoint(args.training_dir, args.manifest, args.norm_stats)

    def test_pointer_cannot_change_midround(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            calls = []
            def runner(command, **kwargs):
                calls.append(command)
                (Path(args.training_dir) / "best_checkpoint.json").write_text(json.dumps({"path": "step_000100", "validation_flow_loss": .2}))
                return SimpleNamespace(returncode=0)
            with self.assertRaisesRegex(RuntimeError, "pointer changed"):
                run_round(args, runner=runner)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
