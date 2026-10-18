"""CPU contract checks; these do not validate model training or robot behavior."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.pi05_g1.model_runtime import QuantileNormalizer, resize_pad_rgb, ModelRuntime, preprocess_g1_observation
from experiments.pi05_g1.training import EpisodeDataset, retain_best_and_last, MasterAdamW, resume_admission_guard
from experiments.pi05_g1.convert_checkpoint import checked_parameter_coverage


class TrainingPipelineChecks(unittest.TestCase):
    def test_fp32_master_accumulates_sub_bf16_ulp_updates_and_resumes(self):
        import copy
        try:
            import torch
        except ImportError:
            self.skipTest("Torch unavailable in this interpreter; use the isolated training venv")
        model = torch.nn.Parameter(torch.tensor([1.], dtype=torch.bfloat16))
        optimizer = MasterAdamW([model], lr=1e-5, weight_decay=0.)
        for _ in range(300):
            model.grad = torch.ones_like(model)
            optimizer.step()
            optimizer.zero_grad()
        saved = copy.deepcopy(optimizer.state_dict())
        restored_model = torch.nn.Parameter(model.detach().clone())
        restored = MasterAdamW([restored_model], lr=1e-5, weight_decay=0.)
        restored.load_state_dict(saved)
        for _ in range(300):
            for parameter, optim in ((model, optimizer), (restored_model, restored)):
                parameter.grad = torch.ones_like(parameter)
                optim.step()
                optim.zero_grad()
        self.assertNotEqual(float(model.detach()), 1.)
        torch.testing.assert_close(model, restored_model, rtol=0., atol=0.)
        torch.testing.assert_close(optimizer.master_parameters[0], restored.master_parameters[0], rtol=0., atol=0.)
        self.assertEqual(optimizer.master_parameters[0].dtype, torch.float32)
        for value in optimizer.optimizer.state[optimizer.master_parameters[0]].values():
            if torch.is_tensor(value):
                self.assertEqual(value.dtype, torch.float32)

    def test_mmap_optimizer_restore_retains_hyperparameters_steps_and_clears_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Use the isolated training venv")
        model = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.bfloat16))
        optimizer = MasterAdamW([model], lr=1e-5, betas=(.9, .95), weight_decay=.01)
        for _ in range(4):
            model.grad = torch.ones_like(model)
            optimizer.step()
            optimizer.zero_grad()
        state = optimizer.state_dict()
        self.assertEqual(state["master_parameters"][0].data_ptr(), optimizer.master_parameters[0].data_ptr())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.pt"
            torch.save(state, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            restored_model = torch.nn.Parameter(torch.zeros_like(model))
            restored = MasterAdamW([restored_model], lr=.1, betas=(.5, .8), weight_decay=0.)
            restored_model.grad = torch.ones_like(restored_model)
            restored.master_parameters[0].grad = torch.ones_like(restored.master_parameters[0])
            restored.load_state_dict(loaded)
            self.assertIsNone(restored_model.grad)
            self.assertIsNone(restored.master_parameters[0].grad)
            group = restored.optimizer.param_groups[0]
            self.assertEqual(group["lr"], 1e-5)
            self.assertEqual(group["betas"], (.9, .95))
            self.assertEqual(group["weight_decay"], .01)
            self.assertEqual(float(restored.optimizer.state[restored.master_parameters[0]]["step"]), 4.)
            for parameter, optim in ((model, optimizer), (restored_model, restored)):
                parameter.grad = torch.ones_like(parameter)
                optim.step()
                optim.zero_grad()
            torch.testing.assert_close(optimizer.master_parameters[0], restored.master_parameters[0], rtol=0., atol=0.)
            torch.testing.assert_close(model, restored_model, rtol=0., atol=0.)

    def test_mmap_step_counter_does_not_keep_whole_checkpoint_mapped(self):
        import gc
        try:
            import torch
        except ImportError:
            self.skipTest("Use the isolated training venv")
        model = torch.nn.Parameter(torch.ones(64, dtype=torch.bfloat16))
        optimizer = MasterAdamW([model], lr=1e-5)
        model.grad = torch.ones_like(model)
        optimizer.step()
        optimizer.zero_grad()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mmap_lifecycle.pt"
            torch.save(optimizer.state_dict(), path)
            loaded = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            restored_model = torch.nn.Parameter(torch.zeros_like(model))
            restored = MasterAdamW([restored_model], lr=1e-5)
            phases = []
            restored.load_state_dict(loaded, progress=phases.append)
            self.assertIn("after_cpu_step_detached", phases)
            # On CUDA these moments have moved off the CPU file mapping. Emulate
            # only that transfer on CPU, leaving the real step-counter handling
            # under test. Without its clone, one scalar retains the entire file.
            values = restored.optimizer.state[restored.master_parameters[0]]
            for name in ("exp_avg", "exp_avg_sq"):
                values[name] = values[name].clone()
            del loaded
            gc.collect()
            self.assertNotIn(str(path), Path("/proc/self/maps").read_text())
            self.assertEqual(float(values["step"]), 1.)

    def test_real_observation_and_official_preprocess_keep_siglip_nchw(self):
        try:
            import torch
            from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
        except ImportError:
            self.skipTest("Use isolated venv with openpi PYTHONPATH for official CPU preprocessing check")
        stats = {key: {"q01": [-1.] * 32, "q99": [1.] * 32} for key in ("state", "actions")}
        class Tokenizer:
            def tokenize(self, prompt, state):
                return np.zeros(200, np.int64), np.ones(200, bool)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            path.write_text(json.dumps(stats))
            runtime = ModelRuntime.__new__(ModelRuntime)
            runtime.torch = torch
            runtime.device = torch.device("cpu")
            runtime.norm = QuantileNormalizer(path)
            runtime.tokenizer = Tokenizer()
            observation = runtime.observation(np.full((480, 640, 3), 255, np.uint8), np.zeros(32, np.float32))
            self.assertEqual(tuple(observation.images["base_0_rgb"].shape), (1, 3, 224, 224))
            self.assertEqual(observation.images["base_0_rgb"].dtype, torch.float32)
            for training in (False, True):
                processed = preprocess_observation_pytorch(observation, train=training)
                for name, image in processed.images.items():
                    self.assertEqual(tuple(image.shape), (1, 3, 224, 224))
                    self.assertTrue(torch.isfinite(image).all())
                    self.assertGreaterEqual(float(image.min()), -1.)
                    self.assertLessEqual(float(image.max()), 1.)
                self.assertTrue(bool(processed.image_masks["base_0_rgb"][0]))
                self.assertFalse(bool(processed.image_masks["left_wrist_0_rgb"][0]))
                self.assertFalse(bool(processed.image_masks["right_wrist_0_rgb"][0]))
                output = torch.nn.Conv2d(3, 4, 14, stride=14)(processed.images["base_0_rgb"])
                self.assertEqual(tuple(output.shape), (1, 4, 16, 16))

    def test_head_only_preserves_official_augmentation_rng_and_head_image(self):
        try:
            import torch
            from openpi.models.model import Observation
        except ImportError:
            self.skipTest("Use isolated venv with openpi PYTHONPATH")
        generator = torch.Generator().manual_seed(99)
        head = torch.rand((1, 3, 224, 224), generator=generator) * 2 - 1
        observation = Observation(images={"base_0_rgb": head, "left_wrist_0_rgb": torch.zeros_like(head), "right_wrist_0_rgb": torch.zeros_like(head)}, image_masks={"base_0_rgb": torch.tensor([True]), "left_wrist_0_rgb": torch.tensor([False]), "right_wrist_0_rgb": torch.tensor([False])}, state=torch.zeros((1, 32)), tokenized_prompt=torch.zeros((1, 200), dtype=torch.int64), tokenized_prompt_mask=torch.ones((1, 200), dtype=torch.bool))
        torch.manual_seed(123)
        full = preprocess_g1_observation(observation, train=True, head_only=False)
        expected_next_random = torch.rand(20)
        only = observation.replace(images={"base_0_rgb": head}, image_masks={"base_0_rgb": torch.tensor([True])})
        torch.manual_seed(123)
        reduced = preprocess_g1_observation(only, train=True, head_only=True)
        actual_next_random = torch.rand(20)
        torch.testing.assert_close(expected_next_random, actual_next_random, rtol=0., atol=0.)
        torch.testing.assert_close(full.images["base_0_rgb"], reduced.images["base_0_rgb"], rtol=0., atol=0.)
        self.assertEqual(list(reduced.images), ["base_0_rgb"])
        self.assertEqual(tuple(reduced.images["base_0_rgb"].shape), (1, 3, 224, 224))

    def test_gb10_resume_admission_checks_mapping_and_memory_limits(self):
        gib = 1024**3
        healthy = {"source_checkpoint_mapping_count": 0, "MemAvailable_bytes": 32 * gib, "cuda_allocated_bytes": 55 * gib}
        self.assertTrue(resume_admission_guard(healthy)["passed"])
        for key, value, reason in (("source_checkpoint_mapping_count", 1, "source_checkpoint_mapping_still_present_or_unknown"), ("MemAvailable_bytes", 32 * gib - 1, "system_available_memory_below_32_GiB"), ("cuda_allocated_bytes", 55 * gib + 1, "cuda_allocation_unknown_or_above_55_GiB")):
            candidate = dict(healthy)
            candidate[key] = value
            result = resume_admission_guard(candidate)
            self.assertFalse(result["passed"])
            self.assertIn(reason, result["reasons"])
        self.assertFalse(resume_admission_guard({})["passed"])

    def test_constant_channels_and_state_padding(self):
        stats = {key: {"q01": [0.] * 32, "q99": [2.] * 31 + [0.]} for key in ("state", "actions")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            path.write_text(json.dumps(stats))
            norm = QuantileNormalizer(path)
            vector = np.ones(32, dtype=np.float32)
            normalized = norm.normalize(vector, "state")
            self.assertTrue(np.isfinite(normalized).all())
            self.assertEqual(normalized[-1], 0.)
            restored = norm.unnormalize(norm.normalize(vector, "actions"))
            np.testing.assert_array_equal(restored[:31], vector[:31])
            self.assertEqual(restored[-1], 0.)

    def test_end_chunk_repeats_only_own_episode_and_masks_repetition(self):
        data = EpisodeDataset.__new__(EpisodeDataset)
        actions = np.arange(3 * 32, dtype=np.float32).reshape(3, 32)
        data.episodes = [dict(actions=actions, states=actions, timestamps=np.array([0., .02, .04]), video=Path("episode_0.mp4"))]
        data.indices = [(0, 0), (0, 1), (0, 2)]
        with patch("experiments.pi05_g1.training.decode_video_frame", return_value=np.zeros((2, 2, 3), np.uint8)) as decode:
            _, state, chunk, mask = data.sample(1)
        self.assertEqual(chunk.shape, (50, 32))
        np.testing.assert_array_equal(chunk[:2], actions[1:])
        np.testing.assert_array_equal(chunk[2:], np.repeat(actions[2:3], 48, axis=0))
        self.assertEqual(int(mask.sum()), 2)
        np.testing.assert_array_equal(state, actions[1])
        decode.assert_called_once_with(Path("episode_0.mp4"), .02)

    def test_conversion_requires_all_real_parameters_but_allows_shared_aliases(self):
        shared = type("Parameter", (), {"requires_grad": True})()
        separate = type("Parameter", (), {"requires_grad": True})()
        class Model:
            def named_parameters(self, remove_duplicate=False):
                return [("embedding", shared), ("tied_head", shared), ("projection", separate)]
            def state_dict(self):
                return {"embedding": None, "tied_head": None, "projection": None}
        report = checked_parameter_coverage(Model(), {"embedding": None, "projection": None})
        self.assertEqual(report["allowed_shared_aliases"], ["tied_head"])
        with self.assertRaises(RuntimeError):
            checked_parameter_coverage(Model(), {"embedding": None})
        with self.assertRaises(RuntimeError):
            checked_parameter_coverage(Model(), {"embedding": None, "projection": None, "alien": None})

    def test_checkpoint_retention_preserves_best_last_and_old_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("step_000100", "step_000200", "step_000300"):
                path = root / name
                path.mkdir()
                (path / "metadata.json").write_text(json.dumps({"step": 100, "identity": {"manifest": "fixed"}}))
                (path / "model.safetensors").write_bytes(b"model")
                (path / "training_state.pt").write_bytes(b"optimizer")
            (root / "best_checkpoint.json").write_text(json.dumps({"path": "step_000100"}))
            self.assertEqual(retain_best_and_last(root, "step_000300"), ["step_000200"])
            for name in ("step_000100", "step_000300"):
                self.assertTrue((root / name / "model.safetensors").is_file())
            old = root / "step_000200"
            self.assertFalse((old / "model.safetensors").exists())
            self.assertEqual(json.loads((old / "metadata.json").read_text())["payload_retention"]["status"], "pruned")

    def test_head_rgb_resize_pad_and_input_validation(self):
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        encoded = resize_pad_rgb(image)
        self.assertEqual(encoded.shape, (224, 224, 3))
        self.assertEqual(float(encoded[0, 0, 0]), -1.)
        self.assertEqual(float(encoded[112, 112, 0]), 1.)
        with self.assertRaises(ValueError):
            resize_pad_rgb(image.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
