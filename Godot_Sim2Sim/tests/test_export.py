"""Export ONNX + schema-2 sidecar: round-trip parity, manifest, PolicyBundle limits."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim2sim.paths import policies_dir
from sim2sim.play_input import TwistLimits
from sim2sim.policy import PolicyBundle, PolicyShapeError

ALPHA_ONNX = policies_dir() / "alpha_walking.onnx"

_ALPHA_META_KEYS = (
    "run_path",
    "joint_names",
    "joint_stiffness",
    "joint_damping",
    "default_joint_pos",
    "command_names",
    "observation_names",
    "action_scale",
)


def _train_extra_installed() -> bool:
    try:
        import onnx  # noqa: F401
        import rsl_rl  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


_SKIP = (not _train_extra_installed()) or (not ALPHA_ONNX.is_file())


def _onnx_max_abs(path_a: Path, path_b: Path, n: int = 10000, seed: int = 0) -> float:
    from sim2sim.train.onnx_import import _onnx_actions_fn, _sample_realistic_obs61

    rng = np.random.default_rng(seed)
    batches = [rng.standard_normal((n, 61), dtype=np.float32), _sample_realistic_obs61(n, rng)]
    fa, fb = _onnx_actions_fn(path_a), _onnx_actions_fn(path_b)
    err = 0.0
    for obs in batches:
        err = max(err, float(np.max(np.abs(fa(obs) - fb(obs)))))
    return err


class TestBuildManifest(unittest.TestCase):
    def test_schema2_single_policy_fields(self) -> None:
        from sim2sim.train.manifest import build_manifest, write_manifest

        man = build_manifest(
            onnx_path=Path("/tmp/Walk_Godot.onnx"),
            source="/tmp/model_1.pt",
            training={"checkpoint": 12, "run": "unit", "init_from": "/tmp/model_1.pt"},
            eval={"note": "unit"},
            twist_limits={"vmax_x": 0.4, "vmin_x": -0.3, "vmax_y": 0.2, "vmin_y": -0.2, "vmax_ang": 1.0},
            use_stand_policy=True,
            description="unit walk",
        )
        self.assertEqual(man["schema_version"], 2)
        self.assertEqual(man["model_api"], 1)
        self.assertEqual(man["obs_len"], 61)
        self.assertEqual(man["action_len"], 14)
        self.assertEqual(man["robot"], {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50})
        self.assertEqual(man["name"], "walk_godot")
        self.assertEqual(man["kind"], "perpetual")
        self.assertEqual(man["slot"], "walk")
        self.assertEqual(man["entry_pose"], "standing")
        self.assertEqual(man["action_scale"], 1.0)
        self.assertEqual(man["description"], "unit walk")
        self.assertEqual(man["command"]["encoding"], "constant")
        self.assertEqual(man["command"]["idle"], [0, 0, 0])
        self.assertIn("twist", man["command"])
        self.assertIn("head", man["command"])
        self.assertIn("body", man["command"])
        tr = man["training"]
        self.assertEqual(tr["task_id"], "Sim2Sim-GodotJolt-Velocity-Walk")
        self.assertEqual(tr["repo"], "MicroDuck/sim2sim")
        self.assertTrue(tr["commit"])
        self.assertIn("branch", tr)
        self.assertIn("dirty", tr)
        self.assertEqual(tr["run"], "unit")
        self.assertEqual(tr["checkpoint"], 12)
        self.assertTrue(str(tr["exported"]).endswith("Z") or "T" in str(tr["exported"]))
        self.assertEqual(tr["init_from"], "/tmp/model_1.pt")
        self.assertEqual(man["eval"], {"note": "unit"})
        s2 = man["sim2sim"]
        self.assertEqual(s2["twist_limits"]["vmax_x"], 0.4)
        self.assertEqual(s2["twist_limits"]["vmin_x"], -0.3)
        self.assertTrue(s2["use_stand_policy"])
        self.assertEqual(s2["fall"], {"tilt_deg": 70, "min_z": 0.055})
        self.assertEqual(s2["control"], {"dt": 0.005, "decimation": 4})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "Walk_Godot.manifest.json"
            write_manifest(path, man)
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded["name"], "walk_godot")


class TestExportRoundtrip(unittest.TestCase):
    @unittest.skipIf(_SKIP, "alpha_walking.onnx or train extra missing")
    def test_roundtrip_onnxruntime_parity(self) -> None:
        from sim2sim.train.export import export_actor
        from sim2sim.train.onnx_import import build_actor_state_dict, parse_mlp_onnx, verify_parity

        rec = parse_mlp_onnx(ALPHA_ONNX)
        sd = build_actor_state_dict(rec, init_std=0.25, count=1_000_000_000)
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td, patch(
            "sim2sim.train.export.publish_to_repo_policies"
        ) as publish:
            out = Path(td) / "Walk_Godot.onnx"
            result = export_actor(
                state_dict=sd,
                source=str(ALPHA_ONNX),
                out=out,
                checkpoint=None,
                run="roundtrip",
                eval_info=None,
            )
            self.assertTrue(out.is_file())
            publish.assert_not_called()
            side = out.with_name(out.stem + ".manifest.json")
            self.assertTrue(side.is_file(), f"missing sidecar {side}")
            self.assertLess(result["parity_max_abs_err"], 1e-5, result)
            bundle = PolicyBundle(out)
            bundle.check_dims(14)
            self.assertAlmostEqual(bundle.twist_limits.vmax_x, 0.4)
            self.assertAlmostEqual(bundle.twist_limits.vmin_x, -0.3)
            self.assertAlmostEqual(bundle.twist_limits.vmax_y, 0.2)
            self.assertAlmostEqual(bundle.twist_limits.vmin_y, -0.2)
            self.assertAlmostEqual(bundle.twist_limits.vmax_ang, 1.0)
            self.assertNotAlmostEqual(bundle.twist_limits.vmax_x, TwistLimits().vmax_x)
            self.assertTrue(bundle.has_standing_partner)
            err = _onnx_max_abs(ALPHA_ONNX, out, n=10000, seed=0)
            self.assertLess(err, 1e-5, f"onnxruntime alpha vs roundtrip max_abs={err}")
            from sim2sim.train.onnx_import import make_rsl_actor_from_state_dict

            actor = make_rsl_actor_from_state_dict(
                sd,
                {"obs_dim": 61, "act_dim": 14, "hidden_dims": rec.hidden_dims, "activation": rec.activation},
                device="cpu",
            )
            torch_err = verify_parity(actor, out, n=10000, seed=0)
            self.assertLess(torch_err, 1e-5, f"torch vs exported onnx max_abs={torch_err}")

    @unittest.skipIf(_SKIP, "alpha_walking.onnx or train extra missing")
    def test_exported_metadata_matches_alpha_keys_and_formats(self) -> None:
        import onnx

        from sim2sim.train.export import export_actor
        from sim2sim.train.onnx_import import build_actor_state_dict, parse_mlp_onnx

        rec = parse_mlp_onnx(ALPHA_ONNX)
        sd = build_actor_state_dict(rec, init_std=0.25, count=1_000_000_000)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "rt.onnx"
            export_actor(state_dict=sd, source=str(ALPHA_ONNX), out=out)
            alpha = {p.key: p.value for p in onnx.load(str(ALPHA_ONNX)).metadata_props}
            got = {p.key: p.value for p in onnx.load(str(out)).metadata_props}
            for key in _ALPHA_META_KEYS:
                self.assertIn(key, got)
                self.assertEqual(got[key], alpha[key], f"metadata {key!r} format mismatch")
            self.assertIn("init_from", got)
            self.assertIn("sim2sim_commit", got)

    @unittest.skipIf(_SKIP, "alpha_walking.onnx or train extra missing")
    def test_cli_actor_pt_and_check_dims(self) -> None:
        import torch

        from sim2sim.train.export import main as export_main
        from sim2sim.train.onnx_import import build_actor_state_dict, parse_mlp_onnx

        rec = parse_mlp_onnx(ALPHA_ONNX)
        sd = build_actor_state_dict(rec, init_std=0.25, count=1_000_000_000)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            actor_pt = td_path / "imported.pt"
            torch.save({"actor_state_dict": sd, "source_onnx": str(ALPHA_ONNX), "hidden_dims": rec.hidden_dims}, actor_pt)
            out = td_path / "from_pt.onnx"
            rc = export_main(["--actor-pt", str(actor_pt), "--out", str(out)])
            self.assertEqual(rc, 0)
            PolicyBundle(out).check_dims(14)
            with self.assertRaises(PolicyShapeError):
                PolicyBundle(out).check_dims(7)


if __name__ == "__main__":
    unittest.main()
