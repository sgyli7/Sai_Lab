"""Skip unless the optional `train` extra is installed."""

from __future__ import annotations

import unittest


def _train_extra_installed() -> bool:
    try:
        import onnx  # noqa: F401
        import rsl_rl  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class TestTrainDeps(unittest.TestCase):
    @unittest.skipUnless(_train_extra_installed(), "train extra not installed")
    def test_import_torch_rsl_rl_onnx(self) -> None:
        import onnx
        import rsl_rl
        import torch

        self.assertTrue(torch.__version__.startswith("2.9.1"))
        self.assertIsNotNone(rsl_rl.__file__)
        self.assertGreaterEqual(tuple(int(p) for p in onnx.__version__.split(".")[:2]), (1, 20))


if __name__ == "__main__":
    unittest.main()
