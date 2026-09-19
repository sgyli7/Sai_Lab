from __future__ import annotations

from pathlib import Path
import signal
import subprocess
from unittest.mock import patch

from sim2sim import workshop


def test_runtime_import_retries_one_transient_godot_abort(tmp_path: Path) -> None:
    results = [
        subprocess.CompletedProcess([], -signal.SIGABRT),
        subprocess.CompletedProcess([], 0),
    ]

    with patch.object(workshop.subprocess, "run", side_effect=results) as run:
        workshop.import_runtime_assets("godot", tmp_path)

    assert run.call_count == 2
