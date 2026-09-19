#!/usr/bin/env python3
"""Fail when a recorded Sai trace leaves the deployment safety envelope."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim2sim.sai_motion_safety import assess_motion_safety


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--riser", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    document = json.loads(args.trace.read_text())
    samples = document.get("samples", document.get("rows", document)) if isinstance(document, dict) else document
    riser = float(args.riser if args.riser is not None else document.get("riser", 0.0))
    report = assess_motion_safety(samples, riser_m=riser)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
