#!/usr/bin/env python3
"""Summarize aligned Sai body, payload, joint, contact, and skill time series."""
import argparse
import json
from pathlib import Path

from sim2sim.sai_trace_analysis import analyze_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze_file(args.trace, args.output), indent=2))


if __name__ == "__main__":
    main()
