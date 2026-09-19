#!/usr/bin/env bash
set -euo pipefail
LEVIATHAN_GAME="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEVIATHAN_DESIGN="${LEVIATHAN_DESIGN_ROOT:-/home/ethan/Projects/RobotDesign/Leviathan_001}"
export SIM2SIM_ROOT="$LEVIATHAN_GAME" DISPLAY="${DISPLAY:-:1}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$LEVIATHAN_DESIGN/.venv/bin/python" "$LEVIATHAN_GAME/scripts/run_leviathan.py" --design "$LEVIATHAN_DESIGN" "$@"
