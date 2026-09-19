#!/usr/bin/env bash
# Documented train entry: exec the venv binary so SIGINT/SIGTERM reach Python
# (and the runner.py checkpoint path). `uv run sim2sim-train` keeps uv as PID 1
# of the tree, so `kill <uv>` does not deliver SIGTERM to the trainer.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
export SIM2SIM_ROOT="$ROOT"
export MICRODUCK_POLICIES="${MICRODUCK_POLICIES:-$HOME/Projects/MicroDuck/policies}"
# Keep torch/OpenMP from hogging the cores pinned to Godot workers.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export SIM2SIM_TORCH_THREADS="${SIM2SIM_TORCH_THREADS:-2}"

# Materialize the project scripts in .venv without `uv sync` (that drops [train]).
uv run --no-sync true

TRAIN="$ROOT/.venv/bin/sim2sim-train"
if [[ ! -x "$TRAIN" ]]; then
  echo "missing $TRAIN (install the train extra; do not uv sync)" >&2
  exit 1
fi
exec "$TRAIN" "$@"
