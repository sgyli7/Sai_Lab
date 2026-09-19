#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export SIM2SIM_ROOT="$PWD" PYTHONPATH="$PWD/src"
export PATH="$HOME/.local/bin:$PATH" DISPLAY="${DISPLAY:-:1}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-sai"
uv sync --frozen --extra sai
WORKSHOP_CPUS="${WORKSHOP_CPUS:-$("$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[-2:])))')}"
exec nice -n 10 taskset -c "$WORKSHOP_CPUS" "$UV_PROJECT_ENVIRONMENT/bin/python" -m sim2sim.workshop "$@"
