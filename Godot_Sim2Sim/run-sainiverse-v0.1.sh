#!/usr/bin/env bash
set -euo pipefail
SAINIVERSE_GAME="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SAINIVERSE_RELEASE="${SAINIVERSE_RELEASE_ROOT:-/home/ethan/Projects/RobotDesign/delivery/Sai_Design}"
SAINIVERSE_PYTHON="${SAINIVERSE_PYTHON:-python3}"
export DISPLAY="${DISPLAY:-:1}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export SAINIVERSE_USE_GAME_ROBOTS=1
if [[ ! -f "$SAINIVERSE_GAME/results/workshop-hub/runtime/robot.gd" ]]; then
  "$SAINIVERSE_GAME/run-workshop.sh" --prepare-only
fi
exec "$SAINIVERSE_PYTHON" "$SAINIVERSE_RELEASE/run.py" --game "$SAINIVERSE_GAME" "$@"
