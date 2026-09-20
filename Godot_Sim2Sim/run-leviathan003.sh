#!/usr/bin/env bash
set -euo pipefail
SAINIVERSE_GAME="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SAINIVERSE_SIBLING="$(cd -- "$SAINIVERSE_GAME/../.." && pwd)/Sai_Art"
SAINIVERSE_LEGACY="/home/ethan/Projects/RobotDesign/delivery/Sai_Design"
SAINIVERSE_RELEASE="${SAINIVERSE_RELEASE_ROOT:-$SAINIVERSE_SIBLING}"
if [[ ! -f "$SAINIVERSE_RELEASE/run.py" && -z "${SAINIVERSE_RELEASE_ROOT:-}" ]]; then
  SAINIVERSE_RELEASE="$SAINIVERSE_LEGACY"
fi
SAINIVERSE_PYTHON="${SAINIVERSE_PYTHON:-/home/ethan/Projects/RobotDesign/Leviathan_001/.venv/bin/python}"
export DISPLAY="${DISPLAY:-:1}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "$SAINIVERSE_PYTHON" "$SAINIVERSE_RELEASE/run.py" --game "$SAINIVERSE_GAME" "$@"
