#!/usr/bin/env bash
# Independent kick_left/right headless gate. Soft known-fail by default —
# does NOT change ./run.sh walk SIM2SIM_RUN green semantics.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export GODOT="${GODOT:-$HOME/.local/bin/godot}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
MODE="${1:-soft}"
shift || true
exec uv run sim2sim-kick-gate --mode "$MODE" "$@"
