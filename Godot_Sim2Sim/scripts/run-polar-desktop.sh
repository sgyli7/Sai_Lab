#!/usr/bin/env bash
set -euo pipefail
POLAR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
POLAR_STATE="${XDG_STATE_HOME:-$HOME/.local/state}/robot-godot-polar"
mkdir -p -- "$POLAR_STATE"
exec 9>"$POLAR_STATE/desktop.lock"
if ! flock -n 9; then
    if command -v xdotool >/dev/null; then
        xdotool search --onlyvisible --name '03 · 极地雪原' windowactivate >/dev/null 2>&1 || true
    fi
    exit 0
fi
export PYTHONUNBUFFERED=1
exec "$POLAR_ROOT/run-leviathan.sh" "$@" >"$POLAR_STATE/desktop.log" 2>&1
