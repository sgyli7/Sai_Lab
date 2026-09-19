#!/usr/bin/env bash
set -euo pipefail
WORKSHOP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSHOP_STATE="${XDG_STATE_HOME:-$HOME/.local/state}/robot-godot-workshop"
mkdir -p -- "$WORKSHOP_STATE"

# Desktop launches share the prepared runtime. Repeated clicks raise one game.
exec 9>"$WORKSHOP_STATE/desktop.lock"
if ! flock -n 9; then
    if command -v xdotool >/dev/null; then
        xdotool search --onlyvisible --class 'Robot Godot Workshop' windowactivate >/dev/null 2>&1 || true
    fi
    exit 0
fi
export PYTHONUNBUFFERED=1
if "$WORKSHOP_ROOT/run-workshop.sh" "$@" >"$WORKSHOP_STATE/desktop.log" 2>&1; then
    exit 0
else
    WORKSHOP_STATUS=$?
    if command -v notify-send >/dev/null; then
        notify-send --app-name='小小维修站' '小小维修站启动失败' "运行记录：$WORKSHOP_STATE/desktop.log" || true
    fi
    exit "$WORKSHOP_STATUS"
fi
