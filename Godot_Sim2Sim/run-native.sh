#!/usr/bin/env bash
set -euo pipefail
NATIVE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_RUNTIME="${SAI_NATIVE_RUNTIME:-$NATIVE_ROOT/results/workshop-hub/runtime}"
NATIVE_GODOT="${GODOT:-$(command -v godot || true)}"
if [ -z "$NATIVE_GODOT" ] || [ ! -x "$NATIVE_GODOT" ]; then
    echo "Godot 4.7.2 is required (set GODOT)." >&2
    exit 2
fi
if [ ! -f "$NATIVE_RUNTIME/project.godot" ] || [ ! -f "$NATIVE_RUNTIME/sai_policy/flat-motion-v1.onnx" ]; then
    echo "Prepared native runtime is missing. Run ./run-workshop.sh --prepare-only once." >&2
    exit 2
fi
NATIVE_OUTPUT="${SAI_NATIVE_OUTPUT:-$NATIVE_ROOT/results/workshop-hub/native-play}"
mkdir -p -- "$NATIVE_OUTPUT"
NATIVE_ENGINE_ARGS=()
NATIVE_USER_ARGS=("--sai-controller=native" "--output=$NATIVE_OUTPUT")
while [ "$#" -gt 0 ]; do
    case "$1" in
        --headless)
            NATIVE_ENGINE_ARGS+=("--headless")
            shift
            ;;
        --fast-check)
            NATIVE_ENGINE_ARGS+=("--fixed-fps" "30")
            NATIVE_USER_ARGS+=("--fast-check")
            shift
            ;;
        --plan|--output)
            if [ "$#" -lt 2 ]; then echo "$1 requires a value" >&2; exit 2; fi
            NATIVE_VALUE="$2"
            case "$NATIVE_VALUE" in
                /*) ;;
                *) NATIVE_VALUE="$PWD/$NATIVE_VALUE" ;;
            esac
            NATIVE_USER_ARGS+=("$1=$NATIVE_VALUE")
            shift 2
            ;;
        --scene|--robot|--task|--sai-controller|--drive-speed)
            if [ "$#" -lt 2 ]; then echo "$1 requires a value" >&2; exit 2; fi
            NATIVE_USER_ARGS+=("$1=$2")
            shift 2
            ;;
        *)
            NATIVE_USER_ARGS+=("$1")
            shift
            ;;
    esac
done
set +e
"$NATIVE_GODOT" --path "$NATIVE_RUNTIME" res://hub/main.tscn --disable-vsync --max-fps 30 \
    "${NATIVE_ENGINE_ARGS[@]}" -- "${NATIVE_USER_ARGS[@]}"
NATIVE_STATUS=$?
set -e
exit "$NATIVE_STATUS"
